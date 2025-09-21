
import sys
import os
from logging_setup import get_logger

log = get_logger(__name__)
# Ensure root path is added
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backtrader as bt
import pandas as pd

import pandas as pd

from backend.analyzer.analyzer_vwap_sl_tp_function_tool import *

class VWAPBounceStrategy(bt.Strategy):
    params = dict(
        # Required
        strategy_name=None,       # e.g., vwap_bounce_01_sl_candle_low_tp_2R
        bias=None,                # Optional directional bias (bullish/bearish)

        # Entry logic params
        vwap_distance_pct=None,   # Entry trigger: proximity below VWAP
        volume_zscore_min=None,   # Entry trigger: volume confirmation (z-score)

        # Exit logic / SL/TP
        stop_loss_rule=None,      # e.g., "candle_low", "atr"
        take_profit_rule=None,    # e.g., "2R", "ema_trail"
        atr_mult_sl=None,         # Used if SL rule = ATR based
        atr_mult_tp=None,         # Used if TP rule = ATR based or trail
        r_multiple=None,          # Multiplier for Risk × Reward exits
        trailing_offset=None,     # For trailing SL cases (optional)
        trail_lookback=None,      # Optional lookback window for trailing TP

        # Routing to SL/TP logic
        sl_tp_function=None       # Optional callable name (via dispatch)
    )


    def __init__(self):
        # Trade tracking
        self.order = None
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0.0
        self.max_equity = self.broker.getvalue()
        self.equity_curve = []

        # === Strategy Variant ID ===
        # Full string like: "01_sl_candle_low_tp_2R" from strategy name: "vwap_bounce_01_sl_candle_low_tp_2R"
        try:
            parts = self.p.strategy_name.split("_")
            self.strategy_id = "_".join(parts[2:])  # Keeps "01_sl_candle_low_tp_2R"
        except Exception:
            raise ValueError(f"Invalid strategy_name format: {self.p.strategy_name}")
        if not hasattr(self, "df_ohlc"):
            try:
                self.df_ohlc = self.datas[0].p.dataname.copy()
            except Exception:
                raise AttributeError("df_ohlc must be provided for entry logic")
        # Common indicators
        self.VWAP = self.datas[0].VWAP
        self.EMA_9 = self.datas[0].EMA_9
        self.volume_zscore = self.datas[0].volume_zscore
        self.ATR_14 = self.datas[0].ATR_14

    def log(self, txt, dt=None, level="info"):
        # Only log entry/exit events and critical errors
        important = True
        if not important:
            return  # skip non-important logs

        dt = dt or self.datas[0].datetime.datetime(0)
        log.info(f"[{dt}] {txt}")

    def check_entry_signal_vwap_bounce_both_directions(self, df: pd.DataFrame, i: int) -> list:
        """
        Detects VWAP Bounce entry conditions in both directions.

        Args:
            df (pd.DataFrame): The OHLCV + indicator dataframe.
            i (int): Index of the current bar.

        Returns:
            list: [True/False, direction ("long"/"short"/"")]
        """
        row = df.iloc[i]
        row_prev = df.iloc[i - 1] if i > 0 else row
        row_range = df.iloc[i - 2:i + 1] if i >= 2 else df.iloc[:i + 1]

        close = row["Close"]
        open_ = row["Open"]
        vwap = row["VWAP"]
        zscore = row["volume_zscore"]
        ema9 = row["EMA_9"]
        atr14 = row["ATR_14"]

        dist_pct = self.p.vwap_distance_pct or 0.3
        min_vol_zscore = self.p.volume_zscore_min or 0.5

        # --- Deep Pullback Reversal ---
        condition1_long = (
                close > open_ and
                close < vwap and
                min(row_range["Low"]) < vwap * (1 - dist_pct) and
                zscore > min_vol_zscore
        )
        condition1_short = (
                close < open_ and
                close > vwap and
                max(row_range["High"]) > vwap * (1 + dist_pct) and
                zscore > min_vol_zscore
        )

        # --- Shallow Touch + Breakout ---
        condition2_long = (
                abs(close - vwap) / vwap < 0.002 and
                close > row_prev["High"] and
                ema9 > row["EMA_20"] and
                zscore > min_vol_zscore
        )
        condition2_short = (
                abs(close - vwap) / vwap < 0.002 and
                close < row_prev["Low"] and
                ema9 < row["EMA_20"] and
                zscore > min_vol_zscore
        )

        if i > 20 and pd.notna(zscore):
            near_pullback = (
                    close < vwap and
                    (vwap - close) / vwap < 0.01 and
                    min(row_range["Low"]) < vwap * (1 - 0.005)  # Slightly relaxed
            )

            near_breakout = (
                    abs(close - vwap) / vwap < 0.005 and
                    (close > row_prev["High"] or close < row_prev["Low"])
            )

            # if near_pullback or near_breakout:
            #     self.log(
            #         f"[DEBUG] i={i} Date={row.name} Close={close:.2f} VWAP={vwap:.2f} EMA9={ema9:.2f} EMA20={row['EMA_20']:.2f} Z={zscore:.2f}")

        if condition1_long or condition2_long:
            return [True, "long"]
        elif condition1_short or condition2_short:
            return [True, "short"]
        else:
            return [False, ""]

    from typing import Tuple, Optional

    def calculate_sl_tp_for_current_bar(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Computes SL/TP for current bar using one of the 6 standard VWAP exit functions,
        depending on the strategy variant (01–04) inside VWAPBounceStrategy.

        Returns:
            Tuple (stop_loss, take_profit)
        """
        if self.df_ohlc is None or self.entry_price is None:
            self.log("Missing df_ohlc or entry_price")
            return None, None

        try:
            strategy_id = self.strategy_id
            entry_index = self.data._idx
            side = self.current_trade["direction"]

            if strategy_id == "01_sl_candle_low_tp_2R":
                candle_low = self.data.low[0]
                stop_ticks = self.entry_price - candle_low
                if self.p.r_multiple is None:
                    raise ValueError("r_multiple is required for R-multiple exit logic")
                result = calc_exit_by_r_multiple(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    stop_ticks=stop_ticks,
                    r_multiple=self.p.r_multiple
                )

            elif strategy_id == "02_sl_1.2atr14_tp_2R":
                if self.p.atr_mult_sl is None or self.p.atr_mult_tp is None:
                    raise ValueError("ATR multipliers must be set for strategy requiring ATR-based SL/TP")

                result = calc_exit_by_atr(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    atr_mult_sl=self.p.atr_mult_sl,
                    atr_mult_tp=self.p.atr_mult_tp
                )

            elif strategy_id == "03_sl_candle_low_tp_ema9":
                result = calc_exit_by_trailing_ema_vwap(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    trailing_type="EMA_9"
                )

            elif strategy_id == "04_sl_1.2atr14_tp_ema9":
                result = calc_exit_by_trailing_ema_vwap(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    trailing_type="EMA_9"
                )

            else:
                self.log(f"Unknown strategy_id: {strategy_id}")
                return None, None

            return result.get("sl"), result.get("tp")

        except Exception as e:
            self.log(f"[SL/TP ERROR] {e}")
            return None, None

    def is_in_trade(self) -> bool:
        return self.position or self.order

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        close = self.datas[0].close[0]


        # === Entry ===
        if not self.is_in_trade():
            bar_dt = self.datas[0].datetime.datetime(0)
            i = self.df_ohlc.index.get_loc(bar_dt)

            signal, direction = self.check_entry_signal_vwap_bounce_both_directions(self.df_ohlc, i=i)
            if signal:
                self.log(f"[ENTRY] {self.strategy_id} {direction.upper()} @ {close:.2f}")

                if direction == "long":
                    self.order = self.buy()
                elif direction == "short":
                    self.order = self.sell()
                else:
                    self.log("[ERROR] Unknown direction; no order placed.")
                    return

                self.entry_price = close
                self.entry_time = dt
                self.current_trade = {
                    "entry_time": dt,
                    "entry_price": close,
                    "strategy": self.p.strategy_name,
                    "direction": direction
                }

        # === Exit ===
        elif self.position:
            stop_loss, take_profit = self.calculate_sl_tp_for_current_bar()
            if stop_loss is None and take_profit is None:
                return  # No SL/TP logic triggered

            if not self.current_trade or "direction" not in self.current_trade:
                raise ValueError("Trade direction is missing in current_trade context")
            trade_dir = self.current_trade["direction"]

            # --- Exit for Long Trades ---
            if trade_dir == "long":
                if take_profit is not None and close >= take_profit:
                    self.log(f"[TP HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} >= {take_profit:.2f}")

                    self.order = self.sell()
                elif stop_loss is not None and close <= stop_loss:
                    self.log(f"[SL HIT {self.strategy_id} {trade_dir.upper()} {close:.2f} <= {stop_loss:.2f}")
                    self.order = self.sell()

            # --- Exit for Short Trades ---
            elif trade_dir == "short":
                if take_profit is not None and close <= take_profit:
                    self.log(f"[TP HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} >= {take_profit:.2f}")
                    self.order = self.buy()
                elif stop_loss is not None and close >= stop_loss:
                    self.log(f"[SL HIT {self.strategy_id} {trade_dir.upper()} {close:.2f} <= {stop_loss:.2f}")
                    self.order = self.buy()

        # === Equity Tracking ===
        equity = self.broker.getvalue()
        self.equity_curve.append(equity)
        self.max_equity = max(self.max_equity, equity)
        drawdown = self.max_equity - equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: price={order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED: price={order.executed.price:.2f}")
        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(f"{side} EXECUTED @ {order.executed.price:.2f}")
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            self.pnl += pnl
            win = pnl > 0

            if win:
                self.wins += 1
            else:
                self.losses += 1

            entry_price = trade.price  # this is the entry
            exit_price = self.datas[0].close[0]  # current close
            exit_time = self.datas[0].datetime.datetime(0)

            self.log(
                f"TRADE CLOSED: Entry={entry_price:.2f}, Exit={exit_price:.2f}, PnL={pnl:.2f}, Time={exit_time}"
            )

            trade_record = {
                "entry_time": self.entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "win": win,
                "strategy": self.p.strategy_name
            }
            self.trades.append(trade_record)

            # reset state
            self.entry_price = None
            self.entry_time = None
            self.current_trade = None

    def stop(self):
        n_trades = len(self.trades)
        win_rate = self.wins / n_trades * 100 if n_trades > 0 else 0
        self.metrics = {
            "PnL": self.pnl,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "num_trades": n_trades,
            "strategy": self.p.strategy_name
        }
        self.log(f"Strategy stopped. PnL={self.pnl:.2f}, Win rate={win_rate:.1f}%, Max DD={self.max_drawdown:.2f}, Trades={n_trades}")


class VWAPReclaimStrategy(bt.Strategy):
    params = dict(
        strategy_name=None,
        bias=None,

        # Entry logic for reclaim
        vwap_cross_back_pct=None,  # << renamed
        volume_zscore_min=None,

        # SL/TP
        stop_loss_rule=None,
        take_profit_rule=None,
        atr_mult_sl=None,
        atr_mult_tp=None,
        r_multiple=None,
        trailing_offset=None,
        trail_lookback=None,

        sl_tp_function=None
    )

    def __init__(self):
        # Trade tracking
        self.order = None
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0.0
        self.max_equity = self.broker.getvalue()
        self.equity_curve = []

        # === Strategy Variant ID ===
        # Full string like: "01_sl_candle_low_tp_2R" from strategy name: "vwap_bounce_01_sl_candle_low_tp_2R"
        try:
            parts = self.p.strategy_name.split("_")
            self.strategy_id = "_".join(parts[2:])  # Keeps "01_sl_candle_low_tp_2R"
        except Exception:
            raise ValueError(f"Invalid strategy_name format: {self.p.strategy_name}")
        if not hasattr(self, "df_ohlc"):
            try:
                self.df_ohlc = self.datas[0].p.dataname.copy()
            except Exception:
                raise AttributeError("df_ohlc must be provided for entry logic")
        # Common indicators
        self.VWAP = self.datas[0].VWAP
        self.EMA_9 = self.datas[0].EMA_9
        self.volume_zscore = self.datas[0].volume_zscore
        self.ATR_14 = self.datas[0].ATR_14

    def log(self, txt, dt=None, level="info"):
        # Only log entry/exit events and critical errors
        important = True
        if not important:
            return  # skip non-important logs

        dt = dt or self.datas[0].datetime.datetime(0)
        log.info(f"[{dt}] {txt}")

    from typing import Tuple, Optional
    import pandas as pd

    def check_entry_signal_vwap_reclaim_both_directions(self, df: pd.DataFrame, i: int) -> list:
        """
        Detects VWAP Reclaim entry conditions in both directions (long and short).
        Includes debug logging to explain why an entry was or wasn't triggered.

        Args:
            df (pd.DataFrame): OHLCV + indicator dataframe with VWAP, volume_zscore, etc.
            i (int): Index of the current bar in backtesting loop.

        Returns:
            list: [bool (entry_found), direction ("long"/"short"/"")]
        """
        if i < 2:
            return [False, ""]

        row = df.iloc[i]
        row_prev = df.iloc[i - 1]

        close = row["Close"]
        open_ = row["Open"]
        vwap = row["VWAP"]
        zscore = row["volume_zscore"]
        ema9 = row["EMA_9"]

        cross_back_pct = self.p.vwap_cross_back_pct or 0.2
        min_zscore = self.p.volume_zscore_min or 1.0

        # === LONG RECLAIM ===
        dipped_below = row_prev["Low"] < vwap * (1 - cross_back_pct)
        reclaimed = close > vwap and close > open_
        strong_volume = zscore >= min_zscore
        condition_long = dipped_below and reclaimed and strong_volume

        if dipped_below and reclaimed:
            self.log(
                f"[DEBUG] Reclaim UP trigger @ {row.name} zscore={zscore:.2f} (min={min_zscore}) ✅" if strong_volume else f"[DEBUG] Reclaim UP SKIPPED @ {row.name} zscore={zscore:.2f} ❌")

        # === SHORT RECLAIM FAIL ===
        spiked_above = row_prev["High"] > vwap * (1 + cross_back_pct)
        rejected = close < vwap and close < open_
        condition_short = spiked_above and rejected and strong_volume

        if spiked_above and rejected:
            self.log(
                f"[DEBUG] Reclaim DOWN trigger @ {row.name} zscore={zscore:.2f} (min={min_zscore}) ✅" if strong_volume else f"[DEBUG] Reclaim DOWN SKIPPED @ {row.name} zscore={zscore:.2f} ❌")
        condition_long = close > vwap
        condition_short = close < vwap

        if condition_long:
            return [True, "long"]
        elif condition_short:
            return [True, "short"]
        else:
            return [False, ""]

    def calculate_sl_tp_for_current_bar(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Computes SL/TP for current bar using one of the 3 VWAP Reclaim strategy variants:
        R1: candle low + 2R
        R2: entry zone + exit on VWAP loss
        R3: ATR based SL + 1.5R
        """
        if self.df_ohlc is None or self.entry_price is None:
            self.log("Missing df_ohlc or entry_price")
            return None, None

        try:
            strategy_id = self.strategy_id
            i_array = self.df_ohlc.index.get_indexer([self.entry_time])
            if i_array[0] == -1:
                self.log(f"[SL/TP ERROR] Could not find index for entry_time {self.entry_time}")
                return None, None
            entry_index = i_array[0]
            if entry_index >= len(self.df_ohlc) - 1:
                self.log(f"[SL/TP ERROR] entry_index {entry_index} out of range for df of length {len(self.df_ohlc)}")
                return None, None

            side = self.current_trade["direction"]

            if strategy_id == "05_sl_candle_low_tp_2R":
                candle_low = self.data.low[0]
                stop_ticks = self.entry_price - candle_low
                if self.p.r_multiple is None:
                    raise ValueError("r_multiple is required for R-multiple exit logic")
                result = calc_exit_by_r_multiple(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    stop_ticks=stop_ticks,
                    r_multiple=self.p.r_multiple
                )

            elif strategy_id == "06_sl_entry_zone_tp_vwaploss":
                if self.is_in_trade() and not self.position:
                    self.log(f"[WARNING] In trade but no open position? Manual check needed.")

                result = calc_exit_on_vwap_loss(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side
                )

            elif strategy_id == "07_sl_atr_tp_1.5R":
                if self.p.atr_mult_sl is None or self.p.atr_mult_tp is None:
                    raise ValueError("ATR multipliers must be set for strategy requiring ATR-based SL/TP")

                result = calc_exit_by_atr(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    atr_mult_sl=self.p.atr_mult_sl,
                    atr_mult_tp=self.p.atr_mult_tp
                )

            else:
                self.log(f"Unknown strategy_id: {strategy_id}")
                return None, None

            return result.get("sl"), result.get("tp")

        except Exception as e:
            self.log(f"[SL/TP ERROR] {e}")
            return None, None

    def is_in_trade(self) -> bool:
        return self.position or self.order

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        close = self.datas[0].close[0]

        # === Entry ===
        if not self.position and not self.order:
            bar_dt = self.datas[0].datetime.datetime(0)
            i_array = self.df_ohlc.index.get_indexer([bar_dt])
            if i_array[0] == -1:
                self.log(f"[SKIP] No index match for {bar_dt}")
                return
            i = i_array[0]

            signal, direction = self.check_entry_signal_vwap_reclaim_both_directions(self.df_ohlc, i=i)

            if signal and not self.order:
                self.log(f"[ENTRY] {self.strategy_id} {direction.upper()} @ {close:.2f}")

                if direction == "long":
                    self.order = self.buy()
                elif direction == "short":
                    self.order = self.sell()
                else:
                    self.log("[ERROR] Unknown direction; no order placed.")
                    return

                self.entry_price = close
                self.entry_time = dt
                self.current_trade = {
                    "entry_time": dt,
                    "entry_price": close,
                    "strategy": self.p.strategy_name,
                    "direction": direction
                }

                # Optional: store exit index if available
                sl, tp = self.calculate_sl_tp_for_current_bar()
                if sl is not None or tp is not None:
                    self.current_trade["exit_index"] = self.data._idx  # default to current bar unless overwritten
                    if isinstance(sl, dict):  # special case: sl might be a dict with exit_index
                        sl_data = sl
                        sl = sl_data.get("sl")
                        tp = sl_data.get("tp")
                        self.current_trade["exit_index"] = sl_data.get("exit_index", self.data._idx)

        # === Exit ===
        elif self.position:
            stop_loss, take_profit = self.calculate_sl_tp_for_current_bar()
            if stop_loss is None and take_profit is None:
                return  # No SL/TP logic triggered

            if not self.current_trade or "direction" not in self.current_trade:
                raise ValueError("Trade direction is missing in current_trade context")

            trade_dir = self.current_trade["direction"]
            current_idx = self.data._idx
            planned_exit_idx = self.current_trade.get("exit_index")

            # --- Exit for Long Trades ---
            if trade_dir == "long":
                if take_profit is not None and close >= take_profit:
                    self.log(f"[TP HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} >= {take_profit:.2f}")
                    self.order = self.sell()
                elif stop_loss is not None and close <= stop_loss:
                    self.log(f"[SL HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} <= {stop_loss:.2f}")
                    self.order = self.sell()
                elif planned_exit_idx is not None and current_idx == planned_exit_idx:
                    self.log(f"[VWAP LOSS SL] Forced Exit LONG at index {current_idx}")
                    self.order = self.sell()

            # --- Exit for Short Trades ---
            elif trade_dir == "short":
                if take_profit is not None and close <= take_profit:
                    self.log(f"[TP HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} <= {take_profit:.2f}")
                    self.order = self.buy()
                elif stop_loss is not None and close >= stop_loss:
                    self.log(f"[SL HIT] {self.strategy_id} {trade_dir.upper()} {close:.2f} >= {stop_loss:.2f}")
                    self.order = self.buy()
                elif planned_exit_idx is not None and current_idx == planned_exit_idx:
                    self.log(f"[VWAP LOSS SL] Forced Exit SHORT at index {current_idx}")
                    self.order = self.buy()

        # === Equity Tracking ===
        equity = self.broker.getvalue()
        self.equity_curve.append(equity)
        self.max_equity = max(self.max_equity, equity)
        drawdown = self.max_equity - equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: price={order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED: price={order.executed.price:.2f}")
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            self.pnl += pnl
            win = pnl > 0
            if win:
                self.wins += 1
            else:
                self.losses += 1
            trade_record = {
                "entry_time": self.entry_time,
                "exit_time": self.datas[0].datetime.datetime(0),
                "entry_price": self.entry_price,
                "exit_price": self.datas[0].close[0],
                "pnl": pnl,
                "win": win,
                "strategy": self.p.strategy_name
            }
            self.trades.append(trade_record)
            self.entry_price = None
            self.entry_time = None
            self.current_trade = None

    def stop(self):
        n_trades = len(self.trades)
        win_rate = self.wins / n_trades * 100 if n_trades > 0 else 0
        self.metrics = {
            "PnL": self.pnl,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "num_trades": n_trades,
            "strategy": self.p.strategy_name
        }
        self.log(f"Strategy stopped. PnL={self.pnl:.2f}, Win rate={win_rate:.1f}%, Max DD={self.max_drawdown:.2f}, Trades={n_trades}")


