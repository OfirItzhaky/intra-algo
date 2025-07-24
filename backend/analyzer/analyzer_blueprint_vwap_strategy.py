import inspect
import math

import backtrader as bt
import pandas as pd

from backend.analyzer.analyzer_mcp_sl_tp_router import dispatch
from analyzer_vwap_sl_tp_function_tool import *

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

        # Common indicators
        self.VWAP = self.datas[0].VWAP
        self.EMA_9 = self.datas[0].EMA_9
        self.volume_zscore = self.datas[0].volume_zscore
        self.ATR_14 = self.datas[0].ATR_14

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"[{dt}] {txt}")

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
                result = calc_exit_by_r_multiple(
                    df=self.df_ohlc,
                    entry_index=entry_index,
                    entry_price=self.entry_price,
                    side=side,
                    stop_ticks=stop_ticks,
                    r_multiple=self.p.r_multiple
                )

            elif strategy_id == "02_sl_1.2atr14_tp_2R":
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
            signal, direction = self.check_entry_signal_vwap_bounce_both_directions(self.df_ohlc, i=len(self.df_ohlc) - 1)
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



