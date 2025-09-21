import backtrader as bt
import pandas as pd
import pytz
import datetime

from research_agent.config import REGRESSION_STRATEGY_DEFAULTS
from logging_setup import get_logger

log = get_logger(__name__)

class ElasticNetStrategy(bt.Strategy):
    """
    A basic strategy using predicted highs from regression models (e.g., ElasticNet),
    operating on a single timeframe (no intrabar logic yet).

    Trades are triggered when the difference between predicted and actual close falls
    within a defined range. Optionally filters trades based on classifier predictions.
    """

    params = dict(
        min_dist=3.0,
        max_dist=20.0,
        target_ticks=10,
        stop_ticks=10,
        tick_size=0.25,  # ✅ ADD THIS

        slippage=0.0,
        force_exit=True,
        session_start='10:00',
        session_end='23:00',
        tick_value=1.25,
        contract_size=1,
        min_classifier_signals=0,  # Set > 0 to use RF/LGBM/XG signals
        use_multi_class=False,
        multi_class_threshold=3,
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.open_trade_time = None
        self.trades = []

    def next(self):
        """
        Entry logic for the strategy (non-intrabar version).

        Checks for entry signals based on predicted high vs. close, and optionally
        filters by classifier signals. Creates a bracket order if criteria are met.
        """
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.strftime('%H:%M')

        # Exit at session end
        if self.position and self.p.force_exit and current_time >= self.p.session_end:
            log.info(f"[{dt}] ⏹️ Session end. Closing position.")
            self.close()
            self.order = None
            return

        if current_time < self.p.session_start or current_time >= self.p.session_end:
            return

        if self.position or self.order:
            return

        close = self.data.close[0]
        predicted = self.data.predicted_high[0]
        delta = predicted - close

        log.info(f"[{dt}] Close: {close:.2f}, Predicted: {predicted:.2f}, Delta: {delta:.2f}")

        if self.p.min_classifier_signals > 0:
            try:
                rf_val = self.data.RandomForest[0]
                lt_val = self.data.LightGBM[0]
                xg_val = self.data.XGBoost[0]

                if any(map(pd.isna, [rf_val, lt_val, xg_val])):
                    log.info(f"[{dt}] 🕳️ Skipping bar — classifier signal is NaN")
                    return

                rf = int(rf_val)
                lt = int(lt_val)
                xg = int(xg_val)

            except Exception as e:
                raise ValueError(f"❌ Error accessing classifier columns: {e}")

            green_count = rf + lt + xg

            if self.p.min_classifier_signals > 3:
                raise ValueError("❌ min_classifier_signals cannot be greater than 3.")

            if green_count < self.p.min_classifier_signals:
                log.info(f"[{dt}] 🚫 Not enough green signals ({green_count}) for entry.")
                return

        if self.p.use_multi_class:
            log.info(f"[{dt}] 🔄 Using multi-class signals with threshold {self.p.multi_class_threshold}")
            # Multi-class approach
            if hasattr(self.data, 'multi_class_label'):
                multi_class_value = self.data.multi_class_label[0]
                if pd.isna(multi_class_value):
                    log.info(f"🕳️ Skipping bar — multi_class_label is NaN")
                    return
                
                # Check if the class value meets or exceeds our threshold
                if multi_class_value < self.p.multi_class_threshold:
                    log.info(f"🚫 Multi-class signal ({multi_class_value}) below threshold ({self.p.multi_class_threshold})")
                    return
                else:
                    log.info(f"✅ Multi-class signal ({multi_class_value}) meets threshold")
            else:
                log.info(f"❌ multi_class_label not available")
                return
        else:
            log.info(f"[{dt}] 🔄 Using binary classification signals")
            # Original binary approach
            signal_count = sum(
                int(getattr(self.data, clf, [0])[0] == 1)
                for clf in ['RandomForest', 'LightGBM', 'XGBoost']
                if hasattr(self.data, clf)
            )
            if signal_count < self.p.min_classifier_signals:
                return

        tick_size = 0.25
        if self.p.min_dist <= delta <= self.p.max_dist:
            entry_price = self.data.open[1] + self.p.slippage
            stop_price = entry_price - (self.p.stop_ticks * tick_size)
            target_price = entry_price + (self.p.target_ticks * tick_size)

            log.info(f"💥 Entry signal | Entry: {entry_price:.2f}, TP: {target_price:.2f}, SL: {stop_price:.2f}")

            self.order = self.buy_bracket(
                price=entry_price,
                size=1,
                stopprice=stop_price,
                limitprice=target_price
            )
            self.entry_price = entry_price
            self.open_trade_time = dt

    def notify_order(self, order):
        """
        Handles order status updates and trade logging for the non-intrabar strategy.

        Tracks entry and exit execution, and records PnL for closed trades.
        """
        if order.status in [order.Submitted, order.Accepted]:
            log.info(f"[ORDER DEBUG] Order submitted/accepted: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}")
            return

        if order.status in [order.Completed]:
            log.info(f"[ORDER DEBUG] Order completed: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}, executed price={order.executed.price}")
            if order.isbuy():
                self.log(f"✅ BUY EXECUTED @ {order.executed.price:.2f}")
                self.entry_price = order.executed.price
                self.open_trade_time = self.data.datetime.datetime(0)
            elif order.issell():
                self.log(f"🏁 SELL EXECUTED @ {order.executed.price:.2f}")
                tick_size = 0.25
                ticks_moved = (order.executed.price - self.entry_price) / tick_size
                pnl = ticks_moved * self.p.contract_size * self.p.tick_value

                self.trades.append({
                    "entry_time": self.open_trade_time,
                    "exit_time": self.data.datetime.datetime(0),
                    "entry_price": self.entry_price,
                    "exit_price": order.executed.price,
                    "pnl": pnl
                })

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"❌ Order Failed: {order.Status[order.status]}")
            self.order = None

    def log(self, txt: str) -> None:
        """
        Logs a message with current bar timestamp.
        """
        dt = self.datas[0].datetime.datetime(0)
        log.info(f"[{dt}] {txt}")

class Long5min1minStrategy(bt.Strategy):
    params = dict(
        min_dist=3.0,
        max_dist=20.0,
        target_ticks=10,
        stop_ticks=10,
        slippage=0.0,
        force_exit=True,
        session_start='10:00',
        session_end='23:00',
        tick_value=1.25,
        contract_size=1,
        tick_size=0.25,
        min_classifier_signals=0,
        use_multi_class=False,
        multi_class_threshold=3,
        max_daily_profit=36.0,
        max_daily_loss=-36.0,
    )

    def __init__(self):
        self.order = None
        self.pending_entry = None
        self.last_bar_time = None
        self.trades = []
        self.last_entry_price = None
        self.last_exit_price = None
        self.daily_pnl = 0
        self.current_trade_date = None

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        
        if self.current_trade_date != dt.date():
            self.current_trade_date = dt.date()
            self.daily_pnl = 0
            # log.info(f"📅 New trading day: {dt.date()}, Daily PnL reset to 0")

        log.info(f"🕒 Current dt: {dt}, Previous bar time: {self.last_bar_time}")

        if self.last_bar_time == dt:
            return
        self.last_bar_time = dt

        current_time = dt.time()

        if current_time < datetime.datetime.strptime(self.p.session_start, "%H:%M").time():
            return
        if current_time >= datetime.datetime.strptime(self.p.session_end, "%H:%M").time():
            if self.position and self.p.force_exit:
                self.close()
            return

        if self.position and self.position.size > 0 and self.last_entry_price is None:
            self.last_entry_price = self.position.price
            # log.info(f"📥 Position opened at {self.last_entry_price:.2f}")
            return
        if self.position or self.order:
            return

        if dt.minute % 5 != 0:
            return

        if self.daily_pnl >= self.p.max_daily_profit:
            # log.info(f"💰 Max daily profit reached (${self.daily_pnl:.2f}). No new trades.")
            return
        if self.daily_pnl <= self.p.max_daily_loss:
            # log.info(f"⚠️ Max daily loss reached (${self.daily_pnl:.2f}). No new trades.")
            return

        pred = self.data.predicted_high[0]
        log.info(f"🔍 At {self.datas[0].datetime.datetime(0)}, Predicted High = {pred}")
        close = self.data.close[0]
        if pred is None or pd.isna(pred):
            return

        delta = pred - close
        if not (self.p.min_dist <= delta <= self.p.max_dist):
            return
        else:
            if self.p.min_classifier_signals > 0:
                if self.p.use_multi_class:
                    if hasattr(self.data, 'multi_class_label'):
                        multi_class_value = self.data.multi_class_label[0]
                        if pd.isna(multi_class_value):
                            # log.info(f"🕳️ Skipping bar — multi_class_label is NaN")
                            return
                        
                        if multi_class_value < self.p.multi_class_threshold:
                            # log.info(f"🚫 Multi-class signal ({multi_class_value}) below threshold ({self.p.multi_class_threshold})")
                            return
                        else:
                            # log.info(f"✅ Multi-class signal ({multi_class_value}) meets threshold")
                            pass
                    else:
                        # log.info(f"❌ multi_class_label not available")
                        return
                else:
                    signal_count = sum(
                        int(getattr(self.data, clf, [0])[0] == 1)
                        for clf in ['RandomForest', 'LightGBM', 'XGBoost']
                        if hasattr(self.data, clf)
                    )
                    if signal_count < self.p.min_classifier_signals:
                        return

            log.info(f"🟡 SIGNAL MATCH at {dt} → delta: {delta:.2f}, close: {close:.2f}, predicted: {pred:.2f}")
            tp = close + self.p.target_ticks * self.p.tick_size
            sl = close - self.p.stop_ticks * self.p.tick_size

            self.buy_bracket(
                exectype=bt.Order.Market,
                price=close,
                size=1,
                limitprice=tp,
                stopprice=sl
            )
            orders = list(self.broker.orders)
            orders_df = pd.DataFrame([{
                'ref': o.ref,
                'type': 'SELL' if o.issell() else 'BUY',
                'status': o.getstatusname(),
                'exec_type': o.exectype,
                'submitted_price': o.created.price,
                'filled_price': o.executed.price if o.status == bt.Order.Completed else None,
                'executed_dt': bt.num2date(o.executed.dt) if o.status == bt.Order.Completed else None
            } for o in orders])
            log.info("test")

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_time = bt.num2date(trade.dtopen, tz=pytz.UTC)
            exit_time = bt.num2date(trade.dtclose, tz=pytz.UTC)
            entry_price = self.last_entry_price
            exit_price = self.last_exit_price if self.last_exit_price is not None else trade.price
            side = self.entry_side
            # Debug: show raw trade entry/exit and stored entry
            log.info(f"[DEBUG] Raw trade exit price from trade object: {getattr(trade, 'price', 'N/A')}, Stored last_entry_price: {entry_price}, Stored last_exit_price: {self.last_exit_price}")
            log.info(f"[DEBUG] Calculated tick delta: {(exit_price - entry_price) / self.p.tick_size if entry_price is not None else 'N/A'}")
            # Manual PnL calculation
            direction_multiplier = 1 if side == 'long' else -1 if side == 'short' else 0
            pnl_dollars = (exit_price - entry_price) * direction_multiplier * self.p.tick_value / self.p.tick_size
            log.info(f"[DEBUG] Manual PnL: {pnl_dollars:.2f}")
            # Compute TP/SL for this trade
            tp_offset_ticks = 10
            sl_offset_ticks = 10
            if side == 'long':
                tp_level = entry_price + tp_offset_ticks * self.p.tick_size
                sl_level = entry_price - sl_offset_ticks * self.p.tick_size
            elif side == 'short':
                tp_level = entry_price - tp_offset_ticks * self.p.tick_size
                sl_level = entry_price + sl_offset_ticks * self.p.tick_size
            else:
                tp_level = sl_level = None
            self.daily_pnl += pnl_dollars
            trade_duration = None
            if hasattr(trade, 'dtopen') and hasattr(trade, 'dtclose'):
                trade_duration = trade.dtclose - trade.dtopen
            # Print trade summary
            log.info(f"\n🔻 TRADE COMPLETED")
            log.info(f"Entry: {entry_price:.2f}  14 Exit: {exit_price:.2f} | Side: {side} | PnL: {pnl_dollars:.2f}")
            log.info(f"TP Level: {tp_level:.2f} | SL Level: {sl_level:.2f}")
            log.info(f"Trade Duration: {trade_duration if trade_duration is not None else 'N/A'} bars")
            log.info(f"Entry Time: {entry_time}, Exit Time: {exit_time}")
            log.info(f"---")
            self.trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": side,
                "pnl": pnl_dollars,
                "tp_level": tp_level,
                "sl_level": sl_level
            })
            self.entry_dt = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            log.info(f"[ORDER DEBUG] Order submitted/accepted: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}")
            return
        if order.status in [order.Completed]:
            log.info(f"[ORDER DEBUG] Order completed: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}, executed price={order.executed.price}")
            # If this is a closing order (not the entry order), update last_exit_price
            if order.isbuy() or order.issell():
                # Heuristic: if we already have an entry price and side, and this is not the entry order, treat as exit
                if self.last_entry_price is not None and self.entry_side is not None:
                    self.last_exit_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            log.info(f"[ORDER DEBUG] Order failed: ref={order.ref}, status={order.getstatusname()}")
            self.order = None

    def log(self, txt: str) -> None:
        dt = self.datas[0].datetime.datetime(0)
        log.info(f"[{dt}] {txt}")


import backtrader as bt

class RegressionScalpingStrategy(bt.Strategy):
    params = dict(
        # Dynamic grid-varying parameters (must be passed explicitly)
        long_threshold=None,
        short_threshold=None,
        min_volume_pct_change=None,
        bar_color_filter=None,
        max_predicted_low_for_long=None,
        min_predicted_high_for_short=None,

        # Constant required parameters (from config)
        target_ticks=REGRESSION_STRATEGY_DEFAULTS['target_ticks'],
        stop_ticks=REGRESSION_STRATEGY_DEFAULTS['stop_ticks'],
        tick_size=REGRESSION_STRATEGY_DEFAULTS['tick_size'],
        tick_value=REGRESSION_STRATEGY_DEFAULTS['tick_value'],
        contract_size=REGRESSION_STRATEGY_DEFAULTS['contract_size'],
        initial_cash=REGRESSION_STRATEGY_DEFAULTS['initial_cash'],
        session_start=REGRESSION_STRATEGY_DEFAULTS['session_start'],
        session_end=REGRESSION_STRATEGY_DEFAULTS['session_end'],
        max_daily_profit=REGRESSION_STRATEGY_DEFAULTS['maxdailyprofit_dollars'],
        max_daily_loss=REGRESSION_STRATEGY_DEFAULTS['maxdailyloss_dollars'],

        # All other params explicitly defaulted to None or ''
        min_classifier_signals=None,
        slippage=REGRESSION_STRATEGY_DEFAULTS['slippage'],
        force_exit=True,
        max_drawdown=None,
        min_dist=None,
        max_dist=None,
        max_risk_per_trade=None,
        max_daily_risk=None,
        min_trades=None,
        min_profit_factor=None,
        use_multi_class=False,
        multi_class_threshold=None,
        bias_summary='',
        stop_loss_dollars=None,
        suggested_contracts=None,
        direction_tag='',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.order = None
        self.last_bar_time = None
        self.trades = []
        self.last_entry_price = None
        self.last_exit_price = None
        self.daily_pnl = 0
        self.current_trade_date = None
        self.entry_side = None
        self.entry_dt = None

        # Special dynamic thresholds
        self.max_pred_low = self.p.max_predicted_low_for_long
        self.min_pred_high = self.p.min_predicted_high_for_short

        # Strict validation of critical runtime-required params
        critical_params = [
            "long_threshold", "short_threshold",
            "target_ticks", "stop_ticks",
            "tick_size", "tick_value",
            "initial_cash", "session_start", "session_end"
        ]
        missing_params = [p for p in critical_params if getattr(self.p, p) in (None, '')]
        if missing_params:
            raise ValueError(f"❌ Missing critical params: {missing_params}")


    def next(self):
        dt5 = self.datas[0].datetime.datetime(0)  # 5m bar datetime
        # Reset daily PnL at new day
        if self.current_trade_date != dt5.date():
            self.current_trade_date = dt5.date()
            self.daily_pnl = 0
            # log.info(f"📅 New trading day: {dt5.date()}, Daily PnL reset to 0")

        # Session time checks
        current_time = dt5.time()
        session_start = datetime.datetime.strptime(self.p.session_start, "%H:%M").time()
        session_end = datetime.datetime.strptime(self.p.session_end, "%H:%M").time()
        if current_time < session_start:
            return
        if current_time >= session_end:
            if self.position and self.p.force_exit:
                self.close()
            return

        # Risk control
        if self.daily_pnl >= self.p.max_daily_profit:
            return
        if self.daily_pnl <= self.p.max_daily_loss:
            return

        # Only act on new 5m bar
        if self.last_bar_time == dt5:
            return
        self.last_bar_time = dt5

        # Only one trade at a time
        if self.position or self.order:
            return

        # --- ENTRY LOGIC ---
        try:
            pred_high = self.datas[0].predicted_high[0]
            pred_low = self.datas[0].predicted_low[0]
        except AttributeError:
            log.info("[ERROR] Data feed is missing 'predicted_high' or 'predicted_low' columns. Ensure your PandasData feed includes these as custom lines.")
            return
        close5 = self.datas[0].close[0]
        long_signal = (pred_high - close5) > self.p.long_threshold
        short_signal = (close5 - pred_low) > self.p.short_threshold

        # --- Cross-comparison filters ---
        # Only go long if predicted_low <= self.max_pred_low (if set)
        if long_signal and self.max_pred_low is not None:
            if pred_low > self.max_pred_low:
                return  # skip this long trade
        # Only go short if predicted_high >= self.min_pred_high (if set)
        if short_signal and self.min_pred_high is not None:
            if pred_high < self.min_pred_high:
                return  # skip this short trade

        # Volume and bar color filters (unchanged, but suppress debug unless trade is placed)
        vol_change_pct = None
        close = None
        open_ = None
        if self.p.min_volume_pct_change > 0:
            vol_now = self.datas[0].volume[0]
            vol_prev = self.datas[0].volume[-1]
            if vol_prev == 0:
                return
            vol_change_pct = abs((vol_now - vol_prev) / vol_prev) * 100
            if vol_change_pct < self.p.min_volume_pct_change:
                return
        if self.p.bar_color_filter:
            close = self.datas[0].close[0]
            open_ = self.datas[0].open[0]
            if long_signal and close <= open_:
                return
            if short_signal and close >= open_:
                return

        if not (long_signal or short_signal):
            return

        # Use 5m close for execution
        tp_offset_ticks = self.p.stop_ticks
        sl_offset_ticks = self.p.target_ticks
        entry_price = self.datas[0].close[0] + self.p.slippage if long_signal else self.datas[0].close[0] - self.p.slippage
        tp = entry_price + tp_offset_ticks * self.p.tick_size if long_signal else entry_price - tp_offset_ticks * self.p.tick_size
        sl = entry_price - sl_offset_ticks * self.p.tick_size if long_signal else entry_price + sl_offset_ticks * self.p.tick_size

        # Only print entry debug when a trade is about to be placed
        # try:
        #     log.info(f"\n🟢 ENTRY SIGNAL @ {self.datas[0].datetime.datetime(0)} | Side: {'LONG' if long_signal else 'SHORT'}")
        #     log.info(f"Close: {close5}, PredHigh: {pred_high}, PredLow: {pred_low}")
        #     if self.p.bar_color_filter:
        #         log.info(f"Candle Color: {'Green' if close > open_ else 'Red' if close < open_ else 'Doji'}")
        #     log.info(f"Entry: {entry_price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}")
        #     log.info(f"---")
        # except Exception as e:
        #     log.info(f"[DEBUG] Could not print entry signal values: {e}")

        # Clear any stale state before placing a new order
        self.entry_side = None
        self.last_entry_price = None
        self.entry_dt = None

        if long_signal:
            self.order = self.buy_bracket(
                data=self.datas[0],
                price=entry_price,
                size=1,
                limitprice=tp,
                stopprice=sl
            )
            self.entry_side = 'long'
            self.last_entry_price = entry_price
            self.entry_dt = dt5
        elif short_signal:
            self.order = self.sell_bracket(
                data=self.datas[0],
                price=entry_price,
                size=1,
                limitprice=tp,
                stopprice=sl
            )
            self.entry_side = 'short'
            self.last_entry_price = entry_price
            self.entry_dt = dt5

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_time = bt.num2date(trade.dtopen, tz=pytz.UTC)
            exit_time = bt.num2date(trade.dtclose, tz=pytz.UTC)
            entry_price = self.last_entry_price
            exit_price = self.last_exit_price if self.last_exit_price is not None else trade.price
            side = self.entry_side
            # Debug: show raw trade entry/exit and stored entry
            # log.info(f"[DEBUG] Raw trade exit price from trade object: {getattr(trade, 'price', 'N/A')}, Stored last_entry_price: {entry_price}, Stored last_exit_price: {self.last_exit_price}")
            # log.info(f"[DEBUG] Calculated tick delta: {(exit_price - entry_price) / self.p.tick_size if entry_price is not None else 'N/A'}")
            # Manual PnL calculation
            direction_multiplier = 1 if side == 'long' else -1 if side == 'short' else 0
            pnl_dollars = (exit_price - entry_price) * direction_multiplier * self.p.tick_value / self.p.tick_size
            # log.info(f"[DEBUG] Manual PnL: {pnl_dollars:.2f}")
            # Compute TP/SL for this trade
            tp_offset_ticks = 10
            sl_offset_ticks = 10
            if side == 'long':
                tp_level = entry_price + tp_offset_ticks * self.p.tick_size
                sl_level = entry_price - sl_offset_ticks * self.p.tick_size
            elif side == 'short':
                tp_level = entry_price - tp_offset_ticks * self.p.tick_size
                sl_level = entry_price + sl_offset_ticks * self.p.tick_size
            else:
                tp_level = sl_level = None
            self.daily_pnl += pnl_dollars
            trade_duration = None
            if hasattr(trade, 'dtopen') and hasattr(trade, 'dtclose'):
                trade_duration = trade.dtclose - trade.dtopen
            # Print trade summary
            # log.info(f"\n🔻 TRADE COMPLETED")
            # log.info(f"Entry: {entry_price:.2f}  14 Exit: {exit_price:.2f} | Side: {side} | PnL: {pnl_dollars:.2f}")
            # log.info(f"TP Level: {tp_level:.2f} | SL Level: {sl_level:.2f}")
            # log.info(f"Trade Duration: {trade_duration if trade_duration is not None else 'N/A'} bars")
            # log.info(f"Entry Time: {entry_time}, Exit Time: {exit_time}")
            # log.info(f"---")
            self.trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": side,
                "pnl": pnl_dollars,
                "tp_level": tp_level,
                "sl_level": sl_level
            })
            self.entry_dt = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # log.info(f"[ORDER DEBUG] Order submitted/accepted: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}")
            return
        if order.status in [order.Completed]:
            # log.info(f"[ORDER DEBUG] Order completed: ref={order.ref}, status={order.getstatusname()}, type={'BUY' if order.isbuy() else 'SELL'}, executed price={order.executed.price}")
            # If this is a closing order (not the entry order), update last_exit_price
            if order.isbuy() or order.issell():
                # Heuristic: if we already have an entry price and side, and this is not the entry order, treat as exit
                if self.last_entry_price is not None and self.entry_side is not None:
                    self.last_exit_price = order.executed.price
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # log.info(f"[ORDER DEBUG] Order failed: ref={order.ref}, status={order.getstatusname()}")
            self.order = None

    def log(self, txt: str) -> None:
        dt = self.datas[0].datetime.datetime(0)
        log.info(f"[{dt}] {txt}")







