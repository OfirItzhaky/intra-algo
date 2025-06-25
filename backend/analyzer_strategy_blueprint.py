import backtrader as bt
import pandas as pd
from matplotlib.dates import num2date
from backtrader.utils.date import num2date
import pytz
import datetime

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
            print(f"[{dt}] ⏹️ Session end. Closing position.")
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

        print(f"[{dt}] Close: {close:.2f}, Predicted: {predicted:.2f}, Delta: {delta:.2f}")

        if self.p.min_classifier_signals > 0:
            try:
                rf_val = self.data.RandomForest[0]
                lt_val = self.data.LightGBM[0]
                xg_val = self.data.XGBoost[0]

                if any(map(pd.isna, [rf_val, lt_val, xg_val])):
                    print(f"[{dt}] 🕳️ Skipping bar — classifier signal is NaN")
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
                print(f"[{dt}] 🚫 Not enough green signals ({green_count}) for entry.")
                return

        if self.p.use_multi_class:
            print(f"[{dt}] 🔄 Using multi-class signals with threshold {self.p.multi_class_threshold}")
            # Multi-class approach
            if hasattr(self.data, 'multi_class_label'):
                multi_class_value = self.data.multi_class_label[0]
                if pd.isna(multi_class_value):
                    print(f"🕳️ Skipping bar — multi_class_label is NaN")
                    return
                
                # Check if the class value meets or exceeds our threshold
                if multi_class_value < self.p.multi_class_threshold:
                    print(f"🚫 Multi-class signal ({multi_class_value}) below threshold ({self.p.multi_class_threshold})")
                    return
                else:
                    print(f"✅ Multi-class signal ({multi_class_value}) meets threshold")
            else:
                print(f"❌ multi_class_label not available")
                return
        else:
            print(f"[{dt}] 🔄 Using binary classification signals")
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

            print(f"💥 Entry signal | Entry: {entry_price:.2f}, TP: {target_price:.2f}, SL: {stop_price:.2f}")

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
            return

        if order.status in [order.Completed]:
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
        print(f"[{dt}] {txt}")

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
            print(f"📅 New trading day: {dt.date()}, Daily PnL reset to 0")

        print(f"🕒 Current dt: {dt}, Previous bar time: {self.last_bar_time}")

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
            print(f"📥 Position opened at {self.last_entry_price:.2f}")
            return
        if self.position or self.order:
            return

        if dt.minute % 5 != 0:
            return

        if self.daily_pnl >= self.p.max_daily_profit:
            print(f"💰 Max daily profit reached (${self.daily_pnl:.2f}). No new trades.")
            return
        if self.daily_pnl <= self.p.max_daily_loss:
            print(f"⚠️ Max daily loss reached (${self.daily_pnl:.2f}). No new trades.")
            return

        pred = self.data.predicted_high[0]
        print(f"🔍 At {self.datas[0].datetime.datetime(0)}, Predicted High = {pred}")
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
                            print(f"🕳️ Skipping bar — multi_class_label is NaN")
                            return
                        
                        if multi_class_value < self.p.multi_class_threshold:
                            print(f"🚫 Multi-class signal ({multi_class_value}) below threshold ({self.p.multi_class_threshold})")
                            return
                        else:
                            print(f"✅ Multi-class signal ({multi_class_value}) meets threshold")
                    else:
                        print(f"❌ multi_class_label not available")
                        return
                else:
                    signal_count = sum(
                        int(getattr(self.data, clf, [0])[0] == 1)
                        for clf in ['RandomForest', 'LightGBM', 'XGBoost']
                        if hasattr(self.data, clf)
                    )
                    if signal_count < self.p.min_classifier_signals:
                        return

            print(f"🟡 SIGNAL MATCH at {dt} → delta: {delta:.2f}, close: {close:.2f}, predicted: {pred:.2f}")
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
            print("test")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.daily_pnl += trade.pnl
            print(f"💵 Trade PnL: ${trade.pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}")

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

            last_filled_sell = orders_df[(orders_df['type'] == 'SELL') & (orders_df['status'] == 'Completed')].iloc[-1]
            exit_price_from_orders = last_filled_sell['filled_price']

            pnl = trade.pnl
            entry_time = bt.num2date(trade.dtopen, tz=pytz.UTC)
            exit_time = bt.num2date(trade.dtclose, tz=pytz.UTC)

            entry_price = self.last_entry_price
            exit_price = trade.price
            self.exit_orders_sent = False

            print(f"🌐 Trade closed. PnL: {pnl:.2f}")
            print(f"🧾 Stored from trade — Entry: {entry_price}, Exit: {exit_price}")

            self.trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price_from_orders,
                "pnl": pnl
            })

class RegressionScalpingStrategy(bt.Strategy):
    params = dict(
        long_threshold=3.0,
        short_threshold=3.0,
        target_ticks=10,
        stop_ticks=10,
        tick_size=0.25,
        session_start='10:00',
        session_end='23:00',
        max_daily_profit=36.0,
        max_daily_loss=-36.0,
        slippage=0.0,
        force_exit=True,
    )

    def __init__(self):
        self.order = None
        self.last_bar_time = None
        self.trades = []
        self.last_entry_price = None
        self.last_exit_price = None
        self.daily_pnl = 0
        self.current_trade_date = None
        self.entry_side = None  # 'long' or 'short'
        self.entry_dt = None

    def next(self):
        dt5 = self.datas[0].datetime.datetime(0)  # 5m bar datetime
        dt1 = self.datas[1].datetime.datetime(0)  # 1m bar datetime (should be aligned)

        # Reset daily PnL at new day
        if self.current_trade_date != dt5.date():
            self.current_trade_date = dt5.date()
            self.daily_pnl = 0
            print(f"📅 New trading day: {dt5.date()}, Daily PnL reset to 0")

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
            print(f"💰 Max daily profit reached (${self.daily_pnl:.2f}). No new trades.")
            return
        if self.daily_pnl <= self.p.max_daily_loss:
            print(f"⚠️ Max daily loss reached (${self.daily_pnl:.2f}). No new trades.")
            return

        # Only act on new 5m bar
        if self.last_bar_time == dt5:
            return
        self.last_bar_time = dt5

        # Only one trade at a time
        if self.position or self.order:
            return

        # --- ENTRY LOGIC ---
        close5 = self.datas[0].close[0]
        pred_high = self.datas[0].predicted_high[0]
        pred_low = self.datas[0].predicted_low[0]
        long_signal = (pred_high - close5) > self.p.long_threshold
        short_signal = (close5 - pred_low) > self.p.short_threshold

        if not (long_signal or short_signal):
            return

        # Use 1m data for execution
        open1 = self.datas[1].open[0]
        entry_price = open1 + self.p.slippage if long_signal else open1 - self.p.slippage
        tp = entry_price + self.p.target_ticks * self.p.tick_size if long_signal else entry_price - self.p.target_ticks * self.p.tick_size
        sl = entry_price - self.p.stop_ticks * self.p.tick_size if long_signal else entry_price + self.p.stop_ticks * self.p.tick_size

        if long_signal:
            print(f"🟢 LONG SIGNAL at {dt5} | Entry: {entry_price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}")
            self.order = self.buy_bracket(
                data=self.datas[1],
                price=entry_price,
                size=1,
                limitprice=tp,
                stopprice=sl
            )
            self.entry_side = 'long'
            self.entry_dt = dt1
            self.last_entry_price = entry_price
        elif short_signal:
            print(f"🔴 SHORT SIGNAL at {dt5} | Entry: {entry_price:.2f}, TP: {tp:.2f}, SL: {sl:.2f}")
            self.order = self.sell_bracket(
                data=self.datas[1],
                price=entry_price,
                size=1,
                limitprice=tp,
                stopprice=sl
            )
            self.entry_side = 'short'
            self.entry_dt = dt1
            self.last_entry_price = entry_price

    def notify_trade(self, trade):
        if trade.isclosed:
            self.daily_pnl += trade.pnl
            print(f"💵 Trade PnL: ${trade.pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}")
            entry_time = bt.num2date(trade.dtopen, tz=pytz.UTC)
            exit_time = bt.num2date(trade.dtclose, tz=pytz.UTC)
            self.trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": self.last_entry_price,
                "exit_price": trade.price,
                "side": self.entry_side,
                "pnl": trade.pnl
            })
            self.last_exit_price = trade.price
            self.order = None
            self.entry_side = None
            self.entry_dt = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"✅ BUY EXECUTED @ {order.executed.price:.2f}")
            elif order.issell():
                print(f"✅ SELL EXECUTED @ {order.executed.price:.2f}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"❌ Order Failed: {order.Status[order.status]}")
            self.order = None

    def log(self, txt: str) -> None:
        dt = self.datas[0].datetime.datetime(0)
        print(f"[{dt}] {txt}")







