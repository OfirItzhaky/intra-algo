import backtrader as bt
import pandas as pd

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
        min_classifier_signals=0  # Set > 0 to use RF/LGBM/XG signals
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

import backtrader as bt
from typing import Optional, List, Dict


class ElasticNetIntrabarStrategy(bt.Strategy):
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
        min_classifier_signals=3,
        tick_size=0.25,
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.open_trade_time = None
        self.trades = []
        self.last_entry_time = None

    def next(self):
        dt_5min = self.datas[0].datetime.datetime(0)
        dt_1min = self.datas[1].datetime.datetime(0)

        print(f"🧠 Bar time: {dt_5min} | {dt_1min}")
        print(f"🧪 Checking bar: {dt_5min} → minute = {dt_5min.minute}")

        # ✅ Skip if already processed this 5-min bar
        if self.last_entry_time == dt_5min:
            return
        self.last_entry_time = dt_5min

        # ✅ Only allow entries at 5-minute intervals
        if dt_5min.minute % 5 != 0:
            print(f"🚫 Skipping non-5-min bar: {dt_5min}")
            return

        current_time = dt_5min.strftime('%H:%M')

        if self.position and self.p.force_exit and current_time >= self.p.session_end:
            self.close()
            self.order = None
            return

        if self.position or self.order:
            return

        main = self.datas[0]
        intrabar = self.datas[1]

        close = main.close[0]
        predicted = main.predicted_high[0]
        delta = predicted - close

        print(f"🔬 At {dt_5min} — Close: {close:.2f}, Predicted: {predicted:.2f}, Delta: {delta:.2f}")

        if self.p.min_classifier_signals > 0:
            try:
                green_count = sum(
                    int(self.datas[0].__getattr__(clf)[0]) for clf in ['RandomForest', 'LightGBM', 'XGBoost'])
            except Exception:
                return
            if green_count < self.p.min_classifier_signals:
                return

        if self.p.min_dist <= delta <= self.p.max_dist:
            print(f"🔍 Trade candidate found at {dt_5min}, delta: {delta:.2f}")
            entry_price = main.open[0] + self.p.slippage
            tp = entry_price + self.p.target_ticks * self.p.tick_size
            sl = entry_price - self.p.stop_ticks * self.p.tick_size
            entry_time = intrabar.datetime.datetime(0)

            for i in range(1, 6):
                if len(intrabar) <= i:
                    break
                hi = intrabar.high[i]
                lo = intrabar.low[i]
                print(f"⏱ Scanning 1-min at {intrabar.datetime.datetime(i)} → hi: {hi}, lo: {lo}")

                exit_price = None
                if lo <= sl:
                    exit_price = sl
                elif hi >= tp:
                    exit_price = tp

                if exit_price is not None:
                    ticks = (exit_price - entry_price) / self.p.tick_size
                    pnl = ticks * self.p.contract_size * self.p.tick_value
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": intrabar.datetime.datetime(i),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                    print(
                        f"✅ Trade recorded — ENTRY: {entry_time}, EXIT: {intrabar.datetime.datetime(i)}, PnL: {pnl:.2f}")
                    return




