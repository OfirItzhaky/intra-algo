import sys
import os
from tabulate import tabulate
from research_agent.logging_setup import get_logger

log = get_logger(__name__)
# === Force Project Root to PYTHONPATH ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtrader as bt
import pandas as pd
import pandas_ta as ta
from backend.analyzer.analyzer_blueprint_vwap_strategy import VWAPBounceStrategy, VWAPReclaimStrategy

# === Prepare Data Feed ===
df = pd.read_csv("sample_ohlc.csv")
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df.set_index("Datetime", inplace=True)

# Add indicators
df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Vol"])
df["EMA_9"] = ta.ema(df["Close"], length=9)
df["EMA_20"] = ta.ema(df["Close"], length=20)
df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
df["volume_zscore"] = (df["Vol"] - df["Vol"].rolling(20).mean()) / df["Vol"].rolling(20).std()
df.drop(columns=["Date", "Time"], inplace=True)

# VWAP reclaim stats (for validation)
df["prev_close_vs_vwap"] = df["Close"].shift(1) - df["VWAP"].shift(1)
df["curr_close_vs_vwap"] = df["Close"] - df["VWAP"]
df["reclaim_up"] = (df["prev_close_vs_vwap"] < 0) & (df["curr_close_vs_vwap"] > 0)
df["reclaim_down"] = (df["prev_close_vs_vwap"] > 0) & (df["curr_close_vs_vwap"] < 0)
log.info("Reclaim Up:", df["reclaim_up"].sum())
log.info("Reclaim Down:", df["reclaim_down"].sum())

# === Backtrader Feed Class ===
class PandasVWAPData(bt.feeds.PandasData):
    lines = ('VWAP', 'EMA_9', 'EMA_20', 'volume_zscore', 'ATR_14',)
    params = (
        ('VWAP', -1),
        ('EMA_9', -1),
        ('EMA_20', -1),
        ('volume_zscore', -1),
        ('ATR_14', -1),
    )

data = PandasVWAPData(dataname=df)

# === Strategy Configurations ===
strategies_to_test = [
    ("vwap_bounce_01_sl_candle_low_tp_2R", VWAPBounceStrategy, dict(vwap_distance_pct=0.003, volume_zscore_min=0, r_multiple=2.0)),
    # ("vwap_bounce_02_sl_1.2atr14_tp_2R", VWAPBounceStrategy, dict(vwap_distance_pct=0.003, volume_zscore_min=0, atr_mult_sl=1.2, atr_mult_tp=2.0)),
    ("vwap_bounce_03_sl_candle_low_tp_ema9", VWAPBounceStrategy, dict(vwap_distance_pct=0.003, volume_zscore_min=0)),
    ("vwap_bounce_04_sl_1.2atr14_tp_ema9", VWAPBounceStrategy, dict(vwap_distance_pct=0.003, volume_zscore_min=0, atr_mult_sl=1.2, atr_mult_tp=1.5)),

    ("vwap_reclaim_05_sl_candle_low_tp_2R", VWAPReclaimStrategy, dict(volume_zscore_min=0.0, r_multiple=2.0)),
    ("vwap_reclaim_06_sl_entry_zone_tp_vwaploss", VWAPReclaimStrategy, dict(volume_zscore_min=0.0)),
    ("vwap_reclaim_07_sl_atr_tp_1.5R", VWAPReclaimStrategy, dict(volume_zscore_min=0.0, atr_mult_sl=1.0, atr_mult_tp=1.5)),
]

# === Run All Strategies with Metrics ===

all_results = []

for strat_name, strat_class, strat_params in strategies_to_test:
    log.info("\n" + "=" * 60)
    log.info(f"Running strategy: {strat_name}")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.addstrategy(strat_class, strategy_name=strat_name, **strat_params)
    cerebro.runonce = True  # ✅

    results = cerebro.run()
    strat = results[0]
    strat.df_ohlc = df.copy()

    # === Metrics ===
    trades = strat.trades
    num_trades = len(trades)
    tp_hits = sum(t['pnl'] > 0 for t in trades)
    sl_hits = sum(t['pnl'] < 0 for t in trades)
    avg_pnl = sum(t['pnl'] for t in trades) / num_trades if num_trades else 0
    best = max((t['pnl'] for t in trades), default=0)
    worst = min((t['pnl'] for t in trades), default=0)
    win_rate = (tp_hits / num_trades) * 100 if num_trades else 0
    all_results.append({
        "strategy": strat_name,
        "trades": num_trades,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(avg_pnl, 2),
        "best": round(best, 2),
        "worst": round(worst, 2)
    })
    log.info(f"✅ Trades: {num_trades} | TP: {tp_hits} | SL: {sl_hits} | Win%: {win_rate:.1f}%")
    log.info(f"💰 Avg PnL: {avg_pnl:.2f} | Best: {best:.2f} | Worst: {worst:.2f}")
# === Final Summary
log.info("\n" + "=" * 60)
log.info("🔍 Summary of All Strategies:\n")
log.info(tabulate(all_results, headers="keys", tablefmt="pretty", floatfmt=".2f"))