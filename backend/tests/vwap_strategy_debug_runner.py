import sys
import os

# === Force Project Root to PYTHONPATH ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtrader as bt
import backtrader.feeds as btfeeds
from backend.analyzer.analyzer_blueprint_vwap_strategy import VWAPBounceStrategy

# === Prepare Data Feed ===
csv_path = "sample_ohlc.csv"

import pandas as pd
import pandas_ta as ta


class PandasVWAPData(bt.feeds.PandasData):
    lines = ('VWAP', 'EMA_9', 'EMA_20', 'volume_zscore', 'ATR_14',)
    params = (
        ('VWAP', -1),
        ('EMA_9', -1),
        ('EMA_20', -1),
        ('volume_zscore', -1),
        ('ATR_14', -1),
    )

# Load your raw test CSV
import pandas as pd
import pandas_ta as ta

df = pd.read_csv("sample_ohlc.csv", parse_dates=[["Date", "Time"]])
df = pd.read_csv("sample_ohlc.csv")
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df.set_index("Datetime", inplace=True)

df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Vol"])
df["EMA_9"] = ta.ema(df["Close"], length=9)
df["EMA_20"] = ta.ema(df["Close"], length=20)
df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
df["volume_zscore"] = (df["Vol"] - df["Vol"].rolling(20).mean()) / df["Vol"].rolling(20).std()
df.drop(columns=["Date", "Time"], inplace=True)

# Save enriched version

data = PandasVWAPData(dataname=df)

# === Setup Backtrader ===
cerebro = bt.Cerebro()
cerebro.adddata(data)

# Add strategy
cerebro.addstrategy(
    VWAPBounceStrategy,
    strategy_name="vwap_bounce_01_sl_candle_low_tp_2R",
    vwap_distance_pct=0.002,
    volume_zscore_min=0,
    r_multiple=2.0
)

# Disable preloading to inject df_ohlc earlier
cerebro.runonce = False

# Run and inject
results = cerebro.run()
strat = results[0]
strat.df_ohlc = df.copy()

