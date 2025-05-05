import yfinance as yf
import pandas as pd
import pandas_ta as ta

SYMBOL_LIST = [
    "SPY", "QQQ", "RSP", "SDS", "SQQQ", "TLT", "GLD", "IBIT",
    "SOXL", "UUP", "DBC", "UVXY", "XBI", "AAPL", "NVDA", "TSLA", "EEM", "EFA"
]

RULE_MATRIX = {
    ("gt", "gt"): "green",
    ("gt", "lt"): "yellow",
    ("lt", "gt"): "yellow",
    ("lt", "lt"): "red"
}

class MarketMomentumScorer:
    def __init__(self, symbols=SYMBOL_LIST):
        self.symbols = symbols
        self.weekly_data = {}
        self.daily_data = {}

    def fetch_data(self):
        print("\n📥 Downloading data for symbols...")
        for symbol in self.symbols:
            try:
                weekly = yf.download(symbol, interval="1wk", period="1y", auto_adjust=True)
                daily = yf.download(symbol, interval="1d", period="1y", auto_adjust=True)

                weekly.columns = weekly.columns.get_level_values(0) if isinstance(weekly.columns, pd.MultiIndex) else [col.split()[0] for col in weekly.columns]
                daily.columns = daily.columns.get_level_values(0) if isinstance(daily.columns, pd.MultiIndex) else [col.split()[0] for col in daily.columns]

                self.weekly_data[symbol] = weekly
                self.daily_data[symbol] = daily
                print(f"✅ {symbol}: Weekly={len(weekly)} Daily={len(daily)}")
            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")

    def compute_indicators(self):
        print("\n📊 Computing SMA and Pivot indicators...")
        for name, dataset in [("weekly", self.weekly_data), ("daily", self.daily_data)]:
            for symbol, df in dataset.items():
                if df.empty:
                    continue
                df = df.copy()
                df["SMA_10"] = df["Close"].rolling(window=10).mean()
                df["SMA_20"] = df["Close"].rolling(window=20).mean()

                df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
                dataset[symbol] = df
                print(f"🔍 {symbol} [{name}] → Columns: {list(df.columns)}")

    def get_momentum_color(self, df):
        latest = df.iloc[-1]
        sma10, sma20, close = latest.get("SMA_10"), latest.get("SMA_20"), latest.get("Close")
        if pd.isna(sma10) or pd.isna(sma20) or pd.isna(close):
            return "gray"
        trend = "gt" if sma10 > sma20 else "lt"
        close_vs_sma20 = "gt" if close > sma20 else "lt"
        return RULE_MATRIX.get((trend, close_vs_sma20), "red")

    def is_near_ma(self, df):
        for i in [1, 2]:
            row = df.iloc[-i]
            if pd.isna(row.get("Low")) or pd.isna(row.get("SMA_10")):
                continue
            if abs(row["Low"] - row["SMA_10"]) / row["SMA_10"] < 0.01:
                return True
        return False

    def score_all_symbols(self):
        def build_results(data_dict, label):
            results = {}
            for symbol, df in data_dict.items():
                color = self.get_momentum_color(df)
                near_ma = self.is_near_ma(df)
                results[symbol] = {
                    "momentum_color": color,
                    "near_ma": near_ma
                }
            return results

        return build_results(self.weekly_data, "weekly"), build_results(self.daily_data, "daily")

    def build_summary_table(self):
        records = []

        for symbol in self.symbols:
            row = {"symbol": symbol}

            for timeframe, data in [("weekly", self.weekly_data), ("daily", self.daily_data)]:
                df = data.get(symbol)
                if df is None or df.empty:
                    row[f"momentum_color_{timeframe}"] = "gray"
                    row[f"touch_recent_ma_{timeframe}"] = False
                    continue

                row[f"momentum_color_{timeframe}"] = self.get_momentum_color(df)
                row[f"touch_recent_ma_{timeframe}"] = self.is_near_ma(df)

            records.append(row)

        return pd.DataFrame(records)

