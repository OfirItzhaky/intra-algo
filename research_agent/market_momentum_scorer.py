import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time

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

    import time

    def fetch_data(self):
        import time
        import yfinance as yf
        import pandas as pd
        from alpha_vantage.timeseries import TimeSeries

        print("\n📥 Downloading data for symbols...")
        alpha_ts = TimeSeries(key="4E6RVEHAR2K6R5O7", output_format="pandas")

        for symbol in self.symbols:
            weekly, daily = pd.DataFrame(), pd.DataFrame()

            # --- Weekly ---
            print(f"⏳ Fetching {symbol} weekly data via YF...")
            try:
                yf_weekly = yf.download(symbol, interval="1wk", period="1y", auto_adjust=True)
                if not yf_weekly.empty:
                    weekly = yf_weekly
                    print(f"✅ YF weekly fetched for {symbol}")
                else:
                    raise ValueError("YF weekly empty")
            except Exception as e:
                print(f"⚠️ YF weekly failed for {symbol}: {e}. Trying Alpha Vantage...")
                try:
                    av_weekly, _ = alpha_ts.get_weekly(symbol=symbol)
                    av_weekly = av_weekly.rename(columns={
                        '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                        '4. close': 'Close', '6. volume': 'Volume'
                    })
                    weekly = av_weekly[::-1]
                    print(f"✅ AV weekly fallback success for {symbol}")
                except Exception as e2:
                    raise RuntimeError(f"❌ Both YF and AV weekly failed for {symbol}: {e2}")

            # --- Daily ---
            print(f"⏳ Fetching {symbol} daily data via YF...")
            try:
                yf_daily = yf.download(symbol, interval="1d", period="1y", auto_adjust=True)
                if not yf_daily.empty:
                    daily = yf_daily
                    print(f"✅ YF daily fetched for {symbol}")
                else:
                    raise ValueError("YF daily empty")
            except Exception as e:
                print(f"⚠️ YF daily failed for {symbol}: {e}. Trying Alpha Vantage...")
                try:
                    av_daily, _ = alpha_ts.get_daily(symbol=symbol, outputsize="full")
                    av_daily = av_daily.rename(columns={
                        '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                        '4. close': 'Close', '6. volume': 'Volume'
                    })
                    daily = av_daily[::-1]
                    print(f"✅ AV daily fallback success for {symbol}")
                except Exception as e2:
                    raise RuntimeError(f"❌ Both YF and AV daily failed for {symbol}: {e2}")

            # Clean column names
            weekly.columns = weekly.columns.get_level_values(0) if isinstance(weekly.columns, pd.MultiIndex) else [
                col.split()[0] for col in weekly.columns]
            daily.columns = daily.columns.get_level_values(0) if isinstance(daily.columns, pd.MultiIndex) else [
                col.split()[0] for col in daily.columns]

            self.weekly_data[symbol] = weekly
            self.daily_data[symbol] = daily
            print(f"✅ {symbol}: Weekly={len(weekly)} Daily={len(daily)}")
            time.sleep(1.5)  # Prevent AV rate limits

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

