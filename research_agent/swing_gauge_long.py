import yfinance as yf
import pandas as pd
import os

class SwingGaugeLong:
    def __init__(self, symbols, image_dir=None):
        """
        :param symbols: List of tradable symbols (e.g., ['SPY', 'QQQ', 'Sds'])
        :param image_dir: Optional directory path with uploaded indicator snapshots
        """
        self.symbols = symbols
        self.image_dir = image_dir
        self.yf_data = {}
        self.image_analysis = {}
        self.signal_scores = {}
        self.output = pd.DataFrame()

    def fetch_price_data(self):
        """
        Pulls historical daily and weekly data using yfinance for each symbol.
        Saves to self.yf_data
        """
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data_daily = ticker.history(period="3mo", interval="1d")
            data_weekly = ticker.history(period="6mo", interval="1wk")
            self.yf_data[symbol] = {
                "daily": data_daily,
                "weekly": data_weekly
            }

    def calculate_spy_vix_ratio(self):
        """
        Pull SPY and ^VIX from yfinance, compute ratio
        """
        spy = yf.download("SPY", period="3mo")["Close"]
        vix = yf.download("^VIX", period="3mo")["Close"]
        df = pd.DataFrame({"SPY": spy, "VIX": vix})
        df["SPY_VIX_Ratio"] = df["SPY"] / df["VIX"]
        return df

    def analyze_market_snapshot_images(self):
        """
        Placeholder for GPT-Vision-based image logic.
        For now, checks if expected files exist.
        """
        expected_files = ["nyad", "nyhl", "nysi", "bpna", "spxa150r", "nya150r"]
        for name in expected_files:
            file_path = os.path.join(self.image_dir, f"{name}.png") if self.image_dir else None
            if file_path and os.path.exists(file_path):
                self.image_analysis[name.upper()] = "✅ Found (awaiting model)"
            else:
                self.image_analysis[name.upper()] = "❌ Missing"

    def evaluate_weekly_signal(self, symbol):
        """
        Evaluate the weekly trend signal logic as described:
        +1 if EMA10 > EMA20 and close > EMA10
        +1 if close or previous green candle touched EMA10
        -1 if not
        0 if within 1.5 ATR and below pivot
        """
        df = self.yf_data[symbol]["weekly"].copy()
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["Prev_Close"] = df["Close"].shift(1)
        df["Prev_Open"] = df["Open"].shift(1)
    
        # Calculate ATR
        df["ATR"] = self.calculate_atr(df)
    
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        signal = 0
    
        # Trend condition
        if last["EMA10"] > last["EMA20"] and last["Close"] > last["EMA10"]:
            signal += 1
    
            # Touch EMA10 check
            touched = abs(last["Close"] - last["EMA10"]) < 0.01 * last["Close"] or \
                      (prev["Close"] > prev["Open"] and abs(prev["Close"] - last["EMA10"]) < 0.01 * prev["Close"])
            signal += 1 if touched else -1
    
            # Additional filter: within ATR zone but below pivot
            pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
            within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
            if within_atr:
                signal = 0
    
        self.signal_scores[symbol] = signal
        return signal

    def summarize(self):
        """
        Simple output summary of available data, images, and scores
        """
        print("🔹 Symbols fetched:", list(self.yf_data.keys()))
        print("📸 Snapshot status:")
        for k, v in self.image_analysis.items():
            print(f"  - {k}: {v}")
        print("📊 Weekly Signals:")
        for sym, score in self.signal_scores.items():
            print(f"  - {sym}: {score}")

    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR) for given OHLC DataFrame.
        """
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def check_pivot_proximity(self, symbol, timeframe="weekly"):
        """
        Checks if current close is below yesterday's pivot AND within 1.5x ATR.
        :return: True if close is below pivot and within ATR range, else False
        """
        df = self.yf_data[symbol][timeframe].copy()
        df["ATR"] = self.calculate_atr(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        # Classic pivot = (high + low + close) / 3
        pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
        
        return within_atr

    def evaluate_daily_signal(self, symbol):
        """
        Evaluate the daily entry timing logic:
        +1 if EMA10 > EMA20 and close > EMA10
        +1 if close or previous green bar touched EMA10
        -1 if not
        0 if within 1.5 ATR and below pivot
        """
        df = self.yf_data[symbol]["daily"].copy()
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["Prev_Close"] = df["Close"].shift(1)
        df["Prev_Open"] = df["Open"].shift(1)
    
        # Calculate ATR
        df["ATR"] = self.calculate_atr(df)
    
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        signal = 0
    
        # Trend alignment
        if last["EMA10"] > last["EMA20"] and last["Close"] > last["EMA10"]:
            signal += 1
    
            # Touch EMA10 check
            touched = abs(last["Close"] - last["EMA10"]) < 0.01 * last["Close"] or \
                      (prev["Close"] > prev["Open"] and abs(prev["Close"] - last["EMA10"]) < 0.01 * prev["Close"])
            signal += 1 if touched else -1
    
            # Pivot/ATR zone condition
            pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
            within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
            if within_atr:
                signal = 0
    
        return signal 