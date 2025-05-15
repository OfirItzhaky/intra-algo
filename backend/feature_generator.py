import numpy as np
import pandas as pd
import pandas_ta as ta
class FeatureGenerator:
    """
    A class to generate technical indicators and features for trading simulations.
    """

    def __init__(self):
        """
        Initializes the FeatureGenerator class.
        """
        self.df_with_labels = None  # ✅ Store processed dataset for classification

        print("✅ FeatureGenerator initialized!")

    def calculate_indicators(self, df):
        """
        Generates technical indicators for a given DataFrame with Open, High, Low, Close, and Volume.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing required price and volume data.

        Returns:
        pd.DataFrame: Updated DataFrame with calculated technical indicators.
        """

        df = df.copy()

        # ✅ Moving Averages (Fast, Slow, EMA)
        df["FastAvg"] = df["Close"].rolling(window=9).mean()
        df["SlowAvg"] = df["Close"].rolling(window=18).mean()
        df["FastEMA"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["MedEMA"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["SlowEMA"] = df["Close"].ewm(span=50, adjust=False).mean()

        # ✅ ADX & DMI
        adx_indicators = df.ta.adx()
        df["ADX"] = adx_indicators["ADX_14"]
        df["DMI_plus"] = adx_indicators["DMP_14"]
        df["DMI_minus"] = adx_indicators["DMN_14"]

        # ✅ Aroon Indicators
        aroon_indicators = df.ta.aroon()
        df["AroonUp"] = aroon_indicators["AROONU_14"]
        df["AroonDn"] = aroon_indicators["AROOND_14"]
        df["AroonOsc"] = aroon_indicators["AROONOSC_14"]

        print("✅ Successfully calculated all technical indicators with correct column names!")
        return df

    def calculate_indicators(self, df):
        """
        Generates technical indicators for a given DataFrame with Open, High, Low, Close, and Volume.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing required price and volume data.

        Returns:
        pd.DataFrame: Updated DataFrame with calculated technical indicators.
        """
        import pandas_ta as ta

        df = df.copy()

        # ✅ Moving Averages (Fast, Slow, EMA)
        df["FastAvg"] = df["Close"].rolling(window=9).mean()
        df["SlowAvg"] = df["Close"].rolling(window=18).mean()
        df["FastEMA"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["MedEMA"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["SlowEMA"] = df["Close"].ewm(span=50, adjust=False).mean()

        # ✅ ADX & DMI
        adx_indicators = df.ta.adx()
        df["ADX"] = adx_indicators["ADX_14"]
        df["DMI_plus"] = adx_indicators["DMP_14"]
        df["DMI_minus"] = adx_indicators["DMN_14"]

        # ✅ Aroon Indicators
        aroon_indicators = df.ta.aroon()
        df["AroonUp"] = aroon_indicators["AROONU_14"]
        df["AroonDn"] = aroon_indicators["AROOND_14"]
        df["AroonOsc"] = aroon_indicators["AROONOSC_14"]

        print("✅ Successfully calculated all technical indicators with correct column names!")
        return df

    def add_constant_columns(self, df, constants=None):
        """
        Adds predefined columns with constant values to a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        constants (dict, optional): Dictionary of column names and their constant values.
                                    Default values:
                                    {
                                        "OverBot": 70,
                                        "OverSld": 30,
                                        "OverBought": 70,
                                        "OverSold": 30,
                                        "Zero Cross": 0,
                                        "CROSS FIB.1": 0,
                                        "CROSS FIB.2": 0
                                    }

        Returns:
        pd.DataFrame: Updated DataFrame with the added columns.
        """
        df = df.copy()  # Avoid modifying the original dataset

        if constants is None:
            constants = {
                "OverBot": 70,
                "OverSld": 30,
                "OverBought": 70,
                "OverSold": 30,
                "Zero Cross": 0,
                "CROSS FIB.1": 0,
                "CROSS FIB.2": 0
            }

        for col, value in constants.items():
            df[col] = value

        print(f"✅ Added constant columns: {list(constants.keys())}")
        return df


    def add_macd_indicators(self, df, macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Adds MACD indicators (MACD, MACDAvg, MACDDiff) to the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing 'Close' price.
            macd_fast (int, optional): Fast EMA period for MACD (default: 12).
            macd_slow (int, optional): Slow EMA period for MACD (default: 26).
            macd_signal (int, optional): Signal line EMA period (default: 9).

        Returns:
            pd.DataFrame: Updated DataFrame with MACD-related indicators.
        """
        df = df.copy()  # ✅ Avoid modifying the original dataset

        # ✅ Compute MACD Line (Fast EMA - Slow EMA)
        df["MACD"] = df["Close"].ewm(span=macd_fast, adjust=False).mean() - df["Close"].ewm(span=macd_slow,
                                                                                            adjust=False).mean()

        # ✅ Compute MACD Signal Line (9-period EMA of MACD)
        df["MACDAvg"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

        # ✅ Compute MACD Histogram (MACD - MACD Signal Line)
        df["MACDDiff"] = df["MACD"] - df["MACDAvg"]

        print("✅ Successfully added MACD indicators!")
        return df

    def add_multi_ema_indicators(self, df, ema_periods=None):
        """
        Adds multiple Exponential Moving Averages (EMAs) for the Close price
        and their distance from both the Close and the Open price.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing 'Close' and 'Open' prices.
        ema_periods (list, optional): List of EMA periods to compute. Default: [5, 10, 15, ..., 50].

        Returns:
        pd.DataFrame: Updated DataFrame with EMAs and their distances from Close and Open.
        """
        df = df.copy()

        # Default EMA timeframes if not provided
        if ema_periods is None:
            ema_periods = list(range(5, 55, 5))  # 5, 10, 15, ..., 50

        # Compute EMAs and distances
        for period in ema_periods:
            ema_col = f"EMA_{period}"
            close_distance_col = f"Close_vs_EMA_{period}"
            open_distance_col = f"Open_vs_EMA_{period}"

            # Compute EMA based on Close price
            df[ema_col] = df["Close"].ewm(span=period, adjust=False).mean()

            # Distances: Close - EMA, Open - EMA
            df[close_distance_col] = df["Close"] - df[ema_col]
            df[open_distance_col] = df["Open"] - df[ema_col]

        print("✅ Added EMAs (5-50) and their Distances (Close/Open) Successfully!")
        return df

    def add_high_based_indicators_combined(self, df, ema_periods=None, rolling_periods=None):
        """
        Computes High-based EMA indicators, rolling highs, and periodic highs
        while ensuring column names match the original dataset.

        Parameters:
            df (pd.DataFrame): The input dataset containing 'High', 'Date', and 'Time'.
            ema_periods (list, optional): List of EMA periods for High price. Default: [5, 10, ..., 50].
            rolling_periods (list, optional): List of rolling periods for Highest Highs. Default: [3, 4, 5, 6, 8, 10, 12, 15, 20].

        Returns:
            pd.DataFrame: Updated dataset with added high-based indicators.
        """
        df = df.copy()

        # Default EMA timeframes if not provided
        if ema_periods is None:
            ema_periods = list(range(5, 55, 5))

        # Default rolling periods if not provided
        if rolling_periods is None:
            rolling_periods = [3, 4, 5, 6, 8, 10, 12, 15, 20]

        # Compute EMAs Based on High Price
        for period in ema_periods:
            ema_col = f"EMA_{period}_High"
            distance_col = f"High_vs_{ema_col}"

            # Compute EMA of the High price
            df[ema_col] = df["High"].ewm(span=period, adjust=False).mean()

            # Compute the distance between High and its EMA
            df[distance_col] = df["High"] - df[ema_col]

        # Expand Rolling Highest Highs for More Lookbacks
        for period in rolling_periods:
            rolling_high_col = f"High_Max_{period}"
            df[rolling_high_col] = df["High"].rolling(window=period).max()

        # Compute Periodic Highs (Hourly, 15-Min, Daily)
        df["Hour"] = pd.to_datetime(df["Date"] + " " + df["Time"]).dt.hour
        df["Minute"] = pd.to_datetime(df["Date"] + " " + df["Time"]).dt.minute

        # Hourly High
        df["High_Hourly"] = df.groupby(["Date", "Hour"])["High"].transform("max")

        # 15-Minute High
        df["Minute_Bucket"] = (df["Minute"] // 15) * 15
        df["High_15Min"] = df.groupby(["Date", "Hour", "Minute_Bucket"])["High"].transform("max")

        # Daily High
        df["High_Daily"] = df.groupby(["Date"])["High"].transform("max")

        # Drop unnecessary time columns
        df = df.drop(columns=["Hour", "Minute", "Minute_Bucket"], errors="ignore")

        # Drop NaN values that may arise from rolling calculations
        # df = df.dropna()

        print("✅ Successfully added High-Based Indicators (Unified Method)!")
        return df

    def remove_exactly_correlated_features(self, df):   #todo:currently not used
        """
        Removes only features that are perfectly correlated (correlation = 1.0), keeping one.

        Parameters:
        df (pd.DataFrame): The dataset with features.

        Returns:
        pd.DataFrame: DataFrame with only unique features.
        """
        df = df.copy()

        # Compute feature correlation matrix
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        features_to_drop = set()
        for column in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[column] == 1.0].tolist()
            if correlated_features:
                features_to_drop.add(column)

        print(f"📊 Removing {len(features_to_drop)} features with perfect correlation (1.0): {features_to_drop}")

        # Drop the selected features
        df = df.drop(columns=features_to_drop, errors="ignore")

        return df

    def add_volatility_momentum_volume_features(self, df, williams_period=14, volume_period=20):
        """
        Adds Williams %R and Relative Volume features, using ATR computed earlier.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing 'High', 'Low', 'Open', 'Close', 'ATR'.

        Returns:
        pd.DataFrame: Updated DataFrame with volatility, momentum, and volume-based indicators.
        """
        df = df.copy()  # ✅ Avoid modifying the original dataset

        # ✅ Ensure ATR is already computed
        if "ATR" not in df.columns:
            print("⚠️ ATR is missing! Ensure ATR is computed before adding volatility features.")
            return df

        # ✅ Compute Williams %R
        high_n = df["High"].rolling(window=williams_period).max()
        low_n = df["Low"].rolling(window=williams_period).min()
        df["Williams_R"] = ((high_n - df["Close"]) / (high_n - low_n)) * -100

        # ✅ Compute Relative Volume using "Volume"
        if "Volume" in df.columns:
            df["SMA_Volume"] = df["Volume"].rolling(window=volume_period).mean()
            df["Relative_Volume"] = df["Volume"] / df["SMA_Volume"]
        else:
            print("⚠️ Warning: 'Volume' column not found, Relative Volume not calculated!")

        # ✅ Drop intermediate calculation columns
        df = df.drop(columns=["SMA_Volume"], errors="ignore")

        print("✅ Successfully added volatility, Williams %R, and Relative Volume features!")
        return df

    def add_ichimoku_cloud(self, df, price_high="High", price_low="Low", price_close="Close", tenkan_period=9,
                           kijun_period=26, senkou_b_period=52):
        """
        Computes Ichimoku Cloud indicators and adds engineered features.
        Only keeps the top 5 engineered ones based on prior gain evaluation.
        """
        df = df.copy()

        # === Step 1: Raw Ichimoku calculation
        df["Tenkan"] = (df[price_high].rolling(window=tenkan_period).max() + df[price_low].rolling(
            window=tenkan_period).min()) / 2
        df["Kijun"] = (df[price_high].rolling(window=kijun_period).max() + df[price_low].rolling(
            window=kijun_period).min()) / 2
        df["Chikou"] = df[price_close].shift(-kijun_period)
        df["SenkouSpan_A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(kijun_period)
        df["SenkouSpan_B"] = ((df[price_high].rolling(window=senkou_b_period).max() +
                               df[price_low].rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)

        # === Step 2: Add only top engineered features
        if all(col in df.columns for col in ["Chikou", "Close", "SenkouSpan_A", "SenkouSpan_B"]):
            df["Chikou_minus_Close"] = df["Chikou"] - df["Close"]
            df["Chikou_vs_Close_sign"] = (df["Chikou"] > df["Close"]).astype(int)
            df["Chikou_vs_SpanA"] = df["Chikou"] - df["SenkouSpan_A"]
            df["Chikou_gt_SpanA"] = (df["Chikou"] > df["SenkouSpan_A"]).astype(int)
            df["SpanA_gt_SpanB"] = (df["SenkouSpan_A"] > df["SenkouSpan_B"]).astype(int)

        # === Step 3: Drop raw Ichimoku components
        df.drop(columns=[
            "Tenkan", "Kijun", "Chikou", "SenkouSpan_A", "SenkouSpan_B"
        ], errors="ignore", inplace=True)

        print("✅ Ichimoku Cloud indicators added successfully (Top 5 features only)!")
        return df

    import pandas_ta as ta

    def add_atr_price_features(self, df, atr_period=14):
        """
        Computes ATR using pandas-ta and adds ATR-based price levels to the dataset.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing 'High', 'Low', 'Open', 'Close'.
        atr_period (int, optional): The ATR calculation period (default: 14).

        Returns:
        pd.DataFrame: Updated DataFrame with ATR and ATR-adjusted price features.
        """
        df = df.copy()  # ✅ Avoid modifying the original dataset
        # ✅ Ensure 'Previous Close' is available
        df["Prev_Close"] = df["Close"].shift(1)
        # ✅ Compute True Range (TR)
        df["TR1"] = df["High"] - df["Low"]
        df["TR2"] = abs(df["High"] - df["Prev_Close"])
        df["TR3"] = abs(df["Low"] - df["Prev_Close"])
        df["True_Range"] = df[["TR1", "TR2", "TR3"]].max(axis=1)

        # ✅ Compute ATR using SMA (Rolling Window)
        df["ATR"] = df["True_Range"].rolling(window=atr_period).mean()

        # ✅ Ensure ATR is available before creating new features
        if df["ATR"].isna().all():
            print("⚠️ ATR calculation failed! Ensure input data is valid.")
            return df

        # ✅ Compute ATR-based price levels
        df["ATR+ High"] = df["High"] + df["ATR"]
        df["ATR+ Low"] = df["Low"] - df["ATR"]
        df["ATR+ Close"] = df["Close"] + df["ATR"]
        df["ATR+ Open"] = df["Open"] + df["ATR"]


        print("✅ Successfully computed ATR and added ATR-adjusted price features!")
        return df

    def add_vwap_features(self, df):
        """
        Adds VWAP to a DataFrame, ensuring it matches the original dataset.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing 'High', 'Low', 'Close', and 'Volume'.

        Returns:
        pd.DataFrame: Updated DataFrame with VWAP.
        """
        df = df.copy()  # ✅ Avoid modifying the original dataset

        # ✅ Compute Typical Price
        df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3

        # ✅ Compute Cumulative Price x Volume
        df["Cum_PxV"] = (df["Typical_Price"] * df["Volume"]).cumsum()

        # ✅ Compute Cumulative Volume
        df["Cum_Volume"] = df["Volume"].cumsum()

        # ✅ Calculate VWAP (Renamed to match `data_labeled`)
        df["VWAP"] = df["Cum_PxV"] / df["Cum_Volume"]


        print("✅ VWAP added successfully!")
        return df

    def add_cci_average(self, df, cci_length=14, cci_avg_length=9):
        """
        Computes the CCI and its moving average.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        cci_length (int, optional): The length used to compute the CCI (default is 14).
        cci_avg_length (int, optional): The length used for the CCI moving average (default is 9).

        Returns:
        pd.DataFrame: Updated DataFrame with 'CCI' and 'CCI_Avg' columns.
        """

        # ✅ Always Compute CCI (Even if it already exists)
        print(f"🔹 Computing CCI with length={cci_length}...")
        df["CCI"] = df.ta.cci(length=cci_length)

        # ✅ Compute CCI Average (Rolling mean of CCI)
        df["CCI_Avg"] = df["CCI"].rolling(window=cci_avg_length).mean()

        print(f"✅ Successfully computed CCI and CCI_Avg (CCI: {cci_length}, CCI_Avg: {cci_avg_length})")
        return df

    def add_fibonacci_levels(self, df, price_high="High", price_low="Low", length=50, retrace=0.382):
        """
        Computes Fibonacci retracement levels using the highest and lowest prices over a rolling window.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing 'High' and 'Low' price columns.

        Returns:
        pd.DataFrame: Updated DataFrame with Fibonacci retracement features.
        """
        df = df.copy()

        df["Fibo_High"] = df[price_high].rolling(window=length).max()
        df["Fibo_Low"] = df[price_low].rolling(window=length).min()

        # ✅ Rename Fibonacci Bands
        df["UpperBand.2"] = df["Fibo_High"]
        df["LowerBand.2"] = df["Fibo_Low"]

        print(f"✅ Fibonacci levels added successfully! (Length={length}, Retrace={retrace})")
        return df

    def create_all_features(self, df):
        """
        Runs all feature engineering methods in sequence on the input DataFrame.

        @param df: The DataFrame containing raw market data (Date, Time, Open, High, Low, Close, Volume).
        @return: Processed DataFrame with all added features.
        """
        print("🚀 Starting full feature generation pipeline...")

        # Step 1: Start with price-only data (this is already the input `df`)
        df_features = df.copy()

        # Step 2: Re-run indicator calculations
        df_features = self.calculate_indicators(df_features)

        # Step 3: Add VWAP Features (MedEMA is now available)
        df_features = self.add_vwap_features(df_features)

        # Step 4: Add Fibonacci Levels
        df_features = self.add_fibonacci_levels(df_features)

        # Step 5: Add CCI Average
        df_features = self.add_cci_average(df_features)

        # Step 6: Add Ichimoku Cloud Features
        df_features = self.add_ichimoku_cloud(df_features)

        # Step 7: Add ATR-Based Price Features
        df_features = self.add_atr_price_features(df_features)

        # Step 8: Add Multiple EMA Indicators
        df_features = self.add_multi_ema_indicators(df_features)

        # Step 9: Add High-Based Indicators (Unified)
        df_features = self.add_high_based_indicators_combined(df_features)

        # Step 10: Add Constant Columns
        df_features = self.add_constant_columns(df_features)

        # Step 11: Add MACD Indicators
        df_features = self.add_macd_indicators(df_features)

        # Step 12: Add Volatility, Momentum, and Volume Features
        df_features = self.add_volatility_momentum_volume_features(df_features)

        # ✅ Print Summary of Features
        print(f"✅ Feature generation complete! Final DataFrame: {df_features.shape[1]} columns")

        return df_features



