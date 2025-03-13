from dataclasses import dataclass
from typing import Optional
import pandas_ta as ta  # Ensure pandas_ta is available
import pandas as pd


@dataclass
class NewBar:
    """Data class to store the latest market bar for feature processing and predictions."""

    # Core Price Data
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Date: str
    Time: str

    # Technical Indicators (Initially None, Will Be Computed)
    FastAvg: Optional[float] = None
    SlowAvg: Optional[float] = None
    FastEMA: Optional[float] = None
    MedEMA: Optional[float] = None
    SlowEMA: Optional[float] = None
    ADX: Optional[float] = None
    DMI_minus: Optional[float] = None
    DMI_plus: Optional[float] = None
    AroonUp: Optional[float] = None
    AroonDn: Optional[float] = None
    AroonOsc: Optional[float] = None
    VWAP: Optional[float] = None
    CCI: Optional[float] = None
    CCI_Avg: Optional[float] = None
    ATR: Optional[float] = None
    MACD: Optional[float] = None
    MACDAvg: Optional[float] = None
    MACDDiff: Optional[float] = None
    Williams_R: Optional[float] = None
    Relative_Volume: Optional[float] = None

    def _1_calculate_indicators_new_bar(self, historical_data):
        """
        Computes the core technical indicators for the new bar using historical data.

        Parameters:
            historical_data (pd.DataFrame): Past bars used for rolling calculations.
        """
        # ✅ Compute Moving Averages
        self.FastAvg = historical_data["Close"].rolling(window=9).mean().iloc[-1]
        self.SlowAvg = historical_data["Close"].rolling(window=18).mean().iloc[-1]
        self.FastEMA = historical_data["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
        self.MedEMA = historical_data["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
        self.SlowEMA = historical_data["Close"].ewm(span=50, adjust=False).mean().iloc[-1]

        # ✅ Compute ADX & DMI (Ensure no NaNs)
        adx_indicators = historical_data.ta.adx()

        self.ADX = adx_indicators["ADX_14"].iloc[-1] if "ADX_14" in adx_indicators.columns and not pd.isna(
            adx_indicators["ADX_14"].iloc[-1]) else 0
        self.DMI_plus = adx_indicators["DMP_14"].iloc[-1] if "DMP_14" in adx_indicators.columns and not pd.isna(
            adx_indicators["DMP_14"].iloc[-1]) else 0
        self.DMI_minus = adx_indicators["DMN_14"].iloc[-1] if "DMN_14" in adx_indicators.columns and not pd.isna(
            adx_indicators["DMN_14"].iloc[-1]) else 0

        # ✅ Debugging: Print the values immediately after assignment
        print("\n🔍 **Debugging ADX & DMI Values After Assignment:**")
        print(f"ADX: {self.ADX}")
        print(f"DMI_plus: {self.DMI_plus}")
        print(f"DMI_minus: {self.DMI_minus}")

        # ✅ Debugging: Print warning if missing values
        missing_dmi = [col for col in ["ADX_14", "DMP_14", "DMN_14"] if col not in adx_indicators.columns]
        if missing_dmi:
            print(f"⚠️ Warning: Missing ADX/DMI columns → {missing_dmi}")

        # ✅ Compute Aroon Indicators
        aroon_indicators = historical_data.ta.aroon()
        self.AroonUp = aroon_indicators["AROONU_14"].iloc[-1] if "AROONU_14" in aroon_indicators.columns else 0
        self.AroonDn = aroon_indicators["AROOND_14"].iloc[-1] if "AROOND_14" in aroon_indicators.columns else 0
        self.AroonOsc = aroon_indicators["AROONOSC_14"].iloc[-1] if "AROONOSC_14" in aroon_indicators.columns else 0

        # ✅ Validate populated fields
        populated_fields = ["FastAvg", "SlowAvg", "FastEMA", "MedEMA", "SlowEMA",
                            "ADX", "DMI_plus", "DMI_minus", "AroonUp", "AroonDn", "AroonOsc"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated Successfully:** {populated_fields}")

    def _2_add_vwap_new_bar(self, historical_data):
        """
        Computes VWAP for the new bar while maintaining cumulative values internally.

        Parameters:
            historical_data (pd.DataFrame): Past bars used for cumulative calculations.
        """

        # ✅ Compute Typical Price
        self.Typical_Price = (self.High + self.Low + self.Close) / 3
        print(f"\n📌 **New Bar - Typical Price:** {self.Typical_Price}")

        # ✅ Ensure historical data has necessary columns
        required_columns = ["High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            print(f"⚠️ Missing columns in historical data: {missing_columns}")
            return

        # ✅ Compute Cumulative Price x Volume and Cumulative Volume for historical data
        historical_data["Typical_Price"] = (historical_data["High"] + historical_data["Low"] + historical_data[
            "Close"]) / 3
        historical_data["Cum_PxV"] = (historical_data["Typical_Price"] * historical_data["Volume"]).cumsum()
        historical_data["Cum_Volume"] = historical_data["Volume"].cumsum()

        # ✅ Assign the last available cumulative values from historical data
        if not historical_data["Cum_PxV"].isna().all():
            last_cum_pxv = historical_data["Cum_PxV"].iloc[-1]
        else:
            print("⚠️ `Cum_PxV` is NaN in historical data!")
            last_cum_pxv = 0  # Default fallback

        if not historical_data["Cum_Volume"].isna().all():
            last_cum_volume = historical_data["Cum_Volume"].iloc[-1]
        else:
            print("⚠️ `Cum_Volume` is NaN in historical data!")
            last_cum_volume = 0  # Default fallback

        print(f"\n🔍 **Last Known Cumulative Values Before New Bar:**")
        print(f"   🔹 Last Cum_PxV: {last_cum_pxv}")
        print(f"   🔹 Last Cum_Volume: {last_cum_volume}")

        # ✅ Compute new cumulative values by **adding** the new bar's contribution
        self.Cum_PxV = last_cum_pxv + (self.Typical_Price * self.Volume)
        self.Cum_Volume = last_cum_volume + self.Volume

        print(f"\n🔍 **New Bar Cumulative Updates:**")
        print(f"   🔹 Updated Cum_PxV: {self.Cum_PxV}")
        print(f"   🔹 Updated Cum_Volume: {self.Cum_Volume}")

        # ✅ Compute VWAP using adjusted cumulative values
        if self.Cum_Volume != 0:
            self.VWAP = self.Cum_PxV / self.Cum_Volume
            print(f"\n📌 **New Bar - Adjusted VWAP:** {self.VWAP}")
        else:
            self.VWAP = self.Typical_Price  # Fallback
            print("⚠️ Warning: Cum_Volume is zero, falling back to Typical Price for VWAP.")

        # ✅ Validate populated fields
        populated_fields = ["Typical_Price", "Cum_PxV", "Cum_Volume", "VWAP"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated Successfully:** {populated_fields}")

    def _3_add_fibonacci_levels_new_bar(self, historical_data, price_high="High", price_low="Low", length=50):
        """
        Computes Fibonacci retracement levels using the highest and lowest prices over a rolling window.

        Parameters:
            historical_data (pd.DataFrame): DataFrame containing past bars for calculations.
            price_high (str, optional): Column name for high prices. Default: "High".
            price_low (str, optional): Column name for low prices. Default: "Low".
            length (int, optional): Lookback period for Fibonacci calculations. Default: 50.
        """
        # ✅ Ensure historical data is sufficient
        if len(historical_data) < length:
            print(f"⚠️ Not enough data for Fibonacci Levels (Need at least {length}, have {len(historical_data)})")
            return

        # ✅ Compute Fibonacci levels
        self.Fibo_High = historical_data[price_high].rolling(window=length).max().iloc[-1]
        self.Fibo_Low = historical_data[price_low].rolling(window=length).min().iloc[-1]

        # ✅ Assign exact column names for consistency with `FeatureGenerator`
        self.__dict__["UpperBand.2"] = self.Fibo_High
        self.__dict__["LowerBand.2"] = self.Fibo_Low

        # ✅ Validate populated fields
        populated_fields = ["Fibo_High", "Fibo_Low", "UpperBand.2", "LowerBand.2"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated:** {populated_fields}")

    def _4_add_cci_average_new_bar(self, historical_data, cci_length=14, cci_avg_length=9):
        """
        Computes the Commodity Channel Index (CCI) and its moving average for the latest bar using historical data.

        Parameters:
        historical_data (pd.DataFrame): The historical dataset to derive CCI values.
        cci_length (int, optional): The length used to compute the CCI (default is 14).
        cci_avg_length (int, optional): The length used for the CCI moving average (default is 9).

        Updates:
        - self.CCI
        - self.CCI_Avg
        """

        # Ensure sufficient historical data is available
        if len(historical_data) < cci_length:
            print(f"⚠️ Not enough data for CCI calculation (Required: {cci_length}, Available: {len(historical_data)})")
            return

        # Compute CCI for the latest row
        self.CCI = \
        ta.cci(historical_data["High"], historical_data["Low"], historical_data["Close"], length=cci_length).iloc[-1]

        # Compute CCI Moving Average
        if len(historical_data) >= cci_avg_length:
            self.CCI_Avg = historical_data["CCI"].rolling(window=cci_avg_length).mean().iloc[-1]

        # ✅ Identify and print successfully populated fields
        relevant_columns = ["CCI", "CCI_Avg"]
        populated_fields = [col for col in relevant_columns if not pd.isna(getattr(self, col))]
        print(f"\n✅ **Fields Populated:** {populated_fields}")

    def _5_add_ichimoku_cloud_new_bar(self, historical_data, tenkan_period=9, kijun_period=26, senkou_b_period=52):
        """
        Computes Ichimoku Cloud indicators and assigns only the latest values to the new bar.

        Parameters:
            historical_data (pd.DataFrame): DataFrame containing past bars for calculations.
            tenkan_period (int, optional): Period for Tenkan-sen (default: 9).
            kijun_period (int, optional): Period for Kijun-sen (default: 26).
            senkou_b_period (int, optional): Period for Senkou Span B (default: 52).
        """

        # ✅ Ensure historical data is sufficient
        if len(historical_data) < max(tenkan_period, kijun_period, senkou_b_period):
            print(
                f"⚠️ Not enough data for Ichimoku Cloud Calculation (Need at least {max(tenkan_period, kijun_period, senkou_b_period)}, have {len(historical_data)})")
            return

        # ✅ Compute Ichimoku Cloud components
        self.Tenkan = (
                              historical_data["High"].rolling(window=tenkan_period).max().iloc[-1] +
                              historical_data["Low"].rolling(window=tenkan_period).min().iloc[-1]
                      ) / 2

        self.Kijun = (
                             historical_data["High"].rolling(window=kijun_period).max().iloc[-1] +
                             historical_data["Low"].rolling(window=kijun_period).min().iloc[-1]
                     ) / 2

        self.Chikou = historical_data["Close"].iloc[-1]

        self.SenkouSpan_A = (
                (self.Tenkan + self.Kijun) / 2
        )  # Senkou A is the mid-point of Tenkan & Kijun

        self.SenkouSpan_B = (
                                    historical_data["High"].rolling(window=senkou_b_period).max().iloc[-1] +
                                    historical_data["Low"].rolling(window=senkou_b_period).min().iloc[-1]
                            ) / 2  # Senkou B is the mid-point of max/min over the longer period

        # ✅ Validate populated fields
        populated_fields = ["Tenkan", "Kijun", "Chikou", "SenkouSpan_A", "SenkouSpan_B"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated:** {populated_fields}")

    def _6_add_atr_price_features_new_bar(self, historical_data, atr_period=14):
        """
        Computes ATR using historical data and adds ATR-based price levels for the new bar.

        Parameters:
            historical_data (pd.DataFrame): Past bars used for rolling ATR calculations.
            atr_period (int, optional): ATR calculation period. Default is 14.
        """
        print("\n🔍 **Debugging `_6_add_atr_price_features_new_bar` Method**")

        # ✅ Ensure historical data is sufficient
        if len(historical_data) < atr_period:
            print(f"⚠️ Not enough data for ATR Calculation (Need at least {atr_period}, have {len(historical_data)})")
            return

        # ✅ Compute True Range (TR) for historical data
        historical_data["Prev_Close"] = historical_data["Close"].shift(1)
        historical_data["TR1"] = historical_data["High"] - historical_data["Low"]
        historical_data["TR2"] = abs(historical_data["High"] - historical_data["Prev_Close"])
        historical_data["TR3"] = abs(historical_data["Low"] - historical_data["Prev_Close"])
        historical_data["True_Range"] = historical_data[["TR1", "TR2", "TR3"]].max(axis=1)

        # ✅ Compute TR for the new bar
        self.Prev_Close = historical_data["Close"].iloc[-1]
        self.TR1 = self.High - self.Low
        self.TR2 = abs(self.High - self.Prev_Close)
        self.TR3 = abs(self.Low - self.Prev_Close)
        self.True_Range = max(self.TR1, self.TR2, self.TR3)

        # ✅ Compute ATR using a rolling window
        atr_series = historical_data["True_Range"].rolling(window=atr_period).mean()
        self.ATR = atr_series.iloc[-1]

        # ✅ Assign ATR-based price levels
        self.__dict__["ATR+ High"] = self.High + self.ATR
        self.__dict__["ATR+ Low"] = self.Low - self.ATR
        self.__dict__["ATR+ Close"] = self.Close + self.ATR
        self.__dict__["ATR+ Open"] = self.Open + self.ATR

        # ✅ Validate populated fields
        populated_fields = ["Prev_Close", "TR1", "TR2", "TR3", "True_Range", "ATR", "ATR+ High", "ATR+ Low",
                            "ATR+ Close", "ATR+ Open"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated:** {populated_fields}")

    def _7_add_multi_ema_indicators_new_bar(self, historical_features, ema_periods=None):
        """
        Computes multiple EMAs using historical data and updates the `NewBar` object.

        Parameters:
        historical_features (pd.DataFrame): DataFrame containing past bars for EMA calculation.
        ema_periods (list, optional): List of EMA periods to compute (Default: [5, 10, ..., 50]).
        """

        # ✅ Default EMA timeframes if not provided
        if ema_periods is None:
            ema_periods = list(range(5, 55, 5))  # 5, 10, 15, ..., 50

        # ✅ Ensure historical data is sufficient for EMA calculation
        if len(historical_features) < max(ema_periods):
            print(
                f"⚠️ Not enough data for EMA calculations (Need at least {max(ema_periods)}, have {len(historical_features)})")
            return

        # ✅ Compute EMAs and distances for each period
        populated_fields = []
        for period in ema_periods:
            ema_col = f"EMA_{period}"
            close_distance_col = f"Close_vs_EMA_{period}"
            open_distance_col = f"Open_vs_EMA_{period}"

            # Compute EMA using historical Close prices
            ema_series = historical_features["Close"].ewm(span=period, adjust=False).mean()
            setattr(self, ema_col, ema_series.iloc[-1])  # Assign latest EMA value to `NewBar`

            # Compute distances
            setattr(self, close_distance_col, self.Close - getattr(self, ema_col))
            setattr(self, open_distance_col, self.Open - getattr(self, ema_col))

            # Track populated fields
            populated_fields.extend([ema_col, close_distance_col, open_distance_col])

        # ✅ Print populated fields
        print(f"✅ **Fields Populated:** {populated_fields}")

    def _8_add_high_based_indicators_combined_new_bar(self, historical_features, ema_periods=None,
                                                      rolling_periods=None):
        """
        Computes High-based EMA indicators, rolling highs, and periodic highs for the new bar.

        Parameters:
            historical_features (pd.DataFrame): DataFrame containing past bars for calculations.
            ema_periods (list, optional): List of EMA periods for High price. Default: [5, 10, ..., 50].
            rolling_periods (list, optional): List of rolling periods for Highest Highs. Default: [3, 4, 5, 6, 8, 10, 12, 15, 20].
        """

        # ✅ Default EMA timeframes if not provided
        if ema_periods is None:
            ema_periods = list(range(5, 55, 5))  # 5, 10, ..., 50

        # ✅ Default rolling periods if not provided
        if rolling_periods is None:
            rolling_periods = [3, 4, 5, 6, 8, 10, 12, 15, 20]

        # ✅ Ensure historical data is sufficient
        if len(historical_features) < max(rolling_periods):
            print(
                f"⚠️ Not enough data for High-Based Indicators (Need at least {max(rolling_periods)}, have {len(historical_features)})")
            return

        # ✅ Compute EMAs Based on High Price
        populated_fields = []
        for period in ema_periods:
            ema_col = f"EMA_{period}_High"
            distance_col = f"High_vs_{ema_col}"

            # Compute EMA using historical High prices
            ema_series = historical_features["High"].ewm(span=period, adjust=False).mean()
            setattr(self, ema_col, ema_series.iloc[-1])  # Assign latest EMA value to `NewBar`

            # Compute the distance between High and its EMA
            setattr(self, distance_col, self.High - getattr(self, ema_col))

            populated_fields.extend([ema_col, distance_col])

        # ✅ Compute Rolling Highest Highs
        for period in rolling_periods:
            rolling_high_col = f"High_Max_{period}"
            rolling_high_series = historical_features["High"].rolling(window=period).max()
            setattr(self, rolling_high_col, rolling_high_series.iloc[-1])  # Assign latest value

            populated_fields.append(rolling_high_col)

        # ✅ Compute Periodic Highs (Hourly, 15-Min, Daily)
        historical_features["Datetime"] = pd.to_datetime(
            historical_features["Date"] + " " + historical_features["Time"])
        self_datetime = pd.to_datetime(self.Date + " " + self.Time)

        # ✅ Hourly High
        hourly_high = historical_features[historical_features["Datetime"].dt.hour == self_datetime.hour]["High"].max()
        self.High_Hourly = hourly_high
        populated_fields.append("High_Hourly")

        # ✅ Corrected 15-Minute High Calculation
        start_time = self_datetime - pd.Timedelta(minutes=15)
        matching_rows = historical_features[
            (historical_features["Datetime"] >= start_time) &
            (historical_features["Datetime"] < self_datetime)
            ]

        # ✅ Assign 15-Minute High (Max from Matching Rows)
        self.High_15Min = max(
            self.High,
            historical_features.iloc[-1]["High"],
            historical_features.iloc[-2]["High"]
        )
        populated_fields.append("High_15Min")

        # ✅ Daily High
        daily_high = historical_features[historical_features["Date"] == self.Date]["High"].max()
        self.High_Daily = daily_high
        populated_fields.append("High_Daily")

        # ✅ Validation: Check if NaNs were introduced
        nan_fields = [field for field in populated_fields if
                      getattr(self, field) is None or pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ **Warning: The following fields contain NaNs after computation: {nan_fields}**")
        else:
            print(f"✅ **Fields Populated Successfully:** {populated_fields}")

    def _9_add_constant_columns_new_bar(self, constants=None):
        """
        Adds predefined constant values to the new bar.

        Parameters:
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
        """

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

        # ✅ Assign constant values to the `NewBar` object
        for col, value in constants.items():
            setattr(self, col, value)

        # ✅ Print populated fields
        print(f"✅ **Constant Fields Populated:** {list(constants.keys())}")

    def _10_add_macd_indicators_new_bar(self, macd_fast=12, macd_slow=26, macd_signal=9, historical_data=None):
        """
        Computes MACD indicators (MACD, MACDAvg, MACDDiff) for the new bar using historical data.

        Parameters:
        macd_fast (int, optional): Fast EMA period for MACD (default: 12).
        macd_slow (int, optional): Slow EMA period for MACD (default: 26).
        macd_signal (int, optional): Signal line EMA period (default: 9).
        historical_data (pd.DataFrame, optional): DataFrame containing past 'Close' prices to compute MACD properly.
        """

        # ✅ Ensure historical data is provided
        if historical_data is None or "Close" not in historical_data.columns:
            print("⚠️ Cannot compute MACD - Historical data is missing or incomplete.")
            return

        # ✅ Combine historical Close prices with the new bar Close price
        close_prices = pd.concat([historical_data["Close"], pd.Series([self.Close])], ignore_index=True)

        # ✅ Compute MACD Line (Fast EMA - Slow EMA)
        macd = close_prices.ewm(span=macd_fast, adjust=False).mean() - close_prices.ewm(span=macd_slow,
                                                                                        adjust=False).mean()

        # ✅ Compute MACD Signal Line (9-period EMA of MACD)
        macd_signal_line = macd.ewm(span=macd_signal, adjust=False).mean()

        # ✅ Compute MACD Histogram (MACD - MACD Signal Line)
        macd_diff = macd - macd_signal_line

        # ✅ Assign the last computed values to the new bar
        self.MACD = macd.iloc[-1]
        self.MACDAvg = macd_signal_line.iloc[-1]
        self.MACDDiff = macd_diff.iloc[-1]

        # ✅ Print populated fields
        print(f"✅ **Fields Populated:** ['MACD', 'MACDAvg', 'MACDDiff']")

    def _11_add_volatility_momentum_volume_features_new_bar(self, historical_data, williams_period=14,
                                                            volume_period=20):
        """
        Computes Williams %R and Relative Volume features for the new bar.

        Parameters:
            historical_data (pd.DataFrame): DataFrame containing past bars for calculations.
            williams_period (int, optional): Lookback period for Williams %R. Default: 14.
            volume_period (int, optional): Lookback period for Relative Volume. Default: 20.
        """
        # ✅ Ensure historical data is sufficient for Williams %R calculation
        if len(historical_data) < williams_period:
            print(
                f"⚠️ Not enough data for Williams %R Calculation (Need at least {williams_period}, have {len(historical_data)})")
            self.__dict__["Williams_R"] = float("nan")
        else:
            # ✅ Compute Williams %R
            high_n = historical_data["High"].rolling(window=williams_period).max()
            low_n = historical_data["Low"].rolling(window=williams_period).min()

            if pd.isna(high_n.iloc[-1]) or pd.isna(low_n.iloc[-1]):
                print(f"⚠️ Warning: Williams %R calculation has NaNs! Debugging:")
                print(historical_data.tail(3))  # Print last few rows for reference
                self.__dict__["Williams_R"] = float("nan")
            else:
                self.__dict__["Williams_R"] = ((high_n.iloc[-1] - self.Close) / (
                            high_n.iloc[-1] - low_n.iloc[-1])) * -100

        # ✅ Ensure historical data is sufficient for Volume calculations
        if len(historical_data) < volume_period:
            print(
                f"⚠️ Not enough data for Relative Volume Calculation (Need at least {volume_period}, have {len(historical_data)})")
            self.__dict__["Relative_Volume"] = float("nan")
        else:
            # ✅ Compute Relative Volume (Ensure volume is not all zeros)
            if "Volume" in historical_data.columns and historical_data["Volume"].sum() > 0:
                sma_volume = historical_data["Volume"].rolling(window=volume_period).mean().iloc[-1]
                self.__dict__["Relative_Volume"] = self.Volume / sma_volume if sma_volume > 0 else 0
            else:
                print("⚠️ Warning: 'Volume' column not found or all zeroes. Relative Volume set to NaN.")
                self.__dict__["Relative_Volume"] = float("nan")

        # ✅ Validate populated fields
        populated_fields = ["Williams_R", "Relative_Volume"]

        # ✅ Check for NaNs
        nan_fields = [field for field in populated_fields if pd.isna(getattr(self, field))]

        if nan_fields:
            print(f"⚠️ Warning: NaNs detected in {nan_fields}")
        else:
            print(f"✅ **Fields Populated:** {populated_fields}")

    def validate_new_bar(new_bar):
        """
        Validates that all attributes in `new_bar` are populated before conversion to DataFrame.

        Parameters:
        - new_bar (NewBar): The newly created bar object.

        Raises:
        - ValueError: If any attribute is None (indicating incomplete feature calculation).
        """

        # ✅ Extract all attributes from the `new_bar` object
        new_bar_attributes = vars(new_bar)

        # ✅ Find any attributes that are still `None`
        missing_fields = {key: value for key, value in new_bar_attributes.items() if value is None}

        # ✅ Check if there are missing values
        if missing_fields:
            print("\n❌ CRITICAL ERROR: `new_bar` contains None values before conversion!")
            for key, value in missing_fields.items():
                print(f"🔹 {key}: {value}")

            # 🚨 Stop execution immediately (you can choose between sys.exit() or raising an error)
            raise ValueError(f"NewBar validation failed - missing fields: {list(missing_fields.keys())}")
            # sys.exit(1)  # Alternative if you want a hard stop (works better in standalone scripts)

        else:
            print("\n✅ All fields in `new_bar` are properly populated before conversion.")

        return True  # Validation passed
