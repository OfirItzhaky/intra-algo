import numpy as np
import pandas as pd


class LabelGenerator:
    """
    LabelGenerator is responsible for generating trade labels based on predefined rules.

    This class is designed to apply custom labeling logic to trading data,
    such as classifying bars as 'good' or 'bad' based on market conditions.

    Methods:
    --------
    - add_good_bar_label(df): Placeholder for the `add_good_bar` method (to be moved here).
    """

    def __init__(self):
        """
        Initializes the LabelGenerator class.

        Currently, this class contains only the `add_good_bar_label` method,
        but more labeling functions can be added in the future.
        """
        pass  # Placeholder, more methods will be added later

    def add_good_bar_label(self, df):
        """
        Adds the `good_bar_prediction_outside_of_boundary` label based on Elastic Net predictions.

        Parameters:
            df (pd.DataFrame): Dataframe containing OHLC prices and Predicted_High.

        Returns:
            pd.DataFrame: Updated dataframe with the new label.
        """
        df = df.copy()  # ✅ Avoid modifying the original dataset

        # ✅ Ensure required columns exist
        required_columns = ["High", "Close", "Prev_Close", "Prev_Predicted_High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ✅ Apply labeling conditions
        conditions = [
            (df["High"] >= df["Prev_Predicted_High"]) & (df["High"] > df["Prev_Close"]),
            (df["High"] > df["Prev_Close"]) & (df["High"] < df["Prev_Predicted_High"]),
            (df["High"] <= df["Prev_Close"]) & (df["High"] <= df["Prev_Predicted_High"]),
            (df["High"] < df["Prev_Close"]) & (df["High"] > df["Prev_Predicted_High"])
        ]

        labels = [1, 0, 0, 0]

        # ✅ Ensure all rows get a valid label (change default from -1 to 0)
        df["good_bar_prediction_outside_of_boundary"] = np.select(conditions, labels, default=0)

        print("✅ Successfully assigned labels without -1!")
        return df

    import pandas as pd

    def elasticnet_label_next_high(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a label for Elastic Net regression by storing the next period's High price.

        @param df: DataFrame containing market data with a "High" column.
        @return: DataFrame with an additional "Next_High" column.
        """
        if "High" not in df.columns:
            raise ValueError("❌ Error: 'High' column not found in DataFrame. Cannot create label.")

        df["Next_High"] = df["High"].shift(-1)  # Shift next high price
        df = df.dropna(subset=["Next_High"])  # Drop last row (NaN due to shift)

        print("✅ Elastic Net label 'Next_High' added.")
        return df

    def long_good_bar_label_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labels all bars. Label = 1 if predicted high > prev close AND actual high ≥ predicted high.
        Others are labeled as 0.
        """
        df = df.copy()

        required_columns = ["Prev_Close", "Prev_Predicted_High", "High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["long_good_bar_label"] = np.where(
            (df["Prev_Predicted_High"] > df["Prev_Close"]) & (df["High"] >= df["Prev_Predicted_High"]),
            1,
            0
        )

        print("✅ All-bar long label applied (1 = bullish + actual ≥ predicted).")
        return df

    def long_good_bar_label_bullish_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labels only bars where previous predicted high > previous close and actual high ≥ predicted high.
        Only applies to bullish setups.
        """
        df = df.copy()

        required_columns = ["Prev_Close", "Prev_Predicted_High", "High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df[df["Prev_Predicted_High"] > df["Prev_Close"]]  # Filter bullish only

        df["long_good_bar_label"] = np.where(
            df["High"] >= df["Prev_Predicted_High"],
            1,
            0
        )

        print("✅ Bullish-only label applied (1 = actual ≥ predicted).")
        return df

    def long_good_bar_label_bullish_only_goal_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Goal B: Labels bar T as 1 if current bar's High ≥ current Predicted_High AND Predicted_High > Close.
        Focused on bullish cases.
        """
        df = df.copy()

        required_columns = ["Close", "Predicted_High", "High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Only consider bullish bars where prediction > close
        df = df[df["Predicted_High"] > df["Close"]]

        df["long_good_bar_label"] = (df["High"] >= df["Predicted_High"]).astype(int)

        print("✅ Goal B Bullish Label applied (1 = this bar's High ≥ this bar's Predicted_High).")
        return df

    def long_good_bar_label_all_goal_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Goal B: Labels bar T as 1 if this bar's High ≥ Predicted_High.
        No bullish condition applied — all bars are labeled.
        """
        df = df.copy()

        required_columns = ["Predicted_High", "High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["long_good_bar_label"] = (df["High"] >= df["Predicted_High"]).astype(int)

        print("✅ Goal B ALL Label applied (1 = High ≥ Predicted_High).")
        return df

    def long_good_bar_label_all_goal_c(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Goal C: Labels bar T as 1 if next bar's High ≥ this bar's Predicted_High.
        Applies to ALL bars.
        """
        df = df.copy()

        # ✅ Drop unnecessary columns if present (leftover from other flows)
        for col in ["Prev_Close", "Prev_Predicted_High"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # ✅ Ensure required columns exist
        required_columns = ["Predicted_High", "High"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ✅ Shift High by -1 to access bar T+1's High
        df["Next_High"] = df["High"].shift(-1)

        # ✅ Assign label = 1 if Next_High ≥ Predicted_High
        df["long_good_bar_label"] = (df["Next_High"] >= df["Predicted_High"]).astype(int)

        print("✅ Goal C ALL Label applied (1 = Next High ≥ Predicted High).")
        return df

    def long_good_bar_label_bullish_only_goal_c(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Goal C: Labels bar T as 1 if:
        - bar's Predicted_High > Close (bullish setup)
        - AND next bar's High ≥ this bar's Predicted_High

        Applies to bullish setups only.
        """
        df = df.copy()

        required_columns = ["Predicted_High", "High", "Close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["Next_High"] = df["High"].shift(-1)

        # Filter bullish only setups
        df = df[df["Predicted_High"] > df["Close"]]

        # Assign label
        df["long_good_bar_label"] = (df["Next_High"] >= df["Predicted_High"]).astype(int)

        print("✅ Goal C Bullish Label applied (1 = bullish setup + Next High ≥ Predicted High).")
        return df

    def green_red_bar_label_goal_d(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Goal D: Simple direction classifier.
        Label = 1 if bar is green (Close > Open), else 0.

        Parameters:
            df (pd.DataFrame): DataFrame with "Open" and "Close" columns.

        Returns:
            pd.DataFrame: DataFrame with new "long_good_bar_label" column.
        """
        df = df.copy()

        required_columns = ["Open", "Close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["Next_Close"] = df["Close"].shift(-1)
        df["long_good_bar_label"] = (df["Next_Close"] > df["Close"]).astype(int)

        df = df.drop(columns=["Next_Close"], errors="ignore")  # ⛔ Prevent leakage

        print("✅ Goal D Label applied (1 = green bar) TEST.")
        return df

    def green_bar_with_tick_threshold_label(self, df: pd.DataFrame, tick_size: float = 0.5) -> pd.DataFrame:
        """
        Label = 1 if (Next Close - Current Close) > tick_size threshold, else 0.
        Helps reduce noise from small price fluctuations.

        Parameters:
            df (pd.DataFrame): DataFrame with "Close" column.
            tick_size (float): Minimum price movement (in points/ticks) to consider as a valid upward bar.

        Returns:
            pd.DataFrame: Updated DataFrame with "tick_threshold_label" column.
        """
        df = df.copy()

        if "Close" not in df.columns:
            raise ValueError("Missing required column: 'Close'")

        df["Next_Close"] = df["Close"].shift(-1)
        df["tick_threshold_label"] = ((df["Next_Close"] - df["Close"]) > tick_size).astype(int)
        df = df.drop(columns=["Next_Close"], errors="ignore")  # ⛔ Prevent leakage

        print(f"✅ Tick-threshold label applied (tick size = {tick_size})")
        return df

    def option_d_multiclass_next_bar_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Option D: Multi-Class labeling based on next bar's movement (Close - Open).

        Creates a 5-class label based on how strong the next bar's move is.
        Used for multi-class classification tasks.

        Label classes:
            0 = Big Drop     (<= -2.5 pts)
            1 = Medium Down  (-2.5 to -1.0 pts)
            2 = Flat / Noise (-1.0 to +1.0 pts)
            3 = Medium Up    (+1.0 to +2.5 pts)
            4 = Big Jump     (>= +2.5 pts)
        """
        df = df.copy()

        required_columns = ["Open", "Close"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute next bar's movement
        df["Next_Open"] = df["Open"].shift(-1)
        df["Next_Close"] = df["Close"].shift(-1)
        df["next_bar_movement"] = df["Next_Close"] - df["Next_Open"]

        # Define bin edges and labels
        bins = [-float("inf"), -2.5, -1.0, 1.0, 2.5, float("inf")]
        labels = [0, 1, 2, 3, 4]

        df["multi_class_label"] = pd.cut(df["next_bar_movement"], bins=bins, labels=labels).astype("Int64")

        # 🚫 These must not leak into training
        df.drop(columns=["Next_Open", "Next_Close", "next_bar_movement"], inplace=True)

        print("✅ Option D Multi-Class Labels applied (based on next bar movement).")

        return df
