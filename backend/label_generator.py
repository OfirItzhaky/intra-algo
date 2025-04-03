import numpy as np


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

        labels = [1, 0, 1, 0]

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
