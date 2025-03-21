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

