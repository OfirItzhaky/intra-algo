import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Ensure using Tk backend
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode ON


class ModelLoaderAndExplorer:
    def __init__(self, regression_path, classifier_path):
        self.regression_path = regression_path
        self.classifier_path = classifier_path

    def load_and_explore(self):
        # === Load Regression Model ===
        with open(self.regression_path, "rb") as f:
            regression_model = pickle.load(f)
        print("\n📦 Regression Model Attributes:")
        for attr in dir(regression_model):
            if not attr.startswith("__"):
                print(f"🔹 {attr}: {type(getattr(regression_model, attr))}")

        # === Load Classifier Model ===
        with open(self.classifier_path, "rb") as f:
            classifier_model = pickle.load(f)
        print("\n📦 Classifier Model Attributes:")
        for attr in dir(classifier_model):
            if not attr.startswith("__"):
                print(f"🔸 {attr}: {type(getattr(classifier_model, attr))}")

        # === Extract Classifier Predictions ===
        df_classifier_preds = classifier_model.classifier_predictions_df.copy()

        return regression_model, classifier_model, df_classifier_preds

    import pandas as pd
    import matplotlib.pyplot as plt

    def plot_delta_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot the distribution of delta values (Predicted_High - Close) in categorized buckets.

        Args:
            df (pd.DataFrame): DataFrame containing 'Predicted_High' and 'Close' columns.
                               It will be modified to include 'Delta' and 'Delta_Bucket' columns.

        Returns:
            None. Displays a bar chart of the delta bucket distribution.
        """
        df["Delta"] = df["Predicted_High"] - df["Close"]
        bins = [-float("inf"), 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float("inf")]
        labels = ["<0", "0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–2.5", "2.5–3", "3–3.5", "3.5–4", "4+"]

        df["Delta_Bucket"] = pd.cut(df["Delta"], bins=bins, labels=labels, right=False)
        bucket_counts = df["Delta_Bucket"].value_counts().sort_index()

        plt.figure(figsize=(10, 4))
        bucket_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title("Delta Buckets (Predicted - Close)")
        plt.xlabel("Delta Range (points)")
        plt.ylabel("Number of Bars")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def evaluate_prediction_distance(
            self,
            df: pd.DataFrame,
            threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Evaluate how many predictions exceed a certain threshold distance from the close,
        and how many of those were actually correct based on future highs.

        Args:
            df (pd.DataFrame): DataFrame containing at least the columns:
                - 'Predicted': Model's predicted value (e.g. next high)
                - 'Actual': Actual target value
                - 'Close': Close price of the current bar
            threshold (float): Minimum required distance between predicted and close
                to consider a prediction high-confidence. Default is 1.0.

        Returns:
            pd.DataFrame: Filtered DataFrame with high-confidence predictions
                          and a new 'Is_Good' column indicating success.
        """
        df_copy = df.copy()

        # Step 1: Calculate difference between predicted and close
        df_copy["Predicted_Diff"] = df_copy["Predicted"] - df_copy["Close"]

        # Step 2: Filter predictions above threshold
        high_conf_preds = df_copy[df_copy["Predicted_Diff"] >= threshold].copy()

        # Step 3: Determine if actual high reached target
        high_conf_preds["Is_Good"] = high_conf_preds["Actual"] >= (high_conf_preds["Close"] + threshold)

        # Step 4: Summary stats
        total_candidates = len(high_conf_preds)
        num_good = high_conf_preds["Is_Good"].sum()
        percent_good = 100 * num_good / total_candidates if total_candidates > 0 else 0

        print(f"📈 Total predictions >= {threshold:.2f} points: {total_candidates}")
        print(f"✅ Number of 'good' bars (Actual >= Close + {threshold:.2f}): {num_good}")
        print(f"🎯 Percentage good: {percent_good:.2f}%")

        return high_conf_preds

    import matplotlib.pyplot as plt
    import pandas as pd

    def plot_high_confidence_by_hour(self, df: pd.DataFrame) -> None:
        """
        Plot the number of high-confidence predictions (Predicted - Close >= 1.0)
        grouped by 2-hour time bins based on the 'Time' column.

        Args:
            df (pd.DataFrame): A DataFrame with columns 'Predicted', 'Close', and 'Time'.
                               It will be filtered and plotted.

        Returns:
            None. Displays a bar chart of trade distribution by time.
        """
        # Step 1: Calculate prediction delta
        df["prediction_minus_close"] = df["Predicted"] - df["Close"]

        # Step 2: Filter 1+ point trades
        df_above_1 = df[df["prediction_minus_close"] >= 1.0].copy()

        # Step 3: Convert Time to hour
        df_above_1["Hour"] = pd.to_datetime(df_above_1["Time"], format="%H:%M").dt.hour

        # Step 4: Create 2-hour bins
        bins = list(range(0, 25, 2))
        labels = [f"{h:02d}-{h + 2:02d}" for h in bins[:-1]]
        df_above_1["HourRange"] = pd.cut(df_above_1["Hour"], bins=bins, labels=labels, right=False)

        # Step 5: Count and plot
        time_distribution = df_above_1["HourRange"].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        time_distribution.plot(kind='bar', color='skyblue')
        plt.title("📊 Number of 1+ Point Trades by 2-Hour Windows")
        plt.xlabel("Time Window")
        plt.ylabel("Number of Trades")
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



    def plot_trade_volume_and_avg(self, df: pd.DataFrame) -> None:
        """
        Plot the total and average number of 1+ point predictions (Predicted - Close >= 1.0)
        per 2-hour time window.

        Args:
            df (pd.DataFrame): A DataFrame that must include 'Predicted', 'Close', 'Time', and 'Date' columns.

        Returns:
            None. Displays a grouped bar chart comparing total vs. average trades per day.
        """
        # Step 1: Filter high-confidence predictions
        df["prediction_minus_close"] = df["Predicted"] - df["Close"]
        df_above_1 = df[df["prediction_minus_close"] >= 1.0].copy()

        # Step 2: Extract hour and assign 2-hour windows
        df_above_1["Hour"] = pd.to_datetime(df_above_1["Time"], format="%H:%M").dt.hour
        df_above_1["Time Window"] = pd.cut(
            df_above_1["Hour"],
            bins=list(range(0, 26, 2)),
            labels=[f"{str(i).zfill(2)}-{str(i + 2).zfill(2)}" for i in range(0, 24, 2)],
            right=False
        )

        # Step 3: Compute total count and average per day
        trade_counts = df_above_1["Time Window"].value_counts().sort_index()
        num_days = df_above_1["Date"].nunique()
        trade_avg_per_day = trade_counts / num_days

        # Step 4: Plot side-by-side bars
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.4
        x = np.arange(len(trade_counts))

        ax.bar(x - bar_width / 2, trade_counts, width=bar_width, alpha=0.6, label="Total Count", color="skyblue")
        ax.bar(x + bar_width / 2, trade_avg_per_day, width=bar_width, alpha=0.9, label="Avg per Day", color="orange")

        ax.set_xticks(x)
        ax.set_xticklabels(trade_counts.index, rotation=45)
        ax.set_ylabel("Number of Trades")
        ax.set_xlabel("Time Window")
        ax.set_title("📊 Number of 1+ Point Trades by 2-Hour Windows (Total vs Avg per Day)")
        ax.legend()
        plt.tight_layout()
        plt.show()
