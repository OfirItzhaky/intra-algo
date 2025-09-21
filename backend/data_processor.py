import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from research_agent.logging_setup import get_logger

log = get_logger(__name__)

class DataProcessor:
    """
    A class to perform Exploratory Data Analysis (EDA) and data preprocessing tasks.
    """

    def __init__(self):
        """
        Initializes the DataProcessor class.
        """
        pass

    def remove_redundant_features(self, df: pd.DataFrame, redundant_columns: list) -> pd.DataFrame:
        """
        Removes redundant features from the DataFrame based on feature similarity and constant columns.

        @param df: The input DataFrame with all indicators.
        @param redundant_columns: List of columns to be removed due to redundancy.
        @return: The DataFrame with redundant features removed.
        """
        # Remove columns that are in the list and exist in the DataFrame
        df_cleaned = df.drop(columns=[col for col in redundant_columns if col in df], errors='ignore')

        log.info(f"Removed {len(redundant_columns)} redundant columns.")
        return df_cleaned

    # Other methods like validate_data, etc.


    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        @param data: The DataFrame to clean by removing rows with missing values.
        @return: The cleaned DataFrame without missing values.
        """
        data_cleaned = data.dropna()
        log.info(f"Removed {data.shape[0] - data_cleaned.shape[0]} rows with missing values.")
        return data_cleaned

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        @param data: The DataFrame to scale.
        @return: The scaled DataFrame (standardized data).
        """
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        log.info(f"Scaled {len(numeric_columns)} numerical columns.")
        return data

    def remove_correlated_features(self, data: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """
        @param data: The DataFrame containing the features.
        @param threshold: The correlation threshold above which features are removed.
        @return: A DataFrame with less correlated features.
        """
        corr_matrix = data.corr()
        upper_triangle = corr_matrix.where(
            pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        data_cleaned = data.drop(columns=to_drop)
        log.info(f"Removed {len(to_drop)} highly correlated features.")
        return data_cleaned


    def validate_data(self, data: pd.DataFrame) -> None:
        """
        @param data: The DataFrame to validate and analyze.
        @return: None
        """
        # Calculate the necessary information
        start_date = data.index.min()
        end_date = data.index.max()
        bar_count = len(data)
        null_count = data.isnull().any(axis=1).sum()
        zero_rows = (data == 0).all(axis=1).sum()

        # Display the results
        log.info(f"Start Date: {start_date}")
        log.info(f"End Date: {end_date}")
        log.info(f"Total Bars: {bar_count}")
        log.info(f"Rows with any Null values: {null_count}")
        log.info(f"Rows where all values are Zero: {zero_rows}")

        # Explanation for why there are more bars than requested
        log.info("The number of bars retrieved may be more than the specified count due to the data "
              "aggregation behavior of the Yahoo Finance API, especially around the beginning or end of trading sessions.")

    # Add another method to the DataProcessor class
    def movement_eda(self, data: pd.DataFrame, ticks_per_point: int = 4, plot: bool = True) -> float:
        """
        Perform exploratory data analysis (EDA) on bar movements.
        """
        data['MovementSize'] = (data['High'] - data['Low']) * ticks_per_point
        avg_movement = data['MovementSize'].mean()
        log.info(f"Average Movement Size: {avg_movement:.2f} ticks")

        if plot:
            import matplotlib.pyplot as plt
            plt.hist(data['MovementSize'], bins=50, color='skyblue', edgecolor='black')
            plt.title('Distribution of Movement Sizes')
            plt.xlabel('Movement Size (in ticks)')
            plt.ylabel('Frequency')
            plt.show()

        return avg_movement

    def print_label_distribution(self, data: pd.DataFrame, label_column: str = 'strategy_1_label') -> None:
        """
        Print and visualize the distribution of the labels in the DataFrame.

        @param data: The DataFrame containing the label column.
        @param label_column: The name of the label column to analyze (default is 'strategy_1_label').
        @return: None
        """
        # Calculate the value counts
        label_distribution = data[label_column].value_counts()

        # Display the distribution as a bar plot
        label_distribution.plot(kind='bar', color=['skyblue', 'orange'], edgecolor='black')
        plt.title(f"Distribution of {label_column}")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Also, print the distribution statistics
        log.info(f"Label Distribution:\n{label_distribution}\n")
        log.info(f"\nPercentage Distribution:\n{(label_distribution / len(data) * 100).round(2)}")



    def show_label_distribution_by_day_time(self, original_data, label_column='strategy_1_label', date_column='Date', time_column='Time'):
        """
        Show the distribution of label 1 across days of the week and time of day.

        Parameters:
            data (pd.DataFrame): DataFrame with the data including Date, Time, and label column.
            label_column (str): The label column name (default 'strategy_1_label').
            date_column (str): The column containing the Date (default 'Date').
            time_column (str): The column containing the Time (default 'Time').

        Returns:
            None
        """
        data = original_data.copy()
        # Step 1: Remove Saturday data from the DataFrame
        if "Saturday" in data.columns:
            data = data.drop(columns='Saturday')

        # Group by HourWindow and DayOfWeek and sum the labels
        label_sum_per_window = data.groupby(['HourWindow', 'DayOfWeek'])['strategy_1_label'].sum()

        # Transpose to swap rows and columns
        label_sum_per_window = label_sum_per_window.unstack(level='DayOfWeek')


        unique_dates = data['Date'].drop_duplicates()

        # Step 2: Count frequency of each day of the week based on unique dates
        unique_day_of_week_count = unique_dates.apply(lambda x: pd.to_datetime(x).day_name()).value_counts()

        # Display the results
        log.info(unique_day_of_week_count)

        # Step 1: Loop over the columns (days of the week) in the corrected_label_avg_per_window
        for day, count in unique_day_of_week_count.items():
            # Step 2: Apply the division dynamically based on the occurrences
            if day in data.columns:
                data[day] = data[day] / count




        # Check if the time contains seconds, and adjust format accordingly
        try:
            data[time_column] = pd.to_datetime(data[time_column], format='mixed').dt.time  # If no seconds
        except:
            data[time_column] = pd.to_datetime(data[time_column], format='mixed').dt.time  # If there are seconds

        # Extract Day of Week and Hour for visualization
        data['DayOfWeek'] = data[date_column].dt.day_name()  # Get Day of the Week (e.g., Monday, Tuesday)
        data['Hour'] = data[time_column].apply(lambda x: x.hour)  # Extract hour from time

        # Group by HourWindow and DayOfWeek and sum the labels
        label_sum_per_window = data.groupby(['Hour', 'DayOfWeek'])[label_column].sum().unstack(level='DayOfWeek')

        # Step 1: Calculate the mean for each HourWindow across the days
        aggregated_data = label_sum_per_window.mean(axis=1)

        # Step 2: Reorder the times to start from 00:00-01:00 to 23:00-00:00
        time_order = [
            '0:00-1:00', '1:00-2:00', '2:00-3:00', '3:00-4:00', '4:00-5:00', '5:00-6:00', '6:00-7:00', '7:00-8:00',
            '8:00-9:00', '9:00-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00', '13:00-14:00', '14:00-15:00',
            '15:00-16:00', '16:00-17:00', '17:00-18:00', '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00',
            '22:00-23:00', '23:00-00:00'
        ]

        # Sort the aggregated data by the predefined time order
        aggregated_data = aggregated_data[time_order]

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        aggregated_data.plot(kind='bar', color=plt.cm.viridis(aggregated_data / max(aggregated_data)), width=0.8)

        # Add labels and title
        plt.title("Label 1 Occurrences by Hour (Bar Chart)")
        plt.xlabel("Hour Window")
        plt.ylabel("Label 1 Occurrences")

        plt.xticks(rotation=90)  # Rotate the x labels for readability
        plt.tight_layout()

        # Show the plot
        plt.show()

        # For Calendar-like Heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(label_sum_per_window, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5)
        plt.title("Label 1 Distribution Across Days of the Week and Time Windows")
        plt.ylabel("Hour Window")
        plt.xlabel("Day of the Week")
        plt.show()

    def plot_feature_correlation(self, data):
        """Generates and displays a correlation matrix heatmap for the given DataFrame, excluding non-numeric columns."""
        try:
            # Ensure data is numeric by excluding non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            # Check if there are any non-numeric columns that were excluded
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_columns) > 0:
                log.info(f"Excluded non-numeric columns: {non_numeric_columns}")

            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()

            # Plot the heatmap
            plt.figure(figsize=(20, 15))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.show()
        except Exception as e:
            log.info(f"An error occurred: {e}")

    def remove_highly_correlated(self, data, threshold=1.0):
        """Remove features that have a correlation coefficient of the specified threshold or higher."""
        numeric_data = data.select_dtypes(include=[np.number])

        # Check if there are any non-numeric columns that were excluded
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            log.info(f"Excluded non-numeric columns: {non_numeric_columns}")
        corr_matrix = numeric_data.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
        reduced_data = data.drop(to_drop, axis=1)
        log.info(f"Removed columns: {to_drop}")
        return reduced_data

    def prepare_dataset_for_regression_sequential(self, data: pd.DataFrame, target_column: str = 'Next_High',
                                                  split_ratio: float = 0.8, drop_target: bool = True):
        """
        Prepares dataset for regression, ensuring correct alignment of target values.

        @param data: The input dataset (must include target_column).
        @param target_column: The target column for regression (default: 'Next_High').
        @param split_ratio: Proportion of the dataset to include in the training set.
        @param drop_target: Whether to exclude the target column from the feature set.

        @return: X_train, y_train, X_test, y_test.
        """

        # ✅ Step 2: Perform sequential split
        split_index = int(len(data) * split_ratio)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        # ✅ Step 3: Fill NaNs in all other feature columns (AFTER the split)
        train_data = train_data.ffill().bfill()
        test_data = test_data.ffill().bfill()

        # ✅ Step 4: Extract numeric features and target
        def extract_features_and_target(df, target_column, drop_target):
            X = df.select_dtypes(include=['number']).copy()
            if drop_target:
                X = X.drop(target_column, axis=1)
            y = df[target_column].copy()
            return X, y

        X_train, y_train = extract_features_and_target(train_data, target_column, drop_target)
        X_test, y_test = extract_features_and_target(test_data, target_column, drop_target)

        # ✅ Step 5: Debugging sample check
        log.info("\n✅ Sample Data Alignment Check:")
        comparison_df = pd.DataFrame({
            "Sample Target (y_test)": y_test.head(5).tolist(),
            "Corresponding Feature Row": X_test.head(5).index.tolist()
        })
        log.info(comparison_df)

        return X_train, y_train, X_test, y_test

    def visualize_regression_predictions_for_pycharm(self, data, y_test, predictions, n=20, tick_size=0.25,
                                                     title="Regression Predictions (PyCharm)"):
        # ✅ Ensure correct alignment by using `data` rows that match `y_test.index`
        aligned_data = data.loc[y_test.index].copy()
        aligned_data["Actual Value"] = y_test.values
        aligned_data["Predicted Value"] = predictions

        # ✅ Slice only the last `n` rows for visualization
        df_to_visualize = aligned_data.tail(n)

        if df_to_visualize.empty:
            log.info("Warning: The DataFrame for visualization is empty. Check your data alignment or filtering criteria.")
            return None

        # ✅ Convert timestamp for plotting
        df_to_visualize['Timestamp'] = pd.to_datetime(df_to_visualize['Date'] + ' ' + df_to_visualize['Time'])
        df_to_visualize.set_index('Timestamp', inplace=True)

        # ✅ Calculate dynamic Y-axis limits based on min/max prices
        min_price = df_to_visualize[['Low', 'Actual Value', 'Predicted Value']].min().min()
        max_price = df_to_visualize[['High', 'Actual Value', 'Predicted Value']].max().max()

        # ✅ Dynamic buffer based on data range (~2% of the range)
        buffer = (max_price - min_price) * 0.02  # 2% padding

        y_min = min_price - buffer
        y_max = max_price + buffer

        # ✅ Creating a figure with a dark background
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="black")
        ax.set_facecolor("black")  # ✅ Set background to dark

        # ✅ Set Y-axis limits dynamically
        ax.set_ylim(y_min, y_max)

        # ✅ Manually drawing the candlesticks with white wicks
        for idx, row in df_to_visualize.iterrows():
            color = 'red' if row['Open'] > row['Close'] else 'green'
            ax.plot([idx, idx], [row['Low'], row['High']], color='white')  # ✅ White wick
            ax.plot([idx, idx], [row['Open'], row['Close']], color=color, linewidth=10)  # Body

        # ✅ Overlaying the line plots for actual and predicted values
        ax.plot(df_to_visualize.index, df_to_visualize['Actual Value'], label='Actual High', marker='o', linestyle='-',
                color='blue')
        ax.plot(df_to_visualize.index, df_to_visualize['Predicted Value'], label='Predicted High', marker='o',
                linestyle='--', color='red')

        # ✅ Annotate points for clarity
        for x, actual, predicted in zip(df_to_visualize.index, df_to_visualize['Actual Value'],
                                        df_to_visualize['Predicted Value']):
            ax.text(x, actual, f'{actual:.2f}', color='blue', ha='center', va='bottom')
            ax.text(x, predicted, f'{predicted:.2f}', color='red', ha='center', va='top')

        # ✅ Add annotation box to clarify next-bar target concept
        note_text = "**EACH ACTUAL REPRESENTS THE NEXT BAR'S TARGET (Next_High)**"
        ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', bbox=dict(facecolor='white', alpha=0.75, edgecolor='black'))

        # ✅ Customizing the x-axis to handle date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
        plt.title(title, color="white")  # ✅ Title in white
        plt.xlabel("Timestamp", color="white")  # ✅ X-axis label in white
        plt.ylabel("Price", color="white")  # ✅ Y-axis label changed to "Price"
        ax.tick_params(axis='x', colors="white")
        ax.tick_params(axis='y', colors="white")
        ax.legend(facecolor="black", edgecolor="white")  # ✅ Legend with dark background
        # ✅ Change legend text color to match line colors
        legend = ax.legend(facecolor="black", edgecolor="white")
        for text, line in zip(legend.get_texts(), legend.get_lines()):
            text.set_color(line.get_color())

        plt.grid(True, color="gray", linestyle="--")
        plt.tight_layout()
        plt.show()

        return fig

    def visualize_classifiers_pycharm(self, trainer, classifier_trainer, n=20):
        """
        Plots classifier predictions alongside regression plot in a new figure.

        Parameters:
            classifier_trainer: ClassifierModelTrainer instance (stores classifier results).
            trainer: RegressionModelTrainer instance (stores regression results).
            n: Number of last bars to visualize.

        Returns:
            fig: Matplotlib figure containing the combined visualization.
        """

        log.info("📊 Debugging visualize_classifiers_pycharm...")

        # ✅ Extract stored regression figure
        if trainer.regression_figure is None:
            log.info("⚠️ No stored regression figure! Showing classifier plot only.")
            return None

        # ✅ Get regression data for the last `n` bars
        df_to_visualize = trainer.x_test_with_meta.loc[trainer.y_test.index].copy()
        df_to_visualize["Actual Value"] = trainer.y_test.values
        df_to_visualize["Predicted Value"] = trainer.predictions
        df_to_visualize = df_to_visualize.tail(n)  # ✅ Only last `n` bars

        if df_to_visualize.empty:
            log.info("❌ Regression data for visualization is empty!")
            return None

        # ✅ Convert 'Date' and 'Time' columns into a proper DatetimeIndex if available
        if 'Date' in df_to_visualize.columns and 'Time' in df_to_visualize.columns:
            df_to_visualize['Timestamp'] = pd.to_datetime(df_to_visualize['Date'] + ' ' + df_to_visualize['Time'])
            df_to_visualize.set_index('Timestamp', inplace=True)

        # ✅ Ensure classifier_df matches the same datetime-based index
        classifier_df = classifier_trainer.classifier_predictions_df.copy()
        classifier_df = classifier_df.reindex(df_to_visualize.index).tail(n)  # Align and limit

        if classifier_df.empty:
            log.info("❌ No classifier predictions found!")
            return None

        log.info("✅ Checking df_to_visualize shape:", df_to_visualize.shape)
        log.info("✅ Checking classifier_df shape before reindexing:", classifier_df.shape)

        # ✅ Round classifier values to ensure binary classification (0 or 1)
        classifier_df = classifier_df.round()

        # ✅ Create a figure with a dark background
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="black")
        ax.set_facecolor("black")

        # ✅ Plot Candlesticks (Ensure existing elements remain intact)
        for idx, row in df_to_visualize.iterrows():
            color = 'red' if row['Open'] > row['Close'] else 'green'
            ax.plot([idx, idx], [row['Low'], row['High']], color='white')  # Wick
            ax.plot([idx, idx], [row['Open'], row['Close']], color=color, linewidth=10)  # Body

        # ✅ Overlay Actual and Predicted Highs
        ax.plot(df_to_visualize.index, df_to_visualize['Actual Value'], label='Actual High', marker='o', linestyle='-',
                color='blue')
        ax.plot(df_to_visualize.index, df_to_visualize['Predicted Value'], label='Predicted High', marker='o',
                linestyle='--', color='red')

        # ✅ Restore axis labels, grid, and ticks
        ax.set_xlabel("Timestamp", color="white")
        ax.set_ylabel("Price", color="white")
        ax.tick_params(axis="both", colors="white")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # ✅ Create a secondary Y-axis for classifier results
        ax2 = ax.twinx()
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["No Signal", "Good Signal"])
        ax2.set_ylabel("Classifier Prediction", color="white")

        # ✅ Define marker shapes and positioning offsets
        classifier_styles = {
            "RandomForest": ("D", 2.0),  # Diamond
            "LightGBM": ("*", 4),  # Star
            "XGBoost": ("P", 6)  # Thick Plus
        }

        # ✅ Plot classifier results below the price bars with dynamic coloring
        for model, (marker, offset) in classifier_styles.items():
            if model in classifier_df.columns:
                colors = classifier_df[model].map(lambda x: "green" if x == 1 else "red")  # Dynamic color mapping
                ax.scatter(
                    classifier_df.index, df_to_visualize["Low"] - offset,  # Position markers below the low price
                    label=f"{model} Prediction", c=colors, marker=marker, s=100  # Increase `s` from 50 to 100
                )

        # ✅ Restore title and legend
        plt.title("Regression + Classifier Predictions (PyCharm)", color="white")

        # ✅ Fix legend to have all text in white
        legend1 = ax.legend(loc="upper left", facecolor="black", edgecolor="white", fontsize=10)

        # ✅ Loop through legend items and set all text to white
        for text in legend1.get_texts():
            text.set_color("white")  # ✅ Make all legend text white

        plt.xticks(rotation=45, ha="right")  # ✅ Ensure readable x-axis timestamps
        plt.tight_layout()
        plt.show()

        return fig

    def validate_simulation(self, training_df, simulation_df):
        """
        Validates simulation data against training data.
        - Ensures no missing expected bars.
        - Removes overlapping bars.
        - Ensures simulation extends past training data.
        - Adjusts simulation start time if needed.
        """
        if training_df is None or simulation_df is None:
            return {"status": "error", "message": "Training or simulation data is not loaded."}

        # ✅ Get last training timestamp
        last_training_bar = training_df.iloc[-1]
        last_training_timestamp = pd.to_datetime(last_training_bar["Date"] + " " + last_training_bar["Time"])
        expected_next_bar = last_training_timestamp + pd.Timedelta(minutes=5)

        # ✅ Get first & last simulation timestamps
        simulation_timestamps = pd.to_datetime(simulation_df["Date"] + " " + simulation_df["Time"])
        first_simulation_bar = simulation_timestamps.iloc[0]
        last_simulation_bar = simulation_timestamps.iloc[-1]

        # ✅ Missing Data Warning (if expected time is not at the start)
        missing_data_alert = None
        if first_simulation_bar != expected_next_bar:
            if expected_next_bar in simulation_timestamps.values:
                # ✅ Slice simulation data to start from expected timestamp
                simulation_df = simulation_df.loc[simulation_timestamps >= expected_next_bar]
                first_simulation_bar = expected_next_bar  # ✅ Update first bar reference
            else:
                # If expected timestamp doesn't exist at all, show warning
                missing_data_alert = f"⚠️ Missing Data! Expected {expected_next_bar}, but found {first_simulation_bar}"

        # ✅ Insufficient Simulation Data Warning
        insufficient_simulation_alert = None
        if last_simulation_bar <= last_training_timestamp:
            insufficient_simulation_alert = (
                f"❌ Simulation data is too short! Training ends at {last_training_timestamp}, "
                f"but simulation ends at {last_simulation_bar}."
            )

        # ✅ Remove Overlapping Data
        training_timestamps = set(pd.to_datetime(training_df["Date"] + " " + training_df["Time"]))
        cleaned_simulation_df = simulation_df[
            ~pd.to_datetime(simulation_df["Date"] + " " + simulation_df["Time"]).isin(training_timestamps)
        ]
        # ✅ Ensure we keep one extra bar at the start
        if not cleaned_simulation_df.empty:
            cleaned_simulation_df = pd.concat([simulation_df.iloc[:1], cleaned_simulation_df], ignore_index=True)

        if not cleaned_simulation_df.empty:
            first_sliced_timestamp = pd.to_datetime(
                cleaned_simulation_df.iloc[0]["Date"] + " " + cleaned_simulation_df.iloc[0]["Time"])

            first_sliced_timestamp - pd.Timedelta(minutes=5)
            # ✅ Find the previous bar in the original simulation data
            prev_bar = last_training_bar.copy()

            # ✅ Add the found previous bar back if it exists
            log.info("\n📊 BEFORE Adding prev_bar_df:")
            log.info(cleaned_simulation_df.head(3))  # Print first 3 rows for visibility

            if not prev_bar.empty:
                prev_bar_df = prev_bar.to_frame().T.reset_index(drop=True)
                cleaned_simulation_df = pd.concat([prev_bar_df, cleaned_simulation_df], ignore_index=True)

            log.info("\n📊 AFTER Adding prev_bar_df:")
            log.info(cleaned_simulation_df.head(3))  # Check if duplicates appear

        original_size = len(simulation_df)
        sliced_size = len(cleaned_simulation_df)

        log.info(f"🔍 Debug: Original Simulation Size -> {original_size}")
        log.info(f"🔍 Debug: Sliced Simulation Size -> {sliced_size}")

        # ✅ Ensure overlap fix is correctly detected
        overlap_fixed = sliced_size < original_size
        log.info(f"✅ Debug: Overlap Fixed Status -> {overlap_fixed}")
        return {
            "missing_data_warning": missing_data_alert,
            "insufficient_simulation_warning": insufficient_simulation_alert,
            "overlap_fixed": overlap_fixed,  # ✅ Updated flag
            "fixed_simulation_df": cleaned_simulation_df.drop_duplicates(subset=["Date", "Time"])

        }












