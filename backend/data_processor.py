import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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

        print(f"Removed {len(redundant_columns)} redundant columns.")
        return df_cleaned

    # Other methods like validate_data, etc.


    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        @param data: The DataFrame to clean by removing rows with missing values.
        @return: The cleaned DataFrame without missing values.
        """
        data_cleaned = data.dropna()
        print(f"Removed {data.shape[0] - data_cleaned.shape[0]} rows with missing values.")
        return data_cleaned

    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        @param data: The DataFrame to scale.
        @return: The scaled DataFrame (standardized data).
        """
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        print(f"Scaled {len(numeric_columns)} numerical columns.")
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
        print(f"Removed {len(to_drop)} highly correlated features.")
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
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Total Bars: {bar_count}")
        print(f"Rows with any Null values: {null_count}")
        print(f"Rows where all values are Zero: {zero_rows}")

        # Explanation for why there are more bars than requested
        print("The number of bars retrieved may be more than the specified count due to the data "
              "aggregation behavior of the Yahoo Finance API, especially around the beginning or end of trading sessions.")

    # Add another method to the DataProcessor class
    def movement_eda(self, data: pd.DataFrame, ticks_per_point: int = 4, plot: bool = True) -> float:
        """
        Perform exploratory data analysis (EDA) on bar movements.
        """
        data['MovementSize'] = (data['High'] - data['Low']) * ticks_per_point
        avg_movement = data['MovementSize'].mean()
        print(f"Average Movement Size: {avg_movement:.2f} ticks")

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
        print(f"Label Distribution:\n{label_distribution}\n")
        print(f"\nPercentage Distribution:\n{(label_distribution / len(data) * 100).round(2)}")



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
        print(unique_day_of_week_count)

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
                print(f"Excluded non-numeric columns: {non_numeric_columns}")

            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()

            # Plot the heatmap
            plt.figure(figsize=(20, 15))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")

    def remove_highly_correlated(self, data, threshold=1.0):
        """Remove features that have a correlation coefficient of the specified threshold or higher."""
        numeric_data = data.select_dtypes(include=[np.number])

        # Check if there are any non-numeric columns that were excluded
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            print(f"Excluded non-numeric columns: {non_numeric_columns}")
        corr_matrix = numeric_data.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
        reduced_data = data.drop(to_drop, axis=1)
        print(f"Removed columns: {to_drop}")
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
        print("\n✅ Sample Data Alignment Check:")
        comparison_df = pd.DataFrame({
            "Sample Target (y_test)": y_test.head(5).tolist(),
            "Corresponding Feature Row": X_test.head(5).index.tolist()
        })
        print(comparison_df)

        return X_train, y_train, X_test, y_test

    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.dates as mdates

    def visualize_predictions_with_table(self, data, y_test, predictions, n=10, tick_size=0.25,
                                         title="Model Predictions vs Actuals with Table"):
        # ✅ Ensure correct alignment by using `data` rows that match `y_test.index`
        aligned_data = data.copy()
        aligned_data["Actual Value"] = y_test.values
        aligned_data["Predicted Value"] = predictions

        # ✅ Use the last `n` rows for visualization
        df_to_visualize = aligned_data.loc[y_test.index].tail(n)

        if df_to_visualize.empty:
            print("Warning: The DataFrame for visualization is empty. Check your data alignment or filtering criteria.")
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

        # ✅ Creating a figure and a grid of subplots
        fig, ax = plt.subplots(figsize=(14, 7))

        # ✅ Set Y-axis limits dynamically
        ax.set_ylim(y_min, y_max)

        # ✅ Manually drawing the candlesticks
        for idx, row in df_to_visualize.iterrows():
            color = 'red' if row['Open'] > row['Close'] else 'green'
            ax.plot([idx, idx], [row['Low'], row['High']], color='black')  # Wick
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
        plt.title(title)
        plt.xlabel("Timestamp")
        plt.ylabel("High Price")
        ax.set_ylim(y_min, y_max)  # Extend Y-axis range dynamically
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return df_to_visualize


