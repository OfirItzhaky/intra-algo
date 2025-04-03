import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from data_processor import DataProcessor

class RegressionModelTrainer:
    def __init__(self, include_prices: bool, apply_filter: bool, filter_threshold: float = 4.0):
        """
        Initializes the Regression Model Trainer.

        :param include_prices: Boolean flag to decide whether to include price columns in training.
        :param apply_filter: Boolean flag to decide whether to filter extreme errors before computing metrics.
        :param filter_threshold: Maximum allowed error for filtering predictions.
        """
        self.include_prices = include_prices
        self.apply_filter = apply_filter
        self.filter_threshold = filter_threshold
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_test_filtered = None  # ✅ Store filtered dataset for classifiers
        self.y_test_filtered = None
        self.predictions_filtered = None
        self.predictions = None
        self.model = None
        self.mse_filtered = None
        self.r2_filtered = None
        self.mse_unfiltered = None
        self.r2_unfiltered = None
        self.last_train_timestamp = None
        self.last_test_timestamp = None
        self.filter_size = 0  # Number of removed rows
        self.regression_figure = None

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def prepare_data(self, data: pd.DataFrame):
        """
        Prepares training and testing datasets for regression.

        :param data: DataFrame containing training data with features and target.
        """
        processor = DataProcessor()
        self.x_train, self.y_train, self.x_test, self.y_test = processor.prepare_dataset_for_regression_sequential(
            data=data,
            target_column="Next_High",
            drop_target=True,
            split_ratio=0.8
        )

        # ✅ Store last timestamps for tracking
        self.last_train_timestamp = f"{data.iloc[len(self.x_train) - 1]['Date']} {data.iloc[len(self.x_train) - 1]['Time']}"
        self.last_test_timestamp = f"{data.iloc[len(self.x_train) + len(self.x_test) - 1]['Date']} {data.iloc[len(self.x_train) + len(self.x_test) - 1]['Time']}"

        # ✅ Drop price columns if `include_prices` is **False**
        if not self.include_prices:
            price_columns = ["Open", "High", "Low", "Close"]
            self.x_train = self.x_train.drop(columns=price_columns, errors="ignore")
            self.x_test = self.x_test.drop(columns=price_columns, errors="ignore")

    def train_model(self):
        """
        Trains an ElasticNet regression model on the prepared dataset.
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data is not prepared. Call `prepare_data()` first.")

        self.model = ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=5000)
        self.model.fit(self.x_train, self.y_train)

    def make_predictions(self):
        """
        Makes predictions on the test set.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call `train_model()` first.")

        self.predictions = self.model.predict(self.x_test)

    def get_model_summary(self):
        """
        Returns summary statistics of the trained model.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not generated. Call `make_predictions()` first.")

        return {
            "train_size": len(self.x_train),
            "test_size": len(self.x_test),
            "num_features": self.x_train.shape[1],
            "last_train_timestamp": self.last_train_timestamp,
            "last_test_timestamp": self.last_test_timestamp,
            "mse_unfiltered": self.mse_unfiltered,
            "r2_unfiltered": self.r2_unfiltered,
            "mse_filtered": self.mse_filtered,
            "r2_filtered": self.r2_filtered,
            "filter_threshold": self.filter_threshold,
            "filter_size": self.filter_size
        }
