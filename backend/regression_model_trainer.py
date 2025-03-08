import pandas as pd
from sklearn.linear_model import ElasticNet
from data_processor import DataProcessor


class RegressionModelTrainer:
    def __init__(self, include_prices: bool):
        """
        Initializes the Regression Model Trainer.

        :param include_prices: Boolean flag to decide whether to include price columns in training.
        """
        self.include_prices = include_prices
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.predictions = None

    def prepare_data(self, data: pd.DataFrame):
        """
        Prepares training and testing datasets for regression.

        :param data: DataFrame containing training data with features and target.
        """
        processor = DataProcessor()

        # ✅ Prepare dataset
        self.x_train, self.y_train, self.x_test, self.y_test = processor.prepare_dataset_for_regression_sequential(
            data=data,
            target_column="Next_High",
            drop_target=True,
            split_ratio=0.8
        )

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
        return self.predictions

    def get_model_summary(self):
        """
        Returns summary statistics of the trained model.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not generated. Call `make_predictions()` first.")

        return {
            "train_size": len(self.x_train),
            "test_size": len(self.x_test),
            "num_features": self.x_train.shape[1]
        }
