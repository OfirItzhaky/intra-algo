import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

class ClassifierModelTrainer:
    """
    A class to train and evaluate classifiers for predicting 'good bar' classifications.
    Supports RandomForest, LightGBM, and XGBoost.
    """

    def __init__(self):
        """Initializes the ClassifierModelTrainer and storage for per-bar predictions."""

        self.rf_results = None
        self.lgbm_results = None
        self.xgb_results = None

        # ✅ Store combined predictions for visualization
        self.classifier_predictions_df = None

        print("✅ ClassifierModelTrainer initialized!")

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        print("\n🚀 Training RandomForest...")

        rf_model = RandomForestClassifier(
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            n_estimators=100,
            random_state=42
        )

        rf_model.fit(X_train, y_train)
        predictions = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 RandomForest Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 RandomForest Accuracy: {accuracy:.4f}")

        return {
            "model": rf_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "predictions": predictions
        }

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        print("\n🚀 Training LightGBM...")

        lightgbm_params = {
            "objective": "binary",
            "metric": "binary_error",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 10,
            "min_data_in_leaf": 10,
            "verbose": -1
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

        lgb_model = lgb.train(lightgbm_params, lgb_train, valid_sets=[lgb_test])

        probabilities = lgb_model.predict(X_test)
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 LightGBM Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 LightGBM Accuracy: {accuracy:.4f}")

        return {
            "model": lgb_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "predictions": predictions
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("\n🚀 Training XGBoost...")

        xgboost_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 10,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 1
        }

        xgb_model = xgb.XGBClassifier(**xgboost_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        probabilities = xgb_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 XGBoost Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 XGBoost Accuracy: {accuracy:.4f}")

        return {
            "model": xgb_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "predictions": predictions
        }

    def train_all_classifiers(self, X_train, y_train, X_test, y_test, trainer):
        """
        Trains all classifiers and stores results in a single DataFrame.

        Parameters:
            X_train, y_train: Training dataset
            X_test, y_test: Test dataset
        """
        print("\n🚀 Training all classifiers...")

        # ✅ Train all classifiers
        self.rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
        self.lgbm_results = self.train_lightgbm(X_train, y_train, X_test, y_test)
        self.xgb_results = self.train_xgboost(X_train, y_train, X_test, y_test)

        # ✅ Extract timestamps from trainer.x_test_with_meta using X_test.index
        timestamps = trainer.x_test_with_meta.loc[X_test.index, ["Date", "Time"]].copy()  # ✅ Extract correct timestamps

        # ✅ Convert Date + Time to a single datetime index
        timestamps["Timestamp"] = pd.to_datetime(timestamps["Date"] + " " + timestamps["Time"])
        timestamps = timestamps["Timestamp"].values  # ✅ Convert to array for indexing

        # ✅ Store classifier predictions with aligned timestamps
        self.classifier_predictions_df = pd.DataFrame({
            "RandomForest": self.rf_results["predictions"],
            "LightGBM": self.lgbm_results["predictions"],
            "XGBoost": self.xgb_results["predictions"]
        }, index=timestamps)

        print("✅ All classifier predictions stored successfully!")

    def predict_all_classifiers(self, X_input: pd.DataFrame):
        """
        Runs all trained classifiers on the given input data and returns predictions.

        Parameters:
            X_input (pd.DataFrame): Input data for classification (should match training feature set).

        Returns:
            dict: A dictionary containing predictions from all classifiers.
        """

        if self.rf_results is None or self.lgbm_results is None or self.xgb_results is None:
            raise ValueError("❌ Classifier models are not trained! Call `train_all_classifiers` first.")

        print("\n🚀 Running classifier predictions...")

        # ✅ Ensure input is aligned with training features
        required_features = self.rf_results["model"].feature_names_in_  # Assuming all classifiers use the same features
        X_input = X_input[required_features]  # ✅ Keep only relevant columns

        # ✅ Run predictions for each classifier
        rf_pred = self.rf_results["model"].predict(X_input)
        lgbm_pred = (self.lgbm_results["model"].predict(X_input) >= 0.5).astype(int)
        xgb_pred = (self.xgb_results["model"].predict_proba(X_input)[:, 1] >= 0.5).astype(int)

        print("✅ Predictions generated for all classifiers!")

        return {
            "RandomForest": rf_pred[0],  # Extract single prediction
            "LightGBM": lgbm_pred[0],
            "XGBoost": xgb_pred[0]
        }
