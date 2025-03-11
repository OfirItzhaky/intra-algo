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
        """Initializes the ClassifierModelTrainer."""

        self.rf_results = None
        self.lgbm_results = None
        self.xgb_results = None

        print("✅ ClassifierModelTrainer initialized!")

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Trains and evaluates a RandomForestClassifier.

        Parameters:
            X_train, y_train: Training dataset
            X_test, y_test: Test dataset

        Returns:
            dict: Contains evaluation metrics, feature importance, predictions, and trained model.
        """
        print("\n🚀 Training RandomForest...")

        # ✅ Define the model
        rf_model = RandomForestClassifier(
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            n_estimators=100,
            random_state=42
        )

        # ✅ Train the model
        rf_model.fit(X_train, y_train)
        print("✅ RandomForest training complete.")

        # ✅ Predict on test data
        predictions = rf_model.predict(X_test)

        # ✅ Evaluate performance
        evaluation_metrics = classification_report(y_test, predictions, output_dict=True)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 RandomForest Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 RandomForest Accuracy: {accuracy:.4f}")

        # ✅ Feature Importance
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        )
        print("\n📊 Top 10 Feature Importances:\n", feature_importance_df.head(10))

        return {
            "model": rf_model,
            "accuracy": accuracy,
            "evaluation_metrics": evaluation_metrics,
            "feature_importance": feature_importance_df,
            "predictions": predictions
        }

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """
        Trains and evaluates a LightGBM classifier.

        Parameters:
            X_train, y_train: Training dataset
            X_test, y_test: Test dataset

        Returns:
            dict: Contains evaluation metrics, feature importance, predictions, and trained model.
        """
        print("\n🚀 Training LightGBM...")

        # ✅ Define LightGBM parameters
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

        # ✅ Prepare dataset for LightGBM
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

        # ✅ Training LightGBM with progress bar
        with tqdm(total=100, desc="LightGBM Training Progress") as pbar:
            def update_progress(env):
                pbar.update()

            lgb_model = lgb.train(lightgbm_params, lgb_train, valid_sets=[lgb_test], callbacks=[update_progress])

        # ✅ Predict & Evaluate LightGBM
        probabilities = lgb_model.predict(X_test)
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 LightGBM Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 LightGBM Accuracy: {accuracy:.4f}")

        # ✅ Feature Importance
        feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": lgb_model.feature_importance()}).sort_values(
            by="Importance", ascending=False
        )
        print("\n📊 Top 10 Feature Importances:\n", feature_importance_df.head(10))

        return {
            "model": lgb_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "feature_importance": feature_importance_df,
            "predictions": predictions
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Trains and evaluates an XGBoost classifier.

        Parameters:
            X_train, y_train: Training dataset
            X_test, y_test: Test dataset

        Returns:
            dict: Contains evaluation metrics, feature importance, predictions, and trained model.
        """
        print("\n🚀 Training XGBoost...")

        # ✅ Define XGBoost parameters
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

        # ✅ Train XGBoost with progress bar
        xgb_model = xgb.XGBClassifier(**xgboost_params)

        with tqdm(total=100, desc="XGBoost Training Progress") as pbar:
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            pbar.update(100)  # Ensure full progress bar update

        # ✅ Predict & Evaluate XGBoost
        probabilities = xgb_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 XGBoost Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 XGBoost Accuracy: {accuracy:.4f}")

        # ✅ Feature Importance
        feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": xgb_model.feature_importances_}).sort_values(
            by="Importance", ascending=False
        )
        print("\n📊 Top 10 Feature Importances:\n", feature_importance_df.head(10))

        return {
            "model": xgb_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "feature_importance": feature_importance_df,
            "predictions": predictions
        }
