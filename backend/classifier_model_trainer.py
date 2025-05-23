import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
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
        self.xgboost_num_classes = 0  # 0 = binary by default; set to >2 for multi-class

        # ✅ Store combined predictions for visualization
        self.classifier_predictions_df = None

        print("✅ ClassifierModelTrainer initialized!")

    def copy(self):
        import copy
        return copy.deepcopy(self)

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

        lgbm_model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            min_child_samples=10,
            verbose=-1,
            random_state=42
        )

        lgbm_model.fit(X_train, y_train)
        probabilities = lgbm_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 LightGBM Evaluation:")
        print(classification_report(y_test, predictions))
        print(f"\n🎯 LightGBM Accuracy: {accuracy:.4f}")

        return {
            "model": lgbm_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True),
            "predictions": predictions
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        print("\n🚀 Training XGBoost...")

        # ✅ Choose objective based on number of classes
        if self.xgboost_num_classes > 2:
            xgboost_params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": self.xgboost_num_classes,
                "learning_rate": 0.05,
                "max_depth": 10,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "verbosity": 1
            }
        else:
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

        # ✅ Train model
        xgb_model = xgb.XGBClassifier(**xgboost_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ✅ Predict and evaluate
        if self.xgboost_num_classes > 2:
            predictions = xgb_model.predict(X_test)
        else:
            probabilities = xgb_model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, predictions)

        print("\n📊 XGBoost Evaluation:")
        print(classification_report(y_test, predictions, zero_division=0))
        print(f"\n🎯 XGBoost Accuracy: {accuracy:.4f}")

        return {
            "model": xgb_model,
            "accuracy": accuracy,
            "evaluation_metrics": classification_report(y_test, predictions, output_dict=True, zero_division=0),
            "predictions": predictions
        }

    def train_all_classifiers(self, X_train, y_train, X_test, y_test, meta_timestamps_df):
        """
        Trains all classifiers and stores results in a single DataFrame.

        Parameters:
            X_train, y_train: Training dataset
            X_test, y_test: Test dataset
            meta_timestamps_df: DataFrame with ['Date', 'Time'], aligned with X_test
        """
        print("\n🚀 Training all classifiers...")

        self.rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
        self.lgbm_results = self.train_lightgbm(X_train, y_train, X_test, y_test)
        self.xgb_results = self.train_xgboost(X_train, y_train, X_test, y_test)

        # ✅ Use provided timestamp DataFrame
        timestamps = meta_timestamps_df.loc[X_test.index, ["Date", "Time"]].copy()
        timestamps["Timestamp"] = pd.to_datetime(timestamps["Date"] + " " + timestamps["Time"])
        timestamps = timestamps["Timestamp"].values

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

        # Use RF feature list as reference (all models now use wrappers)
        required_features = self.rf_results["model"].feature_names_in_
        X_input = X_input[required_features]

        rf_pred = (self.rf_results["model"].predict_proba(X_input)[:, 1] >= 0.5).astype(int)
        lgbm_pred = (self.lgbm_results["model"].predict_proba(X_input)[:, 1] >= 0.5).astype(int)
        xgb_pred = (self.xgb_results["model"].predict_proba(X_input)[:, 1] >= 0.5).astype(int)

        print("✅ Predictions generated for all classifiers!")

        return {
            "RandomForest": rf_pred[0],
            "LightGBM": lgbm_pred[0],
            "XGBoost": xgb_pred[0]
        }

    def evaluate_with_cross_val_smote(self, X_all, y_all, label="N/A"):
        """
        Perform StratifiedKFold Cross-Validation with SMOTE applied in each fold.

        Parameters:
            X_all (pd.DataFrame): Combined train + test features
            y_all (pd.Series): Combined train + test labels
            label (str): Optional label tag to group this variant

        Returns:
            List[Dict]: Metrics per model (F1, Precision for Label 1 & 0)
        """
        print(f"\n📊 CV + SMOTE Evaluation – {label}")

        # ✅ Use tuned model builders (same params as your main training flow)
        model_builders = {
            'RandomForest': lambda: RandomForestClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=5,
                min_samples_split=10,
                n_estimators=100,
                random_state=42
            ),
            'LightGBM': lambda: lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                learning_rate=0.05,
                num_leaves=31,
                max_depth=10,
                min_child_samples=10,
                verbose=-1,
                random_state=42
            ),
            'XGBoost': lambda: xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                learning_rate=0.05,
                max_depth=10,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=1,
                random_state=42
            )
        }

        results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, build_model in model_builders.items():
            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in skf.split(X_all, y_all):
                X_fold_train, y_fold_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
                X_fold_test, y_fold_test = X_all.iloc[test_idx], y_all.iloc[test_idx]

                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_fold_train, y_fold_train)

                model = build_model()
                model.fit(X_resampled, y_resampled)
                y_pred = model.predict(X_fold_test)

                y_true_all.extend(y_fold_test)
                y_pred_all.extend(y_pred)

            report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
            results.append({
                "Variant": label,
                "Model": model_name,
                "F1_Label_1": report['1']['f1-score'],
                "Precision_Label_1": report['1']['precision'],
                "Precision_Label_0": report['0']['precision']
            })
            print(
                f"  ✅ {model_name} | F1: {report['1']['f1-score']:.3f} | Prec1: {report['1']['precision']:.3f} | Prec0: {report['0']['precision']:.3f}"
            )

        return results
