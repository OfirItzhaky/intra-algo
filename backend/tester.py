# This section tests classifier results for "green_red_bar_label_goal_d"
# The goal: Quickly evaluate and improve classifier performance independently.

# === 1. Imports ===
import pandas as pd
from data_loader import DataLoader
from feature_generator import FeatureGenerator
from label_generator import LabelGenerator
from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor
from regression_model_trainer import RegressionModelTrainer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# === 2. Load Data ===
loader = DataLoader()
df = loader.load_from_csv("data/training/mes_new_2_15_mins_20000_bars.txt")
df.drop_duplicates(subset=["Date", "Time"], inplace=True)

# === 3. Generate Features ===
# === 3. Generate Features ===
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda col: col.replace(" ", "_").replace("+_", "+").replace("FIB.", "FIB_").replace("Zero Cross", "Zero_Cross"))

feature_gen = FeatureGenerator()

# ✅ Apply normalization right after generation for training
df_features = normalize_column_names(feature_gen.create_all_features(df))


# === 4. Generate Labels for Regression ===
label_gen = LabelGenerator()
df_regression_labels = label_gen.elasticnet_label_next_high(df_features)

# === 5. Train Regression Model ===
meta_columns = df_regression_labels[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
regression_trainer = RegressionModelTrainer(include_prices=False, apply_filter=False)
regression_trainer.prepare_data(df_regression_labels)
regression_trainer.train_model()
regression_trainer.make_predictions()

# ✅ Explicitly create x_test_with_meta and reset index for alignment
regression_trainer.x_test_with_meta = regression_trainer.x_test.copy()
regression_trainer.x_test_with_meta = regression_trainer.x_test_with_meta.join(
    meta_columns[["Date", "Time"]].loc[regression_trainer.x_test.index]
).reset_index(drop=True)

# === 6. Generate Green/Red Labels ===
df_labels = label_gen.green_red_bar_label_goal_d(df_features)

# Verify labels
print("Label distribution:")
print(df_labels["long_good_bar_label"].value_counts(normalize=True))

# === 7. Prepare Data for Classifier ===
processor = DataProcessor()
X_train, y_train, X_test, y_test = processor.prepare_dataset_for_regression_sequential(
    data=df_labels,
    target_column="long_good_bar_label",
    drop_target=True,
    split_ratio=0.8
)

# ✅ Reset index to match x_test_with_meta
X_test = X_test.reset_index(drop=True)

# === 8. Balance Training Set (SMOTE) ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === 9. Train Classifiers (now regression trainer properly initialized) ===
classifier_trainer = ClassifierModelTrainer()
# Drop label-leakage columns from both train/test sets before training
leakage_cols = ["Next_High", "Next_Close", "Next_Open"]
X_train_bal = X_train_bal.drop(columns=leakage_cols, errors="ignore")
X_test = X_test.drop(columns=leakage_cols, errors="ignore")

classifier_trainer.train_all_classifiers(
    X_train_bal, y_train_bal, X_test, y_test, trainer=regression_trainer
)


# === 10. Evaluate and Print Metrics ===
for model_name, results in [
    ("RandomForest", classifier_trainer.rf_results),
    ("LightGBM", classifier_trainer.lgbm_results),
    ("XGBoost", classifier_trainer.xgb_results)
]:
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, results["predictions"], digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, results["predictions"]))

# === 11. Save Results (Optional) ===
# results_df = pd.DataFrame({
#     "Actual": y_test,
#     "RF_Predictions": classifier_trainer.rf_results["predictions"],
#     "LGBM_Predictions": classifier_trainer.lgbm_results["predictions"],
#     "XGB_Predictions": classifier_trainer.xgb_results["predictions"]
# })
# results_df.to_csv("classifier_test_results.csv", index=False)
## === 12. Inference on New Forward Data (for all models) ===
from sklearn.metrics import classification_report, confusion_matrix

# Separate threshold sweeps for each model
THRESHOLDS_MAP = {
    "XGBoost": [0.50, 0.60, 0.70, 0.80],
    "RandomForest": [0.50, 0.60, 0.70, 0.80, 0.90],
    "LightGBM":  [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
}

# Load new data and generate features
forward_df = loader.load_from_csv("data/training/mes_new_2_15_mins_2_months_forward_for_testing.txt")
forward_df.drop_duplicates(subset=["Date", "Time"], inplace=True)

# ✅ Apply normalization to forward data as well
forward_features = normalize_column_names(feature_gen.create_all_features(forward_df).copy())


# Drop any potential leakage
forward_features.drop(columns=["Next_High", "Next_Close", "Next_Open"], errors="ignore", inplace=True)

# Add true labels for evaluation
forward_features["Next_Open"] = forward_features["Open"].shift(-1)
forward_features["Next_Close"] = forward_features["Close"].shift(-1)
forward_features["True_Label"] = (forward_features["Next_Close"] > forward_features["Next_Open"]).astype(int)
forward_eval_base = forward_features.dropna(subset=["True_Label"]).copy()

## === 13. Evaluate All Classifiers ===
print("\n📊 === Step 13: Evaluation Across Multiple Thresholds ===")

for model_name, results in [
    ("RandomForest", classifier_trainer.rf_results),
    ("LightGBM", classifier_trainer.lgbm_results),
    ("XGBoost", classifier_trainer.xgb_results)
]:
    model = results["model"]

    features_needed = None

    # Try the sklearn API first
    if hasattr(model, "feature_names_in_"):
        features_needed = model.feature_names_in_
        print(f"✅ [{model_name}] Using model.feature_names_in_: {len(features_needed)} features")

    # Fall back to .feature_name() only if needed
    elif hasattr(model, "feature_name") and callable(model.feature_name):
        features_needed = model.feature_name()
        print(f"✅ [{model_name}] Using model.feature_name(): {len(features_needed)} features")

    # Final fallback
    elif hasattr(model, "feature_names"):
        features_needed = model.feature_names
        print(f"✅ [{model_name}] Using model.feature_names: {len(features_needed)} features")

    else:
        raise AttributeError(f"❌ Cannot determine feature names for {model_name}")

    missing = set(features_needed) - set(forward_features.columns)
    if missing:
        print(f"⚠️ Skipping {model_name}: missing model features ({len(missing)}): {missing}")
        continue

    X_forward = forward_features[features_needed].copy()
    probs = model.predict_proba(X_forward)[:, 1]

    print(f"\n📍 {model_name} Threshold Sweep:")

    for thresh in THRESHOLDS_MAP[model_name]:
        forward_eval = forward_eval_base.copy()
        forward_eval["Predicted_Label"] = (probs >= thresh).astype(int)

        predicted_1 = forward_eval[forward_eval["Predicted_Label"] == 1]
        n_total = len(predicted_1)
        n_correct = (predicted_1["True_Label"] == 1).sum()
        precision = n_correct / n_total if n_total > 0 else 0

        report = classification_report(
            forward_eval["True_Label"],
            forward_eval["Predicted_Label"],
            digits=3,
            output_dict=True
        )

        print(f"\n🔹 Threshold = {thresh}")
        print(f"✅ Bars predicted as label=1: {n_total}")
        print(f"🎯 Correct predictions: {n_correct}")
        print(f"📌 Precision (label=1): {precision:.3f}")
        print(f"📈 Recall (label=1): {report['1']['recall']:.3f}")
        print(f"📊 F1 Score: {report['1']['f1-score']:.3f}")
