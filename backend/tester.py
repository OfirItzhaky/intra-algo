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
df = loader.load_from_csv("data/training/mes_2_new.csv")
df.drop_duplicates(subset=["Date", "Time"], inplace=True)

# === 3. Generate Features ===
feature_gen = FeatureGenerator()
df_features = feature_gen.create_all_features(df)

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
