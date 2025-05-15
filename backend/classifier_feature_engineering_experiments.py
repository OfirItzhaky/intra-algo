# classifier_feature_engineering_experiments.py
# ✅ Purpose: Isolate and evaluate changes to feature engineering (starting with Ichimoku features)

# === 1. Imports ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from label_generator import LabelGenerator
from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import lightgbm as lgb

# === 2. Custom Feature Generator with Ichimoku modifications ===
from feature_generator import FeatureGenerator

class CustomFeatureGenerator(FeatureGenerator):
    def create_all_features(self, df):
        df = super().create_all_features(df)

        # Core Ichimoku-style distance features
        df["High_26_bars_ago_minus_Close"] = df["High"].shift(-26) - df["Close"]
        df["Low_26_bars_ago_minus_Close"] = df["Low"].shift(-26) - df["Close"]
        df.rename(columns={"Chikou_minus_Close": "Close_minus_Close_26"}, inplace=True)

        # === Distance Features with Lookbacks 13 and 9 ===
        df["High_13_bars_ago_minus_Close"] = df["High"].shift(-13) - df["Close"]
        df["Low_13_bars_ago_minus_Close"] = df["Low"].shift(-13) - df["Close"]
        df["Close_minus_Close_13"] = df["Close"] - df["Close"].shift(13)

        df["High_9_bars_ago_minus_Close"] = df["High"].shift(-9) - df["Close"]
        df["Low_9_bars_ago_minus_Close"] = df["Low"].shift(-9) - df["Close"]
        df["Close_minus_Close_9"] = df["Close"] - df["Close"].shift(9)
        # === Short-Term Distance Features (3, 5 bars)
        df["High_5_bars_ago_minus_Close"] = df["High"].shift(-5) - df["Close"]
        df["Low_5_bars_ago_minus_Close"] = df["Low"].shift(-5) - df["Close"]
        df["Close_minus_Close_5"] = df["Close"] - df["Close"].shift(5)

        df["High_3_bars_ago_minus_Close"] = df["High"].shift(-3) - df["Close"]
        df["Low_3_bars_ago_minus_Close"] = df["Low"].shift(-3) - df["Close"]
        df["Close_minus_Close_3"] = df["Close"] - df["Close"].shift(3)

        df["High_2_bars_ago_minus_Close"] = df["High"].shift(-2) - df["Close"]
        df["High_1_bars_ago_minus_Close"] = df["High"].shift(-1) - df["Close"]
        df["Low_2_bars_ago_minus_Close"] = df["Low"].shift(-2) - df["Close"]
        df["Low_1_bars_ago_minus_Close"] = df["Low"].shift(-1) - df["Close"]

        return df


# === 3. Load Data ===
loader = DataLoader()
df = loader.load_from_csv("data/training/mes_new_2_15_mins_20000_bars.txt")
df.drop_duplicates(subset=["Date", "Time"], inplace=True)

# === 4. Generate Features ===
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda col: col.replace(" ", "_").replace("+_", "+").replace("FIB.", "FIB_").replace("Zero Cross", "Zero_Cross"))

feature_gen = CustomFeatureGenerator()
df_features = normalize_column_names(feature_gen.create_all_features(df))

# === 5. Generate Labels ===
label_gen = LabelGenerator()
df_labels = label_gen.green_red_bar_label_goal_d(df_features)

# === 6. Prepare Data ===
meta_columns = df_labels[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
processor = DataProcessor()
X_train, y_train, X_test, y_test = processor.prepare_dataset_for_regression_sequential(
    data=df_labels,
    target_column="long_good_bar_label",
    drop_target=True,
    split_ratio=0.8
)

X_test = X_test.reset_index(drop=True)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
leakage_cols = ["Next_High", "Next_Close", "Next_Open"]
X_train_bal = X_train_bal.drop(columns=leakage_cols, errors="ignore")
X_test = X_test.drop(columns=leakage_cols, errors="ignore")

# === 7. Train Classifiers ===
classifier_trainer = ClassifierModelTrainer()
meta_timestamps = df_labels[["Date", "Time"]].copy()
classifier_trainer.train_all_classifiers(X_train_bal, y_train_bal, X_test, y_test, meta_timestamps_df=meta_timestamps)

# === 8. Evaluate ===
for model_name, results in [
    ("RandomForest", classifier_trainer.rf_results),
    ("LightGBM", classifier_trainer.lgbm_results),
    ("XGBoost", classifier_trainer.xgb_results)
]:
    print(f"\n--- {model_name} Results After Ichimoku Enhancements ---")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(classification_report(y_test, results["predictions"], digits=3))

# === 9. LightGBM Feature Importance (Top 50 Gain) ===
lgb_model = classifier_trainer.lgbm_results["model"]
lgb.plot_importance(lgb_model, max_num_features=50, importance_type='gain', figsize=(12, 10))
plt.title("Top 50 LightGBM Features by Gain")
plt.tight_layout()
plt.show()

# === 10. EDA: KDEs for Ichimoku Features ===
ichimoku_cols = [
    "High_26_bars_ago_minus_Close",
    "High_26_bars_ago_minus_High",
    "Chikou_minus_Close",
    "Volume",  # baseline anchor for comparison
]



print("\n📊 Correlation with target (long_good_bar_label):")
df_labeled = label_gen.green_red_bar_label_goal_d(df_features.copy())
for col in ichimoku_cols:
    if col in df_labeled.columns:
        corr = df_labeled[[col, "long_good_bar_label"]].corr().iloc[0, 1]
        print(f"{col}: {corr:.4f}")

# print("\n📊 Plotting KDEs for engineered Ichimoku features:")
# for col in ichimoku_cols:
#     if col in df_labeled.columns:
#         plt.figure(figsize=(6, 3))
#         sns.kdeplot(data=df_labeled, x=col, hue="long_good_bar_label", fill=True)
#         plt.title(f"{col} by Green/Red Label")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

print ("Done!")