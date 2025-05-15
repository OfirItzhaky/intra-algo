# classifier_feature_engineering_experiments.py
# ✅ Purpose: Isolate and evaluate changes to feature engineering (starting with Ichimoku features)

# === 1. Imports ===
import pandas as pd
from data_loader import DataLoader
from label_generator import LabelGenerator
from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor
from regression_model_trainer import RegressionModelTrainer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# === 2. Custom Feature Generator with Ichimoku modifications ===
from feature_generator import FeatureGenerator

class CustomFeatureGenerator(FeatureGenerator):
    def create_all_features(self, df):
        df = super().create_all_features(df)

        # ❌ Option: Drop Ichimoku features completely
        df.drop(columns=[
            "Tenkan", "Kijun", "Chikou", "SenkouSpan_A", "SenkouSpan_B"
        ], errors="ignore", inplace=True)

        # 🧪 Option: Add Ichimoku variants later (e.g., trend strength, differences)
        # Example:
        # df["Chikou_vs_Close"] = df["Chikou"] - df["Close"]
        # df["Tenkan_minus_Kijun"] = df["Tenkan"] - df["Kijun"]

        return df

# === 3. Load Data ===
loader = DataLoader()
df = loader.load_from_csv("data/training/mes_new_2_15_mins_20000_bars.txt")
df.drop_duplicates(subset=["Date", "Time"], inplace=True)

# === 4. Generate Features ===
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda col: col.replace(" ", "_").replace("+_", "+")
                     .replace("FIB.", "FIB_").replace("Zero Cross", "Zero_Cross"))

feature_gen = CustomFeatureGenerator()
df_features = normalize_column_names(feature_gen.create_all_features(df))

# === 5. Generate Labels ===
label_gen = LabelGenerator()
df_labels = label_gen.green_red_bar_label_goal_d(df_features)

# === 6. Train Regression Model for alignment ===
meta_columns = df_labels[["Date", "Time", "Open", "High", "Low", "Close"]].copy()

# === 7. Prepare Data ===
processor = DataProcessor()
X_train, y_train, X_test, y_test = processor.prepare_dataset_for_regression_sequential(
    data=df_labels,
    target_column="long_good_bar_label",
    drop_target=True,
    split_ratio=0.8
)

X_test = X_test.reset_index(drop=True)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === 8. Train Classifiers ===
leakage_cols = ["Next_High", "Next_Close", "Next_Open"]
X_train_bal = X_train_bal.drop(columns=leakage_cols, errors="ignore")
X_test = X_test.drop(columns=leakage_cols, errors="ignore")

classifier_trainer = ClassifierModelTrainer()
# ✅ Extract timestamps separately
meta_timestamps = df_labels[["Date", "Time"]].copy()

classifier_trainer.train_all_classifiers(
    X_train_bal, y_train_bal, X_test, y_test,
    meta_timestamps_df=meta_timestamps
)


# === 9. Evaluate ===
for model_name, results in [
    ("RandomForest", classifier_trainer.rf_results),
    ("LightGBM", classifier_trainer.lgbm_results),
    ("XGBoost", classifier_trainer.xgb_results)
]:
    print(f"\n--- {model_name} Results After Ichimoku Removal ---")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(classification_report(y_test, results["predictions"], digits=3))
# classifier_feature_engineering_experiments.py

"# This file explores why Ichimoku components show high importance in LightGBM."
# Goal: Understand whether to preserve, expand, or drop them based on behavior, correlation, and label relevance.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from feature_generator import FeatureGenerator
from label_generator import LabelGenerator
from sklearn.preprocessing import StandardScaler

# === Step 1: Load and Preprocess Data ===
loader = DataLoader()
df = loader.load_from_csv("data/training/mes_new_2_15_mins_20000_bars.txt")
df.drop_duplicates(subset=["Date", "Time"], inplace=True)

feature_gen = FeatureGenerator()
df_features = feature_gen.create_all_features(df)

# Normalize column names
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda col: col.replace(" ", "_").replace("+_", "+").replace("FIB.", "FIB_").replace("Zero Cross", "Zero_Cross"))

df_features = normalize_column_names(df_features)

# === Step 2: Explore Ichimoku Components ===
ichimoku_cols = ["Tenkan", "Kijun", "Chikou", "SenkouSpan_A", "SenkouSpan_B"]
print("\n📌 Ichimoku columns found:", [col for col in ichimoku_cols if col in df_features.columns])

# === Step 3: Plot Distributions ===
print("\n📊 Plotting Ichimoku Component Distributions...")
for col in ichimoku_cols:
    if col in df_features.columns:
        plt.figure(figsize=(6, 3))
        sns.histplot(df_features[col].dropna(), bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Step 4: Add Labels and Check Correlation ===
label_gen = LabelGenerator()
df_labeled = label_gen.green_red_bar_label_goal_d(df_features.copy())

print("\n📊 Correlation with target (long_good_bar_label):")
for col in ichimoku_cols:
    if col in df_labeled.columns:
        corr = df_labeled[[col, "long_good_bar_label"]].corr().iloc[0, 1]
        print(f"{col}: {corr:.4f}")

# === Step 5: Plot KDE by Class ===
print("\n📊 Plotting KDE plots grouped by label:")
for col in ichimoku_cols:
    if col in df_labeled.columns:
        plt.figure(figsize=(6, 3))
        sns.kdeplot(data=df_labeled, x=col, hue="long_good_bar_label", fill=True)
        plt.title(f"{col} by Green/Red Label")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
