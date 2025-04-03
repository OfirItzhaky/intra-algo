#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_generator import FeatureGenerator
from label_generator import LabelGenerator
from regression_model_trainer import RegressionModelTrainer
from classifier_model_trainer import ClassifierModelTrainer
from data_processor import DataProcessor


# In[2]:


# ✅ Load CSV
loader = DataLoader()
training_df_raw = loader.load_from_csv("data/training/mes_2_new.csv")  # change filename accordingly

# ✅ Drop any duplicate timestamps
training_df_raw = training_df_raw.drop_duplicates(subset=["Date", "Time"])

training_df_raw.head()


# In[3]:


feature_generator = FeatureGenerator()
training_df_features = feature_generator.create_all_features(training_df_raw)

training_df_features.head()


# In[4]:


label_generator = LabelGenerator()
training_df_labeled = label_generator.elasticnet_label_next_high(training_df_features)


# In[5]:


trainer = RegressionModelTrainer(
    include_prices=False,         # Set to True if you want price columns
    apply_filter=True,            # Filter based on error threshold
    filter_threshold=4.0          # Can adjust this value
)

trainer.prepare_data(training_df_features)
trainer.train_model()
trainer.make_predictions()


# In[6]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(trainer.y_test, trainer.predictions)
r2 = r2_score(trainer.y_test, trainer.predictions)

print("📊 Regression Metrics (Unfiltered):")
print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")
print(f"Test Samples: {len(trainer.y_test)}")
print(f"Features Used: {trainer.x_train.shape[1]}")


# In[7]:


# Make a full copy before modifying to avoid SettingWithCopyWarning
training_df_labeled = training_df_labeled.copy()

# Safely create and set Timestamp index
training_df_labeled["Timestamp"] = pd.to_datetime(training_df_labeled["Date"] + " " + training_df_labeled["Time"])
training_df_labeled = training_df_labeled.set_index("Timestamp").sort_index()


# In[8]:


# ✅ Step 1: Reset index to allow access by integer
training_df_labeled = training_df_labeled.reset_index(drop=True)

# ✅ Step 2: Clip x_test indices to max valid row index (prevent KeyError)
max_valid_index = len(training_df_labeled) - 1
safe_indices = trainer.x_test.index[trainer.x_test.index <= max_valid_index]

# ✅ Optional: Warn if any were dropped
if len(safe_indices) < len(trainer.x_test.index):
    print(f"⚠️ {len(trainer.x_test.index) - len(safe_indices)} test rows dropped due to index overflow.")

# ✅ Step 3: Build x_test_with_meta with safe indices only
x_test_with_meta = trainer.x_test.loc[safe_indices].copy()

# Add Date and Time using safe indices
x_test_with_meta["Date"] = training_df_labeled.loc[safe_indices, "Date"].values
x_test_with_meta["Time"] = training_df_labeled.loc[safe_indices, "Time"].values

# Create Timestamp index
x_test_with_meta["Timestamp"] = pd.to_datetime(x_test_with_meta["Date"] + " " + x_test_with_meta["Time"])
x_test_with_meta = x_test_with_meta.set_index("Timestamp").sort_index()

# ✅ Step 4: Rebuild training_df_labeled with Timestamp index (only once!)
training_df_labeled["Timestamp"] = pd.to_datetime(training_df_labeled["Date"] + " " + training_df_labeled["Time"])
training_df_labeled = training_df_labeled.set_index("Timestamp").sort_index()

# ✅ Step 5: Join OHLC columns
trainer.x_test_with_meta = x_test_with_meta.join(
    training_df_labeled[["Open", "High", "Low", "Close"]],
    how="left"
)

# ✅ Step 6: Add predictions and shifted columns
trainer.x_test_with_meta["Predicted_High"] = trainer.predictions[:len(trainer.x_test_with_meta)]
trainer.x_test_with_meta["Prev_Close"] = trainer.x_test_with_meta["Close"].shift(1)
trainer.x_test_with_meta["Prev_Predicted_High"] = trainer.x_test_with_meta["Predicted_High"].shift(1)

# ✅ Step 7: Clean result
df_with_meta = trainer.x_test_with_meta.dropna(subset=["Prev_Close", "Prev_Predicted_High"])

print("✅ Metadata attached and df_with_meta ready. Shape:", df_with_meta.shape)


# In[17]:


# Step 1: Attach predictions directly to df
df_for_labeling = trainer.x_test_with_meta.copy()
df_for_labeling["Predicted_High"] = trainer.predictions[:len(df_for_labeling)]

# Step 2: Shift predictions and closes for labeling logic
df_for_labeling["Prev_Predicted_High"] = df_for_labeling["Predicted_High"].shift(1)
df_for_labeling["Prev_Close"] = df_for_labeling["Close"].shift(1)

# Step 3: Drop NaNs due to shifting
df_for_labeling = df_for_labeling.dropna(subset=["Prev_Close", "Prev_Predicted_High"])

# Step 4: Correct labeling
df_labeled_old = label_generator.add_good_bar_label(df_for_labeling)
df_labeled_all = label_generator.long_good_bar_label_all(df_for_labeling)
df_labeled_bullish = label_generator.long_good_bar_label_bullish_only(df_for_labeling)


# In[27]:


processor = DataProcessor()

# 🔵 old version
X_train_old, y_train_old, X_test_old, y_test_old = processor.prepare_dataset_for_regression_sequential(
    data=df_labeled_old,
    target_column="good_bar_prediction_outside_of_boundary",
    drop_target=True,
    split_ratio=0.8
)

classifier_trainer_old = ClassifierModelTrainer()
classifier_trainer_old.train_all_classifiers(X_train_old, y_train_old, X_test_old, y_test_old, trainer)



# 🔵 All-bar version
X_train_all, y_train_all, X_test_all, y_test_all = processor.prepare_dataset_for_regression_sequential(
    data=df_labeled_all,
    target_column="long_good_bar_label",
    drop_target=True,
    split_ratio=0.8
)

classifier_trainer_all = ClassifierModelTrainer()
classifier_trainer_all.train_all_classifiers(X_train_all, y_train_all, X_test_all, y_test_all, trainer)

# 🟣 Bullish-only version
X_train_bullish, y_train_bullish, X_test_bullish, y_test_bullish = processor.prepare_dataset_for_regression_sequential(
    data=df_labeled_bullish,
    target_column="long_good_bar_label",
    drop_target=True,
    split_ratio=0.8
)

classifier_trainer_bullish = ClassifierModelTrainer()
classifier_trainer_bullish.train_all_classifiers(X_train_bullish, y_train_bullish, X_test_bullish, y_test_bullish, trainer)


# 
# # Classifier Analysis

# In[12]:


import matplotlib.pyplot as plt

# Check distribution of labels in the bullish-only labeled dataset
good_bars = df_labeled_bullish[df_labeled_bullish["long_good_bar_label"] == 1]

total_bars = len(df_labeled_bullish)
total_good_bars = len(good_bars)
percentage_good = (total_good_bars / total_bars) * 100

print(f"Total Bars: {total_bars}")
print(f"Total 'Good' Labeled Bars (Potential Trades): {total_good_bars} ({percentage_good:.2f}%)\n")

# Count potential trades per day
good_bars['Date'] = pd.to_datetime(good_bars['Date'])
trades_per_day = good_bars.groupby(good_bars['Date'].dt.date).size()

print("📅 Potential trades per day statistics:")
print(trades_per_day.describe())

# Simple visualization of potential trades per day
plt.figure(figsize=(10, 5))
trades_per_day.plot(kind='bar', color='skyblue')
plt.title('📅 Potential Trades Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Trades')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.show()

# Check timing distribution (hours)
good_bars['Hour'] = pd.to_datetime(good_bars['Time']).dt.hour
trades_by_hour = good_bars.groupby('Hour').size()

print("\n⏰ Trades distribution by hour:")
print(trades_by_hour)

# Visualization of trades by hour
plt.figure(figsize=(10, 5))
trades_by_hour.plot(kind='bar', color='orange')
plt.title('⏰ Trades Distribution by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Trades')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.show()


# In[ ]:





# In[28]:


from sklearn.metrics import classification_report

# Label variants and their associated trainers/datasets
variants = {
    "Old": {
        "trainer": classifier_trainer_old,
        "X_train": X_train_old,
        "y_train": y_train_old,
        "X_test": X_test_old,
        "y_test": y_test_old,
    },
    "All": {
        "trainer": classifier_trainer_all,
        "X_train": X_train_all,
        "y_train": y_train_all,
        "X_test": X_test_all,
        "y_test": y_test_all,
    },
    "Bullish": {
        "trainer": classifier_trainer_bullish,
        "X_train": X_train_bullish,
        "y_train": y_train_bullish,
        "X_test": X_test_bullish,
        "y_test": y_test_bullish,
    },
}

print("🔍 Checking for overfitting and data leakage across all label types:\n")

for label_name, data in variants.items():
    print(f"🏷️ Label Variant: {label_name}")
    models = {
        'RandomForest': data["trainer"].rf_results["model"],
        'LightGBM': data["trainer"].lgbm_results["model"],
        'XGBoost': data["trainer"].xgb_results["model"]
    }

    for name, model in models.items():
        print(f"\nModel: {name}")

        # Predictions on training and test
        if name in ['LightGBM', 'XGBoost']:
            train_preds = (model.predict(data["X_train"]) >= 0.5).astype(int)
            test_preds = (model.predict(data["X_test"]) >= 0.5).astype(int)
        else:
            train_preds = model.predict(data["X_train"])
            test_preds = model.predict(data["X_test"])

        # Reports
        train_report = classification_report(data["y_train"], train_preds, digits=2, output_dict=True)
        test_report = classification_report(data["y_test"], test_preds, digits=2, output_dict=True)

        print("  ➡️ Training vs. Testing Performance (Precision, Recall, F1):")
        print(f"    - Train: {train_report['1']['precision']:.2f}, {train_report['1']['recall']:.2f}, {train_report['1']['f1-score']:.2f}")
        print(f"    - Test:  {test_report['1']['precision']:.2f}, {test_report['1']['recall']:.2f}, {test_report['1']['f1-score']:.2f}")

        # Check for gap
        precision_gap = abs(train_report['1']['precision'] - test_report['1']['precision'])
        recall_gap = abs(train_report['1']['recall'] - test_report['1']['recall'])
        f1_gap = abs(train_report['1']['f1-score'] - test_report['1']['f1-score'])

        if precision_gap > 0.1 or recall_gap > 0.1 or f1_gap > 0.1:
            print("  ⚠️ Potential overfitting or data leakage detected!")
        else:
            print("  ✅ No significant signs of overfitting or leakage.")

    print("\n" + "=" * 65 + "\n")


# # Overfitting check and analysis

# In[29]:


# Sanity check for overlapping indexes between training and testing sets (all label types)
variants = {
    "Old": (X_train_old, X_test_old),
    "All": (X_train_all, X_test_all),
    "Bullish": (X_train_bullish, X_test_bullish),
}

print("🔎 Checking for overlapping indexes (potential data leakage):\n")

for label_name, (X_train, X_test) in variants.items():
    overlap_index = X_train.index.intersection(X_test.index)

    if len(overlap_index) > 0:
        print(f"⚠️ {label_name}: Overlapping indexes found! Count = {len(overlap_index)}")
    else:
        print(f"✅ {label_name}: No overlapping indexes found.")


# In[30]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Define label variants and their datasets
variants = {
    "Old": (X_train_old, y_train_old, X_test_old, y_test_old),
    "All": (X_train_all, y_train_all, X_test_all, y_test_all),
    "Bullish": (X_train_bullish, y_train_bullish, X_test_bullish, y_test_bullish),
}

# Base models (untrained)
model_classes = {
    'RandomForest': RandomForestClassifier,
    'LightGBM': LGBMClassifier,
    'XGBoost': XGBClassifier,
}

print("📊 Cross-Validation Results (F1-score):\n")

for variant_name, (X_train, y_train, X_test, y_test) in variants.items():
    print(f"🔹 {variant_name.upper()} Label:\n")
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    for model_name, model_cls in model_classes.items():
        model = model_cls(random_state=42)
        scores = cross_val_score(model, X_all, y_all, cv=5, scoring='f1')
        print(f"  {model_name}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    print("-" * 50)

    if variant_name in ["ALL", "BULLISH"]:  # Only show these two
        print(f"\n🔹 {variant_name} Label Distribution:")
        print("Training set:")
        print(pd.Series(y_train).value_counts(normalize=True))
        print("\nTest set:")
        print(pd.Series(y_test).value_counts(normalize=True))


# In[ ]:





# # Handling Class Imbalance

# In[32]:


get_ipython().system('jupyter nbconvert --to script notebooks/improving_classifiers.ipynb')


# In[ ]:




