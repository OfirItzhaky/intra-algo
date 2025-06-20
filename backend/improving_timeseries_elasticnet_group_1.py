#!/usr/bin/env python
# coding: utf-8

"""
Sliding Window ElasticNet Evaluation (Group 1 Only)
- 10 randomized windows, 500 bars each (400 train, 100 test)
- Only Group 1 features: ['FastAvg', 'Close_vs_EMA_10', 'High_15Min', 'MACD', 'High_vs_EMA_5_High', 'ATR']
- No ARIMA, no SHAP, no classifier, no post-filtering, no feature group comparison, no refinements
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import DataLoader
from feature_generator import FeatureGenerator
from label_generator import LabelGenerator
from regression_model_trainer import RegressionModelTrainer
import random
from datetime import datetime
import matplotlib.pyplot as plt

# --- Config ---
WINDOW_SIZE = 500
TRAIN_SIZE = 400
TEST_SIZE = 100
NUM_SAMPLES = 50
GROUP1_FEATURES = ['FastAvg', 'Close_vs_EMA_10', 'High_15Min', 'MACD', 'High_vs_EMA_5_High', 'ATR']
CSV_PATH = "data/training/mes_2_new.csv"
MAX_PRED_DISTANCE = 3.0  # Filter predictions above this distance from Close

# --- Load and prepare data ---
loader = DataLoader()
df_raw = loader.load_from_csv(CSV_PATH)
if df_raw.empty:
    print(f"❌ Error: Could not load data from {CSV_PATH}")
    exit(1)

df_raw = df_raw.drop_duplicates(subset=["Date", "Time"])
df_raw['ParsedTime'] = pd.to_datetime(df_raw['Time'], errors='coerce').dt.time
valid_df = df_raw.dropna(subset=['ParsedTime'])
valid_df["Time"] = valid_df["Time"].astype(str)

feature_generator = FeatureGenerator()
label_generator = LabelGenerator()

# --- Select valid window start indices ---
valid_indices = []
while len(valid_indices) < NUM_SAMPLES:
    idx = random.randint(WINDOW_SIZE, len(valid_df) - 1)
    bar_time = valid_df.iloc[idx]['ParsedTime']
    if datetime.strptime("10:00", "%H:%M").time() <= bar_time <= datetime.strptime("23:30", "%H:%M").time():
        start_idx = idx - WINDOW_SIZE
        if start_idx >= 0:
            valid_indices.append(start_idx)

window_results = []
window_testhour_mse = []  # For plotting: (test_start_hour, mse, window_number)
window_mse_pred_dist = []  # For new analysis: (mse, avg_pred_dist, window_number)

for i, start in enumerate(valid_indices):
    print(f"\n🧪 Window {i+1} | Index range: {start}-{start+WINDOW_SIZE}")
    window_df = valid_df.iloc[start:start + WINDOW_SIZE].copy()

    # Generate features and label
    window_df = feature_generator.create_all_features(window_df)
    window_df = label_generator.elasticnet_label_next_high(window_df)
    window_df = window_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Use only Group 1 features + Next_High
    available = [f for f in GROUP1_FEATURES if f in window_df.columns]
    if len(available) < 2:
        print(f"   ⚠️ Not enough Group 1 features available in window. Skipping.")
        continue
    group_data = window_df[available + ['Next_High', 'Close']].copy()
    group_data = group_data.replace([np.inf, -np.inf], np.nan).dropna()

    X = group_data[available]
    y = group_data['Next_High']
    x_train = X.iloc[:TRAIN_SIZE]
    y_train = y.iloc[:TRAIN_SIZE]
    x_test = X.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    y_test = y.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    reg = RegressionModelTrainer(
        include_prices=False,
        apply_filter=False
    )
    reg.x_train = x_train
    reg.y_train = y_train
    reg.x_test = x_test
    reg.y_test = y_test
    reg.train_model()
    reg.make_predictions()

    mse = mean_squared_error(y_test, reg.predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, reg.predictions)

    # Extract start and end Date+Time
    start_dt = f"{window_df.iloc[0]['Date']} {window_df.iloc[0]['Time']}"
    end_dt = f"{window_df.iloc[-1]['Date']} {window_df.iloc[-1]['Time']}"

    # Extract test start hour for plotting
    try:
        test_time = window_df.iloc[TRAIN_SIZE]['Time']
        test_start_hour = pd.to_datetime(str(test_time)).hour
    except Exception:
        test_start_hour = None

    window_results.append([
        f"Window {i+1}", start_dt, end_dt, mse, rmse, r2
    ])
    window_testhour_mse.append((test_start_hour, mse, f"Window {i+1}"))

    # --- New: Calculate avg prediction distance from Close over test set ---
    try:
        close_test = group_data['Close'].iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE].values
        avg_pred_dist = np.mean(np.abs(reg.predictions - close_test))
    except Exception:
        avg_pred_dist = np.nan
    window_mse_pred_dist.append((mse, avg_pred_dist, f"Window {i+1}"))

# --- Print summary table ---
print("\n\n📊 SLIDING WINDOW RESULTS (Group 1 Only)")
print("=" * 90)
print(f"{'Window':<10} | {'Start':<16} | {'End':<16} | {'MSE':<10} | {'RMSE':<10} | {'R²':<10}")
print("-" * 90)
for row in window_results:
    print(f"{row[0]:<10} | {row[1]:<16} | {row[2]:<16} | {row[3]:<10.4f} | {row[4]:<10.4f} | {row[5]:<10.4f}")
print("=" * 90)

# --- Plot: MSE vs. Test Start Hour ---
hours = [h for h, mse, w in window_testhour_mse if h is not None]
mses = [mse for h, mse, w in window_testhour_mse if h is not None]
labels = [w for h, mse, w in window_testhour_mse if h is not None]

plt.figure(figsize=(10, 6))
plt.scatter(hours, mses, color='royalblue')

# Annotate window numbers
for x, y, label in zip(hours, mses, labels):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.title("MSE vs. Test Start Hour")
plt.xlabel("Test Start Hour (0–23)")
plt.ylabel("MSE")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Plot: MSE vs. Prediction Distance from Close ---
mse_vals = [m for m, d, w in window_mse_pred_dist if not np.isnan(d)]
dist_vals = [d for m, d, w in window_mse_pred_dist if not np.isnan(d)]
labels = [w for m, d, w in window_mse_pred_dist if not np.isnan(d)]

plt.figure(figsize=(10, 6))
plt.scatter(dist_vals, mse_vals, color='darkorange')

# Annotate window numbers
for x, y, label in zip(dist_vals, mse_vals, labels):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.title("MSE vs. Prediction Distance from Close")
plt.xlabel("Avg Distance from Close ($)")
plt.ylabel("MSE")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Filtered Results Table by Max Prediction Distance ---
filtered_results = []
for i, (row, (_, avg_pred_dist, _)) in enumerate(zip(window_results, window_mse_pred_dist)):
    if not np.isnan(avg_pred_dist) and avg_pred_dist <= MAX_PRED_DISTANCE:
        filtered_results.append(row + [avg_pred_dist])

print("\n\n📊 SLIDING WINDOW RESULTS (Group 1 Only, Filtered by Max Distance from Close)")
print("=" * 110)
print(f"{'Window':<10} | {'Start':<16} | {'End':<16} | {'MSE':<10} | {'RMSE':<10} | {'R²':<10} | {'Avg_Dist':<10}")
print("-" * 110)
for row in filtered_results:
    print(f"{row[0]:<10} | {row[1]:<16} | {row[2]:<16} | {row[3]:<10.4f} | {row[4]:<10.4f} | {row[5]:<10.4f} | {row[6]:<10.4f}")
print("=" * 110)

# --- Directional Metrics Analysis ---
directional_results = []
for i, start in enumerate(valid_indices):
    # Re-extract window_df for this window
    window_df = valid_df.iloc[start:start + WINDOW_SIZE].copy()
    window_df = feature_generator.create_all_features(window_df)
    window_df = label_generator.elasticnet_label_next_high(window_df)
    window_df = window_df.replace([np.inf, -np.inf], np.nan).dropna()
    available = [f for f in GROUP1_FEATURES if f in window_df.columns]
    if len(available) < 2:
        continue
    group_data = window_df[available + ['Next_High', 'Close']].copy()
    group_data = group_data.replace([np.inf, -np.inf], np.nan).dropna()
    X = group_data[available]
    y = group_data['Next_High']
    x_train = X.iloc[:TRAIN_SIZE]
    y_train = y.iloc[:TRAIN_SIZE]
    x_test = X.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    y_test = y.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    # Get closes for test set and benchmark close
    close_benchmark = group_data['Close'].iloc[TRAIN_SIZE - 1]
    close_test = group_data['Close'].iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE].values
    # Re-run model for this window
    reg = RegressionModelTrainer(
        include_prices=False,
        apply_filter=False
    )
    reg.x_train = x_train
    reg.y_train = y_train
    reg.x_test = x_test
    reg.y_test = y_test
    reg.train_model()
    reg.make_predictions()
    preds = reg.predictions
    actuals = y_test.values
    # MASE: mean absolute error
    mase = np.mean(np.abs(preds - actuals))
    # PSME: penalized scaled miss error (only penalize when prediction is in wrong direction)
    # Direction: (actual - close_benchmark), (pred - close_benchmark)
    actual_dir = np.sign(actuals - close_benchmark)
    pred_dir = np.sign(preds - close_benchmark)
    wrong_dir_mask = actual_dir != pred_dir
    psme = np.mean(np.abs(preds[wrong_dir_mask] - actuals[wrong_dir_mask])) if np.any(wrong_dir_mask) else 0.0
    # DMR: directional miss rate
    dmr = 100.0 * np.mean(wrong_dir_mask)
    # Start/End for table
    start_dt = f"{window_df.iloc[0]['Date']} {window_df.iloc[0]['Time']}"
    end_dt = f"{window_df.iloc[-1]['Date']} {window_df.iloc[-1]['Time']}"
    directional_results.append([
        f"Window {i+1}", start_dt, end_dt, mase, psme, dmr
    ])

# Print table
print("\n\n📊 SLIDING WINDOW RESULTS (Directional Metrics)")
print("=" * 68)
print(f"{'Window':<10} | {'Start':<16} | {'End':<16} | {'MASE':<7} | {'PSME':<7} | {'DMR':<6}")
print("-" * 68)
for row in directional_results:
    print(f"{row[0]:<10} | {row[1]:<16} | {row[2]:<16} | {row[3]:<7.4f} | {row[4]:<7.4f} | {row[5]:<5.1f}%")
print("=" * 68)

# Line chart comparing all three metrics per window
window_labels = [row[0] for row in directional_results]
mase_vals = [row[3] for row in directional_results]
psme_vals = [row[4] for row in directional_results]
dmr_vals = [row[5] for row in directional_results]

plt.figure(figsize=(12, 6))
plt.plot(window_labels, mase_vals, marker='o', label='MASE')
plt.plot(window_labels, psme_vals, marker='o', label='PSME')
plt.plot(window_labels, dmr_vals, marker='o', label='DMR (%)')
plt.title('Directional Metrics per Window')
plt.xlabel('Window')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
