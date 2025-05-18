# classifier_cnn_experiments.py
# ✅ Purpose: Structured testing of 1D CNN for predicting next-bar outcome using selective features

# === 1. Imports and Configuration ===
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score
from cnn_feature_insight import CNNFeatureInsightHelper
from data_loader import DataLoader
from label_generator import LabelGenerator
from functools import partial

from feature_generator import FeatureGenerator

# === 2. Parameter Grid (Looped Testing Options) ===
SEQUENCE_OPTIONS = [10, 15]
LEARNING_RATES = [0.001, 0.0005]
THRESHOLDS = [0.5]
BATCH_SIZE = 64
LABEL_COLUMN = "long_good_bar_label"
DATA_PATH = "data/training/mes_new_2_15_mins_20000_bars.txt"

# === 3. GPU Configuration ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.list_physical_devices('GPU')
use_gpu = False
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        use_gpu = True
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")

# === 4. Functions ===
def selective_feature_pipeline(df_raw):
    df = df_raw.copy()
    print("\n✅ Using raw price and volume only (no engineered features)")
    return df

def prepare_cnn_dataset(df, label_col, sequence_length=15):
    df = df.dropna().copy()
    y = df[label_col].values.astype(int)
    X = df.select_dtypes(include='number').drop(columns=[label_col]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_cnn_model(input_shape, learning_rate=0.001):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def get_removal_impact():
    base_output = np.squeeze(model(X_test).numpy())
    base_preds = (base_output > THRESHOLD).astype(int)
    base_score = precision_score(y_test, base_preds, zero_division=0)
    impact = []
    for i in range(X_test.shape[2]):
        X_mod = X_test.copy()
        X_mod[:, :, i] = 0
        mod_output = np.squeeze(model(X_mod).numpy())
        mod_preds = (mod_output > THRESHOLD).astype(int)
        mod_score = precision_score(y_test, mod_preds, zero_division=0)
        impact.append(mod_score - base_score)
    return impact

# === 5. Main Execution ===
if __name__ == "__main__":
    feature_rows = []
    temporal_rows = []

    for SEQUENCE_LENGTH in SEQUENCE_OPTIONS:
        for LEARNING_RATE in LEARNING_RATES:
            for THRESHOLD in THRESHOLDS:
                print(f"\n===== RUN: seq={SEQUENCE_LENGTH}, lr={LEARNING_RATE}, threshold={THRESHOLD} =====")
                print("\n===== Loading Data =====")
                loader = DataLoader()
                df_raw = loader.load_from_csv(DATA_PATH)
                df_raw = df_raw.drop_duplicates(subset=["Date", "Time"])

                print("\n===== Feature Engineering =====")
                df_with_features = selective_feature_pipeline(df_raw)
                label_gen = LabelGenerator()
                df_labeled = label_gen.green_red_bar_label_goal_d(df_with_features)

                print("\n===== Preparing Dataset =====")
                X, y = prepare_cnn_dataset(df_labeled, label_col=LABEL_COLUMN, sequence_length=SEQUENCE_LENGTH)
                split_idx = int(0.8 * len(X))
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_test, y_test = X[split_idx:], y[split_idx:]

                log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                val_idx = int(0.9 * len(X_train))
                X_train_final, X_val = X_train[:val_idx], X_train[val_idx:]
                y_train_final, y_val = y_train[:val_idx], y_train[val_idx:]

                device = '/GPU:0' if use_gpu else '/CPU:0'
                with tf.device(device):
                    model = build_cnn_model(input_shape=X_train.shape[1:], learning_rate=LEARNING_RATE)
                    print(f"\n===== Training CNN on {device} =====")
                    training_start_time = time.time()
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                    history = model.fit(
                        train_dataset,
                        epochs=25,
                        validation_data=val_dataset,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                            tensorboard_callback
                        ],
                        verbose=1
                    )
                    training_time = time.time() - training_start_time

                print("\n===== Evaluating Model =====")
                X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                test_dataset = tf.data.Dataset.from_tensor_slices(X_test_tensor).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                with tf.device(device):
                    eval_start_time = time.time()
                    y_pred_prob = model.predict(test_dataset)
                    eval_time = time.time() - eval_start_time

                y_pred = (y_pred_prob > THRESHOLD).astype(int).flatten()
                print("\n===== Classification Report =====")
                print(classification_report(y_test, y_pred))

                print("\n===== Running Feature Insight Helper =====")
                cnn_insight = CNNFeatureInsightHelper(
                    model,
                    torch.tensor(X_test).float(),
                    torch.tensor(y_test).float(),
                    feature_names=list(df_labeled.select_dtypes(include='number').drop(columns=[LABEL_COLUMN]).columns),
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                cnn_insight.summarize_feature_gradient_impact()
                cnn_insight.test_feature_removal_impact(
                    torch.tensor(X_test).float(),
                    torch.tensor(y_test).float(),
                    metric_fn=partial(precision_score, zero_division=0),
                    threshold=THRESHOLD
                )

                print("\n📊 To view training metrics, run:")
                print(f"tensorboard --logdir={log_dir}")
                print("\n===== Summary =====")
                print(f"Device used: {device}")
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Evaluation time: {eval_time:.2f} seconds")

                # === Collect insight data ===
                saliency = cnn_insight.compute_saliency()
                avg_importance = saliency.mean(axis=(0, 1))
                temporal_importance = saliency.mean(axis=(0, 2))
                fp_mask = ((torch.tensor(y_pred) == 1) & (torch.tensor(y_test) == 0)).numpy()
                removal_impact = get_removal_impact()

                run_id = f"seq{SEQUENCE_LENGTH}_lr{LEARNING_RATE}_th{THRESHOLD}"
                for i, name in enumerate(cnn_insight.feature_names):
                    feature_rows.append({
                        "Feature": name,
                        "Impact": avg_importance[i],
                        "Precision Change": removal_impact[i],
                        "Unused": int(avg_importance[i] < 0.01),
                        "FP_Impact": float(saliency[fp_mask].mean(axis=(0, 1))[i]) if fp_mask.any() else 0.0,
                        "Run_ID": run_id
                    })

                temporal_rows.append({
                    "Run_ID": run_id,
                    **{f"Step_{i}": v for i, v in enumerate(temporal_importance.tolist())}
                })

    # === Save to Excel with 2 sheets ===
    feature_df = pd.DataFrame(feature_rows)
    temporal_df = pd.DataFrame(temporal_rows)

    with pd.ExcelWriter("cnn_feature_insight_all.xlsx", engine="openpyxl") as writer:
        feature_df.to_excel(writer, sheet_name="Feature Insights", index=False)
        temporal_df.to_excel(writer, sheet_name="Temporal Importance", index=False)
    print("🧠 TF using GPU:", tf.config.list_logical_devices('GPU'))
