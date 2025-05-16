# classifier_cnn_experiments.py
# ✅ Purpose: Structured testing of 1D CNN for predicting next-bar outcome using selective features

# === 1. Imports ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import time
import os
import sys
print("✅ TensorFlow Keras imports successful!")
print(f"TensorFlow version: {tf.__version__}")

# === GPU Configuration ===
print("\n===== GPU Configuration =====")
# Set TensorFlow log level to show info but not warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s)!")
    print(f"GPU devices: {gpus}")
    
    # Configure GPU memory growth to prevent OOM errors
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
        
        # Print GPU details
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
        use_gpu = True
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
        use_gpu = False
else:
    print("⚠️ No GPU found by TensorFlow.")
    print("If you have an NVIDIA GPU, please:\n"
          "1. Run setup_gpu.bat as Administrator\n"
          "2. Download and install cuDNN (see setup_gpu.bat for instructions)\n"
          "3. Restart your computer\n"
          "4. Try running this script again")
    use_gpu = False

# Run a small test to verify GPU performance
if use_gpu:
    print("\n===== GPU Performance Test =====")
    # Create test matrices for matrix multiplication
    test_size = 2000
    print(f"Testing with matrix multiplication {test_size}x{test_size}...")
    
    # Test matrices
    a = tf.random.normal([test_size, test_size])
    b = tf.random.normal([test_size, test_size])
    
    # Warm up GPU
    with tf.device('/GPU:0'):
        warmup = tf.matmul(a[:100, :100], b[:100, :100])
    
    # Benchmark GPU
    start_time = time.time()
    with tf.device('/GPU:0'):
        c_gpu = tf.matmul(a, b)
        # Force execution
        _ = c_gpu.numpy()
    gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.4f} seconds")
    
    # Benchmark CPU
    start_time = time.time()
    with tf.device('/CPU:0'):
        c_cpu = tf.matmul(a, b)
        # Force execution
        _ = c_cpu.numpy()
    cpu_time = time.time() - start_time
    print(f"CPU computation time: {cpu_time:.4f} seconds")
    
    # Compare performance
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"✅ GPU is {speedup:.2f}x faster than CPU!")
    else:
        print(f"⚠️ GPU is not faster than CPU. GPU may not be correctly configured.")
        use_gpu = False  # Fall back to CPU if GPU isn't faster

print(f"\n===== Running with {'GPU' if use_gpu else 'CPU'} =====")

from data_loader import DataLoader
from label_generator import LabelGenerator
from data_processor import DataProcessor
from feature_generator import FeatureGenerator

# === 2. Selectively Add Features ===
def selective_feature_pipeline(df_raw):
    fg = FeatureGenerator()

    # Start with core set (simplified)
    df = fg.add_vwap_features(df_raw)
    df = fg.add_atr_price_features(df)
    df = fg.add_cci_average(df)
    df = fg.add_volatility_momentum_volume_features(df)
    df = fg.add_macd_indicators(df)
    df = fg.add_constant_columns(df)

    print("\n✅ Selected Features Generated.")
    return df

# === 3. Prepare CNN Dataset ===
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

# === 4. Build 1D CNN Model ===
def build_cnn_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Add another conv layer for more capacity
    model.add(tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Use Adam with a learning rate schedule for better training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

# === 5. Load and Run Pipeline ===
print("\n===== Loading Data =====")
loader = DataLoader()
df_raw = loader.load_from_csv("data/training/mes_new_2_15_mins_20000_bars.txt")
df_raw = df_raw.drop_duplicates(subset=["Date", "Time"])

# === 6. Generate Features and Labels ===
print("\n===== Feature Engineering =====")
df_with_features = selective_feature_pipeline(df_raw)
label_gen = LabelGenerator()
df_labeled = label_gen.green_red_bar_label_goal_d(df_with_features)

# === 7. CNN Dataset and Split ===
print("\n===== Preparing Dataset =====")
X, y = prepare_cnn_dataset(df_labeled, label_col="long_good_bar_label", sequence_length=15)
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# === 8. Train CNN ===
# Create TensorBoard callback for monitoring training
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Creating validation dataset
val_idx = int(0.9 * len(X_train))  # 10% validation split
X_train_final, X_val = X_train[:val_idx], X_train[val_idx:]
y_train_final, y_val = y_train[:val_idx], y_train[val_idx:]

# Set the compute device
device = '/GPU:0' if use_gpu else '/CPU:0'
with tf.device(device):
    # Build model
    model = build_cnn_model(input_shape=X_train.shape[1:])
    print(f"\n===== Training CNN on {device} =====")
    
    # Start time for performance comparison
    training_start_time = time.time()
    
    # Convert data to tensors for faster processing
    X_train_tensor = tf.convert_to_tensor(X_train_final, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train_final, dtype=tf.float32)
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)
    
    # Create dataset with prefetching
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor))
    train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_tensor, y_val_tensor))
    val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Training with callbacks
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
    
    # Calculate training time
    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

# === 9. Evaluate ===
print("\n===== Evaluating Model =====")
# Convert test data to tensor and create dataset
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test_tensor).batch(64).prefetch(tf.data.AUTOTUNE)

# Evaluate
with tf.device(device):
    eval_start_time = time.time()
    y_pred_prob = model.predict(test_dataset)
    eval_time = time.time() - eval_start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")

y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Flatten to match y_test shape

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred))

# Print how to view TensorBoard
print("\n📊 To view training metrics, run:")
print(f"tensorboard --logdir={log_dir}")

# Final GPU vs CPU summary
if use_gpu:
    print("\n===== GPU Acceleration Summary =====")
    print(f"GPU was used successfully for training and evaluation!")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
else:
    print("\n===== CPU Processing Summary =====")
    print(f"Training was performed on CPU only")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    print("\nTo enable GPU acceleration, follow the steps in setup_gpu.bat")
