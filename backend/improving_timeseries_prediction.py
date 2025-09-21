#!/usr/bin/env python
# coding: utf-8

# ===== ElasticNet Regression Benchmark for Time Series Prediction =====
# Goal: Set up baseline ElasticNet benchmark for predicting next high in intraday trading bars
# Based on: improving_classifiers.py structure

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
from feature_generator import FeatureGenerator
from label_generator import LabelGenerator
from regression_model_trainer import RegressionModelTrainer
import warnings
from logging_setup import get_logger

log = get_logger(__name__)
# Import ARIMA models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    arima_available = True
except ImportError:
    arima_available = False
    log.info("⚠️ Warning: statsmodels not available. ARIMA models will be skipped.")

log.info("🚀 Starting ElasticNet Regression Benchmark for Time Series Prediction")

# ===== Step 1: Load CSV using DataLoader =====
log.info("\n📁 Step 1: Loading CSV data...")
loader = DataLoader()
training_df_raw = loader.load_from_csv("data/training/mes_2_new.csv")

if training_df_raw.empty:
    log.info("❌ Error: Failed to load training data!")
    exit(1)

log.info(f"✅ Data loaded successfully: {training_df_raw.shape}")
log.info(f"   Columns: {list(training_df_raw.columns)}")

# ===== Step 2: Drop duplicate rows using Date and Time =====
log.info("\n🔍 Step 2: Removing duplicate timestamps...")
initial_rows = len(training_df_raw)
training_df_raw = training_df_raw.drop_duplicates(subset=["Date", "Time"])
final_rows = len(training_df_raw)
duplicates_removed = initial_rows - final_rows

log.info(f"✅ Duplicates removed: {duplicates_removed}")
log.info(f"   Final dataset: {final_rows} rows")

# ===== Step 3: Generate all features using FeatureGenerator =====
log.info("\n⚙️ Step 3: Generating technical features...")
feature_generator = FeatureGenerator()
training_df_features = feature_generator.create_all_features(training_df_raw.copy())

log.info(f"✅ Features generated: {training_df_features.shape[1]} total columns")
new_features = training_df_features.shape[1] - training_df_raw.shape[1]
log.info(f"   New features added: {new_features}")

# ===== Step 4: Generate regression labels using LabelGenerator =====
log.info("\n🏷️ Step 4: Generating regression labels...")
label_generator = LabelGenerator()
training_df_labeled = label_generator.elasticnet_label_next_high(training_df_features.copy())
training_df_labeled = training_df_labeled.tail(500).copy()

log.info(f"✅ Labels generated: 'Next_High' column added")
log.info(f"   Labeled dataset: {training_df_labeled.shape}")

# ===== Step 5: Train ElasticNet regression model =====
log.info("\n🤖 Step 5: Training ElasticNet regression model...")


trainer = RegressionModelTrainer(
    include_prices=False,         # Set to False as specified
    apply_filter=True,            # Apply filtering as specified
    filter_threshold=4.0          # Use threshold of 4.0 as specified
)

# Prepare data for training
trainer.prepare_data(training_df_labeled.copy())

# Train the model
trainer.train_model()

# Make predictions
trainer.make_predictions()

log.info(f"✅ Model trained successfully!")
log.info(f"   Training samples: {len(trainer.x_train)}")
log.info(f"   Test samples: {len(trainer.x_test)}")
log.info(f"   Features used: {trainer.x_train.shape[1]}")

# ===== Step 6: Evaluate model using MSE and R² =====
log.info("\n📊 Step 6: Evaluating model performance...")

# Calculate metrics
mse = mean_squared_error(trainer.y_test, trainer.predictions)
r2 = r2_score(trainer.y_test, trainer.predictions)

# Print evaluation results
log.info("=" * 50)
log.info("📈 ELASTICNET REGRESSION BENCHMARK RESULTS")
log.info("=" * 50)
log.info(f"MSE (Mean Squared Error): {mse:.3f}")
log.info(f"R² (R-squared Score): {r2:.3f}")
log.info(f"Number of Test Samples: {len(trainer.y_test)}")
log.info(f"Number of Features: {trainer.x_train.shape[1]}")
log.info("=" * 50)

# ===== Extract Top ElasticNet Features =====
log.info("\n🧠 Extracting Top ElasticNet Features by Importance...")

# Get the trained ElasticNet model coefficients
elasticnet_coefficients = trainer.model.coef_
feature_names = trainer.x_train.columns

# Pair coefficients with feature names and sort by absolute value
feature_importance = list(zip(feature_names, elasticnet_coefficients))
feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

# Extract top 10 features
top_10_features = feature_importance_sorted[:10]

log.info("\n📌 Top ElasticNet Features by Importance:")
log.info("-" * 50)
for i, (feature_name, coef) in enumerate(top_10_features, 1):
    log.info(f"{i:2d}. {feature_name:<25}: Coef = {coef:>8.4f}")
log.info("-" * 50)

# Store top features for future use (ARIMAX/SARIMAX testing)
top_elasticnet_features = [feature_name for feature_name, _ in top_10_features]

log.info(f"✅ Top {len(top_elasticnet_features)} features extracted and stored for future ARIMAX/SARIMAX testing")

# ===== Step 7: Apply post-prediction error filter =====
log.info("\n🔍 Step 7: Applying post-prediction error filter...")
log.info(f"   Filter threshold: {trainer.filter_threshold} points")

# Compute absolute prediction errors
prediction_errors = abs(trainer.y_test - trainer.predictions)

# Apply filter: keep only predictions with error <= 4.0
filter_mask = prediction_errors <= trainer.filter_threshold
y_test_filtered = trainer.y_test[filter_mask]
predictions_filtered = trainer.predictions[filter_mask]

# Calculate how many samples were filtered out
total_samples = len(trainer.y_test)
filtered_samples = len(y_test_filtered)
removed_samples = total_samples - filtered_samples

log.info(f"✅ Error filtering applied:")
log.info(f"   Original samples: {total_samples}")
log.info(f"   Filtered samples: {filtered_samples}")
log.info(f"   Removed samples: {removed_samples} ({removed_samples/total_samples*100:.1f}%)")

# Recalculate metrics on filtered data
mse_filtered = mean_squared_error(y_test_filtered, predictions_filtered)
r2_filtered = r2_score(y_test_filtered, predictions_filtered)

# Print filtered evaluation results
log.info("\n" + "=" * 50)
log.info("📉 ELASTICNET FILTERED EVALUATION RESULTS")
log.info("=" * 50)
log.info(f"Filtered Sample Count: {filtered_samples}")
log.info(f"Filtered MSE: {mse_filtered:.3f}")
log.info(f"Filtered R²: {r2_filtered:.3f}")
log.info("=" * 50)

# ===== Step 8: Add ARIMA and ARIMAX benchmark models =====
if arima_available:
    log.info("\n📈 Step 8: Training ARIMA and ARIMAX benchmark models...")
    
    # Get the same train/test split indices as ElasticNet
    train_size = len(trainer.x_train)
    test_size = len(trainer.x_test)
    total_size = train_size + test_size
    
    log.info(f"   Using same split: {train_size} train, {test_size} test samples")
    
    # ===== MODEL 1: ARIMA (Univariate) =====
    log.info("\n🔵 Training ARIMA (Univariate) model...")
    try:
        # Use High column from the labeled dataset, matching the split
        high_series = training_df_labeled["High"].iloc[:total_size].copy()
        
        # Split using the same indices as ElasticNet
        high_train = high_series.iloc[:train_size]
        high_test = high_series.iloc[train_size:]
        
        # Fit ARIMA model (5,1,0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMA(high_train, order=(5,1,0))
            arima_fitted = arima_model.fit()
        
        # Make predictions on test set
        arima_predictions = arima_fitted.forecast(steps=len(high_test))
        
        # Calculate metrics
        arima_mse = mean_squared_error(trainer.y_test, arima_predictions)
        arima_r2 = r2_score(trainer.y_test, arima_predictions)
        
        log.info(f"✅ ARIMA model trained successfully")
        log.info(f"   MSE: {arima_mse:.3f}, R²: {arima_r2:.3f}")
        
    except Exception as e:
        log.info(f"❌ ARIMA training failed: {e}")
        arima_mse, arima_r2 = float('nan'), float('nan')
    
    # ===== MODEL 2: ARIMAX (Multivariate) =====
    log.info("\n🟡 Training ARIMAX (Multivariate) model...")
    try:
        # Select top 5-10 features for ARIMAX
        feature_cols = trainer.x_train.columns
        
        # Select key technical indicators (if available)
        priority_features = ['MACD', 'MACDAvg', 'FastEMA', 'SlowEMA', 'RSI', 'ADX', 'CCI', 'Williams_R', 'ATR', 'VWAP']
        selected_features = []
        
        for feat in priority_features:
            matching_cols = [col for col in feature_cols if feat in col]
            if matching_cols:
                selected_features.extend(matching_cols[:1])  # Take first match
        
        # If we don't have enough, add more features
        if len(selected_features) < 5:
            remaining_features = [col for col in feature_cols if col not in selected_features]
            selected_features.extend(remaining_features[:10-len(selected_features)])
        
        # Limit to 8 features to avoid overfitting
        selected_features = selected_features[:8]
        
        log.info(f"   Selected features: {selected_features}")
        
        # Get the exogenous variables from the same dataset split
        X_train_arimax = trainer.x_train[selected_features].copy()
        X_test_arimax = trainer.x_test[selected_features].copy()
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arimax)
        X_test_scaled = scaler.transform(X_test_arimax)
        
        # Use High series as endogenous variable
        high_train_arimax = training_df_labeled["High"].iloc[:train_size].copy()
        
        # Fit ARIMAX model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arimax_model = ARIMA(high_train_arimax, exog=X_train_scaled, order=(5,1,0))
            arimax_fitted = arimax_model.fit()
        
        # Make predictions with exogenous variables
        arimax_predictions = arimax_fitted.forecast(steps=len(X_test_scaled), exog=X_test_scaled)
        
        # Calculate metrics
        arimax_mse = mean_squared_error(trainer.y_test, arimax_predictions)
        arimax_r2 = r2_score(trainer.y_test, arimax_predictions)
        
        log.info(f"✅ ARIMAX model trained successfully")
        log.info(f"   MSE: {arimax_mse:.3f}, R²: {arimax_r2:.3f}")
        
    except Exception as e:
        log.info(f"❌ ARIMAX training failed: {e}")
        arimax_mse, arimax_r2 = float('nan'), float('nan')
    
    # ===== Apply Error Filtering to ARIMA/ARIMAX (Same as ElasticNet) =====
    log.info("\n🔍 Applying error filtering to ARIMA/ARIMAX predictions...")
    
    # Reuse the existing ElasticNet filter mask from Step 7
    filter_mask = abs(trainer.y_test - trainer.predictions) <= trainer.filter_threshold
    y_test_filtered_step8 = trainer.y_test[filter_mask]
    
    # Apply filter to ARIMA predictions
    if not np.isnan(arima_mse):
        arima_filtered_preds = pd.Series(arima_predictions).reset_index(drop=True)[filter_mask.reset_index(drop=True)]
        filtered_arima_mse = mean_squared_error(y_test_filtered_step8, arima_filtered_preds)
        filtered_arima_r2 = r2_score(y_test_filtered_step8, arima_filtered_preds)
        log.info(f"✅ ARIMA filtered: MSE={filtered_arima_mse:.3f}, R²={filtered_arima_r2:.3f}")
    else:
        filtered_arima_mse, filtered_arima_r2 = float('nan'), float('nan')

    # Apply filter to ARIMAX predictions
    if not np.isnan(arimax_mse):
        arimax_filtered_preds = pd.Series(arimax_predictions).reset_index(drop=True)[filter_mask.reset_index(drop=True)]
        filtered_arimax_mse = mean_squared_error(y_test_filtered_step8, arimax_filtered_preds)
        filtered_arimax_r2 = r2_score(y_test_filtered_step8, arimax_filtered_preds)
        log.info(f"✅ ARIMAX filtered: MSE={filtered_arimax_mse:.3f}, R²={filtered_arimax_r2:.3f}")
    else:
        filtered_arimax_mse, filtered_arimax_r2 = float('nan'), float('nan')

    log.info(f"   Filter applied: {len(y_test_filtered_step8)}/{len(trainer.y_test)} samples retained")
    
    # ===== Model Comparison Summary =====
    log.info("\n" + "=" * 50)
    log.info("📊 MODEL COMPARISON SUMMARY")
    log.info("=" * 50)
    log.info(f"{'Model':<12} {'MSE':<8} {'R²':<8} {'Samples':<8}")
    log.info("-" * 50)
    log.info(f"{'ElasticNet':<12} {mse:<8.3f} {r2:<8.3f} {len(trainer.y_test):<8}")
    log.info(f"{'ARIMA':<12} {arima_mse:<8.3f} {arima_r2:<8.3f} {len(trainer.y_test):<8}")
    log.info(f"{'ARIMAX':<12} {arimax_mse:<8.3f} {arimax_r2:<8.3f} {len(trainer.y_test):<8}")
    log.info("=" * 50)
    
    # ===== Filtered Model Comparison =====
    log.info("\n" + "=" * 50)
    log.info("📉 FILTERED MODEL COMPARISON")
    log.info("=" * 50)
    log.info(f"{'Model':<12} | {'Filtered MSE':<12} | {'Filtered R²':<11} | {'Samples':<8}")
    log.info("-" * 50)
    log.info(f"{'ElasticNet':<12} | {mse_filtered:<12.3f} | {r2_filtered:<11.3f} | {len(y_test_filtered_step8):<8}")
    log.info(f"{'ARIMA':<12} | {filtered_arima_mse:<12.3f} | {filtered_arima_r2:<11.3f} | {len(y_test_filtered_step8):<8}")
    log.info(f"{'ARIMAX':<12} | {filtered_arimax_mse:<12.3f} | {filtered_arimax_r2:<11.3f} | {len(y_test_filtered_step8):<8}")
    log.info("=" * 50)
    
    # Determine best model (unfiltered)
    models_results = [
        ('ElasticNet', mse, r2),
        ('ARIMA', arima_mse, arima_r2),
        ('ARIMAX', arimax_mse, arimax_r2)
    ]
    
    # Determine best model (filtered)
    filtered_models_results = [
        ('ElasticNet', mse_filtered, r2_filtered),
        ('ARIMA', filtered_arima_mse, filtered_arima_r2),
        ('ARIMAX', filtered_arimax_mse, filtered_arimax_r2)
    ]
    
    # Find best model by R² (higher is better) - unfiltered
    valid_models = [(name, mse_val, r2_val) for name, mse_val, r2_val in models_results if not np.isnan(r2_val)]
    
    if valid_models:
        best_model = max(valid_models, key=lambda x: x[2])
        log.info(f"\n🏆 Best unfiltered model: {best_model[0]} (R² = {best_model[2]:.3f})")
    
    # Find best model by R² (higher is better) - filtered
    valid_filtered_models = [(name, mse_val, r2_val) for name, mse_val, r2_val in filtered_models_results if not np.isnan(r2_val)]
    
    if valid_filtered_models:
        best_filtered_model = max(valid_filtered_models, key=lambda x: x[2])
        log.info(f"🏆 Best filtered model: {best_filtered_model[0]} (R² = {best_filtered_model[2]:.3f})")

    # ===== Step 10: Group-Based Model Benchmarking =====
    log.info("\n🧪 Step 10: Group-Based Model Benchmarking...")
    training_df_labeled["High - Low"] = training_df_labeled["High"] - training_df_labeled["Low"]
    training_df_labeled["Close - Open"] = training_df_labeled["Close"] - training_df_labeled["Open"]
    log.info("\n🧪 Step 10: Group-Based Model Benchmarking...")

    # Core bar calculations
    training_df_labeled["High - Low"] = training_df_labeled["High"] - training_df_labeled["Low"]
    training_df_labeled["Close - Open"] = training_df_labeled["Close"] - training_df_labeled["Open"]

    # Volume-based features
    training_df_labeled["Volume_MA_10"] = training_df_labeled["Volume"].rolling(window=10).mean()
    training_df_labeled["Volume_MA_30"] = training_df_labeled["Volume"].rolling(window=30).mean()
    training_df_labeled["Volume_Change"] = training_df_labeled["Volume"].diff()
    training_df_labeled["Volume_Change_Pct"] = training_df_labeled["Volume"].pct_change()
    training_df_labeled["Volume_vs_MA10"] = training_df_labeled["Volume"] / training_df_labeled["Volume_MA_10"]
    training_df_labeled["RelVolume"] = training_df_labeled["Volume"] / training_df_labeled["Volume"].rolling(
        window=20).mean()
    training_df_labeled["Volume_Accum_30m"] = training_df_labeled["Volume"].rolling(window=6).sum()
    training_df_labeled["Volume_Accum_15m"] = training_df_labeled["Volume"].rolling(window=3).sum()

    # Interaction features
    training_df_labeled["PriceChange_vs_VolumeChange"] = training_df_labeled["Close"].diff() / (
                training_df_labeled["Volume"].diff() + 1e-6)
    training_df_labeled["Volatility_vs_Volume"] = (training_df_labeled["High"] - training_df_labeled["Low"]) / (
                training_df_labeled["Volume"] + 1e-6)
    training_df_labeled["Range_vs_Volume"] = training_df_labeled["High - Low"] / (training_df_labeled["Volume"] + 1e-6)
    training_df_labeled["abs(Close - Open) / Volume"] = training_df_labeled["Close - Open"].abs() / (
                training_df_labeled["Volume"] + 1e-6)
    training_df_labeled["High_Extension_vs_Volume"] = (training_df_labeled["High"] - training_df_labeled["Open"]) / (
                training_df_labeled["Volume"] + 1e-6)
    training_df_labeled["Wick_Skewness"] = (training_df_labeled["High"] - training_df_labeled["Close"]) - (
                training_df_labeled["Close"] - training_df_labeled["Low"])
    training_df_labeled["Volume_vs_Range"] = training_df_labeled["Volume"] / (training_df_labeled["High - Low"] + 1e-6)
    training_df_labeled["Range_x_Volume"] = (training_df_labeled["High - Low"]) * training_df_labeled["Volume"]

    # Define 7 feature groups
    feature_groups = {
        "Group 1 - Top ElasticNet": ['FastAvg', 'Close_vs_EMA_10', 'High_15Min', 'MACD', 'High_vs_EMA_5_High', 'ATR'],
        "Group 2 - Momentum/Trend": ['FastEMA', 'SlowEMA', 'ADX', 'AroonUp', 'AroonDn', 'DMI_plus', 'DMI_minus'],
        "Group 3 - High-Specific": ['High_Max_5', 'High_Max_15', 'High_15Min', 'High_vs_EMA_10_High'],
        "Group 4 - Volatility+MACD": ['ATR', 'MACD', 'MACDAvg', 'MACDDiff', 'CCI', 'CCI_Avg', 'Williams_R'],
        "Group 5 - Price vs EMAs": ['Close_vs_EMA_5', 'Close_vs_EMA_10', 'Close_vs_EMA_20', 'Close_vs_EMA_30',
                                    'Close_vs_EMA_50'],
        "Group 6 - Price Relatives": ['High - EMA_10', 'High - VWAP', 'Close_vs_EMA_5', 'High_Max_8', 'Open - Close'],
        "Group 7 - OHLC + Bar Features": ['Open', 'High', 'Low', 'Close', 'High - Low', 'Close - Open'],

        # New Volume-Based Groups
        "Group 8 - Volume Level": ['Volume', 'Volume_MA_10', 'Volume_MA_30'],
        "Group 9 - Volume Momentum": ['Volume_Change', 'Volume_Change_Pct'],
        "Group 10 - Relative Volume": ['Volume_vs_MA10', 'RelVolume'],
        "Group 11 - Price-Vol Pressure": ['PriceChange_vs_VolumeChange', 'Volatility_vs_Volume'],
        "Group 12 - Volume Accumulation": ['Volume_Accum_30m', 'Volume_Accum_15m'],
        "Group 13 - Interaction-1": ['Range_vs_Volume', 'abs(Close - Open) / Volume'],
        "Group 14 - Interaction-2": ['High_Extension_vs_Volume', 'Wick_Skewness'],
        "Group 15 - Interaction-3": ['Volume_vs_Range', 'Range_x_Volume'],

        # Optional hybrid group (top few drivers)
        # "Group 16 - Hybrid Driver Set": ['Volume', 'Volume_vs_MA10', 'RelVolume', 'PriceChange_vs_VolumeChange', 'Range_x_Volume']
    }

    # Initialize results storage
    group_results = []
    
    # Get the same train/test split indices as the original models
    train_size = len(trainer.x_train)
    test_size = len(trainer.x_test)
    total_size = train_size + test_size
    
    log.info(f"   Evaluating {len(feature_groups)} feature groups...")
    log.info(f"   Using same split: {train_size} train, {test_size} test samples")
    
    for group_name, feature_list in feature_groups.items():
        log.info(f"\n🔍 Evaluating {group_name}...")
        
        # Check which features exist in the dataset
        available_features = [f for f in feature_list if f in training_df_labeled.columns]
        missing_features = [f for f in feature_list if f not in training_df_labeled.columns]
        
        if len(available_features) < 2:
            log.info(f"   ⚠️ Skipping {group_name}: insufficient features ({len(available_features)}/5)")
            group_results.append([group_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", len(available_features)])
            continue
        
        if missing_features:
            log.info(f"   ⚠️ Missing features: {missing_features}")
        
        log.info(f"   ✅ Using {len(available_features)} features: {available_features}")
        
        # Create subset DataFrame with available features
        try:
            # Get the subset of features for this group
            group_data = training_df_labeled[available_features + ['Next_High', 'Date', 'Time']].copy()

            # 🚨 Sanitize bad values (inf or NaN)
            group_data = group_data.replace([np.inf, -np.inf], np.nan).dropna()

            # === ElasticNet on this group ===
            group_trainer = RegressionModelTrainer(
                include_prices=False,
                apply_filter=False,  # No filtering for Step 10
                filter_threshold=4.0
            )
            
            group_trainer.prepare_data(group_data)
            group_trainer.train_model()
            group_trainer.make_predictions()
            
            elasticnet_group_mse = mean_squared_error(group_trainer.y_test, group_trainer.predictions)
            elasticnet_group_r2 = r2_score(group_trainer.y_test, group_trainer.predictions)
            
            log.info(f"      ElasticNet: MSE={elasticnet_group_mse:.3f}, R²={elasticnet_group_r2:.3f}")

            # === ARIMAX on this group ===
            try:
                # Normalize features for ARIMAX
                scaler_group = StandardScaler()
                X_train_group_scaled = scaler_group.fit_transform(group_trainer.x_train)
                X_test_group_scaled = scaler_group.transform(group_trainer.x_test)

                # ✅ Fix: Match endogenous variable exactly to exogenous shape
                high_train_group = group_trainer.y_train.reset_index(drop=True)
                high_test_group = group_trainer.y_test.reset_index(drop=True)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arimax_group_model = ARIMA(high_train_group, exog=X_train_group_scaled, order=(3, 1, 0))
                    arimax_group_fitted = arimax_group_model.fit()

                arimax_group_predictions = arimax_group_fitted.forecast(steps=len(X_test_group_scaled),
                                                                        exog=X_test_group_scaled)

                arimax_group_mse = mean_squared_error(high_test_group, arimax_group_predictions)
                arimax_group_r2 = r2_score(high_test_group, arimax_group_predictions)

                log.info(f"      ARIMAX: MSE={arimax_group_mse:.3f}, R²={arimax_group_r2:.3f}")

            except Exception as e:
                log.info(f"      ARIMAX failed: {str(e)[:50]}...")
                arimax_group_mse, arimax_group_r2 = float('nan'), float('nan')

            # === SARIMAX on this group ===
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sarimax_group_model = SARIMAX(high_train_group, exog=X_train_group_scaled,
                                                  order=(3, 1, 0), seasonal_order=(1, 1, 1, 12))
                    sarimax_group_fitted = sarimax_group_model.fit()

                sarimax_group_predictions = sarimax_group_fitted.forecast(steps=len(X_test_group_scaled),
                                                                          exog=X_test_group_scaled)

                sarimax_group_mse = mean_squared_error(high_test_group, sarimax_group_predictions)
                sarimax_group_r2 = r2_score(high_test_group, sarimax_group_predictions)

                log.info(f"      SARIMAX: MSE={sarimax_group_mse:.3f}, R²={sarimax_group_r2:.3f}")

            except Exception as e:
                log.info(f"      SARIMAX failed: {str(e)[:50]}...")
                sarimax_group_mse, sarimax_group_r2 = float('nan'), float('nan')

            # Store results
            group_results.append([
                group_name, 
                elasticnet_group_mse, elasticnet_group_r2,
                arimax_group_mse, arimax_group_r2,
                sarimax_group_mse, sarimax_group_r2,
                len(available_features)
            ])
            
        except Exception as e:
            log.info(f"   ❌ Group evaluation failed: {e}")
            group_results.append([group_name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", 0])
    
    # === Print Group Comparison Summary ===
    log.info("\n" + "=" * 100)
    log.info("🧪 GROUP-BASED MODEL BENCHMARKING RESULTS")
    log.info("=" * 100)
    log.info(f"{'Group':<25} | {'ElasticNet MSE':<13} | {'ElasticNet R²':<12} | {'ARIMAX MSE':<11} | {'ARIMAX R²':<10} | {'SARIMAX MSE':<11} | {'SARIMAX R²':<10} | {'Features':<8}")
    log.info("-" * 100)
    
    for result in group_results:
        group_name, en_mse, en_r2, ar_mse, ar_r2, sar_mse, sar_r2, feat_count = result
        
        # Format values for display
        def format_val(val):
            if isinstance(val, str) or np.isnan(val):
                return "N/A"
            return f"{val:.3f}"
        
        log.info(f"{group_name:<25} | {format_val(en_mse):<13} | {format_val(en_r2):<12} | {format_val(ar_mse):<11} | {format_val(ar_r2):<10} | {format_val(sar_mse):<11} | {format_val(sar_r2):<10} | {feat_count:<8}")
    
    log.info("=" * 100)
    
    # Find best performing group for each model type
    valid_results = [r for r in group_results if isinstance(r[1], float) and not np.isnan(r[1])]
    
    if valid_results:
        best_elasticnet = min(valid_results, key=lambda x: x[1])  # Lowest MSE
        log.info(f"\n🏆 Best ElasticNet group: {best_elasticnet[0]} (MSE = {best_elasticnet[1]:.3f})")
        
        arimax_valid = [r for r in valid_results if isinstance(r[3], float) and not np.isnan(r[3])]
        if arimax_valid:
            best_arimax = min(arimax_valid, key=lambda x: x[3])  # Lowest MSE
            log.info(f"🏆 Best ARIMAX group: {best_arimax[0]} (MSE = {best_arimax[3]:.3f})")
        
        sarimax_valid = [r for r in valid_results if isinstance(r[5], float) and not np.isnan(r[5])]
        if sarimax_valid:
            best_sarimax = min(sarimax_valid, key=lambda x: x[5])  # Lowest MSE
            log.info(f"🏆 Best SARIMAX group: {best_sarimax[0]} (MSE = {best_sarimax[5]:.3f})")
    
else:
    log.info("\n⚠️ Steps 8-10 skipped: statsmodels not available for ARIMA modeling")

# === Optional RMSE Table ===
log.info("\n" + "=" * 100)
log.info("📉 GROUP-BASED RMSE RESULTS")
log.info("=" * 100)
log.info(f"{'Group':<25} | {'ElasticNet RMSE':<16} | {'ARIMAX RMSE':<13} | {'SARIMAX RMSE':<13}")
log.info("-" * 100)

for result in group_results:
    group_name, en_mse, _, ar_mse, _, sar_mse, _, _ = result

    # Compute RMSE from MSE
    def safe_rmse(mse_val):
        return np.sqrt(mse_val) if isinstance(mse_val, float) and not np.isnan(mse_val) else "N/A"

    en_rmse = safe_rmse(en_mse)
    ar_rmse = safe_rmse(ar_mse)
    sar_rmse = safe_rmse(sar_mse)

    def format_rmse(val):
        return f"{val:.3f}" if isinstance(val, float) else "N/A"

    log.info(f"{group_name:<25} | {format_rmse(en_rmse):<16} | {format_rmse(ar_rmse):<13} | {format_rmse(sar_rmse):<13}")

log.info("=" * 100)

log.info("\n✅ Time series prediction benchmark completed successfully!")
log.info("📌 ElasticNet baseline established and compared with ARIMA models.")
log.info("📌 Ready for DeepAR and advanced time series models in future steps.")

# ===== Step 11: Sliding Window Stability Test for Top Groups =====
log.info("\n🪟 Step 11: Sliding Window Evaluation for Top Groups")

import random
from datetime import datetime

# Step 11 config
WINDOW_SIZE = 500
TRAIN_SIZE = 400
TEST_SIZE = 100
NUM_SAMPLES = 10
TOP_GROUPS = {
    "Group 1 - Top ElasticNet": ['FastAvg', 'Close_vs_EMA_10', 'High_15Min', 'MACD', 'High_vs_EMA_5_High', 'ATR'],
    "Group 2 - Momentum/Trend": ['FastEMA', 'SlowEMA', 'ADX', 'AroonUp', 'AroonDn', 'DMI_plus', 'DMI_minus'],
    "Group 3 - High-Specific": ['High_Max_5', 'High_Max_15', 'High_15Min', 'High_vs_EMA_10_High'],
    "Group 4 - Volatility+MACD": ['ATR', 'MACD', 'MACDAvg', 'MACDDiff', 'CCI', 'CCI_Avg', 'Williams_R'],
    "Group 5 - Price vs EMAs": ['Close_vs_EMA_5', 'Close_vs_EMA_10', 'Close_vs_EMA_20', 'Close_vs_EMA_30', 'Close_vs_EMA_50']
}

# Fix Time for filtering and features
training_df_raw['ParsedTime'] = pd.to_datetime(training_df_raw['Time'], errors='coerce').dt.time
valid_df = training_df_raw.dropna(subset=['ParsedTime'])
valid_df["Time"] = valid_df["Time"].astype(str)

# Select valid window start indices
valid_indices = []
while len(valid_indices) < NUM_SAMPLES:
    idx = random.randint(WINDOW_SIZE, len(valid_df) - 1)
    bar_time = valid_df.iloc[idx]['ParsedTime']
    if datetime.strptime("10:00", "%H:%M").time() <= bar_time <= datetime.strptime("23:30", "%H:%M").time():
        start_idx = idx - WINDOW_SIZE
        if start_idx >= 0:
            valid_indices.append(start_idx)

window_results = []

for i, start in enumerate(valid_indices):
    log.info(f"\n🧪 Window {i+1} | Index range: {start}-{start+WINDOW_SIZE}")
    window_df = valid_df.iloc[start:start + WINDOW_SIZE].copy()

    # Generate features and labels
    window_df = feature_generator.create_all_features(window_df)
    window_df = label_generator.elasticnet_label_next_high(window_df)
    window_df.dropna(inplace=True)

    for group_name, feature_list in TOP_GROUPS.items():
        available = [f for f in feature_list if f in window_df.columns]
        if len(available) < 2:
            continue

        group_data = window_df[available + ['Next_High']].copy()
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

        window_results.append([
            f"Window {i+1}", group_name, mse, rmse, r2
        ])

# Print summary
log.info("\n\n📊 SLIDING WINDOW RESULTS")
log.info("=" * 70)
log.info(f"{'Window':<10} | {'Group':<30} | {'MSE':<10} | {'RMSE':<10} | {'R²':<10}")
log.info("-" * 70)
for row in window_results:
    log.info(f"{row[0]:<10} | {row[1]:<30} | {row[2]:<10.4f} | {row[3]:<10.4f} | {row[4]:<10.4f}")

