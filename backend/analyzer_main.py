import numpy as np
import pandas as pd
from pandasgui import show
import backtrader as bt
import joblib
from imblearn.over_sampling import SMOTE

from analyzer_load_eda import ModelLoaderAndExplorer
from analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from analyzer_dashboard import AnalyzerDashboard
from analyzer_strategy_blueprint import Long5min1minStrategy
from backend.classifier_model_trainer import ClassifierModelTrainer
from backend.data_loader import DataLoader
from backend.data_processor import DataProcessor
from backend.label_generator import LabelGenerator
from backend.feature_generator import FeatureGenerator
from backend.regression_model_trainer import RegressionModelTrainer



def get_model_save_path(model_type, data_size):
    """Helper function to generate model save paths with timestamp and data size"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    return f"{model_type}_model_{timestamp}_size_{data_size}.pkl"

def run_analyzer_with_pretrained():
    """Load and prepare pre-trained models for analysis."""
    # Initialize and Load Everything
    model_loader = ModelLoaderAndExplorer(
        regression_path=REGRESSION_MODEL_PATH,
        classifier_path=CLASSIFIER_MODEL_PATH
    )

    regression_trainer, classifier_trainer, df_classifier_preds = model_loader.load_and_explore()

    # Create Regression Prediction Frame
    df_regression_preds = regression_trainer.x_test_with_meta[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
    df_regression_preds["Next_High"] = regression_trainer.y_test.values
    df_regression_preds["Predicted_High"] = regression_trainer.predictions
    # Add aliases for prediction evaluation
    df_regression_preds["Predicted"] = df_regression_preds["Predicted_High"]
    df_regression_preds["Actual"] = df_regression_preds["Next_High"]

    # Initial Analysis
    model_loader.plot_delta_distribution(df_regression_preds)
    model_loader.evaluate_prediction_distance(df_regression_preds, threshold=1.0)
    model_loader.plot_high_confidence_by_hour(df_regression_preds)
    model_loader.plot_trade_volume_and_avg(df_regression_preds)

    return regression_trainer, classifier_trainer, df_classifier_preds, df_regression_preds, model_loader

def run_analyzer_with_training():
    """Train new models and prepare them for analysis."""
    print(f"\nStarting training phase using {TRAINING_DATA_PATH}")
    
    # Initialize classes
    data_loader = DataLoader()
    feature_generator = FeatureGenerator()
    label_generator = LabelGenerator()
    
    # Load data
    df = data_loader.load_from_csv(TRAINING_DATA_PATH)
    
    # Validate data
    if df.empty or "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError(f"Training file could not be loaded or is empty: {TRAINING_DATA_PATH}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["Date", "Time"])
    
    # Generate features
    df = feature_generator.create_all_features(df)
    
    # Generate regression label
    df = label_generator.elasticnet_label_next_high(df)
    
    # Store meta columns before feature selection
    meta_columns = df[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
    
    # Initialize regression trainer with constants
    regression_trainer = RegressionModelTrainer(
        include_prices=not DROP_PRICE_COLUMNS,
        apply_filter=APPLY_FILTER,
        filter_threshold=FILTER_THRESHOLD
    )
    
    # Prepare data and train regression model
    regression_trainer.prepare_data(df)
    regression_trainer.train_model()
    regression_trainer.make_predictions()
    
    # Add meta columns back to x_test
    regression_trainer.x_test_with_meta = regression_trainer.x_test.copy()
    regression_trainer.x_test_with_meta = regression_trainer.x_test_with_meta.join(
        meta_columns.loc[regression_trainer.x_test.index]
    )

    # Prepare data for classifier labels
    data_selected_filtered = regression_trainer.x_test_with_meta.loc[regression_trainer.y_test.index].copy()
    data_selected_filtered["Predicted_High"] = regression_trainer.predictions
    data_selected_filtered["Prev_Close"] = data_selected_filtered["Close"].shift(1)
    data_selected_filtered["Prev_Predicted_High"] = data_selected_filtered["Predicted_High"].shift(1)
    
    # Drop NaNs before classification label generation
    data_selected_filtered = data_selected_filtered.dropna(subset=["Predicted_High", "Prev_Close", "Prev_Predicted_High"])
    
    # Generate classifier labels using specified method
    label_method = getattr(label_generator, LABEL_GENERATION_METHOD)
    df_with_labels = label_method(data_selected_filtered)
    
    # Get the appropriate target column based on method
    if LABEL_GENERATION_METHOD == "add_good_bar_label":
        target_column = "good_bar_prediction_outside_of_boundary"
    elif LABEL_GENERATION_METHOD in ["long_good_bar_label_all", "long_good_bar_label_bullish_only",
                                   "long_good_bar_label_all_goal_b", "long_good_bar_label_bullish_only_goal_b",
                                   "long_good_bar_label_all_goal_c", "long_good_bar_label_bullish_only_goal_c",
                                   "green_red_bar_label_goal_d"]:
        target_column = "long_good_bar_label"
    elif LABEL_GENERATION_METHOD == "option_d_multiclass_next_bar_movement":
        target_column = "multi_class_label"
    else:
        raise ValueError(f"Unsupported label generation method: {LABEL_GENERATION_METHOD}")
    
    # Prepare classifier training data
    processor = DataProcessor()
    classifier_X_train, classifier_y_train, classifier_X_test, classifier_y_test = processor.prepare_dataset_for_regression_sequential(
        data=df_with_labels.drop(columns=["Predicted_High", "Next_High"], errors="ignore"),
        target_column=target_column,
        drop_target=True,
        split_ratio=0.8
    )
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    classifier_X_train_bal, classifier_y_train_bal = smote.fit_resample(classifier_X_train, classifier_y_train)
    
    # Train classifiers
    classifier_trainer = ClassifierModelTrainer()
    classifier_trainer.train_all_classifiers(
        classifier_X_train_bal, 
        classifier_y_train_bal, 
        classifier_X_test, 
        classifier_y_test, 
        regression_trainer
    )
    
    # Save models if enabled
    if SAVE_TRAINED_MODELS:
        n_bars = len(df)
        regression_path = get_model_save_path("regression", n_bars)
        classifier_path = get_model_save_path("classifier", n_bars)
        
        joblib.dump(regression_trainer.model, regression_path)
        joblib.dump(classifier_trainer.model, classifier_path)
        print(f"Saved models:\n- {regression_path}\n- {classifier_path}")
    
    # Create classifier predictions dataframe
    df_classifier_preds = classifier_trainer.classifier_predictions_df
    
    # Handle multi-class vs binary classification
    if USE_MULTI_CLASS:
        print(f"🔀 Using multi-class labels with threshold {MULTI_CLASS_THRESHOLD}")
        if hasattr(classifier_trainer, 'multi_class_label') and classifier_trainer.multi_class_label is not None:
            df_classifier_preds['multi_class_label'] = classifier_trainer.multi_class_label
        else:
            print("⚠️ Warning: Multi-class labels not found in classifier trainer.")
    else:
        print("🔢 Using binary classification.")
    
    # Create regression predictions dataframe
    df_regression_preds = regression_trainer.x_test_with_meta[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
    df_regression_preds["Next_High"] = regression_trainer.y_test.values
    df_regression_preds["Predicted_High"] = regression_trainer.predictions
    df_regression_preds["Predicted"] = df_regression_preds["Predicted_High"]
    df_regression_preds["Actual"] = df_regression_preds["Next_High"]
    
    # Initialize ModelLoaderAndExplorer without paths
    model_loader = ModelLoaderAndExplorer(regression_path=None, classifier_path=None)
    
    return regression_trainer, classifier_trainer, df_classifier_preds, df_regression_preds, model_loader

# === Training Constants ===
USE_TRAINING_MODE = True  # Switch between training and pre-trained modes
TRAINING_DATA_PATH = "data/training/mes_2_new.csv"  # 5-min data for training
TRAIN_TEST_SPLIT = 0.8
DROP_PRICE_COLUMNS = True
APPLY_FILTER = True
FILTER_THRESHOLD = 4.0
SAVE_TRAINED_MODELS = False  # If True, save models with timestamp

# === Label Generation Methods ===
AVAILABLE_LABEL_METHODS = {
    "good_bar": "add_good_bar_label",                      # Binary classification for good bars
    "next_high": "elasticnet_label_next_high",            # Regression label for next high
    "long_all": "long_good_bar_label_all",                # All potential good bars
    "long_bullish": "long_good_bar_label_bullish_only",   # Only bullish good bars
    "long_bullish_b": "long_good_bar_label_bullish_only_goal_b",  # Bullish bars with goal B
    "long_all_b": "long_good_bar_label_all_goal_b",       # All bars with goal B
    "long_all_c": "long_good_bar_label_all_goal_c",       # All bars with goal C
    "long_bullish_c": "long_good_bar_label_bullish_only_goal_c",  # Bullish bars with goal C
    "green_red": "green_red_bar_label_goal_d",            # Green/Red bar classification
    "multi_class": "option_d_multiclass_next_bar_movement"  # Multi-class next bar movement
}

LABEL_GENERATION_METHOD = AVAILABLE_LABEL_METHODS["long_bullish_b"]  # Choose your method here

# === Analysis Constants ===
TICK_SIZE = 0.25
TICK_DOLLAR_VALUE = 1.25
CONTRACT_SIZE = 1
STARTING_CASH = 10000.0

TARGET_TICKS = 10
STOP_TICKS = 10
MIN_DIST = 4
MAX_DIST = 20.0
MIN_CLASSIFIER_SIGNALS = 1
SESSION_START = "17:00"
SESSION_END = "23:00"

# === Multi-class Settings ===
MULTI_CLASS_THRESHOLD = 3  # Classes >= 3 are considered positive signals
USE_MULTI_CLASS = True    # Set to True to use multi-class instead of binary

# === File Paths ===
REGRESSION_MODEL_PATH = "regression_trainer_model.pkl"
CLASSIFIER_MODEL_PATH = "classifier_trainer_model.pkl"
INTRABAR_DATA_PATH = "MES_1_MINUTE_JAN_13_JAN_21.txt"

# === Daily PnL Limits ===
MAX_DAILY_PROFIT = 36.0  # Equivalent to about 30 ticks
MAX_DAILY_LOSS = -36.0   # Negative value for losses

if __name__ == "__main__":
    # Choose which function to use
    # regression_trainer, classifier_trainer, df_classifier_preds, df_regression_preds, model_loader = run_analyzer_with_pretrained()
    regression_trainer, classifier_trainer, df_classifier_preds, df_regression_preds, model_loader = run_analyzer_with_training()

# === Load 1-minute bars for exit logic ===
    df_1min = model_loader.load_1min_bars(INTRABAR_DATA_PATH)

    # ✅ Validate and fill 1-min bars needed for exit simulation
    df_1min_updated, missing_1min_bars = model_loader.validate_and_fill_1min_bars(
        df_1min=df_1min,
        df_test_results=df_regression_preds,
        session_start=SESSION_START,
        session_end=SESSION_END
    )
    df_1min_updated.drop(columns=["PredHigh"], inplace=True)

    df_1min_enriched = model_loader.enrich_1min_with_predictions(
        df_1min=df_1min_updated,
        df_regression_preds=df_regression_preds,
        df_classifier_preds=df_classifier_preds
    )

    df_1min_enriched.drop(columns=["Actual"], inplace=True)

    # === Run New Strategy on 1-min enriched data ===

    strategy_engine = CerebroStrategyEngine(
        df_strategy=df_regression_preds,   # used to extract base predictions
        df_classifiers=df_classifier_preds,
        initial_cash=STARTING_CASH,
        tick_size=TICK_SIZE,
        tick_value=TICK_DOLLAR_VALUE,
        contract_size=CONTRACT_SIZE,
        target_ticks=TARGET_TICKS,
        stop_ticks=STOP_TICKS,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        min_classifier_signals=MIN_CLASSIFIER_SIGNALS,
        session_start=SESSION_START,
        session_end=SESSION_END,
        max_daily_profit=MAX_DAILY_PROFIT,
        max_daily_loss=MAX_DAILY_LOSS
    )

    results_5min1min, cerebro = strategy_engine.run_backtest_Long5min1minStrategy(
        df_5min=df_regression_preds,
        df_1min=df_1min_enriched,
                strategy_class=Long5min1minStrategy,
                use_multi_class=USE_MULTI_CLASS,
                multi_class_threshold=MULTI_CLASS_THRESHOLD
    )

    dashboard_intrabar = AnalyzerDashboard(
        df_strategy=df_regression_preds,
        df_classifiers=df_classifier_preds
    )
    df_trades_intrabar = dashboard_intrabar.build_trade_dataframe_from_orders(list(cerebro.broker.orders))

    df_trades_intrabar["pnl"] = df_trades_intrabar["pnl"] / TICK_SIZE * TICK_DOLLAR_VALUE

    final_value_intrabar = final_value = cerebro.broker.getvalue()

    print(f"📦 Final Portfolio Value (Intrabar): {final_value_intrabar:.2f}")

    dashboard_intrabar.plot_equity_curve_with_drawdown(df_trades_intrabar)

    def get_next_5_high(i, highs):
        future_window = highs[i+1:i+6]  # i+1 to i+5 inclusive (5 bars ahead)
        return future_window.max() if len(future_window) == 5 else np.nan

    high_series = df_1min_enriched["High"].values
    df_1min_enriched["Next_High"] = [
        get_next_5_high(i, high_series) if pd.notna(df_1min_enriched["Predicted"].iloc[i]) else np.nan
        for i in range(len(df_1min_enriched))
    ]

    # Apply only where Predicted is not zero
    df_1min_enriched["Next_High"] = np.where(df_1min_enriched["Predicted"] != 0, df_1min_enriched["Next_High"], np.nan)
    # Clean up
    df_1min_enriched["Predicted"] = df_1min_enriched["Predicted"].replace(0, np.nan)
    df_1min_enriched.loc[df_1min_enriched["Predicted"].isna(), "Next_High"] = np.nan

    # ✅ Plot intrabar trades with new method
    dashboard_intrabar.plot_trades_and_predictions_intrabar_1_min(
        trade_df=df_trades_intrabar,
        df_1min=df_1min_enriched
    )

    show(df_trades_intrabar)
    df_metrics = dashboard_intrabar.calculate_strategy_metrics(df_trades_intrabar)

    strategy_params = {
        "Strategy Class": type(results_5min1min).__name__,
        "Tick Size": TICK_SIZE,
        "Tick Value ($)": TICK_DOLLAR_VALUE,
        "Contract Size": CONTRACT_SIZE,
        "Target Ticks": TARGET_TICKS,
        "Stop Ticks": STOP_TICKS,
        "Min Distance (Points)": MIN_DIST,
        "Max Distance (Points)": MAX_DIST,
        "Min Classifier Signals": MIN_CLASSIFIER_SIGNALS,
        "Session Start": SESSION_START,
        "Session End": SESSION_END,
        "Initial Cash ($)": STARTING_CASH,
        "Using Multi-Class": USE_MULTI_CLASS,
        "Multi-Class Threshold": MULTI_CLASS_THRESHOLD if USE_MULTI_CLASS else "N/A",
        "Max Daily Profit": MAX_DAILY_PROFIT,
        "Max Daily Loss": MAX_DAILY_LOSS
    }
    dashboard_intrabar.display_strategy_and_metrics_side_by_side(df_metrics, strategy_params)

    print("Done")