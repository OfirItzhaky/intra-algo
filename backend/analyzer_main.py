from analyzer_load_eda import ModelLoaderAndExplorer
from analyzer_cerebro_strategy_engine import CerebroStrategyEngine

# === Initialize and Load Everything ===
model_loader = ModelLoaderAndExplorer(
    regression_path="regression_trainer_model.pkl",
    classifier_path="classifier_trainer_model.pkl"
)

regression_trainer, classifier_trainer, df_classifier_preds = model_loader.load_and_explore()

# ✅ Create Regression Prediction Frame
df_regression_preds = regression_trainer.x_test_with_meta[["Date", "Time", "Open", "High", "Low", "Close"]].copy()
df_regression_preds["Next_High"] = regression_trainer.y_test.values
df_regression_preds["Predicted_High"] = regression_trainer.predictions
# 🔄 Add aliases for prediction evaluation
df_regression_preds["Predicted"] = df_regression_preds["Predicted_High"]
df_regression_preds["Actual"] = df_regression_preds["Next_High"]

model_loader.plot_delta_distribution(df_regression_preds)
 # or any df that has Predicted_High & Close

model_loader.evaluate_prediction_distance(df_regression_preds, threshold=1.0)

model_loader.plot_high_confidence_by_hour(df_regression_preds)

model_loader.plot_trade_volume_and_avg(df_regression_preds)



# === Constants ===
TICK_SIZE = 0.25
TICK_DOLLAR_VALUE = 1.25
CONTRACT_SIZE = 1
STARTING_CASH = 10000.0

TARGET_TICKS = 10
STOP_TICKS = 10
MIN_DIST = 3.0
MAX_DIST = 20.0
MIN_CLASSIFIER_SIGNALS = 0
SESSION_START = "10:00"
SESSION_END = "23:00"

# === Run Strategy via Cerebro Engine ===
strategy_engine = CerebroStrategyEngine(
    df_strategy=df_regression_preds,
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
    session_end=SESSION_END
)

results = strategy_engine.run_backtest()


print("Done")
