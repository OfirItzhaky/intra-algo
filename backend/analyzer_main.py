from analyzer_load_eda import ModelLoaderAndExplorer

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
TICK_VALUE = 1.25
CONTRACT_SIZE = 1
STARTING_CASH = 10000.0

# === Export core variables ===
__all__ = [
    "regression_trainer",
    "classifier_trainer",
    "df_regression_preds",
    "df_classifier_preds",
    "TICK_SIZE",
    "TICK_VALUE",
    "CONTRACT_SIZE",
    "STARTING_CASH"
]
