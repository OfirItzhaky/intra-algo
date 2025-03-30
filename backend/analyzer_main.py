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
