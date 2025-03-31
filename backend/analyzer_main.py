import pandas as pd
from pandasgui import show

from analyzer_load_eda import ModelLoaderAndExplorer
from analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from analyzer_dashboard import AnalyzerDashboard

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

# === Access Results After Backtest ===
strat = results[0]  # The strategy instance

# You already have a reference to the engine, so just use it:
final_value = strategy_engine.cerebro.broker.getvalue()
print(f"📦 Final Portfolio Value: {final_value:.2f}")

print(f"✅ Total Closed Trades: {len(strat.trades)}")


# === Initialize the Dashboard Plotting Utility ===
dashboard = AnalyzerDashboard(
    df_strategy=df_regression_preds,
    df_classifiers=df_classifier_preds
)

# === Step 1: Validate Trades for Plotting ===
valid_trades = dashboard.validate_trades_for_plotting(strat.trades)

# === Step 2: Plot Using Plotly ===
# dashboard.plot_trades_and_predictions(valid_trades)

# === Show Trades in Interactive Table ===
df_trades = pd.DataFrame(strat.trades)
# show(df_trades)


# === Show Equity Curve with Drawdown ===
# dashboard.plot_equity_curve_with_drawdown(df_trades)

# dashboard.analyze_trade_duration_buckets(df_trades)
# === Load 1-minute bars for exit logic ===
df_1min = model_loader.load_1min_bars("MES_1_MINUTE_JAN_13_JAN_21.txt")

# ✅ Validate and fill 1-min bars needed for exit simulation
df_1min_updated, missing_1min_bars = model_loader.validate_and_fill_1min_bars(
    df_1min=df_1min,
    df_test_results=df_regression_preds,
    session_start=SESSION_START,
    session_end=SESSION_END
)

# 🚨 Optional: Alert for missing 1-min bars
if missing_1min_bars:
    print(f"🚨 Missing bars detected: {len(missing_1min_bars)} total")
    for m in missing_1min_bars[:10]:  # show only a few
        print(m)

# === Run Intrabar Strategy using the updated 1-min data
results_intrabar, cerebro_intrabar = strategy_engine.run_intrabar_backtest_fresh(
    df_regression_preds, df_classifier_preds, df_1min_updated
)
strat_intrabar = results_intrabar[0]

# === Final Portfolio Stats
final_value_intrabar = strategy_engine.cerebro.broker.getvalue()
print(f"📦 Final Portfolio Value (Intrabar): {final_value_intrabar:.2f}")
print(f"✅ Total Closed Trades (Intrabar): {len(strat_intrabar.trades)}")

# === Format and Visualize Trades
df_trades_intrabar = pd.DataFrame(strat_intrabar.trades)
dashboard.plot_equity_curve_with_drawdown(df_trades_intrabar)


show(df_trades_intrabar)

df_metrics = dashboard.calculate_strategy_metrics(df_trades_intrabar)

# Option 4 – Plotly
dashboard.display_metrics_plotly(df_metrics)




print("Done")
