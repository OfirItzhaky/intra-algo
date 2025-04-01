import pandas as pd
from pandasgui import show
import backtrader as bt

from analyzer_load_eda import ModelLoaderAndExplorer
from analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from analyzer_dashboard import AnalyzerDashboard
from analyzer_strategy_blueprint import Long5min1minStrategy  # Make sure path is correct

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
# strategy_engine = CerebroStrategyEngine(
#     df_strategy=df_regression_preds,
#     df_classifiers=df_classifier_preds,
#     initial_cash=STARTING_CASH,
#     tick_size=TICK_SIZE,
#     tick_value=TICK_DOLLAR_VALUE,
#     contract_size=CONTRACT_SIZE,
#     target_ticks=TARGET_TICKS,
#     stop_ticks=STOP_TICKS,
#     min_dist=MIN_DIST,
#     max_dist=MAX_DIST,
#     min_classifier_signals=MIN_CLASSIFIER_SIGNALS,
#     session_start=SESSION_START,
#     session_end=SESSION_END
# )
#
# results = strategy_engine.run_backtest()

# === Access Results After Backtest ===
# strat = results[0]  # The strategy instance

# # You already have a reference to the engine, so just use it:
# final_value = strategy_engine.cerebro.broker.getvalue()
# print(f"📦 Final Portfolio Value: {final_value:.2f}")
#
# print(f"✅ Total Closed Trades: {len(strat.trades)}")
#
#
# # === Initialize the Dashboard Plotting Utility ===
# dashboard = AnalyzerDashboard(
#     df_strategy=df_regression_preds,
#     df_classifiers=df_classifier_preds
# )

# # === Step 1: Validate Trades for Plotting ===
# valid_trades = dashboard.validate_trades_for_plotting(strat.trades)
#
# # === Step 2: Plot Using Plotly ===
# dashboard.plot_trades_and_predictions(valid_trades)
#
# # === Show Trades in Interactive Table ===
# df_trades = pd.DataFrame(strat.trades)
# show(df_trades)


# === Show Equity Curve with Drawdown ===
# dashboard.plot_equity_curve_with_drawdown(df_trades)
#
# dashboard.analyze_trade_duration_buckets(df_trades)

#todo#################### INTRABAR ###############################


# === Load 1-minute bars for exit logic ===
df_1min = model_loader.load_1min_bars("MES_1_MINUTE_JAN_13_JAN_21.txt")

# ✅ Validate and fill 1-min bars needed for exit simulation
df_1min_updated, missing_1min_bars = model_loader.validate_and_fill_1min_bars(
    df_1min=df_1min,
    df_test_results=df_regression_preds,
    session_start=SESSION_START,
    session_end=SESSION_END
)

df_1min_enriched = model_loader.enrich_1min_with_predictions(
    df_1min=df_1min_updated,
    df_regression_preds=df_regression_preds,
    df_classifier_preds=df_classifier_preds
)




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
    session_end=SESSION_END
)

results_5min1min, cerebro = strategy_engine.run_backtest_Long5min1minStrategy(
    df_5min=df_regression_preds,
    df_1min=df_1min_enriched,
    strategy_class=Long5min1minStrategy
)

dashboard_intrabar = AnalyzerDashboard(
    df_strategy=df_regression_preds,
    df_classifiers=df_classifier_preds
)
df_trades_intrabar = dashboard_intrabar.build_trade_dataframe_from_orders(list(cerebro.broker.orders))




final_value_intrabar = final_value = cerebro.broker.getvalue()

print(f"📦 Final Portfolio Value (Intrabar): {final_value_intrabar:.2f}")
# trade_analysis = results_5min1min.analyzers.trades.get_analysis()
# closed_trades = trade_analysis.total.closed if 'total' in trade_analysis and 'closed' in trade_analysis.total else 0
# print(f"✅ Total Closed Trades (Intrabar): {closed_trades}")







dashboard_intrabar.plot_equity_curve_with_drawdown(df_trades_intrabar)



# ✅ Plot intrabar trades with new method
dashboard_intrabar.plot_trades_and_predictions_intrabar_1_min(
    trade_df=df_trades_intrabar,
    df_1min=df_1min_updated
)


show(df_trades_intrabar)
df_metrics = dashboard_intrabar.calculate_strategy_metrics(df_trades_intrabar)



strategy_params = {
    "Strategy Class": type(strat_intrabar).__name__,
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
}
dashboard_intrabar.display_strategy_and_metrics_side_by_side(df_metrics, strategy_params)



print("Done")
