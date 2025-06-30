import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from backend.analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from backend.analyzer_dashboard import AnalyzerDashboard
import requests, os, json

from research_agent.config import CONFIG
import traceback
import time
import math  # Ensure math is imported for isinf/isnan
try:
    from research_agent.app import regression_backtest_tracker
except ImportError:
    regression_backtest_tracker = None
from research_agent.scalp_agent.scalp_base_agent import BaseAgent

class RegressionPredictorAgent:
    """
    RegressionPredictorAgent forecasts both next-bar high and next-bar low using dual regression models.
    It supports trade filtering based on user-defined thresholds for long/short setups:
      - Only consider long if predicted_high - current_close > long_threshold
      - Only consider short if current_close - predicted_low > short_threshold
    The agent is retrainable per session using the uploaded CSV. All configs and thresholds are set via user_params.
    Results are exposed in a clean dict for downstream evaluation or backtesting, including predictions, filters, and metadata.
    Includes data size and feature shape validation for robust inference.
    Also supports trade simulation, performance evaluation, and result summarization.
    """
    def __init__(self, user_params=None):
        self.user_params = user_params or {}
        self.model_high = None
        self.model_low = None
        self.scaler = None
        self.feature_cols = None
        self.feature_names = None
        self.metadata = {}
        self.last_feedback = None
        self.all_strategy_sims = []  # Store all sim dicts for later use
        self.all_strategy_results = []  # Store all results for later use

    def _compute_group_1_features(self, df):
        # Ensure all column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        # Compute FastAvg: (high + low + close) / 3
        if 'fastavg' not in df.columns:
            df['fastavg'] = (df['high'] + df['low'] + df['close']) / 3
        # Compute Close_vs_EMA_10: close - EMA_10(close)
        if 'close_vs_ema_10' not in df.columns:
            ema_10 = df['close'].ewm(span=10, adjust=False).mean()
            df['close_vs_ema_10'] = df['close'] - ema_10
        # Compute High_15Min: rolling max of high over 3 bars (assuming 5min bars)
        if 'high_15min' not in df.columns:
            df['high_15min'] = df['high'].rolling(window=3, min_periods=1).max()
        # Compute MACD: EMA_12(close) - EMA_26(close)
        if 'macd' not in df.columns:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
        # Compute High_vs_EMA_5_High: high - EMA_5(high)
        if 'high_vs_ema_5_high' not in df.columns:
            ema_5_high = df['high'].ewm(span=5, adjust=False).mean()
            df['high_vs_ema_5_high'] = df['high'] - ema_5_high
        # Compute ATR: average true range (14)
        if 'atr' not in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14, min_periods=1).mean()
        return df

    def fit(self, df):
        # --- Clean up/reset internal state ---
        self.model_high = None
        self.model_low = None
        self.scaler = None
        self.feature_cols = None
        self.feature_names = None
        self.metadata = {}
        self.last_feedback = None
        # --- Minimum data check ---
        if len(df) < 500:
            self.last_feedback = "Insufficient data — at least 500 bars required for regression-based prediction."
            return False
        # --- Feature engineering: Use benchmark Group 1 features only ---
        features = ['fastavg', 'close_vs_ema_10', 'high_15min', 'macd', 'high_vs_ema_5_high', 'atr']
        # Compute features if missing
        df = self._compute_group_1_features(df)
        # --- Targets ---
        df = df.copy()
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        # --- Drop NaNs in features and targets ---
        self.feature_names = features
        df.dropna(subset=[*self.feature_names, 'next_high', 'next_low'], inplace=True)
        # --- Features and targets ---
        X = df[self.feature_names]
        y_high = df['next_high'].values
        y_low = df['next_low'].values
        # --- Scaling ---
        self.scaler = StandardScaler().fit(X.values)
        X_scaled = self.scaler.transform(X.values)
        # --- Fit models ---
        self.model_high = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_scaled, y_high)
        self.model_low = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X_scaled, y_low)
        self.feature_cols = self.feature_names
        self.metadata['n_train'] = len(df)
        self.metadata['model_high_score'] = self.model_high.score(X_scaled, y_high)
        self.metadata['model_low_score'] = self.model_low.score(X_scaled, y_low)
        return True

    def predict(self, df):
        # Use the last row for prediction
        if self.last_feedback:
            return {'feedback': self.last_feedback}
        last = df.iloc[[-1]].copy()
        last['return'] = last['close'].pct_change().fillna(0)
        last['volatility'] = df['close'].rolling(10).std().iloc[-1]
        last['ma_5'] = df['close'].rolling(5).mean().iloc[-1]
        last['ma_10'] = df['close'].rolling(10).mean().iloc[-1]
        last['rsi_7'] = self._rsi(df['close'], 7).iloc[-1]
        last = last.fillna(0)
        # --- Feature shape consistency check ---
        if self.feature_names is not None:
            X_last = last[self.feature_names]
        else:
            X_last = last[self.feature_cols]
        X_last_scaled = self.scaler.transform(X_last.values)
        predicted_high = float(self.model_high.predict(X_last_scaled)[0])
        predicted_low = float(self.model_low.predict(X_last_scaled)[0])
        current_close = float(last['close'].values[0])
        # --- Filtering logic ---
        long_threshold = float(self.user_params.get('long_threshold', 0.5))
        short_threshold = float(self.user_params.get('short_threshold', 0.5))
        filtered_long_ok = (predicted_high - current_close) > long_threshold
        filtered_short_ok = (current_close - predicted_low) > short_threshold
        # --- Output dict ---
        result = {
            'predicted_high': predicted_high,
            'predicted_low': predicted_low,
            'current_close': current_close,
            'filtered_long_ok': filtered_long_ok,
            'filtered_short_ok': filtered_short_ok,
            'long_threshold': long_threshold,
            'short_threshold': short_threshold,
            'model_high_score': self.metadata.get('model_high_score'),
            'model_low_score': self.metadata.get('model_low_score'),
            'n_train': self.metadata.get('n_train'),
        }
        return result

    def simulate_trades(self, config_idx=None, config_dict=None, label=None):
        """
        Export trades and metrics for a given config (by index or config dict) using stored results.
        Never re-runs the simulation or calls Backtrader/Cerebro.
        Continues with all post-processing: metrics, CSV export, summary, etc.
        If label is provided (e.g., 'best', 'worst'), saves to research_agent/uploaded_csvs/{label}_strategy_trades.csv
        """
        if config_idx is not None:
            result = self.all_strategy_results[config_idx]
        elif config_dict is not None:
            # Find the result matching the config_dict
            result = next((r for r in self.all_strategy_results if r['config'] == config_dict), None)
            if result is None:
                print("[simulate_trades] No result found for given config.")
                return None
        else:
            print("[simulate_trades] Must provide config_idx or config_dict.")
            return None
        trades = result['trades']
        metrics = result['metrics']
        trades_df = pd.DataFrame(trades)
        # Save to CSV in uploaded_csvs with clear name if label is provided
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs')
        os.makedirs(out_dir, exist_ok=True)
        if label:
            trades_path = os.path.abspath(os.path.join(out_dir, f"{label}_strategy_trades.csv"))
        else:
            trades_path = os.path.abspath(os.path.join(out_dir, f"exported_trades_config_{config_idx if config_idx is not None else 'custom'}.csv"))
        trades_df.to_csv(trades_path, index=False)
        print(f"[simulate_trades] Saved trades to {trades_path}")
        # Instantiate AnalyzerDashboard to call instance method
        dashboard = AnalyzerDashboard(pd.DataFrame(), pd.DataFrame())
        metrics_df = dashboard.calculate_strategy_metrics_for_ui(trades_df)
        if metrics_df is not None and not metrics_df.empty:
            backtest_summary = metrics_df.to_dict(orient='records')[0]
        else:
            backtest_summary = metrics or {}
        # LLM summary (optional, keep if used downstream)
        num_trades = backtest_summary.get('num_trades', 0)
        win_rate = backtest_summary.get('win_rate', 0.0)
        avg_return = backtest_summary.get('avg_return', 0.0)
        avg_duration = backtest_summary.get('avg_duration', 0.0)
        max_drawdown = backtest_summary.get('max_drawdown', 0.0)
        profit_factor = backtest_summary.get('profit_factor', 0.0)
        llm_summary = self._summarize_trades(
            num_trades if isinstance(num_trades, (int, float)) else 0,
            win_rate if isinstance(win_rate, (int, float)) else 0.0,
            avg_return if isinstance(avg_return, (int, float)) else 0.0,
            avg_duration if isinstance(avg_duration, (int, float)) else 0.0,
            max_drawdown if isinstance(max_drawdown, (int, float)) else 0.0,
            profit_factor if isinstance(profit_factor, (int, float)) else 0.0
        )
        # Return full result dict for downstream use
        return {
            "strategy_name": "Regression Predictor (Backtrader)",
            "entry_conditions": result['config'],
            "trades": trades_df.to_dict(orient='records') if not trades_df.empty else [],
            "trades_df": trades_df,
            "backtest_summary": backtest_summary,
            "llm_summary": llm_summary,
            "model_scores": {},  # Optionally fill if available
            "n_train": self.metadata.get('n_train'),
        }

    def find_best_threshold_strategy(self, df, df_1min=None, user_params=None):
        """
        Search for the best long/short threshold config by simulating multiple values and picking the best by profit factor (or win rate).
        Returns a dict with chosen thresholds, summary, trades, backtest, risk metadata, and explanation.
        Also returns top_strategies (top 3 configs) and a recommended_strategy_summary for UI display.
        Always uses df_1min for intrabar simulation if provided.
        """
        if self.last_feedback:
            return {'feedback': self.last_feedback}
        # Save the original 5m OHLCV DataFrame for plotting
        raw_ohlcv_df = df.copy()
        max_drawdown, min_profit_factor, min_trades, results, threshold_grid = self._generate_threshold_grid(
            user_params)
        total_configs = len(threshold_grid)
        if regression_backtest_tracker is not None:
            regression_backtest_tracker["total"] = total_configs
            regression_backtest_tracker["current"] = 0
        strategy_matrix_llm = []  # Ensure always defined
        llm_context = {}         # Ensure always defined
        self.all_strategy_sims = []  # Store all sim dicts for later use
        df_with_preds = None  # Will hold the DataFrame with predictions for plotting
        df_with_preds = self._simulate_all_configs(df, df_1min, df_with_preds, results, threshold_grid, total_configs,
                                                   user_params)
        # --- Rule-based filtering ---
        filtered_out_configs, filtered_results, filtered_results_clean = self._filter_results_by_rules(max_drawdown,
                                                                                                       min_profit_factor,
                                                                                                       min_trades,
                                                                                                       results)
        best, top_strategies = self._rank_and_pick_best_strategies(filtered_results_clean)
        best_result = best['sim'] if best and 'sim' in best else None
        # Compose explanation
        if best:
            explanation = (
                f"Best thresholds: long_delta={best.get('long_threshold')}, short_delta={best.get('short_threshold')}. "
                f"Profit factor: {best.get('profit_factor', 0.0):.2f}, "
                f"Win rate: {best.get('win_rate', 0.0):.1f}%. "
                f"Risk violations: {best.get('risk_violations', 0)}, "
                f"Daily loss violations: {best.get('daily_loss_violations', 0)}."
            )
            if best_result:
                best_result['threshold_search'] = {
                    'chosen_long_threshold': best.get('long_threshold'),
                    'chosen_short_threshold': best.get('short_threshold'),
                    'risk_meta': {
                        'risk_violations': best.get('risk_violations', 0),
                        'daily_loss_violations': best.get('daily_loss_violations', 0)
                    },
                    'explanation': explanation
                }
                best_result['llm_summary'] = best_result.get('llm_summary', '') + ' ' + explanation
                # Add top_strategies and recommendation
                best_result['top_strategies'] = [
                    {
                        'long_threshold': r['long_threshold'],
                        'short_threshold': r['short_threshold'],
                        'min_volume_pct_change': r['min_volume_pct_change'],
                        'bar_color_filter': r['bar_color_filter'],
                        'profit_factor': r['profit_factor'],
                        'win_rate': r['win_rate'],
                        'avg_return': r['avg_return'],
                        'max_drawdown': r['max_drawdown'],
                        'num_trades': r['num_trades'],
                        'risk_violations': r['risk_violations'],
                        'daily_loss_violations': r['daily_loss_violations'],
                        'direction_tag': r['direction_tag']
                    } for r in top_strategies
                ]
                best_result['recommended_strategy_summary'] = (
                    f"Strategy with long threshold {best.get('long_threshold')} and short threshold {best.get('short_threshold')} "
                    f"yielded the highest profit factor ({best.get('profit_factor', 0.0):.2f}) and stayed within risk limits."
                )
                best_result['filtered_out_configs'] = filtered_out_configs
                best_result['filter_meta'] = {
                    'min_trades': min_trades,
                    'max_drawdown': max_drawdown,
                    'min_profit_factor': min_profit_factor
                }
                # --- Build strategy_matrix_llm for LLM use ---
                llm_context, strategy_matrix_llm = self._build_llm_context_and_matrix(best_result, filtered_results,
                                                                                      llm_context, strategy_matrix_llm,
                                                                                      user_params)
                # --- Real LLM output for recommended strategy ---
                model_name = CONFIG.get("image_analysis_model", "gpt-4o")
                provider = CONFIG.get("model_provider", "openai")
                api_key = CONFIG.get("openai_api_key") if provider == "openai" else CONFIG.get("gemini_api_key")
                bias_summary = llm_context.get('bias_summary')
                if not bias_summary or not isinstance(bias_summary, str) or not bias_summary.strip():
                    best_result['llm_strategy_recommendation'] = {
                        'selected_strategy': None,
                        'alternative_strategies': [],
                        'llm_rationale': "Bias context is missing. Please run the Multi-Timeframe Agent first.",
                        'model_used': model_name,
                        'tokens_used': 0,
                        'cost_usd': 0.0,
                        'llm_error': True
                    }
                    return best_result
                # --- Build prompt ---
                top_strats = strategy_matrix_llm[:10]
                user_risk = llm_context['user_risk_settings']
                llm_prompt = """
                    You are a trading strategist assistant. Your task is to select the best strategies from a 256-strategy grid and an optional market bias summary.
                    
                    🎯 Your Goal:
                    Return exactly **three trading strategies**, selected based on the data, and output them in a **fixed JSON format**. This format must always be followed **exactly as shown below** to ensure consistency.
                    
                    🧠 Inputs:
                    - A strategy grid containing 256 rows, each with metrics like: win rate, profit factor, max drawdown, avg daily PnL, etc.
                    - An optional market bias text summary that may include sector flow, macro bias, symbol news, or support/resistance levels.
                    
                    📋 Selection Criteria:
                    - Choose based on meaningful metrics (not only PnL) — including stability, risk, win rate, and drawdown.
                    - You may weight metrics differently if market bias suggests strong long/short skew or risk-aversion.
                    - Use judgment to combine multiple signals into effective strategy logic.
                    
                    📤 **Output Format – This must be strictly followed**:
                    Return your response as a JSON with the following structure:
                    
                    ```json
                    {
                      "top_strategies": [
                        {
                          "name": "Volatile Shorts",
                          "logic": "Trade when predicted short > 0.6 and candle color is red. Use only if volume spike > 5%.",
                          "direction": "short",
                          "stop_loss_ticks": 10,
                          "take_profit_ticks": 10,
                          "key_metrics": {
                            "profit_factor": 1.25,
                            "win_rate": 52.3,
                            "avg_daily_pnl": 5.2,
                            "max_drawdown": 90.0
                          },
                          "rationale": "This strategy favors short trades in high-volume environments. It has stable win rate and low drawdown, making it ideal in bearish or volatile conditions."
                        },
                        {
                          "name": "Balanced Intraday",
                          "logic": "Trade when either long/short predicted > 0.5 and candle matches direction. No volume filter.",
                          "direction": "both",
                          "stop_loss_ticks": 10,
                          "take_profit_ticks": 10,
                          "key_metrics": {
                            "profit_factor": 1.18,
                            "win_rate": 48.7,
                            "avg_daily_pnl": 6.9,
                            "max_drawdown": 100.0
                          },
                          "rationale": "This is a general-purpose strategy that performs well on both sides with decent stability. Recommended for trend-neutral sessions."
                        },
                        {
                          "name": "Momentum Longs",
                          "logic": "Trade when predicted long > 0.7 and min_volume_pct_change > 3%. Only on green candles.",
                          "direction": "long",
                          "stop_loss_ticks": 10,
                          "take_profit_ticks": 10,
                          "key_metrics": {
                            "profit_factor": 1.32,
                            "win_rate": 56.2,
                            "avg_daily_pnl": 7.5,
                            "max_drawdown": 70.0
                          },
                          "rationale": "Best performing long-biased strategy with high win rate and consistent daily returns. Ideal for strong bullish sessions."
                        }
                      ]
                    }
                    ```
                    📌 Rules:
                    
                    Do not return anything outside the JSON block.
                    
                    Always include exactly 3 strategies unless instructed otherwise.
                    
                    Ensure keys and structure are identical in casing and order.
                    
                    If market bias is empty, ignore it. If present, use it to prioritize strategy alignment.
                    
                    Below is the strategy grid and bias summary:
                    
                    STRATEGY GRID:
                    {grid_json}
                    
                    BIAS SUMMARY:
                    {bias_str}
                    """
        df_results = self._save_and_visualize_results(best_result, df_with_preds, results)
        # Now call LLM selector
        print('[LLM] Calling LLM for top strategy selection...')
        self._call_llm_and_attach(best_result, df_results, user_params)

        # --- Ensure result is serializable before returning ---
        def to_serializable(obj):
            basic_types = (str, int, float, bool, type(None))
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items() if k not in ['strategy', 'cerebro'] and (isinstance(v, basic_types) or isinstance(v, (dict, list)))}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, basic_types):
                if isinstance(obj, float):
                    if math.isinf(obj) or math.isnan(obj):
                        return None  # or 0.0 if you prefer
                return obj
            else:
                return str(obj)
        best_result_serializable = to_serializable(best_result)
        return best_result_serializable

    def _save_and_visualize_results(self, best_result, df_with_preds, results):
        df_results = pd.DataFrame(results)
        print(f"[SUMMARY] Total rows: {len(df_results)}")
        print(f"[SUMMARY] Columns: {df_results.columns.tolist()}")
        # Find best/worst by total_pnl if present, else by avg_return
        best_idx, worst_idx = None, None
        if 'avg_return' in df_results.columns and df_results['avg_return'].notnull().any():
            best_idx = df_results['avg_return'].idxmax()
            worst_idx = df_results['avg_return'].idxmin()

        # Save cleaned results (excluding sim_trades) to CSV
        save_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs', 'strategy_grid_results.csv'))
        df_results.to_csv(save_path, index=False)
        # Save trades for best/worst using simulate_trades (ensures consistent formatting)
        if best_idx is not None:
            self.simulate_trades(config_idx=best_idx, label='best')
        if worst_idx is not None:
            # If only one strategy, best and worst are the same
            if best_idx == worst_idx:
                self.simulate_trades(config_idx=best_idx, label='worst')
            else:
                self.simulate_trades(config_idx=worst_idx, label='worst')
        # --- Multi-Heatmap Visualization ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        df = pd.DataFrame(results)
        print(f"[DEBUG] Heatmap DataFrame columns: {df.columns.tolist()} shape: {df.shape}")
        required_cols = ['long_threshold', 'short_threshold', 'min_volume_pct_change', 'bar_color_filter', 'win_rate',
                         'avg_pnl']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARNING] Cannot plot heatmaps, missing columns: {missing_cols}")
        else:
            min_vols = [0.0, 0.025, 0.05]
            bar_colors = [False, True]
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for i, min_vol in enumerate(min_vols):
                for j, bar_color in enumerate(bar_colors):
                    ax = axes[j, i]
                    sub = df[(df['min_volume_pct_change'] == min_vol) & (df['bar_color_filter'] == bar_color)]
                    if sub.empty:
                        ax.set_title(f"min_vol={min_vol}, bar_color={bar_color}\n(No data)")
                        ax.axis('off')
                        continue
                    pivot = sub.pivot_table(index='long_threshold', columns='short_threshold', values='win_rate',
                                            aggfunc='mean')
                    if pivot.isnull().all().all():
                        # fallback to avg_pnl
                        pivot = sub.pivot_table(index='long_threshold', columns='short_threshold', values='avg_pnl',
                                                aggfunc='mean')
                        value_label = 'avg_pnl'
                    else:
                        value_label = 'win_rate'
                    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', cbar=True, ax=ax)
                    ax.set_title(f"min_vol={min_vol}, bar_color={bar_color}\n({value_label})")
                    ax.set_xlabel('short_threshold')
                    ax.set_ylabel('long_threshold')
            plt.tight_layout()
            # Save to intra_algo/research_agent/uploaded_csvs/heatmap_debug.png using project-relative path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../research_agent
            heatmap_path = os.path.join(project_root, 'uploaded_csvs', 'heatmap_debug.png')
            plt.savefig(heatmap_path)
            plt.close()
            print(f"[INFO] Heatmap saved to {heatmap_path}")
        # After saving the heatmap, show trades and metrics visuals, then call LLM selector
        if best_result and 'trades' in best_result and best_result['trades']:
            regression_plot_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs', 'regression_trades_plot.html'))
            dashboard = AnalyzerDashboard(df_with_preds, pd.DataFrame())
            dashboard.plot_trades_and_predictions_regression_agent(
                trade_df=pd.DataFrame(best_result['trades']),
                max_trades=50,
                save_path=regression_plot_path
            )
            # Show metrics/params table
            metrics_df = None
            params_dict = None
            if 'top_strategies' in best_result and best_result['top_strategies']:
                params_dict = best_result['top_strategies'][0]
            if 'trades' in best_result and best_result['trades']:
                trades_df = pd.DataFrame(best_result['trades'])
                dashboard_metrics = dashboard.calculate_strategy_metrics_for_ui(trades_df)
                if dashboard_metrics is not None and not dashboard_metrics.empty:
                    metrics_df = dashboard_metrics
            if metrics_df is not None and params_dict is not None:
                dashboard.display_strategy_and_metrics_side_by_side(metrics_df, params_dict)
        return df_results

    def _call_llm_and_attach(self, best_result, df_results, user_params):
        bias_summary = (user_params or self.user_params).get('bias_summary', None)
        llm_result = self.llm_select_top_strategies_from_grid(df_results, bias_summary=bias_summary)
        best_result['llm_top_strategies'] = llm_result
        # Add cost/token info if available
        cost = None
        tokens = None
        if isinstance(llm_result, dict):
            cost = llm_result.get('cost_usd') or llm_result.get('cost') or 0.0
            tokens = llm_result.get('tokens_used') or llm_result.get('token_usage') or 0
        best_result['llm_top_strategies_cost'] = {'cost_usd': cost, 'tokens': tokens}

    def _build_llm_context_and_matrix(self, best_result, filtered_results, llm_context, strategy_matrix_llm,
                                      user_params):
        strategy_matrix_llm = []
        for r in filtered_results:
            entry = {
                'long_threshold': r.get('long_threshold'),
                'short_threshold': r.get('short_threshold'),
                'min_volume_pct_change': r.get('min_volume_pct_change'),
                'bar_color_filter': r.get('bar_color_filter'),
                'direction_tag': r.get('direction_tag'),
                'profit_factor': r.get('profit_factor'),
                'win_rate': r.get('win_rate'),
                'avg_return': r.get('avg_return'),
                'num_trades': r.get('num_trades'),
                'max_drawdown': r.get('max_drawdown'),
                'risk_violations': r.get('risk_violations'),
                'daily_loss_violations': r.get('daily_loss_violations'),
            }
            # Optionally add more contextually useful fields here
            strategy_matrix_llm.append(entry)
        # --- Prepare LLM context ---
        # User risk settings
        user_risk_settings = {
            'max_risk_per_trade': float((user_params or self.user_params).get('max_risk_per_trade', 30)),
            'max_daily_risk': float((user_params or self.user_params).get('max_daily_risk', 100)),
            'tick_size': float((user_params or self.user_params).get('tick_size', 0.25)),
            'tick_value': float((user_params or self.user_params).get('tick_value', 1.25)),
        }
        # Bias summary placeholder
        bias_summary = (user_params or self.user_params).get('bias_summary',
                                                             "Bullish bias detected on 15min and hourly. Volatility elevated.")
        # Add suggested_contracts to each strategy
        strategy_matrix_llm_with_contracts = []
        for strat in strategy_matrix_llm:
            stop_ticks = int((user_params or self.user_params).get('stop_ticks', 10))
            tick_value = user_risk_settings['tick_value']
            stop_loss_dollars = stop_ticks * tick_value
            max_risk = user_risk_settings['max_risk_per_trade']
            suggested_contracts = int(max(1, max_risk // stop_loss_dollars)) if stop_loss_dollars > 0 else 1
            strat_with_contracts = dict(strat)
            strat_with_contracts['suggested_contracts'] = suggested_contracts
            strat_with_contracts['stop_ticks'] = stop_ticks
            strat_with_contracts['stop_loss_dollars'] = stop_loss_dollars
            strategy_matrix_llm_with_contracts.append(strat_with_contracts)
        llm_context = {
            'user_risk_settings': user_risk_settings,
            'bias_summary': bias_summary,
            'strategy_matrix': strategy_matrix_llm_with_contracts,
            'auto_contract_calculation': True
        }
        best_result['strategy_matrix_llm'] = strategy_matrix_llm
        best_result['llm_context'] = llm_context
        return llm_context, strategy_matrix_llm

    @staticmethod
    def _safe_float(val):
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    def _rank_and_pick_best_strategies(self, filtered_results_clean):
        results_sorted = sorted(filtered_results_clean,
                                key=lambda x: (self._safe_float(x.get('profit_factor')), self._safe_float(x.get('win_rate'))),
                                reverse=True)
        # Filter for risk-respecting configs
        risk_ok = [r for r in results_sorted if
                   r.get('risk_violations', 1) == 0 and r.get('daily_loss_violations', 1) == 0]
        top_strategies = (risk_ok if risk_ok else results_sorted)[:3]
        # Pick best
        best = top_strategies[0] if top_strategies else None
        return best, top_strategies

    def _filter_results_by_rules(self, max_drawdown, min_profit_factor, min_trades, results):
        filtered_results = []
        filtered_out_configs = []
        for r in results:
            required_keys = ['profit_factor', 'num_trades', 'max_drawdown']
            missing_keys = [k for k in required_keys if k not in r]
            if missing_keys:
                print(f"[DEBUG] Skipping config due to missing keys: {missing_keys}. Available keys: {list(r.keys())}")
                r['error'] = f"Missing keys: {missing_keys}"
                filtered_out_configs.append(r)
                continue
            pf = r['profit_factor']
            nt = r['num_trades']
            drawdown = r['max_drawdown']
            discard = False
            if (min_profit_factor > 0 and pf < min_profit_factor):
                discard = True
            if (min_trades > 0 and nt < min_trades):
                discard = True
            if (max_drawdown < float('inf') and drawdown > max_drawdown):
                discard = True
            if discard:
                filtered_out_configs.append(r)
            else:
                filtered_results.append(r)

        # Sort by profit factor, fallback win rate
        def safe_float(val):
            try:
                return float(val) if val is not None else 0.0
            except Exception:
                return 0.0

        # Filter out configs with None for profit_factor or win_rate, print warning
        filtered_results_clean = []
        for r in filtered_results:
            pf = r.get('profit_factor', 0.0)
            wr = r.get('win_rate', 0.0)
            if pf is None or wr is None:
                print(
                    f"[WARNING] Skipping config due to NoneType in metrics: profit_factor={pf}, win_rate={wr}, config={r}")
                filtered_out_configs.append(r)
            else:
                filtered_results_clean.append(r)
        return filtered_out_configs, filtered_results, filtered_results_clean

    def _simulate_all_configs(self, df, df_1min, df_with_preds, results, threshold_grid, total_configs, user_params):
        for i, (long_t, short_t, min_vol, bar_color) in enumerate(threshold_grid):
            # Fallback metrics always defined at the start of each iteration
            metrics = {
                'profit_factor': None,
                'avg_pnl': None,
                'avg_return': None,
                'direction_tag': None,
                'risk_violations': None,
                'daily_loss_violations': None,
                'num_trades': 0,
                'total_pnl': None,
                'max_pnl': None,
                'min_pnl': None,
                'win_rate': None,
                'max_drawdown': None
            }
            # TEMP DEV MODE: Early-stop after 5 strategy simulations for faster development. Remove or increase for full runs.
            if i >= 5:
                print("\U0001F6D1 TEMP DEV MODE: Early-stop after 5 strategy simulations.")
                break
            if regression_backtest_tracker is not None:
                if regression_backtest_tracker.get("cancel_requested"):
                    regression_backtest_tracker["status"] = "cancelled"
                    break
                regression_backtest_tracker["current"] = i + 1
            # Use a copy of user_params to avoid mutating shared state
            params_copy = (user_params or self.user_params).copy()
            params_copy['long_threshold'] = long_t
            params_copy['short_threshold'] = short_t
            params_copy['min_volume_pct_change'] = min_vol
            params_copy['bar_color_filter'] = bar_color
            # Temporarily set self.user_params for simulate_trades
            old_params = self.user_params
            self.user_params = params_copy
            # --- Ensure predicted_high and predicted_low are present ---
            df = df.copy()
            if 'predicted_high' not in df.columns or 'predicted_low' not in df.columns:
                # Compute features if needed
                df = self._compute_group_1_features(df)
                features = self.feature_names or ['fastavg', 'close_vs_ema_10', 'high_15min', 'macd',
                                                  'high_vs_ema_5_high', 'atr']
                X = df[features]
                X_scaled = self.scaler.transform(X.values)
                df['predicted_high'] = self.model_high.predict(X_scaled)
                df['predicted_low'] = self.model_low.predict(X_scaled)
            if df_with_preds is None:
                # Save the first version of df with predictions for plotting
                df_with_preds = df.copy()
            # --- Enhanced: pass config info to engine for better print ---
            sim = None
            try:
                engine = CerebroStrategyEngine(
                    df_strategy=df,
                    df_classifiers=pd.DataFrame(),
                    initial_cash=params_copy.get('initial_cash', 10000),
                    tick_size=params_copy.get('tick_size', 0.25),
                    tick_value=params_copy.get('tick_value', 1.25),
                    contract_size=params_copy.get('contract_size', 1),
                    target_ticks=params_copy.get('target_ticks', 10),
                    stop_ticks=params_copy.get('stop_ticks', 10),
                    min_dist=params_copy.get('min_dist', 3.0),
                    max_dist=params_copy.get('max_dist', 20.0),
                    min_classifier_signals=params_copy.get('min_classifier_signals', 0),
                    session_start=params_copy.get('session_start', '10:00'),
                    session_end=params_copy.get('session_end', '23:00'),
                    max_daily_profit=params_copy.get('max_daily_profit', 36.0),
                    max_daily_loss=params_copy.get('max_daily_loss', -36.0)
                )
                sim, strategy, cerebro = engine.run_backtest_RegressionScalpingStrategy(
                    df, df_1min,
                    params_copy,
                    config_index=i,
                    total_configs=total_configs,
                    long_t=long_t,
                    short_t=short_t,
                    min_vol=min_vol,
                    bar_color=bar_color
                )
                # Store the sim dict for this config
                self.all_strategy_sims.append(sim)
                # --- Extract trades from strategy if present ---
                trades = sim.get('trades', [])
                if hasattr(strategy, 'trades'):
                    trades = strategy.trades
                df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
                # --- Compute metrics ---
                if not df_trades.empty:
                    metrics = {
                        "num_trades": len(df_trades),
                        "total_pnl": df_trades["pnl"].sum(),
                        "avg_pnl": df_trades["pnl"].mean(),
                        "max_pnl": df_trades["pnl"].max(),
                        "min_pnl": df_trades["pnl"].min(),
                        "win_rate": (df_trades["pnl"] > 0).mean(),
                    }
                    gross_profit = df_trades[df_trades["pnl"] > 0]["pnl"].sum()
                    gross_loss = abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum())
                    metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else None
                    metrics["max_drawdown"] = (df_trades["pnl"].cumsum().cummax() - df_trades[
                        "pnl"].cumsum()).max() if not df_trades.empty else None
                    metrics["avg_return"] = metrics["avg_pnl"]
                    # direction_tag
                    directions = set()
                    for t in trades:
                        side = t.get('side') or t.get('direction')
                        if side:
                            directions.add(str(side).lower())
                        else:
                            pnl = t.get('pnl', 0)
                            if pnl > 0:
                                directions.add('long')
                            elif pnl < 0:
                                directions.add('short')
                    if directions == {'long'}:
                        metrics['direction_tag'] = 'long_only'
                    elif directions == {'short'}:
                        metrics['direction_tag'] = 'short_only'
                    elif 'long' in directions and 'short' in directions:
                        metrics['direction_tag'] = 'both'
                    else:
                        metrics['direction_tag'] = 'none'
                    # risk_violations and daily_loss_violations
                    max_risk_per_trade = params_copy.get('max_risk_per_trade', 15.0)
                    max_daily_risk = params_copy.get('max_daily_risk', 100.0)
                    trade_returns = [t.get('pnl', 0) for t in trades]
                    metrics['risk_violations'] = sum(abs(r) > max_risk_per_trade for r in trade_returns)
                    daily_pnl = []
                    if trades:
                        df_trades['date'] = pd.to_datetime(df_trades['entry_time']).dt.date
                        daily_pnl = df_trades.groupby('date')['pnl'].sum().values
                    metrics['daily_loss_violations'] = sum(abs(x) > max_daily_risk for x in daily_pnl)
                else:
                    print(
                        f"[WARNING] No trades or no trade signals for config: long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color}")
                    # metrics fallback already defined at top of loop
                print(f"[Debug] Trades simulated: {len(trades)}")
                if not df_trades.empty:
                    print(f"[Debug] First 3 trades: {df_trades.head(3).to_dict(orient='records')}")
                    print(f"[Debug] Sample PnLs: {df_trades['pnl'].head(3).tolist()}")
                print(f"[Debug] Computed metrics: {metrics}")
                # Store all results for this config
                self.all_strategy_results.append({
                    'config': params_copy.copy(),
                    'trades': trades,
                    'metrics': metrics,
                    'sim': sim
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(
                    f"\n❌ Backtest failed at config #{i + 1} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color})\n{tb}")
                sim = {
                    'error': f'Backtest failed at config #{i + 1} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color})\n{tb}',
                    'config': {'long_threshold': long_t, 'short_threshold': short_t, 'min_volume_pct_change': min_vol,
                               'bar_color_filter': bar_color},
                    'traceback': tb}
            self.user_params = old_params
            backtest_summary = sim.get('backtest_summary', {})
            # --- Debug: Print computed metrics ---
            print(f"[Debug] Computed metrics: {backtest_summary}")
            trades = sim.get('trades', [])
            # --- Append config and metrics to results ---
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'min_volume_pct_change': min_vol,
                'bar_color_filter': bar_color,
                'suggested_contracts': sim.get('suggested_contracts', 1),
                'stop_ticks': sim.get('stop_ticks', 10),
                'stop_loss_dollars': sim.get('stop_loss_dollars', 0),
                **metrics,
                'sim': sim
            })
            # After simulating trades for this config, calculate extra metrics
            dashboard = AnalyzerDashboard(df_with_preds, pd.DataFrame(trades))
            df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
            # Compute metrics from per-trade DataFrame
            if not df_trades.empty:
                df_trades['date'] = pd.to_datetime(df_trades['entry_time']).dt.date
                # avg_daily_pnl
                daily_pnl = df_trades.groupby('date')['pnl'].sum()
                avg_daily_pnl = daily_pnl.mean()
                # avg_trades_per_day
                avg_trades_per_day = df_trades.groupby('date').size().mean()
                # max_consec_losses
                loss_streaks = "".join(['L' if p < 0 else 'W' for p in df_trades['pnl']])
                max_consec_losses = max(map(len, loss_streaks.split('W')), default=0)
                # outlier_count_3sigma
                mean_pnl = df_trades['pnl'].mean()
                std_pnl = df_trades['pnl'].std()
                outliers = df_trades[np.abs(df_trades['pnl'] - mean_pnl) > 3 * std_pnl]
                outlier_count_3sigma = len(outliers)
            else:
                avg_daily_pnl = 0
                avg_trades_per_day = 0
                max_consec_losses = 0
                outlier_count_3sigma = 0
            metrics_llm = dashboard.calculate_strategy_metrics_for_llm(df_trades).iloc[0]
            num_trades = metrics_llm.get('num_trades', len(trades))
            total_pnl = metrics_llm.get('total_pnl', 0)
            win_rate = metrics_llm.get('win_rate', 0)
            win_rate_pct = win_rate if win_rate > 1 else win_rate * 100
            # Add new metrics to the result dict for this config
            result_row = {
                'long_threshold': long_t,
                'short_threshold': short_t,
                'min_volume_pct_change': min_vol,
                'bar_color_filter': bar_color,
                'pnl': metrics_llm['total_pnl'],
                'profit_factor': metrics_llm['profit_factor'],
                'win_rate': metrics_llm['win_rate'],
                'win_rate_%': win_rate_pct,
                'drawdown_$': metrics_llm['max_drawdown'],
                'avg_daily_pnl': avg_daily_pnl,
                'avg_trades_per_day': avg_trades_per_day,
                'max_consec_losses': max_consec_losses,
                'outlier_count_3sigma': outlier_count_3sigma,
                'num_trades': num_trades,
                'suggested_contracts': sim.get('suggested_contracts', 1),
                'stop_ticks': sim.get('stop_ticks', 10),
                'stop_loss_dollars': sim.get('stop_loss_dollars', 0),
            }
            results[-1].update(result_row)
        return df_with_preds

    def _generate_threshold_grid(self, user_params):
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4]
        results = []
        max_risk_per_trade = float((user_params or self.user_params).get('max_risk_per_trade', 15.0))
        max_daily_risk = float((user_params or self.user_params).get('max_daily_risk', 100.0))
        max_drawdown_limit = float((user_params or self.user_params).get('max_drawdown', 100.0))
        # --- Configurable rule-based filter params ---
        min_trades = int((user_params or self.user_params).get('min_trades', 0))
        max_drawdown = float((user_params or self.user_params).get('max_drawdown', float('inf')))
        min_profit_factor = float((user_params or self.user_params).get('min_profit_factor', 0.0))
        min_volume_pct_change_values = [0.0, 0.05]
        bar_color_filter_values = [True, False]
        threshold_grid = [
            (long_t, short_t, min_vol, bar_color)
            for long_t in thresholds
            for short_t in thresholds
            for min_vol in min_volume_pct_change_values
            for bar_color in bar_color_filter_values
        ]
        return max_drawdown, min_profit_factor, min_trades, results, threshold_grid

    @staticmethod
    def _rsi(series, period=7):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _summarize_trades(num_trades, win_rate, avg_return, avg_duration, max_drawdown, profit_factor):
        if num_trades == 0:
            return "No valid trades were generated by the regression predictor on this dataset."
        return (
            f"This setup yielded {num_trades} trades, win rate {win_rate*100:.1f}%, "
            f"avg return {avg_return:.2f}, avg duration {avg_duration:.1f} bars, "
            f"max drawdown {max_drawdown:.2f}, profit factor {profit_factor:.2f}."
        )

    def llm_select_top_strategies_from_grid(self, strategy_grid_df, bias_summary=None):
        """
        Use an LLM to select the top 3 strategies from the grid, considering the bias summary.
        Returns the raw JSON response from the LLM.
        """
        import json
        import datetime
        from research_agent.config import CONFIG
        print('[LLM] llm_select_top_strategies_from_grid called')
        print(f'[LLM] strategy_grid_df type: {type(strategy_grid_df)}, length: {len(strategy_grid_df) if hasattr(strategy_grid_df, "__len__") else "N/A"}')
        # Defensive: handle None or unexpected types
        if strategy_grid_df is None:
            print('[LLM] strategy_grid_df is None!')
            return {"error": "No strategy grid data provided to LLM selector."}
        # Accept DataFrame or list of dicts
        if hasattr(strategy_grid_df, 'to_dict'):
            grid_list = strategy_grid_df.to_dict(orient='records')
        elif isinstance(strategy_grid_df, list):
            grid_list = strategy_grid_df
        else:
            try:
                grid_list = list(strategy_grid_df)
            except Exception as e:
                print(f'[LLM] strategy_grid_df could not be converted to list: {e}')
                return {"error": f"Strategy grid data is not a DataFrame or list: {e}"}
        if len(grid_list) > 256:
            grid_list = grid_list[:256]
        grid_list_clean = BaseAgent.clean_for_json(grid_list)
        # Debug: Check for non-serializable objects in grid_list_clean
        for i, row in enumerate(grid_list_clean):
            for k, v in row.items():
                try:
                    json.dumps(v)
                except TypeError:
                    print(f'[LLM][DEBUG] Non-serializable value at row {i}, key "{k}": type={type(v)}, repr={repr(v)}')
        grid_list_clean = [{k: v for k, v in r.items() if k != 'sim'} for r in grid_list_clean]

        grid_json = json.dumps(grid_list_clean, indent=2)[:6000]
        # Prepare bias string
        bias_str = bias_summary if bias_summary else "[No bias summary provided. Please indicate this in your reasoning.]"
        # Define llm_prompt just before use
        llm_prompt = """
You are a trading strategist assistant. Your task is to select the best strategies from a 256-strategy grid and an optional market bias summary.

🎯 Your Goal:
Return exactly **three trading strategies**, selected based on the data, and output them in a **fixed JSON format**. This format must always be followed **exactly as shown below** to ensure consistency.

🧠 Inputs:
- A strategy grid containing 256 rows, each with metrics like: win rate, profit factor, max drawdown, avg daily PnL, etc.
- An optional market bias text summary that may include sector flow, macro bias, symbol news, or support/resistance levels.

📋 Selection Criteria:
- Choose based on meaningful metrics (not only PnL) — including stability, risk, win rate, and drawdown.
- You may weight metrics differently if market bias suggests strong long/short skew or risk-aversion.
- Use judgment to combine multiple signals into effective strategy logic.

📤 **Output Format – This must be strictly followed**:
Return your response as a JSON with the following structure:

```json
{
  "top_strategies": [
    {
      "name": "Volatile Shorts",
      "logic": "Trade when predicted short > 0.6 and candle color is red. Use only if volume spike > 5%.",
      "direction": "short",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.25,
        "win_rate": 52.3,
        "avg_daily_pnl": 5.2,
        "max_drawdown": 90.0
      },
      "rationale": "This strategy favors short trades in high-volume environments. It has stable win rate and low drawdown, making it ideal in bearish or volatile conditions."
    },
    {
      "name": "Balanced Intraday",
      "logic": "Trade when either long/short predicted > 0.5 and candle matches direction. No volume filter.",
      "direction": "both",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.18,
        "win_rate": 48.7,
        "avg_daily_pnl": 6.9,
        "max_drawdown": 100.0
      },
      "rationale": "This is a general-purpose strategy that performs well on both sides with decent stability. Recommended for trend-neutral sessions."
    },
    {
      "name": "Momentum Longs",
      "logic": "Trade when predicted long > 0.7 and min_volume_pct_change > 3%. Only on green candles.",
      "direction": "long",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.32,
        "win_rate": 56.2,
        "avg_daily_pnl": 7.5,
        "max_drawdown": 70.0
      },
      "rationale": "Best performing long-biased strategy with high win rate and consistent daily returns. Ideal for strong bullish sessions."
    }
  ]
}
```
📌 Rules:

Do not return anything outside the JSON block.

Always include exactly 3 strategies unless instructed otherwise.

Ensure keys and structure are identical in casing and order.

If market bias is empty, ignore it. If present, use it to prioritize strategy alignment.

Below is the strategy grid and bias summary:

STRATEGY GRID:
{grid_json}

BIAS SUMMARY:
{bias_str}
"""
        # Escape all literal curly braces except placeholders
        llm_prompt = llm_prompt.replace('{', '{{').replace('}', '}}').replace('{{grid_json}}', '{grid_json}').replace('{{bias_str}}', '{bias_str}')
        # Build prompt
        prompt = llm_prompt.format(grid_json=grid_json, bias_str=bias_str)
        print(f'[LLM] Prompt length: {len(prompt)} chars')
        # LLM config
        model_name = CONFIG.get("model_name", "gpt-4o")
        provider = CONFIG.get("model_provider", "openai")
        api_key = CONFIG.get("openai_api_key") if provider == "openai" else CONFIG.get("gemini_api_key")
        print(f'[LLM] Provider: {provider}, Model: {model_name}')
        # Call LLM
        try:
            if provider == "gemini":
                import requests
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                body = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f'[LLM] Raw Gemini response: {content[:500]}...')
                return json.loads(content) if content.strip().startswith('{') else content
            elif provider == "openai":
                import requests
                endpoint = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                }
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f'[LLM] Raw OpenAI response: {content[:500]}...')
                return json.loads(content) if content.strip().startswith('{') else content
            else:
                print(f'[LLM] Unknown provider: {provider}')
                return {"error": f"Unknown LLM provider: {provider}"}
        except Exception as e:
            print(f'[LLM] Exception: {e}')
            return {"error": f"LLM call failed: {e}"}
