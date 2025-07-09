import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from backend.analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from backend.analyzer_dashboard import AnalyzerDashboard
import requests, os, json
import requests

from research_agent.config import CONFIG
import traceback
import time
import math  # Ensure math is imported for isinf/isnan
import re
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
        config = result['config']
        # Apply cross-comparison filters if present
        max_pred_low = config.get('max_predicted_low_for_long', None)
        min_pred_high = config.get('min_predicted_high_for_short', None)
        if max_pred_low is not None or min_pred_high is not None:
            filtered_trades = []
            for t in trades:
                # Only filter if the relevant fields are present in the trade
                if t.get('side', '').lower() == 'long' and max_pred_low is not None:
                    if t.get('predicted_low', float('-inf')) > max_pred_low:
                        continue  # skip this long trade
                if t.get('side', '').lower() == 'short' and min_pred_high is not None:
                    if t.get('predicted_high', float('inf')) < min_pred_high:
                        continue  # skip this short trade
                filtered_trades.append(t)
            trades = filtered_trades
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

    def _make_serializable(self, obj):
        def helper(o):
            basic_types = (str, int, float, bool, type(None))
            if isinstance(o, dict):
                return {k: helper(v) for k, v in o.items()
                        if k not in ['strategy', 'cerebro'] and isinstance(v, (basic_types + (dict, list)))}
            elif isinstance(o, list):
                return [helper(v) for v in o]
            elif isinstance(o, basic_types):
                if isinstance(o, float):
                    if math.isinf(o) or math.isnan(o):
                        return None
                return o
            else:
                return str(o)

        return helper(obj)



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
        # --- Get diverse strategies instead of rule-based filtering ---

        diverse_strategies = self.pick_top_n_strategies_mixed(pd.DataFrame(results))





        # Now call LLM selector
        print('[LLM] Calling LLM for top strategy selection...')
        bias_summary = llm_context.get('bias_summary')
        final_best_result = self._call_llm_and_attach(diverse_strategies, bias_summary)

        best_result = self.extract_best_result_from_top_3(final_best_result)

        df_results = self._save_and_visualize_results(best_result, df_with_preds, results)

        # --- Ensure result is serializable before returning ---
        best_result_serializable = self._make_serializable(final_best_result)

        if 'llm_top_strategies' in best_result_serializable:
            best_result_serializable['llm_top_strategies'] = parse_llm_response(best_result_serializable['llm_top_strategies'])
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
        import numpy as np
        df = pd.DataFrame(results)
        required_cols = ['long_threshold', 'short_threshold', 'min_volume_pct_change', 'bar_color_filter', 'win_rate', 'avg_pnl']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARNING] Cannot plot heatmaps, missing columns: {missing_cols}")
        else:
            min_vols = sorted(df['min_volume_pct_change'].dropna().unique())
            bar_colors = sorted(df['bar_color_filter'].dropna().unique())
            n_rows = len(bar_colors)
            n_cols = len(min_vols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
            for i, min_vol in enumerate(min_vols):
                for j, bar_color in enumerate(bar_colors):
                    ax = axes[j, i]
                    sub = df[(df['min_volume_pct_change'] == min_vol) & (df['bar_color_filter'] == bar_color)]
                    if sub.empty:
                        ax.set_title(f"min_vol={min_vol}, bar_color={bar_color}\n(No data)")
                        ax.axis('off')
                        continue
                    pivot = sub.pivot_table(index='long_threshold', columns='short_threshold', values='win_rate', aggfunc='mean')
                    if pivot.isnull().all().all():
                        # fallback to avg_pnl
                        pivot = sub.pivot_table(index='long_threshold', columns='short_threshold', values='avg_pnl', aggfunc='mean')
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

    def extract_best_result_from_top_3(self, llm_output: dict) -> dict:
        """
        Extract the best strategy result from LLM output by matching the top strategy
        with the corresponding result from self.all_strategy_results.

        Args:
            llm_output: Dictionary containing LLM response with top_strategies

        Returns:
            Dictionary with config, trades, and top_strategies for _save_and_visualize_results
        """
        if not llm_output or 'llm_top_strategies' not in llm_output:
            print("[extract_best_result_from_top_3] No LLM output or top_strategies found")
            return {}

        import json
        import re

        llm_top = llm_output['llm_top_strategies']
        raw_json_text = llm_top.get('llm_raw_response', '')

        if raw_json_text:
            cleaned = re.sub(r'^```json\s*|\s*```$', '', raw_json_text.strip(), flags=re.DOTALL)
            try:
                parsed = json.loads(cleaned)
                top_strategies = parsed.get("top_strategies", [])
            except json.JSONDecodeError as e:
                print(f"[extract_best_result_from_top_3] JSON decode failed: {e}")
                return {}
        else:
            top_strategies = llm_top.get("top_strategies", [])

        if not top_strategies or len(top_strategies) == 0:
            print("[extract_best_result_from_top_3] No top strategies found in LLM output")
            return {}

        top_strategy = top_strategies[0]
        print(f"[extract_best_result_from_top_3] Looking for match for top strategy: {top_strategy}")

        # Strategy matching logic
        matched_result = None
        for result in self.all_strategy_results:
            config = result.get('config', {})
            if matched_result is None:
                matched_result = result
            else:
                current_metrics = result.get('metrics', {})
                best_metrics = matched_result.get('metrics', {})
                if (
                        current_metrics.get('profit_factor', 0) > best_metrics.get('profit_factor', 0) or
                        (
                                current_metrics.get('profit_factor', 0) == best_metrics.get('profit_factor', 0) and
                                current_metrics.get('win_rate', 0) > best_metrics.get('win_rate', 0)
                        )
                ):
                    matched_result = result

        if matched_result is None:
            print("[extract_best_result_from_top_3] No matching strategy found in all_strategy_results")
            return {}

        print(f"[extract_best_result_from_top_3] Matched strategy with config: {matched_result.get('config', {})}")

        return {
            'config': matched_result['config'],
            'trades': matched_result['trades'],
            'top_strategies': top_strategies,
            'llm_top_strategies': llm_top,
            'llm_cost_usd': llm_output.get('llm_cost_usd'),
            'llm_token_usage': llm_output.get('llm_token_usage'),
            'model_name': llm_output.get('model_name'),
            'provider': llm_output.get('provider'),
            'bias_summary_used': llm_output.get('bias_summary_used'),
            'strategy_matrix_llm': llm_output.get('strategy_matrix_llm'),
        }



    def _call_llm_and_attach(self, diverse_strategies, bias_summary
                             ):
        # Serialize the diverse strategy grid
        try:
            # Drop problematic fields before serializing
            df_clean = diverse_strategies.drop(columns=['sim', 'strategy', 'cerebro'], errors='ignore')
            grid_json = df_clean.to_json(orient='records', indent=2)
        except Exception as e:
            print(f"[LLM] Failed to serialize strategies: {e}")
            return {"error": "Could not prepare strategy input for LLM."}

        # Construct the prompt
        llm_prompt = f"""
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
    Return your response as a JSON with the following structure (starting directly with {{):

    {{
      "top_strategies": [
        {{
          "name": "Example Strategy",
          "logic": "Trade when conditions are met.",
          "direction": "both",
          "stop_loss_ticks": 10,
          "take_profit_ticks": 10,
          "key_metrics": {{
            "profit_factor": 1.0,
            "win_rate": 50.0,
            "avg_daily_pnl": 5.0,
            "max_drawdown": 100.0
          }},
          "rationale": "Explanation of why this strategy is selected."
        }},
        ...
      ]
    }}
    📌 Rules:
    - Output must be clean JSON parseable by json.loads()
    - No markdown formatting, no triple backticks
    - Must always include exactly 3 strategies unless told otherwise

    STRATEGY GRID:
    {grid_json}

    BIAS SUMMARY:
    {bias_summary}
    """

        model_name = CONFIG.get("model_name")
        provider = CONFIG.get("model_provider")
        # Call the appropriate LLM API
        try:
            if provider == "google":
                import requests
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": CONFIG.get("gemini_api_key")}
                body = {"contents": [{"parts": [{"text": llm_prompt}]}]}
                resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})

                prompt_tokens = usage['promptTokenCount']
                candidate_tokens = usage['candidatesTokenCount']
                tokens = prompt_tokens + candidate_tokens
                cost = round(tokens * 0.0025 / 1000, 4)

            elif provider == "openai":
                import requests
                endpoint = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {CONFIG.get('openai_api_key')}",
                           "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": llm_prompt}],
                    "max_tokens": 1000
                }
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                cost = round((usage.get("prompt_tokens", 0) * 0.005 + usage.get("completion_tokens", 0) * 0.015) / 1000,
                             4)

            else:
                return {"error": f"Unsupported LLM provider: {provider}"}

            print(f"[LLM Cost] ${cost} ({tokens} tokens from {provider})")
            print(f"[LLM Response Preview] {content[:300]}...")

            parsed = json.loads(content) if content.strip().startswith('{') else {"llm_raw_response": content}
            best_result = {
                "llm_top_strategies": parsed,
                "llm_cost_usd": cost,
                "llm_token_usage": tokens,
                "model_name": model_name,
                "provider": provider,
                "bias_summary_used": bias_summary,
                "strategy_matrix_llm": diverse_strategies,
            }

            return best_result

        except Exception as e:
            print(f"[LLM Error] {e}")
            return {"error": str(e), "llm_prompt": llm_prompt}

    @staticmethod
    def _safe_float(val):
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0



    def _simulate_all_configs(self, df, df_1min, df_with_preds, results, threshold_grid, total_configs, user_params):
        for i, (long_t, short_t, min_vol, bar_color, max_pred_low, min_pred_high) in enumerate(threshold_grid):
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
            if i >= 15:
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
            params_copy['max_predicted_low_for_long'] = max_pred_low
            params_copy['min_predicted_high_for_short'] = min_pred_high
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
                    bar_color=bar_color,
                    max_pred_low=max_pred_low,
                    min_pred_high=min_pred_high
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
                # print(f"[Debug] Trades simulated: {len(trades)}")
                # if not df_trades.empty:
                #     print(f"[Debug] First 3 trades: {df_trades.head(3).to_dict(orient='records')}")
                #     print(f"[Debug] Sample PnLs: {df_trades['pnl'].head(3).tolist()}")
                # print(f"[Debug] Computed metrics: {metrics}")
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
            # print(f"[Debug] Computed metrics: {backtest_summary}")
            trades = sim.get('trades', [])
            # --- Append config and metrics to results ---
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'min_volume_pct_change': min_vol,
                'bar_color_filter': bar_color,
                'max_predicted_low_for_long': max_pred_low,
                'min_predicted_high_for_short': min_pred_high,
                'suggested_contracts': sim.get('suggested_contracts', 1),
                'stop_ticks': sim.get('stop_ticks', 10),
                'stop_loss_dollars': sim.get('stop_loss_dollars', 0),
                **metrics,
                'sim': sim
            })
            # After simulating trades for this config, calculate extra metrics
            dashboard = AnalyzerDashboard(df_with_preds, pd.DataFrame(trades))
            df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
            # After filtering trades for this config, print debug info
            # print(f"[DEBUG] Config: long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color}, max_pred_low={max_pred_low}, min_pred_high={min_pred_high} | Num trades after filtering: {len(trades)}")
            # if trades:
            #     print(f"[DEBUG] First trade: {trades[0]}")
            # Compute metrics from per-trade DataFrame
            # print(f"[DEBUG] df_trades columns: {df_trades.columns.tolist()} | empty: {df_trades.empty}")
            if not df_trades.empty and 'pnl' in df_trades.columns:
                metrics_llm = dashboard.calculate_strategy_metrics_for_llm(df_trades).iloc[0]
                # Also define these for later use
                avg_daily_pnl = metrics_llm.get('avg_daily_pnl', 0)
                avg_trades_per_day = metrics_llm.get('avg_trades_per_day', 0)
                max_consec_losses = metrics_llm.get('max_consec_losses', 0)
                outlier_count_3sigma = metrics_llm.get('outlier_count_3sigma', 0)
            else:
                print(f"[WARNING] Skipping metrics calculation: df_trades empty or missing 'pnl'.")
                metrics_llm = {
                    'total_pnl': 0,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'max_drawdown': 0,
                    'drawdown_pct': 0,
                    'avg_daily_pnl': 0,
                    'avg_trades_per_day': 0,
                    'max_consec_losses': 0,
                    'outlier_count_3sigma': 0,
                    'total_net_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'win_loss_ratio': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'num_winning_days': 0,
                    'num_losing_days': 0
                }
                avg_daily_pnl = 0
                avg_trades_per_day = 0
                max_consec_losses = 0
                outlier_count_3sigma = 0
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
                'max_predicted_low_for_long': max_pred_low,
                'min_predicted_high_for_short': min_pred_high,
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

    def pick_top_n_strategies_mixed(self, df_results, n_per_slice=5):
        """
        Pick top strategies using multiple slicing criteria to ensure diversity.
        
        Slicing Categories (44 total slices):
        1. Performance Metrics (16 slices):
           - Top by profit_factor (both, longs only, shorts only)
           - Top by win_rate (both, longs only, shorts only)
           - Top by avg_daily_pnl (both, longs only, shorts only)
           - Top by Sharpe-like ratio (pnl/drawdown)
           - Lowest drawdown with profit_factor > 1.2
           - Highest num_trades with win_rate > 50%
           - Best performing strategies with bar_color_filter=True/False
           - Best performing strategies with min_volume_pct_change > 0
        
        2. Risk Metrics (12 slices):
           - Lowest max_consec_losses
           - Lowest outlier_count_3sigma
           - Best risk-adjusted returns (various combinations)
           - Lowest daily_loss_violations
           - Lowest risk_violations
           - Best performing with stop_loss_dollars < median
        
        3. Trade Characteristics (8 slices):
           - Best avg_trade_duration (shortest profitable trades)
           - Most consistent daily_pnl (lowest std dev)
           - Highest avg_trades_per_day with profit_factor > 1
           - Best performing in different volatility regimes
        
        4. Hybrid Metrics (8 slices):
           - Combined profit_factor + win_rate score
           - Combined profit_factor + num_trades score
           - Combined win_rate + avg_daily_pnl score
           - Combined low drawdown + high profit_factor score
        
        Args:
            df_results: DataFrame containing all strategy results
            n_per_slice: Number of top strategies to keep per slice (default 3)
            
        Returns:
            DataFrame with diverse set of top strategies, deduplicated
        """

        if df_results is None or len(df_results) == 0:
            print("[Strategy Picker] Warning: Empty results dataframe")
            return df_results
            
        try:
            # Create a copy to avoid modifying original
            df = df_results.copy()
            all_top_strategies = []
            
            # Helper for safe sorting when NaN present
            def safe_sort(df, column, ascending=False):
                if isinstance(df, pd.DataFrame):
                    return df.sort_values(column, ascending=ascending, na_position='last')
                else:
                    sorted = pd.DataFrame(df)
                    return sorted.sort_values(column, ascending=ascending, na_position='last')
            
            # 1. Performance Metrics Slices
            # Profit Factor slices
            all_top_strategies.extend([
                safe_sort(df, 'profit_factor').head(n_per_slice),  # Overall best
                safe_sort(df[df['direction_tag'] == 'long_only'], 'profit_factor').head(n_per_slice),  # Long only
                safe_sort(df[df['direction_tag'] == 'short_only'], 'profit_factor').head(n_per_slice),  # Short only
            ])
            
            # Win Rate slices
            all_top_strategies.extend([
                safe_sort(df, 'win_rate').head(n_per_slice),
                safe_sort(df[df['direction_tag'] == 'long_only'], 'win_rate').head(n_per_slice),
                safe_sort(df[df['direction_tag'] == 'short_only'], 'win_rate').head(n_per_slice),
            ])
            
            # PnL and Trade metrics
            df['pnl_drawdown_ratio'] = df['total_pnl'] / df['max_drawdown'].replace(0, np.inf)
            all_top_strategies.extend([
                safe_sort(df, 'avg_daily_pnl').head(n_per_slice),
                safe_sort(df, 'pnl_drawdown_ratio').head(n_per_slice),
                safe_sort(df[df['profit_factor'] > 1.2], 'max_drawdown', ascending=True).head(n_per_slice),
                safe_sort(df[df['win_rate'] > 0.5], 'num_trades').head(n_per_slice)
            ])
            
            # Filter-based best performers
            all_top_strategies.extend([
                safe_sort(df[df['bar_color_filter'] == True], 'profit_factor').head(n_per_slice),
                safe_sort(df[df['bar_color_filter'] == False], 'profit_factor').head(n_per_slice),
                safe_sort(df[df['min_volume_pct_change'] > 0], 'profit_factor').head(n_per_slice)
            ])
            
            # 2. Risk Metrics Slices
            all_top_strategies.extend([
                safe_sort(df, 'max_consec_losses', ascending=True).head(n_per_slice),
                safe_sort(df, 'outlier_count_3sigma', ascending=True).head(n_per_slice),
                safe_sort(df, 'daily_loss_violations', ascending=True).head(n_per_slice),
                safe_sort(df, 'risk_violations', ascending=True).head(n_per_slice)
            ])
            
            # Risk-adjusted metrics
            df['risk_adjusted_pnl'] = df['avg_daily_pnl'] / (df['max_drawdown'].replace(0, 0.01))
            all_top_strategies.extend([
                safe_sort(df, 'risk_adjusted_pnl').head(n_per_slice)
            ])
            
            # 3. Trade Characteristics
            df['consistency_score'] = df['avg_daily_pnl'] / (df['drawdown_$'].abs().replace(0, 0.01))
            all_top_strategies.extend([
                safe_sort(df[df['profit_factor'] > 1], 'avg_trades_per_day').head(n_per_slice),
                safe_sort(df, 'consistency_score').head(n_per_slice)
            ])
            
            # 4. Hybrid Metrics
            df['profit_win_score'] = df['profit_factor'] * df['win_rate']
            df['profit_trades_score'] = df['profit_factor'] * np.log1p(df['num_trades'])
            df['win_pnl_score'] = df['win_rate'] * df['avg_daily_pnl']
            all_top_strategies.extend([
                safe_sort(df, 'profit_win_score').head(n_per_slice),
                safe_sort(df, 'profit_trades_score').head(n_per_slice),
                safe_sort(df, 'win_pnl_score').head(n_per_slice)
            ])
            
            # Combine all slices and drop duplicates
            combined_df = pd.concat(all_top_strategies)
            key_fields = ['long_threshold', 'short_threshold', 'bar_color_filter', 'min_volume_pct_change']
            combined_df.drop_duplicates(subset=key_fields,inplace=True)
            
            print(f"[Strategy Picker] Selected {len(combined_df)} unique strategies from {len(all_top_strategies) * n_per_slice} candidates")
            
            return combined_df
            
        except Exception as e:
            print(f"[Strategy Picker] Error filtering strategies: {e}")
            print(f"[Strategy Picker] Full error: {traceback.format_exc()}")
            return df_results

    def _generate_threshold_grid(self, user_params):
        thresholds = [1.0, 1.5, 2.0, 2.5, 3, 3.5]
        results = []
        max_risk_per_trade = float((user_params or self.user_params).get('max_risk_per_trade', 15.0))
        max_daily_risk = float((user_params or self.user_params).get('max_daily_risk', 100.0))
        max_drawdown_limit = float((user_params or self.user_params).get('max_drawdown', 100.0))
        # --- Configurable rule-based filter params ---
        min_trades = int((user_params or self.user_params).get('min_trades', 0))
        max_drawdown = float((user_params or self.user_params).get('max_drawdown', float('inf')))
        min_profit_factor = float((user_params or self.user_params).get('min_profit_factor', 0.0))
        min_volume_pct_change_values = [0.0, 0.5]  # rethink with other volume factors...
        bar_color_filter_values = [True, False]
        cross_low_values = [1, 2, 3]  # max_predicted_low_for_long
        cross_high_values = [1, 2, 3] # min_predicted_high_for_short
        threshold_grid = [
            (long_t, short_t, min_vol, bar_color, max_pred_low, min_pred_high)
            for long_t in thresholds
            for short_t in thresholds
            for min_vol in min_volume_pct_change_values
            for bar_color in bar_color_filter_values
            for max_pred_low in cross_low_values
            for min_pred_high in cross_high_values
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

📌 Rules:

Do not return anything outside the JSON block.

Return the output as a clean JSON object (not markdown, not wrapped in triple backticks). Your response should start directly with { and be parseable using json.loads() in Python. Do not include extra text, formatting, or explanation

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
            if provider == "google":
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                body = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                # Get model version from response
                model_version = data.get('modelVersion', model_name)
                # Get usage data
                usage = data.get('usageMetadata')
                total_tokens = usage.get('totalTokenCount')
                cost_gemini = total_tokens * 0.0025 / 1000  # flat $0.0025 / 1K
                cost_usd = round(float(cost_gemini), 4)
                self.last_llm_cost_usd = cost_usd
                print(f"[LLM Cost] ${cost_usd} (provider: gemini)")
                print(f"[LLM Tokens] {total_tokens} tokens (provider: gemini)")
                print(f"[LLM Raw Response] {content[:500]}...")
                result_parsed = safe_parse_json(content)
                if isinstance(result_parsed, dict):
                    result_parsed['llm_cost_usd'] = cost_usd
                    result_parsed['model_name'] = model_version
                return result_parsed
            elif provider == "openai":
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
                usage = data.get('usage', {})
                total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                cost_openai = (usage.get('prompt_tokens', 0) * 0.005 + usage.get('completion_tokens', 0) * 0.015) / 1000  # $/1K tokens
                cost_usd = round(float(cost_openai), 4)
                self.last_llm_cost_usd = cost_usd
                print(f"[LLM Cost] ${cost_usd} (provider: openai)")
                print(f"[LLM Tokens] {total_tokens} tokens (provider: openai)")
                print(f"[LLM Raw Response] {content[:500]}...")
                result_parsed = safe_parse_json(content)
                if isinstance(result_parsed, dict):
                    result_parsed['llm_cost_usd'] = cost_usd
                    result_parsed['model_name'] = model_name
                    return result_parsed
                else:
                    print(f'[LLM] Unknown provider: {provider}')
                return {"error": f"Unknown LLM provider: {provider}"}
        except Exception as e:
                print(f'[LLM] Exception: {e}')
                return {"error": f"LLM call failed: {e}"}

def _strip_json_markdown_fence(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
def safe_parse_json(content):
    try:
        # Strip markdown fences like ```json ... ```
        content_clean = re.sub(r"^```json\s*|```$", "", content.strip(), flags=re.IGNORECASE)
        return json.loads(content_clean)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {e}")
        return content  # fallback to raw string

def parse_llm_response(raw):
    """
    Accepts a string or a dict (possibly with 'llm_raw_response').
    Cleans markdown-style wrapping, parses JSON, and returns the list at data['top_strategies'].
    Always returns a list (empty if parsing fails or 'top_strategies' is missing).
    Prints the result for debugging.
    """
    import re, json
    result = []
    # If it's a dict with 'llm_raw_response', parse that recursively
    if isinstance(raw, dict) and 'llm_raw_response' in raw:
        result = parse_llm_response(raw['llm_raw_response'])
    # If it's a string, clean and parse
    elif isinstance(raw, str):
        cleaned = re.sub(r'^```json\s*|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and 'top_strategies' in data and isinstance(data['top_strategies'], list):
                result = data['top_strategies']
        except Exception as e:
            print(f"[parse_llm_response] Failed to parse or extract top_strategies: {e}")
    # If it's already a dict with 'top_strategies'
    elif isinstance(raw, dict) and 'top_strategies' in raw and isinstance(raw['top_strategies'], list):
        result = raw['top_strategies']
    print(f"[parse_llm_response] Returning top_strategies: {result}")
    return result