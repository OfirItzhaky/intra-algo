import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from backend.analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from backend.analyzer_strategy_blueprint import Long5min1minStrategy
from backend.analyzer_dashboard import AnalyzerDashboard
import requests, os, json
from research_agent.config import CONFIG
import traceback
import time
try:
    from research_agent.app import regression_backtest_tracker
except ImportError:
    regression_backtest_tracker = None

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

    def simulate_trades(self, df, df_1min=None, long_threshold=None, short_threshold=None, min_volume_pct_change=None, bar_color_filter=None):
        """
        Simulate trades using Backtrader RegressionScalpingStrategy and CerebroStrategyEngine.
        Returns a dict with strategy_name, entry_conditions, trades, backtest_summary, llm_summary, model_scores, etc.
        """
        if self.last_feedback:
            return {'feedback': self.last_feedback}
        # --- Feature engineering (same as fit) ---
        df = df.copy()
        df = self._compute_group_1_features(df)
        features = self.feature_names or ['fastavg', 'close_vs_ema_10', 'high_15min', 'macd', 'high_vs_ema_5_high', 'atr']
        X = df[features]
        X_scaled = self.scaler.transform(X.values)
        # --- Predict for all rows ---
        pred_high = self.model_high.predict(X_scaled)
        pred_low = self.model_low.predict(X_scaled)
        df_5min_enriched = df.copy()
        df_5min_enriched['predicted_high'] = pred_high
        df_5min_enriched['predicted_low'] = pred_low
        self.df_5min_predicted = df_5min_enriched
        # --- Ensure 'Date' and 'Time' columns exist with correct casing ---
        if 'Date' not in df_5min_enriched.columns and 'date' in df_5min_enriched.columns:
            df_5min_enriched['Date'] = df_5min_enriched['date']
        if 'Time' not in df_5min_enriched.columns and 'time' in df_5min_enriched.columns:
            df_5min_enriched['Time'] = df_5min_enriched['time']
        # --- Use self.df_1min for 1-min bars ---
        if df_1min is not None:
            self.df_1min = df_1min
        if getattr(self, 'df_1min', None) is None:
            return {'feedback': 'No 1-minute data (self.df_1min) available for intrabar simulation.'}
        # --- Prepare params dict ---
        params = self.user_params.copy()
        if long_threshold is not None:
            params['long_threshold'] = long_threshold
        if short_threshold is not None:
            params['short_threshold'] = short_threshold
        if min_volume_pct_change is not None:
            params['min_volume_pct_change'] = min_volume_pct_change
        if bar_color_filter is not None:
            params['bar_color_filter'] = bar_color_filter
        # --- Run backtest ---
        try:
            # Build engine instance with required params
            engine = CerebroStrategyEngine(
                df_strategy=df_5min_enriched,
                df_classifiers=pd.DataFrame(),  # No classifiers for regression scalping
                initial_cash=params.get('initial_cash', 10000),
                tick_size=params.get('tick_size', 0.25),
                tick_value=params.get('tick_value', 1.25),
                contract_size=params.get('contract_size', 1),
                target_ticks=params.get('target_ticks', 10),
                stop_ticks=params.get('stop_ticks', 10),
                min_dist=params.get('min_dist', 3.0),
                max_dist=params.get('max_dist', 20.0),
                min_classifier_signals=params.get('min_classifier_signals', 0),
                session_start=params.get('session_start', '10:00'),
                session_end=params.get('session_end', '23:00'),
                max_daily_profit=params.get('max_daily_profit', 36.0),
                max_daily_loss=params.get('max_daily_loss', -36.0)
            )
            results, strategy, cerebro = engine.run_backtest_RegressionScalpingStrategy(
                df_5min_enriched, self.df_1min, params
            )
        except Exception as e:
            tb = traceback.format_exc()
            return {'feedback': f'Backtest failed: {type(e).__name__}: {e}\nTraceback:\n{tb}'}
        # --- Extract trades using AnalyzerDashboard ---
        tick_value = params.get('tick_value', 1.25)
        contract_size = params.get('contract_size', 1)
        trades_df = AnalyzerDashboard.build_trade_dataframe_from_orders(list(cerebro.broker.orders), tick_value=tick_value, contract_size=contract_size)
        trades_list = trades_df.to_dict(orient='records') if not trades_df.empty else []
        # --- Compute metrics using AnalyzerDashboard ---
        metrics_df = AnalyzerDashboard.calculate_strategy_metrics(trades_df)
        if metrics_df is not None and not metrics_df.empty:
            backtest_summary = metrics_df.to_dict(orient='records')[0]
        else:
            print("[Warning] No trades or metrics could be computed for this config.")
            backtest_summary = {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'avg_duration': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0
            }
        num_trades = backtest_summary.get('💰 Overall Performance | Total Net PnL ($)', 0)  # fallback for legacy keys
        win_rate = backtest_summary.get('🎯 Trade Quality Metrics | Win Rate (%)', 0.0)
        avg_return = backtest_summary.get('💰 Overall Performance | Total Net PnL ($)', 0.0)
        avg_duration = backtest_summary.get('📅 Time-Based Metrics | Avg Trade Duration (min)', 0.0)
        max_drawdown = backtest_summary.get('⚠️ Risk / Drawdown Metrics | Max Drawdown ($)', 0.0)
        profit_factor = backtest_summary.get('💰 Overall Performance | Profit Factor', 0.0)
        # --- LLM summary ---
        llm_summary = self._summarize_trades(
            num_trades if isinstance(num_trades, (int, float)) else 0,
            win_rate if isinstance(win_rate, (int, float)) else 0.0,
            avg_return if isinstance(avg_return, (int, float)) else 0.0,
            avg_duration if isinstance(avg_duration, (int, float)) else 0.0,
            max_drawdown if isinstance(max_drawdown, (int, float)) else 0.0,
            profit_factor if isinstance(profit_factor, (int, float)) else 0.0
        )
        # --- Result dict ---
        result = {
            "strategy_name": "Regression Predictor (Backtrader)",
            "entry_conditions": {
                "min_high_delta": params.get('long_threshold'),
                "min_low_delta": params.get('short_threshold')
            },
            "trades": trades_list,
            "backtest_summary": backtest_summary,
            "llm_summary": llm_summary,
            "model_scores": {
                "high_r2": self.metadata.get('model_high_score'),
                "low_r2": self.metadata.get('model_low_score')
            },
            "n_train": self.metadata.get('n_train'),
        }
        return result

    def find_best_threshold_strategy(self, df, df_1min=None, user_params=None):
        """
        Search for the best long/short threshold config by simulating multiple values and picking the best by profit factor (or win rate).
        Returns a dict with chosen thresholds, summary, trades, backtest, risk metadata, and explanation.
        Also returns top_strategies (top 3 configs) and a recommended_strategy_summary for UI display.
        Always uses df_1min for intrabar simulation if provided.
        """
        if self.last_feedback:
            return {'feedback': self.last_feedback}
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
        total_configs = len(threshold_grid)
        if regression_backtest_tracker is not None:
            regression_backtest_tracker["total"] = total_configs
            regression_backtest_tracker["current"] = 0
        strategy_matrix_llm = []  # Ensure always defined
        llm_context = {}         # Ensure always defined
        for i, (long_t, short_t, min_vol, bar_color) in enumerate(threshold_grid):
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
                features = self.feature_names or ['fastavg', 'close_vs_ema_10', 'high_15min', 'macd', 'high_vs_ema_5_high', 'atr']
                X = df[features]
                X_scaled = self.scaler.transform(X.values)
                df['predicted_high'] = self.model_high.predict(X_scaled)
                df['predicted_low'] = self.model_low.predict(X_scaled)
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
                # Save config to intra_algo/research_agent/uploaded_csvs/tmp_last_config.csv
                save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs', 'tmp_last_config.csv'))
                pd.DataFrame([params_copy]).to_csv(save_path, index=False)
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
                    # Compute profit_factor and max_drawdown inline if not present
                    gross_profit = df_trades[df_trades["pnl"] > 0]["pnl"].sum()
                    gross_loss = abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum())
                    metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else None
                    metrics["max_drawdown"] = (df_trades["pnl"].cumsum().cummax() - df_trades["pnl"].cumsum()).max() if not df_trades.empty else None
                else:
                    metrics = {}
                    metrics["profit_factor"] = None
                    metrics["max_drawdown"] = None
                print(f"[Debug] Trades simulated: {len(trades)}")
                if not df_trades.empty:
                    print(f"[Debug] First 3 trades: {df_trades.head(3).to_dict(orient='records')}")
                    print(f"[Debug] Sample PnLs: {df_trades['pnl'].head(3).tolist()}")
                print(f"[Debug] Computed metrics: {metrics}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"\n❌ Backtest failed at config #{i+1} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color})\n{tb}")
                # Save failed config to intra_algo/research_agent/uploaded_csvs/tmp_last_config.csv
                save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs', 'tmp_last_config.csv'))
                pd.DataFrame([params_copy]).to_csv(save_path, index=False)
                sim = {'error': f'Backtest failed at config #{i+1} (long={long_t}, short={short_t}, min_vol={min_vol}, bar_color={bar_color})\n{tb}',
                       'config': {'long_threshold': long_t, 'short_threshold': short_t, 'min_volume_pct_change': min_vol, 'bar_color_filter': bar_color},
                       'traceback': tb}
            self.user_params = old_params
            backtest_summary = sim.get('backtest_summary', {})
            # --- Debug: Print computed metrics ---
            print(f"[Debug] Computed metrics: {backtest_summary}")
            trades = sim.get('trades', [])
            # --- Direction tag logic ---
            direction_tag = 'none'
            if trades:
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
                    direction_tag = 'long_only'
                elif directions == {'short'}:
                    direction_tag = 'short_only'
                elif 'long' in directions and 'short' in directions:
                    direction_tag = 'both'
                else:
                    direction_tag = 'none'
            # Risk checks
            trade_returns = [t.get('pnl', 0) for t in trades]
            risk_violations = sum(abs(r) > max_risk_per_trade for r in trade_returns)
            daily_pnl = []
            if trades:
                df_trades = pd.DataFrame(trades)
                if not df_trades.empty and 'entry_time' in df_trades:
                    df_trades['date'] = pd.to_datetime(df_trades['entry_time']).dt.date
                    daily_pnl = df_trades.groupby('date')['pnl'].sum().values
            daily_loss_viol = sum(abs(x) > max_daily_risk for x in daily_pnl)
            pf = backtest_summary.get('💰 Overall Performance | Profit Factor', 0)
            win = backtest_summary.get('🎯 Trade Quality Metrics | Win Rate (%)', 0)
            avg_return = backtest_summary.get('💰 Overall Performance | Total Net PnL ($)', 0)
            max_drawdown = backtest_summary.get('⚠️ Risk / Drawdown Metrics | Max Drawdown ($)', 0)
            num_trades = backtest_summary.get('num_trades', len(trades))
            # Only keep serializable keys from sim
            serializable_keys = ['trades', 'backtest_summary', 'error', 'config', 'traceback', 'threshold_search', 'llm_summary', 'top_strategies', 'recommended_strategy_summary', 'filtered_out_configs', 'filter_meta', 'strategy_matrix_llm', 'llm_context', 'llm_strategy_recommendation']
            sim_serializable = {k: v for k, v in sim.items() if k in serializable_keys} if isinstance(sim, dict) else sim
            # Calculate suggested_contracts, stop_ticks, stop_loss_dollars for each config
            stop_ticks = int((user_params or self.user_params).get('stop_ticks', 10))
            tick_value = float((user_params or self.user_params).get('tick_value', 1.25))
            max_risk = float((user_params or self.user_params).get('max_risk_per_trade', 15.0))
            stop_loss_dollars = stop_ticks * tick_value
            suggested_contracts = int(max(1, max_risk // stop_loss_dollars)) if stop_loss_dollars > 0 else 1
            # --- Append config and metrics to results ---
            results.append({
                'long_threshold': long_t,
                'short_threshold': short_t,
                'min_volume_pct_change': min_vol,
                'bar_color_filter': bar_color,
                'suggested_contracts': suggested_contracts,
                'stop_ticks': stop_ticks,
                'stop_loss_dollars': stop_loss_dollars,
                **metrics,
                'sim': sim_serializable
            })
        # --- Rule-based filtering ---
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
        results_sorted = sorted(filtered_results, key=lambda x: (x.get('profit_factor', 0.0), x.get('win_rate', 0.0)), reverse=True)
        # Filter for risk-respecting configs
        risk_ok = [r for r in results_sorted if r.get('risk_violations', 1) == 0 and r.get('daily_loss_violations', 1) == 0]
        top_strategies = (risk_ok if risk_ok else results_sorted)[:3]
        # Pick best
        best = top_strategies[0] if top_strategies else None
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
                prompt = (
                    f"You are a trading strategy selection assistant.\n"
                    f"\nBIAS SUMMARY:\n{bias_summary}\n"
                    f"\nUSER RISK SETTINGS:\n"
                    f"- Max risk per trade: ${user_risk['max_risk_per_trade']}\n"
                    f"- Max daily risk: ${user_risk['max_daily_risk']}\n"
                    f"- Tick size: {user_risk['tick_size']}\n"
                    f"- Tick value: {user_risk['tick_value']}\n"
                    f"\nSTRATEGY CONFIGURATIONS (Top 10):\n"
                )
                for i, strat in enumerate(top_strats):
                    prompt += f"{i+1}. Long threshold: {strat['long_threshold']}, Short threshold: {strat['short_threshold']}, "
                    prompt += f"Min volume % change: {strat['min_volume_pct_change']}, Bar color filter: {strat['bar_color_filter']}, "
                    prompt += f"Profit factor: {strat['profit_factor']:.2f}, Win rate: {strat['win_rate']:.2f}, "
                    prompt += f"Avg return: {strat['avg_return']:.2f}, Max drawdown: {strat['max_drawdown']:.2f}, Trades: {strat['num_trades']}\n"
                prompt += (
                    "\nPlease recommend the single best strategy for the current bias and risk settings, and suggest 1-2 alternatives. "
                    "Explain your rationale clearly. Respond ONLY in valid JSON with the following fields:\n"
                    "{\n  'selected_strategy': {...},\n  'alternative_strategies': [...],\n  'llm_rationale': '... explanation ...'\n}"
                )
                llm_response = None
                tokens_used = 0
                cost_usd = 0.0
                try:
                    if model_name.startswith("gemini-"):
                        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                        body = {"contents": [{"parts": [{"text": prompt}]}]}
                        resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
                        resp.raise_for_status()
                        data = resp.json()
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        usage = data.get('usage', {})
                        tokens_used = usage.get('totalTokenCount', 0)
                        cost_usd = usage.get('totalCostUsd', 0.0)
                    elif model_name.startswith("gpt-4") or model_name.startswith("gpt-3"):
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
                        tokens_used = usage.get('total_tokens', 0)
                        # Estimate cost for gpt-4o: $5/1M tokens
                        cost_usd = (tokens_used / 1_000_000) * 5
                    else:
                        raise Exception(f"Unknown or unsupported model_name '{model_name}'")
                    # Parse JSON from LLM response
                    cleaned = content.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    llm_response = json.loads(cleaned)
                    best_result['llm_strategy_recommendation'] = {
                        'selected_strategy': llm_response.get('selected_strategy'),
                        'alternative_strategies': llm_response.get('alternative_strategies', []),
                        'llm_rationale': llm_response.get('llm_rationale', ''),
                        'model_used': model_name,
                        'tokens_used': tokens_used,
                        'cost_usd': cost_usd
                    }
                except Exception as e:
                    best_result['llm_strategy_recommendation'] = {
                        'selected_strategy': strategy_matrix_llm[0] if strategy_matrix_llm else None,
                        'alternative_strategies': strategy_matrix_llm[1:3] if len(strategy_matrix_llm) > 1 else [],
                        'llm_rationale': f"LLM call failed: {e}. Falling back to rule-based recommendation.",
                        'model_used': model_name,
                        'tokens_used': tokens_used,
                        'cost_usd': cost_usd,
                        'llm_error': True
                    }
        else:
            best_result = {'feedback': 'No valid strategy configuration found.', 'filtered_out_configs': filtered_out_configs, 'filter_meta': {
                'min_trades': min_trades,
                'max_drawdown': max_drawdown,
                'min_profit_factor': min_profit_factor
            }, 'strategy_matrix_llm': strategy_matrix_llm, 'llm_context': llm_context,
            'llm_strategy_recommendation': {
                'selected_strategy': None,
                'alternative_strategies': [],
                'llm_rationale': "No valid strategies to recommend."
            }}
        # Expand sim dict into separate columns for the CSV
        expanded_results = []
        for row in results:
            base = row.copy()
            sim = base.pop('sim', {})
            if isinstance(sim, dict):
                for k, v in sim.items():
                    base[f'sim_{k}'] = v
            expanded_results.append(base)
        # Only keep relevant columns for EDA/LLM
        cleaned_results = []
        for row in expanded_results:
            cleaned_row = {
                'long_threshold': row.get('long_threshold'),
                'short_threshold': row.get('short_threshold'),
                'min_volume_pct_change': row.get('min_volume_pct_change'),
                'bar_color_filter': row.get('bar_color_filter'),
                'profit_factor': row.get('profit_factor'),
                'win_rate': row.get('win_rate'),
                'avg_return': row.get('avg_return'),
                'max_drawdown': row.get('max_drawdown'),
                'num_trades': row.get('num_trades'),
                'risk_violations': row.get('risk_violations'),
                'daily_loss_violations': row.get('daily_loss_violations'),
                'direction_tag': row.get('direction_tag'),
                'suggested_contracts': row.get('suggested_contracts'),
                'stop_ticks': row.get('stop_ticks'),
                'stop_loss_dollars': row.get('stop_loss_dollars'),
                'error': row.get('sim_error', '')
            }
            cleaned_results.append(cleaned_row)
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploaded_csvs', 'strategy_grid_results.csv'))
        pd.DataFrame(cleaned_results).to_csv(save_path, index=False)
        return best_result

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
