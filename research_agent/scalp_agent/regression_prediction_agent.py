import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from backend.analyzer_cerebro_strategy_engine import CerebroStrategyEngine
from backend.analyzer_strategy_blueprint import Long5min1minStrategy

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

    def simulate_trades(self, df, long_threshold=None, short_threshold=None):
        """
        Simulate long/short trades based on filtered predictions over historical data.
        Returns a dict with strategy_name, entry_conditions, filtered_signals, backtest_summary, llm_summary, model_scores, etc.
        """
        if self.last_feedback:
            return {'feedback': self.last_feedback}
        # --- Feature engineering (same as fit) ---
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(10).std()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['rsi_7'] = self._rsi(df['close'], 7)
        df = df.dropna().reset_index(drop=True)
        # --- Targets ---
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        # --- Features ---
        features = self.feature_names or ['close', 'return', 'volatility', 'ma_5', 'ma_10', 'rsi_7']
        X = df[features]
        X_scaled = self.scaler.transform(X.values)
        # --- Predict for all rows ---
        pred_high = self.model_high.predict(X_scaled)
        pred_low = self.model_low.predict(X_scaled)
        current_close = df['close'].values
        if long_threshold is None:
            long_threshold = float(self.user_params.get('long_threshold', 0.5))
        if short_threshold is None:
            short_threshold = float(self.user_params.get('short_threshold', 0.5))
        # --- Signal logic ---
        long_signals = (pred_high - current_close) > long_threshold
        short_signals = (current_close - pred_low) > short_threshold
        # --- Simulate trades ---
        trades = []
        for i in range(len(df) - 1):  # -1 because we use next bar for outcome
            entry_time = df.index[i]
            entry_price = current_close[i]
            # Long trade simulation
            if long_signals[i]:
                target = pred_high[i]
                actual_high = df['high'].iloc[i+1]
                ret = actual_high - entry_price
                win = actual_high >= target
                trades.append({
                    'timestamp': df.index[i],
                    'side': 'long',
                    'entry': entry_price,
                    'target': target,
                    'actual': actual_high,
                    'return': ret,
                    'win': win,
                    'duration': 1
                })
            # Short trade simulation
            if short_signals[i]:
                target = pred_low[i]
                actual_low = df['low'].iloc[i+1]
                ret = entry_price - actual_low
                win = actual_low <= target
                trades.append({
                    'timestamp': df.index[i],
                    'side': 'short',
                    'entry': entry_price,
                    'target': target,
                    'actual': actual_low,
                    'return': ret,
                    'win': win,
                    'duration': 1
                })
        # --- Metrics ---
        num_trades = len(trades)
        wins = sum(t['win'] for t in trades)
        win_rate = (wins / num_trades) if num_trades > 0 else 0.0
        avg_return = np.mean([t['return'] for t in trades]) if trades else 0.0
        avg_duration = np.mean([t['duration'] for t in trades]) if trades else 0.0
        # Max drawdown calculation
        equity_curve = np.cumsum([t['return'] for t in trades]) if trades else np.array([0.0])
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = running_max - equity_curve
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        # Profit factor
        gross_profit = sum(t['return'] for t in trades if t['return'] > 0)
        gross_loss = -sum(t['return'] for t in trades if t['return'] < 0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        # Filtered signals (timestamps)
        filtered_signals = [t['timestamp'] for t in trades]
        # --- LLM summary ---
        llm_summary = self._summarize_trades(num_trades, win_rate, avg_return, avg_duration, max_drawdown, profit_factor)
        # --- Result dict ---
        result = {
            "strategy_name": "Regression Predictor",
            "entry_conditions": {
                "min_high_delta": long_threshold,
                "min_low_delta": short_threshold
            },
            "filtered_signals": filtered_signals,
            "backtest_summary": {
                "num_trades": num_trades,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_duration": avg_duration,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor
            },
            "llm_summary": llm_summary,
            "model_scores": {
                "high_r2": self.metadata.get('model_high_score'),
                "low_r2": self.metadata.get('model_low_score')
            },
            "n_train": self.metadata.get('n_train'),
        }
        return result

    def find_best_threshold_strategy(self, df, user_params=None):
        """
        Search for the best long/short threshold config by simulating multiple values and picking the best by profit factor (or win rate).
        Respects user risk settings (max_risk_per_trade, max_daily_risk).
        Returns a dict with chosen thresholds, summary, trades, backtest, risk metadata, and explanation.
        Also returns top_strategies (top 3 configs) and a recommended_strategy_summary for UI display.
        """
        if self.last_feedback:
            return {'feedback': self.last_feedback}
        thresholds = [0.5, 1.0, 1.5, 2.0,2.5,3,3.5,4]
        results = []
        max_risk_per_trade = float((user_params or self.user_params).get('max_risk_per_trade', 15.0))
        max_daily_risk = float((user_params or self.user_params).get('max_daily_risk', 100.0))
        for long_t in thresholds:
            for short_t in thresholds:
                sim = self.simulate_trades(df, long_threshold=long_t, short_threshold=short_t)
                trades = sim.get('filtered_signals', [])
                summary = sim.get('backtest_summary', {})
                # Risk checks
                trade_returns = [t['return'] for t in sim.get('trades', [])] if 'trades' in sim else []
                risk_violations = sum(abs(r) > max_risk_per_trade for r in trade_returns)
                daily_pnl = []
                if 'trades' in sim:
                    df_trades = pd.DataFrame(sim['trades'])
                    if not df_trades.empty and 'timestamp' in df_trades:
                        df_trades['date'] = pd.to_datetime(df_trades['timestamp']).dt.date
                        daily_pnl = df_trades.groupby('date')['return'].sum().values
                daily_loss_viol = sum(abs(x) > max_daily_risk for x in daily_pnl)
                pf = summary.get('profit_factor', 0)
                win = summary.get('win_rate', 0)
                results.append({
                    'long_threshold': long_t,
                    'short_threshold': short_t,
                    'profit_factor': pf,
                    'win_rate': win,
                    'avg_return': summary.get('avg_return', 0),
                    'max_drawdown': summary.get('max_drawdown', 0),
                    'num_trades': summary.get('num_trades', 0),
                    'risk_violations': risk_violations,
                    'daily_loss_violations': daily_loss_viol,
                    'sim': sim
                })
        # Sort by profit factor, fallback win rate
        results_sorted = sorted(results, key=lambda x: (x['profit_factor'], x['win_rate']), reverse=True)
        # Filter for risk-respecting configs
        risk_ok = [r for r in results_sorted if r['risk_violations'] == 0 and r['daily_loss_violations'] == 0]
        top_strategies = (risk_ok if risk_ok else results_sorted)[:3]
        # Pick best
        best = top_strategies[0] if top_strategies else None
        best_result = best['sim'] if best else {'feedback': 'No valid strategy configuration found.'}
        # Compose explanation
        if best:
            explanation = (
                f"Best thresholds: long_delta={best['long_threshold']}, short_delta={best['short_threshold']}. "
                f"Profit factor: {best['profit_factor']:.2f}, "
                f"Win rate: {best['win_rate']*100:.1f}%. "
                f"Risk violations: {best['risk_violations']}, "
                f"Daily loss violations: {best['daily_loss_violations']}."
            )
            best_result['threshold_search'] = {
                'chosen_long_threshold': best['long_threshold'],
                'chosen_short_threshold': best['short_threshold'],
                'risk_meta': {
                    'risk_violations': best['risk_violations'],
                    'daily_loss_violations': best['daily_loss_violations']
                },
                'explanation': explanation
            }
            best_result['llm_summary'] = best_result.get('llm_summary', '') + ' ' + explanation
            # Add top_strategies and recommendation
            best_result['top_strategies'] = [
                {
                    'long_threshold': r['long_threshold'],
                    'short_threshold': r['short_threshold'],
                    'profit_factor': r['profit_factor'],
                    'win_rate': r['win_rate'],
                    'avg_return': r['avg_return'],
                    'max_drawdown': r['max_drawdown'],
                    'num_trades': r['num_trades'],
                    'risk_violations': r['risk_violations'],
                    'daily_loss_violations': r['daily_loss_violations']
                } for r in top_strategies
            ]
            best_result['recommended_strategy_summary'] = (
                f"Strategy with long threshold {best['long_threshold']} and short threshold {best['short_threshold']} "
                f"yielded the highest profit factor ({best['profit_factor']:.2f}) and stayed within risk limits."
            )
        else:
            best_result = {'feedback': 'No valid strategy configuration found.'}
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
