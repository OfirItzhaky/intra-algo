"""
InstinctAgent: A freeform, creative agent that dynamically generates rule-based trading strategies.

This stub implementation returns a mocked analysis for demonstration and testing purposes.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from scalp_agent.input_container import InputContainer
from scalp_agent.scalp_base_agent import BaseAgent

class InstinctAgent(BaseAgent):
    """
    An agent that analyzes the chart and builds creative rule-based strategies on the fly.
    """
    def analyze(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the input container and user parameters to generate a creative, dynamic strategy.

        Args:
            input_container (InputContainer): The current bundle of market and chart data for analysis.
            user_params (Dict[str, Any]): User-defined parameters to customize the agent's behavior or strategy.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - indicators (List[str]): List of indicator names used in the analysis.
                - strategies (List[Dict[str, Any]]): List of generated strategy objects.
                - summary (str): Human-readable summary of the agent's analysis and recommendations.
        """
        return {
            "indicators": ["EMA_20", "VWAP"],
            "strategies": [
                {
                    "entry_rule": "Close > EMA_20",
                    "confirmation": "Volume > Volume_Avg * 1.5",
                    "stop_rule": "Low of last 3 candles",
                    "target_rule": "Risk * 2",
                    "label_style": "next_bar"
                }
            ],
            "summary": "Price appears to bounce off EMA_20 with strong volume — a pullback continuation setup."
        }

    class InstinctSimulator:
        """
        Internal simulator for InstinctAgent to backtest rule-based strategies on historical data.
        """
        def simulate(
            self,
            df: pd.DataFrame,
            strategy: Dict[str, str]
        ) -> Dict[str, Any]:
            """
            Simulate the given strategy over the DataFrame bar-by-bar.

            Args:
                df (pd.DataFrame): Historical price and indicator data.
                strategy (Dict[str, str]): Dictionary of strategy rules (entry, stop, target, etc.).

            Returns:
                Dict[str, Any]: Simulation results including win_rate, avg_rr, trades_tested, and notes.
            """
            trades = []
            for i in range(1, len(df)):
                if self._evaluate_entry(df, i, strategy.get("entry_rule")):
                    entry_price = df.iloc[i]["Close"]
                    stop_price = self._calculate_stop(df, i, strategy.get("stop_rule"))
                    target_price = self._calculate_target(entry_price, stop_price, strategy.get("target_rule"))
                    rr = self._calculate_rr(entry_price, stop_price, target_price, df, i)
                    win = rr >= 1.0
                    trades.append({
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "rr": rr,
                        "win": win
                    })
            trades_tested = len(trades)
            win_rate = sum(t["win"] for t in trades) / trades_tested if trades_tested > 0 else 0.0
            avg_rr = sum(t["rr"] for t in trades) / trades_tested if trades_tested > 0 else 0.0
            notes = f"Simulated {trades_tested} trades. Win rate: {win_rate:.2%}, Avg RR: {avg_rr:.2f}"
            return {
                "win_rate": win_rate,
                "avg_rr": avg_rr,
                "trades_tested": trades_tested,
                "notes": notes
            }

        def _evaluate_entry(self, df: pd.DataFrame, idx: int, rule: Optional[str]) -> bool:
            """
            Evaluate the entry rule at the given index.
            Args:
                df (pd.DataFrame): DataFrame of price/indicator data.
                idx (int): Current bar index.
                rule (Optional[str]): Entry rule as a string (e.g., 'Close > EMA_20').
            Returns:
                bool: True if entry condition is met, else False.
            """
            if not rule:
                return False
            try:
                # Only allow safe names
                allowed_names = {col: df.iloc[idx][col] for col in df.columns}
                return bool(eval(rule, {"__builtins__": {}}, allowed_names))
            except Exception:
                return False

        def _calculate_stop(self, df: pd.DataFrame, idx: int, rule: Optional[str]) -> float:
            """
            Calculate the stop price for the trade.
            Args:
                df (pd.DataFrame): DataFrame of price/indicator data.
                idx (int): Current bar index.
                rule (Optional[str]): Stop rule as a string (e.g., 'Low of last 3 candles').
            Returns:
                float: Stop price.
            """
            if not rule or "low of last" not in rule.lower():
                return df.iloc[idx]["Low"]
            import re
            match = re.search(r"last (\d+) candles", rule.lower())
            if match:
                n = int(match.group(1))
                start = max(0, idx - n + 1)
                return float(df.iloc[start:idx+1]["Low"].min())
            return df.iloc[idx]["Low"]

        def _calculate_target(self, entry: float, stop: float, rule: Optional[str]) -> float:
            """
            Calculate the target price for the trade.
            Args:
                entry (float): Entry price.
                stop (float): Stop price.
                rule (Optional[str]): Target rule as a string (e.g., 'Risk * 2').
            Returns:
                float: Target price.
            """
            if not rule or "risk" not in rule.lower():
                return entry + (entry - stop)
            import re
            match = re.search(r"risk \* ([\d\.]+)", rule.lower())
            if match:
                risk_mult = float(match.group(1))
                risk = abs(entry - stop)
                return entry + risk * risk_mult
            return entry + (entry - stop)

        def _calculate_rr(self, entry: float, stop: float, target: float, df: pd.DataFrame, idx: int) -> float:
            """
            Calculate the realized risk-reward (RR) for the trade.
            Args:
                entry (float): Entry price.
                stop (float): Stop price.
                target (float): Target price.
                df (pd.DataFrame): DataFrame of price/indicator data.
                idx (int): Entry bar index.
            Returns:
                float: Realized RR (positive for win, negative for loss).
            """
            # Simulate forward: did price hit stop or target first?
            for j in range(idx+1, len(df)):
                low = df.iloc[j]["Low"]
                high = df.iloc[j]["High"]
                if low <= stop:
                    return -1.0
                if high >= target:
                    return 1.0
            return 0.0  # Neither hit (trade still open or flat)
