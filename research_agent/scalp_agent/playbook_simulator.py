import pandas as pd
from typing import Dict, Any, Optional, List
import numpy as np

class PlaybookSimulator:
    """
    Simulator for PlaybookAgent to backtest predefined scalping strategies on historical data.

    This class accepts a pandas DataFrame and a strategy dictionary, simulates trades bar-by-bar,
    and computes a comprehensive set of performance metrics.
    """
    def simulate(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate the given strategy over the DataFrame bar-by-bar.

        Args:
            df (pd.DataFrame): Historical price and indicator data.
            strategy (Dict[str, Any]): Dictionary of strategy rules (entry_rule, stop_rule, target_rule, etc.).

        Returns:
            Dict[str, Any]: Simulation results including win_rate, avg_rr, trades_tested, expected_value,
                            avg_profit, std_profit, max_profit, max_loss, sl_hit_rate, profit_factor, and notes.
        """
        trades = []
        for i in range(1, len(df)):
            if self._check_entry(df, i, strategy.get("entry_rule")):
                entry_price = df.iloc[i]["Close"]
                stop_price = self._get_stop(df, i, strategy.get("stop_rule"))
                target_price = self._get_target(entry_price, stop_price, strategy.get("target_rule"))
                rr, profit, sl_hit = self._get_trade_result(entry_price, stop_price, target_price, df, i)
                trades.append({
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "rr": rr,
                    "profit": profit,
                    "sl_hit": sl_hit
                })
        trades_tested = len(trades)
        win_trades = [t for t in trades if t["rr"] > 0]
        loss_trades = [t for t in trades if t["rr"] < 0]
        profits = [t["profit"] for t in trades]
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        win_rate = len(win_trades) / trades_tested if trades_tested > 0 else 0.0
        avg_rr = np.mean([t["rr"] for t in trades]) if trades else 0.0
        expected_value = np.mean(profits) if profits else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        std_profit = np.std(profits) if profits else 0.0
        max_profit = max(profits) if profits else 0.0
        max_loss = min(profits) if profits else 0.0
        sl_hit_rate = sum(t["sl_hit"] for t in trades) / trades_tested if trades_tested > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        notes = f"Simulated {trades_tested} trades. Win rate: {win_rate:.2%}, Avg RR: {avg_rr:.2f}, Profit factor: {profit_factor:.2f}"
        return {
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "trades_tested": trades_tested,
            "expected_value": expected_value,
            "avg_profit": avg_profit,
            "std_profit": std_profit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "sl_hit_rate": sl_hit_rate,
            "profit_factor": profit_factor,
            "notes": notes
        }

    def _check_entry(self, df: pd.DataFrame, idx: int, rule: Optional[str]) -> bool:
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
            allowed_names = {col: df.iloc[idx][col] for col in df.columns}
            return bool(eval(rule, {"__builtins__": {}}, allowed_names))
        except Exception:
            return False

    def _get_stop(self, df: pd.DataFrame, idx: int, rule: Optional[str]) -> float:
        """
        Calculate the stop price for the trade.
        Args:
            df (pd.DataFrame): DataFrame of price/indicator data.
            idx (int): Current bar index.
            rule (Optional[str]): Stop rule as a string.
        Returns:
            float: Stop price.
        """
        if not rule or "low of" not in rule.lower():
            return df.iloc[idx]["Low"]
        import re
        match = re.search(r"low of (signal|entry|\d+(?:st|nd|rd|th)?|\d+) candle", rule.lower())
        if match:
            if match.group(1) in ["signal", "entry"]:
                return df.iloc[idx]["Low"]
            try:
                n = int(re.sub(r'\D', '', match.group(1)))
                start = max(0, idx - n + 1)
                return float(df.iloc[start:idx+1]["Low"].min())
            except Exception:
                return df.iloc[idx]["Low"]
        return df.iloc[idx]["Low"]

    def _get_target(self, entry: float, stop: float, rule: Optional[str]) -> float:
        """
        Calculate the target price for the trade.
        Args:
            entry (float): Entry price.
            stop (float): Stop price.
            rule (Optional[str]): Target rule as a string.
        Returns:
            float: Target price.
        """
        if not rule or "risk" not in rule.lower():
            return entry + (entry - stop)
        import re
        match = re.search(r"risk[\s×x\*]+([\d\.]+)", rule.lower())
        if match:
            risk_mult = float(match.group(1))
            risk = abs(entry - stop)
            return entry + risk * risk_mult
        return entry + (entry - stop)

    def _get_trade_result(self, entry: float, stop: float, target: float, df: pd.DataFrame, idx: int) -> (float, float, bool):
        """
        Calculate the realized risk-reward (RR), profit, and stop-loss hit for the trade.
        Args:
            entry (float): Entry price.
            stop (float): Stop price.
            target (float): Target price.
            df (pd.DataFrame): DataFrame of price/indicator data.
            idx (int): Entry bar index.
        Returns:
            Tuple[float, float, bool]: (RR, profit, sl_hit)
        """
        for j in range(idx+1, len(df)):
            low = df.iloc[j]["Low"]
            high = df.iloc[j]["High"]
            if low <= stop:
                rr = -1.0
                profit = stop - entry
                return rr, profit, True
            if high >= target:
                rr = 1.0
                profit = target - entry
                return rr, profit, False
        # If neither hit, assume flat (no profit/loss)
        return 0.0, 0.0, False
