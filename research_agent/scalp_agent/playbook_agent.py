"""
PlaybookAgent: A disciplined agent that selects from a set of predefined, backtested trading strategies.

This stub implementation returns a mocked analysis for demonstration and testing purposes.
"""
from typing import Dict, Any, List
from scalp_agent.input_container import InputContainer
from scalp_agent.scalp_base_agent import BaseAgent

class PlaybookAgent(BaseAgent):
    """
    An agent that applies methodical, pre-validated strategies based on the current market context.
    """
    # Strategy bank for PlaybookAgent
    STRATEGY_BANK: List[Dict[str, Any]] = [
        {
            "strategy_name": "MACD Pullback V2",
            "description": "Buy when MACD crosses above signal and price is above VWAP. Exit on RSI > 70.",
            "entry_rule": None,
            "stop_rule": None,
            "target_rule": None,
            "required_indicators": ["MACD", "VWAP", "RSI_14"],
            "label_style": "next_bar"
        },
        {
            "strategy_name": "VWAP Bounce V1",
            "description": "Long when price pulls back to VWAP and bounces with increasing volume.",
            "entry_rule": None,
            "stop_rule": None,
            "target_rule": None,
            "required_indicators": ["VWAP", "Volume"],
            "label_style": "next_bar"
        },
        {
            "strategy_name": "EMA Pullback Long",
            "description": "Enter long when price pulls back to EMA(21) and closes back above EMA(9)",
            "entry_rule": "Close > EMA_9_Close and Low <= EMA_21_Close",
            "stop_rule": "Low of the entry candle or 1.2× ATR(14)",
            "target_rule": "Risk × 2 or trail with EMA_9",
            "required_indicators": ["EMA_9_Close", "EMA_21_Close", "ATR_14", "Close"],
            "label_style": "next_bar",
            "tags": ["trend", "scalp", "pullback"],
            "complexity": "🟢 Easy"
        },
        {
            "strategy_name": "VWAP Reclaim Push",
            "description": "Long when price dips below VWAP and then closes back above it with volume confirmation.",
            "entry_rule": "Close > VWAP and Low < VWAP and Volume > Volume_Avg",
            "stop_rule": "Low of signal candle or 1.2× ATR(14)",
            "target_rule": "Risk × 2 or trail above VWAP",
            "required_indicators": ["VWAP", "Volume", "Volume_Avg", "ATR_14"],
            "label_style": "next_bar",
            "tags": ["trend", "VWAP", "volume"],
            "complexity": "🟢 Easy"
        },
        {
            "strategy_name": "RSI Pop Fade",
            "description": "Fade when RSI > 75 and a long upper wick appears; enter short on confirmation.",
            "entry_rule": "RSI_14 > 75 and UpperWickSize > BodySize",
            "stop_rule": "High of the signal candle or 1.5× ATR(14)",
            "target_rule": "Risk × 2 or exit on opposite wick",
            "required_indicators": ["RSI_14", "ATR_14", "High", "Open", "Close"],
            "label_style": "next_bar",
            "tags": ["reversal", "wick", "RSI"],
            "complexity": "🟢 Easy"
        },
        {
            "strategy_name": "Opening Range Break (5m)",
            "description": "Enter when price breaks above the high or below the low of the first 15 minutes (3×5m bars), with volume confirmation.",
            "entry_rule": "Close > OR_High and Volume > Volume_Avg (long) or Close < OR_Low and Volume > Volume_Avg (short)",
            "stop_rule": "Opposite side of the OR range (e.g., below OR_Low if long)",
            "target_rule": "Risk × 2 or midpoint of next 15m range",
            "required_indicators": ["OR_High", "OR_Low", "Volume", "Volume_Avg"],
            "label_style": "next_bar",
            "tags": ["breakout", "opening-range", "volume"],
            "complexity": "🟢 Easy"
        },
        {
            "strategy_name": "3-Bar Pullback + Confirm",
            "description": "After a visible 3-bar pullback, go long if the next candle closes green and volume increases.",
            "entry_rule": "Last 3 candles were red + current Close > Open + Volume > Volume_Avg",
            "stop_rule": "Low of the 3rd pullback candle",
            "target_rule": "Risk × 2 or EMA(9) trail",
            "required_indicators": ["Close", "Open", "Volume", "Volume_Avg"],
            "label_style": "next_bar",
            "tags": ["pattern", "pullback", "volume"],
            "complexity": "🟢 Easy"
        }
    ]

    def analyze(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the input container and user parameters to select suitable predefined strategies.

        Args:
            input_container (InputContainer): The current bundle of market and chart data for analysis.
            user_params (Dict[str, Any]): User-defined parameters to customize the agent's behavior or strategy.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - indicators (List[str]): List of indicator names used in the analysis.
                - strategies (List[Dict[str, Any]]): List of selected predefined strategy objects.
                - summary (str): Human-readable summary of the agent's analysis and recommendations.
        """
        # For now, return all strategies in the bank as viable options
        all_indicators = set()
        for strat in self.STRATEGY_BANK:
            all_indicators.update(strat.get("required_indicators", []))
        return {
            "indicators": list(all_indicators),
            "strategies": self.STRATEGY_BANK,
            "summary": "Based on current market context, several pre-validated strategies fit well, including EMA Pullback Long."
        }
