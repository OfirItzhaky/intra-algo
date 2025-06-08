"""
InputContainer: Bundles all relevant inputs for a trading agent's analysis in the intraday research system.

This container provides a structured, extensible way to pass chart, symbol, and user context to any agent.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

@dataclass
class InputContainer:
    """
    Container for all inputs required by a trading agent for analysis.

    Attributes:
        symbol (str): The trading symbol (e.g., 'AAPL', 'BTCUSD').
        interval (str): The chart interval (e.g., '1m', '5m', '1h').
        last_bar_time (Optional[datetime]): Timestamp of the most recent bar/candle.
        patterns (List[str]): List of detected chart patterns.
        indicators (List[str]): List of indicator names or descriptions.
        support_resistance_zones (List[str]): List of support/resistance zone descriptions.
        rag_insights (Optional[List[str]]): Optional RAG (retrieval-augmented generation) insights.
        user_notes (Optional[str]): Optional user-provided notes or context.
    """
    symbol: str
    interval: str
    last_bar_time: Optional[datetime]
    patterns: List[str]
    indicators: List[str]
    support_resistance_zones: List[str]
    rag_insights: Optional[List[str]] = None
    user_notes: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert the InputContainer to a dictionary for LLM prompt formatting or serialization.

        Returns:
            dict: Dictionary representation of the input container.
        """
        return asdict(self)
