"""
AgentHandler class for managing multiple agents in the intraday dual-agent trading system.

This handler registers and coordinates both dynamic and predefined strategy agents, providing unified methods for analysis and indicator aggregation.
"""

from typing import Dict, Any, List
from scalp_agent.input_container import InputContainer
from scalp_agent.scalp_base_agent import BaseAgent
from scalp_agent.instinct_agent import InstinctAgent
from scalp_agent.multitimeframe3strategies_agent import MultiTimeframe3StrategiesAgent
from research_agent.logging_setup import get_logger

log = get_logger(__name__)
class AgentHandler:
    """
    Manages multiple trading agents and coordinates their analysis in the dual-agent system.
    """
    def __init__(self) -> None:
        """
        Initialize the AgentHandler and register all agents.
        """
        self.agents: List[BaseAgent] = [
            InstinctAgent(),
            MultiTimeframe3StrategiesAgent()
        ]

    def run_all(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the analyze method of all registered agents and collect their results.

        Args:
            input_container (InputContainer): The current bundle of market and chart data for analysis.
            user_params (Dict[str, Any]): User-defined parameters to customize agent behavior.

        Returns:
            Dict[str, Any]: A dictionary mapping agent class names to their analysis results.
        """
        results = {}
        for agent in self.agents:
            agent_name = agent.__class__.__name__
            results[agent_name] = agent.analyze(input_container, user_params)
        return results

    def get_all_indicators(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> List[str]:
        """
        Gather and deduplicate all indicator names required by all registered agents.

        This method runs each agent's analyze method, extracts the 'indicators' list from their output,
        combines all indicators into a flat list, deduplicates them, and returns the result.
        Optionally, agent results can be cached internally to avoid redundant computation.

        Args:
            input_container (InputContainer): The current bundle of market and chart data for analysis.
            user_params (Dict[str, Any]): User-defined parameters to customize agent behavior.

        Returns:
            List[str]: A deduplicated list of all indicator names used by the agents.
        """
        all_indicators = set()
        for agent in self.agents:
            result = agent.analyze(input_container, user_params)
            indicators = result.get('indicators', [])
            all_indicators.update(indicators)
        return list(all_indicators)
