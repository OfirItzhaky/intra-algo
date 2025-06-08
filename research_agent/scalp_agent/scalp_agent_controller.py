import pandas as pd
from scalp_agent.input_container import InputContainer
from scalp_agent.agent_handler import AgentHandler
from typing import Dict, Any

class ScalpAgentController:
    def __init__(self, max_risk=None, csv_file=None, chart_file=None, session_notes=None):
        self.max_risk = max_risk
        self.csv_file = csv_file
        self.chart_file = chart_file
        self.session_notes = session_notes
        # Add any additional state here as needed

    def run_dual_agent_analysis(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run dual-agent analysis using the multi-agent system.

        This method initializes the AgentHandler, runs all registered agents on the provided input container,
        and returns the combined results from both agents.

        Args:
            input_container (InputContainer): The structured input bundle for agent analysis.
            user_params (Dict[str, Any]): User-defined parameters for customizing agent behavior (e.g., max risk).

        Returns:
            Dict[str, Any]: Dictionary containing the outputs from all agents.
        """
        agent_handler = AgentHandler()
        results = agent_handler.run_all(input_container, user_params)
        return results 