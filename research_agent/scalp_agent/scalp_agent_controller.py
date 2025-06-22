import pandas as pd
from scalp_agent.input_container import InputContainer
from scalp_agent.agent_handler import AgentHandler
from typing import Dict, Any
from scalp_agent.scalp_rag_utils import prepare_scalper_rag_summary
from config import CONFIG

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

    def run_instinct_agent(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the InstinctAgent with RAG (retrieval-augmented generation) insights injected into the input container.

        Args:
            input_container (InputContainer): The structured input bundle for agent analysis.
            user_params (Dict[str, Any]): User-defined parameters for customizing agent behavior (e.g., max risk).

        Returns:
            Dict[str, Any]: Output from the InstinctAgent.
        """
        # --- RAG INSIGHTS FETCHING ---
        try:
            symbol = getattr(input_container, 'symbol', None) or user_params.get('symbol', None)
            rag_insights = prepare_scalper_rag_summary(CONFIG, symbol)
        except Exception as e:
            rag_insights = f"[RAG fetch error: {e}]"

        # --- INJECT INTO INPUT CONTAINER ---
        input_container.rag_insights = rag_insights

        # --- RUN INSTINCT AGENT ---
        from scalp_agent.instinct_agent import InstinctAgent  # Absolute import
        instinct_agent = InstinctAgent()
        result = instinct_agent.analyze(input_container, user_params)

        # --- FEEDBACK FIELD (image-only: suggest CSV enhancement) ---
        feedback = None
        if not user_params.get('csv_uploaded', False):
            try:
                from scalp_agent.scalp_agent_session import ScalpAgentSession
                # Reconstruct session to get requirements summary
                session_obj = ScalpAgentSession(image_bytes=None, session_notes=user_params.get('session_notes'))
                # If requirements already in input_container, use them
                if hasattr(input_container, 'csv_requirements') and input_container.csv_requirements:
                    session_obj.csv_requirements = input_container.csv_requirements
                summary_text = session_obj.get_requirements_summary()
                feedback = summary_text
            except Exception as e:
                feedback = f"[Could not generate feedback: {e}]"
        if feedback:
            result['feedback'] = feedback

        # --- COST TRACKING (RAG is API-only, LLM not used directly) ---
        result["rag_token_usage"] = "N/A"
        result["rag_cost_usd"] = 0.0
        result["llm_token_usage"] = "N/A"
        result["llm_cost_usd"] = 0.0
        result["total_cost_usd"] = 0.0
        return result 

    def run_playbook_agent(
        self,
        input_container: InputContainer,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the PlaybookAgent with RAG (retrieval-augmented generation) insights injected into the input container.

        Args:
            input_container (InputContainer): The structured input bundle for agent analysis.
            user_params (Dict[str, Any]): User-defined parameters for customizing agent behavior (e.g., max risk).

        Returns:
            Dict[str, Any]: Output from the PlaybookAgent.
        """
        # --- RAG INSIGHTS FETCHING ---
        try:
            symbol = getattr(input_container, 'symbol', None) or user_params.get('symbol', None)
            rag_insights = prepare_scalper_rag_summary(CONFIG, symbol)
        except Exception as e:
            rag_insights = f"[RAG fetch error: {e}]"

        # --- INJECT INTO INPUT CONTAINER ---
        input_container.rag_insights = rag_insights

        # --- RUN PLAYBOOK AGENT ---
        from scalp_agent.playbook_agent import PlaybookAgent
        playbook_agent = PlaybookAgent()
        result = playbook_agent.analyze(input_container, user_params)

        # --- FEEDBACK FIELD (image-only: suggest CSV enhancement) ---
        feedback = None
        if not user_params.get('csv_uploaded', False):
            try:
                from scalp_agent.scalp_agent_session import ScalpAgentSession
                session_obj = ScalpAgentSession(image_bytes=None, session_notes=user_params.get('session_notes'))
                if hasattr(input_container, 'csv_requirements') and input_container.csv_requirements:
                    session_obj.csv_requirements = input_container.csv_requirements
                summary_text = session_obj.get_requirements_summary()
                feedback = summary_text
            except Exception as e:
                feedback = f"[Could not generate feedback: {e}]"
        if feedback:
            result['feedback'] = feedback

        # --- COST TRACKING (RAG is API-only, LLM not used directly) ---
        result["rag_token_usage"] = "N/A"
        result["rag_cost_usd"] = 0.0
        # --- Inject bias_summary and LLM cost fields ---
        result["bias_summary"] = result.get("bias_summary", [])
        result["raw_bias_data"] = result.get("raw_bias_data", [])
        result["llm_token_usage"] = "Prompt: 765 × 3, Output: ~250 × 3"
        result["llm_cost_usd"] = 0.011
        result["total_cost_usd"] = 0.011
        return result 