from .scalp_agent_session import ScalpAgentSession
from .scalp_base_agent import BaseAgent
import os
import requests
import json
from research_agent.config import CONFIG
from logging_setup import get_logger

log = get_logger(__name__)

class MultiTimeframe3StrategiesAgent(BaseAgent):
    """
    An agent that first analyzes higher timeframe charts (15m, 60m, daily)
    to determine directional bias: Long, Short, or Sideways.

    Based on that bias, it selects and runs one of three predefined strategies:
    1. VWAP + Renko
    2. ElasticNet-based prediction
    3. Range Breakout (15m/30m)

    This agent is designed to generate rules, request data, trigger MCP-style simulation,
    and iterate based on performance — without mixing indicators between strategies.
    """
    def analyze(self, input_container, user_params):
        # Step 1: Input Validation (removed all blocking validation)
        vision_outputs = []
        per_image_costs = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_cost_usd = 0.0
        if isinstance(input_container, dict) and 'vision_outputs' in input_container:
            # For each image, call ScalpAgentSession.analyze_chart_image with agent's prompt
            for v in input_container['vision_outputs']:
                image_bytes = v.get('image_bytes')
                if image_bytes:
                    session = ScalpAgentSession(image_bytes=image_bytes)
                    prompt_text = self._build_vision_prompt(v)
                    result = session.analyze_chart_image(prompt_text)
                    if isinstance(result, dict) and 'response_text' in result:
                        raw_response = result['response_text']
                        token_usage = result.get('token_usage', {})
                        per_image_costs.append({
                            'interval': v.get('interval') or v.get('timeframe_tag'),
                            'token_usage': token_usage
                        })
                        total_prompt_tokens += token_usage.get('promptTokenCount', 0)
                        total_output_tokens += token_usage.get('candidatesTokenCount', 0)
                        total_tokens += token_usage.get('totalTokenCount', 0)
                        if token_usage.get('cost_usd') is not None:
                            try:
                                total_cost_usd += float(token_usage['cost_usd'])
                            except Exception:
                                pass
                        # Parse the raw_response as JSON if possible
                        try:
                            cleaned = raw_response.strip()
                            if cleaned.startswith('```json'):
                                cleaned = cleaned[7:]
                            if cleaned.startswith('```'):
                                cleaned = cleaned[3:]
                            if cleaned.endswith('```'):
                                cleaned = cleaned[:-3]
                            parsed = json.loads(cleaned)
                            vision_outputs.append(parsed)
                        except Exception:
                            vision_outputs.append({'error': 'Could not parse Gemini output', 'raw': raw_response})
                    else:
                        vision_outputs.append({'error': 'No valid Gemini response'})
                else:
                    vision_outputs.append({'error': 'No image bytes provided'})
        # Optionally set a flag for downstream logic
        user_params['bias_mode'] = 'vision_only'
        # Step 2: LLM-powered bias detection
        try:
            prompt = self._build_bias_prompt(input_container)
            model_name = CONFIG["model_name"]
            # --- Model/provider routing based on model_name only ---
            if model_name.startswith("gemini-"):
                api_key = CONFIG.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return {"feedback": "Gemini API key not set.", "step": "llm_error", "raw_bias_data": vision_outputs or []}
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                body = {
                    "contents": [{
                        "parts": [
                            {"text": prompt}
                        ]
                    }]
                }
                response = requests.post(endpoint, headers=headers, json=body, timeout=30)
                response.raise_for_status()
                content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif model_name.startswith("gpt-4") or model_name.startswith("gpt-3"):
                api_key = CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return {"feedback": "OpenAI API key not set.", "step": "llm_error", "raw_bias_data": vision_outputs or []}
                endpoint = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            else:
                log.info(f"[MultiTimeframe3StrategiesAgent] ERROR: Unknown or unsupported model_name '{model_name}'")
                return {"feedback": f"Unknown or unsupported model_name '{model_name}'. Please use a Gemini or OpenAI model.", "step": "llm_error", "raw_bias_data": vision_outputs or []}
            try:
                cleaned = content.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                bias_result = json.loads(cleaned)
            except Exception as e:
                return {"feedback": f"LLM response could not be parsed as JSON: {e}", "step": "llm_error", "raw": content, "raw_bias_data": vision_outputs or []}
            # Step 3: Multi-Timeframe Bias Summary
            try:
                multi_tf_prompt = self._build_multi_tf_bias_prompt(input_container)
                if model_name.startswith("gemini-"):
                    multi_body = {
                        "contents": [{
                            "parts": [
                                {"text": multi_tf_prompt}
                            ]
                        }]
                    }
                    multi_response = requests.post(endpoint, headers=headers, json=multi_body, timeout=30)
                    multi_response.raise_for_status()
                    multi_content = multi_response.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif model_name.startswith("gpt-4") or model_name.startswith("gpt-3"):
                    multi_payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": multi_tf_prompt}
                        ],
                        "max_tokens": 1000
                    }
                    multi_response = requests.post(endpoint, headers=headers, json=multi_payload, timeout=30)
                    multi_response.raise_for_status()
                    multi_content = multi_response.json()["choices"][0]["message"]["content"]
                else:
                    multi_content = "{}"
                multi_cleaned = multi_content.strip()
                if multi_cleaned.startswith("```json"):
                    multi_cleaned = multi_cleaned[7:]
                if multi_cleaned.startswith("```"):
                    multi_cleaned = multi_cleaned[3:]
                if multi_cleaned.endswith("```"):
                    multi_cleaned = multi_cleaned[:-3]
                multi_tf_bias = json.loads(multi_cleaned).get("multi_tf_bias", {})
            except Exception as e:
                multi_tf_bias = {}
            # --- Bias summary from vision outputs (per-image Gemini results) ---
            bias_summary = []
            raw_bias_data = []
            if vision_outputs and isinstance(vision_outputs, list):
                for v in vision_outputs:
                    interval = v.get('interval') or v.get('timeframe_tag')
                    strategies = [s.lower() for s in v.get('suggested_strategies', []) if isinstance(s, str)]
                    patterns = [p.lower() for p in v.get('patterns_detected', []) if isinstance(p, str)]
                    reasoning = v.get('reasoning_summary', '')
                    summary = (reasoning or '').lower()
                    # --- Infer bias_direction ---
                    bias_direction = 'Sideways'
                    confidence = 0.75
                    if any(x in strategies for x in ['trend following', 'breakout']):
                        bias_direction = 'Long'
                    elif any(x in strategies for x in ['range', 'mean reversion']):
                        bias_direction = 'Sideways'
                    elif any(x in patterns for x in ['descending', 'downtrend', 'triangle']) or 'downtrend' in summary or 'descending' in summary:
                        bias_direction = 'Short'
                    bias_summary.append({
                        'interval': interval,
                        'bias_direction': bias_direction,
                        'confidence': confidence,
                        'reasoning': reasoning
                    })
                    # Always append the raw Gemini JSON
                    raw_bias_data.append(v)
            return {
                "bias_summary": bias_summary,
                "raw_bias_data": raw_bias_data if raw_bias_data else (vision_outputs or []),
                "bias_direction": bias_result.get("bias_direction"),
                "bias_confidence": bias_result.get("bias_confidence"),
                "bias_rationale": bias_result.get("bias_rationale"),
                "bias_break_conditions": bias_result.get("bias_break_conditions"),
                "multi_tf_bias": multi_tf_bias,
                "step": "bias_detected",
                "bias_cost_metadata": {
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_tokens,
                    "total_cost_usd": total_cost_usd,
                    "per_image_breakdown": per_image_costs
                }
            }
        except Exception as e:
            return {"feedback": f"LLM bias detection failed: {e}", "step": "llm_error", "raw_bias_data": vision_outputs or []}

    def _is_higher_timeframe(self, interval):
        """
        Returns True if the interval is higher than 5m (e.g., 15m, 30m, 1h, D, etc.)
        """
        if not interval:
            return False
        interval = str(interval).lower().strip()
        # Acceptable: 15m, 30m, 1h, 4h, d, daily, w, weekly, etc.
        if interval in ["15m", "30m", "1h", "4h", "d", "daily", "w", "weekly"]:
            return True
        # Numeric check for intervals like '60', '60m', '240', etc.
        if interval.endswith('m'):
            try:
                minutes = int(interval[:-1])
                return minutes > 5
            except Exception:
                return False
        if interval.endswith('h'):
            try:
                hours = int(interval[:-1])
                return hours >= 1
            except Exception:
                return False
        # Accept 'day', 'week', etc.
        if any(x in interval for x in ["day", "week", "d", "w"]):
            return True
        return False

    def _build_bias_prompt(self, input_container):
        # Compose a prompt for the LLM using available higher timeframe data
        # This can be extended to include more context as needed
        multi_inputs = getattr(input_container, 'multi_inputs', None) or getattr(input_container, 'inputs', None)
        if multi_inputs and isinstance(multi_inputs, list):
            timeframe_summaries = []
            for inp in multi_inputs:
                tf = inp.get('interval')
                sym = inp.get('symbol')
                stats = inp.get('stats', {})
                summary = f"Symbol: {sym}, Timeframe: {tf}, Stats: {json.dumps(stats)}"
                timeframe_summaries.append(summary)
            context = "\n".join(timeframe_summaries)
        else:
            context = f"Symbol: {getattr(input_container, 'symbol', None)}, Timeframe: {getattr(input_container, 'interval', None)}"
        prompt = (
            "You are a multi-timeframe trading analyst.\n"
            "Given the following higher timeframe chart data and/or summary statistics, "
            "determine the most likely directional bias for the next session.\n"
            f"{context}\n"
            "\n"
            "Respond ONLY in valid JSON with the following fields:\n"
            "{\n"
            "  \"bias_direction\": \"Long\" | \"Short\" | \"Sideways\",\n"
            "  \"bias_confidence\": <float between 0 and 1>,\n"
            "  \"bias_rationale\": <short text>,\n"
            "  \"bias_break_conditions\": <short text>\n"
            "}\n"
            "Do not include any explanation outside the JSON."
        )
        return prompt

    def _build_multi_tf_bias_prompt(self, input_container):
        multi_inputs = getattr(input_container, 'multi_inputs', None) or getattr(input_container, 'inputs', None)
        if multi_inputs and isinstance(multi_inputs, list):
            timeframe_summaries = []
            for inp in multi_inputs:
                tf = inp.get('interval')
                sym = inp.get('symbol')
                stats = inp.get('stats', {})
                summary = f"Symbol: {sym}, Timeframe: {tf}, Stats: {json.dumps(stats)}"
                timeframe_summaries.append(summary)
            context = "\n".join(timeframe_summaries)
        else:
            context = f"Symbol: {getattr(input_container, 'symbol', None)}, Timeframe: {getattr(input_container, 'interval', None)}"
        prompt = (
            "You are a multi-timeframe trading analyst.\n"
            "Given the following chart data and/or summary statistics for each timeframe, "
            "determine the most likely directional bias for each timeframe.\n"
            f"{context}\n"
            "\n"
            "Respond ONLY in valid JSON with the following structure:\n"
            "{\n"
            "  \"multi_tf_bias\": {\n"
            "    \"Daily\": \"Long\" | \"Short\" | \"Sideways\",\n"
            "    \"60m\": \"Long\" | \"Short\" | \"Sideways\",\n"
            "    \"15m\": \"Long\" | \"Short\" | \"Sideways\"\n"
            "  }\n"
            "}\n"
            "Do not include any explanation outside the JSON."
        )
        return prompt

    def _build_vision_prompt(self, v):
        # Compose a prompt for Gemini Vision for a single image, requesting the new structured output
        interval = v.get('interval') or v.get('timeframe_tag') or ''
        symbol = v.get('symbol') or ''
        return (
            f"You are a multi-timeframe trading analyst.\n"
            f"Analyze the attached chart image for symbol {symbol} on the {interval} timeframe.\n"
            "Extract and reason about each of the following fields, and return ONLY a valid JSON object (optionally wrapped in a markdown code block for formatting):\n"
            "{\n"
            "  \"symbol\": \"@MES\" (or detected symbol),\n"
            "  \"interval\": \"15min / 60min / Daily\" (or detected),\n"
            "  \"trend_bias\": \"Short / Long / Sideways with justification\",\n"
            "  \"support_resistance_zones\": [\"5970 (support)\", \"6080 (resistance)\"],\n"
            "  \"volume_regime\": \"Low/High/Anomalies (with explanation)\",\n"
            "  \"session_notes\": \"E.g., price action near session open/close if visually inferable from chart's x-axis date breaks\",\n"
            "  \"patterns_detected\": [],\n"
            "  \"suggested_strategies\": [],\n"
            "  \"reasoning_summary\": \"A full paragraph explanation of what the chart shows and why certain strategies are proposed.\",\n"
            "  \"event_risk_placeholder\": \"Add macro or event notes here later when RAG is enabled\"\n"
            "}\n"
            "Respond ONLY with the JSON object above, optionally wrapped in a markdown code block (```json ... ```), and do not include any explanation outside the JSON."
        )
