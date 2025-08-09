from __future__ import annotations

import base64
import json
import os
from typing import List, Optional, Tuple, Dict, Any
from research_agent.five_star_agent.prompt_manager_5_star import MAIN_SYSTEM_PROMPT, NEWS_AND_REPORT_BIAS as BIAS_TEMPLATE
try:
    # Optional: LangChain integration for future memory/agents
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

class FiveStarAgentController:
    """Controller for the Five Star swing-trading agent.

    Provides both a mock response method and a real LLM-backed analysis
    using OpenAI models (GPT-4o family). This keeps all agent logic within
    the 5star_agent package per project constraints.
    """

    REQUIRED_SOURCES = ["ProRealTime", "StockCharts", "Finviz"]

    def generate_placeholder_response(self, instructions: str, images: List[str]) -> str:
        """Return a mock agent response for the provided images + instructions."""
        if not images:
            return "I didn't receive any charts. Please upload weekly charts (ProRealTime, StockCharts, Finviz)."
        score = 3
        recommendation = "No"
        reason = "Placeholder: awaiting full strategy logic."
        if instructions:
            reason += f" Instructions noted: {instructions[:140]}{'...' if len(instructions) > 140 else ''}"
        response = (
            "✅ Received charts: " + ", ".join(images) + "\n"
            f"📊 Score: {score}/5\n"
            f"🟢 Swing-Trade Recommendation: {recommendation}\n"
            f"📝 Reasoning: {reason}"
        )
        return response

    # -----------------------------
    # Real LLM integration (OpenAI)
    # -----------------------------
    def _encode_image_as_data_url(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            # Default to PNG for data URL; browser/LLM will handle
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    def analyze_with_llm(self, instructions: str, image_paths: List[str]) -> str:
        """Call OpenAI model with charts and instructions; return formatted reply (legacy)."""
        reply, _ = self.analyze_with_model(instructions=instructions, image_paths=image_paths, model_choice="gpt-4o-mini")
        return reply

    def analyze_with_model(self, instructions: str, image_paths: List[str], model_choice: str) -> Tuple[str, str, Dict[str, Any]]:
        """Analyze with chosen model; returns (reply_text, model_used, usage).

        Supports OpenAI (gpt-4o family) and Gemini (1.5 pro/flash). Falls back to a
        multimodal-capable model if a non-vision model is selected while images are present.
        """
        # Determine provider and model
        selection = (model_choice or "").strip() or "gpt-4o-mini"
        provider = "openai" if not selection.lower().startswith("gemini") else "gemini"
        requested_model = "gpt-4o" #todo:TEMP HARD CODED

        if provider == "openai":
            # OpenAI image-capable defaults
            vision_defaults = ["gpt-4o", "gpt-4o-mini"]
            model_used = requested_model
            if requested_model not in vision_defaults:
                # Fallback for non-vision choices (e.g., 3.5, 4.1)
                model_used = "gpt-4o" #todo:TEMP HARD CODED
            # For now we still call native OpenAI path to preserve image support
            reply, usage = self._openai_call(instructions, image_paths, model_used)
            reply = f"{reply}\n🤖 Model: {usage.get('model_used', model_used)}\n💰 Tokens: {usage.get('total_tokens', 'N/A')} | Est. Cost: ${usage.get('estimated_cost_usd', 0):.6f}"
            return reply, usage.get('model_used', model_used), usage
        else:
            # Gemini models (1.5 pro / 1.5 flash) including -latest aliases
            allowed = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
            ]
            model_used = requested_model if requested_model in allowed else "gemini-1.5-flash-latest"
            reply, usage = self._gemini_call(instructions, image_paths, model_used)
            reply = f"{reply}\n🤖 Model: {usage.get('model_used', model_used)}\n💰 Tokens: {usage.get('total_tokens', 'N/A')} | Est. Cost: ${usage.get('estimated_cost_usd', 0):.6f}"
            return reply, usage.get('model_used', model_used), usage

    # --- Provider implementations ---
    def _openai_call(self, instructions: str, image_paths: List[str], model_used: str) -> Tuple[str, Dict[str, Any]]:
        """OpenAI implementation using chat.completions (GPT-4o family).

        Returns: (reply_text, usage_dict)
        usage_dict keys: prompt_tokens, completion_tokens, total_tokens, model_used, estimated_cost_usd
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "❌ Missing OpenAI API key. Set environment variable OPENAI_API_KEY to enable analysis.\n"
                "You can still upload charts, but responses will be placeholders."
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        # Lazily import to avoid dependency issues elsewhere
        try:
            from openai import OpenAI
        except Exception:
            return "❌ OpenAI client not available. Ensure 'openai' is installed.", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        client = OpenAI(api_key=api_key)

        # Build message content with text + images
        content = []
        user_text = instructions or ""
        if user_text.strip():
            content.append({"type": "text", "text": user_text})

        attached_names = []
        for p in image_paths:
            data_url = self._encode_image_as_data_url(p)
            if not data_url:
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
            attached_names.append(os.path.basename(p))

        if not attached_names:
            # Allow text-only follow-ups
            return (
                "Please paste or upload at least one weekly chart image to analyze.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        system_prompt = (
            MAIN_SYSTEM_PROMPT
        )
        # Inject real-time bias string dynamically
        bias_text = "Currently No Bias for this Stock"  # you define this
        MAIN_PROMPT_WITH_BIAS = MAIN_SYSTEM_PROMPT.replace(BIAS_TEMPLATE, bias_text)

        try:
            response = client.chat.completions.create(
                model=model_used,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as e:
            return f"❌ LLM call failed: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        raw_text = (response.choices[0].message.content or "").strip()

        # Gather usage and estimate cost
        openai_model_returned = getattr(response, 'model', None) or model_used
        usage_obj = getattr(response, 'usage', None) or {}
        prompt_tokens = getattr(usage_obj, 'prompt_tokens', None) if hasattr(usage_obj, 'prompt_tokens') else usage_obj.get('prompt_tokens') if isinstance(usage_obj, dict) else None
        completion_tokens = getattr(usage_obj, 'completion_tokens', None) if hasattr(usage_obj, 'completion_tokens') else usage_obj.get('completion_tokens') if isinstance(usage_obj, dict) else None
        total_tokens = getattr(usage_obj, 'total_tokens', None) if hasattr(usage_obj, 'total_tokens') else usage_obj.get('total_tokens') if isinstance(usage_obj, dict) else None
        usage_dict = self._estimate_cost(openai_model_returned, prompt_tokens, completion_tokens)

        # Try to parse JSON from the response
        parsed = None
        try:
            # Some models wrap JSON in code fences
            txt = raw_text
            if "```" in txt:
                txt = txt.split("```", 2)[1]
                # Remove an optional language hint like ```json
                txt = "\n".join(line for line in txt.splitlines() if not line.strip().lower().startswith("json"))
            parsed = json.loads(txt)
        except Exception:
            parsed = None

        # Fallbacks
        score = None
        recommendation = None
        reasoning = None
        close_note = None
        if isinstance(parsed, dict):
            score = parsed.get("score")
            recommendation = parsed.get("recommendation")
            reasoning = parsed.get("reasoning")
            close_note = parsed.get("close_note")

        # Validate and coerce
        def to_int_0_5(val):
            try:
                n = int(val)
                if n < 0: n = 0
                if n > 5: n = 5
                return n
            except Exception:
                return None

        score = to_int_0_5(score)
        if recommendation:
            rec = str(recommendation).strip().lower()
            recommendation = "Yes" if rec in ["yes", "y", "true", "1"] else ("No" if rec else None)

        # Build final text
        if score is not None and recommendation and reasoning:
            parts = [
                "✅ Received charts: " + ", ".join(attached_names),
                f"📊 Score: {score}/5",
                f"🟢 Swing-Trade Recommendation: {recommendation}",
                f"📝 Reasoning: {reasoning}",
            ]
            if close_note:
                parts.append(f"ℹ️ Close: {close_note}")
            if instructions:
                parts.append(
                    "📌 Instructions acknowledged: " + (instructions[:140] + ("..." if len(instructions) > 140 else ""))
                )
            return "\n".join(parts), usage_dict

        # If parsing failed, return raw text in our wrapper
        ack = (instructions[:140] + ("..." if instructions and len(instructions) > 140 else "")) if instructions else ""
        return (
            "✅ Received charts: " + ", ".join(attached_names) + "\n" +
            (f"📝 Model Response: {raw_text}\n" if raw_text else "📝 Model Response: (empty)\n") +
            (f"📌 Instructions acknowledged: {ack}" if ack else "")
        ), usage_dict

    def _guess_mime(self, path: str) -> str:
        ext = os.path.splitext(path.lower())[1]
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/png")

    def _gemini_call(self, instructions: str, image_paths: List[str], model_used: str) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                "❌ Missing Gemini API key. Set environment variable GEMINI_API_KEY to enable analysis."
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
        try:
            import google.generativeai as genai
        except Exception:
            return "❌ Gemini client not available. Ensure 'google-generativeai' is installed.", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_used)

        parts = []
        if instructions and instructions.strip():
            parts.append(instructions)
        attached_names = []
        for p in image_paths:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                mime = self._guess_mime(p)
                parts.append({"mime_type": mime, "data": data})
                attached_names.append(os.path.basename(p))
            except Exception:
                continue

        if not attached_names:
            # Allow text-only follow-ups
            return (
                "Please paste or upload at least one weekly chart image to analyze.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        system_prompt = (
            "You are a swing trading assistant. Analyze the uploaded weekly charts and give a score out of 5 with reasoning. "
            "Return a concise JSON with keys: score (0-5), recommendation ('Yes' or 'No'), reasoning (short paragraph), close_note (optional)."
        )
        # Gemini uses system instruction via safety settings or first-turn; prepend as text
        parts = [system_prompt] + parts

        try:
            resp = model.generate_content(parts)
            raw_text = (resp.text or "").strip()
        except Exception as e:
            return f"❌ LLM call failed: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        # Usage and cost
        um = getattr(resp, 'usage_metadata', None)
        prompt_tokens = getattr(um, 'prompt_token_count', None) if um else None
        completion_tokens = getattr(um, 'candidates_token_count', None) if um else None
        usage_dict = self._estimate_cost(model_used, prompt_tokens, completion_tokens)

        # Try to parse JSON from the response
        parsed = None
        try:
            txt = raw_text
            if "```" in txt:
                txt = txt.split("```", 2)[1]
                txt = "\n".join(line for line in txt.splitlines() if not line.strip().lower().startswith("json"))
            parsed = json.loads(txt)
        except Exception:
            parsed = None

        score = None
        recommendation = None
        reasoning = None
        close_note = None
        if isinstance(parsed, dict):
            score = parsed.get("score")
            recommendation = parsed.get("recommendation")
            reasoning = parsed.get("reasoning")
            close_note = parsed.get("close_note")

        def to_int_0_5(val):
            try:
                n = int(val)
                if n < 0: n = 0
                if n > 5: n = 5
                return n
            except Exception:
                return None
        score = to_int_0_5(score)
        if recommendation:
            rec = str(recommendation).strip().lower()
            recommendation = "Yes" if rec in ["yes", "y", "true", "1"] else ("No" if rec else None)

        if score is not None and recommendation and reasoning:
            parts_out = [
                "✅ Received charts: " + ", ".join(attached_names),
                f"📊 Score: {score}/5",
                f"🟢 Swing-Trade Recommendation: {recommendation}",
                f"📝 Reasoning: {reasoning}",
            ]
            if close_note:
                parts_out.append(f"ℹ️ Close: {close_note}")
            if instructions:
                parts_out.append(
                    "📌 Instructions acknowledged: " + (instructions[:140] + ("..." if len(instructions) > 140 else ""))
                )
            return "\n".join(parts_out), usage_dict

        ack = (instructions[:140] + ("..." if instructions and len(instructions) > 140 else "")) if instructions else ""
        return (
            "✅ Received charts: " + ", ".join(attached_names) + "\n" +
            (f"📝 Model Response: {raw_text}\n" if raw_text else "📝 Model Response: (empty)\n") +
            (f"📌 Instructions acknowledged: {ack}" if ack else "")
        ), usage_dict

    # --- Cost estimation ---
    def _estimate_cost(self, model_used: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Dict[str, Any]:
        # Prices per 1K tokens (approximate; adjust as needed)
        pricing = {
            # OpenAI
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.00060},
            "gpt-4.1": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
            # Gemini
            "gemini-1.5-pro": {"input_per_1k": 0.0035, "output_per_1k": 0.0100},
            "gemini-1.5-flash": {"input_per_1k": 0.00035, "output_per_1k": 0.00105},
            "gemini-1.5-pro-latest": {"input_per_1k": 0.0035, "output_per_1k": 0.0100},
            "gemini-1.5-flash-latest": {"input_per_1k": 0.00035, "output_per_1k": 0.00105},
        }
        price = pricing.get(model_used) or pricing.get(model_used.split(":")[0], None)
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        total = pt + ct
        if not price:
            return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": total, "model_used": model_used, "estimated_cost_usd": 0.0}
        cost = (pt / 1000.0) * price["input_per_1k"] + (ct / 1000.0) * price["output_per_1k"]
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": total, "model_used": model_used, "estimated_cost_usd": round(cost, 6)}

