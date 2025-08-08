from __future__ import annotations

import base64
import json
import os
from typing import List, Optional


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

    def analyze_with_model(self, instructions: str, image_paths: List[str], model_choice: str) -> tuple[str, str]:
        """Analyze with chosen model; returns (reply_text, model_used).

        Supports OpenAI (gpt-4o family) and Gemini (1.5 pro/flash). Falls back to a
        multimodal-capable model if a non-vision model is selected while images are present.
        """
        # Determine provider and model
        selection = (model_choice or "").strip() or "gpt-4o-mini"
        provider = "openai" if not selection.lower().startswith("gemini") else "gemini"
        requested_model = selection

        if provider == "openai":
            # OpenAI image-capable defaults
            vision_defaults = ["gpt-4o", "gpt-4o-mini"]
            model_used = requested_model
            if requested_model not in vision_defaults:
                # Fallback for non-vision choices (e.g., 3.5, 4.1)
                model_used = "gpt-4o"
            reply = self._openai_call(instructions, image_paths, model_used)
            # Append model used line
            reply = f"{reply}\n🤖 Model: {model_used}"
            return reply, model_used
        else:
            # Gemini models (1.5 pro / 1.5 flash)
            allowed = ["gemini-1.5-pro", "gemini-1.5-flash"]
            model_used = requested_model if requested_model in allowed else "gemini-1.5-flash"
            reply = self._gemini_call(instructions, image_paths, model_used)
            reply = f"{reply}\n🤖 Model: {model_used}"
            return reply, model_used

    # --- Provider implementations ---
    def _openai_call(self, instructions: str, image_paths: List[str], model_used: str) -> str:
        """OpenAI implementation using chat.completions (GPT-4o family)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "❌ Missing OpenAI API key. Set environment variable OPENAI_API_KEY to enable analysis.\n"
                "You can still upload charts, but responses will be placeholders."
            )

        # Lazily import to avoid dependency issues elsewhere
        try:
            from openai import OpenAI
        except Exception:
            return "❌ OpenAI client not available. Ensure 'openai' is installed."

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
            return "Please paste or upload at least one weekly chart image to analyze."

        system_prompt = (
            "ANSWER ACCURAETLY! "
        )

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
            return f"❌ LLM call failed: {e}"

        raw_text = (response.choices[0].message.content or "").strip()

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
            return "\n".join(parts)

        # If parsing failed, return raw text in our wrapper
        ack = (instructions[:140] + ("..." if instructions and len(instructions) > 140 else "")) if instructions else ""
        return (
            "✅ Received charts: " + ", ".join(attached_names) + "\n" +
            (f"📝 Model Response: {raw_text}\n" if raw_text else "📝 Model Response: (empty)\n") +
            (f"📌 Instructions acknowledged: {ack}" if ack else "")
        )

    def _guess_mime(self, path: str) -> str:
        ext = os.path.splitext(path.lower())[1]
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/png")

    def _gemini_call(self, instructions: str, image_paths: List[str], model_used: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                "❌ Missing Gemini API key. Set environment variable GEMINI_API_KEY to enable analysis."
            )
        try:
            import google.generativeai as genai
        except Exception:
            return "❌ Gemini client not available. Ensure 'google-generativeai' is installed."

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
            return "Please paste or upload at least one weekly chart image to analyze."

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
            return f"❌ LLM call failed: {e}"

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
            return "\n".join(parts_out)

        ack = (instructions[:140] + ("..." if instructions and len(instructions) > 140 else "")) if instructions else ""
        return (
            "✅ Received charts: " + ", ".join(attached_names) + "\n" +
            (f"📝 Model Response: {raw_text}\n" if raw_text else "📝 Model Response: (empty)\n") +
            (f"📌 Instructions acknowledged: {ack}" if ack else "")
        )

