from __future__ import annotations

import base64
import json
import os
from typing import List, Optional, Tuple, Dict, Any

import tiktoken
from mimetypes import guess_type

from .prompt_manager_5_star import MAIN_SYSTEM_PROMPT, NEWS_AND_REPORT_BIAS as BIAS_TEMPLATE
from ..config import PRICING
try:
    # Optional: LangChain integration for OpenAI with conversation memory
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    LANGCHAIN_AVAILABLE = False

# Optional: LangChain integration for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_LC_AVAILABLE = True
except Exception as e:
    GOOGLE_LC_AVAILABLE = False

class FiveStarAgentController:
    """Controller for the Five Star swing-trading agent.

    Provides both a mock response method and a real LLM-backed analysis
    using OpenAI models (GPT-4o family). This keeps all agent logic within
    the 5star_agent package per project constraints.
    """

    REQUIRED_SOURCES = ["ProRealTime", "StockCharts", "Finviz"]

    def _format_summary_block(self, summary: str) -> str:
        return f"Context from previous chart analysis (summary):\n{summary}"

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
        except Exception as e:
            return None


    def analyze_with_model(self, instructions: str, image_paths: List[str], model_choice: str, session_id: Optional[str] = None, image_summary: Optional[str] = None, active_user: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
        """Analyze with chosen model; returns (reply_text, model_used, usage).

        Supports OpenAI (gpt-4o family) and Gemini (1.5 pro/flash). Falls back to a
        multimodal-capable model if a non-vision model is selected while images are present.
        """
        try:
            print(
                f"[FiveStar][DEBUG] analyze_with_model: Session={session_id or 'default'} | "
                f"Model selected={model_choice} | Images={image_paths} | Summary injected={bool(image_summary)}"
            )
        except Exception as e:
            pass

        # If we have a cached summary and no images for this turn, prepend the
        # summary text to the instructions and avoid passing it separately to
        # lower-level calls to prevent duplicate injection.
        if image_summary and not image_paths:
            try:
                summary_block = self._format_summary_block(image_summary)
                instructions = f"{summary_block}\n\n{instructions or ''}"
                print(f"[FiveStar][OPT] Prepended summary to instructions (no images). words={len(summary_block.split())}")
            except Exception as e:
                pass
            # Prevent duplicate injection in lower-level functions
            image_summary = None
        # Determine provider and model
        selection = (model_choice or "").strip() or "gpt-4o"
        provider = "openai" if not selection.lower().startswith("gemini") else "gemini"
        requested_model = selection

        if provider == "openai":
            # OpenAI image-capable defaults
            model_used = requested_model
            # Prefer LangChain (if available) to enable conversation memory
            if LANGCHAIN_AVAILABLE:
                reply, usage = self._openai_call_langchain(instructions, image_paths, model_used, session_id or "default", image_summary=image_summary, active_user=active_user)
            else:
                # Fallback to native SDK path
                reply, usage = self._openai_call(instructions, image_paths, model_used, image_summary=image_summary, active_user=active_user)
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

            # Prefer LangChain Gemini if available to mirror OpenAI LC flow
            try:
                print(f"[FiveStar][MODE] Gemini path selected | GOOGLE_LC_AVAILABLE={GOOGLE_LC_AVAILABLE}")
            except Exception:
                pass
            if GOOGLE_LC_AVAILABLE:
                try:
                    print("[FiveStar][MODE] Using Gemini LangChain path")
                except Exception:
                    pass
                reply, usage = self._gemini_call_langchain(instructions, image_paths, model_used, session_id or "default", image_summary=image_summary)
            else:
                try:
                    print("[FiveStar][MODE] Using Gemini native SDK path")
                except Exception:
                    pass
                reply, usage = self._gemini_call(instructions, image_paths, model_used, image_summary=image_summary)
            reply = f"{reply}\n🤖 Model: {usage.get('model_used', model_used)}\n💰 Tokens: {usage.get('total_tokens', 'N/A')} | Est. Cost: ${usage.get('estimated_cost_usd', 0):.6f}"
            return reply, usage.get('model_used', model_used), usage

    # --- Provider implementations ---
    _history_store: Dict[str, InMemoryChatMessageHistory] = {}

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        hist = self._history_store.get(session_id)
        if hist is None:
            hist = InMemoryChatMessageHistory()
            self._history_store[session_id] = hist
        return hist

    def _openai_call_langchain(self, instructions: str, image_paths: List[str], model_used: str, session_id: str, image_summary: Optional[str] = None, active_user: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """OpenAI via LangChain with message history. Supports images using image_url blocks.

        Returns (reply_text, usage_dict) like the native path.
        """
        # Select OpenAI key based on active user (supports itz01 override)
        active_user = (active_user or os.getenv("RESEARCH_AGENT_ACTIVE_USER") or os.getenv("USERNAME") or "").strip().lower()
        api_key = None
        if active_user == "itz01":
            api_key = os.getenv("ITZ_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            print("[FiveStar][KEY] Using ITZ_OPENAI_API_KEY for user itz01")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            print("[FiveStar][KEY] Using OPENAI_API_KEY for user", active_user or "(unknown)")
        if not api_key:
            return (
                "❌ Missing OpenAI API key. Set environment variable OPENAI_API_KEY to enable analysis.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        try:
            # Pass through key for LC as well
            llm = ChatOpenAI(model=model_used, temperature=0.2, api_key=api_key)
        except Exception as e:
            # Fallback to native path if LC model init fails
            print(f"[FiveStar][FALLBACK] OpenAI LC init failed → native path. err={e}")
            return self._openai_call(instructions, image_paths, model_used, image_summary=image_summary, active_user=active_user)

        # Compose human content blocks (text + image urls)
        human_content: List[Dict[str, Any]] = []
        # Prepend summary if present
        if image_summary:
            human_content.append({"type": "text", "text": self._format_summary_block(image_summary)})
        if (instructions or "").strip():
            human_content.append({"type": "text", "text": instructions})
        attached_names: List[str] = []
        for p in image_paths:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                human_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                attached_names.append(os.path.basename(p))
            except Exception as e:
                continue

        if not human_content:
            # No content – mirror native behavior
            return (
                "Please paste or upload at least one weekly chart image to analyze.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        messages = [
            SystemMessage(content=MAIN_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # Diagnostics: LC mode inputs
        try:
            text_len = sum(len(block.get("text", "")) for block in human_content if isinstance(block, dict) and block.get("type") == "text")
            print(
                f"[FiveStar][OPT] LangChain Mode: Using image summary: {bool(image_summary)} | "
                f"Image block count: {len([b for b in human_content if isinstance(b, dict) and b.get('type') == 'image_url'])} | "
                f"Text size (chars): {text_len}"
            )
        except Exception as e:
            pass

        # Build LCEL: extract list[BaseMessage] from dict → chat model
        extract_messages = RunnableLambda(lambda x: x["messages"])  # x is a dict with key 'messages'
        pipeline = extract_messages | llm
        chain = RunnableWithMessageHistory(
            pipeline,
            get_session_history=self._get_history,
            input_messages_key="messages",
        )

        try:
            print("[FiveStar][DEBUG] Prompt breakdown:")
            print("SystemMessage tokens:", len(tiktoken.encoding_for_model("gpt-4o").encode(MAIN_SYSTEM_PROMPT)))
            print("HumanMessage tokens (summary + instruction):",
                  len(tiktoken.encoding_for_model("gpt-4o").encode(str(human_content))))
            print("Messages length:", len(messages))

            resp = chain.invoke({"messages": messages}, config={"configurable": {"session_id": session_id}})
        except Exception as e:
            # Fallback to native path on any LC runtime issue
            print(f"[FiveStar][FALLBACK] OpenAI LC invoke failed → native path. err={e}")
            return self._openai_call(instructions, image_paths, model_used, image_summary=image_summary, active_user=active_user)

        # Extract text
        raw_text = getattr(resp, "content", "") or ""

        # Usage metadata (best-effort: handle multiple LC/OpenAI schema variants)
        meta = getattr(resp, "response_metadata", {}) or {}
        token_usage = meta.get("token_usage", {}) or {}
        # Try common keys
        prompt_tokens = token_usage.get("prompt_tokens") or meta.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens") or meta.get("completion_tokens")
        # Some responses expose usage under additional_kwargs
        if prompt_tokens is None or completion_tokens is None:
            ak = getattr(resp, "additional_kwargs", {}) or {}
            usage = ak.get("usage") or ak.get("token_usage") or {}
            prompt_tokens = prompt_tokens or usage.get("prompt_tokens")
            completion_tokens = completion_tokens or usage.get("completion_tokens")
        # Fallback: estimate tokens if LC wrapper provided none
        if prompt_tokens is None or completion_tokens is None:
            try:
                human_text_len = sum(len(block.get("text", "")) for block in human_content if isinstance(block, dict) and block.get("type") == "text")
                est_prompt_tokens = (len(MAIN_SYSTEM_PROMPT) + human_text_len) // 4
                if prompt_tokens is None:
                    prompt_tokens = max(1, est_prompt_tokens)
                    print(f"[FiveStar][OPT] LC Gemini prompt_tokens rough-estimated from chars={len(MAIN_SYSTEM_PROMPT)+human_text_len} -> tokens~{prompt_tokens}")
                if completion_tokens is None:
                    completion_tokens = max(0, len(raw_text) // 4)
                    print(f"[FiveStar][OPT] LC Gemini completion_tokens rough-estimated from chars={len(raw_text)} -> tokens~{completion_tokens}")
            except Exception:
                # Ensure integers
                prompt_tokens = int(prompt_tokens or 0)
                completion_tokens = int(completion_tokens or 0)

        # Fallback: estimate tokens if LC wrapper provided none or zeros
        if (prompt_tokens is None or int(prompt_tokens or 0) == 0) or (completion_tokens is None or int(completion_tokens or 0) == 0):
            try:
                human_text_len = sum(len(block.get("text", "")) for block in human_content if isinstance(block, dict) and block.get("type") == "text")
                est_prompt_tokens = (len(MAIN_SYSTEM_PROMPT) + human_text_len) // 4
                if prompt_tokens is None or int(prompt_tokens or 0) == 0:
                    prompt_tokens = max(1, est_prompt_tokens)
                    print(f"[FiveStar][OPT] LC Gemini prompt_tokens rough-estimated from chars={len(MAIN_SYSTEM_PROMPT)+human_text_len} -> tokens~{prompt_tokens}")
                if completion_tokens is None or int(completion_tokens or 0) == 0:
                    completion_tokens = max(0, len(raw_text) // 4)
                    print(f"[FiveStar][OPT] LC Gemini completion_tokens rough-estimated from chars={len(raw_text)} -> tokens~{completion_tokens}")
            except Exception:
                # Ensure integers
                prompt_tokens = int(prompt_tokens or 0)
                completion_tokens = int(completion_tokens or 0)

        usage_dict = self._estimate_cost(model_used, prompt_tokens, completion_tokens)
        try:
            print(
                f"[FiveStar][DEBUG][COST] provider=GeminiLC model={model_used} pt={prompt_tokens} ct={completion_tokens} total={usage_dict.get('total_tokens')} cost=${usage_dict.get('estimated_cost_usd')}"
            )
        except Exception:
            pass
        try:
            print(f"[FiveStar][OPT] LC Gemini usage (final): prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} est_cost=${usage_dict.get('estimated_cost_usd', 0)}")
        except Exception:
            pass

        try:
            print(
                f"[FiveStar][OPT] LC OpenAI call OK | Model={model_used} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens} | total={usage_dict.get('total_tokens')}"
            )
        except Exception as e:
            pass

        # Try parse JSON block as before
        parsed = None
        try:
            txt = raw_text
            if "```" in txt:
                txt = txt.split("```", 2)[1]
                txt = "\n".join(line for line in txt.splitlines() if not line.strip().lower().startswith("json"))
            parsed = json.loads(txt)
        except Exception as e:
            parsed = None

        score = recommendation = reasoning = close_note = None
        if isinstance(parsed, dict):
            score = parsed.get("score")
            recommendation = parsed.get("recommendation")
            reasoning = parsed.get("reasoning")
            close_note = parsed.get("close_note")

        def to_int_0_5(val):
            try:
                n = int(val)
                return 0 if n < 0 else 5 if n > 5 else n
            except Exception as e:
                return None

        score = to_int_0_5(score)
        if recommendation:
            rec = str(recommendation).strip().lower()
            recommendation = "Yes" if rec in ["yes", "y", "true", "1"] else ("No" if rec else None)

        if score is not None and recommendation and reasoning:
            parts = [
                "✅ Received charts: " + ", ".join(attached_names) if attached_names else "✅ Received charts",
                f"📊 Score: {score}/5",
                f"🟢 Swing-Trade Recommendation: {recommendation}",
                f"📝 Reasoning: {reasoning}",
            ]
            if close_note:
                parts.append(f"ℹ️ Close: {close_note}")
            if instructions:
                parts.append("📌 Instructions acknowledged: " + (instructions[:140] + ("..." if len(instructions) > 140 else "")))
            return "\n".join(parts), usage_dict

        ack = (instructions[:140] + ("..." if instructions and len(instructions) > 140 else "")) if instructions else ""
        return (
            ("✅ Received charts: " + ", ".join(attached_names) + "\n" if attached_names else "") +
            (f"📝 Model Response: {raw_text}\n" if raw_text else "📝 Model Response: (empty)\n") +
            (f"📌 Instructions acknowledged: {ack}" if ack else "")
        ), usage_dict

    def _openai_call(self, instructions: str, image_paths: List[str], model_used: str, image_summary: Optional[str] = None, active_user: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """OpenAI implementation using chat.completions (GPT-4o family).

        Returns: (reply_text, usage_dict)
        usage_dict keys: prompt_tokens, completion_tokens, total_tokens, model_used, estimated_cost_usd
        """
        # Choose API key based on active user (passed from app) with safe fallback
        active_user_lc = (active_user or os.getenv("RESEARCH_AGENT_ACTIVE_USER") or os.getenv("USERNAME") or "").strip().lower()
        if active_user_lc == "itz01":
            api_key = os.getenv("ITZ_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            print("[FiveStar][KEY] Using ITZ_OPENAI_API_KEY for user itz01")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            print(f"[FiveStar][KEY] Using OPENAI_API_KEY for user {active_user_lc or '(unknown)'}")
        if not api_key:
            return (
                "❌ Missing OpenAI API key. Set environment variable OPENAI_API_KEY to enable analysis.\n"
                "You can still upload charts, but responses will be placeholders."
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        # Lazily import to avoid dependency issues elsewhere
        try:
            from openai import OpenAI
        except Exception as e:
            return "❌ OpenAI client not available. Ensure 'openai' is installed.", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        client = OpenAI(api_key=api_key)

        # Build message content with text + images
        content = []
        user_text = instructions or ""
        if image_summary:
            content.append({"type": "text", "text": self._format_summary_block(image_summary)})
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
        try:
            print(f"[FiveStar][IMAGES] OpenAI payload includes image blocks: {len(attached_names) > 0}")
        except Exception as e:
            pass

        if not attached_names and not image_summary:
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

        try:
            # Rough content size diagnostics
            text_parts = [blk.get("text", "") for blk in content if isinstance(blk, dict) and blk.get("type") == "text"]
            text_size = sum(len(t) for t in text_parts)
            print(
                f"[FiveStar][OPT] OpenAI call OK | Model={openai_model_returned} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens} | total={usage_dict.get('total_tokens')} | text_size_chars={text_size} | image_blocks={len(attached_names)} | using_summary={bool(image_summary)}"
            )
        except Exception as e:
            pass

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
        except Exception as e:
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
            except Exception as e:
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

    def _gemini_call_langchain(self, instructions: str, image_paths: List[str], model_used: str, session_id: str, image_summary: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Gemini via LangChain with message history. Mirrors OpenAI LC flow."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                "❌ Missing Gemini API key. Set environment variable GEMINI_API_KEY to enable analysis.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        if not GOOGLE_LC_AVAILABLE:
            return self._gemini_call(instructions, image_paths, model_used, image_summary=image_summary)

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_used,
                temperature=0.2,
                convert_system_message_to_human=True,
                google_api_key=api_key,
            )
        except Exception:
            return self._gemini_call(instructions, image_paths, model_used, image_summary=image_summary)

        # Compose human content (text + images) using LC message format of dict blocks
        human_content: List[Dict[str, Any]] = []
        if image_summary:
            human_content.append({"type": "text", "text": self._format_summary_block(image_summary)})
        if (instructions or "").strip():
            human_content.append({"type": "text", "text": instructions})
        attached_names: List[str] = []
        for p in image_paths:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                human_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                attached_names.append(os.path.basename(p))
            except Exception:
                continue

        if not human_content:
            return (
                "Please paste or upload at least one weekly chart image to analyze.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
            )

        messages = [
            SystemMessage(content=MAIN_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # Diagnostics
        try:
            text_len = sum(len(block.get("text", "")) for block in human_content if isinstance(block, dict) and block.get("type") == "text")
            print(
                f"[FiveStar][OPT] LC Gemini: Using image summary={bool(image_summary)} | images={len([b for b in human_content if isinstance(b, dict) and b.get('type')=='image_url'])} | text_chars={text_len}"
            )
        except Exception:
            pass

        extract_messages = RunnableLambda(lambda x: x["messages"])  # x is dict with key 'messages'
        pipeline = extract_messages | llm
        chain = RunnableWithMessageHistory(
            pipeline,
            get_session_history=self._get_history,
            input_messages_key="messages",
        )

        try:
            resp = chain.invoke({"messages": messages}, config={"configurable": {"session_id": session_id}})
        except Exception:
            return self._gemini_call(instructions, image_paths, model_used, image_summary=image_summary)

        raw_text = getattr(resp, "content", "") or ""

        # Try to extract token usage from LC wrapper (response_metadata / additional_kwargs)
        meta = getattr(resp, "response_metadata", {}) or {}
        token_usage = meta.get("token_usage", {}) or {}
        prompt_tokens = token_usage.get("prompt_tokens") or meta.get("prompt_token_count")
        completion_tokens = token_usage.get("completion_tokens") or meta.get("candidates_token_count")
        if prompt_tokens is None or completion_tokens is None:
            ak = getattr(resp, "additional_kwargs", {}) or {}
            # Some wrappers may pass through usage_metadata
            um = ak.get("usage_metadata") or ak.get("usage") or {}
            prompt_tokens = prompt_tokens or um.get("prompt_token_count") or um.get("prompt_tokens")
            completion_tokens = completion_tokens or um.get("candidates_token_count") or um.get("completion_tokens")

        # Fallback estimation for LC Gemini if usage missing or zero
        try:
            if prompt_tokens is None or int(prompt_tokens or 0) == 0:
                human_text_len = sum(len(block.get("text", "")) for block in human_content if isinstance(block, dict) and block.get("type") == "text")
                prompt_tokens = max(1, (len(MAIN_SYSTEM_PROMPT) + human_text_len) // 4)
                print(f"[FiveStar][OPT] LC Gemini prompt_tokens rough-estimated from chars={len(MAIN_SYSTEM_PROMPT)+human_text_len} -> tokens~{prompt_tokens}")
            if completion_tokens is None or int(completion_tokens or 0) == 0:
                completion_tokens = max(0, len(raw_text) // 4)
                print(f"[FiveStar][OPT] LC Gemini completion_tokens rough-estimated from chars={len(raw_text)} -> tokens~{completion_tokens}")
        except Exception:
            # Ensure integers
            prompt_tokens = int(prompt_tokens or 0)
            completion_tokens = int(completion_tokens or 0)

        usage_dict = self._estimate_cost(model_used, prompt_tokens, completion_tokens)
        try:
            print(
                f"[FiveStar][DEBUG][COST] provider=GeminiLC model={model_used} pt={prompt_tokens} ct={completion_tokens} total={usage_dict.get('total_tokens')} cost=${usage_dict.get('estimated_cost_usd')}"
            )
        except Exception:
            pass

        # Try to parse structured JSON
        parsed = None
        try:
            txt = raw_text
            if "```" in txt:
                txt = txt.split("```", 2)[1]
                txt = "\n".join(line for line in txt.splitlines() if not line.strip().lower().startswith("json"))
            parsed = json.loads(txt)
        except Exception:
            parsed = None

        score = recommendation = reasoning = close_note = None
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
                parts_out.append("📌 Instructions acknowledged: " + (instructions[:140] + ("..." if len(instructions) > 140 else "")))
            return "\n".join(parts_out), usage_dict

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

    def _to_gemini_inline_part(self, path: str) -> Optional[Dict[str, Any]]:
        """Convert local file path to Gemini inlineData content part.

        Returns None if file cannot be read or is empty.
        """
        try:
            with open(path, "rb") as f:
                raw = f.read()
            if not raw:
                return None
            encoded = base64.b64encode(raw).decode("utf-8")
            mime_type = guess_type(path)[0] or self._guess_mime(path)
            # Gemini Python SDK expects snake_case keys: inline_data -> {mime_type, data}
            return {"inline_data": {"mime_type": mime_type, "data": encoded}}
        except Exception as e:
            print(f"[FiveStar][ERROR] Failed to build Gemini inlineData for {path}: {e}")
            return None

    def _gemini_call(self, instructions: str, image_paths: List[str], model_used: str, image_summary: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return (
                "❌ Missing Gemini API key. Set environment variable GEMINI_API_KEY to enable analysis."
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
        try:
            import google.generativeai as genai
        except Exception as e:
            return "❌ Gemini client not available. Ensure 'google-generativeai' is installed.", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_used)

        # Build a flat list of Part objects for Gemini generate_content
        parts: List[Dict[str, Any]] = []
        if image_summary:
            parts.append({"text": self._format_summary_block(image_summary)})
        if instructions and instructions.strip():
            parts.append({"text": instructions})
        attached_names: List[str] = []
        for p in image_paths:
            part = self._to_gemini_inline_part(p)
            if part and part.get("inline_data", {}).get("data"):
                parts.append(part)
                attached_names.append(os.path.basename(p))
            else:
                print(f"[FiveStar][WARN] Skipping empty/invalid Gemini inline_data for: {p}")
        try:
            print(f"[FiveStar][IMAGES] Gemini payload includes image parts: {len(attached_names) > 0}")
        except Exception as e:
            pass

        if not attached_names and not image_summary:
            # If no images, allow text-only follow-ups as long as there is any instruction text
            if not (instructions and instructions.strip()):
                return (
                    "Please paste or upload at least one weekly chart image to analyze.",
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}
                )
            try:
                print(f"[FiveStar][OPT] Proceeding with text-only follow-up (no image parts). instructions_chars={len(instructions)}")
            except Exception:
                pass

        # Use the full system prompt from prompt_manager to match OpenAI behavior
        parts = [{"text": MAIN_SYSTEM_PROMPT}] + parts

        try:
            # Validate no empty inline_data blocks before call
            for prt in parts:
                if isinstance(prt, dict) and "inline_data" in prt:
                    inline = prt["inline_data"]
                    assert inline.get("data"), "Empty inline_data in Gemini part"
            resp = model.generate_content(parts)
            raw_text = (resp.text or "").strip()
        except Exception as e:
            return f"❌ LLM call failed: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model_used": model_used, "estimated_cost_usd": 0.0}

        # Usage and cost
        um = getattr(resp, 'usage_metadata', None)
        # Google SDK may expose token counts as ints
        prompt_tokens = getattr(um, 'prompt_token_count', None) if um else None
        completion_tokens = getattr(um, 'candidates_token_count', None) if um else None
        # Some SDK versions return dict-like usage
        if (prompt_tokens is None or completion_tokens is None) and isinstance(um, dict):
            prompt_tokens = prompt_tokens or um.get('prompt_token_count') or um.get('promptTokens')
            completion_tokens = completion_tokens or um.get('candidates_token_count') or um.get('candidatesTokens')

        # If SDK didn't return usage, estimate using model.count_tokens as a fallback
        if prompt_tokens is None:
            try:
                ct = model.count_tokens(parts)
                prompt_tokens = int(getattr(ct, 'total_tokens', None) or ct.get('total_tokens'))
                print(f"[FiveStar][OPT] Gemini prompt_tokens estimated via count_tokens={prompt_tokens}")
            except Exception:
                try:
                    # Rough fallback: sum text char lengths / 4
                    est_chars = sum(len(prt.get('text', '')) for prt in parts if isinstance(prt, dict) and 'text' in prt)
                    prompt_tokens = max(1, est_chars // 4)
                    print(f"[FiveStar][OPT] Gemini prompt_tokens rough-estimated from chars={est_chars} -> tokens~{prompt_tokens}")
                except Exception:
                    prompt_tokens = 0
        if completion_tokens is None:
            try:
                ct_out = model.count_tokens(raw_text)
                completion_tokens = int(getattr(ct_out, 'total_tokens', None) or ct_out.get('total_tokens'))
                print(f"[FiveStar][OPT] Gemini completion_tokens estimated via count_tokens={completion_tokens}")
            except Exception:
                try:
                    completion_tokens = max(0, len(raw_text) // 4)
                    print(f"[FiveStar][OPT] Gemini completion_tokens rough-estimated from chars={len(raw_text)} -> tokens~{completion_tokens}")
                except Exception:
                    completion_tokens = 0
        usage_dict = self._estimate_cost(model_used, prompt_tokens, completion_tokens)

        try:
            # Compute text size across text parts
            parts_text = "\n".join(prt.get("text", "") for prt in parts if isinstance(prt, dict) and "text" in prt)
            print(
                f"[FiveStar][OPT] Gemini call OK | Model={model_used} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens} | total={usage_dict.get('total_tokens')} | text_size_chars={len(parts_text)} | image_parts={len(attached_names)} | using_summary={bool(image_summary)}"
            )
        except Exception as e:
            pass

        # Try to parse JSON from the response
        parsed = None
        try:
            txt = raw_text
            if "```" in txt:
                txt = txt.split("```", 2)[1]
                txt = "\n".join(line for line in txt.splitlines() if not line.strip().lower().startswith("json"))
            parsed = json.loads(txt)
        except Exception as e:
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
            except Exception as e:
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
        # Prices loaded from research_agent.config.PRICING
        pricing = PRICING
        # Normalize model string to a pricing key
        key = model_used or ""
        key_lower = key.lower()
        if key_lower.startswith("gpt-5"):
            norm = "gpt-5"
        elif key_lower.startswith("gpt-4o-mini"):
            norm = "gpt-4o-mini"
        elif key_lower.startswith("gpt-4o"):
            norm = "gpt-4o"
        elif key_lower.startswith("gpt-4.1"):
            norm = "gpt-4.1"
        elif key_lower.startswith("gpt-3.5-turbo"):
            norm = "gpt-3.5-turbo"
        elif key_lower.startswith("gemini-1.5-pro"):
            norm = "gemini-1.5-pro"
        elif key_lower.startswith("gemini-1.5-flash"):
            norm = "gemini-1.5-flash"
        else:
            # Fallback for prefixes like "models/gemini-..." or vendor-style names
            if "gemini-1.5-pro" in key_lower:
                norm = "gemini-1.5-pro"
            elif "gemini-1.5-flash" in key_lower:
                norm = "gemini-1.5-flash"
            else:
                norm = key

        price = pricing.get(norm)
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        total = pt + ct
        if not price:
            return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": total, "model_used": model_used, "estimated_cost_usd": 0.0}
        cost = (pt / 1000.0) * price["input_per_1k"] + (ct / 1000.0) * price["output_per_1k"]
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": total, "model_used": model_used, "estimated_cost_usd": round(cost, 6)}

