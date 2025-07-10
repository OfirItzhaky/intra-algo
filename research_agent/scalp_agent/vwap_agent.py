import os
import base64
import imghdr
from research_agent.scalp_agent.prompt_manager import VWAP_PROMPT_SINGLE_IMAGE, VWAP_PROMPT_4_IMAGES
from research_agent.config import CONFIG
import requests
import json

class VWAPAgent:
    """
    VWAPAgent handles multi-image VWAP prompt construction and LLM calls for strategy suggestion.
    For now, only image+prompt+LLM flow is implemented. CSV and advanced logic will be added later.
    """
    def __init__(self, user_params=None):
        self.user_params = user_params or {}
        self.model_name = CONFIG.get("model_name")
        self.provider = "gemini" if self.model_name and self.model_name.startswith("gemini-") else "openai"

    def build_prompt(self, num_images):
        """
        Selects the correct VWAP prompt based on the number of images.
        """
        if num_images == 1:
            return VWAP_PROMPT_SINGLE_IMAGE, "single_image"
        else:
            return VWAP_PROMPT_4_IMAGES, "multi_image"

    def call_llm_with_images(self, images, user_params=None):
        """
        Calls the configured LLM (OpenAI or Gemini) with the VWAP prompt and images.
        Returns a dict with raw response, cost, model, etc.
        """
        num_images = len(images)
        prompt_text, prompt_type = self.build_prompt(num_images)
        print(f"[VWAP_AGENT] Using prompt type: {prompt_type}")
        model_name = self.model_name
        provider = self.provider
        llm_response = None
        raw_response_text = None
        llm_cost_usd = None
        llm_token_usage = None
        if provider == "gemini":
            api_key = CONFIG.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return {"error": "GEMINI_API_KEY not set in config or environment."}
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            parts = [{"text": prompt_text}]
            for img_bytes in images:
                mime_type = imghdr.what(None, h=img_bytes) or "png"
                mime_type = f"image/{mime_type}"
                encoded_image = base64.b64encode(img_bytes).decode("utf-8")
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": encoded_image
                    }
                })
            body = {"contents": [{"parts": parts}]}
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            response = requests.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            response_data = response.json()
            raw_response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            llm_response = response.text
            usage = response_data.get('usage', response_data.get('usageMetadata', {}))
            llm_token_usage = usage.get('totalTokenCount')
            llm_cost_usd = usage.get('totalCostUsd')
            if llm_cost_usd is None and llm_token_usage is not None:
                try:
                    llm_cost_usd = float(llm_token_usage) * 0.0025 / 1000
                except Exception:
                    llm_cost_usd = None
        elif provider == "openai":
            api_key = CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"error": "OPENAI_API_KEY not set in config or environment."}
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            # Support multi-image payload for debug (OpenAI only uses the first image, but we want to inspect)
            content_list = [
                {"type": "text", "text": prompt_text}
            ]
            for img_bytes in images:
                mime_type = imghdr.what(None, h=img_bytes) or "png"
                image_data = f"data:image/{mime_type};base64,{base64.b64encode(img_bytes).decode()}"
                content_list.append({"type": "image_url", "image_url": {"url": image_data}})
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content_list
                    }
                ],
                "max_tokens": 1000
            }
            print("🧪 DEBUG: OpenAI payload:\n", json.dumps(payload, indent=2))
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            raw_response_text = response_data["choices"][0]["message"]["content"]
            llm_response = response.text
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            llm_token_usage = prompt_tokens + completion_tokens
            try:
                llm_cost_usd = (prompt_tokens * 0.01 + completion_tokens * 0.03) / 1000
            except Exception:
                llm_cost_usd = None
        else:
            return {"error": f"Unknown or unsupported model_name '{model_name}'."}
        return {
            "llm_raw_response": raw_response_text,
            "model_name": model_name,
            "provider": provider,
            "llm_cost_usd": llm_cost_usd,
            "llm_token_usage": llm_token_usage,
            "num_images": num_images,
            "prompt_type": prompt_type
        }

    # TODO: Add parsing, result formatting, and CSV logic in future steps. 