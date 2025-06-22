import json
import base64
from config import CONFIG
import os
from google.oauth2 import service_account
from google.generativeai import configure, GenerativeModel
import imghdr
import requests
from dotenv import load_dotenv
import re

class ScalpAgentSession:
    """
    Session object for a scalping agent workflow.
    Holds context for the uploaded chart image, parsed results, and CSV requirements.
    
    Gemini Vision prompt used:
    -------------------------
    You are an expert trading analyst preparing to backtest and simulate trade strategies based on the attached chart image.
    Visually analyze the chart and infer what data would be required to realistically simulate and evaluate the most appropriate trading strategies for this setup.
    Do not rely on user-specified logic. Instead, recommend the indicators, timeframe, volume or volatility data, bar count, and any other features you would need to test strategies that fit the visible chart structure (e.g., trends, flags, breakouts, ranges, etc).
    Output ONLY a valid JSON object with the following fields:
    {
      "symbol": "<symbol shown on chart, or null if not visible>",
      "interval": "<chart interval, e.g., '5m', '15m', '1h', or null if not visible>",
      "last_bar_datetime": "<ISO 8601 datetime of the last visible bar, or null>",
      "detected_indicators": [
        {
          "name": "<indicator name, e.g., 'EMA', 'MACD', 'RSI'>",
          "parameters": "<parameters as shown, e.g., '10', '12,26,9'>"
        }
        // ... more indicators
      ],
      "required_bar_count": "<number of bars visible on the chart, or estimate>",
      "required_timeframe": "<date range covered by the chart, e.g., '2024-05-01 to 2024-05-10'>",
      "special_requests": [
        "<any special data needs, e.g., 'volume average', 'VWAP', or null>"
      ],
      "support_resistance_zones": [
        "<zone description, e.g., '5200–5220 (resistance)', '5050 (support)'>"
      ],
      "patterns_detected": [
        "<pattern name, e.g., 'bull flag', 'double top', 'ascending triangle'>"
      ],
      "suggested_strategies": [
        "<strategy type, e.g., 'trend breakout', 'pullback continuation', 'range fade'>"
      ],
      "reasoning_summary": "<short explanation of why these strategies were suggested, based on the chart>"
    }
    If any field is not visible, set it to null or an empty list as appropriate. Do not include any explanation outside the JSON.

    Example JSON output:
    --------------------
    {
      "symbol": "AAPL",
      "interval": "5m",
      "last_bar_datetime": "2024-06-07T15:55:00Z",
      "detected_indicators": [
        {"name": "EMA", "parameters": "10"},
        {"name": "MACD", "parameters": "12,26,9"}
      ],
      "required_bar_count": 120,
      "required_timeframe": "2024-06-07T09:30:00Z to 2024-06-07T15:55:00Z",
      "special_requests": ["volume average"],
      "support_resistance_zones": ["5200–5220 (resistance)", "5050 (support)"],
      "patterns_detected": ["bull flag", "double top"],
      "suggested_strategies": ["trend breakout", "pullback continuation", "range fade"],
      "reasoning_summary": "The chart shows a clear trend with some consolidation. A trend breakout strategy would be effective for capturing the trend, while a pullback continuation strategy could be used for entering the market after a pullback. A range fade strategy could be used for trading the range between 5200 and 5220."
    }
    """
    def __init__(self, image_bytes=None, session_notes=None):
        self.image_bytes = image_bytes
        self.session_notes = session_notes
        self.csv_requirements = None
        self.gemini_response_raw = None

    def analyze_chart_image(self, prompt_text):
        """
        Sends the image and prompt_text to Gemini Vision API and returns the raw response text.
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                load_dotenv()
                api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            model_name = "gemini-1.5-pro-latest"
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            mime_type = imghdr.what(None, h=self.image_bytes) or "png"
            mime_type = f"image/{mime_type}"
            encoded_image = base64.b64encode(self.image_bytes).decode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            body = {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": encoded_image
                            }
                        }
                    ]
                }]
            }
            print(f"[ScalpAgentSession] Calling Gemini Vision API at {endpoint} (HTTP, API key)...")
            print(f"[ScalpAgentSession] Prompt: {prompt_text[:500]}{'...' if len(prompt_text) > 500 else ''}")
            print(f"[ScalpAgentSession] Image byte size: {len(self.image_bytes) if self.image_bytes else 0}")
            response = requests.post(endpoint, headers=headers, json=body)
            print(f"[ScalpAgentSession] Gemini Vision raw response: {response.text[:1000]}{'...' if len(response.text) > 1000 else ''}")
            response.raise_for_status()
            response_data = response.json()
            # Return the raw Gemini response text (not parsed)
            response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            return response_text
        except Exception as e:
            print(f"[ScalpAgentSession] Gemini Vision call failed: {e}")
            return {"error": f"Gemini Vision call failed: {e}"}

    def get_requirements_summary(self):
        """
        Returns a user-friendly summary of what CSV(s) are needed, including support/resistance and patterns if present.
        """
        if not self.csv_requirements:
            return "No requirements extracted yet."
        if "error" in self.csv_requirements:
            return self.csv_requirements["error"]
        req = self.csv_requirements
        symbol = req.get('symbol') or 'UNKNOWN'
        interval = req.get('interval') or 'UNKNOWN interval'
        timeframe = req.get('required_timeframe') or 'an unspecified timeframe'
        bar_count = req.get('required_bar_count')
        indicators = req.get('detected_indicators', [])
        special = req.get('special_requests', [])
        zones = req.get('support_resistance_zones', [])
        patterns = req.get('patterns_detected', [])

        # Build indicator description
        if indicators:
            ind_list = []
            for i in indicators:
                name = i.get('name')
                params = i.get('parameters')
                if name and params:
                    ind_list.append(f"{name} ({params})")
                elif name:
                    ind_list.append(name)
            ind_str = ", ".join(ind_list)
        else:
            ind_str = None

        # Build special requests description
        special_str = ", ".join([s for s in special if s]) if special else None

        # Build support/resistance description
        zones_str = "; ".join(zones) if zones else None

        # Build patterns description
        patterns_str = ", ".join(patterns) if patterns else None

        # Compose the summary
        summary = f"Please upload data for the symbol {symbol} covering {timeframe} using {interval} intervals."
        if bar_count:
            summary += f" The data should include at least {bar_count} bars."
        if ind_str:
            summary += f" Required indicators: {ind_str}."
        else:
            summary += " No indicators are required, just price."
        if special_str:
            summary += f" Special requests: {special_str}."
        if zones_str:
            summary += f" Support/Resistance zones to note: {zones_str}."
        if patterns_str:
            summary += f" Patterns to look for: {patterns_str}."
        # Always include suggested strategies and reasoning summary if present
        strategies = req.get('suggested_strategies')
        reasoning = req.get('reasoning_summary')
        if strategies and isinstance(strategies, list) and any(strategies):
            summary += f"\n\nSuggested strategies: {', '.join([s for s in strategies if s])}."
        if reasoning:
            summary += f"\nRationale: {reasoning.strip()}"
        return summary 