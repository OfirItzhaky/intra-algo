import json
from config import CONFIG

class ScalpAgentSession:
    """
    Session object for a scalping agent workflow.
    Holds context for the uploaded chart image, parsed results, and CSV requirements.
    
    Gemini Vision prompt used:
    -------------------------
    You are a trading assistant. Analyze the attached chart image and extract the following information in JSON format:
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
      ]
    }
    If any field is not visible, set it to null. Only return the JSON object, no explanation.

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
      "patterns_detected": ["bull flag", "double top"]
    }
    """
    def __init__(self, image_bytes=None, session_notes=None):
        self.image_bytes = image_bytes
        self.session_notes = session_notes
        self.csv_requirements = None
        self.gemini_response_raw = None
        self.gemini_prompt = self._build_gemini_prompt()

    def _build_gemini_prompt(self):
        return (
            "You are a trading assistant. Analyze the attached chart image and extract the following information in JSON format:\n\n"
            "{\n"
            '  "symbol": "<symbol shown on chart, or null if not visible>",\n'
            '  "interval": "<chart interval, e.g., \'5m\', \'15m\', \'1h\', or null if not visible>",\n'
            '  "last_bar_datetime": "<ISO 8601 datetime of the last visible bar, or null>",\n'
            '  "detected_indicators": [\n'
            '    {\n'
            '      "name": "<indicator name, e.g., \'EMA\', \'MACD\', \'RSI\'>",\n'
            '      "parameters": "<parameters as shown, e.g., \'10\', \'12,26,9\'>"\n'
            "    }\n"
            "    // ... more indicators\n"
            "  ],\n"
            '  "required_bar_count": "<number of bars visible on the chart, or estimate>",\n'
            '  "required_timeframe": "<date range covered by the chart, e.g., \'2024-05-01 to 2024-05-10\'>",\n'
            '  "special_requests": [\n'
            '    "<any special data needs, e.g., \'volume average\', \'VWAP\', or null>"\n'
            "  ],\n"
            '  "support_resistance_zones": [\n'
            '    "<zone description, e.g., \'5200–5220 (resistance)\', \'5050 (support)\'>"\n'
            "  ],\n"
            '  "patterns_detected": [\n'
            '    "<pattern name, e.g., \'bull flag\', \'double top\', \'ascending triangle\'>"\n'
            "  ]\n"
            "}\n\n"
            "If any field is not visible, set it to null. Only return the JSON object, no explanation."
        )

    def analyze_chart_image(self):
        """
        Sends the image and prompt to Gemini Vision API, parses the response,
        and returns a dictionary describing required CSV data.
        """
        try:
            import google.generativeai as genai
            api_key = CONFIG["gemini_api_key"]
            model_name = CONFIG.get("scalping_agent_model_name", "gemini-1.5-pro-latest")
            print("[ScalpAgentSession] Calling Gemini Vision API...")
            print(f"[ScalpAgentSession] Prompt: {self.gemini_prompt[:500]}{'...' if len(self.gemini_prompt) > 500 else ''}")
            print(f"[ScalpAgentSession] Image byte size: {len(self.image_bytes) if self.image_bytes else 0}")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            parts = [
                {"text": self.gemini_prompt},
                {"image": self.image_bytes}
            ]
            try:
                response = model.generate_content(parts)
                response_text = response.text if hasattr(response, 'text') else str(response)
                print(f"[ScalpAgentSession] Raw Gemini response: {response_text[:1000]}{'...' if len(response_text) > 1000 else ''}")
                self.gemini_response_raw = response_text
                try:
                    self.csv_requirements = json.loads(response_text)
                except Exception as e:
                    print(f"[ScalpAgentSession] JSON parse error: {e}")
                    self.csv_requirements = {"error": f"Gemini response could not be parsed as JSON. Raw response: {response_text}"}
            except Exception as e:
                print(f"[ScalpAgentSession] Gemini Vision call failed: {e}")
                self.csv_requirements = {"error": f"Gemini Vision call failed: {e}"}
        except Exception as e:
            print(f"[ScalpAgentSession] Unexpected error: {e}")
            self.gemini_response_raw = str(e)
            self.csv_requirements = {"error": f"Error calling Gemini Vision API: {e}"}
        return self.csv_requirements

    def get_requirements_summary(self):
        """
        Returns a user-friendly summary of what CSV(s) are needed, including support/resistance and patterns if present.
        """
        if not self.csv_requirements:
            return "No requirements extracted yet."
        if "error" in self.csv_requirements:
            return self.csv_requirements["error"]
        summary = [
            f"Symbol: {self.csv_requirements.get('symbol')}",
            f"Interval: {self.csv_requirements.get('interval')}",
            f"Indicators: {', '.join([i['name'] for i in self.csv_requirements.get('detected_indicators', [])])}",
            f"Bars Needed: {self.csv_requirements.get('required_bar_count')}",
            f"Timeframe: {self.csv_requirements.get('required_timeframe')}",
            f"Special Requests: {', '.join(self.csv_requirements.get('special_requests', []))}"
        ]
        zones = self.csv_requirements.get('support_resistance_zones')
        if zones:
            summary.append("Support/Resistance Zones:")
            for z in zones:
                summary.append(f"  • {z}")
        patterns = self.csv_requirements.get('patterns_detected')
        if patterns:
            summary.append("Patterns Detected:")
            for p in patterns:
                summary.append(f"  • {p}")
        return "\n".join(summary) 