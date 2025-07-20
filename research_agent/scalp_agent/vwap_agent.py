import itertools
import os
import base64
import imghdr

import pandas as pd

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

    @staticmethod
    def parse_llm_response_text(raw_response_text: str) -> dict:
        """
        Cleans and parses LLM response text as JSON. Removes markdown code block markers and strips whitespace.
        Prints the cleaned string before parsing. Raises ValueError if parsing fails.
        """
        import json
        cleaned = raw_response_text.strip()
        # Remove markdown code block markers if present
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):]
        if cleaned.startswith('```'):
            cleaned = cleaned[len('```'):]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-len('```')]
        cleaned = cleaned.strip()
        print(f"[VWAPAgent] Cleaned LLM response for JSON parsing:\n{cleaned[:1000]}")
        try:
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e:
            print(f"[VWAPAgent] ERROR: Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nCleaned string: {cleaned[:500]}")

    def build_grid_from_llm_response(self, llm_response: dict) -> pd.DataFrame:
        """
        Given an LLM response with suggested_strategies, thresholds, entry_conditions, and risk_management,
        build a parameter grid DataFrame. Each row is a unique combination of threshold values for a strategy,
        with strategy_name, bias, all threshold params, risk management fields, and entry_conditions.

        Example input:
        {
          "bias": "bullish",
          "suggested_strategies": [
            {
              "name": "VWAP_Bounce",
              "entry_conditions": ["Price pulls back near VWAP", "Volume Z-score > 1.5"],
              "thresholds": {
                "vwap_distance_pct": [0.1, 0.2],
                "volume_zscore_min": [1.0, 1.5],
                "ema_bias_filter": ["bullish_9_20"]
              },
              "risk_management": {
                "stop_loss": "below entry bar low",
                "take_profit": "VWAP mean or 2R",
                "risk_type": "technical"
              }
            }
          ]
        }
        """
        bias = llm_response.get('bias', None)
        strategies = llm_response.get('suggested_strategies', [])
        all_rows = []
        for strat in strategies:
            name = strat.get('name', None)
            thresholds = strat.get('thresholds', {})
            entry_conditions = strat.get('entry_conditions', [])
            risk_management = strat.get('risk_management', {})
            if not thresholds or not name:
                continue
            keys = list(thresholds.keys())
            value_lists = [thresholds[k] for k in keys]
            for combo in itertools.product(*value_lists):
                row = {
                    'strategy_name': name,
                    'bias': bias
                }
                row.update({k: v for k, v in zip(keys, combo)})
                # Risk management fields
                row['stop_loss_rule'] = risk_management.get('stop_loss')
                row['take_profit_rule'] = risk_management.get('take_profit')
                row['risk_type'] = risk_management.get('risk_type')
                # Entry conditions as concatenated string (optional)
                if entry_conditions:
                    row['entry_conditions'] = '\n'.join(entry_conditions)
                all_rows.append(row)
        df = pd.DataFrame(all_rows)
        return df

    def run_backtests_from_grid(self, df_grid, df_5m, df_1m=None):
        """
        Runs backtests for each strategy config in the parameter grid using run_backtest_VWAPStrategy.
        Args:
            df_grid: DataFrame of parameter grid (from build_grid_from_llm_response)
            df_5m: DataFrame of 5m data (with indicators)
            df_1m: Optional DataFrame of 1m data (for intrabar simulation)
        Returns:
            all_results: List of dicts, each with 'config', 'trades', 'metrics'
            summary_df: DataFrame of metrics for all strategies
        """
        from backend.analyzer.analyzer_cerebro_strategy_engine import run_backtest_VWAPStrategy
        print(f"[Step 8] Running backtest for {len(df_grid)} strategy configs...")
        all_results = []
        metrics_rows = []
        total = len(df_grid)
        for idx, row in df_grid.iterrows():
            config_dict = row.to_dict()
            print(f"[Step 8] Running backtest {idx+1}/{total} for strategy: {config_dict.get('strategy_name', 'N/A')}")
            try:
                result = run_backtest_VWAPStrategy(config_dict, df_5m, df_1m)
                metrics = result.get('metrics', {})
                trades = result.get('trades', [])
            except Exception as e:
                print(f"[Step 8] ERROR in backtest {idx+1}: {e}")
                metrics = {'error': str(e)}
                trades = []
            all_results.append({
                'config': config_dict,
                'trades': trades,
                'metrics': metrics
            })
            metrics_row = {'config_idx': idx}
            metrics_row.update(config_dict)
            metrics_row.update(metrics)
            metrics_rows.append(metrics_row)
        summary_df = pd.DataFrame(metrics_rows)
        print(f"[Step 8] Finished backtests. Total results: {len(all_results)}")
        return all_results, summary_df

    def generate_natural_language_rules(self, top_n=5):
        """
        Selects the top N strategies from self.summary_df, builds a structured summary for each,
        formats into a prompt, and calls the LLM to generate natural language rules.
        Stores the result in self.generated_rules.
        """
        if not hasattr(self, 'summary_df') or self.summary_df is None or self.summary_df.empty:
            print("[VWAPAgent] No summary_df available. Run backtests first.")
            return None
        df = self.summary_df.copy()
        # Sort by PnL (or win_rate if not present)
        sort_col = 'PnL' if 'PnL' in df.columns else ('win_rate' if 'win_rate' in df.columns else None)
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
        top_df = df.head(top_n)
        summaries = []
        for _, row in top_df.iterrows():
            summary = {
                'strategy_name': row.get('strategy_name', ''),
                'entry_conditions': row.get('entry_conditions', ''),
                'stop_loss_rule': row.get('stop_loss_rule', ''),
                'take_profit_rule': row.get('take_profit_rule', ''),
                'risk_type': row.get('risk_type', ''),
                'PnL': row.get('PnL', row.get('pnl', '')),  # support both
                'win_rate': row.get('win_rate', ''),
                'max_drawdown': row.get('max_drawdown', ''),
            }
            summaries.append(summary)
        # Format prompt for LLM
        prompt_lines = [
            "You are an expert trading rules analyst. Given the following top VWAP strategies, generate clear, natural language rules for each:",
            ""
        ]
        for i, s in enumerate(summaries, 1):
            prompt_lines.append(f"Strategy {i}: {s['strategy_name']}")
            prompt_lines.append(f"Entry: {s['entry_conditions']}")
            prompt_lines.append(f"Stop Loss: {s['stop_loss_rule']}")
            prompt_lines.append(f"Take Profit: {s['take_profit_rule']}")
            prompt_lines.append(f"Risk Type: {s['risk_type']}")
            prompt_lines.append(f"Metrics: PnL={s['PnL']}, Win Rate={s['win_rate']}, Max Drawdown={s['max_drawdown']}")
            prompt_lines.append("")
        prompt_text = '\n'.join(prompt_lines)
        print(f"\n📊 Sent top {top_n} strategies to LLM for rule generation...")
        print(f"[VWAPAgent] LLM prompt:\n{prompt_text}\n")
        # Call LLM (simulate if not implemented)
        if hasattr(self, 'llm_client') and hasattr(self.llm_client, 'ask_rules_from_grid_summary'):
            rules = self.llm_client.ask_rules_from_grid_summary(prompt_text)
        else:
            rules = f"[SIMULATED LLM OUTPUT]\nRules for {top_n} strategies would be generated here."
        print(f"[VWAPAgent] LLM-generated rules:\n{rules}\n")
        self.generated_rules = rules
        # === Debug output for backend validation ===
        print("\n=== Top Strategy Rules from LLM ===")
        print(self.generated_rules)
        print("\n--- Top 5 Strategy Grid Rows Used ---")
        if hasattr(self, 'summary_df') and self.summary_df is not None:
            print(self.summary_df.head(5).to_string(index=False))
        print("\n====================================\n")
        return rules

    @staticmethod
    def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Lowercase all column names. No alias mapping. User must upload with columns: open, high, low, close, volume.
        """
        df = df.copy()
        print(f"[VWAPAgent] [DEBUG] Columns before normalization: {list(df.columns)}")
        df.columns = [col.lower() for col in df.columns]
        print(f"[VWAPAgent] [DEBUG] Columns after normalization: {list(df.columns)}")
        return df

    def enrich_with_all_vwap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required indicators across all 4 VWAP strategies:
        - VWAP + VWAP upper/lower bands
        - EMA(9), EMA(20)
        - ATR(14)
        - DMI: ADX, +DI, -DI
        - Volume Z-score
        - Any additional indicators used in thresholds (ema_bias_filter, dmi_crossover, etc.)
        """
        import pandas_ta as ta
        # Normalize columns first
        df = self.normalize_ohlcv_columns(df)
        df = df.copy()

        # VWAP and bands
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['VWAP_upper'] = df['VWAP'] + df['ATR_14']
        df['VWAP_lower'] = df['VWAP'] - df['ATR_14']

        # EMAs
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_20'] = ta.ema(df['Close'], length=20)

        # DMI (includes ADX, +DI, -DI)
        dmi = ta.dmi(df['High'], df['Low'], df['Close'], length=14)
        df = df.join(dmi)

        # Volume Z-score (based on 20-bar rolling mean)
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

        # Optional: preprocess ema_bias_filter, dmi_crossover, etc.
        df['ema_bias_filter'] = df['EMA_9'] > df['EMA_20']
        df['dmi_crossover'] = df['DMP_14'] > df['DMN_14']

        return df

    def run(self, images, df_5m, user_params=None, top_n=5, df_1m=None):
        """
        Full pipeline: LLM call, grid build, backtest, rule generation, with debug prints after each major step.
        """
        # Step 0: Normalize columns and enrich DataFrame with all required VWAP indicators
        df_5m_norm = self.normalize_ohlcv_columns(df_5m)
        df_5m_enriched = self.enrich_with_all_vwap_indicators(df_5m_norm)
        new_cols = set(df_5m_enriched.columns) - set(df_5m_norm.columns)
        print(f"[VWAPAgent] [DEBUG] Enriched DataFrame with VWAP indicators. New columns: {list(new_cols)}")

        # Step 1: LLM call
        llm_result = self.call_llm_with_images(images, user_params=user_params)
        raw_response_text = llm_result.get("llm_raw_response")
        llm_structured = self.parse_llm_response_text(raw_response_text)

        # Step 2: Build grid
        grid_df = self.build_grid_from_llm_response(llm_structured)
        print(f"[VWAPAgent] [DEBUG] Parameter grid length: {len(grid_df)}")
        if not grid_df.empty:
            print(f"[VWAPAgent] [DEBUG] Parameter grid head(3):\n{grid_df.head(3).to_string(index=False)}")
        else:
            print("[VWAPAgent] [DEBUG] Parameter grid is empty!")

        # Step 3: Run backtests
        all_results, summary_df = self.run_backtests_from_grid(grid_df, df_5m_enriched, df_1m)
        self.summary_df = summary_df
        if not summary_df.empty:
            print(f"[VWAPAgent] [DEBUG] Backtest summary describe():\n{summary_df.describe(include='all').to_string()}")
            # Print top strategy metrics
            sort_col = 'PnL' if 'PnL' in summary_df.columns else ('win_rate' if 'win_rate' in summary_df.columns else None)
            if sort_col:
                top_row = summary_df.sort_values(sort_col, ascending=False).head(1)
                print(f"[VWAPAgent] [DEBUG] Top strategy metrics:\n{top_row.to_string(index=False)}")
        else:
            print("[VWAPAgent] [DEBUG] Backtest summary is empty!")

        # Step 4: Generate rules
        rules = self.generate_natural_language_rules(top_n=top_n)
        if rules:
            print("[VWAPAgent] [DEBUG] Natural language rules (line by line):")
            for line in str(rules).splitlines():
                print(f"    {line}")
        else:
            print("[VWAPAgent] [DEBUG] No rules generated.")
        return {
            "llm_result": llm_result,
            "llm_structured": llm_structured,
            "parameter_grid": grid_df,
            "backtest_results": all_results,
            "backtest_summary": summary_df,
            "natural_language_rules": rules
        }
