import inspect
import itertools
import os
import base64
import imghdr

import pandas as pd
import pandas_ta as ta

from copy import deepcopy

from backend.analyzer.analyzer_blueprint_vwap_strategy import VWAPBounceStrategy
from .prompt_manager import VWAP_PROMPT_SINGLE_IMAGE, VWAP_PROMPT_4_IMAGES, VWAP_OPTIMIZATION_PROMPT
from .scalp_rag_utils import extract_json_block
from ..config import CONFIG, VWAP_STRATEGY_PARAM_TEMPLATE, PRICING
import requests
import json
import plotly.graph_objects as go
from backend.analyzer.analyzer_dashboard import AnalyzerDashboard



class VWAPAgent:
    """
    VWAPAgent handles multi-image VWAP prompt construction and LLM calls for strategy suggestion.
    For now, only image+prompt+LLM flow is implemented. CSV and advanced logic will be added later.
    """
    def __init__(self, user_params=None):
        self.user_params = user_params or {}
        # --- NEW: Allow override from user_params ---
        self.model_name = self.user_params.get('model_name') or CONFIG.get("model_name")
        self.provider = self.user_params.get('provider') or ("gemini" if self.model_name and self.model_name.startswith("gemini-") else "openai")
        print(f"[VWAPAgent][INIT] model_name: {self.model_name}, provider: {self.provider}")

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
        user_params = user_params or {}
        num_images = len(images)  # Always calculate num_images regardless of prompt type
        
        # Check for prompt override
        if 'prompt_override' in user_params:
            prompt_text = user_params['prompt_override']
            prompt_type = 'renko_override'
        else:
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
            # Gemini usage fields may include promptTokenCount and candidatesTokenCount
            prompt_tokens = usage.get('promptTokenCount') or usage.get('prompt_tokens') or 0
            completion_tokens = usage.get('candidatesTokenCount') or usage.get('output_tokens') or 0
            total_tokens = usage.get('totalTokenCount')
            if (not prompt_tokens and not completion_tokens) and total_tokens:
                # Fallback split if only total is available
                prompt_tokens = int(total_tokens * 0.6)
                completion_tokens = int(total_tokens - prompt_tokens)
            llm_token_usage = (prompt_tokens or 0) + (completion_tokens or 0) or total_tokens
            # Pricing-based cost estimate
            pricing = PRICING.get(model_name) or next((PRICING[k] for k in PRICING if str(model_name).startswith(k)), None)
            if pricing:
                in_rate = pricing.get('input_per_1k') or 0.0
                out_rate = pricing.get('output_per_1k') or 0.0
                llm_cost_usd = ((prompt_tokens or 0) * in_rate + (completion_tokens or 0) * out_rate) / 1000
            else:
                llm_cost_usd = usage.get('totalCostUsd')
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
            llm_token_usage = (prompt_tokens or 0) + (completion_tokens or 0)
            # Pricing-based cost estimate
            pricing = PRICING.get(model_name) or next((PRICING[k] for k in PRICING if str(model_name).startswith(k)), None)
            if pricing:
                in_rate = pricing.get('input_per_1k') or 0.0
                out_rate = pricing.get('output_per_1k') or 0.0
                llm_cost_usd = ((prompt_tokens or 0) * in_rate + (completion_tokens or 0) * out_rate) / 1000
            else:
                llm_cost_usd = None
        else:
            return {"error": f"Unknown or unsupported model_name '{model_name}'."}
        print(f"[VWAPAgent] Provider: {provider}, Model: {model_name}, Tokens: {llm_token_usage}, Cost: ${llm_cost_usd}")

        return {
            "llm_raw_response": raw_response_text,
            "model_name": model_name,
            "provider": provider,
            "llm_cost_usd": llm_cost_usd,
            "llm_token_usage": llm_token_usage,
            "num_images": num_images,
            "prompt_type": prompt_type
        }

    def call_llm_text(self, prompt_text: str):
        """
        Calls the configured LLM with a text-only prompt (no images). Returns a dict mirroring call_llm_with_images.
        """
        model_name = self.model_name
        provider = self.provider
        raw_response_text = None
        llm_cost_usd = None
        llm_token_usage = None
        if provider == "gemini":
            api_key = CONFIG.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return {"error": "GEMINI_API_KEY not set in config or environment."}
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            body = {"contents": [{"parts": [{"text": prompt_text}]}]}
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            response = requests.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            response_data = response.json()
            raw_response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            usage = response_data.get('usage', response_data.get('usageMetadata', {}))
            # Attempt to use detailed fields when available
            prompt_tokens = usage.get('promptTokenCount') or usage.get('prompt_tokens') or 0
            completion_tokens = usage.get('candidatesTokenCount') or usage.get('output_tokens') or 0
            total_tokens = usage.get('totalTokenCount')
            if (not prompt_tokens and not completion_tokens) and total_tokens:
                prompt_tokens = int(total_tokens * 0.6)
                completion_tokens = int(total_tokens - prompt_tokens)
            llm_token_usage = (prompt_tokens or 0) + (completion_tokens or 0) or total_tokens
            pricing = PRICING.get(model_name) or next((PRICING[k] for k in PRICING if str(model_name).startswith(k)), None)
            if pricing:
                in_rate = pricing.get('input_per_1k') or 0.0
                out_rate = pricing.get('output_per_1k') or 0.0
                llm_cost_usd = ((prompt_tokens or 0) * in_rate + (completion_tokens or 0) * out_rate) / 1000
            else:
                llm_cost_usd = usage.get('totalCostUsd')
        elif provider == "openai":
            api_key = CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"error": "OPENAI_API_KEY not set in config or environment."}
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 1000
            }
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            raw_response_text = response_data["choices"][0]["message"]["content"]
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            llm_token_usage = (prompt_tokens or 0) + (completion_tokens or 0)
            pricing = PRICING.get(model_name) or next((PRICING[k] for k in PRICING if str(model_name).startswith(k)), None)
            if pricing:
                in_rate = pricing.get('input_per_1k') or 0.0
                out_rate = pricing.get('output_per_1k') or 0.0
                llm_cost_usd = ((prompt_tokens or 0) * in_rate + (completion_tokens or 0) * out_rate) / 1000
            else:
                llm_cost_usd = None
        else:
            return {"error": f"Unknown or unsupported model_name '{model_name}'."}

        return {
            "llm_raw_response": raw_response_text,
            "model_name": model_name,
            "provider": provider,
            "llm_cost_usd": llm_cost_usd,
            "llm_token_usage": llm_token_usage,
            "num_images": 0,
            "prompt_type": "text_only"
        }

    @staticmethod
    def parse_llm_response_text(raw_response_text: str) -> dict:
        """
        Cleans and parses LLM response text as JSON. Removes markdown code block markers and strips whitespace.
        Prints the cleaned string before parsing. Raises ValueError if parsing fails.
        """
        llm_structured = extract_json_block(raw_response_text)
        return llm_structured

    def build_grid_from_llm_response(self, llm_response: dict) -> pd.DataFrame:
        """
        Converts the structured LLM response into a parameter grid DataFrame.
        Each row represents a strategy configuration for backtesting.

        Includes:
        - strategy_name and bias
        - all threshold combinations (vwap_distance_pct, zscore, etc.)
        - SL/TP function and parameters if provided by translator
        - risk description fields (for traceability)
        """
        bias = llm_response.get('bias', None)
        strategies = llm_response.get('suggested_strategies', [])
        all_rows = []

        for strat in strategies:
            name = strat.get('name')
            thresholds = strat.get('thresholds', {})
            risk = strat.get('risk_management', {})
            entry_conditions = strat.get('entry_conditions', [])

            if not thresholds or not name:
                continue

            threshold_keys = list(thresholds.keys())
            threshold_values = [thresholds[k] for k in threshold_keys]

            # Safeguard for threshold list mismatch
            if not all(isinstance(v, list) for v in threshold_values):
                print(f"[WARN] Skipping strategy '{name}' due to malformed threshold values.")
                continue

            # Extract sl_tp_function and its params
            sl_tp_function = strat.get("sl_tp_function")
            sl_tp_params = {
                k: v for k, v in strat.items()
                if k not in ['name', 'thresholds', 'risk_management', 'entry_conditions', 'sl_tp_function']
            }

            # Manual combo logic (instead of itertools.product)
            total_combos = 1
            for vlist in threshold_values:
                total_combos *= len(vlist)

            for i in range(total_combos):
                idxs = []
                divisor = total_combos
                for vlist in threshold_values:
                    divisor = divisor // len(vlist)
                    idxs.append((i // divisor) % len(vlist))

                row = {
                    "strategy_name": name,
                    "bias": bias,
                    "sl_tp_function": sl_tp_function,
                    "stop_loss_rule": risk.get("stop_loss"),
                    "take_profit_rule": risk.get("take_profit"),
                    "risk_type": risk.get("risk_type")
                }

                for k, vlist, idx in zip(threshold_keys, threshold_values, idxs):
                    row[k] = vlist[idx]

                if entry_conditions:
                    row["entry_conditions"] = "\n".join(entry_conditions)

                row.update(sl_tp_params)
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        if not df.empty:
            debug_cols = [c for c in df.columns if "sl_tp" in c or "multiplier" in c]
            print(f"[DEBUG] Grid includes SL/TP-related columns: {debug_cols}")

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
            config_dict = self.filter_strategy_params(row.to_dict())
            print(f"[Step 8] Running backtest {idx+1}/{total} for strategy: {config_dict.get('strategy_name', 'N/A')}")
            try:
                result = run_backtest_VWAPStrategy(config_dict, df_5m)
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
            metrics_row["trades"] = trades
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

    def normalize_strategy_params(self, llm_params: dict) -> dict:
        """
        Merge raw LLM params into a safe VWAP strategy config dict,
        using the global param template for missing defaults.
        """
        full_config = deepcopy(VWAP_STRATEGY_PARAM_TEMPLATE)

        for k, v in llm_params.items():
            if k in full_config:
                full_config[k] = v
            else:
                print(f"[VWAPAgent] ⚠️ Ignoring unexpected param: {k}")

        return full_config

    import inspect

    def filter_strategy_params(self, config: dict) -> dict:
        """
        Clean config dict by:
        - Removing unserializable types (Backtrader line objects, etc.)
        - Filtering SL/TP params to only include those required by the sl_tp_function
        """
        allowed_types = (str, int, float, bool, list, dict, type(None))
        cleaned = {k: v for k, v in config.items() if isinstance(v, allowed_types)}

        # Optional SL/TP param filtering if function is known
        sl_tp_func_name = cleaned.get("sl_tp_function")
        if sl_tp_func_name:
            from backend.analyzer.analyzer_mcp_sl_tp_logic import SL_TP_FUNCTIONS  # or wherever your map lives
            sl_tp_func = SL_TP_FUNCTIONS.get(sl_tp_func_name)
            if sl_tp_func:
                expected_args = set(inspect.signature(sl_tp_func).parameters.keys())
                # Always keep standard fields
                standard_keys = {"sl_tp_function", "strategy_name", "entry_price", "entry_time"}
                cleaned = {k: v for k, v in cleaned.items() if k in expected_args or k in standard_keys}

        return cleaned

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
        # Normalize columns first
        df = self.normalize_ohlcv_columns(df)
        df = df.copy()

        # VWAP and bands
        # Combine 'date' and 'time' into a datetime column
        df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        # Set it as index
        df.set_index('Datetime', inplace=True)

        # Drop original columns if not needed
        df.drop(columns=['date', 'time'], inplace=True)

        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['vol'])
        # Optional cleanup: drop raw 'vwap' if exists to avoid confusion
        if 'vwap' in df.columns:
            print("[FIX] Dropping lowercase 'vwap' to prevent conflict with computed VWAP")
            df.drop(columns=['vwap'], inplace=True)

        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['VWAP_upper'] = df['VWAP'] + df['ATR_14']
        df['VWAP_lower'] = df['VWAP'] - df['ATR_14']

        # EMAs
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_20'] = ta.ema(df['close'], length=20)

        # DMI (includes ADX, +DI, -DI)
        dmi_df = df.ta.dm(length=14)
        adx_df = df.ta.adx(length=14)
        # print("[DEBUG] Existing cols dmi_df:", dmi_df.columns)
        # print("[DEBUG] Existing cols adx_df:", adx_df.columns)
        # print("[DEBUG] Existing cols before join:", df.columns)

        overlap = dmi_df.columns.intersection(adx_df.columns)
        if not overlap.empty:
            print(f"[DEBUG] Dropping overlap columns from adx_df: {overlap}")
            adx_df = adx_df.drop(columns=overlap)

        # Safe join
        df = df.join([dmi_df, adx_df])
        # Volume Z-score (based on 20-bar rolling mean)
        df['volume_zscore'] = (df['vol'] - df['vol'].rolling(20).mean()) / df['vol'].rolling(20).std()

        # Optional: preprocess ema_bias_filter, dmi_crossover, etc.
        df['ema_bias_filter'] = df['EMA_9'] > df['EMA_20']
        print("[DEBUG] Existing cols after join befor cross over", df.columns)
        df['dmi_crossover'] = df['DMP_14'] > df['DMN_14']
        # --- Place immediately after enrichment ---
        required = ["VWAP", "VWAP_upper", "VWAP_lower", "EMA_9", "EMA_20", "DMP_14", "DMN_14", "ADX_14",
                    "volume_zscore"]

        valid_mask = df[required].notna().all(axis=1) & (df[required] != 0.0).all(axis=1)
        first_valid_idx = valid_mask[valid_mask].index.min()

        if pd.isna(first_valid_idx):
            raise ValueError("No valid rows found with all indicators present!")

        df_cleaned = df.loc[first_valid_idx:]
        print(f"[CLEANUP] Trimmed df_5m_enriched to start at first valid indicator row: {first_valid_idx}. leaving with {len(df_cleaned)} bars for testing...")

        return df_cleaned

    def display_top_results_grid_and_metrics(self, summary_df, top_n=5, show_trades=True):
        """
        Display a heatmap of PnL and win rate for the top N strategies using Plotly.
        Show metrics and trades for the best strategy using AnalyzerDashboard.
        """
        if summary_df is None or summary_df.empty:
            print("[VWAPAgent] No summary DataFrame to display.")
            return
        # Determine sort column
        sort_col = 'PnL' if 'PnL' in summary_df.columns else ('win_rate' if 'win_rate' in summary_df.columns else None)
        if sort_col is None:
            print("[VWAPAgent] No PnL or win_rate column in summary_df.")
            return
        top_df = summary_df.sort_values(sort_col, ascending=False).head(top_n)
        # Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=top_df[[c for c in ['PnL', 'win_rate'] if c in top_df.columns]].values.T,
            x=top_df['strategy_name'] if 'strategy_name' in top_df.columns else top_df.index.astype(str),
            y=[c for c in ['PnL', 'win_rate'] if c in top_df.columns],
            colorscale='Viridis',
            colorbar=dict(title='Value'),
            showscale=True
        ))
        fig.update_layout(title=f"Top {top_n} VWAP Strategies: PnL & Win Rate Heatmap", xaxis_title="Strategy", yaxis_title="Metric")
        fig.show()
        # Best strategy
        best_row = top_df.iloc[0]
        trades = best_row.get('trades', None)
        # Display metrics and params
        dashboard = AnalyzerDashboard(pd.DataFrame(), pd.DataFrame())
        trade_df = pd.DataFrame(trades)

        metrics_df = dashboard.calculate_strategy_metrics_for_ui(trade_df)

        dashboard.display_strategy_and_metrics_side_by_side(metrics_df, best_row.to_dict())
        # Optionally show trades
        if show_trades and trades is not None and isinstance(trades, (list, pd.DataFrame)) and len(trades) > 0:
            trade_df = pd.DataFrame(trades) if isinstance(trades, list) else trades
            dashboard.plot_trades_and_predictions_regression_agent(trade_df, max_trades=50)

    @staticmethod
    def camel_to_snake(name):
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def normalize_params_dict(self, params_dict):
        normalized = {}
        for original, value in params_dict.items():
            normalized_key = self.camel_to_snake(original)
            print(f"[VWAPAgent] Normalized param key: {original}  {normalized_key}")
            normalized[normalized_key] = value
        return normalized

    def run(self, images, df_5m, user_params=None, top_n=5, df_1m=None):
        """
        Full pipeline: LLM call, grid build, backtest, rule generation, with debug prints after each major step.
        """

        # Step 1: LLM call
        llm_result = self.call_llm_with_images(images, user_params=user_params)
        raw_response_text = llm_result.get("llm_raw_response")
        llm_structured = self.parse_llm_response_text(raw_response_text)
        # Debug: print number of strategies received
        recommendations = llm_structured.get("strategy_recommendations", []) if llm_structured else []
        print(f"[VWAPAgent] Parsed {len(recommendations)} strategy recommendations")
        # Normalize param keys in all strategies and warn if missing params_to_optimize
        if recommendations:
            for rec in recommendations:
                if "params_to_optimize" in rec:
                    rec["params_to_optimize"] = self.normalize_params_dict(rec["params_to_optimize"])
                else:
                    print(f"[WARN] Strategy {rec.get('name')} missing 'params_to_optimize'")

        # # === Extract final top strategy ===
        # final_strategy = None
        # sort_col = 'PnL' if 'PnL' in summary_df.columns else ('win_rate' if 'win_rate' in summary_df.columns else None)
        # if sort_col:
        #     top_row = summary_df.sort_values(sort_col, ascending=False).head(1)
        #     if not top_row.empty:
        #         final_strategy = top_row.iloc[0].to_dict()
        #

        # --- Optimization file flow (if present) ---
        optimization_files = getattr(self, 'optimization_files', None)
        optimization_cost_metadata = None
        if optimization_files:
            dashboard = AnalyzerDashboard(pd.DataFrame(), pd.DataFrame())
            # 1. Parse optimization files
            opt_result = dashboard.parse_optimization_reports_from_tradestation_to_df(optimization_files)
            grid_df = opt_result['grid_df']
            # 2. Prepare for LLM
            llm_input_df = self.prepare_optimization_results_for_llm(grid_df)
            # 3. Retrieve session bias (assume self.session_bias or similar)
            session_bias = getattr(self, 'session_bias', None)
            # 4. Format prompt
            prompt_template = """
            [VWAP OPTIMIZATION]
            Bias: {{BIAS}}
            Top parameter sets:
            {llm_input}
            """
            llm_input = llm_input_df.to_string(index=False)
            prompt_text = prompt_template.replace("{{BIAS}}", str(session_bias)).replace("{llm_input}", llm_input)
            # 5. Call LLM (text-only)
            opt_llm = self.call_llm_text(prompt_text)
            print("[VWAPAgent] LLM optimization response:\n", opt_llm.get("llm_raw_response"))
            # Capture optimization LLM cost metadata for UI
            optimization_cost_metadata = {
                "model_name": opt_llm.get("model_name"),
                "provider": opt_llm.get("provider"),
                "llm_cost_usd": opt_llm.get("llm_cost_usd"),
                "llm_token_usage": opt_llm.get("llm_token_usage"),
            }

        return {
            "llm_raw_response": llm_result.get("llm_raw_response"),
            "llm_structured": llm_structured,
            "model_name": llm_result.get("model_name"),
            "provider": llm_result.get("provider"),
            "llm_cost_usd": llm_result.get("llm_cost_usd"),
            "llm_token_usage": llm_result.get("llm_token_usage"),
            "num_images": len(images),
            "parameter_grid": [],
            "backtest_summary": [],
            "natural_language_rules": None,
            "final_strategy": None,
            "optimization_cost_metadata": optimization_cost_metadata
        }

    def run_optimizations_only(self, optimization_files):
        """
        Process optimization .txt files without requiring images and return LLM recommendation.
        """
        print(f"[VWAPAgent] DEBUG: Number of optimization files sent to LLM: {len(optimization_files)}")
        dashboard = AnalyzerDashboard(pd.DataFrame(), pd.DataFrame())
        opt_result = dashboard.parse_optimization_reports_from_tradestation_to_df(optimization_files)
        grid_df = opt_result['grid_df']
        llm_input_df = self.prepare_optimization_results_for_llm(grid_df)
        session_bias = getattr(self, 'session_bias', None)
        prompt_template = VWAP_OPTIMIZATION_PROMPT
        llm_input = llm_input_df.to_string(index=False)
        prompt_text = prompt_template.replace("{{BIAS}}", str(session_bias)).replace("{llm_input}", llm_input)
        llm_result = self.call_llm_text(prompt_text)

        try:
            structured = self.parse_llm_response_text(llm_result.get("llm_raw_response") or "")
        except Exception:
            structured = None
        return {
            "llm_raw_response": llm_result.get("llm_raw_response"),
            "llm_structured": structured,
            "model_name": llm_result.get("model_name"),
            "provider": llm_result.get("provider"),
            "llm_cost_usd": llm_result.get("llm_cost_usd"),
            "llm_token_usage": llm_result.get("llm_token_usage"),
            "num_images": 0,
            "final_strategy": None,
            "optimization_cost_metadata": {
                "model_name": llm_result.get("model_name"),
                "provider": llm_result.get("provider"),
                "llm_cost_usd": llm_result.get("llm_cost_usd"),
                "llm_token_usage": llm_result.get("llm_token_usage"),
            }
        }

    def filter_strategy_params(self, config: dict) -> dict:
        """
        Filters the config dict to only include params declared in VWAPScalpingStrategy.params.
        Prevents Backtrader from breaking due to unknown kwargs.
        """

        allowed_keys = [k for k, _ in VWAPBounceStrategy.params._getitems()]
        return {k: v for k, v in config.items() if k in allowed_keys}

    def prepare_optimization_results_for_llm(self, grid_df):
        """
        Filter and prepare TradeStation optimization results for LLM input.
        Args:
            grid_df (pd.DataFrame): Combined DataFrame from parse_optimization_reports_from_tradestation_to_df.
        Returns:
            pd.DataFrame: Filtered and sorted DataFrame ready for LLM.
        """
        import pandas as pd
        df = grid_df.copy()
        # === Per-line filters (toggle on/off as needed) ===
        if 'ProfitFactor' in df.columns:
            df = df[df['ProfitFactor'] > 1.5]
        if 'MaxStrategyDrawdown' in df.columns:
            df = df[df['MaxStrategyDrawdown'] < 20]
        if 'TotalNumberOfTrades' in df.columns:
            df = df[df['TotalNumberOfTrades'] > 10]
        if 'PercentProfitable' in df.columns:
            df = df[df['PercentProfitable'] > 50]
        # === Sort and take top 5 per strategy ===
        if 'strategy' in df.columns and 'NetProfit' in df.columns:
            df = df.sort_values(['strategy', 'NetProfit'], ascending=[True, False])
            df = df.groupby('strategy').head(5).reset_index(drop=True)
        return df
