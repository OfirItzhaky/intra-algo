import itertools
import os
import base64
import imghdr

import pandas as pd
from copy import deepcopy

from research_agent.scalp_agent.prompt_manager import VWAP_PROMPT_SINGLE_IMAGE, VWAP_PROMPT_4_IMAGES, OPTION_A_SL_TP_TRANSLATOR_PROMPT
from research_agent.config import CONFIG, VWAP_STRATEGY_PARAM_TEMPLATE
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

    def translate_risk_blocks_option_a(self, llm_structured):
        """
        For each suggested_strategy, extract the risk_management block and call the LLM with the translator prompt.
        Attach sl_tp_function and numeric parameters to each strategy dict.
        """
        import json
        strategies = llm_structured.get('suggested_strategies', [])
        # List of available SL/TP functions (should match backend)
        function_list = [
            "sl_tp_from_r_multiple",
            "sl_tp_fixed_dollar",
            "sl_tp_swing_low_high",
            "sl_tp_dynamic_atr",
            "sl_tp_bar_by_bar_trailing",
            "sl_tp_vwap_bands",
            "sl_tp_custom_zscore",
            "sl_tp_pivot_level_trailing",
            "sl_tp_volume_spike",
            "sl_tp_dmi_bias",
            "sl_tp_ema_cross",
            "sl_tp_bias_based",
            "sl_tp_trailing_update"
        ]
        for strat in strategies:
            risk = strat.get('risk_management', {})
            stop_loss = risk.get('stop_loss', '')
            take_profit = risk.get('take_profit', '')
            risk_type = risk.get('risk_type', '')
            function_list_str = "\n".join(function_list)
            prompt = OPTION_A_SL_TP_TRANSLATOR_PROMPT.format(
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_type=risk_type,
                function_list=function_list_str
            )

            # Call LLM (same provider as main agent)
            try:
                if self.provider == "gemini":
                    api_key = CONFIG.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        print("[VWAPAgent] GEMINI_API_KEY not set for SL/TP translation.")
                        continue
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                    body = {"contents": [{"parts": [{"text": prompt}]}]}
                    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                    response = requests.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    response_data = response.json()
                    text = response_data["candidates"][0]["content"]["parts"][0]["text"]
                elif self.provider == "openai":
                    api_key = CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        print("[VWAPAgent] OPENAI_API_KEY not set for SL/TP translation.")
                        continue
                    endpoint = "https://api.openai.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 400
                    }
                    response = requests.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    response_data = response.json()
                    text = response_data["choices"][0]["message"]["content"]
                else:
                    print(f"[VWAPAgent] Unknown provider {self.provider} for SL/TP translation.")
                    continue
                # Parse LLM response (should be JSON)
                try:
                    parsed = json.loads(text.strip().split('```')[-1])
                    sl_tp_func = parsed.get("sl_tp_function")
                    params = parsed.get("parameters", {})
                    if sl_tp_func:
                        strat["sl_tp_function"] = sl_tp_func
                        for k, v in params.items():
                            strat[k] = v
                except Exception as parse_exc:
                    print(f"[VWAPAgent] Failed to parse SL/TP translation for strategy {strat.get('name')}: {parse_exc}\nRaw: {text}")
            except Exception as e:
                print(f"[VWAPAgent] SL/TP translation failed for strategy {strat.get('name')}: {e}")

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
        # Combine 'date' and 'time' into a datetime column
        df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        # Set it as index
        df.set_index('Datetime', inplace=True)

        # Drop original columns if not needed
        df.drop(columns=['date', 'time'], inplace=True)

        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['vol'])
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

        return df

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

        # === Option A SL/TP Translator Integration ===
        self.translate_risk_blocks_option_a(llm_structured)

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
        # === Display top results and metrics ===
        self.display_top_results_grid_and_metrics(self.summary_df)

        # === Extract final top strategy ===
        final_strategy = None
        sort_col = 'PnL' if 'PnL' in summary_df.columns else ('win_rate' if 'win_rate' in summary_df.columns else None)
        if sort_col:
            top_row = summary_df.sort_values(sort_col, ascending=False).head(1)
            if not top_row.empty:
                final_strategy = top_row.iloc[0].to_dict()

        return {
            "llm_raw_response": llm_result.get("llm_raw_response"),
            "llm_structured": llm_structured,
            "model_name": llm_result.get("model_name"),
            "provider": llm_result.get("provider"),
            "llm_cost_usd": llm_result.get("llm_cost_usd"),
            "llm_token_usage": llm_result.get("llm_token_usage"),
            "num_images": llm_result.get("num_images"),
            "prompt_type": llm_result.get("prompt_type"),
            "parameter_grid": grid_df.to_dict(orient="records"),
            "backtest_summary": summary_df.to_dict(orient="records"),
            "natural_language_rules": rules,
            "final_strategy": final_strategy
        }

