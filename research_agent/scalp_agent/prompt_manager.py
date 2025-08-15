from research_agent.config import MAX_SUGGESTED_VWAP_STRATEGIES
VWAP_STRATEGY_OPTIMIZATION_PARAMS = {
    "vwap_bounce_01_sl_candle_low_tp_2R": {
        "VWAPDistancePct": {"min": 0.1, "max": 0.5, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_bounce_02_sl_1.2atr14_tp_2R": {
        "VWAPDistancePct": {"min": 0.1, "max": 0.5, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_bounce_03_sl_candle_low_tp_ema9": {
        "VWAPDistancePct": {"min": 0.1, "max": 0.5, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_bounce_04_sl_1.2atr14_tp_ema9": {
        "VWAPDistancePct": {"min": 0.1, "max": 0.5, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_reclaim_05_sl_candle_low_tp_2R": {
        "VWAPCrossBackPct": {"min": 0.05, "max": 0.3, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 1.0, "max": 3.0, "step": 0.5}
    },
    "vwap_reclaim_06_sl_entry_zone_tp_vwaploss": {
        "VWAPCrossBackPct": {"min": 0.05, "max": 0.3, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 1.0, "max": 3.0, "step": 0.5}
    },
    "vwap_reclaim_07_sl_atr_tp_1.5R": {
        "VWAPCrossBackPct": {"min": 0.05, "max": 0.3, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 1.0, "max": 3.0, "step": 0.5}
    },
    "vwap_compression_08_sl_range_tp_fade_exit": {
        "RangePctLast3": {"min": 0.2, "max": 0.5, "step": 0.05},
        "ATRDropRatio": {"min": 0.5, "max": 0.9, "step": 0.1}
    },
    "vwap_compression_09_sl_atr_tp_fade_exit": {
        "RangePctLast3": {"min": 0.2, "max": 0.5, "step": 0.05},
        "ATRDropRatio": {"min": 0.5, "max": 0.9, "step": 0.1}
    },
    "vwap_ema_cross_10_sl_candle_low_tp_2R": {
        "EmaCrossStrengthLong": {"min": 0.1, "max": 0.5, "step": 0.1},
        "EmaSlopeThresholdLong": {"min": 0.0005, "max": 0.003, "step": 0.0005},
        "VolumeZScoreThresholdLong": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_ema_cross_11_sl_atr_tp_1.5R": {
        "EmaCrossStrengthLong": {"min": 0.1, "max": 0.5, "step": 0.1},
        "EmaSlopeThresholdLong": {"min": 0.0005, "max": 0.003, "step": 0.0005},
        "VolumeZScoreThresholdLong": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_ema_cross_12_sl_candle_low_tp_trail": {
        "EmaCrossStrengthLong": {"min": 0.1, "max": 0.5, "step": 0.1},
        "EmaSlopeThresholdLong": {"min": 0.0005, "max": 0.003, "step": 0.0005},
        "VolumeZScoreThresholdLong": {"min": 0.5, "max": 3.0, "step": 0.5}
    },
    "vwap_trend_13_sl_candle_low_tp_2R": {
        "EMABiasPct": {"min": 0.1, "max": 0.5, "step": 0.1},
        "PullbackDepthPct": {"min": 0.1, "max": 0.5, "step": 0.1}
    },
    "vwap_trend_14_sl_ema20_tp_1.5R": {
        "EMABiasPct": {"min": 0.1, "max": 0.5, "step": 0.1},
        "PullbackDepthPct": {"min": 0.1, "max": 0.5, "step": 0.1}
    },
    "vwap_trend_15_sl_candle_low_tp_trail_vwap": {
        "EMABiasPct": {"min": 0.1, "max": 0.5, "step": 0.1},
        "PullbackDepthPct": {"min": 0.1, "max": 0.5, "step": 0.1}
    },
    "vwap_fade_16_sl_wick_tp_2R": {
        "RejectionLookback": {"min": 1, "max": 4, "step": 1},
        "WickRatioThreshold": {"min": 0.4, "max": 0.8, "step": 0.1},
        "VolumeZScoreThreshold": {"min": 0.2, "max": 2.0, "step": 0.2}
    },
    "vwap_breakfail_17_sl_atr_tp_vwap": {
        "FailureRangePct": {"min": 0.2, "max": 0.5, "step": 0.05},
        "VolumeZScoreThreshold": {"min": 0.5, "max": 2.0, "step": 0.25}
    },
    "vwap_magnet_18_sl_range_tp_vwapband": {
        "RangeCoilPct": {"min": 0.1, "max": 0.4, "step": 0.05},
        "MeanReversionStrength": {"min": 0.5, "max": 1.5, "step": 0.25}
    }
}

VWAP_STRATEGY_OPTIMIZATION_PARAMS_NO_VALUES = {
  "vwap_bounce_01_sl_candle_low_tp_2R": ["VWAPDistancePct", "VolumeZScoreThreshold"],
  "vwap_bounce_02_sl_1.2atr14_tp_2R": ["VWAPDistancePct", "VolumeZScoreThreshold", "StopATRMultiplier"],
  "vwap_bounce_03_sl_candle_low_tp_ema9": ["VWAPDistancePct", "VolumeZScoreThreshold"],
  "vwap_bounce_04_sl_1.2atr14_tp_ema9": ["VWAPDistancePct", "VolumeZScoreThreshold", "StopATRMultiplier"],
  "vwap_reclaim_05_sl_candle_low_tp_2R": ["VWAPCrossBackPct", "VolumeZScoreThreshold", "LookbackBars"],
    "vwap_reclaim_06_sl_entry_zone_tp_vwaploss": ["VWAPCrossBackPct", "VolumeZScoreThreshold"],
    "vwap_reclaim_07_sl_atr_tp_1.5R": ["VWAPCrossBackPct", "VolumeZScoreThreshold"],

  "vwap_compression_08_sl_range_tp_fade_exit": ["RangePctThreshold", "ATRDropRatio", "VolumeZScoreThreshold", "ExitVolumeZScoreFade"],
"vwap_compression_09_sl_atr_tp_fade_exit": ["ATRStopMultiplier", "ATRDropRatio", "VolumeZScoreThreshold", "ExitVolumeZScoreFade"],

    "vwap_ema_cross_10_sl_candle_low_tp_2R": ["EmaCrossStrengthLong", "EmaSlopeThresholdLong",
                                              "VolumeZScoreThresholdLong"],
    "vwap_ema_cross_11_sl_atr_tp_1.5R": ["EmaCrossStrengthLong", "EmaSlopeThresholdLong", "VolumeZScoreThresholdLong"],
    "vwap_ema_cross_12_sl_candle_low_tp_trail": ["EmaCrossStrengthLong", "EmaSlopeThresholdLong",
                                                 "VolumeZScoreThresholdLong"],

    "vwap_trend_13_sl_candle_low_tp_2R": ["EMA9SlopeThreshold", "MaxPullbackPct"],
"vwap_trend_14_sl_ema20_tp_1.5R": ["EMA9SlopeThreshold", "MaxPullbackPct"],
"vwap_trend_15_sl_candle_low_tp_trail_vwap": ["EMA9SlopeThreshold", "MaxPullbackPct"],

  "vwap_fade_16_sl_wick_tp_2R": ["WickRatioThreshold", "VolumeZScoreThreshold"],
"vwap_breakfail_17_sl_atr_tp_vwap": ["WickRatioThreshold", "VWAPBufferPct"],
"vwap_magnet_18_sl_range_tp_vwapband": ["MaxRangeConversionTicks", "VWAPBufferPct"]

}



VWAP_PROMPT_SINGLE_IMAGE = f"""
You are a senior intraday futures trader and VWAP-based scalping strategist with deep expertise in 18 known VWAP strategies.

🎯 YOUR GOAL:
Analyze the provided multi-timeframe chart image and return only the VWAP strategies that fit the structure and bias shown. 
For each valid strategy, also suggest optimization parameters to tune for today's session.

---

1️⃣ **Determine Intraday Bias**  
Pick one of: ["bullish", "bearish", "range", "volatile_chop"]

---

2️⃣ **Evaluate All 18 VWAP Strategies**

You have access to the full strategy library, grouped as follows:

🔁 VWAP Bounce (01–04) – Price pulls back near VWAP and bounces with structure and volume

📈 VWAP Reclaim (05–07) – Price dips under VWAP, then reclaims with momentum

📉 VWAP Compression (08–09) – Tight range forms near VWAP, sets up breakout/fade

🔄 VWAP EMA Cross (10–12) – EMA(9) crosses VWAP with slope and volume confirmation

🔼 VWAP Trend (13–15) – Price rides VWAP in the direction of a clean EMA trend

🔽 VWAP Reversal (16–17) – Sharp rejection from VWAP after extended move or fakeout

🧲 VWAP Magnet (18) – Price drifts from VWAP without trend, then mean-reverts to value

Each strategy uses a unique stop-loss and target method (e.g., fixed R, trailing EMA, VWAP band), embedded in its name (e.g., sl_1.2atr14_tp_2R).



3️⃣ **Suggest Parameter Ranges**

Use this constant to determine which input parameters to optimize for each strategy:

{VWAP_STRATEGY_OPTIMIZATION_PARAMS_NO_VALUES}
You decide appropriate min, max, and step values for each based on the actual chart structure.
Use trading-based cues such as:
• size of pullbacks,
• strength of volume spikes,
• slope of EMAs,
• range tightness,
• ATR contraction,
• number of failed breakouts,
…to guide your grid values.
⚠️ Keep grid sizes reasonable. Usually 2–3 parameters per strategy. Avoid overly granular steps.

4️⃣ Return only valid strategies

Only include strategies where structure clearly fits the image. For each:

"name": Strategy name (e.g., "vwap_bounce_01_sl_candle_low_tp_2R")

"recommend": true

"rank": Integer (starting at 1 for best fit)

"reason": Clear explanation based on chart structure

"params_to_optimize": Dict with param: {{{{min, max, step}}}}


Example:

{{
  "name": "vwap_bounce_01_sl_candle_low_tp_2R",
  "recommend": true,
  "rank": 1,
  "reason": "5m shows bounce from lower VWAP band with EMA support and rising volume",
  "params_to_optimize": {{
    "vwap_distance_pct": {{{{"min": 0.1, "max": 0.5, "step": 0.1}}}},
    "volume_zscore": {{{{"min": 0.5, "max": 1.5, "step": 0.25}}}}
  }}
}}

🧠 Return a full JSON like:

{{
  "bias": "bullish",
  "strategy_recommendations": [ ... ]
}}
📌 CHART OVERVIEW (1 image with 4 panels)

🟥 Daily – VWAP, Volume MA, ATR(14), DMI
🟪 60-Min – EMA(9/20), VWAP, Volume, ATR, DMI
🟨 15-Min – Candles, Volume MA, DMI, ATR
🟦 5-Min – VWAP bands, EMA(9/20), Volume, DMI ← Main execution chart

🔁 Additional Rules:

Return up to {MAX_SUGGESTED_VWAP_STRATEGIES}. If none apply, return an empty list.

Use 5-minute chart as execution anchor. Confirm bias with higher timeframes.

Do not include entry/exit logic — only evaluate structure and suggest parameters to optimize.

Think like a scalper risking real capital. Be precise, realistic, and strict.

"""




VWAP_PROMPT_4_IMAGES = f"""
You are a senior intraday futures trader and VWAP-based scalping strategist with deep expertise in 18 known VWAP strategies.

🎯 YOUR GOAL:
Analyze the 4 uploaded multi-timeframe chart images and return only the VWAP strategies that fit the structure and bias shown.
For each valid strategy, also suggest optimization parameters to tune for today's session.

---

1️⃣ **Determine Intraday Bias**  
Start with the smallest time frame one (for example 5 min), then work backwards for bias confirmation.

Pick one of: ["bullish", "bearish", "range", "volatile_chop"]

---

2️⃣ **Evaluate All 18 VWAP Strategies**

You have access to the full strategy library, grouped as follows:

🔁 VWAP Bounce (01–04) – Price pulls back near VWAP and bounces with structure and volume  
📈 VWAP Reclaim (05–07) – Price dips under VWAP, then reclaims with momentum  
📉 VWAP Compression (08–09) – Tight range forms near VWAP, sets up breakout/fade  
🔄 VWAP EMA Cross (10–12) – EMA(9) crosses VWAP with slope and volume confirmation  
🔼 VWAP Trend (13–15) – Price rides VWAP in the direction of a clean EMA trend  
🔽 VWAP Reversal (16–17) – Sharp rejection from VWAP after extended move or fakeout  
🧲 VWAP Magnet (18) – Price drifts from VWAP without trend, then mean-reverts to value

Each strategy uses a unique stop-loss and target method (e.g., fixed R, trailing EMA, VWAP band), embedded in its name (e.g., sl_1.2atr14_tp_2R).

Only include a strategy if the 5m chart clearly supports the entry conditions. Other timeframes are secondary!!!
---

3️⃣ **Suggest Parameter Ranges**

Use this constant to determine which input parameters to optimize for each strategy:

{VWAP_STRATEGY_OPTIMIZATION_PARAMS_NO_VALUES}

You decide appropriate min, max, and step values for each based on the actual chart structure.

Use trading-based cues such as:
• size of pullbacks,  
• strength of volume spikes,  
• slope of EMAs,  
• range tightness,  
• ATR contraction,  
• number of failed breakouts,  
…to guide your grid values.

⚠️ Keep grid sizes reasonable. Usually 2–3 parameters per strategy. Avoid overly granular steps.

---

4️⃣ Return only valid strategies

Only include strategies where structure clearly fits the images. For each:

"name": Strategy name (e.g., "vwap_bounce_01_sl_candle_low_tp_2R")  
"recommend": true  
"rank": Integer (starting at 1 for best fit)  
"reason": Clear explanation based on chart structure  
"params_to_optimize": Dict with param: {{{{min, max, step}}}}

Example:

{{
  "name": "vwap_bounce_01_sl_candle_low_tp_2R",
  "recommend": true,
  "rank": 1,
  "reason": "5m chart shows VWAP bounce with volume spike and rising EMA slope",
  "params_to_optimize": {{
    "vwap_distance_pct": {{{{"min": 0.1, "max": 0.5, "step": 0.1}}}},
    "volume_zscore": {{{{"min": 0.5, "max": 1.5, "step": 0.25}}}}
  }}
}}

⚠️ When in doubt, adjust parameters to fit the microstructure on the 5m chart — that’s the chart to trade.

🧠 Return a full JSON like:

{{
  "bias": "bullish",
  "strategy_recommendations": [ ... ]
}}

---

🖼 IMAGE PANELS (uploaded as 4 separate images):

🟥 Image 1 – Daily: VWAP, Volume MA, ATR, DMI  
🟪 Image 2 – 60-Min: EMA(9/20), VWAP, Volume, ATR, DMI  
🟨 Image 3 – 15-Min: Candles, Volume MA, DMI, ATR  
🟦 Image 4 – 5-Min: VWAP bands, EMA(9/20), Volume MA, ATR, DMI ← Main execution chart

---

🔁 Additional Instructions:

Return up to {MAX_SUGGESTED_VWAP_STRATEGIES} strategies. If none apply, return an empty list.  
Use the 5-minute chart as the execution anchor. Confirm bias using the higher timeframes.  
Do not include entry/exit logic — only evaluate structure and recommend strategy + param ranges.

Act like a scalper trading live capital. Be strict, context-aware, and technically accurate.
"""

LLM_PROMPT_OPTIMIZATION_SELECTION = f"""
You are a senior VWAP scalping expert reviewing TradeStation optimization results from a live intraday session.

📊 These results come from different VWAP strategies that were backtested on the most recent market session using candidate parameter grids.

Your goal: 
👉 For each strategy, **select up to 2 optimal parameter configurations** that best fit the current market bias and show solid risk-adjusted performance.

---

📈 **Session Bias**: "{{BIAS}}"  
This bias was detected using 5m/15m/60m charts and confirms the overall structure of the session (bullish, bearish, range, or volatile).

---

🏁 **Selection Criteria** (you may prioritize these as needed):
• High ProfitFactor (>1.5)
• Win rate above 50%
• Lower drawdowns (MaxStrategyDrawdown)
• Solid NetProfit but not at the expense of deep risk
• Reasonable trade count (avoid 1-off lucky setups)

---

🧪 **What You Get**:
You'll receive a table (per strategy) with:
• Parameter values tested (e.g. VWAPDistancePct, VolumeZScoreThreshold, etc.)  
• Performance metrics (ProfitFactor, NetProfit, WinRate, etc.)  
• Strategy name for each row

---

📤 **What You Should Return**:
Return a JSON structure like:

{{
  "strategy_picks": [
    {{
      "strategy": "vwap_bounce_01_sl_candle_low_tp_2R",
      "rank": 1,
      "reason": "Strong profit factor, 68% win rate, fits bullish bounce context",
      "params": {{
        "VWAPDistancePct": 0.2,
        "VolumeZScoreThreshold": 1.0
      }}
    }},
    ...
  ]
}}
• Return up to 2 configs per strategy (sorted by preference).
• Do NOT fabricate values. Only select from the provided grid.
• If no config meets criteria, skip that strategy.

Think like a real trader — you’re managing live capital in a fast session. Be realistic, not theoretical.
"""


PROMPT_REGRESSION_AGENT = """
You are a trading strategist assistant. Your task is to select the best strategies from a 256-strategy grid and an optional market bias summary.

🎯 Your Goal:
Return exactly **three trading strategies**, selected based on the data, and output them in a **fixed JSON format**. This format must always be followed **exactly as shown below** to ensure consistency.

🧠 Inputs:
- A strategy grid containing 256 rows, each with metrics like: win rate, profit factor, max drawdown, avg daily PnL, etc.
- An optional market bias text summary that may include sector flow, macro bias, symbol news, or support/resistance levels.

📋 Selection Criteria:
- Choose based on meaningful metrics (not only PnL) — including stability, risk, win rate, and drawdown.
- You may weight metrics differently if market bias suggests strong long/short skew or risk-aversion.
- Use judgment to combine multiple signals into effective strategy logic.

📤 **Output Format – This must be strictly followed**:
Return your response as a JSON with the following structure:


{
  "top_strategies": [
    {
      "name": "Volatile Shorts",
      "logic": "Trade when predicted short > 0.6 and candle color is red. Use only if volume spike > 5%.",
      "direction": "short",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.25,
        "win_rate": 52.3,
        "avg_daily_pnl": 5.2,
        "max_drawdown": 90.0
      },
      "rationale": "This strategy favors short trades in high-volume environments. It has stable win rate and low drawdown, making it ideal in bearish or volatile conditions."
    },
    {
      "name": "Balanced Intraday",
      "logic": "Trade when either long/short predicted > 0.5 and candle matches direction. No volume filter.",
      "direction": "both",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.18,
        "win_rate": 48.7,
        "avg_daily_pnl": 6.9,
        "max_drawdown": 100.0
      },
      "rationale": "This is a general-purpose strategy that performs well on both sides with decent stability. Recommended for trend-neutral sessions."
    },
    {
      "name": "Momentum Longs",
      "logic": "Trade when predicted long > 0.7 and min_volume_pct_change > 3%. Only on green candles.",
      "direction": "long",
      "stop_loss_ticks": 10,
      "take_profit_ticks": 10,
      "key_metrics": {
        "profit_factor": 1.32,
        "win_rate": 56.2,
        "avg_daily_pnl": 7.5,
        "max_drawdown": 70.0
      },
      "rationale": "Best performing long-biased strategy with high win rate and consistent daily returns. Ideal for strong bullish sessions."
    }
  ]
}

📌 Rules:

Do not return anything outside the JSON block.

Return the output as a clean JSON object (not markdown, not wrapped in triple backticks). Your response should start directly with { and be parseable using json.loads() in Python. Do not include extra text, formatting, or explanation

Always include exactly 3 strategies unless instructed otherwise.

Ensure keys and structure are identical in casing and order.

If market bias is empty, ignore it. If present, use it to prioritize strategy alignment.

Below is the strategy grid and bias summary:

STRATEGY GRID:
{grid_json}

BIAS SUMMARY:
{bias_str}
"""

VWAP_OPTIMIZATION_PROMPT = """
You are a trading assistant helping select the best intraday VWAP strategy based on **backtest results** from a recent session.

Each section below represents a different VWAP strategy with its parameter grid. For each, perform a clear evaluation and recommendation.

---

📊 **What you are analyzing:**
These are in-sample backtest results. Each row contains a different parameter combination and metrics like:
- Net PnL
- Profit Factor (PF)
- Expectancy
- Max Drawdown

Your job is to:
1. Identify strong performers
2. Recommend the best parameter set
3. Explain trade logic and risk caveats

---

🧪 **Section Format Per Strategy**

Use this exact format **per strategy**:

**Strategy Name:** `vwap_trend_13_sl_candle_low_tp_2R`

**Overall Assessment:**
- Many parameter sets show consistent high performance
- Robust across small param shifts (e.g. EMA slope or pullback %)

**Top Parameter Sets:**

(Use pipe `|` as column separator. Do NOT wrap in Markdown.)

| Test ID | Param Summary                 | Net PnL | PF  | Expectancy | Max Drawdown |
|---------|-------------------------------|---------|-----|------------|---------------|
| 505     | EMA9Slope: 1.0, Pullback: 1.0 | $705.00 | 1.45| 212        | -322.50       |
| 504     | EMA9Slope: 0.9, Pullback: 1.0 | $687.50 | 1.42| 206        | -305.00       |
| 502     | EMA9Slope: 1.0, Pullback: 0.9 | $661.25 | 1.38| 197        | -290.00       |

**Suggested Parameter Set:**
- `EMA9SlopeThreshold`: 1.0  
- `MaxPullbackPct`: 1.0  
- Reason: Strongest NetPnL and expectancy with tolerable drawdown

**Trade Logic:**
Enter long when price reclaims VWAP after a <1% pullback and EMA9 slope > 1.0. TP at 2R, SL at candle low.

**Notes:**
High drawdown relative to risk; good candidate for size scaling with caution.

---

📌 **Repeat this block for each strategy provided in the input**.

--- 

✅ **Final Summary Section (After All Strategies)**

After analyzing all strategies, return a summary with:

- 📈 Best overall strategy for today
- ✅ Exact parameter values
- 🔁 1-sentence trade logic
- ⚠️ Risk management warning
- 🧠 Use today’s bias: `{{BIAS}}` and metric stability across tests to decide.

---

🧠 Output Rules:
- Keep JSONs strict: lowercase keys, snake_case or camelCase as shown
- Use true/false for booleans, not "true"/"false"
- Do not invent values. Only use values from the backtest grid
- Use realistic trading language — this is for live capital

---

📥 **Grid Data Input:**
{{llm_input}}
"""
