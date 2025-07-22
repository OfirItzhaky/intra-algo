import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load from .env

today = datetime.today().strftime("%Y-%m-%d")
#todo: Alternative LLMs
# provider- google
# gemini-1.5-pro-latest
#  provider openai
# gpt-4o
# gpt-4.1 provider openai
# Choose Model:
MODEL = "gemini-1.5-pro-latest"

company = "google" if MODEL=="gemini-1.5-pro-latest" else "openai"
CONFIG = {
    "model_provider": company,
    "model_name": MODEL,
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),

    "newsapi_key": os.getenv("NEWSAPI_KEY"),
    "finnhub_key": os.getenv("FINNHUB_KEY"),
    "fmp_key": os.getenv("FMP_KEY"),

    # 🔹 Reddit API credentials (added for sentiment fetcher)
    "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
    "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "reddit_username": os.getenv("REDDIT_USERNAME"),
    "reddit_password": os.getenv("REDDIT_PASSWORD"),
    "reddit_user_agent": os.getenv("REDDIT_USER_AGENT"),

    "markets": ["US"],
    "symbols": ["SPY", "QQQ", "AAPL", "NVDA"],
    "focus_sectors": ["Technology", "Healthcare", "Energy"],

    "only_major_events": False,
    "print_live_summary": True,
    "copy_to_clipboard": False,

    "save_directory": "research_outputs",
    "date": today,
}

SUMMARY_CACHE = {}
EVENT_CACHE = {}

# --- Explicit Regression Strategy Constants ---
REGRESSION_STRATEGY_DEFAULTS = {
    'target_ticks': 10,
    'stop_ticks': 10,
    'tick_size': 0.25,
    'tick_value': 1.25,
    'contract_size': 1,
    'initial_cash': 10000.0,
    'session_start': '01:00',
    'session_end': '23:00',
    'maxdailyprofit_dollars': 36.0,
    'maxdailyloss_dollars': -36.0,
    'slippage': 0.0,
    'min_dist': 1.0,
    'max_dist': 20.0,
    'min_classifier_signals': 0
}

VWAP_STRATEGY_DEFAULTS = {
    "tick_size": 0.25,
    "tick_value": 1.25,
    "contract_size": 1,
    "initial_cash": 10000.0,
    "slippage": 0.0,
    "session_start": "01:00",
    "session_end": "23:00",
    "maxdailyprofit_dollars": 36.0,
    "maxdailyloss_dollars": -36.0,

}
VWAP_STRATEGY_PARAM_TEMPLATE = {
    "strategy_name": None,
    "bias": None,
    "vwap_distance_pct": None,
    "volume_zscore_min": None,
    "volume_increase_pct": None,
    "volume_increase": None,
    "ema_bias_filter": None,
    "dmi_crossover": None,
    "dmi_bias_filter": None,
    "dmi_trend": None,
    "dmi_positive_slope": None,
    "dmi_divergence": None,
    "ema_vwap_cross": None,
    "ema_vwap_relationship": None,
    "vwap_reclaim_time": None,
    "vwap_reclaim_speed": None,
    "volume_confirmation": None,
    "dmi_plus_min": None,
    "entry_conditions": None,
    "stop_loss_rule": None,
    "take_profit_rule": None,
    "risk_type": None,
}

