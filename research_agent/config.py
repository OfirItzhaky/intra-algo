import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load from .env

today = datetime.today().strftime("%Y-%m-%d")

CONFIG = {
    "model_provider": "openai",
    "model_name": "gpt-4o",

    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),

    "newsapi_key": os.getenv("NEWSAPI_KEY"),
    "finnhub_key": os.getenv("FINNHUB_KEY"),
    "fmp_key": os.getenv("FMP_KEY"),

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
