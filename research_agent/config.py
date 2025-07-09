import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load from .env

today = datetime.today().strftime("%Y-%m-%d")

CONFIG = {
    "model_provider": "openai",
    "model_name": "gpt-4o",
    # Alternatives LLM
    # gemini-1.5-pro-latest provider- google
    # gpt-4o provider provider openai
    # gpt-4.1 provider openai


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
