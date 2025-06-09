import requests
from typing import Optional
from datetime import datetime
import praw

MACRO_KEYWORDS = [
    "Fed", "inflation", "rates", "recession", "macro", "FOMC", "CPI", "PPI", "interest", "central bank", "bond", "yields", "unemployment", "jobs", "growth", "oil", "energy"
]

def fetch_macro_headlines_summary(config: dict, max_headlines: int = 5) -> Optional[str]:
    """
    Fetches macro-related news headlines from NewsAPI and returns a string block of 3-5 relevant headlines.
    Args:
        config (dict): Config with NewsAPI key.
        max_headlines (int): Max number of headlines to return.
    Returns:
        str: Block of macro-relevant headlines (title + description if available), or None if unavailable.
    """
    api_key = config.get("newsapi_key")
    if not api_key:
        return None
    # Use NewsAPI 'everything' endpoint for keyword search
    url = (
        "https://newsapi.org/v2/everything?"
        "q=" + "%20OR%20".join(MACRO_KEYWORDS) +
        "&language=en&pageSize=10&sortBy=publishedAt&apiKey=" + api_key
    )
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200 or "articles" not in data:
            return None
        # Filter and format headlines
        macro_headlines = []
        for article in data["articles"]:
            title = article.get("title", "").strip()
            desc = article.get("description", "").strip()
            # Only include if any macro keyword is present in title or description
            if any(k.lower() in (title + " " + desc).lower() for k in MACRO_KEYWORDS):
                if desc and desc.lower() != title.lower():
                    macro_headlines.append(f"{title}\n{desc}")
                else:
                    macro_headlines.append(title)
            if len(macro_headlines) >= max_headlines:
                break
        if not macro_headlines:
            return None
        return "\n\n".join(macro_headlines)
    except Exception as e:
        return None


def fetch_vix_summary(config: dict) -> Optional[str]:
    """
    Fetches the current VIX index value from Finnhub and returns a summary string.
    Args:
        config (dict): Config with Finnhub API key.
    Returns:
        str: Summary string about the VIX risk regime, or None if unavailable.
    """
    api_key = config.get("finnhub_key")
    if not api_key:
        return None
    url = f"https://finnhub.io/api/v1/quote?symbol=^VIX&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        vix = data.get("c")  # 'c' is the current price
        if vix is None:
            return None
        try:
            vix = float(vix)
        except Exception:
            return None
        if vix < 14:
            regime = "Low volatility"
        elif 14 <= vix <= 20:
            regime = "Normal volatility"
        else:
            regime = "Volatility elevated, caution advised"
        return f"Current VIX: {vix:.2f} — {regime}."
    except Exception as e:
        return None


def fetch_sector_snapshot(config: dict, max_sectors: int = 5) -> Optional[str]:
    """
    Fetches today's sector performance from Finnhub and returns a formatted markdown block.
    Args:
        config (dict): Config with Finnhub API key.
        max_sectors (int): Number of sectors to show (top 3 + bottom 2 by default).
    Returns:
        str: Markdown-formatted sector performance snapshot, or None if unavailable.
    """
    api_key = config.get("finnhub_key")
    if not api_key:
        return None
    url = f"https://finnhub.io/api/v1/sector-performance?token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        # Each item: { 'sector': 'Technology', 'change': 1.23 }
        sectors = [
            (item.get('sector', ''), item.get('change', 0.0))
            for item in data if 'sector' in item and 'change' in item
        ]
        if not sectors:
            return None
        # Sort by performance descending
        sectors_sorted = sorted(sectors, key=lambda x: x[1], reverse=True)
        # Top 3 and bottom 2
        top = sectors_sorted[:3]
        bottom = sectors_sorted[-2:] if len(sectors_sorted) > 4 else sectors_sorted[3:]
        # Format as markdown block
        lines = ["📈 Sector Performance:"]
        for name, change in top:
            lines.append(f"{name}: {change:+.2f}%")
        if bottom:
            lines.append("")
            for name, change in bottom:
                lines.append(f"{name}: {change:+.2f}%")
        return "\n".join(lines)
    except Exception:
        return None


def fetch_today_economic_events(config: dict) -> Optional[str]:
    """
    Fetches today's key economic events (CPI, FOMC, NFP, Jobs, Retail Sales, etc.) from Finnhub economic calendar.
    Returns a markdown-like string block for LLM use.
    """
    api_key = config.get("finnhub_key")
    if not api_key:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={today}&token={api_key}"
    important_keywords = ["CPI", "FOMC", "NFP", "Jobs", "Retail Sales", "PPI", "GDP", "Unemployment", "Payrolls", "Fed"]
    try:
        response = requests.get(url)
        data = response.json()
        events = data.get("economicCalendar", [])
        filtered = []
        for event in events:
            desc = event.get("event", "")
            time = event.get("time", "")
            if any(k.lower() in desc.lower() for k in important_keywords):
                # Format: "CPI data at 8:30 EST"
                line = desc
                if time:
                    line += f" at {time}"
                filtered.append(line)
        if not filtered:
            return None
        lines = ["📅 Today's Economic Events:"] + filtered
        return "\n".join(lines)
    except Exception:
        return None


def fetch_symbol_news_summary(symbol: str, config: dict, max_headlines: int = 3) -> Optional[str]:
    """
    Fetches top 2-3 news headlines for the given symbol using Finnhub.
    Returns a markdown-like string block for LLM use.
    """
    api_key = config.get("finnhub_key")
    if not api_key:
        return None
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.now().strftime('%Y-%m-%d')}&to={datetime.now().strftime('%Y-%m-%d')}&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        lines = [f"🧨 {symbol.upper()} News:"]
        for article in data[:max_headlines]:
            title = article.get("headline", "").strip()
            desc = article.get("summary", "").strip()
            if desc and desc.lower() != title.lower():
                lines.append(f'"{title}"\n{desc}')
            else:
                lines.append(f'"{title}"')
        return "\n\n".join(lines)
    except Exception:
        return None


def fetch_crowd_sentiment(config: dict) -> str:
    """
    Fetches top 5 hot post titles from /r/stocks and /r/wallstreetbets using Reddit API (praw).
    Returns a markdown-style string block of post titles (2–3 from each subreddit).
    Handles API failures gracefully.
    """
    try:
        reddit = praw.Reddit(
            client_id=config["reddit_client_id"],
            client_secret=config["reddit_client_secret"],
            username=config["reddit_username"],
            password=config["reddit_password"],
            user_agent=config["reddit_user_agent"]
        )
        subreddits = ["stocks", "wallstreetbets"]
        titles = []
        for sub in subreddits:
            posts = reddit.subreddit(sub).hot(limit=5)
            count = 0
            for post in posts:
                # Skip stickied posts
                if getattr(post, "stickied", False):
                    continue
                titles.append(f'"{post.title.strip()}"')
                count += 1
                if count >= 3:
                    break
        if not titles:
            return "🔎 Crowd Sentiment:\n- Reddit data unavailable"
        return "🔎 Crowd Sentiment:\n\n" + "\n\n".join(titles)
    except Exception:
        return "🔎 Crowd Sentiment:\n- Reddit data unavailable"


def prepare_scalper_rag_summary(config: dict, symbol: str) -> str:
    """
    Calls all 6 RAG fetchers and combines their outputs into a markdown-style summary block.
    Args:
        config (dict): Configuration dictionary for API keys and settings.
        symbol (str): The trading symbol (e.g., 'MES', 'AAPL').
    Returns:
        str: Combined markdown-style summary for InputContainer.rag_insights.
    """
    macro = fetch_macro_headlines_summary(config) or "[No macro headlines available]"
    vix = fetch_vix_summary(config) or "[No VIX data available]"
    sector = fetch_sector_snapshot(config) or "[No sector snapshot available]"
    econ = fetch_today_economic_events(config) or "[No economic events found for today]"
    symbol_news = fetch_symbol_news_summary(symbol, config) or f"[No news found for {symbol}]"
    crowd = fetch_crowd_sentiment(config) or "[No crowd sentiment available]"

    summary = f"""
📰 MACRO HEADLINES:
{macro}

📈 SECTOR SNAPSHOT:
{sector}

📅 TODAY'S EVENTS:
{econ}

🧨 SYMBOL NEWS ({symbol.upper()}):
{symbol_news}

🔎 CROWD SENTIMENT:
{crowd}

📉 VIX SUMMARY:
{vix}
""".strip()
    return summary