import requests
import openai
from openai import OpenAI
import feedparser
import numpy as np
from config import SUMMARY_CACHE, EVENT_CACHE
import google.generativeai as genai  # Make sure this is at the top of your file

from logging_setup import get_logger

log = get_logger(__name__)
def summarize_with_cache(fetchers, merged_headlines, force_refresh=False):

    if not force_refresh and "summarized_news" in SUMMARY_CACHE:
        log.info("ℹ️ Using cached summarized news (no additional token cost).")
        fetchers.token_usage = 0
        fetchers.cost_usd = 0
        return SUMMARY_CACHE["summarized_news"]
    else:
        log.info("⚡ Summarizing headlines via LLM...")
        summarized = fetchers.summarize_headlines_with_llm(merged_headlines)
        SUMMARY_CACHE["summarized_news"] = summarized
        return summarized

def summarize_economic_events_with_cache(fetchers, force_refresh=False):

    if not force_refresh and "summarized_events" in EVENT_CACHE:
        log.info("ℹ️ Using cached summarized economic events (no additional token cost).")
        fetchers.token_usage = 0
        fetchers.cost_usd = 0
        return EVENT_CACHE["summarized_events"]
    else:
        log.info("⚡ Summarizing economic events via LLM...")
        events = fetchers.fetch_economic_events()
        summaries = fetchers.summarize_headlines_with_llm([e["event"] for e in events])
        EVENT_CACHE["summarized_events"] = summaries
        return summaries

class ResearchFetchers:
    """
    Visual interface for analyzing strategy performance, trade validity, and classifier signals.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary with API keys and settings
        """
        self.config = config
        self.date = config["date"]
        self.model_provider = config["model_provider"]
        self.openai_key = config["openai_api_key"]
        self.gemini_key = config["gemini_api_key"]
        self.token_usage = 0
        self.cost_usd = 0.0

    def fetch_news_headlines(self, markets, symbols=[]):
        """
        Fetch raw news headlines for given markets and symbols, then summarize them via LLM.
        """
        headlines = []

        try:
            # Fetch headlines from NewsAPI
            api_key = self.config.get("newsapi_key", "")
            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={api_key}"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                for article in data.get("articles", []):
                    headlines.append(article.get("title", ""))
            else:
                log.info(f"⚠️ News API Error: {data.get('message')}")

            if not headlines:
                log.info("⚠️ No headlines fetched.")
                return []

            log.info(f"✅ Pulled {len(headlines)} raw news headlines.")

            # Summarize headlines using OpenAI
            summarized_news = self.summarize_headlines_with_llm(headlines)

            return summarized_news

        except Exception as e:
            log.info(f"⚠️ News fetching or summarizing failed: {e}")

        return []


    def summarize_headlines_with_llm(self, headlines):
        """
        Summarize a list of headlines using the selected LLM provider.
        """
        summary = []

        prompt_text = "Summarize today's market tone based on these headlines:\n\n"
        prompt_text += "\n".join(f"- {h}" for h in headlines)

        if self.model_provider == "openai":
            client = OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model=self.config["model_name"],
                messages=[
                    {"role": "system", "content": "You are a financial news summarizer."},
                    {"role": "user", "content": prompt_text}
                ]
            )

            summary_text = response.choices[0].message.content
            usage = response.usage
            self.token_usage = usage.total_tokens
            self.cost_usd = self.calculate_cost_estimate(usage.total_tokens)

            log.info(f"📊 LLM Token usage: {usage.total_tokens} tokens")
            log.info(f"💵 Estimated LLM Cost: ${self.cost_usd:.4f}")

            summary.append(summary_text)

        elif self.model_provider == "gemini":
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(self.config["model_name"])
            response = model.generate_content(prompt_text)

            summary_text = response.text
            self.token_usage = 0  # Gemini does not return token usage yet
            self.cost_usd = self.calculate_cost_estimate(0)

            log.info("📊 Gemini LLM used (token usage unknown)")
            summary.append(summary_text)

        return summary

    def calculate_cost_estimate(self, total_tokens):
        """
        Estimate cost based on selected model.
        """
        cost_per_million_tokens = {
            "gpt-4o": 5,  # $5 per 1M tokens
            "gpt-4-turbo": 10,  # $10 per 1M tokens
            "gemini-1.5-pro-latest": 3,  # Approx $3 per 1M tokens
            "gemini-1.0-pro": 4,  # Approx $4 per 1M tokens
        }

        model = self.config["model_name"]
        per_million = cost_per_million_tokens.get(model, 5)

        estimated_cost = (total_tokens / 1_000_000) * per_million
        return estimated_cost

    def fetch_smart_money_flow(self, symbols: list = None):
        """
        Estimates smart money flow using Finnhub sentiment.
        If symbols are provided, analyze them. Otherwise, use broad ETFs.
        """
        api_key = self.config.get("finnhub_api_key")
        headers = {"X-Finnhub-Token": api_key}
    
        if not symbols:
            # Use broad ETFs for general market sentiment
            symbols_to_check = {
                                "SPY": "S&P 500",
                                "QQQ": "Nasdaq 100",
                                "XLK": "Technology",
                                "XLF": "Financials",
                                "XLE": "Energy",
                                "XLV": "Healthcare",
                                "XLY": "Consumer Discretionary",
                                "XLP": "Consumer Staples",
                                "XLI": "Industrials",
                                "XLU": "Utilities",
                                "XLRE": "Real Estate",
                                "XLB": "Materials",
                                "GLD": "Gold",
                                "USO": "Oil",
                                "TLT": "Long-Term Bonds"
                            }

        else:
            # Use user-provided symbols (e.g., AAPL, TSLA)
            symbols_to_check = {sym: sym for sym in symbols}
    
        results = {}
        for sym, label in symbols_to_check.items():
            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={sym}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                sentiment = response.json().get("sentiment", {})
                results[label] = sentiment.get("score", 0)
            else:
                results[label] = 0  # fallback
    
        avg_score = sum(results.values()) / len(results)
        trend = "Accumulation" if avg_score > 0.2 else "Distribution" if avg_score < -0.2 else "Neutral"
        top_sectors = sorted(results, key=results.get, reverse=True)[:3]
    
        return trend, top_sectors

    def fetch_economic_events(self):
        """
        Fetches upcoming economic events from Investing.com's Economic Calendar RSS feed,
        and extracts country information if possible.
        """
        rss_url = "https://www.investing.com/rss/news_285.rss"
        
        try:
            feed = feedparser.parse(rss_url)
            events = []
    
            def extract_country_from_event(event_title):
                title = event_title.lower()
                if "u.s." in title or "fed" in title:
                    return "United States 🇺🇸"
                elif "eurozone" in title or "ecb" in title:
                    return "Eurozone 🇪🇺"
                elif "u.k." in title or "boe" in title:
                    return "United Kingdom 🇬🇧"
                elif "japan" in title or "boj" in title:
                    return "Japan 🇯🇵"
                elif "china" in title:
                    return "China 🇨🇳"
                elif "germany" in title:
                    return "Germany 🇩🇪"
                elif "canada" in title:
                    return "Canada 🇨🇦"
                else:
                    return "Unknown 🌎"
    
            for entry in feed.entries:
                events.append({
                    "event": entry.title,
                    "published": entry.published,
                    "country": extract_country_from_event(entry.title)
                })
    
            return events[:10]  # Limit to first 10 events for now

        except Exception as e:
            log.info(f"⚠️ Failed to fetch economic events: {e}")
            return []

    # 1. Fetch ETF Sector Sentiment
    def fetch_etf_sentiment_heatmap(self):
        """
        Fetches ETF sector sentiment scores from Finnhub.
        Returns a dictionary: {sector_name: sentiment_score}.
        """
        api_key = self.config.get("finnhub_api_key")
        headers = {"X-Finnhub-Token": api_key}
    
        etf_symbols = {
            "Technology": "XLK",
            "Energy": "XLE",
            "Financials": "XLF",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Materials": "XLB",
            "Real Estate": "XLRE"
        }
    
        results = {}
        for sector, symbol in etf_symbols.items():
            url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                sentiment = response.json().get("sentiment", {})
                results[sector] = sentiment.get("score", 0)
            else:
                results[sector] = 0  # fallback
    
        return results
    
    # 2. Analyze News Volume Spikes
    def analyze_news_volume_spikes(self, headlines):
        """
        Analyzes headline volume. Flags if headline count exceeds threshold.
        Returns 'High' or 'Normal'.
        """
        threshold = 150  # You can adjust this
        volume = len(headlines)
        if volume > threshold:
            return "High"
        else:
            return "Normal"
    
    # 3. Economic Risk Tagging
    def economic_risk_tagging(self, economic_calendar):
        """
        Tags the day as 'High Risk' if major events like Fed, CPI, Jobs appear.
        Otherwise 'Normal'.
        """
        high_risk_keywords = ["Fed", "FOMC", "CPI", "PPI", "Jobs", "NFP", "GDP", "Interest Rate"]
        joined_events = " ".join(event["event"] for event in economic_calendar)
    
        if any(keyword.lower() in joined_events.lower() for keyword in high_risk_keywords):
            return "High Risk"
        else:
            return "Normal" 