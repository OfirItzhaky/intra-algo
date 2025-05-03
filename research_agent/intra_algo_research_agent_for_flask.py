#!/usr/bin/env python
# coding: utf-8

# ## 🔧 Global Configuration & API Keys
# 
# This section initializes global variables, including API keys and caching dictionaries (`SUMMARY_CACHE`, `EVENT_CACHE`).  
# It also defines the date and prepares a centralized `CONFIG` dictionary used throughout the agent.
# 

# In[5]:


# === [ NEW GLOBAL CACHE (clears on restart) ] ===
SUMMARY_CACHE = {}
EVENT_CACHE = {}

gemini_key = "AIzaSyChXSlJ4AvJVmcXi1d4R_fwM34cHuew8Ck"


openai_key = "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"



newsapi_key = "245fedd277794f78a3ada25372c13e2d"

finnhub_key = "d08dvh9r01qh1ecbk2agd08dvh9r01qh1ecbk2b0"

fmp_key = "v5AN061l6WNC9G4FykCkswxSqa22PqOQ"


# In[ ]:





# In[4]:


# 📄 Configuration Cell: Research Agent Setup

# ✅ API Keys (define or import these before running this cell)
OPENAI_API_KEY = openai_key
GEMINI_API_KEY = gemini_key
NEWSAPI_KEY = newsapi_key
FINNHUB_KEY = finnhub_key
FMP_KEY = fmp_key

# ✅ Default Research Settings
DEFAULT_MARKETS = ["US Market"]
DEFAULT_SYMBOLS = []             # (e.g., ["AAPL", "BAC", "GOLD"])
FOCUS_SECTORS = ["Oil", "Coffee"]

# ✅ File Saving Settings
SAVE_DIRECTORY = "./research_outputs/"

# ✅ Behavior Settings
ONLY_MAJOR_EVENTS = True
PRINT_LIVE_SUMMARY = True
COPY_TO_CLIPBOARD = True

# ✅ Date Handling
import datetime
import pytz

ny_timezone = pytz.timezone("America/New_York")
today = datetime.datetime.now(ny_timezone).strftime("%Y-%m-%d")

# ✅ Central Configuration Dictionary
CONFIG = {
    # Model provider and model name will be injected dynamically later
    "model_provider": None,
    "model_name": None,

    # LLM Providers
    "openai_api_key": OPENAI_API_KEY,
    "gemini_api_key": GEMINI_API_KEY,

    # News Data Sources
    "newsapi_key": NEWSAPI_KEY,
    "finnhub_key": FINNHUB_KEY,
    "fmp_key": FMP_KEY,

    # Market/Strategy Parameters
    "markets": DEFAULT_MARKETS,
    "symbols": DEFAULT_SYMBOLS,
    "focus_sectors": FOCUS_SECTORS,

    # Behavior Flags
    "only_major_events": ONLY_MAJOR_EVENTS,
    "print_live_summary": PRINT_LIVE_SUMMARY,
    "copy_to_clipboard": COPY_TO_CLIPBOARD,

    # File/Session
    "save_directory": SAVE_DIRECTORY,
    "date": today,
}


# ## 📦 ResearchFetchers Class: Market Data Collection
# 
# This class provides methods to:
# - Pull raw headlines from NewsAPI, Finnhub, Yahoo RSS
# - Fetch ETF sentiment and economic events
# - Summarize data using LLM (OpenAI or Gemini)
# - Estimate token cost
# 
# It acts as the backend fetch-and-process layer of the research agent.
# 

# In[5]:


# 📦 Updated Fetchers Module for Research Agent

import requests
import openai
from openai import OpenAI
import feedparser

class ResearchFetchers:
    def __init__(self, config):
        self.config = config
        self.date = config["date"]
        self.model_provider = config["model_provider"]
        self.openai_key = config["openai_api_key"]
        self.gemini_key = config["gemini_api_key"]
        self.token_usage = 0
        self.cost_usd = 0.0

    def fetch_news_headlines(self, markets, symbols):
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
                print(f"⚠️ News API Error: {data.get('message')}")

            if not headlines:
                print("⚠️ No headlines fetched.")
                return []

            print(f"✅ Pulled {len(headlines)} raw news headlines.")

            # Summarize headlines using OpenAI
            summarized_news = self.summarize_headlines_with_llm(headlines)

            return summarized_news

        except Exception as e:
            print(f"⚠️ News fetching or summarizing failed: {e}")

        return []

    def summarize_headlines_with_llm(self, headlines):
        """
        Summarize a list of headlines using the selected LLM provider.
        """
        summary = []
        if self.model_provider == "openai":
            client = OpenAI(api_key=self.openai_key)
            prompt_text = "Summarize today's market tone based on these headlines:\n\n"
            prompt_text += "\n".join(f"- {h}" for h in headlines)

            response = client.chat.completions.create(
                model=self.config["model_name"],
                messages=[
                    {"role": "system", "content": "You are a financial news summarizer."},
                    {"role": "user", "content": prompt_text}
                ]
            )

            summary_text = response.choices[0].message.content

            # Capture token usage
            usage = response.usage
            self.token_usage = usage.total_tokens
            self.cost_usd = self.calculate_cost_estimate(usage.total_tokens)

            print(f"📊 LLM Token usage: {usage.total_tokens} tokens")
            print(f"💵 Estimated LLM Cost: ${self.cost_usd:.4f}")

            summary.append(summary_text)
        
        elif self.model_provider == "gemini":
            # Placeholder: Add Gemini call later if needed
            pass

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
            print(f"⚠️ Failed to fetch economic events: {e}")
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


# In[6]:


# !pip install pyperclip
# !pip install feedparser


# ## 🧠 News Aggregator Class: Collecting Multi-Source Headlines
# 
# Aggregates financial headlines from:
# - NewsAPI
# - Finnhub
# - Yahoo Finance RSS
# - Investing.com RSS
# 
# Used to build a diverse market context from multiple data providers.
# 

# In[7]:


# 📦 News Aggregator Class for Research Agent

import requests
import feedparser

class NewsAggregator:
    def __init__(self, config):
        self.config = config
        self.headlines = []
        self.date = config["date"]

        # API keys needed
        self.newsapi_key = config.get("newsapi_key", "")
        self.finnhub_key = config.get("finnhub_key", "")
        self.fmp_key = config.get("fmp_key", "")

    def fetch_from_newsapi(self):
        """
        Fetch latest business headlines using NewsAPI.
        """
        try:
            if not self.newsapi_key:
                print("⚠️ NewsAPI key missing, skipping NewsAPI fetch.")
                return

            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=30&apiKey={self.newsapi_key}"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                headlines = [article.get("title", "") for article in data.get("articles", [])]
                self.headlines.extend(headlines)
                print(f"✅ Pulled {len(headlines)} headlines from NewsAPI.")
            else:
                print(f"⚠️ NewsAPI error: {data.get('message')}")

        except Exception as e:
            print(f"❌ Error fetching from NewsAPI: {e}")

    def fetch_from_finnhub(self):
        """
        Fetch general financial news using Finnhub API.
        """
        try:
            if not self.finnhub_key:
                print("⚠️ Finnhub key missing, skipping Finnhub fetch.")
                return

            url = f"https://finnhub.io/api/v1/news?category=general&token={self.finnhub_key}"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                headlines = [article.get("headline", "") for article in data]
                self.headlines.extend(headlines)
                print(f"✅ Pulled {len(headlines)} headlines from Finnhub.")
            else:
                print(f"⚠️ Finnhub error: {response.json().get('error')}")

        except Exception as e:
            print(f"❌ Error fetching from Finnhub: {e}")

    

    def fetch_from_yahoo_finance_rss(self):
        """
        Fetch headlines from Yahoo Finance RSS feed.
        """
        try:
            url = "https://finance.yahoo.com/news/rssindex"
            feed = feedparser.parse(url)

            headlines = [entry.title for entry in feed.entries]
            self.headlines.extend(headlines)
            print(f"✅ Pulled {len(headlines)} headlines from Yahoo Finance RSS.")

        except Exception as e:
            print(f"❌ Error fetching from Yahoo RSS: {e}")

    def aggregate_news(self):
        """
        Fetch from all sources and merge headlines.
        """
        print("📡 Aggregating news from multiple sources...")

        self.fetch_from_newsapi()
        self.fetch_from_finnhub()
        self.fetch_from_yahoo_finance_rss()

        print(f"🧠 Total aggregated headlines: {len(self.headlines)}")

        # Remove duplicates
        merged_headlines = list(set(self.headlines))
        print(f"🧹 After removing duplicates: {len(merged_headlines)} unique headlines.")

        return merged_headlines

    def fetch_from_investing_rss(self):
        """
        Fetch headlines from Investing.com RSS feeds (general + economic).
        """
        try:
            import feedparser
            
            feeds = [
                "https://www.investing.com/rss/news_25.rss",   # General market news
                "https://www.investing.com/rss/news_285.rss",  # Economic news
            ]
    
            all_headlines = []
    
            for url in feeds:
                feed = feedparser.parse(url)
                feed_headlines = [entry.title for entry in feed.entries]
                all_headlines.extend(feed_headlines)
    
            self.headlines.extend(all_headlines)
            print(f"✅ Pulled {len(all_headlines)} headlines from Investing.com RSS.")
    
        except Exception as e:
            print(f"❌ Error fetching from Investing.com RSS: {e}")
    
    import requests

    def fetch_company_profile(self, symbol: str) -> dict:
        """
        Fetches basic company profile from FMP (free endpoint).
        
        Parameters:
            symbol (str): Stock ticker symbol (e.g., AAPL, MSFT)
    
        Returns:
            dict: A dictionary with sector, industry, CEO, and description if successful.
        """
        api_key = self.config.get("fmp_api_key")
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
    
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    profile = data[0]
                    return {
                        "companyName": profile.get("companyName", "N/A"),
                        "sector": profile.get("sector", "N/A"),
                        "industry": profile.get("industry", "N/A"),
                        "ceo": profile.get("ceo", "N/A"),
                        "description": profile.get("description", "N/A")
                    }
                else:
                    print(f"⚠️ No profile data found for {symbol}")
                    return {}
            else:
                print(f"⚠️ Error fetching FMP profile for {symbol}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"⚠️ Exception during FMP profile fetch: {e}")
        return {}


# ## 📊 ResearchAnalyzer Class: Market Summary & Signal Generator
# 
# This is the main analysis engine. It:
# - Builds metadata and market context
# - Runs general and symbol-level analysis
# - Summarizes bias, top sectors, and events
# - Saves JSON reports and prints summaries
# 

# In[8]:


# 🧩 Analyzer Module for Research Agent

from datetime import datetime
import os
import json
import pyperclip

class ResearchAnalyzer:
    def __init__(self, config, fetchers):
        self.config = config
        self.fetchers = fetchers
        self.outputs = {
            "metadata": {},
            "general": {},
            "symbols": {}
        }

    def build_metadata(self):
        """
        Create metadata block with generation time and user preferences.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.outputs["metadata"] = {
            "generation_time": now,
            "requested_markets": self.config["markets"],
            "requested_symbols": self.config["symbols"]
        }

    def analyze_general(self):
        """
        Fetch news, smart money, economic events and generate general market insights.
        """
        news_headlines = self.fetchers.fetch_news_headlines(
            self.config["markets"], 
            self.config["symbols"]
        )

        smart_money_trend, top_sectors = self.fetchers.fetch_smart_money_flow()
        economic_events = self.fetchers.fetch_economic_events()

        # Summarize bias based on simple logic
        if "Fed" in " ".join([e["event"] for e in economic_events]):
            general_bias = "Cautious"
        elif smart_money_trend == "Accumulation":
            general_bias = "Bullish"
        elif smart_money_trend == "Distribution":
            general_bias = "Bearish"
        else:
            general_bias = "Neutral"

        self.outputs["general"] = {
            "general_bias": general_bias,
            "smart_money_flow": {
                "trend": smart_money_trend,
                "top_sectors": top_sectors
            },
            "economic_calendar": economic_events,
            "headline_sample": news_headlines[:5]  # First 5 headlines
        }
    import requests


    def analyze_symbols(self):
        """
        Analyze requested symbols one by one.
        """
        symbols = self.config["symbols"]
        for symbol in symbols:
            # Simplified: check if any news contains the symbol
            found = False
            symbol_summary = {
                "sentiment": "Unknown",
                "unusual_volume": False,
                "smart_money_flow": "Unknown",
                "recommended_action": "No strong recommendation"
            }

            # Simulate symbol search in headlines
            sample_headlines = self.outputs["general"].get("headline_sample", [])
            for headline in sample_headlines:
                if symbol in headline:
                    found = True
                    symbol_summary["sentiment"] = "Positive"
                    symbol_summary["recommended_action"] = "Watch for Long Opportunity"
                    break

            if not found:
                symbol_summary["status"] = "No fresh news found today."

            self.outputs["symbols"][symbol] = symbol_summary



    def save_report(self):
        """
        Saves the current analysis output (self.outputs) as a JSON file.
        Now includes summarized_event_text from config, if available.
        """
        # 📂 Ensure output folder exists
        os.makedirs("research_outputs", exist_ok=True)
    
        # 🗓️ File name with today's date
        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = f"research_outputs/research_{today_str}.json"
    
        # ✅ Add summarized LLM economic event text to outputs if available
        if "general" in self.outputs and "summarized_events" in self.config:
            self.outputs["general"]["summarized_event_text"] = self.config["summarized_events"]
    
        # 💾 Save the output to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.outputs, f, ensure_ascii=False, indent=2)
    
        print(f"✅ Research report saved to {file_path}.")


    def print_summary(self):
        """
        Print live short summary and optionally copy to clipboard.
        """
        bias = self.outputs["general"].get("general_bias", "Unknown")
        sectors = self.outputs["general"].get("smart_money_flow", {}).get("top_sectors", [])
        upcoming_events = [e["event"] for e in self.outputs["general"].get("economic_calendar", [])]

        print("\n🔔 Daily Summary:")
        print(f"🧠 Market Bias: {bias}")
        print(f"🏛️ Focus Sectors: {', '.join(sectors) if sectors else 'None'}")
        print(f"📅 Key Events: {', '.join(upcoming_events) if upcoming_events else 'None'}\n")

        if self.config["copy_to_clipboard"]:
            summary_text = f"Bias: {bias} | Sectors: {', '.join(sectors)} | Events: {', '.join(upcoming_events)}"
            pyperclip.copy(summary_text)
            print("📋 Summary copied to clipboard!")

    def run_daily_analysis(self):
        """
        Complete run: metadata → analysis → save → display
        """
        self.build_metadata()
        self.analyze_general()
        self.analyze_symbols()
        self.save_report()
        if self.config["print_live_summary"]:
            self.print_summary()

    def analyze_symbols(self):
        """
        Analyze each symbol separately: fetch news, fetch company profile, and prepare outputs.
        """
        self.outputs["symbols"] = {}
    
        for symbol in self.config.get("symbols", []):
            print(f"\n🔎 Analyzing Symbol: {symbol}")
    
            # Fetch top headlines
            headlines = self.fetchers.fetch_news_headlines([symbol])
    
            # Fetch company profile using FMP
            profile_data = self.fetchers.fetch_company_profile(symbol)
    
            # Optional: If you want to fetch sentiment separately later (not mandatory yet)
            # sentiment_data = self.fetchers.fetch_sentiment(symbol)
    
            # Build output for this symbol
            self.outputs["symbols"][symbol] = {
                "headline_sample": headlines[:5],
                "company_profile": profile_data,
                # "sentiment": sentiment_data,  # If you add it later
            }
    
            # Print a brief summary
            print(f"✅ {symbol}: Found {len(headlines)} headlines.")
            if profile_data:
                print(f"✅ {symbol}: Sector: {profile_data.get('sector', 'N/A')}, Industry: {profile_data.get('industry', 'N/A')}")



# ## ⚙️ Final Runner: Execute Full Daily Analysis Flow
# 
# This cell:
# 1. Initializes model and fetchers
# 2. Pulls & summarizes news and events (with caching)
# 3. Runs the analyzer
# 4. Saves and prints the full summary
# 

# In[9]:


# === [ Step 0: Recommended models ] ===
RECOMMENDED_MODELS = {
    "openai": {
        "primary": "gpt-4o",
        "alternative": "gpt-4-turbo"
    },
    "gemini": {
        "primary": "gemini-1.5-pro-latest",
        "alternative": "gemini-1.0-pro"
    }
}

# === [ Step 1: Manual selection ] ===
SELECTED_PROVIDER = "openai"
SELECTED_MODEL = RECOMMENDED_MODELS[SELECTED_PROVIDER]["primary"]

# === [ Step 2: Inject into config ] ===
CONFIG["model_provider"] = SELECTED_PROVIDER
CONFIG["model_name"] = SELECTED_MODEL

# === [ Step 3: Init components ] ===
fetchers = ResearchFetchers(config=CONFIG)
news_aggregator = NewsAggregator(config=CONFIG)

# === [ Step 4: Fetch all news sources ] ===
merged_headlines = news_aggregator.aggregate_news()

# === [ Step 5: Summarize news headlines (cache-aware) ] ===
def summarize_with_cache(fetchers, merged_headlines, force_refresh=False):
    global SUMMARY_CACHE

    if not force_refresh and "summarized_news" in SUMMARY_CACHE:
        print("ℹ️ Using cached summarized news (no additional token cost).")
        fetchers.token_usage = 0
        fetchers.cost_usd = 0
        return SUMMARY_CACHE["summarized_news"]
    else:
        print("⚡ Summarizing headlines via LLM...")
        summarized = fetchers.summarize_headlines_with_llm(merged_headlines)
        SUMMARY_CACHE["summarized_news"] = summarized
        return summarized

# === [ Step 6: Summarize economic events (cache-aware) ] ===
def summarize_economic_events_with_cache(fetchers, force_refresh=False):
    global EVENT_CACHE

    if not force_refresh and "summarized_events" in EVENT_CACHE:
        print("ℹ️ Using cached summarized economic events (no additional token cost).")
        fetchers.token_usage = 0
        fetchers.cost_usd = 0
        return EVENT_CACHE["summarized_events"]
    else:
        print("⚡ Summarizing economic events via LLM...")
        events = fetchers.fetch_economic_events()
        summaries = fetchers.summarize_headlines_with_llm([e["event"] for e in events])
        EVENT_CACHE["summarized_events"] = summaries
        return summaries

# === [ Step 7: Control switches ] ===
FORCE_NEW_SUMMARY = False
FORCE_NEW_EVENTS = False

# === [ Step 8: Run both summaries ] ===
summarized_news = summarize_with_cache(fetchers, merged_headlines, force_refresh=FORCE_NEW_SUMMARY)
summary_token_usage = fetchers.token_usage
summary_cost_usd = fetchers.cost_usd

summarized_events = summarize_economic_events_with_cache(fetchers, force_refresh=FORCE_NEW_EVENTS)
econ_token_usage = fetchers.token_usage
econ_cost_usd = fetchers.cost_usd

# === [ Step 9: Inject summaries into config ] ===
CONFIG["summarized_news"] = summarized_news
CONFIG["summarized_events"] = summarized_events

# === [ Step 10: Run analyzer ] ===
analyzer = ResearchAnalyzer(config=CONFIG, fetchers=fetchers)
analyzer.run_daily_analysis()

# === [ Step 11: Print FINAL COMBINED usage & cost ] ===
total_tokens = summary_token_usage + econ_token_usage
total_cost = summary_cost_usd + econ_cost_usd

print("\n🔢 FINAL USAGE SUMMARY:")
print(f"🧾 Total Tokens Used: {total_tokens}")
print(f"💰 Total Estimated Cost: ${total_cost:.4f}")


# ## 🧾 Report Loader: View Latest JSON Output
# 
# This section loads the most recent saved daily analysis file from disk and prints the `general` section for review.
# 

# In[10]:


import json
from datetime import datetime

# Get today's file path
today_str = datetime.now().strftime("%Y-%m-%d")
file_path = f"./research_outputs/research_{today_str}.json"

# Load the file
with open(file_path, "r", encoding="utf-8") as f:
    saved_output = json.load(f)

# Pretty-print the general summary
from pprint import pprint
pprint(saved_output["general"])


# ## 🔥 Smart Money Proxy Heatmap (ETF Sentiment + Risk Layers)
# 
# Builds a sector-level heatmap using:
# - ETF sentiment (Finnhub or Yahoo fallback)
# - News volume spike detection
# - Economic event risk classification
# 
# Displays a quick "smart money" overview across sectors.
# 

# In[11]:


# === [ Smart Money Proxy Heatmap Cell - Updated ] ===

# Step 0: (Re)-Init fetchers to ensure latest methods are loaded
fetchers = ResearchFetchers(config=CONFIG)

# Step 1: Fetch ETF sector sentiment
etf_sentiment_scores = fetchers.fetch_etf_sentiment_heatmap()

# Step 2: Analyze news volume spike
news_volume_tag = fetchers.analyze_news_volume_spikes(merged_headlines)

# Step 3: Analyze economic risk from calendar
economic_risk_tag = fetchers.economic_risk_tagging(fetchers.fetch_economic_events())

# Step 4: Build and Display smart money proxy heatmap
import pandas as pd

# Create DataFrame
heatmap_df = pd.DataFrame.from_dict(etf_sentiment_scores, orient="index", columns=["Sentiment Score"])
heatmap_df["News Volume"] = news_volume_tag
heatmap_df["Economic Risk"] = economic_risk_tag

# Display sorted table
print("\n🔥 Smart Money Proxy Heatmap 🔥")
display(heatmap_df.sort_values("Sentiment Score", ascending=False))


# In[12]:


import yfinance as yf
import pandas as pd
import os

class SwingGaugeLong:
    def __init__(self, symbols, image_dir=None):
        """
        :param symbols: List of tradable symbols (e.g., ['SPY', 'QQQ', 'Sds'])
        :param image_dir: Optional directory path with uploaded indicator snapshots
        """
        self.symbols = symbols
        self.image_dir = image_dir
        self.yf_data = {}
        self.image_analysis = {}
        self.signal_scores = {}
        self.output = pd.DataFrame()

    def fetch_price_data(self):
        """
        Pulls historical daily and weekly data using yfinance for each symbol.
        Saves to self.yf_data
        """
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data_daily = ticker.history(period="3mo", interval="1d")
            data_weekly = ticker.history(period="6mo", interval="1wk")
            self.yf_data[symbol] = {
                "daily": data_daily,
                "weekly": data_weekly
            }

    def calculate_spy_vix_ratio(self):
        """
        Pull SPY and ^VIX from yfinance, compute ratio
        """
        spy = yf.download("SPY", period="3mo")["Close"]
        vix = yf.download("^VIX", period="3mo")["Close"]
        df = pd.DataFrame({"SPY": spy, "VIX": vix})
        df["SPY_VIX_Ratio"] = df["SPY"] / df["VIX"]
        return df

    def analyze_market_snapshot_images(self):
        """
        Placeholder for GPT-Vision-based image logic.
        For now, checks if expected files exist.
        """
        expected_files = ["nyad", "nyhl", "nysi", "bpna", "spxa150r", "nya150r"]
        for name in expected_files:
            file_path = os.path.join(self.image_dir, f"{name}.png") if self.image_dir else None
            if file_path and os.path.exists(file_path):
                self.image_analysis[name.upper()] = "✅ Found (awaiting model)"
            else:
                self.image_analysis[name.upper()] = "❌ Missing"

    def evaluate_weekly_signal(self, symbol):
        """
        Evaluate the weekly trend signal logic as described:
        +1 if EMA10 > EMA20 and close > EMA10
        +1 if close or previous green candle touched EMA10
        -1 if not
        0 if within 1.5 ATR and below pivot
        """
        df = self.yf_data[symbol]["weekly"].copy()
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["Prev_Close"] = df["Close"].shift(1)
        df["Prev_Open"] = df["Open"].shift(1)
    
        # Calculate ATR
        df["ATR"] = self.calculate_atr(df)
    
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        signal = 0
    
        # Trend condition
        if last["EMA10"] > last["EMA20"] and last["Close"] > last["EMA10"]:
            signal += 1
    
            # Touch EMA10 check
            touched = abs(last["Close"] - last["EMA10"]) < 0.01 * last["Close"] or \
                      (prev["Close"] > prev["Open"] and abs(prev["Close"] - last["EMA10"]) < 0.01 * prev["Close"])
            signal += 1 if touched else -1
    
            # Additional filter: within ATR zone but below pivot
            pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
            within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
            if within_atr:
                signal = 0
    
        self.signal_scores[symbol] = signal
        return signal


    def summarize(self):
        """
        Simple output summary of available data, images, and scores
        """
        print("🔹 Symbols fetched:", list(self.yf_data.keys()))
        print("📸 Snapshot status:")
        for k, v in self.image_analysis.items():
            print(f"  - {k}: {v}")
        print("📊 Weekly Signals:")
        for sym, score in self.signal_scores.items():
            print(f"  - {sym}: {score}")

    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR) for given OHLC DataFrame.
        """
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def check_pivot_proximity(self, symbol, timeframe="weekly"):
        """
        Checks if current close is below yesterday's pivot AND within 1.5x ATR.
        :return: True if close is below pivot and within ATR range, else False
        """
        df = self.yf_data[symbol][timeframe].copy()
        df["ATR"] = self.calculate_atr(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        # Classic pivot = (high + low + close) / 3
        pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
        
        return within_atr

    def evaluate_daily_signal(self, symbol):
        """
        Evaluate the daily entry timing logic:
        +1 if EMA10 > EMA20 and close > EMA10
        +1 if close or previous green bar touched EMA10
        -1 if not
        0 if within 1.5 ATR and below pivot
        """
        df = self.yf_data[symbol]["daily"].copy()
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["Prev_Close"] = df["Close"].shift(1)
        df["Prev_Open"] = df["Open"].shift(1)
    
        # Calculate ATR
        df["ATR"] = self.calculate_atr(df)
    
        last = df.iloc[-1]
        prev = df.iloc[-2]
    
        signal = 0
    
        # Trend alignment
        if last["EMA10"] > last["EMA20"] and last["Close"] > last["EMA10"]:
            signal += 1
    
            # Touch EMA10 check
            touched = abs(last["Close"] - last["EMA10"]) < 0.01 * last["Close"] or \
                      (prev["Close"] > prev["Open"] and abs(prev["Close"] - last["EMA10"]) < 0.01 * prev["Close"])
            signal += 1 if touched else -1
    
            # Pivot/ATR zone condition
            pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
            within_atr = (last["Close"] > pivot - 1.5 * last["ATR"]) and (last["Close"] < pivot)
            if within_atr:
                signal = 0
    
        return signal



# ## ImageAnalyzerAI

# In[29]:


import base64
import json
import requests
import ipywidgets as widgets
from IPython.display import display

class ImageAnalyzerAI:
    def __init__(self, model_provider, model_name, api_key):
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.image_analysis = {}

        self.prompts = [
            {
                "id": "trend_strength",
                "name": "Trend Strength Index",
                "timeframe": "weekly",
                "prompt": (
                    "Analyze the uploaded weekly chart using the following logic:\n"
                    "1. Add EMA(10) and EMA(20) to the chart.\n"
                    "2. If EMA(10) > EMA(20) AND the bar closes above EMA(10), assign +1 for positive momentum.\n"
                    "3. If the current bar OR the previous green bar touches EMA(10), assign +1. If not, assign -1.\n"
                    "4. If momentum is positive but the close is below the previous pivot and within 1.5 ATR, override score to 0.\n"
                    "Return a label (Bullish, Neutral, or Bearish), a numeric score (-1, 0, 1, or 2), and a short natural language explanation."
                )
            },
            {
                "id": "entry_timing",
                "name": "Entry Timing Index",
                "timeframe": "daily",
                "prompt": (
                    "Analyze the uploaded daily chart using the following logic:\n"
                    "1. Add EMA(10) and EMA(20).\n"
                    "2. If EMA(10) > EMA(20) AND the bar closes above EMA(10), assign +1.\n"
                    "3. If the current bar OR previous green bar touches EMA(10), assign +1. If not, assign -1.\n"
                    "4. If momentum is positive but the close is below yesterday’s pivot and within 1.5 ATR, override to 0.\n"
                    "Return a label (Bullish, Neutral, or Bearish), a score, and a brief explanation."
                )
            }
        ]

    def get_prompt_by_id(self, rule_id):
        for rule in self.prompts:
            if rule["id"] == rule_id:
                return rule
        return None

    def upload_and_analyze_images(self, rule_id):
        self._uploader = widgets.FileUpload(
            accept='.png,.jpg,.jpeg',
            multiple=True,
            description="Upload Snapshot Images"
        )
        display(self._uploader)
        self._uploader.observe(lambda change: self._handle_uploaded_files(rule_id), names='value')

    def _handle_uploaded_files(self, rule_id):
        prompt_block = self.get_prompt_by_id(rule_id)
        if not prompt_block:
            print(f"❌ Invalid rule_id: {rule_id}")
            return

        results = {}
        for filename, fileinfo in self._uploader.value.items():
            symbol = filename.split(".")[0].upper()
            image_bytes = fileinfo['content']
            print(f"🧠 Analyzing {symbol}...")

            try:
                temp_path = f"/tmp/{filename}"
                with open(temp_path, "wb") as temp:
                    temp.write(image_bytes)

                response = self.analyze_image_with_bytes(temp_path, rule_id)

                if isinstance(response, dict) and "label" in response.get("raw_output", "").lower():
                    results[symbol] = {
                        "raw_output": response["raw_output"],
                        "timeframe": prompt_block["timeframe"]
                    }
                else:
                    results[symbol] = {
                        "label": "Unknown",
                        "score": 0,
                        "timeframe": prompt_block["timeframe"],
                        "explanation": f"⚠️ No valid pattern detected. Please upload the appropriate image for {symbol}."
                    }

            except Exception as e:
                results[symbol] = {
                    "label": "Error",
                    "score": 0,
                    "timeframe": prompt_block["timeframe"],
                    "explanation": f"⚠️ Failed to process image for {symbol}: {e}"
                }

        self.image_analysis = results
        print("✅ Image analysis completed.")

    def analyze_image_with_bytes(self, image_bytes, rule_id):
        """
        Analyze an in-memory image using the selected rule.
        Accepts bytes directly (no file path needed).
        """
        prompt_block = self.get_prompt_by_id(rule_id)
        if not prompt_block:
            raise ValueError(f"Unknown rule_id: {rule_id}")
    
        if self.model_provider == "openai":
            return self._analyze_with_openai(image_bytes, prompt_block["prompt"])
        elif self.model_provider == "gemini":
            return self._analyze_with_gemini(image_bytes, prompt_block["prompt"])
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")


    def _analyze_with_openai(self, image_bytes, prompt):
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                    }}
                ]}
            ],
            "max_tokens": 500
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return {"raw_output": content}

    def _analyze_with_gemini(self, image_bytes, prompt):
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": encoded_image
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(endpoint, headers=headers, data=json.dumps(body))
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return {"raw_output": content}


# In[23]:


# !pip install ipywidgets jupyterlab_widgets --quiet


# In[24]:


# === [ Cell: Upload + Analyze Snapshot Image with Selected Vision Model ] ===

# === Vision Model Options ===
# OPENAI:
#   - "gpt-4-vision-preview" 🧠 (~$0.01 / image + $0.03 per 1K output tokens)
#   - "gpt-4o"                ⚡ Faster + cheaper ($0.005 / image + $0.01 per 1K tokens) ✅ Recommended
#
# GEMINI:
#   - "gemini-pro-vision"     🧠 $0.0025 / 1K tokens (image cost depends on API wrapper)
#   - "gemini-1.5-pro"        ⚡ Best model, limited free quota (Vision + text, priced via Vertex AI)

# === CHOOSE MODEL PROVIDER ===
MODEL_PROVIDER = "openai"  # Options: "openai" or "gemini"

# === Load API keys from variable (assumed already defined) ===
OPENAI_API_KEY = openai_key
GEMINI_API_KEY = gemini_key
api_key = OPENAI_API_KEY if MODEL_PROVIDER == "openai" else GEMINI_API_KEY

# === Set model name based on provider ===
if MODEL_PROVIDER == "openai":
    MODEL_NAME = "gpt-4o"
    ESTIMATED_COST = 0.005 + 600 / 1000 * 0.01  # image + output
else:
    MODEL_NAME = "gemini-1.5-pro"
    ESTIMATED_COST = 600 / 1000 * 0.0025

# === Launch Analyzer ===
print(f"🔍 Selected: {MODEL_PROVIDER.upper()} - Model: {MODEL_NAME}")
analyzer = ImageAnalyzerAI(model_provider=MODEL_PROVIDER, model_name=MODEL_NAME, api_key=api_key)
analyzer.upload_and_analyze_images(rule_id="trend_strength")

# We'll display cost after upload completes
import time
time.sleep(1.0)
print(f"💰 Estimated cost per image: ${ESTIMATED_COST:.4f}")


# In[25]:


import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

def pick_snapshots_and_analyze(analyzer, rule_id="trend_strength"):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select snapshot images",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    results = {}
    for path in file_paths:
        symbol = Path(path).stem.upper()
        print(f"🧠 Analyzing {symbol}...")

        try:
            result = analyzer.analyze_image_with_bytes(str(path), rule_id)
            results[symbol] = {
                "raw_output": result["raw_output"],
                "timeframe": analyzer.get_prompt_by_id(rule_id)["timeframe"]
            }
        except Exception as e:
            results[symbol] = {
                "label": "Error",
                "score": 0,
                "timeframe": analyzer.get_prompt_by_id(rule_id)["timeframe"],
                "explanation": f"⚠️ Failed to process image: {e}"
            }

    analyzer.image_analysis = results
    print("✅ All snapshots processed.")
    return results


# In[26]:


# pick_snapshots_and_analyze(analyzer)


# In[ ]:





# In[27]:


from PIL import ImageGrab
from datetime import datetime
from pathlib import Path

def analyze_clipboard_snapshot(analyzer, rule_id="trend_strength"):
    print("📋 Waiting for snapshot from clipboard...")

    image = ImageGrab.grabclipboard()
    if image is None:
        print("❌ No image found in clipboard. Use Snipping Tool or press PrtScr, then try again.")
        return

    file_name = f"snapshot_{datetime.now().strftime('%H%M%S')}.png"
    image_path = Path(file_name)
    image.save(image_path)

    print(f"✅ Snapshot saved as: {image_path.name}")
    result = analyzer.analyze_image_with_bytes(str(image_path), rule_id)
    print("🔍 Analysis Result:")
    print(result["raw_output"])
    return result


# In[ ]:




