import requests
import feedparser

from logging_setup import get_logger

log = get_logger(__name__)

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
                log.info("⚠️ NewsAPI key missing, skipping NewsAPI fetch.")
                return

            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=30&apiKey={self.newsapi_key}"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                headlines = [article.get("title", "") for article in data.get("articles", [])]
                self.headlines.extend(headlines)
                log.info(f"✅ Pulled {len(headlines)} headlines from NewsAPI.")
            else:
                log.info(f"⚠️ NewsAPI error: {data.get('message')}")

        except Exception as e:
            log.info(f"❌ Error fetching from NewsAPI: {e}")

    def fetch_from_finnhub(self):
        """
        Fetch general financial news using Finnhub API.
        """
        try:
            if not self.finnhub_key:
                log.info("⚠️ Finnhub key missing, skipping Finnhub fetch.")
                return

            url = f"https://finnhub.io/api/v1/news?category=general&token={self.finnhub_key}"
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                headlines = [article.get("headline", "") for article in data]
                self.headlines.extend(headlines)
                log.info(f"✅ Pulled {len(headlines)} headlines from Finnhub.")
            else:
                log.info(f"⚠️ Finnhub error: {response.json().get('error')}")

        except Exception as e:
            log.info(f"❌ Error fetching from Finnhub: {e}")

    def fetch_from_yahoo_finance_rss(self):
        """
        Fetch headlines from Yahoo Finance RSS feed.
        """
        try:
            url = "https://finance.yahoo.com/news/rssindex"
            feed = feedparser.parse(url)

            headlines = [entry.title for entry in feed.entries]
            self.headlines.extend(headlines)
            log.info(f"✅ Pulled {len(headlines)} headlines from Yahoo Finance RSS.")

        except Exception as e:
            log.info(f"❌ Error fetching from Yahoo RSS: {e}")

    def aggregate_news(self):
        """
        Fetch from all sources and merge headlines.
        """
        log.info("📡 Aggregating news from multiple sources...")

        self.fetch_from_newsapi()
        self.fetch_from_finnhub()
        self.fetch_from_yahoo_finance_rss()

        log.info(f"🧠 Total aggregated headlines: {len(self.headlines)}")

        # Remove duplicates
        merged_headlines = list(set(self.headlines))
        log.info(f"🧹 After removing duplicates: {len(merged_headlines)} unique headlines.")

        return merged_headlines

    def fetch_from_investing_rss(self):
        """
        Fetch headlines from Investing.com RSS feeds (general + economic).
        """
        try:
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
            log.info(f"✅ Pulled {len(all_headlines)} headlines from Investing.com RSS.")
    
        except Exception as e:
            log.info(f"❌ Error fetching from Investing.com RSS: {e}")
    
    def fetch_company_profile(self, symbol: str) -> dict:
        """
        Fetches basic company profile from FMP (free endpoint).
        
        Parameters:
            symbol (str): Stock ticker symbol (e.g., AAPL, MSFT)
    
        Returns:
            dict: A dictionary with sector, industry, CEO, and description if successful.
        """
        api_key = self.config.get("fmp_key")
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
                    log.info(f"⚠️ No profile data found for {symbol}")
                    return {}
            else:
                log.info(f"⚠️ Error fetching FMP profile for {symbol}: {response.status_code}")
                return {}
        except Exception as e:
            log.info(f"⚠️ Exception during FMP profile fetch: {e}")
        return {} 