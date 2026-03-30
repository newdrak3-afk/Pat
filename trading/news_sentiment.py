# trading/news_sentiment.py
"""
News Sentiment — Fetches financial news and gauges market sentiment.

Sources (free, no API key required):
1. Alpha Vantage News Sentiment (free tier: 25/day)
2. RSS feeds from major financial outlets
3. Fear & Greed Index (CNN)

Used as a soft filter — doesn't block trades, but adjusts confidence
and adds context to trade reasoning.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class NewsSentiment:
    """Sentiment summary for a symbol or the overall market."""
    symbol: str
    sentiment: str           # "bullish", "bearish", "neutral", "mixed"
    score: float             # -1.0 (very bearish) to +1.0 (very bullish)
    headline_count: int
    top_headlines: list[str]
    fear_greed: Optional[int]  # 0-100, None if unavailable
    last_updated: str


# Forex economic calendar keywords that move markets
HIGH_IMPACT_KEYWORDS = [
    "nfp", "non-farm", "fomc", "fed rate", "interest rate decision",
    "cpi", "inflation", "gdp", "employment", "unemployment",
    "ecb", "boe", "boj", "rba", "rbnz",
    "tariff", "trade war", "sanctions", "geopolitical",
    "recession", "default", "crisis", "crash",
]


class NewsReader:
    """
    Fetches news sentiment from free sources.

    Caches results to avoid hitting rate limits.
    """

    def __init__(self):
        self._cache: dict[str, tuple[NewsSentiment, float]] = {}
        self._cache_ttl = 1800  # 30 min cache
        self._market_sentiment: Optional[NewsSentiment] = None
        self._market_sentiment_time: float = 0

    def get_sentiment(self, symbol: str) -> NewsSentiment:
        """Get news sentiment for a symbol. Returns cached if recent."""
        # Check cache
        if symbol in self._cache:
            cached, ts = self._cache[symbol]
            if time.time() - ts < self._cache_ttl:
                return cached

        # Try Alpha Vantage first (has sentiment scoring)
        result = self._fetch_alphavantage(symbol)
        if not result:
            # Fallback to RSS
            result = self._fetch_rss(symbol)
        if not result:
            result = NewsSentiment(
                symbol=symbol,
                sentiment="neutral",
                score=0.0,
                headline_count=0,
                top_headlines=[],
                fear_greed=None,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        self._cache[symbol] = (result, time.time())
        return result

    def get_market_sentiment(self) -> NewsSentiment:
        """Get overall market sentiment (Fear & Greed + top news)."""
        if self._market_sentiment and time.time() - self._market_sentiment_time < self._cache_ttl:
            return self._market_sentiment

        fear_greed = self._fetch_fear_greed()
        headlines = self._fetch_market_headlines()

        # Compute overall sentiment from fear/greed
        if fear_greed is not None:
            if fear_greed >= 70:
                sentiment, score = "bullish", 0.5 + (fear_greed - 70) / 60
            elif fear_greed <= 30:
                sentiment, score = "bearish", -0.5 - (30 - fear_greed) / 60
            else:
                sentiment, score = "neutral", (fear_greed - 50) / 50
        else:
            sentiment, score = "neutral", 0.0

        # Check headlines for high-impact events
        high_impact = []
        for h in headlines:
            h_lower = h.lower()
            for kw in HIGH_IMPACT_KEYWORDS:
                if kw in h_lower:
                    high_impact.append(h)
                    break

        result = NewsSentiment(
            symbol="MARKET",
            sentiment=sentiment,
            score=score,
            headline_count=len(headlines),
            top_headlines=headlines[:5],
            fear_greed=fear_greed,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self._market_sentiment = result
        self._market_sentiment_time = time.time()
        return result

    def has_high_impact_event(self) -> tuple[bool, list[str]]:
        """Check if there are high-impact news events right now."""
        market = self.get_market_sentiment()
        impacts = []
        for h in market.top_headlines:
            h_lower = h.lower()
            for kw in HIGH_IMPACT_KEYWORDS:
                if kw in h_lower:
                    impacts.append(h)
                    break
        return len(impacts) > 0, impacts

    def _fetch_alphavantage(self, symbol: str) -> Optional[NewsSentiment]:
        """Fetch news from Alpha Vantage News Sentiment API (free)."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
        # Map forex symbols to tickers AV understands
        ticker = symbol.replace("_", "").replace("/", "")
        if ticker.startswith("BTC") or ticker.startswith("ETH"):
            topics = "blockchain"
        else:
            topics = "forex"

        url = (
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers=FOREX:{ticker}&topics={topics}"
            f"&sort=LATEST&limit=10&apikey={api_key}"
        )

        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()

            if "feed" not in data:
                return None

            articles = data["feed"]
            if not articles:
                return None

            # Compute average sentiment
            scores = []
            headlines = []
            for article in articles[:10]:
                headlines.append(article.get("title", ""))
                # AV provides sentiment for each ticker mentioned
                for ticker_info in article.get("ticker_sentiment", []):
                    if ticker in ticker_info.get("ticker", ""):
                        score = float(ticker_info.get("ticker_sentiment_score", 0))
                        scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            if avg_score > 0.15:
                sentiment = "bullish"
            elif avg_score < -0.15:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            return NewsSentiment(
                symbol=symbol,
                sentiment=sentiment,
                score=avg_score,
                headline_count=len(articles),
                top_headlines=headlines[:5],
                fear_greed=None,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.debug(f"Alpha Vantage news fetch failed: {e}")
            return None

    def _fetch_rss(self, symbol: str) -> Optional[NewsSentiment]:
        """Fetch news from free RSS feeds as fallback."""
        # Use Forex Factory / Investing.com RSS
        feeds = [
            "https://www.forexfactory.com/rss",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US",
        ]

        headlines = []
        for feed_url in feeds:
            try:
                resp = requests.get(feed_url, timeout=8, headers={
                    "User-Agent": "Mozilla/5.0"
                })
                # Simple XML parsing for <title> tags
                import re
                titles = re.findall(r"<title[^>]*>(.*?)</title>", resp.text)
                headlines.extend(titles[:5])
            except Exception:
                continue

        if not headlines:
            return None

        # Simple keyword sentiment
        bullish_words = ["rally", "surge", "gain", "rise", "bullish", "high", "up", "growth", "strong"]
        bearish_words = ["drop", "fall", "crash", "bear", "decline", "low", "down", "weak", "fear", "recession"]

        bull_count = sum(1 for h in headlines for w in bullish_words if w in h.lower())
        bear_count = sum(1 for h in headlines for w in bearish_words if w in h.lower())

        total = bull_count + bear_count
        if total > 0:
            score = (bull_count - bear_count) / total
        else:
            score = 0.0

        sentiment = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"

        return NewsSentiment(
            symbol=symbol,
            sentiment=sentiment,
            score=score,
            headline_count=len(headlines),
            top_headlines=headlines[:5],
            fear_greed=None,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def _fetch_fear_greed(self) -> Optional[int]:
        """Fetch CNN Fear & Greed Index (free, no key needed)."""
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0"
            })
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score")
            if score is not None:
                return int(score)
        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
        return None

    def _fetch_market_headlines(self) -> list[str]:
        """Fetch general market headlines."""
        headlines = []
        try:
            url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            import re
            titles = re.findall(r"<title[^>]*>(.*?)</title>", resp.text)
            headlines.extend(titles[1:6])  # Skip feed title
        except Exception:
            pass
        return headlines

    def format_sentiment_report(self) -> str:
        """Format a full sentiment report for Telegram."""
        market = self.get_market_sentiment()

        lines = [
            "NEWS & SENTIMENT",
            "",
        ]

        if market.fear_greed is not None:
            fg = market.fear_greed
            if fg <= 25:
                fg_label = "Extreme Fear"
            elif fg <= 45:
                fg_label = "Fear"
            elif fg <= 55:
                fg_label = "Neutral"
            elif fg <= 75:
                fg_label = "Greed"
            else:
                fg_label = "Extreme Greed"
            lines.append(f"  Fear & Greed: {fg} ({fg_label})")
        else:
            lines.append("  Fear & Greed: unavailable")

        lines.append(f"  Market Mood:  {market.sentiment.upper()}")
        lines.append("")

        if market.top_headlines:
            lines.append("  Top Headlines:")
            for h in market.top_headlines[:5]:
                # Truncate long headlines
                h_short = h[:80] + "..." if len(h) > 80 else h
                lines.append(f"  - {h_short}")

        has_impact, impacts = self.has_high_impact_event()
        if has_impact:
            lines.append("")
            lines.append("  HIGH IMPACT EVENTS:")
            for h in impacts[:3]:
                lines.append(f"  - {h[:80]}")

        return "\n".join(lines)
