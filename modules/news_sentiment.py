"""
news_sentiment.py — News fetching and sentiment analysis for the options trading bot.

Features:
- Yahoo Finance RSS headlines (no API key required)
- NewsAPI.org integration (optional, free tier 100 req/day)
- TextBlob sentiment scoring on headlines
- Earnings calendar awareness (flags earnings within 5 days)
- Macro event detection (FOMC, CPI, NFP, etc.)
- Distinguishes bullish / bearish / uncertain / mixed sentiment
- Provides catalyst risk summary for the reasoning engine
"""

import os
import json
import time
import hashlib
import feedparser
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Optional

# Macro events that should trigger trade caution flags
MACRO_KEYWORDS = [
    "fed", "fomc", "interest rate", "rate decision", "powell",
    "cpi", "inflation", "ppi", "nfp", "jobs report", "unemployment",
    "gdp", "recession", "debt ceiling", "treasury", "yield curve",
]

# Earnings-related keywords
EARNINGS_KEYWORDS = [
    "earnings", "quarterly results", "eps", "revenue miss", "revenue beat",
    "guidance", "outlook", "profit warning",
]


class NewsSentiment:
    """Fetch and score news headlines for a given ticker."""

    CACHE_TTL = 900  # 15 minutes cache to avoid hammering feeds

    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self._cache: dict = {}  # {ticker: (timestamp, data)}

    # ──────────────────────────────────────────────
    # HEADLINE FETCHING
    # ──────────────────────────────────────────────

    def get_news(self, ticker: str, max_items: int = 15) -> list:
        """
        Fetch recent headlines for a ticker.
        Tries Yahoo Finance RSS first, then NewsAPI if key is present.

        Returns list of dicts:
            {headline, source, published, url, age_hours}
        """
        cache_key = ticker.upper()
        now = time.time()
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < self.CACHE_TTL:
                return cached[:max_items]

        headlines = []
        headlines.extend(self._fetch_yahoo_rss(ticker))
        if self.news_api_key and len(headlines) < 5:
            headlines.extend(self._fetch_newsapi(ticker))

        # Deduplicate by headline hash
        seen = set()
        unique = []
        for item in headlines:
            h = hashlib.md5(item["headline"].lower().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(item)

        unique.sort(key=lambda x: x.get("age_hours", 999))
        self._cache[cache_key] = (now, unique)
        return unique[:max_items]

    def _fetch_yahoo_rss(self, ticker: str) -> list:
        """Fetch from Yahoo Finance RSS feed."""
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries[:20]:
                published = entry.get("published_parsed")
                if published:
                    pub_dt = datetime(*published[:6])
                    age_h = (datetime.utcnow() - pub_dt).total_seconds() / 3600
                else:
                    age_h = 999

                items.append({
                    "headline": entry.get("title", "").strip(),
                    "source": "Yahoo Finance",
                    "published": entry.get("published", ""),
                    "url": entry.get("link", ""),
                    "age_hours": round(age_h, 1),
                })
            return items
        except Exception:
            return []

    def _fetch_newsapi(self, ticker: str) -> list:
        """Fetch from NewsAPI.org (requires NEWS_API_KEY)."""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": self.news_api_key,
            }
            r = requests.get(url, params=params, timeout=8)
            if r.status_code != 200:
                return []
            articles = r.json().get("articles", [])
            items = []
            for a in articles:
                pub = a.get("publishedAt", "")
                try:
                    pub_dt = datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
                    age_h = (datetime.utcnow() - pub_dt).total_seconds() / 3600
                except Exception:
                    age_h = 999
                items.append({
                    "headline": a.get("title", "").strip(),
                    "source": a.get("source", {}).get("name", "NewsAPI"),
                    "published": pub,
                    "url": a.get("url", ""),
                    "age_hours": round(age_h, 1),
                })
            return items
        except Exception:
            return []

    # ──────────────────────────────────────────────
    # SENTIMENT SCORING
    # ──────────────────────────────────────────────

    def score_sentiment(self, ticker: str) -> dict:
        """
        Score sentiment for a ticker using TextBlob polarity on recent headlines.

        Returns:
            {
                score: float (-1.0 to 1.0),
                label: 'bullish' | 'bearish' | 'neutral' | 'mixed',
                confidence: float (0-1, based on agreement among headlines),
                headlines_scored: int,
                top_headlines: list of str,
                catalyst_risk: bool,
                catalyst_reason: str,
            }
        """
        headlines = self.get_news(ticker, max_items=15)

        if not headlines:
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "headlines_scored": 0,
                "top_headlines": [],
                "catalyst_risk": False,
                "catalyst_reason": "",
            }

        # Only score recent headlines (< 48 hours)
        recent = [h for h in headlines if h.get("age_hours", 999) < 48]
        if not recent:
            recent = headlines[:5]

        scores = []
        for item in recent:
            text = item["headline"]
            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Measure agreement: how many go same direction as average
        if avg_score > 0:
            agreement = sum(1 for s in scores if s > 0) / len(scores)
        elif avg_score < 0:
            agreement = sum(1 for s in scores if s < 0) / len(scores)
        else:
            agreement = 0.5

        # Label
        if avg_score > 0.15 and agreement > 0.55:
            label = "bullish"
        elif avg_score < -0.15 and agreement > 0.55:
            label = "bearish"
        elif abs(avg_score) < 0.08:
            label = "neutral"
        else:
            label = "mixed"

        # Catalyst risk check
        catalyst_risk, catalyst_reason = self._check_catalyst_risk(ticker, headlines)

        top = [h["headline"] for h in recent[:3]]

        return {
            "score": round(avg_score, 3),
            "label": label,
            "confidence": round(agreement, 2),
            "headlines_scored": len(recent),
            "top_headlines": top,
            "catalyst_risk": catalyst_risk,
            "catalyst_reason": catalyst_reason,
        }

    # ──────────────────────────────────────────────
    # EARNINGS & MACRO RISK
    # ──────────────────────────────────────────────

    def _check_catalyst_risk(self, ticker: str, headlines: list) -> tuple:
        """
        Detect earnings announcements or macro events in recent headlines.
        Returns (risk_flag: bool, reason: str).
        """
        all_text = " ".join(h["headline"].lower() for h in headlines)

        # Check for earnings mentions
        for kw in EARNINGS_KEYWORDS:
            if kw in all_text:
                return True, f"Earnings-related news detected ('{kw}' in headlines)"

        # Check for macro mentions
        for kw in MACRO_KEYWORDS:
            if kw in all_text:
                return True, f"Macro event detected ('{kw}' in headlines)"

        # Check for upcoming earnings via yfinance
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not cal.empty:
                # Earnings Date column
                earnings_col = None
                for col in cal.columns:
                    if "earnings" in str(col).lower():
                        earnings_col = col
                        break
                if earnings_col:
                    ed = cal[earnings_col].iloc[0]
                    if hasattr(ed, "date"):
                        ed = ed.date()
                    days_away = (ed - datetime.today().date()).days
                    if 0 <= days_away <= 5:
                        return True, f"Earnings in {days_away} day(s) ({ed})"
        except Exception:
            pass

        return False, ""

    def get_catalyst_summary(self, ticker: str) -> str:
        """
        Return a plain-English summary of news and catalyst risk for the reasoning engine.

        Example output:
            "AAPL: 3 bullish headlines in last 24h including 'Apple beats earnings estimates by 12%'.
             No major catalyst risk detected."
        """
        sentiment = self.score_sentiment(ticker)
        count = sentiment["headlines_scored"]
        label = sentiment["label"].upper()
        top = sentiment["top_headlines"]

        if count == 0:
            summary = f"{ticker}: No recent news found."
        else:
            top_str = f" — top headline: \"{top[0]}\"" if top else ""
            summary = f"{ticker}: {count} headlines scored {label}{top_str}."

        if sentiment["catalyst_risk"]:
            summary += f" ⚠ CATALYST RISK: {sentiment['catalyst_reason']}"
        else:
            summary += " No major catalyst risk detected."

        return summary
