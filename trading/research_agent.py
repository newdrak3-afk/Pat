"""
Research Agent — Parallel scraping of Twitter, Reddit, RSS feeds.

Runs sentiment analysis and compares narrative against market odds.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree

import requests

from trading.config import ResearchConfig, SystemConfig
from trading.models import Market, ResearchResult

logger = logging.getLogger(__name__)

# Simple sentiment lexicon for fast analysis without heavy ML deps
POSITIVE_WORDS = {
    "bullish", "surge", "soar", "rally", "breakout", "moon", "pump", "win",
    "victory", "confirmed", "approved", "passed", "success", "growth", "gain",
    "profit", "strong", "increase", "up", "higher", "likely", "certain",
    "definitely", "positive", "optimistic", "boost", "momentum", "trending",
    "landslide", "dominant", "leading", "ahead", "favored", "lock",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "plunge", "collapse", "tank", "sell", "loss",
    "fail", "rejected", "denied", "scandal", "fraud", "risk", "decline",
    "drop", "down", "lower", "unlikely", "doubtful", "negative", "pessimistic",
    "weak", "losing", "behind", "trailing", "underdog", "worried", "concern",
    "controversy", "investigation", "lawsuit", "indicted",
}


class ResearchAgent:
    """
    Scrapes Twitter, Reddit, and RSS feeds in parallel.
    Runs sentiment analysis and compares narrative with market odds.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.research_cfg = self.config.research

    def research_market(self, market: Market) -> ResearchResult:
        """Run parallel research on a single market. Returns aggregated result."""
        logger.info(f"Researching market: {market.question[:60]}...")

        # Extract search keywords from market question
        keywords = self._extract_keywords(market.question)
        logger.info(f"Search keywords: {keywords}")

        # Run all sources in parallel
        twitter_data = []
        reddit_data = []
        rss_data = []

        with ThreadPoolExecutor(
            max_workers=self.research_cfg.parallel_workers
        ) as executor:
            futures = {}

            # Twitter
            futures[executor.submit(
                self._scrape_twitter, keywords
            )] = "twitter"

            # Reddit
            futures[executor.submit(
                self._scrape_reddit, keywords
            )] = "reddit"

            # RSS
            futures[executor.submit(
                self._scrape_rss, keywords
            )] = "rss"

            for future in as_completed(
                futures, timeout=self.config.research_timeout_seconds
            ):
                source = futures[future]
                try:
                    result = future.result()
                    if source == "twitter":
                        twitter_data = result
                    elif source == "reddit":
                        reddit_data = result
                    elif source == "rss":
                        rss_data = result
                    logger.info(f"  {source}: {len(result)} items collected")
                except Exception as e:
                    logger.warning(f"  {source} failed: {e}")

        # Sentiment analysis on each source
        twitter_sentiment = self._analyze_sentiment(twitter_data)
        reddit_sentiment = self._analyze_sentiment(reddit_data)
        rss_sentiment = self._analyze_sentiment(rss_data)

        # Weighted combination (Reddit and Twitter weigh more for prediction markets)
        weights = {"twitter": 0.35, "reddit": 0.35, "rss": 0.30}
        combined = (
            twitter_sentiment * weights["twitter"]
            + reddit_sentiment * weights["reddit"]
            + rss_sentiment * weights["rss"]
        )

        # Label
        if combined > 0.3:
            label = "very_bullish"
        elif combined > 0.1:
            label = "bullish"
        elif combined < -0.3:
            label = "very_bearish"
        elif combined < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        # Compare narrative vs odds
        # Market price near 1.0 = market says very likely
        # Positive sentiment = narrative says likely
        narrative_implied = (combined + 1) / 2  # map [-1,1] to [0,1]
        alignment = 1.0 - abs(narrative_implied - market.current_price)

        # Key signals
        all_texts = twitter_data + reddit_data + rss_data
        key_signals = self._extract_key_signals(all_texts, keywords)

        # Narrative summary
        narrative = self._build_narrative(
            market, label, combined, alignment, key_signals
        )

        result = ResearchResult(
            market_id=market.market_id,
            twitter_sentiment=twitter_sentiment,
            reddit_sentiment=reddit_sentiment,
            rss_sentiment=rss_sentiment,
            combined_sentiment=combined,
            sentiment_label=label,
            narrative_summary=narrative,
            source_count=len(all_texts),
            key_signals=key_signals[:10],
            narrative_vs_odds_alignment=alignment,
        )

        logger.info(
            f"Research complete: sentiment={label} ({combined:.2f}), "
            f"alignment={alignment:.2f}"
        )
        return result

    def research_markets_parallel(
        self, markets: list[Market]
    ) -> list[ResearchResult]:
        """Research multiple markets in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_market = {
                executor.submit(self.research_market, m): m for m in markets
            }
            for future in as_completed(future_to_market):
                market = future_to_market[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Research failed for {market.market_id}: {e}"
                    )

        return results

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract meaningful search keywords from a market question."""
        # Remove common question words
        stop_words = {
            "will", "the", "be", "is", "are", "was", "were", "has", "have",
            "had", "do", "does", "did", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below", "between",
            "this", "that", "these", "those", "it", "its", "what", "which",
            "who", "whom", "when", "where", "why", "how", "not", "no", "yes",
            "than", "more", "most", "other", "some", "such", "only", "own",
            "same", "so", "can", "could", "would", "should", "shall", "may",
            "might", "must", "need", "if", "then", "there", "about", "over",
            "under", "again", "once", "also", "any", "each", "every", "both",
            "few", "many", "much", "very",
        }

        # Clean and split
        cleaned = re.sub(r"[^\w\s]", " ", question)
        words = cleaned.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Take top keywords (preserve order, limit to 5)
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:5]

    def _scrape_twitter(self, keywords: list[str]) -> list[str]:
        """Scrape Twitter/X for relevant posts."""
        token = self.research_cfg.twitter_bearer_token
        if not token:
            logger.debug("Twitter bearer token not set, using fallback search")
            return self._fallback_twitter_search(keywords)

        query = " OR ".join(keywords) + " -is:retweet lang:en"
        texts = []

        try:
            resp = requests.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "query": query,
                    "max_results": min(
                        self.research_cfg.max_posts_per_source, 100
                    ),
                    "tweet.fields": "created_at,public_metrics",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for tweet in data.get("data", []):
                texts.append(tweet.get("text", ""))

        except requests.RequestException as e:
            logger.warning(f"Twitter API error: {e}")
            return self._fallback_twitter_search(keywords)

        return texts

    def _fallback_twitter_search(self, keywords: list[str]) -> list[str]:
        """Fallback: use Nitter or similar public scraping endpoint."""
        # Use a public search aggregator as fallback
        texts = []
        query = "+".join(keywords)

        try:
            # Try to get social media sentiment from a public aggregator
            resp = requests.get(
                f"https://api.duckduckgo.com/?q={query}+site:twitter.com&format=json",
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                for topic in data.get("RelatedTopics", []):
                    text = topic.get("Text", "")
                    if text:
                        texts.append(text)
        except requests.RequestException:
            pass

        return texts

    def _scrape_reddit(self, keywords: list[str]) -> list[str]:
        """Scrape Reddit for relevant posts and comments."""
        texts = []
        query = " ".join(keywords)

        # Relevant subreddits for prediction markets / events
        subreddits = [
            "polymarket", "predictions", "wallstreetbets", "politics",
            "sports", "cryptocurrency", "news", "worldnews",
        ]

        try:
            # Reddit search API (no auth needed for public posts)
            for subreddit in subreddits[:4]:
                try:
                    resp = requests.get(
                        f"https://www.reddit.com/r/{subreddit}/search.json",
                        params={
                            "q": query,
                            "sort": "relevance",
                            "t": "day",
                            "limit": self.research_cfg.max_posts_per_source
                            // len(subreddits),
                        },
                        headers={
                            "User-Agent": self.research_cfg.reddit_user_agent
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})
                        title = post_data.get("title", "")
                        selftext = post_data.get("selftext", "")[:500]
                        if title:
                            texts.append(f"{title} {selftext}".strip())

                    time.sleep(0.5)  # Reddit rate limiting

                except requests.RequestException as e:
                    logger.debug(f"Reddit r/{subreddit} search failed: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Reddit scraping error: {e}")

        return texts

    def _scrape_rss(self, keywords: list[str]) -> list[str]:
        """Scrape RSS feeds for relevant articles."""
        texts = []
        keyword_set = set(kw.lower() for kw in keywords)

        for feed_url in self.research_cfg.rss_feeds:
            try:
                resp = requests.get(feed_url, timeout=10)
                resp.raise_for_status()
                root = ElementTree.fromstring(resp.content)

                # Handle both RSS and Atom feeds
                items = root.findall(".//item") or root.findall(
                    ".//{http://www.w3.org/2005/Atom}entry"
                )

                for item in items[: self.research_cfg.max_posts_per_source]:
                    title_el = item.find("title") or item.find(
                        "{http://www.w3.org/2005/Atom}title"
                    )
                    desc_el = item.find("description") or item.find(
                        "{http://www.w3.org/2005/Atom}summary"
                    )

                    title = title_el.text if title_el is not None else ""
                    desc = desc_el.text if desc_el is not None else ""
                    text = f"{title} {desc}".strip()
                    text_lower = text.lower()

                    # Only include if relevant to our keywords
                    if any(kw in text_lower for kw in keyword_set):
                        # Strip HTML tags
                        clean = re.sub(r"<[^>]+>", "", text)
                        texts.append(clean[:500])

            except Exception as e:
                logger.debug(f"RSS feed {feed_url} failed: {e}")
                continue

        return texts

    def _analyze_sentiment(self, texts: list[str]) -> float:
        """
        Simple lexicon-based sentiment analysis.
        Returns score from -1 (very negative) to +1 (very positive).
        """
        if not texts:
            return 0.0

        total_score = 0.0

        for text in texts:
            words = set(re.findall(r"\w+", text.lower()))
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            total = pos + neg
            if total > 0:
                total_score += (pos - neg) / total

        return max(-1.0, min(1.0, total_score / max(len(texts), 1)))

    def _extract_key_signals(
        self, texts: list[str], keywords: list[str]
    ) -> list[str]:
        """Extract the most relevant sentences as key signals."""
        signals = []
        keyword_set = set(kw.lower() for kw in keywords)

        for text in texts:
            sentences = re.split(r"[.!?]+", text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                words = set(sentence.lower().split())
                # Relevance: how many keywords appear
                relevance = len(words & keyword_set)
                # Sentiment strength
                pos = len(words & POSITIVE_WORDS)
                neg = len(words & NEGATIVE_WORDS)
                strength = pos + neg

                if relevance > 0 and strength > 0:
                    signals.append((relevance + strength, sentence[:200]))

        # Sort by score descending
        signals.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in signals]

    def _build_narrative(
        self,
        market: Market,
        label: str,
        score: float,
        alignment: float,
        signals: list[str],
    ) -> str:
        """Build a human-readable narrative summary."""
        direction = "YES" if score > 0 else "NO" if score < 0 else "NEUTRAL"

        lines = [
            f"Market: {market.question}",
            f"Current price: {market.current_price:.3f} "
            f"(market implies {market.current_price*100:.1f}% probability)",
            f"Sentiment: {label.upper()} (score: {score:+.2f})",
            f"Narrative direction: {direction}",
            f"Narrative-odds alignment: {alignment:.1%}",
        ]

        if alignment < 0.7:
            lines.append(
                "** DIVERGENCE DETECTED: Narrative disagrees with market odds **"
            )

        if signals:
            lines.append("\nKey signals:")
            for s in signals[:5]:
                lines.append(f"  - {s}")

        return "\n".join(lines)
