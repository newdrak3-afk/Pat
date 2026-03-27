"""
Data models for the trading system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from enum import Enum


class MarketStatus(Enum):
    ACTIVE = "active"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


class TradeOutcome(Enum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    CANCELLED = "cancelled"


class Sentiment(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class Market:
    """A prediction market."""
    market_id: str
    question: str
    category: str = ""
    current_price: float = 0.5        # probability implied by market
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    end_date: Optional[str] = None
    outcomes: list = field(default_factory=lambda: ["Yes", "No"])
    flagged: bool = False
    flag_reasons: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class ResearchResult:
    """Aggregated research for a market."""
    market_id: str
    twitter_sentiment: float = 0.0       # -1 to 1
    reddit_sentiment: float = 0.0
    rss_sentiment: float = 0.0
    combined_sentiment: float = 0.0
    sentiment_label: str = "neutral"
    narrative_summary: str = ""
    source_count: int = 0
    key_signals: list = field(default_factory=list)
    narrative_vs_odds_alignment: float = 0.0  # how much narrative agrees with market
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class Prediction:
    """A calibrated prediction."""
    market_id: str
    market_question: str = ""
    market_price: float = 0.5
    predicted_probability: float = 0.5
    xgboost_probability: float = 0.5
    llm_probability: float = 0.5
    confidence: float = 0.0
    edge: float = 0.0                    # predicted_prob - market_price
    recommended_side: str = ""           # "Yes" or "No"
    reasoning: str = ""
    features_used: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class Trade:
    """A placed trade."""
    trade_id: str
    market_id: str
    market_question: str = ""
    side: str = ""                       # "Yes" or "No"
    amount: float = 0.0
    entry_price: float = 0.0
    predicted_probability: float = 0.5
    confidence: float = 0.0
    edge: float = 0.0
    outcome: str = "pending"
    pnl: float = 0.0
    placed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved_at: Optional[str] = None
    loss_analysis: Optional[str] = None
    lessons_learned: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Lesson:
    """A lesson learned from a losing trade."""
    lesson_id: str
    trade_id: str
    market_id: str
    category: str = ""                   # e.g. "sentiment_mismatch", "low_liquidity"
    description: str = ""
    rule_added: str = ""                 # new rule/filter added to prevent recurrence
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)
