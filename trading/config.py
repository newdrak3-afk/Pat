"""
Configuration for the multi-agent trading system.
All thresholds, API keys, and tunable parameters live here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ScanConfig:
    """Market scanner settings."""
    min_liquidity_usd: float = 5_000.0
    min_24h_volume_usd: float = 1_000.0
    max_spread_pct: float = 5.0          # flag if spread > 5%
    price_move_zscore: float = 2.0       # flag if price move > 2 std devs
    time_resolutions: list = field(default_factory=lambda: ["1h", "4h", "1d"])
    max_markets: int = 500               # scan up to this many
    min_active_markets: int = 300        # ensure we scan at least 300


@dataclass
class ResearchConfig:
    """Research agent settings."""
    twitter_bearer_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TWITTER_BEARER_TOKEN")
    )
    reddit_client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_ID")
    )
    reddit_client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET")
    )
    reddit_user_agent: str = "PatTradingBot/1.0"
    rss_feeds: list = field(default_factory=lambda: [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://polymarket.com/rss",
    ])
    max_posts_per_source: int = 50
    sentiment_window_hours: int = 24
    parallel_workers: int = 3


@dataclass
class PredictionConfig:
    """Prediction agent settings."""
    confidence_threshold: float = 0.60       # only fire when confidence > 60%
    edge_threshold: float = 0.05             # min 5% edge over market price
    xgboost_n_estimators: int = 200
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    model_save_path: str = "trading/data/xgb_model.json"
    training_data_path: str = "trading/data/training_data.json"


@dataclass
class RiskConfig:
    """Risk management settings."""
    max_bet_pct: float = 0.02           # max 2% of bankroll per trade
    max_daily_loss_pct: float = 0.05    # stop trading if daily loss > 5%
    max_open_positions: int = 5
    max_correlation: float = 0.7        # block correlated bets
    min_bankroll: float = 10.0          # minimum bankroll to keep trading
    kelly_fraction: float = 0.25        # quarter-Kelly for safety
    cooldown_after_loss_minutes: int = 30


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    scan: ScanConfig = field(default_factory=ScanConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Data paths
    trades_log: str = "trading/data/trades.json"
    lessons_log: str = "trading/data/lessons.json"
    positions_file: str = "trading/data/positions.json"
    bankroll_file: str = "trading/data/bankroll.json"

    # Polling
    scan_interval_seconds: int = 300     # scan every 5 minutes
    research_timeout_seconds: int = 60

    # Prediction market API
    polymarket_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
