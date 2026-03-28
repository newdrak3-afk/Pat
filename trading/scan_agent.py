"""
Scan Agent — Filters 300+ active prediction markets.

Checks liquidity, volume, time resolution, flags weird price moves and wide spreads.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np

from trading.config import ScanConfig, SystemConfig
from trading.models import Market

logger = logging.getLogger(__name__)


class ScanAgent:
    """
    Scans prediction markets (Polymarket) and filters by:
    - Minimum liquidity
    - Minimum 24h volume
    - Time resolution (how soon the market resolves)
    - Flags anomalous price movements (z-score)
    - Flags wide spreads
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.scan_cfg = self.config.scan
        self.api_url = self.config.gamma_api_url
        self._price_history: dict[str, list[float]] = {}
        self._lessons: list[dict] = self._load_lessons()

    def _load_lessons(self) -> list[dict]:
        """Load learned lessons to apply as additional filters."""
        try:
            with open(self.config.lessons_log) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def scan(self) -> list[Market]:
        """Main scan: fetch markets, filter, flag anomalies."""
        logger.info("Scan agent starting — fetching active markets...")

        raw_markets = self._fetch_markets()
        logger.info(f"Fetched {len(raw_markets)} raw markets")

        # Step 1: Filter by liquidity and volume
        filtered = self._filter_by_liquidity_volume(raw_markets)
        logger.info(f"After liquidity/volume filter: {len(filtered)} markets")

        # Step 2: Filter by time resolution
        filtered = self._filter_by_time_resolution(filtered)
        logger.info(f"After time resolution filter: {len(filtered)} markets")

        # Step 3: Apply learned lesson filters
        filtered = self._apply_lesson_filters(filtered)
        logger.info(f"After lesson filters: {len(filtered)} markets")

        # Step 4: Flag anomalies
        flagged = self._flag_anomalies(filtered)

        flagged_count = sum(1 for m in flagged if m.flagged)
        logger.info(
            f"Scan complete: {len(flagged)} active markets, "
            f"{flagged_count} flagged for anomalies"
        )

        return flagged

    def _fetch_markets(self) -> list[dict]:
        """Fetch active markets from Polymarket Gamma API."""
        markets = []
        offset = 0
        limit = 100

        while len(markets) < self.scan_cfg.max_markets:
            try:
                resp = requests.get(
                    f"{self.api_url}/markets",
                    params={
                        "limit": limit,
                        "offset": offset,
                        "active": True,
                        "closed": False,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                markets.extend(batch)
                offset += limit

                # Rate limiting
                time.sleep(0.3)

            except requests.RequestException as e:
                logger.warning(f"API error fetching markets at offset {offset}: {e}")
                break

        return markets[:self.scan_cfg.max_markets]

    def _filter_by_liquidity_volume(self, raw_markets: list[dict]) -> list[Market]:
        """Keep only markets with sufficient liquidity and volume."""
        filtered = []

        for m in raw_markets:
            try:
                liquidity = float(m.get("liquidity", 0) or 0)
                volume = float(m.get("volume", 0) or 0)
                spread = float(m.get("spread", 0) or 0)

                if (liquidity >= self.scan_cfg.min_liquidity_usd and
                        volume >= self.scan_cfg.min_24h_volume_usd):

                    # Extract best price as midpoint
                    best_bid = float(m.get("bestBid", 0) or 0)
                    best_ask = float(m.get("bestAsk", 0) or 0)
                    outcomePrices = m.get("outcomePrices", "")

                    if isinstance(outcomePrices, str) and outcomePrices:
                        try:
                            prices = json.loads(outcomePrices)
                            current_price = float(prices[0]) if prices else 0.5
                        except (json.JSONDecodeError, IndexError):
                            current_price = 0.5
                    elif best_bid > 0 and best_ask > 0:
                        current_price = (best_bid + best_ask) / 2
                    else:
                        current_price = 0.5

                    # Calculate spread percentage
                    if best_bid > 0 and best_ask > 0:
                        spread_pct = ((best_ask - best_bid) / best_ask) * 100
                    else:
                        spread_pct = spread

                    market = Market(
                        market_id=str(m.get("id", "")),
                        question=m.get("question", "Unknown"),
                        category=m.get("category", ""),
                        current_price=current_price,
                        volume_24h=volume,
                        liquidity=liquidity,
                        spread=spread_pct,
                        end_date=m.get("endDate", None),
                        outcomes=m.get("outcomes", ["Yes", "No"])
                        if isinstance(m.get("outcomes"), list)
                        else ["Yes", "No"],
                    )
                    filtered.append(market)

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed market: {e}")
                continue

        return filtered

    def _filter_by_time_resolution(self, markets: list[Market]) -> list[Market]:
        """Filter by market end date / time resolution."""
        now = datetime.utcnow()
        filtered = []

        for market in markets:
            if not market.end_date:
                # No end date — still include but note it
                filtered.append(market)
                continue

            try:
                end_dt = datetime.fromisoformat(
                    market.end_date.replace("Z", "+00:00")
                ).replace(tzinfo=None)

                # Keep markets that resolve within a reasonable timeframe
                # (1 hour to 90 days out)
                time_to_resolve = end_dt - now
                if timedelta(hours=1) <= time_to_resolve <= timedelta(days=90):
                    filtered.append(market)
            except (ValueError, TypeError):
                filtered.append(market)

        return filtered

    def _apply_lesson_filters(self, markets: list[Market]) -> list[Market]:
        """Apply any learned rules from past losses as additional filters."""
        if not self._lessons:
            return markets

        filtered = []
        block_categories = set()
        block_keywords = set()
        min_liquidity_override = self.scan_cfg.min_liquidity_usd

        for lesson in self._lessons:
            rule = lesson.get("rule_added", "")
            cat = lesson.get("category", "")

            if "block_category:" in rule:
                block_categories.add(rule.split("block_category:")[1].strip())
            if "block_keyword:" in rule:
                block_keywords.add(rule.split("block_keyword:")[1].strip().lower())
            if "min_liquidity:" in rule:
                try:
                    val = float(rule.split("min_liquidity:")[1].strip())
                    min_liquidity_override = max(min_liquidity_override, val)
                except ValueError:
                    pass

        for market in markets:
            # Check category blocks
            if market.category in block_categories:
                logger.info(
                    f"Blocked market {market.market_id} — "
                    f"category '{market.category}' blocked by lesson"
                )
                continue

            # Check keyword blocks
            question_lower = market.question.lower()
            blocked = False
            for kw in block_keywords:
                if kw in question_lower:
                    logger.info(
                        f"Blocked market {market.market_id} — "
                        f"keyword '{kw}' blocked by lesson"
                    )
                    blocked = True
                    break
            if blocked:
                continue

            # Check liquidity override
            if market.liquidity < min_liquidity_override:
                continue

            filtered.append(market)

        return filtered

    def _flag_anomalies(self, markets: list[Market]) -> list[Market]:
        """Flag markets with weird price moves or wide spreads."""
        for market in markets:
            reasons = []

            # Wide spread flag
            if market.spread > self.scan_cfg.max_spread_pct:
                reasons.append(
                    f"Wide spread: {market.spread:.1f}% "
                    f"(threshold: {self.scan_cfg.max_spread_pct}%)"
                )

            # Track price history for z-score calculation
            mid = market.market_id
            if mid not in self._price_history:
                self._price_history[mid] = []
            self._price_history[mid].append(market.current_price)

            # Z-score on price if we have history
            history = self._price_history[mid]
            if len(history) >= 5:
                prices = np.array(history[-20:])  # last 20 readings
                mean = np.mean(prices)
                std = np.std(prices)
                if std > 0:
                    zscore = abs((market.current_price - mean) / std)
                    if zscore > self.scan_cfg.price_move_zscore:
                        reasons.append(
                            f"Anomalous price move: z-score={zscore:.2f} "
                            f"(threshold: {self.scan_cfg.price_move_zscore})"
                        )

            # Very low liquidity relative to volume (potential manipulation)
            if market.volume_24h > 0 and market.liquidity > 0:
                vol_liq_ratio = market.volume_24h / market.liquidity
                if vol_liq_ratio > 10:
                    reasons.append(
                        f"High volume/liquidity ratio: {vol_liq_ratio:.1f}x "
                        f"(possible manipulation)"
                    )

            if reasons:
                market.flagged = True
                market.flag_reasons = reasons

        return markets

    def get_summary(self, markets: list[Market]) -> str:
        """Return a human-readable scan summary."""
        flagged = [m for m in markets if m.flagged]
        lines = [
            f"=== SCAN SUMMARY ===",
            f"Total active markets: {len(markets)}",
            f"Flagged markets: {len(flagged)}",
            "",
        ]

        if flagged:
            lines.append("FLAGGED MARKETS:")
            for m in flagged[:20]:
                lines.append(f"  [{m.market_id[:8]}] {m.question[:60]}")
                lines.append(
                    f"    Price: {m.current_price:.3f} | "
                    f"Vol: ${m.volume_24h:,.0f} | "
                    f"Liq: ${m.liquidity:,.0f} | "
                    f"Spread: {m.spread:.1f}%"
                )
                for r in m.flag_reasons:
                    lines.append(f"    ! {r}")
                lines.append("")

        return "\n".join(lines)
