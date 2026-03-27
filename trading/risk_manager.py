"""
Risk Manager — Bet sizing, account protection, trade blocking.

Ensures bets are small relative to account, blocks risky trades,
and enforces daily loss limits and position limits.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from trading.config import RiskConfig, SystemConfig
from trading.models import Market, Prediction, Trade

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Controls bet sizing and blocks dangerous trades.

    Rules:
    - Max bet = kelly_fraction * Kelly criterion, capped at max_bet_pct of bankroll
    - Block if daily loss exceeds max_daily_loss_pct
    - Block if too many open positions
    - Block if correlated with existing positions
    - Cooldown period after a loss
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.risk_cfg = self.config.risk
        self._bankroll: float = 0.0
        self._positions: list[dict] = []
        self._daily_pnl: float = 0.0
        self._last_loss_time: Optional[datetime] = None
        self._load_state()

    def _load_state(self):
        """Load bankroll and positions from disk."""
        try:
            with open(self.config.bankroll_file) as f:
                data = json.load(f)
                self._bankroll = data.get("bankroll", 100.0)
                self._daily_pnl = data.get("daily_pnl", 0.0)
                last_loss = data.get("last_loss_time")
                if last_loss:
                    self._last_loss_time = datetime.fromisoformat(last_loss)
                # Reset daily PnL if it's a new day
                last_date = data.get("last_date", "")
                today = datetime.utcnow().strftime("%Y-%m-%d")
                if last_date != today:
                    self._daily_pnl = 0.0
        except (FileNotFoundError, json.JSONDecodeError):
            self._bankroll = 100.0  # default starting bankroll
            self._daily_pnl = 0.0

        try:
            with open(self.config.positions_file) as f:
                self._positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._positions = []

    def _save_state(self):
        """Persist bankroll and positions."""
        os.makedirs(os.path.dirname(self.config.bankroll_file), exist_ok=True)

        with open(self.config.bankroll_file, "w") as f:
            json.dump(
                {
                    "bankroll": self._bankroll,
                    "daily_pnl": self._daily_pnl,
                    "last_loss_time": (
                        self._last_loss_time.isoformat()
                        if self._last_loss_time
                        else None
                    ),
                    "last_date": datetime.utcnow().strftime("%Y-%m-%d"),
                },
                f,
                indent=2,
            )

        with open(self.config.positions_file, "w") as f:
            json.dump(self._positions, f, indent=2)

    @property
    def bankroll(self) -> float:
        return self._bankroll

    def evaluate_trade(
        self, prediction: Prediction, market: Market
    ) -> dict:
        """
        Evaluate whether a trade should be placed and at what size.

        Returns:
            {
                "approved": bool,
                "amount": float,
                "reason": str,
                "risk_score": float,  # 0-1, higher = riskier
                "warnings": list[str],
            }
        """
        warnings = []
        block_reasons = []

        # ─── Check 1: Minimum bankroll ───
        if self._bankroll < self.risk_cfg.min_bankroll:
            return {
                "approved": False,
                "amount": 0.0,
                "reason": (
                    f"Bankroll too low: ${self._bankroll:.2f} < "
                    f"${self.risk_cfg.min_bankroll:.2f} minimum"
                ),
                "risk_score": 1.0,
                "warnings": ["CRITICAL: Bankroll depleted"],
            }

        # ─── Check 2: Daily loss limit ───
        max_daily_loss = self._bankroll * self.risk_cfg.max_daily_loss_pct
        if abs(self._daily_pnl) > max_daily_loss and self._daily_pnl < 0:
            return {
                "approved": False,
                "amount": 0.0,
                "reason": (
                    f"Daily loss limit hit: ${abs(self._daily_pnl):.2f} > "
                    f"${max_daily_loss:.2f} ({self.risk_cfg.max_daily_loss_pct:.0%})"
                ),
                "risk_score": 1.0,
                "warnings": ["Daily loss limit reached — no more trades today"],
            }

        # ─── Check 3: Open position limit ───
        open_count = len(self._positions)
        if open_count >= self.risk_cfg.max_open_positions:
            return {
                "approved": False,
                "amount": 0.0,
                "reason": (
                    f"Too many open positions: {open_count} >= "
                    f"{self.risk_cfg.max_open_positions}"
                ),
                "risk_score": 0.8,
                "warnings": ["Max open positions reached"],
            }

        # ─── Check 4: Loss cooldown ───
        if self._last_loss_time:
            cooldown_end = self._last_loss_time + timedelta(
                minutes=self.risk_cfg.cooldown_after_loss_minutes
            )
            if datetime.utcnow() < cooldown_end:
                remaining = (cooldown_end - datetime.utcnow()).seconds // 60
                return {
                    "approved": False,
                    "amount": 0.0,
                    "reason": (
                        f"Cooldown active: {remaining} minutes remaining "
                        f"after last loss"
                    ),
                    "risk_score": 0.6,
                    "warnings": ["Post-loss cooldown in effect"],
                }

        # ─── Check 5: Correlation with existing positions ───
        for pos in self._positions:
            if pos.get("category") == market.category and market.category:
                warnings.append(
                    f"Correlated position exists in category: {market.category}"
                )
                if len([
                    p for p in self._positions
                    if p.get("category") == market.category
                ]) >= 2:
                    block_reasons.append(
                        f"Too many correlated positions in {market.category}"
                    )

        # ─── Check 6: Spread too wide (risky entry) ───
        if market.spread > 8.0:
            block_reasons.append(
                f"Spread too wide for safe entry: {market.spread:.1f}%"
            )

        if block_reasons:
            return {
                "approved": False,
                "amount": 0.0,
                "reason": "; ".join(block_reasons),
                "risk_score": 0.9,
                "warnings": warnings,
            }

        # ─── Calculate bet size using Kelly Criterion ───
        amount = self._calculate_kelly_bet(prediction)

        # Cap at max_bet_pct of bankroll
        max_bet = self._bankroll * self.risk_cfg.max_bet_pct
        if amount > max_bet:
            warnings.append(
                f"Kelly suggests ${amount:.2f}, capped at ${max_bet:.2f} "
                f"({self.risk_cfg.max_bet_pct:.0%} of bankroll)"
            )
            amount = max_bet

        # Floor at $0.50 to avoid dust trades
        if amount < 0.50:
            return {
                "approved": False,
                "amount": 0.0,
                "reason": f"Bet size too small: ${amount:.2f}",
                "risk_score": 0.3,
                "warnings": warnings,
            }

        # ─── Risk score ───
        risk_factors = [
            min(amount / self._bankroll / 0.05, 1.0),  # size relative to bankroll
            min(market.spread / 10.0, 1.0),             # spread risk
            1.0 - prediction.confidence,                 # uncertainty
            min(open_count / self.risk_cfg.max_open_positions, 1.0),
        ]
        risk_score = sum(risk_factors) / len(risk_factors)

        return {
            "approved": True,
            "amount": round(amount, 2),
            "reason": (
                f"Trade approved: ${amount:.2f} on {prediction.recommended_side} "
                f"(edge={prediction.edge:.3f}, conf={prediction.confidence:.3f})"
            ),
            "risk_score": risk_score,
            "warnings": warnings,
        }

    def _calculate_kelly_bet(self, prediction: Prediction) -> float:
        """
        Calculate bet size using fractional Kelly Criterion.

        Kelly % = (bp - q) / b
        where b = odds, p = win probability, q = 1 - p

        We use quarter-Kelly for safety.
        """
        p = prediction.predicted_probability
        q = 1.0 - p

        if prediction.recommended_side == "Yes":
            entry_price = prediction.market_price
        else:
            entry_price = 1.0 - prediction.market_price

        if entry_price <= 0 or entry_price >= 1:
            return 0.0

        # Odds: if you pay entry_price, you win (1 - entry_price)
        b = (1.0 - entry_price) / entry_price

        kelly_pct = (b * p - q) / b if b > 0 else 0.0
        kelly_pct = max(0, kelly_pct)

        # Apply fractional Kelly
        bet_pct = kelly_pct * self.risk_cfg.kelly_fraction
        amount = self._bankroll * bet_pct

        return amount

    def record_trade(self, trade: Trade, market: Market):
        """Record a new open position."""
        self._positions.append(
            {
                "trade_id": trade.trade_id,
                "market_id": trade.market_id,
                "category": market.category,
                "side": trade.side,
                "amount": trade.amount,
                "entry_price": trade.entry_price,
                "placed_at": trade.placed_at,
            }
        )
        self._bankroll -= trade.amount
        self._save_state()
        logger.info(
            f"Recorded trade {trade.trade_id}: "
            f"${trade.amount:.2f} on {trade.side} | "
            f"Bankroll: ${self._bankroll:.2f}"
        )

    def resolve_trade(self, trade_id: str, pnl: float):
        """Resolve a trade and update bankroll."""
        self._positions = [
            p for p in self._positions if p.get("trade_id") != trade_id
        ]
        self._bankroll += pnl
        self._daily_pnl += pnl

        if pnl < 0:
            self._last_loss_time = datetime.utcnow()
            logger.warning(
                f"Trade {trade_id} LOSS: ${pnl:.2f} | "
                f"Bankroll: ${self._bankroll:.2f} | "
                f"Cooldown: {self.risk_cfg.cooldown_after_loss_minutes}min"
            )
        else:
            logger.info(
                f"Trade {trade_id} WIN: +${pnl:.2f} | "
                f"Bankroll: ${self._bankroll:.2f}"
            )

        self._save_state()

    def get_status(self) -> str:
        """Return current risk status summary."""
        max_daily = self._bankroll * self.risk_cfg.max_daily_loss_pct
        return (
            f"=== RISK STATUS ===\n"
            f"Bankroll: ${self._bankroll:.2f}\n"
            f"Open positions: {len(self._positions)}/{self.risk_cfg.max_open_positions}\n"
            f"Daily PnL: ${self._daily_pnl:+.2f} "
            f"(limit: ${max_daily:.2f})\n"
            f"Cooldown: {'ACTIVE' if self._last_loss_time and datetime.utcnow() < self._last_loss_time + timedelta(minutes=self.risk_cfg.cooldown_after_loss_minutes) else 'None'}\n"
        )
