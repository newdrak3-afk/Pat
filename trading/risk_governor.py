"""
Risk Governor — Centralized risk budget system.

Single point of authority that both forex and options traders
must consult before opening any position. Tracks drawdown,
loss streaks, cluster exposure, and daily trade limits.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Currency clusters — any pair containing the currency belongs to its cluster
# ---------------------------------------------------------------------------
CURRENCY_CLUSTERS = {
    "USD": "USD_CLUSTER",
    "EUR": "EUR_CLUSTER",
    "GBP": "GBP_CLUSTER",
    "JPY": "JPY_CLUSTER",
    "AUD": "AUD_CLUSTER",
    "CAD": "CAD_CLUSTER",
    "CHF": "CHF_CLUSTER",
    "NZD": "NZD_CLUSTER",
}

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_CONSECUTIVE_LOSSES = 3
LOSS_PAUSE_HOURS = 1
MAX_DAILY_TRADES = 5
DRAWDOWN_REDUCE_PCT = 5.0      # reduce size by 50% at this drawdown
DRAWDOWN_STOP_PCT = 8.0        # stop opening positions at this drawdown
MAX_CLUSTER_POSITIONS = 3
DEFAULT_RISK_PER_TRADE_PCT = 1.0  # 1% of equity per trade


@dataclass
class BudgetResponse:
    """Returned by request_budget()."""
    allowed: bool
    max_risk_usd: float
    reason: str


@dataclass
class TradeResult:
    """Recorded by add_result()."""
    outcome: str          # "win" or "loss"
    pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskGovernor:
    """
    Centralized risk budget that forex and options traders request from.

    Usage:
        gov = RiskGovernor(account_equity=10_000.0)
        resp = gov.request_budget("EUR/USD", "long", "forex")
        if resp.allowed:
            # open position risking up to resp.max_risk_usd
            ...
        gov.add_result("win", pnl=45.0)
    """

    def __init__(
        self,
        account_equity: float = 10_000.0,
        peak_equity: Optional[float] = None,
        risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    ):
        # Equity tracking
        self.account_equity = account_equity
        self.peak_equity = peak_equity or account_equity

        # Configurable base risk
        self.risk_per_trade_pct = risk_per_trade_pct

        # Open positions: list of {"symbol": str, "side": str, "asset_class": str}
        self.open_positions: list[dict] = []

        # Daily counters (call reset_daily() at start of each day)
        self.daily_trade_count: int = 0

        # Loss streak
        self.consecutive_losses: int = 0
        self.last_loss_time: Optional[datetime] = None

        # Full result history for the session
        self.results: list[TradeResult] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def request_budget(
        self, symbol: str, side: str, asset_class: str
    ) -> BudgetResponse:
        """
        Ask the governor whether a new position is allowed.

        Args:
            symbol:      e.g. "EUR/USD", "SPY_250418_C_500"
            side:        "long" or "short"
            asset_class: "forex" or "options"

        Returns:
            BudgetResponse with allowed flag, max risk in USD, and reason.
        """
        symbol_upper = symbol.upper()

        # --- Circuit breaker: loss streak pause ---
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if self.last_loss_time is not None:
                resume_at = self.last_loss_time + timedelta(hours=LOSS_PAUSE_HOURS)
                if datetime.utcnow() < resume_at:
                    mins_left = (resume_at - datetime.utcnow()).seconds // 60
                    return BudgetResponse(
                        allowed=False,
                        max_risk_usd=0.0,
                        reason=(
                            f"Paused after {self.consecutive_losses} consecutive "
                            f"losses. Resume in ~{mins_left} min."
                        ),
                    )
                # Pause period elapsed — allow trading but keep streak count
                logger.info("Loss-streak pause elapsed, resuming trading.")

        # --- Circuit breaker: daily trade limit ---
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return BudgetResponse(
                allowed=False,
                max_risk_usd=0.0,
                reason=(
                    f"Daily trade limit reached ({MAX_DAILY_TRADES}). "
                    "Done for today."
                ),
            )

        # --- Circuit breaker: hard drawdown stop ---
        dd_pct = self._drawdown_pct()
        if dd_pct >= DRAWDOWN_STOP_PCT:
            return BudgetResponse(
                allowed=False,
                max_risk_usd=0.0,
                reason=(
                    f"Drawdown {dd_pct:.1f}% hit {DRAWDOWN_STOP_PCT}% limit. "
                    "No new positions until equity recovers."
                ),
            )

        # --- Cluster exposure check ---
        clusters = self._get_clusters(symbol_upper)
        for cluster_name in clusters:
            count = self._cluster_count(cluster_name)
            if count >= MAX_CLUSTER_POSITIONS:
                return BudgetResponse(
                    allowed=False,
                    max_risk_usd=0.0,
                    reason=(
                        f"Cluster {cluster_name} already has {count} open "
                        f"positions (max {MAX_CLUSTER_POSITIONS})."
                    ),
                )

        # --- Calculate allowed risk ---
        base_risk_usd = self.account_equity * (self.risk_per_trade_pct / 100.0)

        # Halve risk if in drawdown reduction zone
        if dd_pct >= DRAWDOWN_REDUCE_PCT:
            base_risk_usd *= 0.5
            reason = (
                f"Allowed (reduced 50% — drawdown {dd_pct:.1f}% "
                f">= {DRAWDOWN_REDUCE_PCT}%)"
            )
        else:
            reason = "Allowed"

        logger.info(
            "Budget approved: %s %s %s — max_risk $%.2f (%s)",
            side, symbol, asset_class, base_risk_usd, reason,
        )

        return BudgetResponse(
            allowed=True,
            max_risk_usd=round(base_risk_usd, 2),
            reason=reason,
        )

    def add_result(self, outcome: str, pnl: float) -> None:
        """
        Record a trade result. Call after every closed trade.

        Args:
            outcome: "win" or "loss"
            pnl:     profit/loss in USD (negative for losses)
        """
        now = datetime.utcnow()
        self.results.append(TradeResult(outcome=outcome, pnl=pnl, timestamp=now))

        # Update equity
        self.account_equity += pnl
        if self.account_equity > self.peak_equity:
            self.peak_equity = self.account_equity

        # Update loss streak
        if outcome == "loss":
            self.consecutive_losses += 1
            self.last_loss_time = now
            logger.warning(
                "Loss recorded (PnL $%.2f). Consecutive losses: %d",
                pnl, self.consecutive_losses,
            )
        else:
            if self.consecutive_losses > 0:
                logger.info(
                    "Win breaks %d-loss streak (PnL $%.2f).",
                    self.consecutive_losses, pnl,
                )
            self.consecutive_losses = 0

    def add_position(self, symbol: str, side: str, asset_class: str) -> None:
        """Register a newly opened position."""
        self.open_positions.append(
            {"symbol": symbol.upper(), "side": side, "asset_class": asset_class}
        )
        self.daily_trade_count += 1

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position by symbol."""
        symbol_upper = symbol.upper()
        self.open_positions = [
            p for p in self.open_positions if p["symbol"] != symbol_upper
        ]

    def reset_daily(self) -> None:
        """Reset daily counters. Call at the start of each trading day."""
        self.daily_trade_count = 0
        logger.info("Daily risk counters reset.")

    # ------------------------------------------------------------------
    # Status / display
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        """Human-readable status string suitable for Telegram."""
        dd_pct = self._drawdown_pct()
        total_pnl = sum(r.pnl for r in self.results)
        wins = sum(1 for r in self.results if r.outcome == "win")
        losses = sum(1 for r in self.results if r.outcome == "loss")

        lines = [
            "--- Risk Governor ---",
            f"Equity:       ${self.account_equity:,.2f}",
            f"Peak:         ${self.peak_equity:,.2f}",
            f"Drawdown:     {dd_pct:.1f}%",
            f"Open pos:     {len(self.open_positions)}",
            f"Daily trades: {self.daily_trade_count}/{MAX_DAILY_TRADES}",
            f"Loss streak:  {self.consecutive_losses}",
            f"Session W/L:  {wins}/{losses}",
            f"Session PnL:  ${total_pnl:+,.2f}",
        ]

        # Cluster summary
        cluster_counts = self._all_cluster_counts()
        if cluster_counts:
            lines.append("Clusters:")
            for name, count in sorted(cluster_counts.items()):
                lines.append(f"  {name}: {count}/{MAX_CLUSTER_POSITIONS}")

        # Active circuit breakers
        breakers = []
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if self.last_loss_time:
                resume_at = self.last_loss_time + timedelta(hours=LOSS_PAUSE_HOURS)
                if datetime.utcnow() < resume_at:
                    breakers.append("LOSS STREAK PAUSE")
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            breakers.append("DAILY LIMIT HIT")
        if dd_pct >= DRAWDOWN_STOP_PCT:
            breakers.append("DRAWDOWN STOP")
        elif dd_pct >= DRAWDOWN_REDUCE_PCT:
            breakers.append("DRAWDOWN REDUCED SIZE")

        if breakers:
            lines.append(f"ACTIVE: {', '.join(breakers)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drawdown_pct(self) -> float:
        """Current drawdown from peak as a percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - self.account_equity) / self.peak_equity) * 100.0

    def _get_clusters(self, symbol: str) -> list[str]:
        """Return cluster names that a symbol belongs to."""
        clusters = []
        for currency, cluster_name in CURRENCY_CLUSTERS.items():
            if currency in symbol:
                clusters.append(cluster_name)
        return clusters

    def _cluster_count(self, cluster_name: str) -> int:
        """Count open positions in a given cluster."""
        count = 0
        for pos in self.open_positions:
            if cluster_name in self._get_clusters(pos["symbol"]):
                count += 1
        return count

    def _all_cluster_counts(self) -> dict[str, int]:
        """Return {cluster_name: count} for all clusters with open positions."""
        counts: dict[str, int] = {}
        for pos in self.open_positions:
            for cluster in self._get_clusters(pos["symbol"]):
                counts[cluster] = counts.get(cluster, 0) + 1
        return counts
