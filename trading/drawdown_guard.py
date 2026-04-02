"""
Drawdown Guard — Real-time max drawdown protection.

Monitors equity and blocks trading when drawdown exceeds limits.
Prevents catastrophic losses by enforcing hard stops.
"""

import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STATE_FILE = "trading/data/drawdown_state.json"


@dataclass
class DrawdownStatus:
    """Current drawdown metrics."""
    peak_equity: float
    current_equity: float
    current_drawdown: float
    current_drawdown_pct: float
    max_drawdown_seen: float
    max_drawdown_pct_seen: float
    daily_pnl: float
    daily_loss_limit: float
    can_trade: bool
    reason: str


class DrawdownGuard:
    """
    Monitors equity curve and blocks trading on excessive drawdown.

    Two independent limits:
    1. Max total drawdown from peak (default 10%)
    2. Max daily loss (default 3%)
    """

    def __init__(
        self,
        max_drawdown_pct: float = 5.0,
        daily_loss_limit_pct: float = 2.0,
        state_file: str = STATE_FILE,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.state_file = state_file

        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.max_drawdown_seen: float = 0.0
        self.max_drawdown_pct_seen: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_start_balance: float = 0.0
        self.last_date: str = ""
        self.equity_history: list[float] = []

        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        try:
            with open(self.state_file) as f:
                data = json.load(f)
                self.peak_equity = data.get("peak_equity", 0)
                self.current_equity = data.get("current_equity", 0)
                self.max_drawdown_seen = data.get("max_drawdown_seen", 0)
                self.max_drawdown_pct_seen = data.get("max_drawdown_pct_seen", 0)
                self.daily_pnl = data.get("daily_pnl", 0)
                self.daily_start_balance = data.get("daily_start_balance", 0)
                self.last_date = data.get("last_date", "")
                self.equity_history = data.get("equity_history", [])[-500:]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "max_drawdown_seen": self.max_drawdown_seen,
                "max_drawdown_pct_seen": self.max_drawdown_pct_seen,
                "daily_pnl": self.daily_pnl,
                "daily_start_balance": self.daily_start_balance,
                "last_date": self.last_date,
                "equity_history": self.equity_history[-500:],
            }, f, indent=2)

    def reset_peak(self, current_balance: float):
        """
        Reset peak equity to current balance.

        Call this when starting fresh or when the old peak is stale
        (e.g., practice account balance was reset).
        """
        self.peak_equity = current_balance
        self.current_equity = current_balance
        self.max_drawdown_seen = 0.0
        self.max_drawdown_pct_seen = 0.0
        self.daily_pnl = 0.0
        self.daily_start_balance = current_balance
        self.last_date = datetime.utcnow().strftime("%Y-%m-%d")
        self._save_state()
        logger.info(f"DRAWDOWN GUARD: Peak reset to ${current_balance:,.2f}")

    def update(self, current_balance: float):
        """
        Update with current account balance. Call after every trade.

        Args:
            current_balance: Current account equity
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Reset daily tracking at midnight
        if today != self.last_date:
            self.daily_pnl = 0.0
            self.daily_start_balance = current_balance
            self.last_date = today

        # Track daily PnL
        if self.daily_start_balance > 0:
            self.daily_pnl = current_balance - self.daily_start_balance

        # Auto-reset peak if it's wildly stale (>50% above current balance)
        # This handles practice account resets and stale state files
        if self.peak_equity > 0 and current_balance > 0:
            stale_ratio = self.peak_equity / current_balance
            if stale_ratio > 2.0:  # Peak is 2x+ current = clearly stale
                logger.warning(
                    f"DRAWDOWN GUARD: Peak ${self.peak_equity:,.2f} is {stale_ratio:.1f}x "
                    f"current ${current_balance:,.2f} — auto-resetting peak"
                )
                self.peak_equity = current_balance
                self.max_drawdown_seen = 0.0
                self.max_drawdown_pct_seen = 0.0

        # Update peak and current
        self.current_equity = current_balance
        if current_balance > self.peak_equity:
            self.peak_equity = current_balance

        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = self.peak_equity - current_balance
            drawdown_pct = (drawdown / self.peak_equity) * 100

            if drawdown > self.max_drawdown_seen:
                self.max_drawdown_seen = drawdown
            if drawdown_pct > self.max_drawdown_pct_seen:
                self.max_drawdown_pct_seen = drawdown_pct

        self.equity_history.append(current_balance)
        self._save_state()

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on drawdown limits.

        Returns:
            (can_trade: bool, reason: str)
        """
        if self.peak_equity <= 0:
            return True, "No equity data yet"

        # Check total drawdown
        drawdown = self.peak_equity - self.current_equity
        drawdown_pct = (drawdown / self.peak_equity) * 100

        if drawdown_pct >= self.max_drawdown_pct:
            return False, (
                f"MAX DRAWDOWN BREACHED: {drawdown_pct:.1f}% "
                f"(limit: {self.max_drawdown_pct}%) | "
                f"Peak: ${self.peak_equity:.2f} → Now: ${self.current_equity:.2f}"
            )

        # Check daily loss
        if self.daily_start_balance > 0 and self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance * 100
            if daily_loss_pct >= self.daily_loss_limit_pct:
                return False, (
                    f"DAILY LOSS LIMIT: -{daily_loss_pct:.1f}% "
                    f"(limit: {self.daily_loss_limit_pct}%) | "
                    f"Lost ${abs(self.daily_pnl):.2f} today"
                )

        return True, "OK"

    def get_status(self) -> str:
        """Human-readable status."""
        if self.peak_equity <= 0:
            return "Drawdown Guard: No data yet"

        drawdown = self.peak_equity - self.current_equity
        drawdown_pct = (drawdown / self.peak_equity) * 100
        allowed, reason = self.can_trade()

        return (
            f"=== DRAWDOWN GUARD ===\n"
            f"Peak Equity:     ${self.peak_equity:,.2f}\n"
            f"Current Equity:  ${self.current_equity:,.2f}\n"
            f"Current DD:      {drawdown_pct:.2f}% (${drawdown:,.2f})\n"
            f"Max DD Seen:     {self.max_drawdown_pct_seen:.2f}%\n"
            f"Daily PnL:       ${self.daily_pnl:+,.2f}\n"
            f"DD Limit:        {self.max_drawdown_pct}%\n"
            f"Daily Limit:     {self.daily_loss_limit_pct}%\n"
            f"Trading:         {'ALLOWED' if allowed else 'BLOCKED'}\n"
            f"{'Reason: ' + reason if not allowed else ''}"
        )

    def get_drawdown_info(self) -> DrawdownStatus:
        """Get structured drawdown data."""
        drawdown = self.peak_equity - self.current_equity if self.peak_equity > 0 else 0
        drawdown_pct = (drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0
        allowed, reason = self.can_trade()
        daily_limit = self.daily_start_balance * self.daily_loss_limit_pct / 100 if self.daily_start_balance > 0 else 0

        return DrawdownStatus(
            peak_equity=self.peak_equity,
            current_equity=self.current_equity,
            current_drawdown=drawdown,
            current_drawdown_pct=drawdown_pct,
            max_drawdown_seen=self.max_drawdown_seen,
            max_drawdown_pct_seen=self.max_drawdown_pct_seen,
            daily_pnl=self.daily_pnl,
            daily_loss_limit=daily_limit,
            can_trade=allowed,
            reason=reason,
        )
