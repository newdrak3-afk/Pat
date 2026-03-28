"""
Drift Detector — Monitors if strategy performance is degrading.

Tracks rolling win rate, losing streaks, and PnL trends.
Signals when to pause trading for review.
"""

import json
import logging
import os
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

STATE_FILE = "trading/data/drift_state.json"


class DriftDetector:
    """
    Detects strategy performance degradation in real-time.

    Monitors:
    - Rolling win rate (over last N trades)
    - Consecutive losing streaks
    - PnL trend (is it getting worse?)
    - Time-weighted recent performance
    """

    def __init__(
        self,
        window_size: int = 20,
        min_win_rate: float = 0.35,
        max_losing_streak: int = 5,
        state_file: str = STATE_FILE,
    ):
        self.window_size = window_size
        self.min_win_rate = min_win_rate
        self.max_losing_streak = max_losing_streak
        self.state_file = state_file

        # Track results
        self.results: list[dict] = []
        self.current_streak: int = 0  # negative = losing, positive = winning
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.pnls: list[float] = []

        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        try:
            with open(self.state_file) as f:
                data = json.load(f)
                self.results = data.get("results", [])[-200:]
                self.current_streak = data.get("current_streak", 0)
                self.total_wins = data.get("total_wins", 0)
                self.total_losses = data.get("total_losses", 0)
                self.pnls = data.get("pnls", [])[-200:]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "results": self.results[-200:],
                "current_streak": self.current_streak,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "pnls": self.pnls[-200:],
            }, f, indent=2)

    def add_result(self, outcome: str, pnl: float = 0.0):
        """
        Record a trade result.

        Args:
            outcome: "win" or "loss"
            pnl: Profit/loss amount
        """
        is_win = outcome.lower() == "win"

        self.results.append({
            "outcome": outcome,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat(),
        })

        self.pnls.append(pnl)

        if is_win:
            self.total_wins += 1
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.total_losses += 1
            self.current_streak = min(0, self.current_streak) - 1

        self._save_state()

    def is_drifting(self) -> tuple[bool, str]:
        """
        Check if strategy performance is degrading.

        Returns:
            (is_drifting: bool, reason: str)
        """
        reasons = []

        # Need minimum data
        if len(self.results) < 10:
            return False, "Not enough data yet"

        # Check 1: Rolling win rate
        recent = self.results[-self.window_size:]
        recent_wins = sum(1 for r in recent if r["outcome"] == "win")
        rolling_wr = recent_wins / len(recent)

        if rolling_wr < self.min_win_rate:
            reasons.append(
                f"Rolling win rate LOW: {rolling_wr:.0%} "
                f"(threshold: {self.min_win_rate:.0%}, "
                f"window: last {len(recent)} trades)"
            )

        # Check 2: Losing streak
        if self.current_streak <= -self.max_losing_streak:
            reasons.append(
                f"LOSING STREAK: {abs(self.current_streak)} consecutive losses "
                f"(max allowed: {self.max_losing_streak})"
            )

        # Check 3: PnL trend declining
        if len(self.pnls) >= 10:
            first_half = self.pnls[-10:-5]
            second_half = self.pnls[-5:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second < avg_first and avg_second < 0:
                reasons.append(
                    f"PnL trend DECLINING: "
                    f"recent avg ${avg_second:.2f} vs prior ${avg_first:.2f}"
                )

        # Check 4: Recent performance much worse than overall
        if len(self.results) >= 20:
            overall_wr = self.total_wins / (self.total_wins + self.total_losses) if (self.total_wins + self.total_losses) > 0 else 0.5
            if rolling_wr < overall_wr * 0.6:  # 40%+ degradation
                reasons.append(
                    f"Performance DEGRADED: recent {rolling_wr:.0%} vs "
                    f"overall {overall_wr:.0%} ({((rolling_wr/overall_wr)-1)*100:.0f}%)"
                )

        if reasons:
            return True, " | ".join(reasons)
        return False, "Performance within normal range"

    def should_pause(self) -> bool:
        """
        Should we pause trading? True if multiple drift signals.
        """
        drifting, reason = self.is_drifting()
        if not drifting:
            return False

        # Count how many signals
        signal_count = reason.count("|") + 1
        return signal_count >= 2

    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        total = self.total_wins + self.total_losses
        overall_wr = self.total_wins / total if total > 0 else 0

        recent = self.results[-self.window_size:]
        recent_wins = sum(1 for r in recent if r["outcome"] == "win")
        rolling_wr = recent_wins / len(recent) if recent else 0

        recent_pnl = sum(self.pnls[-self.window_size:]) if self.pnls else 0

        drifting, reason = self.is_drifting()

        return {
            "total_trades": total,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "overall_win_rate": overall_wr,
            "rolling_win_rate": rolling_wr,
            "current_streak": self.current_streak,
            "recent_pnl": recent_pnl,
            "is_drifting": drifting,
            "drift_reason": reason,
            "should_pause": self.should_pause(),
        }

    def get_status(self) -> str:
        """Human-readable status string."""
        m = self.get_metrics()
        streak_str = (
            f"+{m['current_streak']} wins"
            if m["current_streak"] > 0
            else f"{m['current_streak']} losses"
            if m["current_streak"] < 0
            else "0"
        )

        return (
            f"=== DRIFT DETECTOR ===\n"
            f"Total Trades:    {m['total_trades']}\n"
            f"Overall WR:      {m['overall_win_rate']:.0%}\n"
            f"Rolling WR:      {m['rolling_win_rate']:.0%} (last {self.window_size})\n"
            f"Current Streak:  {streak_str}\n"
            f"Recent PnL:      ${m['recent_pnl']:+,.2f}\n"
            f"Drifting:        {'YES' if m['is_drifting'] else 'No'}\n"
            f"Should Pause:    {'YES' if m['should_pause'] else 'No'}\n"
            f"{'Reason: ' + m['drift_reason'] if m['is_drifting'] else ''}"
        )
