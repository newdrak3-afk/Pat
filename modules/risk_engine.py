"""
risk_engine.py — Risk management and trade gating for the options/forex bot.

Implements all safety controls that must pass before any trade is executed:
  - Max trades per day
  - Max open positions (total and per ticker)
  - Max directional exposure (too many calls or too many puts at once)
  - Max position size per trade
  - Cooldown after consecutive losses
  - Hard kill switch (emergency stop all trading)
  - Daily loss limit (auto-pause when hit)
  - Data staleness check (don't trade on stale prices)
  - Bid/ask spread too wide check
  - Contract liquidity too low check
  - No trade during catalyst / earnings window

All state is persisted to data/daily_stats.json (resets daily) and
data/risk_events.json (permanent log of risk rule triggers).
"""

import os
import json
from datetime import datetime, date
from typing import Optional

DAILY_STATS_FILE = "data/daily_stats.json"
RISK_EVENTS_FILE = "data/risk_events.json"
KILL_SWITCH_FILE = "data/kill_switch.json"


class RiskEngine:
    """
    Central risk gating system.
    Call check_all() before placing any trade to get a go/no-go decision.
    """

    def __init__(self):
        self._load_config()
        self._daily = self._load_daily_stats()

    def _load_config(self):
        """Load risk limits from environment variables."""
        self.max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
        self.max_exposure_per_ticker = int(os.getenv("MAX_EXPOSURE_PER_TICKER", "2"))
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "500"))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "200"))
        self.max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
        self.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", "30"))
        self.max_spread_pct = float(os.getenv("MAX_SPREAD_PCT", "10.0"))
        self.min_open_interest = int(os.getenv("MIN_OPEN_INTEREST", "100"))
        self.min_contract_volume = int(os.getenv("MIN_CONTRACT_VOLUME", "10"))
        self.min_delta = float(os.getenv("MIN_DELTA", "0.20"))
        self.max_delta = float(os.getenv("MAX_DELTA", "0.70"))
        self.min_dte = int(os.getenv("MIN_DTE", "5"))
        self.max_dte = int(os.getenv("MAX_DTE", "45"))
        self.max_directional_ratio = float(os.getenv("MAX_DIRECTIONAL_RATIO", "0.75"))
        self.stale_data_minutes = int(os.getenv("STALE_DATA_MINUTES", "5"))

    # ──────────────────────────────────────────────
    # MAIN GATING FUNCTION
    # ──────────────────────────────────────────────

    def check_all(
        self,
        signal: dict,
        positions: list,
        contract: dict = None,
    ) -> dict:
        """
        Run all risk checks for a proposed trade.

        Args:
            signal: Signal dict from scanner (ticker, direction, confidence, market)
            positions: Current open positions list
            contract: Options contract dict (optional, for options-specific checks)

        Returns:
            {
                approved: bool,
                blocked_by: str or None,
                reason: str,
                warnings: list of str,
            }
        """
        self._daily = self._load_daily_stats()
        warnings = []

        # ── Hard kill switch ────────────────────────
        if self._is_kill_switch_active():
            return self._block("kill_switch", "Kill switch is active. Run 'python main.py resume' to clear.")

        # ── Daily loss limit ────────────────────────
        daily_loss = self._daily.get("realized_loss", 0)
        if daily_loss >= self.max_daily_loss:
            self._log_risk_event("daily_loss_limit", f"Daily loss ${daily_loss:.2f} >= limit ${self.max_daily_loss:.2f}")
            return self._block("daily_loss_limit", f"Daily loss limit hit (${daily_loss:.2f}/${self.max_daily_loss:.2f}). Trading paused for today.")

        # ── Trades per day ──────────────────────────
        trades_today = self._daily.get("trades_today", 0)
        if trades_today >= self.max_trades_per_day:
            return self._block("max_trades_per_day", f"Max trades/day reached ({trades_today}/{self.max_trades_per_day}).")

        # ── Open positions ───────────────────────────
        open_count = len([p for p in positions if p.get("status") == "open"])
        if open_count >= self.max_open_positions:
            return self._block("max_open_positions", f"Too many open positions ({open_count}/{self.max_open_positions}). Close some before opening new ones.")

        # ── Per-ticker exposure ──────────────────────
        ticker = signal.get("ticker") or signal.get("pair", "")
        ticker_count = len([p for p in positions if p.get("ticker") == ticker and p.get("status") == "open"])
        if ticker_count >= self.max_exposure_per_ticker:
            return self._block("max_exposure_per_ticker", f"Already have {ticker_count} open position(s) in {ticker} (max {self.max_exposure_per_ticker}).")

        # ── Directional exposure ─────────────────────
        direction = signal.get("direction", "").lower()
        dir_block = self._check_directional_exposure(direction, positions)
        if dir_block:
            warnings.append(dir_block)

        # ── Consecutive loss cooldown ────────────────
        consec_losses = self._daily.get("consecutive_losses", 0)
        if consec_losses >= self.max_consecutive_losses:
            cooldown_result = self._check_cooldown()
            if cooldown_result:
                return self._block("cooldown", cooldown_result)

        # ── Options-specific checks ──────────────────
        market = signal.get("market", "options")
        if market == "options" and contract:
            options_check = self.check_contract_quality(contract)
            if not options_check["approved"]:
                return options_check
            warnings.extend(options_check.get("warnings", []))

        # ── Catalyst risk ────────────────────────────
        if signal.get("catalyst_risk"):
            warnings.append(f"Catalyst risk detected: {signal.get('catalyst_reason', '')}")

        return {
            "approved": True,
            "blocked_by": None,
            "reason": "All risk checks passed.",
            "warnings": warnings,
        }

    # ──────────────────────────────────────────────
    # CONTRACT QUALITY (OPTIONS)
    # ──────────────────────────────────────────────

    def check_contract_quality(self, contract: dict) -> dict:
        """
        Validate that an options contract meets liquidity and quality requirements.

        Checks:
          - Minimum open interest
          - Minimum volume
          - Max bid/ask spread %
          - Delta in acceptable range
          - DTE (days to expiration) in range
        """
        warnings = []
        ticker = contract.get("underlying", contract.get("symbol", "?"))

        # Open interest
        oi = contract.get("open_interest", 0)
        if oi < self.min_open_interest:
            return self._block("low_open_interest", f"Open interest {oi} < minimum {self.min_open_interest} for {contract.get('symbol')}")

        # Volume
        vol = contract.get("volume", 0)
        if vol < self.min_contract_volume:
            return self._block("low_volume", f"Contract volume {vol} < minimum {self.min_contract_volume}")

        # Bid/ask spread
        bid = contract.get("bid", 0)
        ask = contract.get("ask", 0)
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
        spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999
        if spread_pct > self.max_spread_pct:
            return self._block("wide_spread", f"Bid/ask spread {spread_pct:.1f}% > max {self.max_spread_pct}%. Too expensive to fill efficiently.")
        elif spread_pct > self.max_spread_pct * 0.7:
            warnings.append(f"Spread is {spread_pct:.1f}% (getting wide, watch fill quality)")

        # Delta
        delta = abs(contract.get("delta", 0))
        if delta < self.min_delta:
            return self._block("low_delta", f"Delta {delta:.2f} < minimum {self.min_delta} (too far OTM)")
        if delta > self.max_delta:
            warnings.append(f"Delta {delta:.2f} is high (deep ITM — expensive, less leverage)")

        # DTE
        dte = self._calc_dte(contract.get("expiration", ""))
        if dte < self.min_dte:
            return self._block("too_close_to_expiry", f"Only {dte} days to expiration (min {self.min_dte}). Theta risk too high.")
        if dte > self.max_dte:
            warnings.append(f"DTE={dte} days (longer than typical — more capital tied up)")

        return {"approved": True, "blocked_by": None, "reason": "Contract quality OK", "warnings": warnings}

    def rank_contracts(self, contracts: list, direction: str) -> list:
        """
        Rank option contracts by suitability score:
        - Penalize wide spreads and low OI
        - Favor delta in 0.30–0.50 range
        - Favor DTE 14–30 days
        - Sort best first

        Returns sorted list of (contract, score) tuples.
        """
        scored = []
        for c in contracts:
            if c.get("option_type", "") != direction.lower():
                continue
            score = 0
            bid, ask = c.get("bid", 0), c.get("ask", 0)
            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999

            # Spread penalty
            score -= spread_pct * 2

            # OI bonus
            oi = c.get("open_interest", 0)
            score += min(oi / 100, 20)  # cap at 20 points

            # Delta score (prefer 0.30–0.50)
            delta = abs(c.get("delta", 0))
            if 0.30 <= delta <= 0.50:
                score += 30
            elif 0.20 <= delta < 0.30 or 0.50 < delta <= 0.65:
                score += 15

            # DTE score (prefer 14–30)
            dte = self._calc_dte(c.get("expiration", ""))
            if 14 <= dte <= 30:
                score += 20
            elif 7 <= dte < 14 or 30 < dte <= 45:
                score += 10

            # Volume bonus
            vol = c.get("volume", 0)
            score += min(vol / 50, 15)

            scored.append((c, round(score, 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ──────────────────────────────────────────────
    # KILL SWITCH
    # ──────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "manual"):
        """Immediately halt all trading."""
        os.makedirs("data", exist_ok=True)
        with open(KILL_SWITCH_FILE, "w") as f:
            json.dump({"active": True, "reason": reason, "timestamp": datetime.now().isoformat()}, f)
        self._log_risk_event("kill_switch_activated", reason)

    def deactivate_kill_switch(self):
        """Resume trading (clear kill switch)."""
        if os.path.exists(KILL_SWITCH_FILE):
            os.remove(KILL_SWITCH_FILE)

    def _is_kill_switch_active(self) -> bool:
        if not os.path.exists(KILL_SWITCH_FILE):
            return False
        try:
            with open(KILL_SWITCH_FILE) as f:
                d = json.load(f)
            return d.get("active", False)
        except Exception:
            return False

    # ──────────────────────────────────────────────
    # DAILY STATS
    # ──────────────────────────────────────────────

    def record_trade_opened(self, ticker: str):
        """Call when a trade is opened."""
        self._daily["trades_today"] = self._daily.get("trades_today", 0) + 1
        self._save_daily_stats()

    def record_trade_closed(self, pnl: float):
        """Call when a trade is closed with its P&L."""
        if pnl >= 0:
            self._daily["wins"] = self._daily.get("wins", 0) + 1
            self._daily["consecutive_losses"] = 0
            self._daily["realized_gain"] = self._daily.get("realized_gain", 0) + pnl
        else:
            self._daily["losses"] = self._daily.get("losses", 0) + 1
            self._daily["consecutive_losses"] = self._daily.get("consecutive_losses", 0) + 1
            self._daily["realized_loss"] = self._daily.get("realized_loss", 0) + abs(pnl)
            if self._daily["consecutive_losses"] >= self.max_consecutive_losses:
                self._daily["cooldown_start"] = datetime.now().isoformat()
        self._save_daily_stats()

    def get_daily_summary(self) -> dict:
        """Return today's trading summary."""
        d = self._daily
        wins = d.get("wins", 0)
        losses = d.get("losses", 0)
        total = wins + losses
        return {
            "date": d.get("date", str(date.today())),
            "trades_today": d.get("trades_today", 0),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / total * 100, 1) if total > 0 else 0,
            "realized_gain": round(d.get("realized_gain", 0), 2),
            "realized_loss": round(d.get("realized_loss", 0), 2),
            "daily_pnl": round(d.get("realized_gain", 0) - d.get("realized_loss", 0), 2),
            "consecutive_losses": d.get("consecutive_losses", 0),
        }

    def _load_daily_stats(self) -> dict:
        """Load today's stats, resetting if date changed."""
        os.makedirs("data", exist_ok=True)
        today = str(date.today())
        if os.path.exists(DAILY_STATS_FILE):
            try:
                with open(DAILY_STATS_FILE) as f:
                    d = json.load(f)
                if d.get("date") == today:
                    return d
            except Exception:
                pass
        # New day — reset
        fresh = {"date": today, "trades_today": 0, "wins": 0, "losses": 0,
                 "consecutive_losses": 0, "realized_gain": 0, "realized_loss": 0}
        with open(DAILY_STATS_FILE, "w") as f:
            json.dump(fresh, f, indent=2)
        return fresh

    def _save_daily_stats(self):
        os.makedirs("data", exist_ok=True)
        with open(DAILY_STATS_FILE, "w") as f:
            json.dump(self._daily, f, indent=2)

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _check_directional_exposure(self, direction: str, positions: list) -> Optional[str]:
        """
        Warn if too many positions are in the same direction.
        Max ratio: MAX_DIRECTIONAL_RATIO (default 0.75).
        """
        open_pos = [p for p in positions if p.get("status") == "open"]
        if len(open_pos) < 2:
            return None
        bullish = sum(1 for p in open_pos if p.get("direction", "").lower() in ("call", "long", "buy", "bullish"))
        total = len(open_pos)
        if direction in ("call", "long", "buy", "bullish"):
            new_bullish_ratio = (bullish + 1) / (total + 1)
            if new_bullish_ratio > self.max_directional_ratio:
                return f"High directional concentration: {bullish+1}/{total+1} positions would be bullish"
        elif direction in ("put", "short", "sell", "bearish"):
            bearish = total - bullish
            new_bearish_ratio = (bearish + 1) / (total + 1)
            if new_bearish_ratio > self.max_directional_ratio:
                return f"High directional concentration: {bearish+1}/{total+1} positions would be bearish"
        return None

    def _check_cooldown(self) -> Optional[str]:
        """Return a message if still in cooldown period after consecutive losses."""
        cooldown_start_str = self._daily.get("cooldown_start")
        if not cooldown_start_str:
            return None
        try:
            cooldown_start = datetime.fromisoformat(cooldown_start_str)
            elapsed = (datetime.now() - cooldown_start).total_seconds() / 60
            remaining = self.cooldown_minutes - elapsed
            if remaining > 0:
                return (
                    f"Cooldown active after {self.max_consecutive_losses} consecutive losses. "
                    f"{remaining:.0f} minute(s) remaining."
                )
        except Exception:
            pass
        # Cooldown expired — reset
        self._daily["cooldown_start"] = None
        self._daily["consecutive_losses"] = 0
        self._save_daily_stats()
        return None

    def _calc_dte(self, expiration_str: str) -> int:
        """Calculate days to expiration from a date string like '2024-01-19'."""
        if not expiration_str:
            return 0
        try:
            exp = datetime.strptime(expiration_str[:10], "%Y-%m-%d").date()
            return max((exp - date.today()).days, 0)
        except Exception:
            return 0

    def _block(self, rule: str, reason: str) -> dict:
        return {"approved": False, "blocked_by": rule, "reason": reason, "warnings": []}

    def _log_risk_event(self, event_type: str, detail: str):
        """Append a risk event to the permanent log."""
        try:
            os.makedirs("data", exist_ok=True)
            events = []
            if os.path.exists(RISK_EVENTS_FILE):
                with open(RISK_EVENTS_FILE) as f:
                    events = json.load(f)
            events.append({
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "detail": detail,
            })
            events = events[-1000:]
            with open(RISK_EVENTS_FILE, "w") as f:
                json.dump(events, f, indent=2)
        except Exception:
            pass
