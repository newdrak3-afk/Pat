"""
Guard Engine — Centralized trade approval system.

All guards run through ONE pipeline and return ONE verdict.
No more scattered if/continue blocks in auto_trader.

Returns a TradeApproval with:
- approved: bool
- reasons: list of why blocked/approved
- adjusted_confidence: after calibration
- allowed_units: after slippage/sizing
- guard_results: which guards passed/failed
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeApproval:
    """Single verdict from all guards."""
    approved: bool = False
    reasons: list[str] = field(default_factory=list)
    adjusted_confidence: float = 0.0
    allowed_units: int = 0
    guard_results: dict = field(default_factory=dict)
    signal: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "BLOCKED"
        reason_str = " | ".join(self.reasons) if self.reasons else "All clear"
        guards_passed = sum(1 for v in self.guard_results.values() if v)
        guards_total = len(self.guard_results)
        return (
            f"{status} — {reason_str}\n"
            f"Guards: {guards_passed}/{guards_total} passed\n"
            f"Confidence: {self.adjusted_confidence:.0%}\n"
            f"Units: {self.allowed_units}"
        )


class GuardEngine:
    """
    Runs all guards in order and returns a single TradeApproval.

    Guard pipeline:
    1. Max open positions check
    2. Data quality (if enabled)
    3. Regime detection (if enabled)
    4. Portfolio exposure (if enabled)
    5. Calibration adjustment (if enabled)
    6. Slippage adjustment (if enabled)
    7. Position sizing

    Each guard can BLOCK or ADJUST — never both scattered in auto_trader.
    """

    def __init__(
        self,
        settings=None,
        oanda=None,
        data_quality=None,
        regime=None,
        portfolio=None,
        calibration=None,
        slippage=None,
        drawdown=None,
        drift=None,
        db=None,
    ):
        self.settings = settings
        self.oanda = oanda
        self.data_quality = data_quality
        self.regime = regime
        self.portfolio = portfolio
        self.calibration = calibration
        self.slippage = slippage
        self.drawdown = drawdown
        self.drift = drift
        self.db = db

    def evaluate(
        self,
        signal: dict,
        balance: float,
        open_trade_count: int,
        max_open: int = 5,
    ) -> TradeApproval:
        """
        Run all guards on a signal and return one approval verdict.

        Args:
            signal: dict with symbol, side, confidence, entry, stop_loss, take_profit, etc
            balance: current account balance
            open_trade_count: number of currently open trades
            max_open: maximum allowed open positions
        """
        approval = TradeApproval(
            approved=True,
            adjusted_confidence=signal.get("confidence", 0),
            signal=signal,
        )

        symbol = signal.get("symbol", "")
        toggles = self.settings.toggles if self.settings else None

        # ── 1. Max open positions ──
        if open_trade_count >= max_open:
            approval.approved = False
            approval.reasons.append(f"Max {max_open} open positions")
            approval.guard_results["max_positions"] = False
            self._log_verdict(signal, approved=False, reason="Max open positions")
            return approval
        approval.guard_results["max_positions"] = True

        # ── 2. Drawdown guard ──
        if toggles and toggles.drawdown_guard_enabled and self.drawdown:
            can_trade, dd_reason = self.drawdown.can_trade()
            approval.guard_results["drawdown"] = can_trade
            if not can_trade:
                approval.approved = False
                approval.reasons.append(f"Drawdown: {dd_reason}")
                self._log_verdict(signal, approved=False, reason=f"Drawdown: {dd_reason}")
                return approval

        # ── 3. Drift detector ──
        if toggles and toggles.drift_detector_enabled and self.drift:
            if self.drift.should_pause():
                _, drift_reason = self.drift.is_drifting()
                approval.approved = False
                approval.reasons.append(f"Drift: {drift_reason}")
                approval.guard_results["drift"] = False
                self._log_verdict(signal, approved=False, reason=f"Drift: {drift_reason}")
                return approval
            approval.guard_results["drift"] = True

        # ── 4. Data quality ──
        if toggles and toggles.data_quality_check and self.data_quality and self.oanda:
            candles = self.oanda.get_candles(symbol, "H1", 50)
            report = self.data_quality.validate_candles(candles)
            approval.guard_results["data_quality"] = report.is_valid
            if not report.is_valid:
                approval.approved = False
                approval.reasons.append(f"Data quality: {'; '.join(report.issues[:2])}")
                self._log_verdict(signal, approved=False, reason="Data quality failed")
                return approval

        # ── 5. Regime detection ──
        if toggles and toggles.regime_detection_enabled and self.regime and self.oanda:
            candles = self.oanda.get_candles(symbol, "H1", 100)
            regime_info = self.regime.detect_regime(candles)
            should = self.regime.should_trade(regime_info)
            approval.guard_results["regime"] = should
            if not should:
                approval.approved = False
                approval.reasons.append(f"Regime: {regime_info.regime.value}")
                self._log_verdict(signal, approved=False, reason=f"Regime: {regime_info.regime.value}")
                return approval

        # ── 6. Portfolio exposure ──
        if toggles and toggles.portfolio_manager_enabled and self.portfolio:
            can_add, port_reason = self.portfolio.can_add_position(
                symbol, signal.get("side", ""), signal.get("units", 1000), balance
            )
            approval.guard_results["portfolio"] = can_add
            if not can_add:
                approval.approved = False
                approval.reasons.append(f"Portfolio: {port_reason}")
                self._log_verdict(signal, approved=False, reason=f"Portfolio: {port_reason}")
                return approval

        # ── 7. Calibration (adjust, not block) ──
        raw_confidence = signal.get("confidence", 0)
        if toggles and toggles.calibration_enabled and self.calibration:
            calibrated = self.calibration.calibrate(raw_confidence)
            approval.adjusted_confidence = calibrated
            approval.guard_results["calibration"] = True
            if calibrated < 0.15:
                approval.approved = False
                approval.reasons.append(
                    f"Calibration: {raw_confidence:.2f} → {calibrated:.2f}"
                )
                approval.guard_results["calibration"] = False
                self._log_verdict(signal, approved=False, reason="Below calibrated threshold")
                return approval
        else:
            approval.adjusted_confidence = raw_confidence

        # ── 8. Slippage adjustment (adjust targets, not block) ──
        if toggles and toggles.slippage_model_enabled and self.slippage:
            units = signal.get("units", 1000)
            costs = self.slippage.estimate_costs(
                symbol, units, signal.get("side", ""),
                spread_pips=signal.get("spread_pips", 1.5),
            )
            adj_sl, adj_tp = self.slippage.adjust_targets(
                signal["entry"], signal["stop_loss"],
                signal["take_profit"], signal["side"], costs,
            )
            signal["stop_loss"] = adj_sl
            signal["take_profit"] = adj_tp
            approval.guard_results["slippage"] = True

        # ── 9. Same-theme exposure (soft penalty, not hard block) ──
        # Currency overlap is normal in FX (USD appears in most pairs).
        # Instead of blocking, reduce position size proportionally.
        theme_scale = 1.0
        if self.portfolio:
            symbol_currencies = set()
            parts = symbol.replace("_", "/").split("/")
            for p in parts:
                symbol_currencies.add(p.upper())

            try:
                report = self.portfolio.get_exposure_report()
                positions = report.get("positions", {})
                theme_count = 0
                for pos_symbol in positions:
                    pos_parts = pos_symbol.replace("_", "/").split("/")
                    pos_currencies = set(p.upper() for p in pos_parts)
                    if symbol_currencies & pos_currencies:  # overlap
                        theme_count += 1
                if theme_count >= 4:
                    # Only hard-block at extreme concentration (4+ same-currency)
                    approval.approved = False
                    approval.reasons.append(f"Theme cap: {theme_count} positions share {'/'.join(symbol_currencies)}")
                    approval.guard_results["theme_cap"] = False
                    self._log_verdict(signal, approved=False, reason="Same-theme exposure cap (4+)")
                    return approval
                elif theme_count >= 2:
                    # Soft penalty: scale down size by 25% per overlap above 1
                    theme_scale = max(0.25, 1.0 - 0.25 * (theme_count - 1))
                    approval.reasons.append(f"Theme penalty: {theme_count} overlaps → {theme_scale:.0%} size")
            except Exception:
                pass
            approval.guard_results["theme_cap"] = True

        # ── 10. Position sizing (pip-aware, dynamic based on recent performance) ──
        from trading.brokers.oanda import CRYPTO_PAIRS
        is_crypto = symbol in CRYPTO_PAIRS

        # Determine pip value for this pair
        if is_crypto:
            pip = 1.0
        elif "JPY" in symbol:
            pip = 0.01
        else:
            pip = 0.0001

        # Base risk: 0.5% per trade
        risk_pct = 0.005

        # Dynamic adjustment: reduce risk after losing streaks
        if self.db:
            try:
                recent = self.db.get_all_trades(limit=20) or []
                if len(recent) >= 5:
                    wins = sum(1 for t in recent if t.get("outcome") == "win")
                    win_rate = wins / len(recent)

                    # Check for consecutive losses
                    consec_losses = 0
                    for t in recent:
                        if t.get("outcome") == "loss":
                            consec_losses += 1
                        else:
                            break

                    if consec_losses >= 5:
                        risk_pct = 0.001  # 0.1% — heavy drawdown protection
                        approval.reasons.append(f"Sizing: 5+ consec losses → 0.1% risk")
                    elif consec_losses >= 3:
                        risk_pct = 0.002  # 0.2%
                        approval.reasons.append(f"Sizing: {consec_losses} consec losses → 0.2% risk")
                    elif win_rate < 0.35:
                        risk_pct = 0.002  # 0.2% — poor win rate
                        approval.reasons.append(f"Sizing: {win_rate:.0%} win rate → 0.2% risk")
                    elif win_rate < 0.45:
                        risk_pct = 0.003  # 0.3%
                    elif win_rate > 0.60:
                        risk_pct = 0.006  # 0.6% — performing well, slight increase

                    logger.info(
                        f"DYNAMIC SIZING: win_rate={win_rate:.0%} "
                        f"consec_losses={consec_losses} → risk={risk_pct:.1%}"
                    )
            except Exception as e:
                logger.debug(f"Dynamic sizing check failed: {e}")

        # Apply theme overlap scaling
        risk_pct *= theme_scale

        risk_amount = balance * risk_pct
        sl_distance = abs(signal.get("entry", 0) - signal.get("stop_loss", 0))

        # Pip-aware sizing: convert stop distance to pips, then size accordingly
        # units = risk_amount / (sl_pips * pip_value_per_unit)
        # For standard forex: pip_value ≈ pip for most pairs (simplified for micro accounts)
        sl_pips = sl_distance / pip if pip > 0 else 0

        if sl_pips > 0 and pip > 0:
            # risk_amount / (sl_pips * pip) gives units
            units = int(risk_amount / (sl_pips * pip))
            if is_crypto:
                units = max(1, min(units, 5))
            else:
                units = max(1, min(units, 5000))
        else:
            units = 1 if is_crypto else 1000

        approval.allowed_units = units
        approval.guard_results["sizing"] = True

        # All guards passed
        if approval.approved:
            approval.reasons.append("All guards passed")
            self._log_verdict(signal, approved=True, reason="All guards passed")

        return approval

    def _log_verdict(self, signal: dict, approved: bool, reason: str):
        """Log every guard verdict (approved or denied) to the database."""
        if self.db:
            try:
                self.db.save_signal(
                    symbol=signal.get("symbol", ""),
                    side=signal.get("side", ""),
                    confidence=signal.get("confidence", 0),
                    entry=signal.get("entry", 0),
                    sl=signal.get("stop_loss"),
                    tp=signal.get("take_profit"),
                    taken=approved,
                    reason_skipped="" if approved else reason,
                )
            except Exception as e:
                logger.debug(f"Failed to log verdict: {e}")

    def cycle_check(self) -> tuple[bool, str]:
        """
        Pre-cycle check — should we even scan this cycle?
        Checks drawdown and drift before any scanning happens.

        Returns (can_proceed, reason)
        """
        toggles = self.settings.toggles if self.settings else None

        if toggles and toggles.drawdown_guard_enabled and self.drawdown:
            can_trade, reason = self.drawdown.can_trade()
            if not can_trade:
                return False, f"Drawdown guard: {reason}"

        if toggles and toggles.drift_detector_enabled and self.drift:
            if self.drift.should_pause():
                _, reason = self.drift.is_drifting()
                return False, f"Drift detector: {reason}"

        return True, "OK"
