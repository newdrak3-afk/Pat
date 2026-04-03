"""
Position Manager — Handles trade lifecycle (open, monitor, close).

Extracted from auto_trader.py so the trader only orchestrates,
it doesn't manage individual position state.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from trading.trade_db import TradeDB
from trading.drawdown_guard import DrawdownGuard
from trading.drift_detector import DriftDetector
from trading.portfolio_manager import PortfolioManager
from trading.calibration import CalibrationLayer
from trading.loss_analyzer import LossAnalyzer
from trading.notifier import TelegramNotifier
from trading.risk_manager import RiskManager
from trading.models import Trade, Market, Prediction, ResearchResult

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages the lifecycle of open positions.

    Responsibilities:
    - Track open trades (in-memory + DB)
    - Detect closed positions via broker
    - Compute PnL on close
    - Update all downstream systems (drift, calibration, drawdown, portfolio, DB)
    - Send win/loss notifications
    """

    def __init__(
        self,
        oanda=None,
        db: Optional[TradeDB] = None,
        drawdown: Optional[DrawdownGuard] = None,
        drift: Optional[DriftDetector] = None,
        portfolio: Optional[PortfolioManager] = None,
        calibration: Optional[CalibrationLayer] = None,
        loss_analyzer: Optional[LossAnalyzer] = None,
        risk_mgr: Optional[RiskManager] = None,
        notifier: Optional[TelegramNotifier] = None,
    ):
        self.oanda = oanda
        self.db = db
        self.drawdown = drawdown
        self.drift = drift
        self.portfolio = portfolio
        self.calibration = calibration
        self.loss_analyzer = loss_analyzer
        self.risk_mgr = risk_mgr
        self.notifier = notifier

        self.open_trades: dict[str, dict] = {}  # trade_id -> trade info

    def add_trade(self, trade_info: dict):
        """Register a newly placed trade."""
        trade_id = trade_info["trade_id"]
        self.open_trades[trade_id] = trade_info

        # Track in portfolio
        if self.portfolio:
            self.portfolio.add_position(
                trade_info["symbol"],
                trade_info["side"],
                trade_info["units"],
                trade_info["entry"],
            )

        # Log to DB
        if self.db:
            self.db.save_trade(
                trade_id=trade_id,
                symbol=trade_info["symbol"],
                side=trade_info["side"],
                units=trade_info["units"],
                entry_price=trade_info["entry"],
                stop_loss=trade_info["stop_loss"],
                take_profit=trade_info["take_profit"],
                confidence=trade_info["confidence"],
                reasoning=trade_info.get("reasoning", ""),
            )
            self.db.save_signal(
                symbol=trade_info["symbol"],
                side=trade_info["side"],
                confidence=trade_info["confidence"],
                entry=trade_info["entry"],
                sl=trade_info["stop_loss"],
                tp=trade_info["take_profit"],
                taken=True,
                reason_skipped="",
            )

    def check_positions(self) -> dict:
        """
        Check open positions against broker and resolve closed trades.

        Returns stats dict with keys: closed, wins, losses, pnl
        """
        result = {"closed": 0, "wins": 0, "losses": 0, "pnl": 0.0}

        if not self.open_trades or not self.oanda:
            return result

        positions = self.oanda.get_positions()
        open_symbols = {p.symbol for p in positions}
        closed_ids = []

        for trade_id, info in self.open_trades.items():
            if info["symbol"] in open_symbols:
                continue

            closed_ids.append(trade_id)
            outcome, pnl = self._resolve_trade(trade_id, info)

            result["closed"] += 1
            result["pnl"] += pnl
            if outcome == "win":
                result["wins"] += 1
            else:
                result["losses"] += 1

        for tid in closed_ids:
            del self.open_trades[tid]

        # Save daily stats
        if self.db and result["closed"] > 0:
            self._save_daily_stats()

        return result

    def _resolve_trade(self, trade_id: str, info: dict) -> tuple[str, float]:
        """Determine outcome of a closed trade and update all systems.

        Uses actual broker PnL when available (from closed transactions),
        falls back to quote-based calculation.
        """
        quote = self.oanda.get_quote(info["symbol"]) if self.oanda else None
        current = quote.mid if quote else info["entry"]

        entry = info["entry"]
        side = info["side"]
        units = info["units"]

        # Try to get actual PnL from broker's closed transactions
        actual_pnl = self._get_broker_pnl(trade_id, info)

        if actual_pnl is not None:
            pnl = actual_pnl
            outcome = "win" if pnl > 0 else "loss"
        else:
            # Fallback: calculate from current price vs entry
            if side == "buy":
                pnl = (current - entry) * units
            else:
                pnl = (entry - current) * units
            outcome = "win" if pnl > 0 else "loss"

        info["status"] = outcome
        info["pnl"] = pnl
        info["closed_at"] = datetime.now(timezone.utc).isoformat()

        # Update DB
        if self.db:
            self.db.update_trade(trade_id=trade_id, exit_price=current, outcome=outcome, pnl=pnl)

        # Update drift detector
        if self.drift:
            self.drift.add_result(outcome, pnl)

        # Update calibration
        if self.calibration:
            self.calibration.add_outcome(
                info.get("raw_confidence", info["confidence"]),
                1 if outcome == "win" else 0,
            )

        # Update drawdown
        if self.drawdown and self.oanda:
            balance = self.oanda.get_account_balance()
            self.drawdown.update(balance)

        # Remove from portfolio
        if self.portfolio:
            self.portfolio.remove_position(info["symbol"])

        # Notify
        self._notify_outcome(trade_id, info, outcome, pnl)

        return outcome, pnl

    def _notify_outcome(self, trade_id: str, info: dict, outcome: str, pnl: float):
        """Send win/loss notification and log lessons for losses."""
        balance = self.oanda.get_account_balance() if self.oanda else 0

        # Build trade object for notifications
        units = info.get("units", 0)
        entry = info["entry"]
        confidence = info.get("confidence", 0)
        pair = info["symbol"].replace("_", "/")

        if outcome == "win":
            if self.notifier:
                self.notifier.send_forex_result(
                    pair=pair,
                    side=info["side"],
                    entry=entry,
                    units=units,
                    pnl=pnl,
                    confidence=confidence,
                    balance=balance,
                    outcome="WIN",
                )
            logger.info(f"WIN: {info['symbol']} +${pnl:.2f}")
        else:
            # Analyze loss
            lesson_text = ""
            if self.loss_analyzer:
                trade_obj = Trade(
                    trade_id=trade_id,
                    market_id=info["symbol"],
                    market_question=f"Forex: {info['symbol']}",
                    side=info["side"],
                    amount=abs(pnl),
                    entry_price=entry,
                    pnl=pnl,
                    outcome=outcome,
                )
                market = Market(
                    market_id=info["symbol"],
                    question=f"Forex: {info['symbol']}",
                    category="forex",
                    current_price=info.get("exit_price", entry),
                )
                prediction = Prediction(
                    market_id=info["symbol"],
                    market_price=entry,
                    predicted_probability=confidence,
                    confidence=confidence,
                )
                research = ResearchResult(market_id=info["symbol"])

                lesson = self.loss_analyzer.analyze_loss(trade_obj, market, research, prediction)
                lesson_text = lesson.description

                if self.db:
                    from uuid import uuid4
                    self.db.save_lesson(
                        lesson_id=str(uuid4())[:8],
                        trade_id=trade_id,
                        category=lesson.category,
                        description=lesson.description,
                        rule_added=lesson.rule_added,
                    )

                logger.info(f"LOSS: {info['symbol']} -${abs(pnl):.2f} | {lesson.rule_added}")
            else:
                logger.info(f"LOSS: {info['symbol']} -${abs(pnl):.2f}")

            if self.notifier:
                self.notifier.send_forex_result(
                    pair=pair,
                    side=info["side"],
                    entry=entry,
                    units=units,
                    pnl=pnl,
                    confidence=confidence,
                    balance=balance,
                    outcome="LOSS",
                    lesson=lesson_text,
                )

            if self.risk_mgr:
                self.risk_mgr.resolve_trade(trade_id, pnl)

    def _save_daily_stats(self):
        """Persist daily performance stats."""
        if not self.db or not self.oanda:
            return
        balance = self.oanda.get_account_balance()
        dd_info = self.drawdown.get_drawdown_info() if self.drawdown else None

        total_trades = len(self.open_trades)
        # Count from DB for accuracy
        summary = self.db.get_performance_summary() or {}

        self.db.save_daily_stats(
            total_trades=summary.get("total_trades", 0),
            wins=summary.get("wins", 0),
            losses=summary.get("losses", 0),
            pnl=summary.get("total_pnl", 0),
            max_drawdown=dd_info.max_drawdown_pct_seen if dd_info else 0,
            balance=balance,
        )

    def _get_broker_pnl(self, trade_id: str, info: dict) -> Optional[float]:
        """Try to get actual realized PnL from broker for a closed trade.

        Queries OANDA's transaction history for the actual closed PnL
        rather than guessing from TP/SL levels.
        """
        if not self.oanda:
            return None
        try:
            # OANDA provides realized PnL on closed trades via the transactions endpoint
            if hasattr(self.oanda, 'get_closed_trade_pnl'):
                return self.oanda.get_closed_trade_pnl(trade_id)
        except Exception as e:
            logger.debug(f"Could not get broker PnL for {trade_id}: {e}")
        return None

    @property
    def open_count(self) -> int:
        """Count open positions from broker (ground truth), fall back to in-memory."""
        if self.oanda:
            try:
                positions = self.oanda.get_positions()
                broker_count = len(positions)
                # Sync in-memory count if broker disagrees
                if broker_count != len(self.open_trades):
                    logger.warning(
                        f"Position count mismatch: broker={broker_count}, "
                        f"memory={len(self.open_trades)}"
                    )
                return broker_count
            except Exception:
                pass
        return len(self.open_trades)
