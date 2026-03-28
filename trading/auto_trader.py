"""
Auto Trader v2 — Enhanced with all protection layers.

Flow: Connect → [Regime Check → Data Quality → Scan → Portfolio Check →
       Drawdown Check → Drift Check → Slippage Adjust → Calibrate →
       Risk Check → Execute → Log to DB → Monitor → Learn]

New protections:
- Regime detection (skip volatile/unfavorable regimes)
- Data quality validation (reject bad candle data)
- Portfolio exposure limits (max per-currency exposure)
- Drawdown guard (hard stop on max drawdown)
- Drift detector (pause if strategy degrading)
- Slippage model (realistic cost accounting)
- Calibration (adjust overconfident predictions)
- SQLite database (structured trade logging)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from trading.config import SystemConfig
from trading.settings import Settings
from trading.brokers.oanda import OandaBroker
from trading.forex_scanner import ForexScanner
from trading.research_agent import ResearchAgent
from trading.risk_manager import RiskManager
from trading.loss_analyzer import LossAnalyzer
from trading.notifier import TelegramNotifier
from trading.models import Market, Trade, Prediction, ResearchResult

# New v2 modules
from trading.trade_db import TradeDB
from trading.regime_detector import RegimeDetector
from trading.data_quality import DataQualityChecker
from trading.portfolio_manager import PortfolioManager
from trading.drawdown_guard import DrawdownGuard
from trading.drift_detector import DriftDetector
from trading.slippage_model import SlippageModel
from trading.calibration import CalibrationLayer

logger = logging.getLogger(__name__)


class AutoTrader:
    """
    Fully automated trading loop with all protection layers.

    Scans → Guards → Trades → Monitors → Learns → Repeats
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.settings = Settings()

        # Brokers
        self.oanda = OandaBroker()

        # Core agents
        self.scanner = None  # initialized after connect
        self.researcher = ResearchAgent(self.config)
        self.risk_mgr = RiskManager(self.config)
        self.loss_analyzer = LossAnalyzer(self.config)
        self.notifier = TelegramNotifier(self.config)

        # v2 protection layers
        self.db = TradeDB()
        self.regime = RegimeDetector()
        self.data_quality = DataQualityChecker()
        self.portfolio = PortfolioManager()
        self.drawdown = DrawdownGuard()
        self.drift = DriftDetector()
        self.slippage = SlippageModel()
        self.calibration = CalibrationLayer()

        # State
        self._trades: list[dict] = []
        self._open_trades: dict[str, dict] = {}  # order_id -> trade info
        self._stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "cycles": 0,
            "blocked_by_regime": 0,
            "blocked_by_drawdown": 0,
            "blocked_by_drift": 0,
            "blocked_by_portfolio": 0,
            "blocked_by_data_quality": 0,
        }
        self._load_state()

    def _load_state(self):
        """Load saved state."""
        state_file = "trading/data/auto_trader_state.json"
        try:
            with open(state_file) as f:
                data = json.load(f)
                self._trades = data.get("trades", [])
                saved_stats = data.get("stats", {})
                self._stats.update(saved_stats)
                self._open_trades = data.get("open_trades", {})
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Save state to disk."""
        state_file = "trading/data/auto_trader_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({
                "trades": self._trades[-200:],  # keep last 200
                "stats": self._stats,
                "open_trades": self._open_trades,
            }, f, indent=2)

    def start(self, scan_interval: int = 300):
        """
        Start the auto-trading loop.

        Args:
            scan_interval: Seconds between scans (default 5 min)
        """
        # Use settings override if available
        scan_interval = self.settings.toggles.scan_interval_seconds or scan_interval

        # Check if scanning is enabled
        if not self.settings.toggles.scanning_enabled:
            logger.info("Scanning is DISABLED in settings. Exiting.")
            self.notifier.send_system_alert("Bot started but scanning is disabled. Use settings to enable.")
            return

        # Connect to OANDA
        logger.info("Connecting to OANDA...")
        if not self.oanda.connect():
            self.notifier.send_system_alert(
                "Failed to connect to OANDA. Check your API key and account ID."
            )
            logger.error("OANDA connection failed — exiting")
            return

        balance = self.oanda.get_account_balance()
        self.scanner = ForexScanner(self.oanda)

        # Update bankroll from OANDA
        self.risk_mgr._bankroll = balance
        self.risk_mgr._save_state()

        # Initialize drawdown guard with current balance
        self.drawdown.update(balance)

        self.notifier.send_startup(balance, dry_run=not self.settings.toggles.auto_trading_enabled)
        logger.info(f"Auto trader v2 started — Balance: ${balance:.2f}")
        logger.info(f"Scan interval: {scan_interval}s")
        logger.info(f"Auto-trading: {'ON' if self.settings.toggles.auto_trading_enabled else 'OFF (alerts only)'}")
        logger.info(self.settings.get_status())

        while True:
            # Re-check settings each cycle (allows live toggle)
            self.settings = Settings()
            if not self.settings.toggles.scanning_enabled:
                logger.info("Scanning disabled — sleeping...")
                time.sleep(scan_interval)
                continue

            try:
                self._run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.notifier.send_system_alert("Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self.notifier.send_system_alert(f"Error: {str(e)[:200]}")

            try:
                time.sleep(scan_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break

    def _run_cycle(self):
        """Run one full cycle with all protection layers."""
        self._stats["cycles"] += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"CYCLE {self._stats['cycles']}")
        logger.info(f"{'='*50}")

        # Step 0: Update balance and check guards
        balance = self.oanda.get_account_balance()
        self.drawdown.update(balance)

        # ── GUARD: Drawdown check ──
        if self.settings.toggles.drawdown_guard_enabled:
            can_trade, dd_reason = self.drawdown.can_trade()
            if not can_trade:
                logger.warning(f"DRAWDOWN GUARD: {dd_reason}")
                self._stats["blocked_by_drawdown"] += 1
                self.notifier.send_system_alert(f"Trading paused: {dd_reason}")
                self._save_state()
                return

        # ── GUARD: Drift check ──
        if self.settings.toggles.drift_detector_enabled:
            if self.drift.should_pause():
                drifting, drift_reason = self.drift.is_drifting()
                logger.warning(f"DRIFT DETECTOR: {drift_reason}")
                self._stats["blocked_by_drift"] += 1
                self.notifier.send_system_alert(f"Trading paused — strategy drift: {drift_reason}")
                self._save_state()
                return

        # Step 1: Check open positions and resolve completed trades
        self._check_positions()

        # Step 2: Scan for new signals
        signals = self.scanner.scan_all_pairs()

        if not signals:
            logger.info("No signals found this cycle")
            self.notifier.send_scan_summary(
                total_markets=28,
                flagged=0,
                predictions=0,
                trades=0,
                blocked=0,
            )
            self._save_state()
            return

        # Step 3: Filter through all protection layers and trade
        trades_placed = 0
        trades_blocked = 0
        max_trades = self.settings.toggles.max_trades_per_cycle

        for signal in signals[:5]:  # max 5 signals per cycle
            if trades_placed >= max_trades:
                break

            symbol = signal["symbol"]

            # ── GUARD: Data quality ──
            if self.settings.toggles.data_quality_check:
                candles = self.oanda.get_candles(symbol, "H1", 50)
                report = self.data_quality.validate_candles(candles)
                if not report.is_valid:
                    logger.info(f"DATA QUALITY: {symbol} failed — {report.issues}")
                    self._stats["blocked_by_data_quality"] += 1
                    # Log skipped signal to DB
                    self.db.save_signal(
                        symbol=symbol, side=signal["side"],
                        confidence=signal["confidence"],
                        entry=signal["entry"],
                        sl=signal["stop_loss"], tp=signal["take_profit"],
                        taken=False, reason_skipped="Data quality check failed",
                    )
                    trades_blocked += 1
                    continue

            # ── GUARD: Regime detection ──
            if self.settings.toggles.regime_detection_enabled:
                candles = self.oanda.get_candles(symbol, "H1", 100)
                regime_info = self.regime.detect_regime(candles)
                if not self.regime.should_trade(regime_info):
                    logger.info(
                        f"REGIME: {symbol} in {regime_info.regime.value} — skipping"
                    )
                    self._stats["blocked_by_regime"] += 1
                    self.db.save_signal(
                        symbol=symbol, side=signal["side"],
                        confidence=signal["confidence"],
                        entry=signal["entry"],
                        sl=signal["stop_loss"], tp=signal["take_profit"],
                        taken=False,
                        reason_skipped=f"Unfavorable regime: {regime_info.regime.value}",
                    )
                    trades_blocked += 1
                    continue

            # ── GUARD: Portfolio exposure ──
            if self.settings.toggles.portfolio_manager_enabled:
                can_add, port_reason = self.portfolio.can_add_position(
                    symbol, signal["side"], signal.get("units", 1000), balance
                )
                if not can_add:
                    logger.info(f"PORTFOLIO: {symbol} blocked — {port_reason}")
                    self._stats["blocked_by_portfolio"] += 1
                    self.db.save_signal(
                        symbol=symbol, side=signal["side"],
                        confidence=signal["confidence"],
                        entry=signal["entry"],
                        sl=signal["stop_loss"], tp=signal["take_profit"],
                        taken=False, reason_skipped=f"Portfolio: {port_reason}",
                    )
                    trades_blocked += 1
                    continue

            # ── ADJUST: Calibration ──
            raw_confidence = signal["confidence"]
            if self.settings.toggles.calibration_enabled:
                calibrated = self.calibration.calibrate(raw_confidence)
                signal["confidence"] = calibrated
                if calibrated < 0.35:  # Below minimum threshold after calibration
                    logger.info(
                        f"CALIBRATION: {symbol} confidence dropped "
                        f"{raw_confidence:.2f} → {calibrated:.2f} — skipping"
                    )
                    trades_blocked += 1
                    continue

            # ── ADJUST: Slippage costs ──
            if self.settings.toggles.slippage_model_enabled:
                units = signal.get("units", 1000)
                costs = self.slippage.estimate_costs(
                    symbol, units, signal["side"],
                    spread_pips=signal.get("spread_pips", 1.5),
                )
                adj_sl, adj_tp = self.slippage.adjust_targets(
                    signal["entry"], signal["stop_loss"],
                    signal["take_profit"], signal["side"], costs,
                )
                signal["stop_loss"] = adj_sl
                signal["take_profit"] = adj_tp

            # Convert signal to Prediction for risk manager
            prediction = Prediction(
                market_id=signal["symbol"],
                market_question=f"Forex: {signal['symbol']} {signal['side'].upper()}",
                market_price=signal["entry"],
                predicted_probability=signal["confidence"],
                confidence=signal["confidence"],
                edge=signal["confidence"] - 0.5,
                recommended_side=signal["side"],
                reasoning=signal["reasoning"],
            )

            market = Market(
                market_id=signal["symbol"],
                question=f"Forex: {signal['symbol']}",
                category="forex",
                current_price=signal["entry"],
                liquidity=100000,
                spread=0,
            )

            # ── GUARD: Risk check ──
            risk_eval = self.risk_mgr.evaluate_trade(prediction, market)

            if not risk_eval["approved"]:
                logger.info(
                    f"RISK: {signal['symbol']} — {risk_eval['reason']}"
                )
                self.notifier.send_trade_blocked(
                    market, prediction, risk_eval["reason"]
                )
                self.db.save_signal(
                    symbol=symbol, side=signal["side"],
                    confidence=signal["confidence"],
                    entry=signal["entry"],
                    sl=signal["stop_loss"], tp=signal["take_profit"],
                    taken=False, reason_skipped=f"Risk: {risk_eval['reason']}",
                )
                trades_blocked += 1
                continue

            # ── Check if auto-trading is enabled ──
            if not self.settings.toggles.auto_trading_enabled:
                # Alert-only mode: send signal but don't place trade
                self.notifier.send_forex_alert(
                    symbol=signal["symbol"],
                    side=signal["side"],
                    hit_pct=signal["confidence"] * 100,
                    entry_price=signal["entry"],
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    units=0,
                    reasoning=f"[ALERT ONLY] {signal['reasoning']}",
                )
                self.db.save_signal(
                    symbol=symbol, side=signal["side"],
                    confidence=signal["confidence"],
                    entry=signal["entry"],
                    sl=signal["stop_loss"], tp=signal["take_profit"],
                    taken=False, reason_skipped="Auto-trading disabled (alert only)",
                )
                logger.info(f"ALERT ONLY: {signal['side']} {symbol} @ {signal['entry']}")
                continue

            # ── EXECUTE: Place the trade ──
            balance = self.oanda.get_account_balance()
            risk_amount = balance * self.config.risk.max_bet_pct
            sl_distance = abs(signal["entry"] - signal["stop_loss"])
            if sl_distance > 0:
                units = int(risk_amount / sl_distance)
                units = max(1, min(units, 10000))
            else:
                units = 1000

            result = self.oanda.place_order_with_stops(
                symbol=signal["symbol"],
                side=signal["side"],
                quantity=units,
                stop_loss_pips=signal["sl_pips"],
                take_profit_pips=signal["tp_pips"],
            )

            if result.success:
                trade_id = result.order_id
                trade_info = {
                    "trade_id": trade_id,
                    "symbol": signal["symbol"],
                    "side": signal["side"],
                    "units": units,
                    "entry": result.price,
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "confidence": signal["confidence"],
                    "raw_confidence": raw_confidence,
                    "reasoning": signal["reasoning"],
                    "placed_at": datetime.utcnow().isoformat(),
                    "status": "open",
                }

                self._open_trades[trade_id] = trade_info
                self._trades.append(trade_info)
                self._stats["total_trades"] += 1
                trades_placed += 1

                # Track in portfolio manager
                self.portfolio.add_position(symbol, signal["side"], units, result.price)

                # Log to SQLite DB
                self.db.save_trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=signal["side"],
                    units=units,
                    entry_price=result.price,
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    confidence=signal["confidence"],
                    reasoning=signal["reasoning"],
                )

                # Log signal as taken
                self.db.save_signal(
                    symbol=symbol, side=signal["side"],
                    confidence=signal["confidence"],
                    entry=signal["entry"],
                    sl=signal["stop_loss"], tp=signal["take_profit"],
                    taken=True, reason_skipped="",
                )

                # Notify
                self.notifier.send_forex_alert(
                    symbol=signal["symbol"],
                    side=signal["side"],
                    hit_pct=signal["confidence"] * 100,
                    entry_price=result.price,
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    units=units,
                    reasoning=signal["reasoning"],
                )

                logger.info(
                    f"TRADE PLACED: {signal['side'].upper()} "
                    f"{signal['symbol']} {units} units @ {result.price}"
                )

            else:
                logger.warning(
                    f"Order failed for {signal['symbol']}: {result.message}"
                )

        # Summary
        self.notifier.send_scan_summary(
            total_markets=28,
            flagged=len(signals),
            predictions=len(signals),
            trades=trades_placed,
            blocked=trades_blocked,
        )

        self._save_state()

    def _check_positions(self):
        """Check open positions and resolve completed trades."""
        if not self._open_trades:
            return

        positions = self.oanda.get_positions()
        open_symbols = {p.symbol for p in positions}

        closed_trade_ids = []

        for trade_id, trade_info in self._open_trades.items():
            symbol = trade_info["symbol"]

            if symbol not in open_symbols:
                closed_trade_ids.append(trade_id)

                quote = self.oanda.get_quote(symbol)
                current = quote.mid if quote else trade_info["entry"]

                entry = trade_info["entry"]
                side = trade_info["side"]
                units = trade_info["units"]

                tp = trade_info["take_profit"]
                sl = trade_info["stop_loss"]

                if side == "buy":
                    if current >= tp:
                        outcome = "win"
                        pnl = abs(tp - entry) * units
                    else:
                        outcome = "loss"
                        pnl = -abs(entry - sl) * units
                else:
                    if current <= tp:
                        outcome = "win"
                        pnl = abs(entry - tp) * units
                    else:
                        outcome = "loss"
                        pnl = -abs(sl - entry) * units

                trade_info["status"] = outcome
                trade_info["pnl"] = pnl
                trade_info["closed_at"] = datetime.utcnow().isoformat()

                # Update DB
                self.db.update_trade(
                    trade_id=trade_id,
                    exit_price=current,
                    outcome=outcome,
                    pnl=pnl,
                )

                # Update drift detector
                self.drift.add_result(outcome, pnl)

                # Update calibration with actual outcome
                self.calibration.add_outcome(
                    trade_info.get("raw_confidence", trade_info["confidence"]),
                    1 if outcome == "win" else 0,
                )

                # Update drawdown
                balance = self.oanda.get_account_balance()
                self.drawdown.update(balance)

                # Remove from portfolio
                self.portfolio.remove_position(symbol)

                if outcome == "win":
                    self._stats["wins"] += 1
                    self._stats["total_pnl"] += pnl

                    trade_obj = Trade(
                        trade_id=trade_id,
                        market_id=symbol,
                        market_question=f"Forex: {symbol}",
                        side=side,
                        amount=abs(pnl),
                        entry_price=entry,
                        pnl=pnl,
                        outcome="win",
                    )
                    self.notifier.send_win_alert(trade_obj, pnl, balance)
                    logger.info(f"WIN: {symbol} +${pnl:.2f}")

                else:
                    self._stats["losses"] += 1
                    self._stats["total_pnl"] += pnl

                    trade_obj = Trade(
                        trade_id=trade_id,
                        market_id=symbol,
                        market_question=f"Forex: {symbol}",
                        side=side,
                        amount=abs(pnl),
                        entry_price=entry,
                        pnl=pnl,
                        outcome="loss",
                    )

                    market = Market(
                        market_id=symbol,
                        question=f"Forex: {symbol}",
                        category="forex",
                        current_price=current,
                    )

                    research = ResearchResult(market_id=symbol)
                    prediction = Prediction(
                        market_id=symbol,
                        market_price=entry,
                        predicted_probability=trade_info["confidence"],
                        confidence=trade_info["confidence"],
                    )

                    lesson = self.loss_analyzer.analyze_loss(
                        trade_obj, market, research, prediction
                    )

                    # Save lesson to DB
                    self.db.save_lesson(
                        trade_id=trade_id,
                        category=lesson.category,
                        description=lesson.description,
                        rule_added=lesson.rule_added,
                    )

                    self.notifier.send_loss_alert(
                        trade_obj, pnl, balance, lesson.description
                    )

                    logger.info(
                        f"LOSS: {symbol} -${abs(pnl):.2f} | "
                        f"Lesson: {lesson.category} — {lesson.rule_added}"
                    )

                    self.risk_mgr.resolve_trade(trade_id, pnl)

        for tid in closed_trade_ids:
            del self._open_trades[tid]

        # Save daily stats
        if self._stats["total_trades"] > 0:
            balance = self.oanda.get_account_balance()
            dd_info = self.drawdown.get_drawdown_info()
            self.db.save_daily_stats(
                total_trades=self._stats["total_trades"],
                wins=self._stats["wins"],
                losses=self._stats["losses"],
                pnl=self._stats["total_pnl"],
                max_drawdown=dd_info.max_drawdown_pct_seen,
                balance=balance,
            )

        self._save_state()

    def get_status(self) -> str:
        """Get current auto trader status with all guard statuses."""
        total = self._stats["wins"] + self._stats["losses"]
        win_rate = (self._stats["wins"] / total * 100) if total > 0 else 0
        balance = self.oanda.get_account_balance() if self.oanda.connected else 0

        lines = [
            "╔══════════════════════════════════════╗",
            "║     AUTO TRADER v2 STATUS            ║",
            "╚══════════════════════════════════════╝",
            "",
            f"  Balance:       ${balance:,.2f}",
            f"  Cycles:        {self._stats['cycles']}",
            f"  Total Trades:  {self._stats['total_trades']}",
            f"  Wins/Losses:   {self._stats['wins']}/{self._stats['losses']}",
            f"  Win Rate:      {win_rate:.0f}%",
            f"  Total PnL:     ${self._stats['total_pnl']:+,.2f}",
            f"  Open Trades:   {len(self._open_trades)}",
            "",
            "  ── Guards ──",
            f"  Blocked by Regime:     {self._stats.get('blocked_by_regime', 0)}",
            f"  Blocked by Drawdown:   {self._stats.get('blocked_by_drawdown', 0)}",
            f"  Blocked by Drift:      {self._stats.get('blocked_by_drift', 0)}",
            f"  Blocked by Portfolio:  {self._stats.get('blocked_by_portfolio', 0)}",
            f"  Blocked by Data:       {self._stats.get('blocked_by_data_quality', 0)}",
            "",
        ]

        # Add drawdown status
        lines.append(self.drawdown.get_status())
        lines.append("")

        # Add drift status
        lines.append(self.drift.get_status())
        lines.append("")

        # Add portfolio exposure
        exposure = self.portfolio.get_exposure_report()
        lines.append(f"  Portfolio Positions: {exposure.get('total_positions', 0)}")
        lines.append(f"  Total Exposure: {exposure.get('total_exposure_pct', 0):.1f}%")

        return "\n".join(lines)

    def get_full_report(self) -> str:
        """Get comprehensive report including DB stats."""
        lines = [self.get_status(), ""]

        # DB performance summary
        summary = self.db.get_performance_summary()
        if summary:
            lines.extend([
                "  ── Database Stats ──",
                f"  DB Trades:     {summary.get('total_trades', 0)}",
                f"  DB Win Rate:   {summary.get('win_rate', 0):.0%}",
                f"  DB Total PnL:  ${summary.get('total_pnl', 0):+,.2f}",
                "",
            ])

        # Calibration stats
        cal_stats = self.calibration.get_calibration_stats()
        if cal_stats.get("total_samples", 0) > 0:
            lines.extend([
                "  ── Calibration ──",
                f"  Samples:       {cal_stats['total_samples']}",
                f"  Brier Score:   {cal_stats.get('brier_score', 0):.4f}",
                f"  Overconfident: {'Yes' if self.calibration.is_overconfident() else 'No'}",
                "",
            ])

        return "\n".join(lines)
