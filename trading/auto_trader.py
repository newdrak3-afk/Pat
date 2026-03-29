"""
Auto Trader v3 — Refactored with GuardEngine + PositionManager.

Flow: Connect → Telegram Bot → Main Loop →
      [Guard Cycle Check → Scan → GuardEngine.evaluate() →
       Execute → PositionManager → Learn]

Architecture:
- GuardEngine: Centralized trade approval (one pipeline, one verdict)
- PositionManager: Trade lifecycle (open, monitor, close, notify)
- SessionAwareness: Tags trades with active session
- Settings: Runtime profiles (dev/paper/practice/live)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from trading.config import SystemConfig
from trading.settings import Settings
from trading.brokers.oanda import OandaBroker
from trading.forex_scanner import ForexScanner
from trading.research_agent import ResearchAgent
from trading.risk_manager import RiskManager
from trading.loss_analyzer import LossAnalyzer
from trading.notifier import TelegramNotifier
from trading.models import Market, Trade, Prediction, ResearchResult

# v2 modules
from trading.trade_db import TradeDB
from trading.regime_detector import RegimeDetector
from trading.data_quality import DataQualityChecker
from trading.portfolio_manager import PortfolioManager
from trading.drawdown_guard import DrawdownGuard
from trading.drift_detector import DriftDetector
from trading.slippage_model import SlippageModel
from trading.calibration import CalibrationLayer
from trading.telegram_bot import TelegramBot

# v3 modules
from trading.guard_engine import GuardEngine
from trading.position_manager import PositionManager
from trading.session_awareness import tag_trade_session, get_current_session

logger = logging.getLogger(__name__)


class AutoTrader:
    """
    Fully automated trading loop.

    v3: Orchestrates GuardEngine + PositionManager instead of inline guards.
    Scans → Guard → Trade → Monitor → Learn → Repeat
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

        # v3: Centralized guard engine
        self.guard_engine = GuardEngine(
            settings=self.settings,
            oanda=self.oanda,
            data_quality=self.data_quality,
            regime=self.regime,
            portfolio=self.portfolio,
            calibration=self.calibration,
            slippage=self.slippage,
            drawdown=self.drawdown,
            drift=self.drift,
            db=self.db,
        )

        # v3: Position manager
        self.position_mgr = PositionManager(
            oanda=self.oanda,
            db=self.db,
            drawdown=self.drawdown,
            drift=self.drift,
            portfolio=self.portfolio,
            calibration=self.calibration,
            loss_analyzer=self.loss_analyzer,
            risk_mgr=self.risk_mgr,
            notifier=self.notifier,
        )

        # Telegram command bot
        self.telegram_bot = TelegramBot()

        # Stats
        self._stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "cycles": 0,
            "blocked_signals": 0,
        }
        self._load_state()

    def _load_state(self):
        """Load saved state."""
        state_file = "trading/data/auto_trader_state.json"
        try:
            with open(state_file) as f:
                data = json.load(f)
                saved_stats = data.get("stats", {})
                self._stats.update(saved_stats)
                # Restore open trades to position manager
                open_trades = data.get("open_trades", {})
                self.position_mgr.open_trades = open_trades
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Save state to disk."""
        state_file = "trading/data/auto_trader_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({
                "stats": self._stats,
                "open_trades": self.position_mgr.open_trades,
            }, f, indent=2)

    def start(self, scan_interval: int = 300):
        """
        Start the auto-trading loop.

        Always starts the Telegram command bot first so you can
        control the system from your phone even when scanning is off.
        """
        scan_interval = self.settings.toggles.scan_interval_seconds or scan_interval

        # Connect to OANDA first
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

        # Start Telegram command listener FIRST (always runs)
        self.telegram_bot.connect(
            settings=self.settings,
            oanda=self.oanda,
            scanner=self.scanner,
            drawdown=self.drawdown,
            drift=self.drift,
            portfolio=self.portfolio,
            calibration=self.calibration,
            db=self.db,
            loss_analyzer=self.loss_analyzer,
            get_status_fn=self.get_status,
            get_report_fn=self.get_full_report,
        )
        self.telegram_bot.start()

        self.notifier.send_startup(balance, dry_run=not self.settings.toggles.auto_trading_enabled)
        logger.info(f"Auto trader v3 started — Balance: ${balance:.2f}")
        logger.info(f"Scan interval: {scan_interval}s")
        logger.info(f"Profile: {self.settings.toggles.runtime_profile}")
        logger.info(self.settings.get_status())

        if not self.settings.toggles.scanning_enabled:
            logger.info("Scanning is OFF. Send /resume in Telegram to start.")
            self.notifier.send_system_alert(
                "Bot online. Scanning is OFF.\n"
                "Send /resume to start scanning.\n"
                "Send /help to see all commands."
            )

        # Main loop — always runs, checks settings each cycle
        while True:
            self.settings = Settings()
            self.telegram_bot._settings = self.settings
            # Keep guard engine settings in sync
            self.guard_engine.settings = self.settings

            if not self.settings.toggles.scanning_enabled:
                logger.debug("Scanning disabled — waiting for /resume...")
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    self.telegram_bot.stop()
                    break
                continue

            try:
                self._run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.notifier.send_system_alert("Bot stopped by user.")
                self.telegram_bot.stop()
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self.notifier.send_system_alert(f"Error: {str(e)[:200]}")

            try:
                time.sleep(scan_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.telegram_bot.stop()
                break

    def _run_cycle(self):
        """Run one full scan-guard-trade cycle."""
        self._stats["cycles"] += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"CYCLE {self._stats['cycles']}")
        logger.info(f"{'='*50}")

        # Step 0: Update balance
        balance = self.oanda.get_account_balance()
        self.drawdown.update(balance)

        # Step 1: Pre-cycle guard check (drawdown + drift)
        can_proceed, reason = self.guard_engine.cycle_check()
        if not can_proceed:
            logger.warning(f"CYCLE BLOCKED: {reason}")
            self.notifier.send_system_alert(f"Trading paused: {reason}")
            self._save_state()
            return

        # Step 2: Check open positions
        close_result = self.position_mgr.check_positions()
        if close_result["closed"] > 0:
            self._stats["wins"] += close_result["wins"]
            self._stats["losses"] += close_result["losses"]
            self._stats["total_pnl"] += close_result["pnl"]

        # Step 3: Scan for new signals
        signals = self.scanner.scan_all_pairs()

        if not signals:
            logger.info("No signals found this cycle")
            from trading.brokers.oanda import ALL_PAIRS
            self.notifier.send_scan_summary(
                total_markets=len(ALL_PAIRS),
                flagged=0,
                predictions=0,
                trades=0,
                blocked=0,
            )
            self._save_state()
            return

        # Step 4: Run each signal through the guard engine
        trades_placed = 0
        trades_blocked = 0
        max_trades = self.settings.toggles.max_trades_per_cycle

        for signal in signals[:5]:
            if trades_placed >= max_trades:
                break

            # Run all guards via GuardEngine
            approval = self.guard_engine.evaluate(
                signal=signal,
                balance=balance,
                open_trade_count=self.position_mgr.open_count,
                max_open=5,
            )

            if not approval.approved:
                logger.info(f"BLOCKED: {signal['symbol']} — {approval.summary()}")
                trades_blocked += 1
                self._stats["blocked_signals"] += 1
                continue

            # Update signal with guard adjustments
            signal["confidence"] = approval.adjusted_confidence

            # Check auto-trading toggle
            if not self.settings.toggles.auto_trading_enabled:
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
                    symbol=signal["symbol"], side=signal["side"],
                    confidence=signal["confidence"],
                    entry=signal["entry"],
                    sl=signal["stop_loss"], tp=signal["take_profit"],
                    taken=False, reason_skipped="Auto-trading disabled (alert only)",
                )
                continue

            # Execute the trade
            units = approval.allowed_units
            result = self.oanda.place_order_with_stops(
                symbol=signal["symbol"],
                side=signal["side"],
                quantity=units,
                stop_loss_pips=signal["sl_pips"],
                take_profit_pips=signal["tp_pips"],
            )

            if result.success:
                trade_info = {
                    "trade_id": result.order_id,
                    "symbol": signal["symbol"],
                    "side": signal["side"],
                    "units": units,
                    "entry": result.price,
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "confidence": signal["confidence"],
                    "raw_confidence": signal.get("raw_confidence", signal["confidence"]),
                    "reasoning": signal["reasoning"],
                    "placed_at": datetime.utcnow().isoformat(),
                    "status": "open",
                }

                # Tag with session info
                tag_trade_session(trade_info)

                # Register with position manager (handles DB, portfolio, etc)
                self.position_mgr.add_trade(trade_info)

                self._stats["total_trades"] += 1
                trades_placed += 1

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
                    f"{signal['symbol']} {units} units @ {result.price} "
                    f"[{trade_info.get('session', 'unknown')}]"
                )
            else:
                logger.warning(f"Order failed for {signal['symbol']}: {result.message}")

        # Summary
        from trading.brokers.oanda import ALL_PAIRS
        self.notifier.send_scan_summary(
            total_markets=len(ALL_PAIRS),
            flagged=len(signals),
            predictions=len(signals),
            trades=trades_placed,
            blocked=trades_blocked,
        )

        self._save_state()

    def get_status(self) -> str:
        """Get current auto trader status."""
        total = self._stats["wins"] + self._stats["losses"]
        win_rate = (self._stats["wins"] / total * 100) if total > 0 else 0
        balance = self.oanda.get_account_balance() if self.oanda.connected else 0
        session = get_current_session()

        lines = [
            "╔══════════════════════════════════════╗",
            "║     AUTO TRADER v3 STATUS            ║",
            "╚══════════════════════════════════════╝",
            "",
            f"  Profile:       {self.settings.toggles.runtime_profile}",
            f"  Session:       {session.primary.replace('_', ' ').title()}",
            f"  Liquidity:     {session.liquidity.upper()}",
            f"  Balance:       ${balance:,.2f}",
            f"  Cycles:        {self._stats['cycles']}",
            f"  Total Trades:  {self._stats['total_trades']}",
            f"  Wins/Losses:   {self._stats['wins']}/{self._stats['losses']}",
            f"  Win Rate:      {win_rate:.0f}%",
            f"  Total PnL:     ${self._stats['total_pnl']:+,.2f}",
            f"  Open Trades:   {self.position_mgr.open_count}",
            f"  Blocked:       {self._stats.get('blocked_signals', 0)}",
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
        try:
            summary = self.db.get_performance_summary() or {}
            total = summary.get("total_trades") or 0
            win_rate = summary.get("win_rate") or 0
            total_pnl = summary.get("total_pnl") or 0
            lines.extend([
                "  ── Database Stats ──",
                f"  DB Trades:     {total}",
                f"  DB Win Rate:   {win_rate:.0%}",
                f"  DB Total PnL:  ${total_pnl:+,.2f}",
                "",
            ])
        except Exception:
            lines.append("  DB Stats: No data yet\n")

        # Calibration stats
        try:
            cal_stats = self.calibration.get_calibration_stats() or {}
            samples = cal_stats.get("total_samples") or 0
            if samples > 0:
                brier = cal_stats.get("brier_score") or 0
                lines.extend([
                    "  ── Calibration ──",
                    f"  Samples:       {samples}",
                    f"  Brier Score:   {brier:.4f}",
                    f"  Overconfident: {'Yes' if self.calibration.is_overconfident() else 'No'}",
                    "",
                ])
        except Exception:
            pass

        return "\n".join(lines)
