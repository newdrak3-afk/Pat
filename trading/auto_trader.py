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

import hashlib
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

        # Options trader (wired externally by trading_main.py)
        self.options_trader = None

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
        """Load saved state with crash recovery.

        On restart, reconciles saved open trades against broker positions
        to avoid duplicate orders and drop stale entries.
        If no state file exists (fresh deploy), syncs from broker directly.
        """
        state_file = "trading/data/auto_trader_state.json"
        try:
            with open(state_file) as f:
                data = json.load(f)
                saved_stats = data.get("stats", {})
                self._stats.update(saved_stats)

                saved_open = data.get("open_trades", {})
                reconciled = self._reconcile_open_trades(saved_open)
                self.position_mgr.open_trades = reconciled
        except (FileNotFoundError, json.JSONDecodeError):
            # No state file (fresh deploy) — sync from broker
            self._sync_from_broker()

    def _sync_from_broker(self):
        """Sync open trades from broker when no state file exists.

        This handles fresh deploys on Railway where the state file is wiped.
        Queries the broker for all open positions and registers them so
        the bot knows about existing trades and won't open duplicates.
        """
        try:
            if not self.oanda.connect():
                return

            broker_positions = self.oanda.get_positions()
            if not broker_positions:
                return

            logger.info(f"Fresh deploy: syncing {len(broker_positions)} positions from broker")

            for pos in broker_positions:
                side = "buy" if pos.quantity > 0 else "sell"
                trade_id = f"synced_{pos.symbol}_{side}"
                trade_info = {
                    "trade_id": trade_id,
                    "symbol": pos.symbol,
                    "side": side,
                    "units": abs(pos.quantity),
                    "entry": pos.entry_price,
                    "stop_loss": 0,
                    "take_profit": 0,
                    "confidence": 0,
                    "reasoning": "Synced from broker on restart",
                    "placed_at": datetime.utcnow().isoformat(),
                    "status": "open",
                }
                self.position_mgr.open_trades[trade_id] = trade_info

                # Save to DB so FK references work for lessons
                self.db.save_trade(
                    trade_id=trade_id,
                    symbol=pos.symbol,
                    side=side,
                    units=abs(pos.quantity),
                    entry_price=pos.entry_price,
                    stop_loss=0,
                    take_profit=0,
                    confidence=0,
                    reasoning="Synced from broker on restart",
                )

                # Register in portfolio manager
                if self.portfolio:
                    self.portfolio.add_position(
                        pos.symbol, side, abs(pos.quantity), pos.entry_price
                    )

            self._stats["total_trades"] = len(broker_positions)
            logger.info(f"Synced {len(broker_positions)} positions from broker")
            self._save_state()

        except Exception as e:
            logger.warning(f"Could not sync from broker: {e}")

    def _reconcile_open_trades(self, saved_open: dict) -> dict:
        """Reconcile saved open trades with actual broker positions.

        - Drops saved trades that no longer exist on the broker.
        - Keeps broker positions that are still tracked in saved state.
        - Logs any discrepancies for debugging.
        """
        if not saved_open:
            return {}

        try:
            broker_positions = self.oanda.get_positions()
        except Exception as e:
            logger.warning(f"Could not fetch broker positions for reconciliation: {e}")
            # If we can't reach the broker yet, keep saved state as-is;
            # startup_checks() will catch connectivity issues later.
            return saved_open

        # Build a set of (symbol, side) tuples currently open on the broker
        broker_open = set()
        for pos in broker_positions:
            broker_open.add((pos.symbol, pos.side))

        reconciled: dict = {}
        for trade_id, info in saved_open.items():
            key = (info.get("symbol"), info.get("side"))
            if key in broker_open:
                reconciled[trade_id] = info
            else:
                logger.warning(
                    f"Crash recovery: dropping stale trade {trade_id} "
                    f"({info.get('symbol')} {info.get('side')}) — "
                    "no matching broker position"
                )

        if len(reconciled) != len(saved_open):
            logger.info(
                f"Crash recovery: kept {len(reconciled)}/{len(saved_open)} "
                "saved trades after broker reconciliation"
            )

        return reconciled

    def _save_state(self):
        """Save state to disk."""
        state_file = "trading/data/auto_trader_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({
                "stats": self._stats,
                "open_trades": self.position_mgr.open_trades,
            }, f, indent=2)

    def _startup_checks(self):
        """Fail-fast pre-flight checks before starting the trading loop.

        Raises ``RuntimeError`` if any critical requirement is missing so
        the bot never enters the main loop in a broken state.
        """
        # 1. Required environment variables
        required_env = [
            "OANDA_API_KEY",
            "OANDA_ACCOUNT_ID",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        ]
        missing = [v for v in required_env if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}"
            )

        # 2. Broker connectivity
        logger.info("Startup check: verifying OANDA connectivity...")
        if not self.oanda.connect():
            raise RuntimeError(
                "OANDA connection failed — check API key and account ID"
            )

        # 3. DB schema sanity (attempt a lightweight query on each core table)
        logger.info("Startup check: verifying DB schema...")
        for table in ("trades", "signals", "positions", "daily_stats", "lessons"):
            try:
                self.db._execute(
                    f"SELECT 1 FROM {table} LIMIT 1",
                    fetch="one",
                    commit=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"DB schema check failed for table '{table}': {e}"
                )

        logger.info("All startup checks passed")

    def start(self, scan_interval: int = 300):
        """
        Start the auto-trading loop.

        Always starts the Telegram command bot first so you can
        control the system from your phone even when scanning is off.
        """
        # Fail fast if critical requirements are not met
        try:
            self._startup_checks()
        except RuntimeError as e:
            logger.error(f"Startup check FAILED: {e}")
            self.notifier.send_system_alert(f"Startup failed: {e}")
            return

        scan_interval = self.settings.toggles.scan_interval_seconds or scan_interval

        # Connection already verified by _startup_checks()
        balance = self.oanda.get_account_balance()
        self.scanner = ForexScanner(self.oanda, db=self.db)

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

        is_practice = self.settings.toggles.demo_mode or self.settings.toggles.paper_trading
        self.notifier.send_startup(balance, dry_run=is_practice)
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
        block_reasons = []
        # Always allow up to 10 trades per cycle and 10 open positions
        # We have other guards (drawdown, drift) to protect us
        max_trades = 10
        max_open = 10

        for signal in signals:
            approval = self.guard_engine.evaluate(
                signal=signal,
                balance=balance,
                open_trade_count=self.position_mgr.open_count,
                max_open=max_open,
            )

            if not approval.approved:
                block_reason = approval.summary()
                logger.info(f"BLOCKED: {signal['symbol']} — {block_reason}")
                # Send block reason to Telegram so user can see WHY
                self.notifier.send_system_alert(
                    f"BLOCKED: {signal['symbol']} {signal['side'].upper()}\n"
                    f"Confidence: {signal['confidence']:.0%}\n"
                    f"{block_reason}"
                )
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

            # Position sizing
            units = approval.allowed_units

            # Allow up to 2 positions per symbol; block 3rd+
            same_symbol_count = sum(
                1 for t in self.position_mgr.open_trades.values()
                if t["symbol"] == signal["symbol"]
            )
            if same_symbol_count >= 2:
                logger.info(f"SKIP: Already have {same_symbol_count} positions on {signal['symbol']}")
                self.notifier.send_system_alert(
                    f"SKIP: {signal['symbol']} — already have {same_symbol_count} open positions (max 2)"
                )
                trades_blocked += 1
                continue

            # Execute the trade
            result = self.oanda.place_order_with_stops(
                symbol=signal["symbol"],
                side=signal["side"],
                quantity=units,
                stop_loss_pips=signal["sl_pips"],
                take_profit_pips=signal["tp_pips"],
            )

            if not result:
                logger.error(f"Order placement returned None for {signal['symbol']}")
                self.notifier.send_system_alert(f"ORDER FAILED: {signal['symbol']} — no response from broker")
                trades_blocked += 1
                continue

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

    def _get_git_sha(self) -> str:
        """Get current git SHA for version tracking."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def get_status(self) -> str:
        """Get current auto trader status with config snapshot."""
        balance = self.oanda.get_account_balance() if self.oanda.connected else 0
        session = get_current_session()
        t = self.settings.toggles

        # Combine DB + in-memory stats (use whichever is higher since DB resets on deploy)
        db_summary = self.db.get_performance_summary() or {}
        db_wins = db_summary.get("wins", 0) or 0
        db_losses = db_summary.get("losses", 0) or 0
        mem_wins = self._stats["wins"]
        mem_losses = self._stats["losses"]
        # Use the higher count (DB might be from older session, memory from this session)
        total_wins = max(db_wins, mem_wins)
        total_losses = max(db_losses, mem_losses)
        total_closed = total_wins + total_losses
        win_rate = (total_wins / total_closed * 100) if total_closed > 0 else 0
        total_pnl = self._stats["total_pnl"] or db_summary.get("total_pnl", 0) or 0

        # Get open positions from broker (ground truth)
        open_positions = []
        try:
            open_positions = self.oanda.get_positions() or []
        except Exception:
            pass
        open_pnl = sum(getattr(p, "pnl", 0) for p in open_positions)

        total_trades = total_closed + len(open_positions)

        lines = [
            "══ TRADING BOT STATUS ══",
            "",
            f"  Balance:       ${balance:,.2f}",
            "",
            f"  ── TRADES ──",
            f"  Open:          {len(open_positions)}",
            f"  Wins:          {total_wins}",
            f"  Losses:        {total_losses}",
            f"  Total:         {total_trades}",
            f"  Win Rate:      {win_rate:.0f}%",
            f"  Closed PnL:    ${total_pnl:+,.2f}",
            f"  Open PnL:      ${open_pnl:+,.2f}",
            "",
        ]

        # Show open positions
        if open_positions:
            lines.append("  ── OPEN POSITIONS ──")
            for p in open_positions:
                pair = p.symbol.replace("_", "/")
                side = "Long" if p.quantity > 0 else "Short"
                lines.append(
                    f"    {pair} {side} {abs(p.quantity)}"
                    f" @ {p.entry_price:.5f}"
                    f" PnL: ${p.pnl:+,.2f}"
                )
            lines.append("")

        # Lessons learned
        try:
            lessons = self.db.get_lessons(limit=100)
            lesson_count = len(lessons) if lessons else 0
            lines.append(f"  Lessons:       {lesson_count}")
        except Exception:
            pass

        lines.append(f"  Forex Cycles:  {self._stats['cycles']}")
        lines.append(f"  Blocked:       {self._stats.get('blocked_signals', 0)}")
        lines.append("")

        # ── OPTIONS SECTION ──
        if self.options_trader:
            opt = self.options_trader
            opt_stats = opt._stats
            lines.extend([
                "  ── OPTIONS (ALPACA) ──",
                f"  Open:          {len(opt.open_trades)}/{opt.max_open_positions}",
                f"  Trades:        {opt_stats.get('total_trades', 0)}",
                f"  Wins:          {opt_stats.get('wins', 0)}",
                f"  Losses:        {opt_stats.get('losses', 0)}",
                f"  PnL:           ${opt_stats.get('total_pnl', 0):+,.2f}",
                f"  Cycles:        {opt_stats.get('cycles', 0)}",
                "",
            ])

        lines.append(f"  Session:       {session.primary.replace('_', ' ').title()}")
        lines.append(f"  Scanning:      {'ON' if t.scanning_enabled else 'OFF'}")
        lines.append(f"  Auto-Trading:  {'ON' if t.auto_trading_enabled else 'OFF'}")

        return "\n".join(lines)

    def get_full_report(self) -> str:
        """Get comprehensive report including DB stats."""
        lines = [self.get_status(), ""]

        # Lessons learned this session
        try:
            lessons = self.db.get_lessons(limit=100)
            if lessons:
                lines.append("  ── LESSONS LEARNED ──")
                for l in lessons[:5]:
                    lines.append(f"  [{l.get('category', '?')}] {l.get('description', '')[:80]}")
                if len(lessons) > 5:
                    lines.append(f"  ... and {len(lessons) - 5} more")
                lines.append("")
        except Exception:
            pass

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
