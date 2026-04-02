"""
Telegram Bot — Control the entire trading system from your phone.

Commands you can send in Telegram:
    /start      — Welcome message with all commands
    /status     — Current bot status, balance, wins/losses
    /scan       — Toggle scanning on/off
    /trade      — Toggle auto-trading on/off
    /settings   — View all current settings
    /report     — Full performance report
    /balance    — Current OANDA balance
    /positions  — View open positions
    /history    — Recent trade history
    /lessons    — Lessons learned from losses
    /drawdown   — Drawdown guard status
    /drift      — Drift detector status
    /exposure   — Portfolio currency exposure
    /pause      — Pause everything
    /resume     — Resume scanning + trading
    /guards     — Status of all protection guards
    /help       — Show all commands

Runs as a background thread alongside the auto-trader.
Uses Telegram's getUpdates long polling (no webhook needed).
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional, Callable

import requests

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Listens for commands on Telegram and controls the trading system.

    Runs in a background thread using long polling.
    """

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._last_update_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # These get set by the auto_trader after init
        self._options_trader = None
        self._settings = None
        self._oanda = None
        self._scanner = None
        self._drawdown = None
        self._drift = None
        self._portfolio = None
        self._calibration = None
        self._db = None
        self._loss_analyzer = None
        self._get_status_fn: Optional[Callable] = None
        self._get_report_fn: Optional[Callable] = None
        self._heartbeat_enabled = True
        self._last_scan_time: Optional[datetime] = None
        self._guard_engine = None

        if not self.enabled:
            logger.warning("Telegram bot not configured")

    def connect(
        self,
        settings,
        oanda=None,
        scanner=None,
        drawdown=None,
        drift=None,
        portfolio=None,
        calibration=None,
        db=None,
        loss_analyzer=None,
        get_status_fn=None,
        get_report_fn=None,
    ):
        """Connect all system components so the bot can control them."""
        self._settings = settings
        self._oanda = oanda
        self._scanner = scanner
        self._drawdown = drawdown
        self._drift = drift
        self._portfolio = portfolio
        self._calibration = calibration
        self._db = db
        self._loss_analyzer = loss_analyzer
        self._get_status_fn = get_status_fn
        self._get_report_fn = get_report_fn

    def connect_guard_engine(self, guard_engine):
        """Connect the guard engine for heartbeat reporting."""
        self._guard_engine = guard_engine

    def start(self):
        """Start listening for commands in a background thread."""
        if not self.enabled:
            logger.info("Telegram bot disabled — no token/chat_id")
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Telegram command bot started")
        self._send("Bot command listener started. Send /help for commands.")

    def stop(self):
        """Stop the bot."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        """Long-poll Telegram for new messages."""
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                logger.error(f"Telegram bot error: {e}")

            time.sleep(1)  # Small delay between polls

    def _get_updates(self) -> list:
        """Fetch new messages from Telegram."""
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                params={
                    "offset": self._last_update_id + 1,
                    "timeout": 10,  # Long poll for 10 seconds
                    "allowed_updates": '["message"]',
                },
                timeout=15,
            )
            if resp.ok:
                data = resp.json()
                return data.get("result", [])
        except requests.RequestException:
            pass
        return []

    def _handle_update(self, update: dict):
        """Process a single update (message)."""
        self._last_update_id = update.get("update_id", self._last_update_id)

        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "").strip()

        # Only respond to our chat
        if chat_id != self.chat_id:
            return

        if not text.startswith("/"):
            return

        # Parse command
        parts = text.split()
        cmd = parts[0].lower().split("@")[0]  # Remove @botname suffix
        args = parts[1:] if len(parts) > 1 else []

        logger.info(f"Telegram command: {cmd} {args}")

        # Route command
        handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/scan": self._cmd_scan,
            "/trade": self._cmd_trade,
            "/settings": self._cmd_settings,
            "/report": self._cmd_report,
            "/balance": self._cmd_balance,
            "/positions": self._cmd_positions,
            "/history": self._cmd_history,
            "/lessons": self._cmd_lessons,
            "/drawdown": self._cmd_drawdown,
            "/drift": self._cmd_drift,
            "/exposure": self._cmd_exposure,
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
            "/guards": self._cmd_guards,
            "/set": self._cmd_set,
            "/crypto": self._cmd_crypto,
            "/kill": self._cmd_kill,
            "/safe": self._cmd_safe,
            "/mode": self._cmd_mode,
            "/session": self._cmd_session,
            "/why": self._cmd_why,
            "/options": self._cmd_options,
            "/testoptions": self._cmd_testoptions,
            "/optionscan": self._cmd_optionscan,
            "/optiontrade": self._cmd_optiontrade,
            "/optionpositions": self._cmd_optionpositions,
            "/optionrisk": self._cmd_optionrisk,
            "/heartbeat": self._cmd_heartbeat,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                handler(args)
            except Exception as e:
                self._send(f"Error: {str(e)[:200]}")
        else:
            self._send(f"Unknown command: {cmd}\nSend /help for all commands.")

    # ─── COMMANDS ───

    def _cmd_start(self, args):
        self._send(
            "<b>Trading Bot v2</b>\n\n"
            "Control everything from here.\n"
            "Send /help to see all commands."
        )

    def _cmd_help(self, args):
        self._send(
            "<b>COMMANDS</b>\n\n"
            "<b>Info:</b>\n"
            "/status — Bot status, balance, stats\n"
            "/balance — OANDA account balance\n"
            "/positions — Open positions\n"
            "/history — Recent trade history\n"
            "/report — Full performance report\n"
            "/lessons — Lessons from losses\n"
            "/session — Current trading session\n"
            "/why — Why last signal was blocked\n\n"
            "<b>Controls:</b>\n"
            "/scan — Toggle scanning ON/OFF\n"
            "/trade — Toggle auto-trading ON/OFF\n"
            "/crypto — Scan crypto NOW (works weekends)\n"
            "/pause — Pause everything\n"
            "/resume — Resume all\n"
            "/set key value — Change a setting\n"
            "/mode [dev|paper|practice|live] — Switch profile\n"
            "/options — Options trader status\n\n"
            "<b>Options:</b>\n"
            "/optionscan — Force options scan now\n"
            "/optiontrade — Options trading stats\n"
            "/optionpositions — Open options positions + P&L\n"
            "/optionrisk — View/adjust options risk settings\n"
            "/testoptions — Diagnostic: test Alpaca data\n\n"
            "<b>Emergency:</b>\n"
            "/kill — STOP everything immediately\n"
            "/safe — Safe mode (close all, stop trading)\n\n"
            "<b>Guards:</b>\n"
            "/guards — All guard statuses\n"
            "/drawdown — Drawdown guard\n"
            "/drift — Drift detector\n"
            "/exposure — Portfolio exposure\n"
            "/settings — All current settings\n"
            "/heartbeat — Toggle health heartbeat ON/OFF"
        )

    def _cmd_status(self, args):
        if self._get_status_fn:
            status = self._get_status_fn()
            # Truncate for Telegram (4096 char limit)
            self._send(f"<pre>{status[:3900]}</pre>")
        else:
            self._send("Status not available — bot initializing")

    def _cmd_scan(self, args):
        if not self._settings:
            self._send("Settings not available")
            return

        if args and args[0].lower() in ("on", "true", "1"):
            self._settings.set("scanning_enabled", True)
            self._send("Scanning: <b>ON</b>")
        elif args and args[0].lower() in ("off", "false", "0"):
            self._settings.set("scanning_enabled", False)
            self._send("Scanning: <b>OFF</b>")
        else:
            # Toggle
            current = self._settings.toggles.scanning_enabled
            self._settings.set("scanning_enabled", not current)
            state = "ON" if not current else "OFF"
            self._send(f"Scanning: <b>{state}</b>")

    def _cmd_trade(self, args):
        if not self._settings:
            self._send("Settings not available")
            return

        if args and args[0].lower() in ("on", "true", "1"):
            self._settings.set("auto_trading_enabled", True)
            self._send("Auto-Trading: <b>ON</b>\nBot will now place real trades on OANDA.")
        elif args and args[0].lower() in ("off", "false", "0"):
            self._settings.set("auto_trading_enabled", False)
            self._send("Auto-Trading: <b>OFF</b>\nAlert-only mode.")
        else:
            current = self._settings.toggles.auto_trading_enabled
            self._settings.set("auto_trading_enabled", not current)
            state = "ON" if not current else "OFF"
            self._send(f"Auto-Trading: <b>{state}</b>")

    def _cmd_settings(self, args):
        if not self._settings:
            self._send("Settings not available")
            return
        status = self._settings.get_status()
        self._send(f"<pre>{status[:3900]}</pre>")

    def _cmd_report(self, args):
        if self._get_report_fn:
            report = self._get_report_fn()
            self._send(f"<pre>{report[:3900]}</pre>")
        elif self._db:
            summary = self._db.get_performance_summary()
            if summary:
                self._send(
                    f"<b>PERFORMANCE</b>\n\n"
                    f"Trades: {summary.get('total_trades', 0)}\n"
                    f"Win Rate: {summary.get('win_rate', 0):.0%}\n"
                    f"Total PnL: ${summary.get('total_pnl', 0):+,.2f}"
                )
            else:
                self._send("No trade data yet.")
        else:
            self._send("Report not available")

    def _cmd_balance(self, args):
        if self._oanda and self._oanda.connected:
            balance = self._oanda.get_account_balance()
            self._send(f"<b>OANDA Balance:</b> ${balance:,.2f}")
        else:
            self._send("OANDA not connected")

    def _cmd_positions(self, args):
        if not self._oanda or not self._oanda.connected:
            self._send("OANDA not connected")
            return

        positions = self._oanda.get_positions()
        if not positions:
            self._send("No open positions")
            return

        lines = ["<b>OPEN POSITIONS</b>\n"]
        for p in positions:
            pair = p.symbol.replace("_", "/")
            side = "Long" if p.quantity > 0 else "Short"
            lines.append(
                f"{pair}: <b>{side}</b> {abs(p.quantity)} units "
                f"@ {p.entry_price:.5f} (PnL: ${p.pnl:+,.2f})"
            )
        self._send("\n".join(lines))

    def _cmd_history(self, args):
        if not self._db:
            self._send("Database not available")
            return

        trades = self._db.get_all_trades()
        recent = trades[-10:] if trades else []

        if not recent:
            self._send("No trade history yet")
            return

        lines = ["<b>RECENT TRADES</b>\n"]
        for t in reversed(recent):
            symbol = t.get("symbol", "?")
            outcome = t.get("outcome", "open").upper()
            pnl = t.get("pnl", 0)
            side = t.get("side", "?")
            lines.append(
                f"{'W' if outcome == 'WIN' else 'L' if outcome == 'LOSS' else 'O'} "
                f"{symbol} {side} ${pnl:+,.2f}"
            )
        self._send("\n".join(lines))

    def _cmd_lessons(self, args):
        # Try loss analyzer first (JSON file)
        if self._loss_analyzer:
            summary = self._loss_analyzer.get_lessons_summary()
            if summary and summary != "No lessons learned yet.":
                self._send(f"<pre>{summary[:3900]}</pre>")
                return

        # Fallback: read from DB
        if self._db:
            lessons = self._db.get_lessons(limit=20)
            if lessons:
                lines = [f"LESSONS FROM DB ({len(lessons)} total)\n"]
                for l in lessons[:10]:
                    lines.append(
                        f"[{l.get('category', '?')}] {l.get('description', '')[:100]}\n"
                        f"  Rule: {l.get('rule_added', 'N/A')}\n"
                    )
                self._send(f"<pre>{chr(10).join(lines)[:3900]}</pre>")
            else:
                self._send("No lessons learned yet. Lessons are recorded after each losing trade.")
        else:
            self._send("Loss analyzer not available")

    def _cmd_drawdown(self, args):
        if not self._drawdown:
            self._send("Drawdown guard not initialized")
            return

        # /drawdown reset — reset peak to current balance
        if args and args[0].lower() == "reset":
            if self._oanda and self._oanda.connected:
                balance = self._oanda.get_account_balance()
                self._drawdown.reset_peak(balance)
                self._send(
                    f"<b>DRAWDOWN RESET</b>\n\n"
                    f"Peak reset to current balance: ${balance:,.2f}\n"
                    f"Drawdown tracking starts fresh from here."
                )
            else:
                self._send("Cannot reset — OANDA not connected")
            return

        self._send(f"<pre>{self._drawdown.get_status()}</pre>")

    def _cmd_drift(self, args):
        if self._drift:
            self._send(f"<pre>{self._drift.get_status()}</pre>")
        else:
            self._send("Drift detector not initialized")

    def _cmd_exposure(self, args):
        if not self._portfolio:
            self._send("Portfolio manager not initialized")
            return

        report = self._portfolio.get_exposure_report()
        lines = ["<b>PORTFOLIO EXPOSURE</b>\n"]
        lines.append(f"Positions: {report.get('total_positions', 0)}")
        lines.append(f"Total Exposure: {report.get('total_exposure_pct', 0):.1f}%")

        currency_exp = report.get("currency_exposure", {})
        if currency_exp:
            lines.append("\nPer Currency:")
            for curr, exp in sorted(currency_exp.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(exp) > 0.1:
                    lines.append(f"  {curr}: {exp:+,.0f} units")

        self._send("\n".join(lines))

    def _cmd_pause(self, args):
        if self._settings:
            self._settings.set("scanning_enabled", False)
            self._settings.set("auto_trading_enabled", False)
            self._send(
                "<b>PAUSED</b>\n\n"
                "Scanning: OFF\n"
                "Auto-Trading: OFF\n\n"
                "Send /resume to restart."
            )
        else:
            self._send("Settings not available")

    def _cmd_resume(self, args):
        if self._settings:
            self._settings.set("scanning_enabled", True)
            self._settings.set("auto_trading_enabled", True)
            self._send(
                "<b>RESUMED</b>\n\n"
                "Scanning: ON\n"
                "Auto-Trading: ON\n\n"
                "Bot is now scanning and trading."
            )
        else:
            self._send("Settings not available")

    def _cmd_guards(self, args):
        lines = ["<b>GUARD STATUS</b>\n"]

        # Drawdown
        if self._drawdown:
            can, reason = self._drawdown.can_trade()
            lines.append(f"Drawdown: {'OK' if can else 'BLOCKED'}")
            if not can:
                lines.append(f"  {reason}")

        # Drift
        if self._drift:
            drifting, reason = self._drift.is_drifting()
            lines.append(f"Drift: {'DRIFTING' if drifting else 'OK'}")
            if drifting:
                lines.append(f"  {reason}")

        # Calibration
        if self._calibration:
            overconf = self._calibration.is_overconfident()
            lines.append(f"Calibration: {'OVERCONFIDENT' if overconf else 'OK'}")

        # Portfolio
        if self._portfolio:
            report = self._portfolio.get_exposure_report()
            lines.append(f"Portfolio: {report.get('total_positions', 0)} positions, "
                        f"{report.get('total_exposure_pct', 0):.0f}% exposed")

        # Settings guards
        if self._settings:
            t = self._settings.toggles
            lines.append(f"\nGuards enabled:")
            lines.append(f"  Drawdown:  {'ON' if t.drawdown_guard_enabled else 'OFF'}")
            lines.append(f"  Drift:     {'ON' if t.drift_detector_enabled else 'OFF'}")
            lines.append(f"  Portfolio: {'ON' if t.portfolio_manager_enabled else 'OFF'}")
            lines.append(f"  Slippage:  {'ON' if t.slippage_model_enabled else 'OFF'}")
            lines.append(f"  Quality:   {'ON' if t.data_quality_check else 'OFF'}")
            lines.append(f"  Regime:    {'ON' if t.regime_detection_enabled else 'OFF'}")
            lines.append(f"  Calibrate: {'ON' if t.calibration_enabled else 'OFF'}")

        self._send("\n".join(lines))

    def _cmd_crypto(self, args):
        """Run a crypto scan right now (works on weekends)."""
        if not self._scanner:
            self._send("Scanner not initialized. Is OANDA connected?")
            return

        self._send("Scanning crypto pairs... (BTC, ETH, LTC, XRP...)")

        try:
            from trading.brokers.oanda import CRYPTO_PAIRS

            signals = []
            for symbol in CRYPTO_PAIRS:
                try:
                    signal = self._scanner._analyze_pair(symbol)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
                    continue

            signals.sort(key=lambda x: x["confidence"], reverse=True)

            if not signals:
                self._send(
                    "<b>CRYPTO SCAN</b>\n\n"
                    f"Scanned {len(CRYPTO_PAIRS)} crypto pairs.\n"
                    "No signals found right now.\n\n"
                    "This means no pair hit the minimum confidence threshold."
                )
                return

            lines = [f"<b>CRYPTO SCAN — {len(signals)} signals</b>\n"]
            for s in signals[:5]:
                pair = s["symbol"].replace("_", "/")
                side = "BUY" if s["side"] == "buy" else "SELL"
                lines.append(
                    f"\n<b>{pair} — {side}</b>\n"
                    f"Confidence: {s['confidence']*100:.0f}%\n"
                    f"Entry: {s['entry']:.2f}\n"
                    f"SL: {s['stop_loss']:.2f} | TP: {s['take_profit']:.2f}\n"
                    f"{s['reasoning'][:150]}"
                )

            if self._settings and self._settings.toggles.auto_trading_enabled:
                lines.append("\nAuto-trading is ON — trades will be placed automatically.")
            else:
                lines.append("\nAuto-trading is OFF — these are alerts only.\nSend /trade on to enable.")

            self._send("\n".join(lines))

        except Exception as e:
            self._send(f"Crypto scan error: {str(e)[:200]}")

    def _cmd_set(self, args):
        """Set a specific setting. Usage: /set key value"""
        if not self._settings:
            self._send("Settings not available")
            return

        if len(args) < 2:
            self._send("Usage: /set key value\nExample: /set max_trades_per_cycle 5")
            return

        key = args[0]
        value = args[1]

        # Parse value
        if value.lower() in ("true", "on", "yes", "1"):
            parsed = True
        elif value.lower() in ("false", "off", "no", "0"):
            parsed = False
        elif value.replace(".", "").replace("-", "").isdigit():
            parsed = int(value) if "." not in value else float(value)
        else:
            parsed = value

        if self._settings.set(key, parsed):
            self._send(f"Set <b>{key}</b> = <b>{parsed}</b>")
        else:
            self._send(f"Unknown setting: {key}")

    # ─── EMERGENCY & PROFILE COMMANDS ───

    def _cmd_kill(self, args):
        """Emergency stop — disable everything immediately."""
        if self._settings:
            self._settings.set("scanning_enabled", False)
            self._settings.set("auto_trading_enabled", False)
            self._settings.set("paper_trading", True)
            self._send(
                "<b>EMERGENCY STOP</b>\n\n"
                "Scanning: OFF\n"
                "Auto-Trading: OFF\n"
                "Paper Trading: ON\n\n"
                "All trading halted. Open positions remain until manually closed.\n"
                "Send /resume to restart."
            )
        else:
            self._send("Settings not available")

    def _cmd_safe(self, args):
        """Safe mode — close all positions and stop trading."""
        if self._settings:
            self._settings.set("scanning_enabled", False)
            self._settings.set("auto_trading_enabled", False)

        # Try to close all positions
        closed = 0
        if self._oanda and self._oanda.connected:
            try:
                positions = self._oanda.get_positions()
                for p in positions:
                    try:
                        self._oanda.close_position(p.symbol)
                        closed += 1
                    except Exception as e:
                        logger.error(f"Failed to close {p.symbol}: {e}")
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")

        self._send(
            f"<b>SAFE MODE ACTIVATED</b>\n\n"
            f"Closed {closed} position(s).\n"
            f"Scanning: OFF\n"
            f"Auto-Trading: OFF\n\n"
            f"System is now in safe mode.\n"
            f"Send /resume to restart."
        )

    def _cmd_mode(self, args):
        """Switch runtime profile."""
        if not self._settings:
            self._send("Settings not available")
            return

        if not args:
            current = self._settings.toggles.runtime_profile
            self._send(
                f"Current profile: <b>{current}</b>\n\n"
                f"Available profiles:\n"
                f"  <b>dev</b> — All guards off, no trading\n"
                f"  <b>paper</b> — Paper trading, minimal guards\n"
                f"  <b>practice</b> — OANDA practice, all guards on\n"
                f"  <b>live</b> — REAL MONEY, all guards on\n\n"
                f"Usage: /mode practice"
            )
            return

        profile = args[0].lower()
        if profile == "live":
            self._send(
                "<b>WARNING: LIVE MODE</b>\n\n"
                "This will trade with REAL MONEY.\n"
                "Send /mode live_confirm to proceed."
            )
            return

        if profile == "live_confirm":
            profile = "live"

        if self._settings.apply_profile(profile):
            self._send(f"Switched to <b>{profile}</b> profile.\nSend /settings to see changes.")
        else:
            self._send(f"Unknown profile: {profile}\nOptions: dev, paper, practice, live")

    def _cmd_session(self, args):
        """Show current trading session info."""
        from trading.session_awareness import get_session_status
        self._send(f"<pre>{get_session_status()}</pre>")

    def _cmd_options(self, args):
        """Show options trader status."""
        if self._options_trader:
            status = self._options_trader.get_status()
            self._send(f"<pre>{status}</pre>")
        else:
            self._send(
                "Options module not active.\n\n"
                "To enable:\n"
                "1. Get Alpaca API key (alpaca.markets)\n"
                "2. Set ALPACA_API_KEY and ALPACA_SECRET_KEY\n"
                "3. Redeploy the bot\n\n"
                "Symbols: SPY, QQQ, AAPL, MSFT, NVDA\n"
                "Hours: Mon-Fri 9:45 AM - 3:45 PM ET"
            )

    def _cmd_testoptions(self, args):
        """Test Alpaca connection and options data flow."""
        if not self._options_trader:
            self._send("Options trader not loaded. Check ALPACA_API_KEY env var.")
            return

        lines = ["<b>OPTIONS DIAGNOSTIC</b>\n"]

        # 1. Connection
        broker = self._options_trader.broker
        if not broker.connected:
            connected = broker.connect()
            lines.append(f"Alpaca connect: {'OK' if connected else 'FAILED'}")
            if not connected:
                self._send("\n".join(lines))
                return
        else:
            lines.append("Alpaca connect: OK (already connected)")

        # 2. Balance
        balance = broker.get_account_balance()
        lines.append(f"Balance: ${balance:,.2f}")

        # 3. Market status
        market_open = broker.is_market_open()
        lines.append(f"Market open: {'YES' if market_open else 'NO'}")

        from trading.options_scanner import is_options_market_open
        opts_open = is_options_market_open()
        lines.append(f"Options hours: {'YES' if opts_open else 'NO (9:45-3:45 ET Mon-Fri)'}")

        # 4. Test candle data for SPY
        for symbol in ["SPY", "QQQ"]:
            candles_d1 = broker.get_candles(symbol, "1Day", 5)
            candles_h1 = broker.get_candles(symbol, "1Hour", 5)
            candles_h4 = broker.get_candles(symbol, "4Hour", 5)
            lines.append(f"\n{symbol} candles:")
            lines.append(f"  D1: {len(candles_d1)} bars")
            lines.append(f"  H4: {len(candles_h4)} bars")
            lines.append(f"  H1: {len(candles_h1)} bars")
            if candles_d1:
                last = candles_d1[-1]
                lines.append(f"  Last D1: O={last['open']:.2f} H={last['high']:.2f} L={last['low']:.2f} C={last['close']:.2f}")

        # 5. Test quote
        quote = broker.get_quote("SPY")
        if quote:
            lines.append(f"\nSPY quote: bid={quote.bid:.2f} ask={quote.ask:.2f} mid={quote.mid:.2f}")
        else:
            lines.append("\nSPY quote: FAILED")

        # 6. Test options chain
        chain = broker.get_options_chain("SPY")
        if chain:
            with_quotes = sum(1 for c in chain if c.bid > 0)
            lines.append(f"SPY options chain: {len(chain)} contracts, {with_quotes} with live quotes")
        else:
            lines.append("SPY options chain: EMPTY or FAILED")

        # 7. Cycles
        lines.append(f"\nOptions cycles run: {self._options_trader._stats.get('cycles', 0)}")
        lines.append(f"Options trades: {self._options_trader._stats.get('total_trades', 0)}")

        self._send("\n".join(lines))

    def _cmd_optionscan(self, args):
        """Force an options scan or show last scan results."""
        if not self._options_trader:
            self._send("Options trader not active. Set ALPACA_API_KEY to enable.")
            return

        broker = self._options_trader.broker
        if not broker.connected:
            if not broker.connect():
                self._send("Cannot connect to Alpaca.")
                return

        from trading.options_scanner import OptionsScanner, is_options_market_open

        if not is_options_market_open():
            self._send(
                "<b>OPTIONS SCAN</b>\n\n"
                "Market is closed right now.\n"
                "Options hours: Mon-Fri 9:45 AM - 3:45 PM ET\n\n"
                "Scanning anyway for diagnostics..."
            )

        self._send("Scanning options... (SPY, QQQ, AAPL, MSFT, NVDA)")

        try:
            scanner = self._options_trader.scanner or OptionsScanner(broker)
            signals = scanner.scan_all()

            if not signals:
                self._send(
                    "<b>OPTIONS SCAN — 0 signals</b>\n\n"
                    "No setups passed confidence threshold.\n"
                    "Thresholds: Momentum 0.40 | Swing 0.35\n\n"
                    "Use /testoptions to check data quality."
                )
                return

            lines = [f"<b>OPTIONS SCAN — {len(signals)} signal(s)</b>\n"]
            for s in signals[:5]:
                direction = "CALL" if s["side"] == "buy" else "PUT"
                contract = s.get("contract")
                strike = f"${contract.strike:.0f}" if contract else "?"
                exp = contract.expiration if contract else "?"
                premium = f"${contract.mid:.2f}" if contract else "?"

                lines.append(
                    f"\n<b>{s['symbol']} {direction} {strike} exp {exp}</b>\n"
                    f"Mode: {s.get('mode', '?').upper()} | "
                    f"Tier: {'SPY/QQQ' if s.get('tier') == 1 else 'Single'}\n"
                    f"Confidence: <b>{s['confidence']:.0%}</b>\n"
                    f"Premium: {premium}\n"
                    f"{s.get('reasoning', '')[:200]}"
                )

            open_count = len(self._options_trader.open_trades)
            max_pos = self._options_trader.max_open_positions
            lines.append(f"\nOpen: {open_count}/{max_pos}")

            if open_count >= max_pos:
                lines.append("Max positions reached — no new trades until one closes.")

            self._send("\n".join(lines))
        except Exception as e:
            self._send(f"Options scan error: {str(e)[:300]}")

    def _cmd_optiontrade(self, args):
        """Toggle options auto-trading on/off or show status."""
        if not self._options_trader:
            self._send("Options trader not active. Set ALPACA_API_KEY to enable.")
            return

        if not args:
            # Show current state
            stats = self._options_trader._stats
            total = stats["wins"] + stats["losses"]
            win_rate = (stats["wins"] / total * 100) if total > 0 else 0
            open_count = len(self._options_trader.open_trades)

            self._send(
                f"<b>OPTIONS TRADING</b>\n\n"
                f"Total trades: {stats['total_trades']}\n"
                f"Momentum: {stats.get('momentum_trades', 0)} | "
                f"Swing: {stats.get('swing_trades', 0)}\n"
                f"W/L: {stats['wins']}/{stats['losses']} "
                f"({win_rate:.0f}%)\n"
                f"PnL: <b>${stats['total_pnl']:+,.2f}</b>\n"
                f"Open: {open_count}/{self._options_trader.max_open_positions}\n"
                f"Cycles: {stats.get('cycles', 0)}\n\n"
                f"Max premium: ${self._options_trader.max_premium_per_trade:.0f}\n"
                f"Max positions: {self._options_trader.max_open_positions}"
            )
            return

        action = args[0].lower()
        if action in ("on", "true", "1", "enable"):
            self._send(
                "<b>OPTIONS TRADING: ON</b>\n\n"
                "Options auto-trading is controlled by the options thread.\n"
                "It scans every 5 minutes during market hours.\n"
                "Use /optionscan to force a scan now."
            )
        elif action in ("off", "false", "0", "disable"):
            self._send(
                "<b>OPTIONS TRADING</b>\n\n"
                "Options run on a separate thread and can't be paused individually yet.\n"
                "Use /kill to stop ALL trading, or /safe to close all positions."
            )
        else:
            self._send("Usage: /optiontrade [on|off]")

    def _cmd_optionpositions(self, args):
        """Show open options positions with live P&L."""
        if not self._options_trader:
            self._send("Options trader not active. Set ALPACA_API_KEY to enable.")
            return

        open_trades = self._options_trader.open_trades
        if not open_trades:
            self._send(
                "<b>OPTIONS POSITIONS</b>\n\n"
                "No open options positions.\n\n"
                f"Total trades: {self._options_trader._stats['total_trades']}\n"
                f"PnL: ${self._options_trader._stats['total_pnl']:+,.2f}"
            )
            return

        lines = [f"<b>OPTIONS POSITIONS ({len(open_trades)})</b>\n"]

        broker = self._options_trader.broker
        positions = {}
        try:
            pos_list = broker.get_positions()
            positions = {p.symbol: p for p in pos_list}
        except Exception:
            pass

        for tid, info in open_trades.items():
            opt_sym = info.get("option_symbol", "?")
            direction = info.get("option_type", "?").upper()
            mode = info.get("mode", "?").upper()
            entry_premium = info.get("entry_premium", 0)

            # Get live price if available
            pos = positions.get(opt_sym)
            if pos:
                current = pos.current_price
                pnl_pct = ((current - entry_premium) / entry_premium * 100) if entry_premium > 0 else 0
                pnl_dollar = (current - entry_premium) * 100
            else:
                current = entry_premium
                pnl_pct = 0
                pnl_dollar = 0

            # Time held
            try:
                placed = datetime.fromisoformat(info["placed_at"].replace("Z", "+00:00"))
                hours_held = (datetime.now(placed.tzinfo or None) - placed).total_seconds() / 3600
                time_str = f"{hours_held:.1f}h"
            except Exception:
                time_str = "?"

            rules = self._options_trader.exit_rules.get(info.get("mode", "swing"), {})
            partial = "YES" if info.get("partial_taken") else "no"

            lines.append(
                f"\n<b>{info['symbol']} {direction} ${info.get('strike', '?')}</b>\n"
                f"Mode: {mode} | Exp: {info.get('expiration', '?')} ({info.get('dte', '?')} DTE)\n"
                f"Entry: ${entry_premium:.2f} | Now: ${current:.2f}\n"
                f"P&L: <b>{'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%</b> (${pnl_dollar:+.2f})\n"
                f"Held: {time_str} | Partial TP: {partial}\n"
                f"TP: {rules.get('tp_pct', 0):.0%} | SL: {rules.get('sl_pct', 0):.0%} | "
                f"Time stop: {rules.get('time_stop_hours', '?')}h\n"
                f"Confidence: {info.get('confidence', 0):.0%}"
            )

        self._send("\n".join(lines))

    def _cmd_optionrisk(self, args):
        """Show or adjust options risk settings."""
        if not self._options_trader:
            self._send("Options trader not active. Set ALPACA_API_KEY to enable.")
            return

        trader = self._options_trader

        if args and len(args) >= 2:
            key = args[0].lower()
            try:
                value = float(args[1])
            except ValueError:
                self._send(f"Invalid value: {args[1]}")
                return

            if key == "maxpremium":
                trader.max_premium_per_trade = value
                trader._save_state()
                self._send(f"Max premium per trade set to <b>${value:.0f}</b>")
            elif key == "maxpositions":
                trader.max_open_positions = int(value)
                trader._save_state()
                self._send(f"Max open positions set to <b>{int(value)}</b>")
            elif key == "tp_momentum":
                trader.exit_rules["momentum"]["tp_pct"] = value / 100
                self._send(f"Momentum TP set to <b>{value:.0f}%</b>")
            elif key == "sl_momentum":
                trader.exit_rules["momentum"]["sl_pct"] = value / 100
                self._send(f"Momentum SL set to <b>{value:.0f}%</b>")
            elif key == "tp_swing":
                trader.exit_rules["swing"]["tp_pct"] = value / 100
                self._send(f"Swing TP set to <b>{value:.0f}%</b>")
            elif key == "sl_swing":
                trader.exit_rules["swing"]["sl_pct"] = value / 100
                self._send(f"Swing SL set to <b>{value:.0f}%</b>")
            else:
                self._send(
                    f"Unknown setting: {key}\n\n"
                    "Available: maxpremium, maxpositions, "
                    "tp_momentum, sl_momentum, tp_swing, sl_swing"
                )
            return

        # Show current risk settings
        from trading.options_confidence import MOMENTUM_THRESHOLD, SWING_THRESHOLD

        balance = 0
        try:
            balance = trader.broker.get_account_balance()
        except Exception:
            pass

        stats = trader._stats
        total = stats["wins"] + stats["losses"]
        win_rate = (stats["wins"] / total * 100) if total > 0 else 0

        self._send(
            f"<b>OPTIONS RISK SETTINGS</b>\n\n"
            f"<b>Account:</b>\n"
            f"Balance: ${balance:,.2f}\n"
            f"Open: {len(trader.open_trades)}/{trader.max_open_positions}\n"
            f"Max premium/trade: ${trader.max_premium_per_trade:.0f}\n\n"
            f"<b>Confidence thresholds:</b>\n"
            f"Momentum: {MOMENTUM_THRESHOLD:.0%}\n"
            f"Swing: {SWING_THRESHOLD:.0%}\n"
            f"Tier 2 bonus: +10%\n\n"
            f"<b>Momentum exits:</b>\n"
            f"TP: {trader.exit_rules['momentum']['tp_pct']:.0%} | "
            f"SL: {trader.exit_rules['momentum']['sl_pct']:.0%}\n"
            f"Partial: {trader.exit_rules['momentum']['partial_tp_pct']:.0%} | "
            f"Time stop: {trader.exit_rules['momentum']['time_stop_hours']}h\n\n"
            f"<b>Swing exits:</b>\n"
            f"TP: {trader.exit_rules['swing']['tp_pct']:.0%} | "
            f"SL: {trader.exit_rules['swing']['sl_pct']:.0%}\n"
            f"Partial: {trader.exit_rules['swing']['partial_tp_pct']:.0%} | "
            f"Time stop: {trader.exit_rules['swing']['time_stop_hours']}h\n\n"
            f"<b>Performance:</b>\n"
            f"W/L: {stats['wins']}/{stats['losses']} ({win_rate:.0f}%)\n"
            f"PnL: ${stats['total_pnl']:+,.2f}\n\n"
            f"<b>Adjust:</b>\n"
            f"/optionrisk maxpremium 300\n"
            f"/optionrisk maxpositions 5\n"
            f"/optionrisk tp_momentum 40\n"
            f"/optionrisk sl_swing 50"
        )

    def _cmd_news(self, args):
        """Show current news sentiment and headlines."""
        try:
            from trading.news_sentiment import NewsReader
            reader = NewsReader()

            # If a symbol is specified, get sentiment for that symbol
            if args:
                symbol = args[0].upper().replace("/", "_")
                sentiment = reader.get_sentiment(symbol)
                lines = [f"<b>NEWS: {symbol}</b>\n"]
                lines.append(f"Sentiment: <b>{sentiment.sentiment.upper()}</b> ({sentiment.score:+.2f})")
                lines.append(f"Headlines: {sentiment.headline_count}")
                if sentiment.top_headlines:
                    lines.append("")
                    for h in sentiment.top_headlines[:5]:
                        lines.append(f"- {h[:100]}")
                self._send("\n".join(lines))
            else:
                # General market sentiment
                report = reader.format_sentiment_report()
                self._send(f"<pre>{report[:3900]}</pre>")
        except Exception as e:
            self._send(f"News error: {str(e)[:200]}")

    def _cmd_why(self, args):
        """Show why the last signal(s) were blocked."""
        if not self._db:
            self._send("Database not available")
            return

        try:
            # Get recent signal verdicts from DB (both approved and denied)
            import sqlite3
            conn = sqlite3.connect(self._db.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT symbol, side, confidence, taken, reason_skipped, timestamp "
                "FROM signals "
                "ORDER BY timestamp DESC LIMIT 10"
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                self._send("No signal verdicts found recently.")
                return

            lines = ["<b>RECENT SIGNAL VERDICTS</b>\n"]
            for r in rows:
                verdict = "APPROVED" if r['taken'] else "BLOCKED"
                reason = r['reason_skipped'] if r['reason_skipped'] else "All guards passed"
                lines.append(
                    f"\n{'✓' if r['taken'] else '✗'} {r['symbol']} {r['side'].upper()} "
                    f"({r['confidence']*100:.0f}%) — <b>{verdict}</b>\n"
                    f"Reason: {reason}\n"
                    f"Time: {r['timestamp'][:16]}"
                )
            self._send("\n".join(lines))
        except Exception as e:
            self._send(f"Error: {str(e)[:200]}")

    # ─── HEARTBEAT ───

    def _heartbeat_loop(self):
        """Send a health heartbeat every 2 hours."""
        interval = 2 * 60 * 60  # 2 hours in seconds
        while self._running:
            time.sleep(interval)
            if not self._running or not self._heartbeat_enabled:
                continue
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    def _send_heartbeat(self):
        """Build and send the heartbeat status message."""
        lines = ["<b>HEARTBEAT</b>\n"]

        # Brokers up/down
        if self._oanda:
            oanda_ok = getattr(self._oanda, "connected", False)
            lines.append(f"OANDA: {'UP' if oanda_ok else 'DOWN'}")
        else:
            lines.append("OANDA: not configured")

        if self._options_trader:
            opt_ok = getattr(self._options_trader.broker, "connected", False)
            opt_trades = self._options_trader._stats.get("total_trades", 0)
            opt_cycles = self._options_trader._stats.get("cycles", 0)
            opt_open = len(self._options_trader.open_trades)
            lines.append(f"Options: {'UP' if opt_ok else 'DOWN'} | "
                        f"{opt_trades} trades | {opt_open} open | {opt_cycles} cycles")

        # Last scan time
        if self._last_scan_time:
            lines.append(f"Last scan: {self._last_scan_time.strftime('%H:%M:%S')}")
        else:
            lines.append("Last scan: none")

        # Open positions and PnL
        total_pnl = 0.0
        pos_count = 0
        if self._oanda and getattr(self._oanda, "connected", False):
            try:
                positions = self._oanda.get_positions()
                pos_count = len(positions) if positions else 0
                total_pnl = sum(getattr(p, "pnl", 0) for p in (positions or []))
            except Exception:
                pass
        lines.append(f"Open positions: {pos_count}")
        lines.append(f"PnL: ${total_pnl:+,.2f}")

        # Drawdown %
        if self._drawdown:
            try:
                dd_pct = getattr(self._drawdown, "current_drawdown_pct", None)
                if dd_pct is not None:
                    lines.append(f"Drawdown: {dd_pct:.1f}%")
                else:
                    can, reason = self._drawdown.can_trade()
                    lines.append(f"Drawdown: {'OK' if can else reason}")
            except Exception:
                lines.append("Drawdown: unavailable")

        # Guard states
        lines.append("")
        lines.append("<b>Guards:</b>")
        if self._drawdown:
            can, _ = self._drawdown.can_trade()
            lines.append(f"  Drawdown: {'OK' if can else 'BLOCKED'}")
        if self._drift:
            drifting, _ = self._drift.is_drifting()
            lines.append(f"  Drift: {'DRIFTING' if drifting else 'OK'}")
        if self._calibration:
            overconf = self._calibration.is_overconfident()
            lines.append(f"  Calibration: {'OVERCONFIDENT' if overconf else 'OK'}")
        if self._portfolio:
            report = self._portfolio.get_exposure_report()
            lines.append(f"  Portfolio: {report.get('total_positions', 0)} pos, "
                        f"{report.get('total_exposure_pct', 0):.0f}% exposed")

        lines.append(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._send("\n".join(lines))

    def _cmd_heartbeat(self, args):
        """Toggle the periodic heartbeat on/off."""
        if args and args[0].lower() in ("on", "true", "1"):
            self._heartbeat_enabled = True
        elif args and args[0].lower() in ("off", "false", "0"):
            self._heartbeat_enabled = False
        else:
            self._heartbeat_enabled = not self._heartbeat_enabled

        state = "ON" if self._heartbeat_enabled else "OFF"
        self._send(f"Heartbeat: <b>{state}</b> (every 2 hours)")

    # ─── SEND ───

    def _send(self, text: str):
        """Send a message to our Telegram chat."""
        if not self.enabled:
            return

        try:
            # Split long messages (Telegram limit is 4096)
            while text:
                chunk = text[:4000]
                text = text[4000:]

                requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
        except requests.RequestException as e:
            logger.warning(f"Telegram send error: {e}")
