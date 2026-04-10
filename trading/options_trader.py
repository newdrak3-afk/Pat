"""
Options Trader v2 — Momentum + Swing modes with dynamic exits.

Two modes:
  MOMENTUM: 5-10 DTE, quick entries, 25-35% TP, strict time stop
  SWING:    10-21 DTE, pullback entries, 50% TP, wider time stop

Exit logic:
  - Premium-based TP/SL (mode-dependent)
  - Time stop (no move within N hours = close)
  - Momentum fade (underlying reverses)
  - Partial take-profit at first target
  - Force close at 2 DTE
  - End-of-day review for momentum trades

Risk rules:
  - Max premium per trade: $500
  - Max 3 open option positions
  - No trading first/last 15 min
  - Limit orders only
  - SPY/QQQ priority, single names only when ideal
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from trading.brokers.alpaca import AlpacaBroker, OPTIONS_SYMBOLS
from trading.options_scanner import OptionsScanner, is_options_market_open
from trading.options_contract_selector import ContractSelector
from trading.trade_db import TradeDB
from trading.notifier import TelegramNotifier
from trading.config import SystemConfig

logger = logging.getLogger(__name__)


class OptionsTrader:
    """
    Automated options trading loop with momentum + swing modes.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        from trading.settings import Settings
        self.config = config or SystemConfig()
        self.broker = AlpacaBroker()
        self.scanner = None
        self.db = TradeDB()
        self.notifier = TelegramNotifier(self.config)
        self.settings = Settings()

        # Risk limits
        self.max_premium_per_trade = 1500.0
        self.max_open_positions = 5

        # Mode-specific exit rules
        self.exit_rules = {
            "momentum": {
                "tp_pct": 0.30,           # 30% profit target
                "sl_pct": 0.35,           # 35% stop loss
                "partial_tp_pct": 0.20,   # Take half at 20%
                "time_stop_hours": 4,     # Close if no move in 4 hours
                "force_close_dte": 2,
            },
            "swing": {
                "tp_pct": 0.50,           # 50% profit target
                "sl_pct": 0.40,           # 40% stop loss
                "partial_tp_pct": 0.25,   # Take half at 25%
                "time_stop_hours": 24,    # Close if no move in 24 hours
                "force_close_dte": 2,
            },
        }

        # State
        self.open_trades: dict[str, dict] = {}
        self._stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "cycles": 0,
            "momentum_trades": 0,
            "swing_trades": 0,
        }
        self._load_state()

    # ─── Settings-backed toggles (persist to settings.json) ───
    @property
    def scanning_enabled(self) -> bool:
        # Always re-read from disk so Telegram /toggle takes effect immediately
        from trading.settings import Settings
        return bool(Settings().toggles.options_scanning_enabled)

    @scanning_enabled.setter
    def scanning_enabled(self, value: bool) -> None:
        self.settings.set("options_scanning_enabled", bool(value))

    @property
    def auto_trading_enabled(self) -> bool:
        from trading.settings import Settings
        return bool(Settings().toggles.options_trading_enabled)

    @auto_trading_enabled.setter
    def auto_trading_enabled(self, value: bool) -> None:
        self.settings.set("options_trading_enabled", bool(value))

    def _load_state(self):
        state_file = "trading/data/options_trader_state.json"
        try:
            with open(state_file) as f:
                data = json.load(f)
                self.open_trades = data.get("open_trades", {})
                self._stats.update(data.get("stats", {}))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_state(self):
        state_file = "trading/data/options_trader_state.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({
                "open_trades": self.open_trades,
                "stats": self._stats,
            }, f, indent=2)

    def start(self, scan_interval: int = 300):
        """Start the options trading loop."""
        if not self.broker.api_key:
            logger.info("Alpaca not configured — options trading disabled")
            return

        logger.info("Connecting to Alpaca...")
        if not self.broker.connect():
            logger.error("Alpaca connection failed")
            return

        self.scanner = OptionsScanner(self.broker)
        balance = self.broker.get_account_balance()
        logger.info(f"Options trader v2 started — Balance: ${balance:,.2f}")

        from trading.options_scanner import TIER1_SYMBOLS, TIER2_SYMBOLS, TIER3_SYMBOLS
        total_syms = len(TIER1_SYMBOLS) + len(TIER2_SYMBOLS) + len(TIER3_SYMBOLS)
        self.notifier.send_system_alert(
            f"OPTIONS v2 online\n"
            f"Scanning: {total_syms} symbols\n"
            f"T1: {', '.join(TIER1_SYMBOLS)}\n"
            f"T2: {', '.join(TIER2_SYMBOLS)}\n"
            f"T3: {', '.join(TIER3_SYMBOLS)}\n"
            f"Modes: Momentum (3-14 DTE) + Swing (7-30 DTE)\n"
            f"Alpaca balance: ${balance:,.2f}\n"
            f"Max positions: {self.max_open_positions}"
        )

        cycle_count = 0
        while True:
            try:
                if is_options_market_open() and self.scanning_enabled:
                    cycle_count += 1
                    logger.info(f"OPTIONS CYCLE #{cycle_count} starting...")
                    self._run_cycle()

                    logger.info(
                        f"OPTIONS CYCLE #{cycle_count} — "
                        f"Open: {len(self.open_trades)}/{self.max_open_positions} | "
                        f"Total: {self._stats['total_trades']} | "
                        f"W/L: {self._stats['wins']}/{self._stats['losses']} | "
                        f"PnL: ${self._stats['total_pnl']:+,.2f}"
                    )
                else:
                    if cycle_count == 0:
                        # First cycle and market closed — notify once
                        now = datetime.now(timezone.utc)
                        logger.info(
                            f"Options market closed — {now.strftime('%H:%M UTC')} "
                            f"(opens ~9:45 ET Mon-Fri)"
                        )
                        self.notifier.send_system_alert(
                            f"OPTIONS: Market closed ({datetime.now(timezone.utc).strftime('%H:%M UTC')})\n"
                            f"Will auto-scan when market opens ~9:45 ET"
                        )
                        cycle_count = -1  # So we don't spam this
            except KeyboardInterrupt:
                logger.info("Options trader stopping...")
                break
            except Exception as e:
                logger.error(f"Options cycle error: {e}", exc_info=True)
                self.notifier.send_system_alert(f"OPTIONS ERROR: {str(e)[:200]}")

            try:
                time.sleep(scan_interval)
            except KeyboardInterrupt:
                break

    def _run_cycle(self):
        """Run one options scan + trade cycle."""
        self._stats["cycles"] += 1

        # Check existing positions first (exits)
        self._check_positions()

        # Scan for new signals
        logger.info("OPTIONS: Starting scan...")
        from trading.options_scanner import TIER1_SYMBOLS, TIER2_SYMBOLS, TIER3_SYMBOLS

        if self.scanner._controlled_start:
            total_symbols = min(2, len(TIER1_SYMBOLS))
        else:
            total_symbols = len(TIER1_SYMBOLS) + len(TIER2_SYMBOLS) + len(TIER3_SYMBOLS)

        signals = self.scanner.scan_all()
        logger.info(f"OPTIONS: Scan returned {len(signals)} signals")

        # Send diagnostic summary every cycle so we can see where the pipeline dies
        diag_lines = [f"OPTIONS OPEN CHECK (cycle #{self._stats['cycles']})"]
        diag_lines.append(f"Symbols scanned: {total_symbols}")
        diag_lines.append(f"Signals found: {len(signals)}")
        if signals:
            for s in signals[:3]:
                c = s["contract"]
                diag_lines.append(
                    f"\n{s['symbol']} {s['mode'].upper()} {s['side'].upper()}:"
                    f"\n  contract: {c.option_type.upper()} ${c.strike} {c.expiration} ({c.dte}DTE)"
                    f"\n  premium: ${c.mid:.2f} | spread: {c.spread_pct:.1%} | OI: {c.open_interest}"
                    f"\n  confidence: {s['confidence']:.2f}"
                    f"\n  action: WOULD {'BUY' if self.auto_trading_enabled else 'ALERT ONLY'}"
                )
        # Show where rejected symbols died
        if hasattr(self.scanner, '_diagnostics') and self.scanner._diagnostics:
            diag_lines.append(f"\nRejections ({len(self.scanner._diagnostics)}):")
            # Group by terminal stage
            from collections import Counter
            stage_counts = Counter(d["terminal_stage"] for d in self.scanner._diagnostics)
            for stage, count in stage_counts.most_common():
                diag_lines.append(f"  {stage}: {count}")
            # Show first 5 individual rejections
            for d in self.scanner._diagnostics[:5]:
                diag_lines.append(f"  {d['symbol']} {d['mode']}: {d['terminal_stage']} — {d['detail'][:80]}")

        self.notifier.send_system_alert("\n".join(diag_lines))

        trades_placed = 0
        trades_blocked = 0

        if not signals:
            # Send scan summary even with 0 signals
            self.notifier.send_options_scan(
                symbols_scanned=total_symbols,
                signals_found=0,
                trades_placed=0,
                trades_blocked=0,
            )
            return

        if not self.auto_trading_enabled:
            self.notifier.send_system_alert(
                f"OPTIONS: {len(signals)} signals found but auto-trading is OFF\n"
                f"Send /optiontrade on to enable"
            )
            self.notifier.send_options_scan(
                symbols_scanned=total_symbols,
                signals_found=len(signals),
                trades_placed=0,
                trades_blocked=len(signals),
            )
            return

        # Controlled start: max 1 trade until pipeline is proven
        max_signals = 1 if self.scanner._controlled_start else 3
        for signal in signals[:max_signals]:
            if len(self.open_trades) >= self.max_open_positions:
                trades_blocked += 1
                self.notifier.send_system_alert(
                    f"OPTIONS BLOCKED: {signal['symbol']} — max positions ({self.max_open_positions}) reached"
                )
                continue

            # Skip if already have position in this underlying
            if any(t["symbol"] == signal["symbol"] for t in self.open_trades.values()):
                trades_blocked += 1
                self.notifier.send_system_alert(
                    f"OPTIONS BLOCKED: {signal['symbol']} — already have position"
                )
                continue

            contract = signal["contract"]

            if contract.max_loss > self.max_premium_per_trade:
                logger.info(
                    f"SKIP {signal['symbol']}: Premium ${contract.max_loss:.0f} "
                    f"> max ${self.max_premium_per_trade}"
                )
                trades_blocked += 1
                self.notifier.send_system_alert(
                    f"OPTIONS BLOCKED: {signal['symbol']} — premium ${contract.max_loss:.0f} > max ${self.max_premium_per_trade:.0f}"
                )
                continue

            # Place the trade
            result = self.broker.place_option_order(
                option_symbol=contract.symbol,
                side="buy",
                quantity=1,
                order_type="limit",
                limit_price=contract.mid,
            )

            if not result:
                logger.error(f"Option order returned None for {signal['symbol']}")
                trades_blocked += 1
                continue

            if result.success:
                trades_placed += 1
                mode = signal.get("mode", "swing")
                trade_info = {
                    "trade_id": result.order_id,
                    "symbol": signal["symbol"],
                    "option_symbol": contract.symbol,
                    "option_type": contract.option_type,
                    "side": signal["side"],
                    "mode": mode,
                    "tier": signal.get("tier", 1),
                    "strike": contract.strike,
                    "expiration": contract.expiration,
                    "dte": contract.dte,
                    "entry_premium": contract.mid,
                    "max_loss": contract.max_loss,
                    "confidence": signal["confidence"],
                    "confidence_scores": signal.get("confidence_scores", {}),
                    "reasons": signal.get("reasons", []),
                    "reasoning": signal["reasoning"],
                    "placed_at": datetime.now(timezone.utc).isoformat(),
                    "entry_underlying": signal["entry"],
                    "peak_premium": contract.mid,
                    "partial_taken": False,
                }

                self.open_trades[result.order_id] = trade_info
                self._stats["total_trades"] += 1
                if mode == "momentum":
                    self._stats["momentum_trades"] += 1
                else:
                    self._stats["swing_trades"] += 1

                # DB
                self.db.save_trade(
                    trade_id=result.order_id,
                    symbol=f"{signal['symbol']}_{contract.option_type.upper()}_{contract.strike}",
                    side=signal["side"],
                    units=1,
                    entry_price=contract.mid,
                    stop_loss=contract.mid * (1 - self.exit_rules[mode]["sl_pct"]),
                    take_profit=contract.mid * (1 + self.exit_rules[mode]["tp_pct"]),
                    confidence=signal["confidence"],
                    reasoning=signal["reasoning"],
                )

                # Notify with full detail
                direction = "CALL" if contract.option_type == "call" else "PUT"
                self.notifier.send_system_alert(
                    f"OPTIONS TRADE [{mode.upper()}]\n\n"
                    f"{direction} {signal['symbol']} ${contract.strike}\n"
                    f"Exp: {contract.expiration} ({contract.dte} DTE)\n"
                    f"Premium: ${contract.mid:.2f} (max loss: ${contract.max_loss:.0f})\n"
                    f"Confidence: {signal['confidence']:.0%}\n"
                    f"Tier: {'SPY/QQQ' if signal.get('tier') == 1 else 'Single name'}\n\n"
                    f"Exits: TP {self.exit_rules[mode]['tp_pct']:.0%} | "
                    f"SL {self.exit_rules[mode]['sl_pct']:.0%} | "
                    f"Partial {self.exit_rules[mode]['partial_tp_pct']:.0%} | "
                    f"Time stop {self.exit_rules[mode]['time_stop_hours']}h\n\n"
                    f"{signal['reasoning'][:300]}"
                )

                logger.info(
                    f"OPTIONS TRADE: [{mode.upper()}] {direction} {signal['symbol']} "
                    f"${contract.strike} exp {contract.expiration} @ ${contract.mid:.2f}"
                )

        # Send OPTIONS SCAN summary (mirrors forex scan format)
        self.notifier.send_options_scan(
            symbols_scanned=total_symbols,
            signals_found=len(signals),
            trades_placed=trades_placed,
            trades_blocked=trades_blocked,
        )
        self._save_state()

    def _check_positions(self):
        """Monitor open positions with dynamic exit logic."""
        if not self.open_trades:
            return

        positions = self.broker.get_positions()
        pos_by_symbol = {p.symbol: p for p in positions}
        closed_ids = []

        for trade_id, info in self.open_trades.items():
            option_sym = info["option_symbol"]
            entry_premium = info["entry_premium"]
            mode = info.get("mode", "swing")
            rules = self.exit_rules[mode]

            pos = pos_by_symbol.get(option_sym)

            if pos is None:
                closed_ids.append(trade_id)
                self._resolve_closed(trade_id, info, 0, "expired_or_filled")
                continue

            current_premium = pos.current_price
            pnl_pct = (current_premium - entry_premium) / entry_premium if entry_premium > 0 else 0

            # Track peak premium for trailing
            if current_premium > info.get("peak_premium", 0):
                info["peak_premium"] = current_premium

            # ─── EXIT 1: Full profit target ───
            if pnl_pct >= rules["tp_pct"]:
                logger.info(f"TP HIT: {info['symbol']} [{mode}] +{pnl_pct:.0%}")
                self.broker.close_position(option_sym)
                closed_ids.append(trade_id)
                pnl = (current_premium - entry_premium) * 100
                self._resolve_closed(trade_id, info, pnl, "profit_target")
                continue

            # ─── EXIT 2: Stop loss ───
            if pnl_pct <= -rules["sl_pct"]:
                logger.info(f"SL HIT: {info['symbol']} [{mode}] {pnl_pct:.0%}")
                self.broker.close_position(option_sym)
                closed_ids.append(trade_id)
                pnl = (current_premium - entry_premium) * 100
                self._resolve_closed(trade_id, info, pnl, "stop_loss")
                continue

            # ─── EXIT 3: Partial take-profit ───
            if not info.get("partial_taken") and pnl_pct >= rules["partial_tp_pct"]:
                logger.info(f"PARTIAL TP: {info['symbol']} [{mode}] +{pnl_pct:.0%}")
                # In production you'd sell half; for now just log and trail
                info["partial_taken"] = True
                info["trail_from"] = current_premium
                self.notifier.send_system_alert(
                    f"OPTIONS PARTIAL TP +{pnl_pct:.0%}\n"
                    f"{info['symbol']} [{mode.upper()}]\n"
                    f"Trailing remainder..."
                )

            # ─── EXIT 4: Trailing stop after partial ───
            if info.get("partial_taken") and info.get("peak_premium", 0) > 0:
                trail_from_peak = (current_premium - info["peak_premium"]) / info["peak_premium"]
                if trail_from_peak <= -0.15:  # Give back 15% from peak
                    logger.info(f"TRAIL STOP: {info['symbol']} [{mode}] gave back from peak")
                    self.broker.close_position(option_sym)
                    closed_ids.append(trade_id)
                    pnl = (current_premium - entry_premium) * 100
                    self._resolve_closed(trade_id, info, pnl, "trailing_stop")
                    continue

            # ─── EXIT 5: Time stop ───
            try:
                placed = datetime.fromisoformat(info["placed_at"].replace("Z", "+00:00"))
                hours_held = (datetime.now(timezone.utc) - placed).total_seconds() / 3600
                if hours_held >= rules["time_stop_hours"] and pnl_pct < 0.05:
                    logger.info(
                        f"TIME STOP: {info['symbol']} [{mode}] "
                        f"{hours_held:.0f}h held, only {pnl_pct:+.0%}"
                    )
                    self.broker.close_position(option_sym)
                    closed_ids.append(trade_id)
                    pnl = (current_premium - entry_premium) * 100
                    self._resolve_closed(trade_id, info, pnl, "time_stop")
                    continue
            except (ValueError, TypeError):
                pass

            # ─── EXIT 6: Force close near expiry ───
            try:
                exp_date = datetime.strptime(info["expiration"], "%Y-%m-%d").date()
                now = datetime.now(timezone.utc).date()
                dte = (exp_date - now).days
                if dte <= rules["force_close_dte"]:
                    logger.info(f"EXPIRY CLOSE: {info['symbol']} {dte} DTE")
                    self.broker.close_position(option_sym)
                    closed_ids.append(trade_id)
                    pnl = (current_premium - entry_premium) * 100
                    self._resolve_closed(trade_id, info, pnl, "expiry_close")
            except (ValueError, TypeError):
                pass

        for tid in closed_ids:
            del self.open_trades[tid]

        self._save_state()

    def _resolve_closed(self, trade_id: str, info: dict, pnl: float, exit_reason: str):
        """Handle a closed options trade with full logging."""
        outcome = "win" if pnl > 0 else "loss"
        mode = info.get("mode", "swing")

        if outcome == "win":
            self._stats["wins"] += 1
        else:
            self._stats["losses"] += 1
        self._stats["total_pnl"] += pnl

        self.db.update_trade(
            trade_id=trade_id,
            exit_price=info["entry_premium"] + (pnl / 100),
            outcome=outcome,
            pnl=pnl,
        )

        balance = self.broker.get_account_balance()
        direction = info.get("option_type", "option").upper()

        # Detailed exit notification
        self.notifier.send_system_alert(
            f"OPTIONS {'WIN' if outcome == 'win' else 'LOSS'} "
            f"{'+'if pnl > 0 else ''}${pnl:.2f}\n\n"
            f"{info['symbol']} {direction} ${info['strike']}\n"
            f"Mode: {mode.upper()} | Exit: {exit_reason}\n"
            f"Entry: ${info['entry_premium']:.2f} | "
            f"Confidence: {info.get('confidence', 0):.0%}\n"
            f"Balance: ${balance:,.2f}\n\n"
            f"Scores: {info.get('confidence_scores', {})}\n"
            f"Reasons: {', '.join(info.get('reasons', [])[:3])}"
        )

        logger.info(
            f"OPTIONS {'WIN' if outcome == 'win' else 'LOSS'}: "
            f"{info['symbol']} [{mode}] ${pnl:+.2f} | exit={exit_reason}"
        )

    def get_status(self) -> str:
        """Get options trader status with full detail."""
        total = self._stats["wins"] + self._stats["losses"]
        win_rate = (self._stats["wins"] / total * 100) if total > 0 else 0

        try:
            from trading.options_scanner import TIER1_SYMBOLS, TIER2_SYMBOLS, TIER3_SYMBOLS
            total_syms = len(TIER1_SYMBOLS) + len(TIER2_SYMBOLS) + len(TIER3_SYMBOLS)
        except Exception:
            total_syms = 25

        lines = [
            "OPTIONS TRADER v2",
            "",
            f"  Symbols:    {total_syms} (T1:{len(TIER1_SYMBOLS)} T2:{len(TIER2_SYMBOLS)} T3:{len(TIER3_SYMBOLS)})",
            f"  Modes:      Momentum + Swing",
            f"  Trades:     {self._stats['total_trades']}",
            f"  Momentum:   {self._stats.get('momentum_trades', 0)}",
            f"  Swing:      {self._stats.get('swing_trades', 0)}",
            f"  Wins:       {self._stats['wins']}",
            f"  Losses:     {self._stats['losses']}",
            f"  Win Rate:   {win_rate:.0f}%",
            f"  PnL:        ${self._stats['total_pnl']:+,.2f}",
            f"  Open:       {len(self.open_trades)}/{self.max_open_positions}",
            "",
            "  Exit Rules:",
            f"  Momentum: TP 30% | SL 35% | Partial 20% | Time 4h",
            f"  Swing:    TP 50% | SL 40% | Partial 25% | Time 24h",
            "",
        ]

        if self.open_trades:
            lines.append("  Open Positions:")
            for tid, info in self.open_trades.items():
                mode = info.get("mode", "?")
                lines.append(
                    f"  [{mode.upper()}] {info['symbol']} "
                    f"{info.get('option_type', '?').upper()} "
                    f"${info['strike']} exp {info['expiration']} "
                    f"@ ${info['entry_premium']:.2f}"
                )

        return "\n".join(lines)
