"""
Options Trader — Automated options trading with separate risk rules.

Runs alongside the forex auto_trader but on a completely separate track:
- Different broker (Alpaca)
- Different risk rules (premium-based, not pip-based)
- Different market hours (US 9:30-4:00 ET, Mon-Fri)
- Separate performance tracking

Risk Rules (options-specific):
- Max premium per trade: $500
- Max contracts per trade: 2
- Max spread: 15%
- No averaging down
- Force close before expiry (2 DTE)
- No trading first/last 15 min
- Max 3 open option positions
- Exit at 50% profit or 40% loss on premium
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
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
    Automated options trading loop.

    Separate from forex — different broker, different rules, different track.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.broker = AlpacaBroker()
        self.scanner = None
        self.db = TradeDB()
        self.notifier = TelegramNotifier(self.config)

        # Options-specific risk limits
        self.max_premium_per_trade = 500.0
        self.max_contracts_per_trade = 2
        self.max_open_positions = 3
        self.profit_target_pct = 0.50   # Close at 50% profit
        self.stop_loss_pct = 0.40       # Close at 40% loss
        self.force_close_dte = 2        # Close 2 days before expiry

        # State
        self.open_trades: dict[str, dict] = {}
        self._stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "cycles": 0,
        }
        self._load_state()

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
            self.notifier.send_system_alert(
                "Options module: No Alpaca API key set.\n"
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY to enable."
            )
            return

        logger.info("Connecting to Alpaca...")
        if not self.broker.connect():
            logger.error("Alpaca connection failed")
            return

        self.scanner = OptionsScanner(self.broker)
        balance = self.broker.get_account_balance()
        logger.info(f"Options trader started — Alpaca balance: ${balance:,.2f}")

        self.notifier.send_system_alert(
            f"Options module online.\n"
            f"Alpaca balance: ${balance:,.2f}\n"
            f"Symbols: {', '.join(OPTIONS_SYMBOLS)}\n"
            f"Market hours: 9:45 AM - 3:45 PM ET"
        )

        while True:
            try:
                if is_options_market_open():
                    self._run_cycle()
                else:
                    logger.debug("Options market closed — waiting...")
            except KeyboardInterrupt:
                logger.info("Options trader stopping...")
                break
            except Exception as e:
                logger.error(f"Options cycle error: {e}", exc_info=True)

            try:
                time.sleep(scan_interval)
            except KeyboardInterrupt:
                break

    def _run_cycle(self):
        """Run one options scan + trade cycle."""
        self._stats["cycles"] += 1

        # Check open positions first
        self._check_positions()

        # Scan for new signals
        signals = self.scanner.scan_all()
        if not signals:
            return

        for signal in signals[:3]:
            if len(self.open_trades) >= self.max_open_positions:
                break

            # Skip if already have position in this underlying
            if any(t["symbol"] == signal["symbol"] for t in self.open_trades.values()):
                continue

            contract = signal["contract"]

            # Options risk checks
            if contract.max_loss > self.max_premium_per_trade:
                logger.info(f"SKIP {signal['symbol']}: Premium ${contract.max_loss:.0f} > max ${self.max_premium_per_trade}")
                continue

            # Place the trade (limit order at mid price)
            result = self.broker.place_option_order(
                option_symbol=contract.symbol,
                side="buy",
                quantity=1,
                order_type="limit",
                limit_price=contract.mid,
            )

            if result.success:
                trade_info = {
                    "trade_id": result.order_id,
                    "symbol": signal["symbol"],
                    "option_symbol": contract.symbol,
                    "option_type": contract.option_type,
                    "side": signal["side"],
                    "strike": contract.strike,
                    "expiration": contract.expiration,
                    "dte": contract.dte,
                    "entry_premium": contract.mid,
                    "max_loss": contract.max_loss,
                    "confidence": signal["confidence"],
                    "reasoning": signal["reasoning"],
                    "placed_at": datetime.now(timezone.utc).isoformat(),
                }

                self.open_trades[result.order_id] = trade_info
                self._stats["total_trades"] += 1

                # Save to DB
                self.db.save_trade(
                    trade_id=result.order_id,
                    symbol=f"{signal['symbol']}_{contract.option_type.upper()}_{contract.strike}",
                    side=signal["side"],
                    units=1,
                    entry_price=contract.mid,
                    stop_loss=contract.mid * (1 - self.stop_loss_pct),
                    take_profit=contract.mid * (1 + self.profit_target_pct),
                    confidence=signal["confidence"],
                    reasoning=signal["reasoning"],
                )

                # Notify
                direction = "CALL" if contract.option_type == "call" else "PUT"
                self.notifier.send_option_alert(
                    symbol=signal["symbol"],
                    option_type=contract.option_type,
                    strike=contract.strike,
                    expiration=contract.expiration,
                    hit_pct=signal["confidence"] * 100,
                    entry_price=contract.mid,
                    underlying_price=signal["entry"],
                    amount=contract.max_loss,
                    reasoning=signal["reasoning"],
                )

                logger.info(
                    f"OPTIONS TRADE: {direction} {signal['symbol']} "
                    f"${contract.strike} exp {contract.expiration} "
                    f"@ ${contract.mid:.2f}"
                )

        self._save_state()

    def _check_positions(self):
        """Monitor open options positions for exit conditions."""
        if not self.open_trades:
            return

        positions = self.broker.get_positions()
        pos_by_symbol = {p.symbol: p for p in positions}

        closed_ids = []

        for trade_id, info in self.open_trades.items():
            option_sym = info["option_symbol"]
            entry_premium = info["entry_premium"]

            pos = pos_by_symbol.get(option_sym)

            if pos is None:
                # Position closed (filled SL/TP or expired)
                closed_ids.append(trade_id)
                self._resolve_closed(trade_id, info, 0)
                continue

            current_premium = pos.current_price
            pnl_pct = (current_premium - entry_premium) / entry_premium if entry_premium > 0 else 0

            # Exit: profit target
            if pnl_pct >= self.profit_target_pct:
                logger.info(f"PROFIT TARGET: {info['symbol']} +{pnl_pct:.0%}")
                self.broker.close_position(option_sym)
                closed_ids.append(trade_id)
                pnl = (current_premium - entry_premium) * 100
                self._resolve_closed(trade_id, info, pnl, "win")
                continue

            # Exit: stop loss
            if pnl_pct <= -self.stop_loss_pct:
                logger.info(f"STOP LOSS: {info['symbol']} {pnl_pct:.0%}")
                self.broker.close_position(option_sym)
                closed_ids.append(trade_id)
                pnl = (current_premium - entry_premium) * 100
                self._resolve_closed(trade_id, info, pnl, "loss")
                continue

            # Exit: force close near expiry
            try:
                exp_date = datetime.strptime(info["expiration"], "%Y-%m-%d").date()
                now = datetime.now(timezone.utc).date()
                dte = (exp_date - now).days
                if dte <= self.force_close_dte:
                    logger.info(f"EXPIRY CLOSE: {info['symbol']} {dte} DTE remaining")
                    self.broker.close_position(option_sym)
                    closed_ids.append(trade_id)
                    pnl = (current_premium - entry_premium) * 100
                    outcome = "win" if pnl > 0 else "loss"
                    self._resolve_closed(trade_id, info, pnl, outcome)
            except (ValueError, TypeError):
                pass

        for tid in closed_ids:
            del self.open_trades[tid]

        self._save_state()

    def _resolve_closed(self, trade_id: str, info: dict, pnl: float, outcome: str = None):
        """Handle a closed options trade."""
        if outcome is None:
            outcome = "win" if pnl > 0 else "loss"

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

        if outcome == "win":
            self.notifier.send_system_alert(
                f"OPTIONS WIN +${pnl:.2f}\n"
                f"{info['symbol']} {direction} ${info['strike']}\n"
                f"Balance: ${balance:,.2f}"
            )
        else:
            self.notifier.send_system_alert(
                f"OPTIONS LOSS -${abs(pnl):.2f}\n"
                f"{info['symbol']} {direction} ${info['strike']}\n"
                f"Balance: ${balance:,.2f}"
            )

    def get_status(self) -> str:
        """Get options trader status."""
        total = self._stats["wins"] + self._stats["losses"]
        win_rate = (self._stats["wins"] / total * 100) if total > 0 else 0

        lines = [
            "OPTIONS TRADER STATUS",
            "",
            f"  Trades:     {self._stats['total_trades']}",
            f"  Wins:       {self._stats['wins']}",
            f"  Losses:     {self._stats['losses']}",
            f"  Win Rate:   {win_rate:.0f}%",
            f"  PnL:        ${self._stats['total_pnl']:+,.2f}",
            f"  Open:       {len(self.open_trades)}",
            f"  Symbols:    {', '.join(OPTIONS_SYMBOLS)}",
        ]

        for tid, info in self.open_trades.items():
            lines.append(
                f"  {info['symbol']} {info['option_type'].upper()} "
                f"${info['strike']} exp {info['expiration']}"
            )

        return "\n".join(lines)
