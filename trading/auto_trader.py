"""
Auto Trader — Runs the full loop: scan, trade, track, learn.

Connects to OANDA practice account and automatically:
1. Scans forex pairs for signals
2. Places trades with stop loss + take profit
3. Monitors open positions
4. Runs loss analysis on every losing trade
5. Updates the system to avoid repeating mistakes
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from trading.config import SystemConfig
from trading.brokers.oanda import OandaBroker
from trading.forex_scanner import ForexScanner
from trading.research_agent import ResearchAgent
from trading.risk_manager import RiskManager
from trading.loss_analyzer import LossAnalyzer
from trading.notifier import TelegramNotifier
from trading.models import Market, Trade, Prediction, ResearchResult

logger = logging.getLogger(__name__)


class AutoTrader:
    """
    Fully automated trading loop for practice account testing.

    Scans → Trades → Monitors → Learns → Repeats
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # Brokers
        self.oanda = OandaBroker()

        # Agents
        self.scanner = None  # initialized after connect
        self.researcher = ResearchAgent(self.config)
        self.risk_mgr = RiskManager(self.config)
        self.loss_analyzer = LossAnalyzer(self.config)
        self.notifier = TelegramNotifier(self.config)

        # State
        self._trades: list[dict] = []
        self._open_trades: dict[str, dict] = {}  # order_id -> trade info
        self._stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "cycles": 0,
        }
        self._load_state()

    def _load_state(self):
        """Load saved state."""
        state_file = "trading/data/auto_trader_state.json"
        try:
            with open(state_file) as f:
                data = json.load(f)
                self._trades = data.get("trades", [])
                self._stats = data.get("stats", self._stats)
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

        self.notifier.send_startup(balance, dry_run=False)
        logger.info(f"Auto trader started — Balance: ${balance:.2f}")
        logger.info(f"Scan interval: {scan_interval}s")

        while True:
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
        """Run one scan → trade → monitor cycle."""
        self._stats["cycles"] += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"CYCLE {self._stats['cycles']}")
        logger.info(f"{'='*50}")

        # Step 1: Check open positions and resolve completed trades
        self._check_positions()

        # Step 2: Scan for new signals
        signals = self.scanner.scan_all_pairs()

        if not signals:
            logger.info("No signals found this cycle")
            self.notifier.send_scan_summary(
                total_markets=len(self.oanda.get_spread_for_pairs()),
                flagged=0,
                predictions=0,
                trades=0,
                blocked=0,
            )
            self._save_state()
            return

        # Step 3: Filter through risk manager and trade
        trades_placed = 0
        trades_blocked = 0

        for signal in signals[:5]:  # max 5 signals per cycle
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
                liquidity=100000,  # forex is always liquid
                spread=0,
            )

            # Risk check
            risk_eval = self.risk_mgr.evaluate_trade(prediction, market)

            if not risk_eval["approved"]:
                logger.info(
                    f"BLOCKED: {signal['symbol']} — {risk_eval['reason']}"
                )
                self.notifier.send_trade_blocked(
                    market, prediction, risk_eval["reason"]
                )
                trades_blocked += 1
                continue

            # Calculate units based on risk
            balance = self.oanda.get_account_balance()
            risk_amount = balance * self.config.risk.max_bet_pct
            # Units = risk amount / stop loss distance
            sl_distance = abs(signal["entry"] - signal["stop_loss"])
            if sl_distance > 0:
                units = int(risk_amount / sl_distance)
                units = max(1, min(units, 10000))  # cap at 10k units for practice
            else:
                units = 1000

            # Place the trade
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
                    "reasoning": signal["reasoning"],
                    "placed_at": datetime.utcnow().isoformat(),
                    "status": "open",
                }

                self._open_trades[trade_id] = trade_info
                self._trades.append(trade_info)
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
                    f"{signal['symbol']} {units} units @ {result.price}"
                )

            else:
                logger.warning(
                    f"Order failed for {signal['symbol']}: {result.message}"
                )

        # Summary
        self.notifier.send_scan_summary(
            total_markets=len(self.oanda.get_spread_for_pairs()),
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

        # Check if any of our tracked trades have closed
        closed_trade_ids = []

        for trade_id, trade_info in self._open_trades.items():
            symbol = trade_info["symbol"]

            if symbol not in open_symbols:
                # Trade has been closed (hit SL or TP)
                closed_trade_ids.append(trade_id)

                # Figure out PnL
                # Get current price to estimate
                quote = self.oanda.get_quote(symbol)
                if quote:
                    current = quote.mid
                else:
                    current = trade_info["entry"]

                entry = trade_info["entry"]
                side = trade_info["side"]
                units = trade_info["units"]

                if side == "buy":
                    pnl = (current - entry) * units
                else:
                    pnl = (entry - current) * units

                # Determine if win or loss based on where it closed
                # If it hit take profit, it's a win. If stop loss, it's a loss.
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

                if outcome == "win":
                    self._stats["wins"] += 1
                    self._stats["total_pnl"] += pnl
                    balance = self.oanda.get_account_balance()

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
                    balance = self.oanda.get_account_balance()

                    # Run loss analysis
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

                    self.notifier.send_loss_alert(
                        trade_obj, pnl, balance, lesson.description
                    )

                    logger.info(
                        f"LOSS: {symbol} -${abs(pnl):.2f} | "
                        f"Lesson: {lesson.category} — {lesson.rule_added}"
                    )

                    # Update risk manager
                    self.risk_mgr.resolve_trade(trade_id, pnl)

        # Remove closed trades from open list
        for tid in closed_trade_ids:
            del self._open_trades[tid]

        self._save_state()

    def get_status(self) -> str:
        """Get current auto trader status."""
        total = self._stats["wins"] + self._stats["losses"]
        win_rate = (
            self._stats["wins"] / total * 100
        ) if total > 0 else 0

        balance = self.oanda.get_account_balance() if self.oanda.connected else 0

        return (
            f"=== AUTO TRADER STATUS ===\n"
            f"Balance: ${balance:.2f}\n"
            f"Cycles run: {self._stats['cycles']}\n"
            f"Total trades: {self._stats['total_trades']}\n"
            f"Wins: {self._stats['wins']} | Losses: {self._stats['losses']}\n"
            f"Win rate: {win_rate:.0f}%\n"
            f"Total PnL: ${self._stats['total_pnl']:+.2f}\n"
            f"Open trades: {len(self._open_trades)}\n"
            f"Lessons learned: {len(self.loss_analyzer._lessons)}\n"
        )
