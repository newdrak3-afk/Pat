"""
Orchestrator — Coordinates all agents end-to-end.

Flow:
1. ScanAgent filters 300+ markets
2. ResearchAgents run in parallel on flagged/promising markets
3. PredictionAgent calibrates probability and fires when confident
4. RiskManager sizes the bet and blocks if too risky
5. Execute trade (or simulate)
6. LossAnalyzer reviews every loss and updates the system
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from trading.config import SystemConfig
from trading.models import Market, Trade, TradeOutcome
from trading.scan_agent import ScanAgent
from trading.research_agent import ResearchAgent
from trading.prediction_agent import PredictionAgent
from trading.risk_manager import RiskManager
from trading.loss_analyzer import LossAnalyzer
from trading.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator that ties all agents together.

    Runs the full pipeline:
    scan -> research -> predict -> risk check -> execute -> analyze losses
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # Initialize all agents
        self.scanner = ScanAgent(self.config)
        self.researcher = ResearchAgent(self.config)
        self.predictor = PredictionAgent(self.config)
        self.risk_mgr = RiskManager(self.config)
        self.loss_analyzer = LossAnalyzer(self.config)
        self.notifier = TelegramNotifier(self.config)

        # Trade log
        self._trades: list[dict] = []
        self._load_trades()

        # Simulation mode (no real money)
        self.dry_run = True

    def _load_trades(self):
        """Load trade history."""
        try:
            with open(self.config.trades_log) as f:
                self._trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._trades = []

    def _save_trades(self):
        """Persist trade history."""
        os.makedirs(os.path.dirname(self.config.trades_log), exist_ok=True)
        with open(self.config.trades_log, "w") as f:
            json.dump(self._trades, f, indent=2)

    def run_cycle(self) -> dict:
        """
        Run one full trading cycle.

        Returns a summary dict with stats about what happened.
        """
        cycle_start = datetime.utcnow()
        summary = {
            "cycle_id": str(uuid4())[:8],
            "started_at": cycle_start.isoformat(),
            "markets_scanned": 0,
            "markets_flagged": 0,
            "markets_researched": 0,
            "predictions_made": 0,
            "trades_approved": 0,
            "trades_blocked": 0,
            "total_amount_bet": 0.0,
            "errors": [],
        }

        try:
            # ─── PHASE 1: SCAN ───
            logger.info("=" * 60)
            logger.info("PHASE 1: SCANNING MARKETS")
            logger.info("=" * 60)

            markets = self.scanner.scan()
            summary["markets_scanned"] = len(markets)

            flagged = [m for m in markets if m.flagged]
            summary["markets_flagged"] = len(flagged)

            logger.info(self.scanner.get_summary(markets))

            if not markets:
                logger.warning("No markets found — ending cycle")
                return summary

            # Select top markets to research:
            # - All flagged markets (anomalies are opportunities)
            # - Top non-flagged by volume (most liquid = safest)
            non_flagged = sorted(
                [m for m in markets if not m.flagged],
                key=lambda m: m.volume_24h,
                reverse=True,
            )
            to_research = flagged[:10] + non_flagged[:10]
            to_research = to_research[:15]  # cap at 15 for speed

            # ─── PHASE 2: RESEARCH ───
            logger.info("=" * 60)
            logger.info(f"PHASE 2: RESEARCHING {len(to_research)} MARKETS")
            logger.info("=" * 60)

            research_results = self.researcher.research_markets_parallel(
                to_research
            )
            summary["markets_researched"] = len(research_results)

            # Build lookup
            research_by_id = {r.market_id: r for r in research_results}
            market_by_id = {m.market_id: m for m in to_research}

            # ─── PHASE 3: PREDICT ───
            logger.info("=" * 60)
            logger.info("PHASE 3: GENERATING PREDICTIONS")
            logger.info("=" * 60)

            predictions = []
            for market in to_research:
                research = research_by_id.get(market.market_id)
                if not research:
                    continue

                try:
                    prediction = self.predictor.predict(market, research)
                    if prediction:
                        predictions.append(
                            (market, research, prediction)
                        )
                except Exception as e:
                    logger.error(
                        f"Prediction failed for {market.market_id}: {e}"
                    )
                    summary["errors"].append(str(e))

            summary["predictions_made"] = len(predictions)
            logger.info(
                f"Predictions with signal: {len(predictions)} "
                f"(out of {len(to_research)} researched)"
            )

            # ─── PHASE 4: RISK CHECK & EXECUTE ───
            logger.info("=" * 60)
            logger.info("PHASE 4: RISK EVALUATION & EXECUTION")
            logger.info("=" * 60)

            logger.info(self.risk_mgr.get_status())

            for market, research, prediction in predictions:
                risk_eval = self.risk_mgr.evaluate_trade(prediction, market)

                if risk_eval["warnings"]:
                    for w in risk_eval["warnings"]:
                        logger.warning(f"  Warning: {w}")

                if not risk_eval["approved"]:
                    logger.info(
                        f"  BLOCKED: {prediction.market_question[:50]} — "
                        f"{risk_eval['reason']}"
                    )
                    self.notifier.send_trade_blocked(
                        market, prediction, risk_eval["reason"]
                    )
                    summary["trades_blocked"] += 1
                    continue

                # Execute trade
                trade = self._execute_trade(
                    market, prediction, risk_eval["amount"]
                )
                if trade:
                    self.risk_mgr.record_trade(trade, market)
                    summary["trades_approved"] += 1
                    summary["total_amount_bet"] += trade.amount

                    # Send Telegram alert
                    self.notifier.send_trade_alert(
                        market, prediction,
                        risk_eval["amount"], risk_eval["risk_score"],
                    )

                    logger.info(
                        f"  TRADE PLACED: {trade.side} on "
                        f'"{market.question[:50]}" '
                        f"— ${trade.amount:.2f} @ {trade.entry_price:.3f}"
                    )

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            summary["errors"].append(str(e))

        summary["duration_seconds"] = (
            datetime.utcnow() - cycle_start
        ).total_seconds()

        # Send scan summary to Telegram
        self.notifier.send_scan_summary(
            summary["markets_scanned"],
            summary["markets_flagged"],
            summary["predictions_made"],
            summary["trades_approved"],
            summary["trades_blocked"],
        )

        logger.info("=" * 60)
        logger.info(f"CYCLE COMPLETE: {json.dumps(summary, indent=2)}")
        logger.info("=" * 60)

        return summary

    def _execute_trade(
        self, market: Market, prediction, amount: float
    ) -> Optional[Trade]:
        """Place a trade (simulated or live)."""
        trade = Trade(
            trade_id=str(uuid4())[:8],
            market_id=market.market_id,
            market_question=market.question,
            side=prediction.recommended_side,
            amount=amount,
            entry_price=market.current_price,
            predicted_probability=prediction.predicted_probability,
            confidence=prediction.confidence,
            edge=prediction.edge,
        )

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would place: {trade.side} ${amount:.2f}")
        else:
            # Live execution would go here (Polymarket CLOB API)
            logger.info(
                f"  Placing live trade: {trade.side} ${amount:.2f} "
                f"on {market.market_id}"
            )
            # TODO: Integrate with Polymarket CLOB for live execution

        self._trades.append(trade.to_dict())
        self._save_trades()

        return trade

    def resolve_trade(
        self, trade_id: str, outcome: str, market: Optional[Market] = None
    ):
        """
        Resolve a trade (win/loss) and trigger loss analysis if needed.

        Args:
            trade_id: The trade to resolve
            outcome: "win" or "loss"
            market: Optional market data for analysis
        """
        # Find the trade
        trade_data = None
        for t in self._trades:
            if t.get("trade_id") == trade_id:
                trade_data = t
                break

        if not trade_data:
            logger.error(f"Trade {trade_id} not found")
            return

        # Calculate PnL
        amount = trade_data["amount"]
        entry = trade_data["entry_price"]

        if outcome == "win":
            pnl = amount * ((1.0 / entry) - 1)  # profit from binary option
            trade_data["outcome"] = "win"
        else:
            pnl = -amount  # lose the bet amount
            trade_data["outcome"] = "loss"

        trade_data["pnl"] = pnl
        trade_data["resolved_at"] = datetime.utcnow().isoformat()

        # Update risk manager
        self.risk_mgr.resolve_trade(trade_id, pnl if outcome == "win" else pnl)

        # Update prediction agent training data
        trade_obj = Trade(**{
            k: v for k, v in trade_data.items()
            if k in Trade.__dataclass_fields__
        })

        features = trade_data.get("features_used", {})
        if features:
            self.predictor.add_training_sample(
                features=features,
                outcome=1 if outcome == "win" else 0,
                market_id=trade_data["market_id"],
            )

        # ─── LOSS ANALYSIS ───
        if outcome == "loss":
            logger.info("=" * 60)
            logger.info("LOSS DETECTED — Running analysis...")
            logger.info("=" * 60)

            # Reconstruct objects for analysis
            if market is None:
                market = Market(
                    market_id=trade_data["market_id"],
                    question=trade_data.get("market_question", ""),
                )

            research = ResearchResult(market_id=market.market_id)
            prediction = Prediction(
                market_id=market.market_id,
                market_question=trade_data.get("market_question", ""),
                market_price=entry,
                predicted_probability=trade_data.get(
                    "predicted_probability", 0.5
                ),
                confidence=trade_data.get("confidence", 0),
                edge=trade_data.get("edge", 0),
            )

            lesson = self.loss_analyzer.analyze_loss(
                trade_obj, market, research, prediction
            )

            trade_data["loss_analysis"] = lesson.description
            trade_data["lessons_learned"] = [lesson.to_dict()]

            logger.info(f"\nLesson learned: {lesson.category}")
            logger.info(f"New rule: {lesson.rule_added}")
            logger.info(f"Description: {lesson.description[:200]}")

            # Notify loss
            self.notifier.send_loss_alert(
                trade_obj, pnl, self.risk_mgr.bankroll,
                lesson.description,
            )
        else:
            # Notify win
            self.notifier.send_win_alert(
                trade_obj, pnl, self.risk_mgr.bankroll,
            )

        self._save_trades()
        logger.info(
            f"Trade {trade_id} resolved: {outcome} | PnL: ${pnl:+.2f}"
        )

    def run_continuous(self, max_cycles: int = 0):
        """
        Run trading cycles continuously.

        Args:
            max_cycles: Stop after N cycles (0 = run forever)
        """
        cycle_count = 0

        logger.info("Starting continuous trading loop...")
        logger.info(f"Scan interval: {self.config.scan_interval_seconds}s")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(self.risk_mgr.get_status())

        self.notifier.send_startup(self.risk_mgr.bankroll, self.dry_run)

        while True:
            cycle_count += 1
            logger.info(f"\n{'#' * 60}")
            logger.info(f"# CYCLE {cycle_count}")
            logger.info(f"{'#' * 60}\n")

            try:
                summary = self.run_cycle()

                logger.info(
                    f"Cycle {cycle_count} done: "
                    f"{summary['trades_approved']} trades, "
                    f"${summary['total_amount_bet']:.2f} bet"
                )

            except KeyboardInterrupt:
                logger.info("Interrupted — shutting down gracefully")
                break
            except Exception as e:
                logger.error(f"Cycle {cycle_count} crashed: {e}", exc_info=True)

            if max_cycles and cycle_count >= max_cycles:
                logger.info(f"Completed {max_cycles} cycles — stopping")
                break

            # Wait before next cycle
            logger.info(
                f"Sleeping {self.config.scan_interval_seconds}s "
                f"before next cycle..."
            )
            try:
                time.sleep(self.config.scan_interval_seconds)
            except KeyboardInterrupt:
                logger.info("Interrupted during sleep — shutting down")
                break

    def get_dashboard(self) -> str:
        """Generate a text dashboard of the system status."""
        lines = [
            "=" * 60,
            "  MULTI-AGENT TRADING SYSTEM — DASHBOARD",
            "=" * 60,
            "",
            self.risk_mgr.get_status(),
            "",
        ]

        # Trade history
        total_trades = len(self._trades)
        wins = sum(1 for t in self._trades if t.get("outcome") == "win")
        losses = sum(1 for t in self._trades if t.get("outcome") == "loss")
        pending = sum(1 for t in self._trades if t.get("outcome") == "pending")
        total_pnl = sum(t.get("pnl", 0) for t in self._trades)

        lines.extend([
            f"=== TRADE HISTORY ===",
            f"Total trades: {total_trades}",
            f"Wins: {wins} | Losses: {losses} | Pending: {pending}",
            f"Win rate: {wins/(wins+losses)*100:.1f}%" if (wins + losses) > 0 else "Win rate: N/A",
            f"Total PnL: ${total_pnl:+.2f}",
            "",
        ])

        # Recent trades
        if self._trades:
            lines.append("Recent trades:")
            for t in self._trades[-5:]:
                outcome_str = t.get("outcome", "pending").upper()
                pnl = t.get("pnl", 0)
                lines.append(
                    f"  [{t.get('trade_id', '?')}] {t.get('side', '?')} "
                    f"${t.get('amount', 0):.2f} — {outcome_str} "
                    f"(${pnl:+.2f})"
                )
                lines.append(
                    f"    {t.get('market_question', 'Unknown')[:60]}"
                )
            lines.append("")

        # Lessons
        lines.append(self.loss_analyzer.get_lessons_summary())

        return "\n".join(lines)
