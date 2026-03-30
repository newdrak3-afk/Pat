"""
Loss Analyzer — Post-loss root cause analysis + system self-improvement.

After every loss:
1. Figures out what went wrong
2. Categorizes the failure
3. Generates a corrective rule
4. Saves the lesson
5. Updates the system so it doesn't make the same mistake twice
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

from trading.config import SystemConfig
from trading.models import Trade, Market, ResearchResult, Prediction, Lesson

logger = logging.getLogger(__name__)

# Try to import Anthropic for LLM-powered analysis
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Known failure categories and their associated corrective actions
FAILURE_CATEGORIES = {
    "sentiment_mismatch": {
        "description": "Sentiment signals disagreed with actual outcome",
        "default_rule": "Increase sentiment threshold for this category",
    },
    "low_liquidity_trap": {
        "description": "Market had deceptively low liquidity, causing bad fill",
        "default_rule": "min_liquidity:10000",
    },
    "late_entry": {
        "description": "Entered too late — information was already priced in",
        "default_rule": "Require fresher signals (< 2h old)",
    },
    "black_swan": {
        "description": "Unexpected event invalidated all analysis",
        "default_rule": "Reduce position size for similar markets",
    },
    "overconfidence": {
        "description": "Model was too confident on weak evidence",
        "default_rule": "Raise confidence threshold",
    },
    "correlated_loss": {
        "description": "Multiple correlated positions lost simultaneously",
        "default_rule": "Tighten correlation limits",
    },
    "spread_slippage": {
        "description": "Wide spread caused worse entry than expected",
        "default_rule": "Lower max spread threshold",
    },
    "narrative_trap": {
        "description": "Strong narrative turned out to be misleading noise",
        "default_rule": "Require multiple independent signal sources",
    },
    "model_drift": {
        "description": "Market regime changed, model predictions drifted",
        "default_rule": "Trigger model retrain with recent data weighted higher",
    },
    "unknown": {
        "description": "Root cause unclear",
        "default_rule": "Flag for manual review",
    },
}


class LossAnalyzer:
    """
    Analyzes every losing trade, categorizes the failure,
    generates a corrective rule, and updates the system.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._lessons: list[dict] = []
        self._load_lessons()

    def _load_lessons(self):
        """Load existing lessons."""
        try:
            with open(self.config.lessons_log) as f:
                self._lessons = json.load(f)
                logger.info(f"Loaded {len(self._lessons)} existing lessons")
        except (FileNotFoundError, json.JSONDecodeError):
            self._lessons = []

    def _save_lessons(self):
        """Persist lessons to disk."""
        os.makedirs(os.path.dirname(self.config.lessons_log), exist_ok=True)
        with open(self.config.lessons_log, "w") as f:
            json.dump(self._lessons, f, indent=2)

    def analyze_loss(
        self,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
    ) -> Lesson:
        """
        Analyze a losing trade and generate a lesson.

        Steps:
        1. Categorize the failure
        2. Generate root cause analysis
        3. Create corrective rule
        4. Save lesson
        5. Return lesson for system update
        """
        logger.info(
            f"Analyzing loss: trade {trade.trade_id} on "
            f"{market.question[:50]}..."
        )
        logger.info(f"  PnL: ${trade.pnl:.2f}")

        # Step 1: Categorize
        category = self._categorize_failure(trade, market, research, prediction)
        logger.info(f"  Category: {category}")

        # Step 2: Root cause analysis
        analysis = self._root_cause_analysis(
            trade, market, research, prediction, category
        )
        logger.info(f"  Analysis: {analysis[:100]}...")

        # Step 3: Generate corrective rule
        rule = self._generate_rule(
            category, trade, market, research, prediction
        )
        logger.info(f"  New rule: {rule}")

        # Step 4: Create lesson
        lesson = Lesson(
            lesson_id=str(uuid4())[:8],
            trade_id=trade.trade_id,
            market_id=trade.market_id,
            category=category,
            description=analysis,
            rule_added=rule,
        )

        # Step 5: Save
        self._lessons.append(lesson.to_dict())
        self._save_lessons()

        # Update trade with analysis
        trade.loss_analysis = analysis
        trade.lessons_learned = [lesson.to_dict()]

        logger.info(
            f"Lesson saved: [{lesson.lesson_id}] {category} — {rule}"
        )
        return lesson

    def _categorize_failure(
        self,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
    ) -> str:
        """Determine the category of failure."""

        # ─── Forex-specific categorization ───
        is_forex = trade.side in ("buy", "sell")
        if is_forex:
            return self._categorize_forex_loss(trade, prediction)

        # ─── Prediction market categorization ───
        # Check for sentiment mismatch
        if trade.side == "Yes" and research.combined_sentiment > 0.2:
            return "sentiment_mismatch"
        if trade.side == "No" and research.combined_sentiment < -0.2:
            return "sentiment_mismatch"

        # Check for low liquidity
        if market.liquidity < 10_000:
            return "low_liquidity_trap"

        # Check for wide spread
        if market.spread > 5.0:
            return "spread_slippage"

        # Check for overconfidence (high confidence, big miss)
        if prediction.confidence > 0.8 and abs(trade.pnl) > trade.amount * 0.5:
            return "overconfidence"

        # Check for narrative trap (strong signals but wrong outcome)
        if abs(research.combined_sentiment) > 0.4 and research.source_count > 10:
            return "narrative_trap"

        # Check for low source count (weak evidence)
        if research.source_count < 5:
            return "late_entry"

        # Try LLM categorization if available
        if HAS_ANTHROPIC and self.config.prediction.anthropic_api_key:
            return self._llm_categorize(trade, market, research, prediction)

        return "unknown"

    def _categorize_forex_loss(self, trade: Trade, prediction: Prediction) -> str:
        """Categorize a forex trading loss based on available info."""
        confidence = prediction.confidence if prediction else 0

        # Overconfidence: high confidence but still lost
        if confidence > 0.7:
            return "overconfidence"

        # Model drift: moderate confidence, market moved against
        if confidence > 0.5:
            return "model_drift"

        # Low confidence trade that lost — shouldn't have been taken
        if confidence < 0.45:
            return "late_entry"

        return "sentiment_mismatch"

    def _llm_categorize(
        self,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
    ) -> str:
        """Use LLM to categorize the failure."""
        try:
            client = anthropic.Anthropic(
                api_key=self.config.prediction.anthropic_api_key
            )

            categories_str = "\n".join(
                f"- {k}: {v['description']}"
                for k, v in FAILURE_CATEGORIES.items()
            )

            prompt = f"""Categorize this trading loss into one of these categories:

{categories_str}

Trade details:
- Market: {market.question}
- Side: {trade.side} @ {trade.entry_price:.3f}
- PnL: ${trade.pnl:.2f}
- Predicted probability: {prediction.predicted_probability:.3f}
- Market price at entry: {prediction.market_price:.3f}
- Confidence: {prediction.confidence:.3f}
- Sentiment: {research.sentiment_label} ({research.combined_sentiment:+.2f})
- Sources: {research.source_count}
- Liquidity: ${market.liquidity:,.0f}
- Spread: {market.spread:.1f}%

Respond with ONLY the category name (e.g., "sentiment_mismatch")."""

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )

            category = response.content[0].text.strip().lower()
            if category in FAILURE_CATEGORIES:
                return category

        except Exception as e:
            logger.warning(f"LLM categorization failed: {e}")

        return "unknown"

    def _root_cause_analysis(
        self,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
        category: str,
    ) -> str:
        """Generate detailed root cause analysis."""

        # Use LLM if available
        if HAS_ANTHROPIC and self.config.prediction.anthropic_api_key:
            return self._llm_root_cause(
                trade, market, research, prediction, category
            )

        # Heuristic analysis
        is_forex = trade.side in ("buy", "sell")

        lines = [
            f"LOSS ANALYSIS — Trade {trade.trade_id}",
            f"Category: {category}",
            f"",
        ]

        if is_forex:
            pair = trade.market_id.replace("_", "/") if trade.market_id else "Unknown"
            lines.extend([
                f"Pair: {pair}",
                f"Side: {trade.side.upper()} @ {trade.entry_price:.5f}",
                f"PnL: ${trade.pnl:.2f}",
                f"Confidence: {prediction.confidence:.0%}" if prediction else "",
                f"",
                f"What happened:",
            ])
        else:
            lines.extend([
                f"Market: {market.question}",
                f"Placed: {trade.side} @ {trade.entry_price:.3f} "
                f"(${trade.amount:.2f})",
                f"Result: PnL ${trade.pnl:.2f}",
                f"",
                f"What happened:",
            ])

        cat_info = FAILURE_CATEGORIES.get(category, FAILURE_CATEGORIES["unknown"])
        lines.append(f"  {cat_info['description']}")

        # Specific analysis per category
        if category == "sentiment_mismatch":
            if is_forex:
                lines.append(
                    f"  HTF trend aligned but price moved against the trade. "
                    f"Possible late entry or news reversal."
                )
            else:
                lines.append(
                    f"  Sentiment was {research.sentiment_label} "
                    f"({research.combined_sentiment:+.2f}) but outcome went "
                    f"against the narrative."
                )
                lines.append(
                    f"  The market was correct at {prediction.market_price:.3f} "
                    f"while our model predicted {prediction.predicted_probability:.3f}."
                )

        elif category == "low_liquidity_trap":
            lines.append(
                f"  Liquidity was only ${market.liquidity:,.0f}. "
                f"Price may have been manipulated or unreliable."
            )

        elif category == "overconfidence":
            conf = prediction.confidence if prediction else 0
            lines.append(
                f"  Model confidence was {conf:.0%} "
                f"but the prediction was wrong. Edge was overstated."
            )

        elif category == "spread_slippage":
            lines.append(
                f"  Spread was {market.spread:.1f}%, eating into potential profits."
            )

        elif category == "narrative_trap":
            lines.append(
                f"  {research.source_count} sources showed strong signal, "
                f"but the narrative was misleading."
            )

        elif category == "model_drift":
            lines.append(
                f"  The market regime may have shifted. Strategy signals "
                f"that worked before are no longer effective in current conditions."
            )

        elif category == "late_entry":
            lines.append(
                f"  Low confidence entry — signal was weak but trade was still taken. "
                f"Consider raising the confidence threshold."
            )

        if not is_forex and prediction:
            lines.append(
                f"\nPrediction breakdown:"
                f"\n  XGBoost: {prediction.xgboost_probability:.3f}"
                f"\n  LLM: {prediction.llm_probability:.3f}"
                f"\n  Ensemble: {prediction.predicted_probability:.3f}"
                f"\n  Market: {prediction.market_price:.3f}"
            )

        return "\n".join(lines)

    def _llm_root_cause(
        self,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
        category: str,
    ) -> str:
        """Use LLM for detailed root cause analysis."""
        try:
            client = anthropic.Anthropic(
                api_key=self.config.prediction.anthropic_api_key
            )

            prompt = f"""Analyze this losing trade and explain what went wrong in 3-5 sentences.

Market: {market.question}
Category: {category}
Trade: {trade.side} @ {trade.entry_price:.3f} (${trade.amount:.2f})
PnL: ${trade.pnl:.2f}
Prediction: {prediction.predicted_probability:.3f} vs market {prediction.market_price:.3f}
Confidence: {prediction.confidence:.3f}
Sentiment: {research.sentiment_label} ({research.combined_sentiment:+.2f})
Sources: {research.source_count}
Key signals: {'; '.join(research.key_signals[:3])}

Be specific about what the system got wrong and what to do differently."""

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.warning(f"LLM root cause failed: {e}")
            return self._root_cause_analysis(
                trade, market, research, prediction, category
            )

    def _generate_rule(
        self,
        category: str,
        trade: Trade,
        market: Market,
        research: ResearchResult,
        prediction: Prediction,
    ) -> str:
        """Generate a specific corrective rule based on the failure."""
        cat_info = FAILURE_CATEGORIES.get(category, FAILURE_CATEGORIES["unknown"])

        # Generate specific rules based on failure type
        if category == "sentiment_mismatch":
            # Check if this market's category has caused problems before
            same_cat_losses = sum(
                1 for l in self._lessons
                if l.get("category") == "sentiment_mismatch"
            )
            if same_cat_losses >= 2:
                return f"block_category:{market.category}"
            return "raise_sentiment_threshold:0.35"

        elif category == "low_liquidity_trap":
            new_min = max(market.liquidity * 2, 10_000)
            return f"min_liquidity:{new_min}"

        elif category == "overconfidence":
            return "raise_confidence_threshold:0.70"

        elif category == "spread_slippage":
            new_max = max(market.spread * 0.7, 2.0)
            return f"max_spread:{new_max}"

        elif category == "narrative_trap":
            # Block keywords from this market to avoid similar traps
            keywords = market.question.lower().split()
            # Find the most distinctive keyword
            distinctive = [
                w for w in keywords
                if len(w) > 4 and w not in {
                    "will", "does", "market", "price", "before"
                }
            ]
            if distinctive:
                return f"block_keyword:{distinctive[0]}"
            return "require_min_sources:10"

        elif category == "correlated_loss":
            return f"block_category:{market.category}"

        elif category == "model_drift":
            return "trigger_retrain"

        return cat_info.get("default_rule", "flag_for_review")

    def get_lessons_summary(self) -> str:
        """Return a summary of all lessons learned."""
        if not self._lessons:
            return "No lessons learned yet."

        lines = [
            f"=== LESSONS LEARNED ({len(self._lessons)} total) ===",
            "",
        ]

        # Group by category
        by_category: dict[str, list] = {}
        for lesson in self._lessons:
            cat = lesson.get("category", "unknown")
            by_category.setdefault(cat, []).append(lesson)

        for cat, lessons in sorted(
            by_category.items(), key=lambda x: -len(x[1])
        ):
            lines.append(
                f"{cat} ({len(lessons)} occurrences):"
            )
            for l in lessons[-3:]:  # show last 3 per category
                lines.append(f"  Rule: {l.get('rule_added', 'N/A')}")
                desc = l.get("description", "")
                if desc:
                    lines.append(f"  {desc[:100]}...")
            lines.append("")

        return "\n".join(lines)
