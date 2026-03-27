"""
Prediction Agent — XGBoost + LLM probability calibration.

Combines machine learning features with LLM reasoning to estimate
true market probability vs. market price. Only fires when confidence
exceeds the configured threshold.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

from trading.config import PredictionConfig, SystemConfig
from trading.models import Market, ResearchResult, Prediction

logger = logging.getLogger(__name__)

# Try to import XGBoost (optional heavy dependency)
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed — using fallback model")

# Try to import Anthropic SDK for LLM calibration
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("Anthropic SDK not installed — LLM calibration disabled")


class PredictionAgent:
    """
    Calibrates true market probability using:
    1. XGBoost model trained on market features
    2. LLM (Claude) for qualitative reasoning
    3. Weighted ensemble of both

    Only recommends trades when confidence > threshold AND edge > minimum.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pred_cfg = self.config.prediction
        self._model = None
        self._training_data: list[dict] = []
        self._load_model()
        self._load_training_data()

    def _load_model(self):
        """Load saved XGBoost model if it exists."""
        if not HAS_XGBOOST:
            return

        if os.path.exists(self.pred_cfg.model_save_path):
            try:
                self._model = xgb.XGBClassifier()
                self._model.load_model(self.pred_cfg.model_save_path)
                logger.info("Loaded saved XGBoost model")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self._model = None

    def _load_training_data(self):
        """Load historical training data for model updates."""
        try:
            with open(self.pred_cfg.training_data_path) as f:
                self._training_data = json.load(f)
                logger.info(
                    f"Loaded {len(self._training_data)} training samples"
                )
        except (FileNotFoundError, json.JSONDecodeError):
            self._training_data = []

    def _save_training_data(self):
        """Persist training data."""
        os.makedirs(os.path.dirname(self.pred_cfg.training_data_path), exist_ok=True)
        with open(self.pred_cfg.training_data_path, "w") as f:
            json.dump(self._training_data, f, indent=2)

    def predict(
        self, market: Market, research: ResearchResult
    ) -> Optional[Prediction]:
        """
        Generate a calibrated prediction for a market.
        Returns None if confidence is below threshold.
        """
        logger.info(f"Predicting: {market.question[:60]}...")

        # Step 1: Build feature vector
        features = self._build_features(market, research)

        # Step 2: XGBoost prediction
        xgb_prob = self._xgboost_predict(features)
        logger.info(f"  XGBoost probability: {xgb_prob:.3f}")

        # Step 3: LLM calibration
        llm_prob = self._llm_predict(market, research)
        logger.info(f"  LLM probability: {llm_prob:.3f}")

        # Step 4: Ensemble — weight XGBoost more when we have training data
        if len(self._training_data) >= 50:
            xgb_weight = 0.6
        elif len(self._training_data) >= 20:
            xgb_weight = 0.4
        else:
            xgb_weight = 0.2  # lean on LLM when little training data

        llm_weight = 1.0 - xgb_weight
        predicted_prob = (xgb_prob * xgb_weight) + (llm_prob * llm_weight)

        # Step 5: Calculate edge and confidence
        edge = predicted_prob - market.current_price
        abs_edge = abs(edge)

        # Confidence based on agreement between models and signal strength
        model_agreement = 1.0 - abs(xgb_prob - llm_prob)
        signal_strength = abs(research.combined_sentiment)
        source_coverage = min(research.source_count / 20, 1.0)

        confidence = (
            model_agreement * 0.4
            + signal_strength * 0.3
            + source_coverage * 0.15
            + min(abs_edge / 0.2, 1.0) * 0.15
        )

        # Determine recommended side
        if edge > 0:
            side = "Yes"  # market underpricing — buy Yes
        else:
            side = "No"  # market overpricing — buy No
            edge = market.current_price - predicted_prob  # flip for No side

        logger.info(
            f"  Ensemble: {predicted_prob:.3f} | "
            f"Edge: {edge:.3f} | Confidence: {confidence:.3f}"
        )

        # Step 6: Check thresholds
        if confidence < self.pred_cfg.confidence_threshold:
            logger.info(
                f"  SKIP: Confidence {confidence:.3f} < "
                f"threshold {self.pred_cfg.confidence_threshold}"
            )
            return None

        if abs_edge < self.pred_cfg.edge_threshold:
            logger.info(
                f"  SKIP: Edge {abs_edge:.3f} < "
                f"threshold {self.pred_cfg.edge_threshold}"
            )
            return None

        # Build reasoning string
        reasoning = (
            f"XGBoost ({xgb_weight:.0%}): {xgb_prob:.3f} | "
            f"LLM ({llm_weight:.0%}): {llm_prob:.3f} | "
            f"Ensemble: {predicted_prob:.3f}\n"
            f"Market price: {market.current_price:.3f} | "
            f"Edge: {edge:.3f} | Confidence: {confidence:.3f}\n"
            f"Sentiment: {research.sentiment_label} "
            f"({research.combined_sentiment:+.2f})\n"
            f"Narrative alignment: {research.narrative_vs_odds_alignment:.1%}\n"
            f"Recommendation: BUY {side} @ {market.current_price:.3f}"
        )

        prediction = Prediction(
            market_id=market.market_id,
            market_question=market.question,
            market_price=market.current_price,
            predicted_probability=predicted_prob,
            xgboost_probability=xgb_prob,
            llm_probability=llm_prob,
            confidence=confidence,
            edge=edge,
            recommended_side=side,
            reasoning=reasoning,
            features_used=features,
        )

        logger.info(f"  SIGNAL: BUY {side} — edge={edge:.3f}, conf={confidence:.3f}")
        return prediction

    def _build_features(
        self, market: Market, research: ResearchResult
    ) -> dict:
        """Build feature vector from market + research data."""
        return {
            "market_price": market.current_price,
            "volume_24h": market.volume_24h,
            "liquidity": market.liquidity,
            "spread": market.spread,
            "price_change_1h": market.price_change_1h,
            "price_change_24h": market.price_change_24h,
            "twitter_sentiment": research.twitter_sentiment,
            "reddit_sentiment": research.reddit_sentiment,
            "rss_sentiment": research.rss_sentiment,
            "combined_sentiment": research.combined_sentiment,
            "source_count": research.source_count,
            "narrative_alignment": research.narrative_vs_odds_alignment,
            "volume_liquidity_ratio": (
                market.volume_24h / max(market.liquidity, 1)
            ),
            "sentiment_price_divergence": abs(
                (research.combined_sentiment + 1) / 2 - market.current_price
            ),
            "is_flagged": 1.0 if market.flagged else 0.0,
        }

    def _xgboost_predict(self, features: dict) -> float:
        """Get XGBoost probability prediction."""
        if not HAS_XGBOOST or self._model is None:
            return self._fallback_predict(features)

        try:
            feature_array = np.array(
                [list(features.values())], dtype=np.float32
            )
            proba = self._model.predict_proba(feature_array)
            return float(proba[0][1])  # probability of class 1 (Yes)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return self._fallback_predict(features)

    def _fallback_predict(self, features: dict) -> float:
        """
        Fallback prediction when XGBoost is unavailable.
        Uses a simple heuristic based on features.
        """
        base = features["market_price"]

        # Sentiment adjustment
        sentiment = features["combined_sentiment"]
        sentiment_adj = sentiment * 0.15

        # Divergence signal
        divergence = features["sentiment_price_divergence"]
        if divergence > 0.2:
            # Big divergence — sentiment disagrees with market
            div_adj = sentiment * 0.1
        else:
            div_adj = 0

        # Volume/liquidity health
        vol_liq = features["volume_liquidity_ratio"]
        if vol_liq > 5:
            # Unusually high — might be informed trading
            vol_adj = 0.05 if sentiment > 0 else -0.05
        else:
            vol_adj = 0

        predicted = base + sentiment_adj + div_adj + vol_adj
        return max(0.01, min(0.99, predicted))

    def _llm_predict(self, market: Market, research: ResearchResult) -> float:
        """Use Claude to get a calibrated probability estimate."""
        if not HAS_ANTHROPIC or not self.pred_cfg.anthropic_api_key:
            # Fallback to sentiment-adjusted estimate
            base = market.current_price
            adj = research.combined_sentiment * 0.1
            return max(0.01, min(0.99, base + adj))

        try:
            client = anthropic.Anthropic(
                api_key=self.pred_cfg.anthropic_api_key
            )

            prompt = f"""You are a prediction market calibration expert. Estimate the TRUE probability for this market.

Market Question: {market.question}
Current Market Price: {market.current_price:.3f} ({market.current_price*100:.1f}% implied probability)
24h Volume: ${market.volume_24h:,.0f}
Liquidity: ${market.liquidity:,.0f}

Research Summary:
{research.narrative_summary}

Sentiment: {research.sentiment_label} (score: {research.combined_sentiment:+.2f})
Sources analyzed: {research.source_count}

Key signals:
{chr(10).join(f'- {s}' for s in research.key_signals[:5])}

Based on all available information, what is the TRUE probability that the answer is YES?
Respond with ONLY a number between 0.01 and 0.99 (e.g., 0.65).
Think carefully about base rates, current evidence, and potential biases in the market."""

            response = client.messages.create(
                model=self.pred_cfg.llm_model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the probability from response
            text = response.content[0].text.strip()
            # Extract first number that looks like a probability
            import re

            match = re.search(r"0\.\d+", text)
            if match:
                prob = float(match.group())
                return max(0.01, min(0.99, prob))

            logger.warning(f"Could not parse LLM probability from: {text}")

        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")

        # Fallback
        base = market.current_price
        adj = research.combined_sentiment * 0.1
        return max(0.01, min(0.99, base + adj))

    def add_training_sample(
        self, features: dict, outcome: int, market_id: str
    ):
        """Add a resolved trade outcome as training data."""
        sample = {
            "features": features,
            "outcome": outcome,
            "market_id": market_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._training_data.append(sample)
        self._save_training_data()
        logger.info(
            f"Added training sample (total: {len(self._training_data)})"
        )

        # Retrain if we have enough data
        if len(self._training_data) >= 20 and len(self._training_data) % 10 == 0:
            self._retrain()

    def _retrain(self):
        """Retrain XGBoost model on accumulated data."""
        if not HAS_XGBOOST or len(self._training_data) < 20:
            return

        logger.info(
            f"Retraining XGBoost on {len(self._training_data)} samples..."
        )

        try:
            X = np.array(
                [list(s["features"].values()) for s in self._training_data],
                dtype=np.float32,
            )
            y = np.array(
                [s["outcome"] for s in self._training_data], dtype=np.int32
            )

            self._model = xgb.XGBClassifier(
                n_estimators=self.pred_cfg.xgboost_n_estimators,
                max_depth=self.pred_cfg.xgboost_max_depth,
                learning_rate=self.pred_cfg.xgboost_learning_rate,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            self._model.fit(X, y)

            os.makedirs(
                os.path.dirname(self.pred_cfg.model_save_path), exist_ok=True
            )
            self._model.save_model(self.pred_cfg.model_save_path)
            logger.info("XGBoost model retrained and saved")

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
