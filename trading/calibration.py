"""
Probability calibration layer for the forex trading system.

Tracks predicted vs actual outcomes over time and applies Platt scaling
to map raw model predictions to well-calibrated probabilities.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


class CalibrationLayer:
    """Maps raw prediction probabilities to calibrated probabilities using Platt scaling.

    Platt scaling fits a logistic function P(y=1|f) = 1 / (1 + exp(A*f + B))
    to historical (predicted_prob, actual_outcome) pairs, where A and B are
    learned parameters.
    """

    MIN_SAMPLES_TO_FIT = 20

    def __init__(self, persistence_path: Optional[str] = None):
        # Platt scaling parameters: P = 1 / (1 + exp(A*x + B))
        # Defaults are identity-like: A=-1, B=0 maps sigmoid(x) ~ x for moderate x
        self._a: float = -1.0
        self._b: float = 0.0
        self._fitted: bool = False

        # Historical data
        self._predictions: list[float] = []
        self._outcomes: list[int] = []

        # Persistence
        self._persistence_path: Optional[Path] = (
            Path(persistence_path) if persistence_path else None
        )

        # Try to load existing params
        if self._persistence_path and self._persistence_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, raw_probability: float) -> float:
        """Map a raw prediction probability to a calibrated probability.

        Uses Platt scaling: P_calibrated = 1 / (1 + exp(A * raw + B))

        Args:
            raw_probability: Raw model output in [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        raw_probability = float(np.clip(raw_probability, 1e-7, 1.0 - 1e-7))
        # Convert probability to log-odds (the "f" in Platt scaling)
        logit = math.log(raw_probability / (1.0 - raw_probability))
        calibrated = 1.0 / (1.0 + math.exp(self._a * logit + self._b))
        return float(np.clip(calibrated, 0.0, 1.0))

    def add_outcome(self, predicted_prob: float, actual_outcome: int) -> None:
        """Record a prediction-outcome pair.

        Args:
            predicted_prob: The raw probability that was predicted (0-1).
            actual_outcome: Whether the event occurred (1) or not (0).
        """
        if not (0.0 <= predicted_prob <= 1.0):
            raise ValueError(f"predicted_prob must be in [0, 1], got {predicted_prob}")
        if actual_outcome not in (0, 1):
            raise ValueError(f"actual_outcome must be 0 or 1, got {actual_outcome}")

        self._predictions.append(float(predicted_prob))
        self._outcomes.append(int(actual_outcome))

        # Auto-fit when we cross the threshold
        if len(self._predictions) >= self.MIN_SAMPLES_TO_FIT and not self._fitted:
            self.fit()

    def fit(self) -> bool:
        """Refit the Platt scaling calibration curve using collected data.

        Uses Newton's method to find optimal A, B that minimize the
        negative log-likelihood of the logistic model.

        Returns:
            True if fitting succeeded, False if insufficient data.
        """
        n = len(self._predictions)
        if n < self.MIN_SAMPLES_TO_FIT:
            return False

        predictions = np.array(self._predictions, dtype=np.float64)
        outcomes = np.array(self._outcomes, dtype=np.float64)

        # Clip predictions away from 0 and 1 for numerical stability
        predictions = np.clip(predictions, 1e-7, 1.0 - 1e-7)

        # Convert to log-odds space
        f = np.log(predictions / (1.0 - predictions))

        # Target probabilities with Platt's label regularisation
        # to avoid overfitting: t_i = (y_i * N+ + 1) / (N+ + 2) for positives
        n_pos = np.sum(outcomes)
        n_neg = n - n_pos
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(outcomes == 1, t_pos, t_neg)

        # Newton's method for Platt scaling
        a = 0.0
        b = math.log((n_neg + 1.0) / (n_pos + 1.0))

        max_iter = 100
        min_step = 1e-10
        sigma = 1e-12  # regularisation for Hessian

        for _ in range(max_iter):
            # Current predictions
            exponents = np.clip(a * f + b, -500, 500)
            p = 1.0 / (1.0 + np.exp(exponents))
            p = np.clip(p, 1e-15, 1.0 - 1e-15)

            # Gradient
            d1 = targets - p
            g_a = np.dot(f, d1)
            g_b = np.sum(d1)

            # Hessian
            d2 = p * (1.0 - p)
            h_aa = -np.dot(f * f, d2) - sigma
            h_bb = -np.sum(d2) - sigma
            h_ab = -np.dot(f, d2)

            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-15:
                break

            # Newton step
            da = -(h_bb * g_a - h_ab * g_b) / det
            db = -(h_aa * g_b - h_ab * g_a) / det

            a += da
            b += db

            if abs(da) < min_step and abs(db) < min_step:
                break

        self._a = float(a)
        self._b = float(b)
        self._fitted = True

        if self._persistence_path:
            self.save()

        return True

    def get_calibration_stats(self) -> dict:
        """Compute calibration quality metrics.

        Returns:
            Dictionary containing:
            - brier_score: Mean squared error of probability predictions (lower is better).
            - reliability_bins: List of dicts with bin midpoint, mean predicted prob,
              and observed frequency for a reliability diagram.
            - overconfidence_metric: Average (predicted - actual) for bins where predicted > 0.5.
              Positive means overconfident.
            - n_samples: Number of recorded outcomes.
        """
        n = len(self._predictions)
        if n == 0:
            return {
                "brier_score": None,
                "reliability_bins": [],
                "overconfidence_metric": None,
                "n_samples": 0,
            }

        predictions = np.array(self._predictions, dtype=np.float64)
        outcomes = np.array(self._outcomes, dtype=np.float64)

        # Brier score
        brier = float(np.mean((predictions - outcomes) ** 2))

        # Reliability diagram (10 bins)
        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        reliability_bins = []
        overconf_diffs = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (predictions >= lo) & (predictions < hi) if i < n_bins - 1 else (predictions >= lo) & (predictions <= hi)
            count = int(np.sum(mask))
            if count == 0:
                continue

            mean_pred = float(np.mean(predictions[mask]))
            observed_freq = float(np.mean(outcomes[mask]))
            midpoint = float((lo + hi) / 2.0)

            reliability_bins.append({
                "bin_midpoint": round(midpoint, 2),
                "mean_predicted": round(mean_pred, 4),
                "observed_frequency": round(observed_freq, 4),
                "count": count,
            })

            if mean_pred > 0.5:
                overconf_diffs.append(mean_pred - observed_freq)

        overconfidence = (
            float(np.mean(overconf_diffs)) if overconf_diffs else 0.0
        )

        return {
            "brier_score": round(brier, 6),
            "reliability_bins": reliability_bins,
            "overconfidence_metric": round(overconfidence, 6),
            "n_samples": n,
        }

    def is_overconfident(self) -> bool:
        """Check whether predictions are systematically overconfident.

        Returns True if the overconfidence metric exceeds a threshold,
        meaning the model's high-confidence predictions (> 0.5) consistently
        overestimate the true probability of the positive outcome.
        """
        stats = self.get_calibration_stats()
        if stats["overconfidence_metric"] is None:
            return False
        return stats["overconfidence_metric"] > 0.05

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save calibration parameters and history to a JSON file."""
        target = Path(path) if path else self._persistence_path
        if target is None:
            raise ValueError("No persistence path configured")

        data = {
            "a": self._a,
            "b": self._b,
            "fitted": self._fitted,
            "predictions": self._predictions,
            "outcomes": self._outcomes,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """Load calibration parameters and history from a JSON file."""
        target = Path(path) if path else self._persistence_path
        if target is None:
            raise ValueError("No persistence path configured")
        if not target.exists():
            raise FileNotFoundError(f"Calibration file not found: {target}")

        with open(target, "r") as f:
            data = json.load(f)

        self._a = float(data["a"])
        self._b = float(data["b"])
        self._fitted = bool(data["fitted"])
        self._predictions = [float(p) for p in data.get("predictions", [])]
        self._outcomes = [int(o) for o in data.get("outcomes", [])]
