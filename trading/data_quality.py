"""Data feed validation for forex trading system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class QualityReport:
    """Result of a data quality check on candle data."""

    is_valid: bool = True
    issues: list[str] = field(default_factory=list)
    gaps_found: int = 0
    stale_data: bool = False
    suspicious_spikes: int = 0


class DataQualityChecker:
    """Validates incoming candle data for completeness and consistency.

    Each candle is expected to be a dict with keys:
        timestamp (ISO string or datetime), open, high, low, close, volume
    """

    # Typical candle intervals in seconds (auto-detected from data)
    COMMON_INTERVALS = [60, 300, 900, 3600, 14400, 86400]

    def __init__(self, atr_spike_threshold: float = 5.0):
        self.atr_spike_threshold = atr_spike_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_candles(self, candles: list[dict]) -> QualityReport:
        """Run all quality checks on a list of candle dicts."""
        report = QualityReport()

        if not candles:
            report.is_valid = False
            report.issues.append("Empty candle list")
            return report

        timestamps = self._extract_timestamps(candles)

        self._check_duplicates(timestamps, report)
        self._check_gaps(timestamps, report)
        self._check_staleness(timestamps, report)
        self._check_ohlc_consistency(candles, report)
        self._check_zero_volume(candles, report)
        self._check_price_spikes(candles, report)

        if report.issues:
            report.is_valid = False

        return report

    def is_fresh(self, candles: list[dict], max_age_minutes: int = 30) -> bool:
        """Return True if the most recent candle is within *max_age_minutes*."""
        if not candles:
            return False
        last_ts = self._to_datetime(candles[-1].get("timestamp", candles[-1].get("time", "")))
        age = (datetime.now(timezone.utc) - last_ts).total_seconds()
        return age <= max_age_minutes * 60

    def remove_outliers(self, candles: list[dict]) -> list[dict]:
        """Return a copy of *candles* with suspicious spikes removed."""
        if len(candles) < 15:
            return list(candles)

        closes = np.array([float(c["close"]) for c in candles])
        atr = self._compute_atr(candles)

        if atr == 0:
            return list(candles)

        clean: list[dict] = [candles[0]]
        for i in range(1, len(candles)):
            move = abs(closes[i] - closes[i - 1])
            if move <= self.atr_spike_threshold * atr:
                clean.append(candles[i])

        return clean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(ts) -> datetime:
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts
        return datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)

    def _extract_timestamps(self, candles: list[dict]) -> list[datetime]:
        return [self._to_datetime(c.get("timestamp", c.get("time", ""))) for c in candles]

    @staticmethod
    def _detect_interval(timestamps: list[datetime]) -> float:
        """Detect the most common interval (in seconds) between candles."""
        if len(timestamps) < 2:
            return 0
        diffs = np.array(
            [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
        )
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0
        # Use the median to be robust against gaps
        return float(np.median(diffs))

    # --- Individual checks ---

    @staticmethod
    def _check_duplicates(timestamps: list[datetime], report: QualityReport) -> None:
        seen: set[datetime] = set()
        dup_count = 0
        for ts in timestamps:
            if ts in seen:
                dup_count += 1
            seen.add(ts)
        if dup_count:
            report.issues.append(f"Found {dup_count} duplicate timestamp(s)")

    def _check_gaps(self, timestamps: list[datetime], report: QualityReport) -> None:
        if len(timestamps) < 2:
            return
        interval = self._detect_interval(timestamps)
        if interval == 0:
            return
        # Allow 1.5x the expected interval before flagging a gap
        threshold = interval * 1.5
        gap_count = 0
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if diff > threshold:
                gap_count += 1
        if gap_count:
            report.gaps_found = gap_count
            report.issues.append(f"Found {gap_count} time gap(s) exceeding expected interval ({interval:.0f}s)")

    def _check_staleness(self, timestamps: list[datetime], report: QualityReport) -> None:
        last = timestamps[-1]
        age_minutes = (datetime.now(timezone.utc) - last).total_seconds() / 60
        # On weekends forex data can be hours old — only warn, don't block
        is_weekend = datetime.now(timezone.utc).weekday() >= 5
        stale_threshold = 60 * 48 if is_weekend else 30  # 48 hours on weekend
        if age_minutes > stale_threshold:
            report.stale_data = True
            report.issues.append(f"Data is stale: last candle is {age_minutes:.1f} minutes old")

    @staticmethod
    def _check_ohlc_consistency(candles: list[dict], report: QualityReport) -> None:
        bad_count = 0
        for c in candles:
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            if h < o or h < cl or l > o or l > cl or h < l:
                bad_count += 1
        if bad_count:
            report.issues.append(f"Found {bad_count} candle(s) with inconsistent OHLC values")

    @staticmethod
    def _check_zero_volume(candles: list[dict], report: QualityReport) -> None:
        zero_count = sum(1 for c in candles if float(c.get("volume", 1)) == 0)
        if zero_count:
            report.issues.append(f"Found {zero_count} candle(s) with zero volume")

    def _check_price_spikes(self, candles: list[dict], report: QualityReport) -> None:
        if len(candles) < 15:
            return
        closes = np.array([float(c["close"]) for c in candles])
        atr = self._compute_atr(candles)
        if atr == 0:
            return
        spike_count = 0
        for i in range(1, len(closes)):
            if abs(closes[i] - closes[i - 1]) > self.atr_spike_threshold * atr:
                spike_count += 1
        if spike_count:
            report.suspicious_spikes = spike_count
            report.issues.append(f"Found {spike_count} suspicious price spike(s) (>{self.atr_spike_threshold} ATR)")

    @staticmethod
    def _compute_atr(candles: list[dict], period: int = 14) -> float:
        """Compute the Average True Range over *period* candles."""
        if len(candles) < period + 1:
            return 0.0
        highs = np.array([float(c["high"]) for c in candles])
        lows = np.array([float(c["low"]) for c in candles])
        closes = np.array([float(c["close"]) for c in candles])

        tr_values = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        # Simple moving average of the last *period* true range values
        return float(np.mean(tr_values[-period:]))
