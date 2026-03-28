"""
Market Regime Detector — Classifies current market conditions.

Detects whether a forex pair is trending, ranging, or volatile,
and provides regime-specific trading recommendations.
"""

import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeInfo:
    """Result of regime detection for a symbol."""
    regime: MarketRegime
    confidence: float          # 0-1, how confident we are in this classification
    adx: float                 # Average Directional Index (0-100)
    atr_pct: float             # ATR as percentage of price
    bb_width: float            # Bollinger Band width as percentage of middle band
    trend_strength: float      # -1 (strong down) to +1 (strong up)
    description: str           # Human-readable summary


class RegimeDetector:
    """
    Detects market regime using multiple indicators:
    - ADX for trend strength
    - ATR (as % of price) for volatility
    - Bollinger Band width for squeeze / expansion
    - Price position relative to moving averages for trend direction
    """

    # ── Thresholds ──────────────────────────────────────────────

    ADX_TREND_THRESHOLD = 25.0        # ADX above this = trending
    ADX_STRONG_TREND = 40.0           # ADX above this = strong trend
    ATR_PCT_VOLATILE = 0.015          # ATR% above this = elevated volatility
    ATR_PCT_EXTREME = 0.025           # ATR% above this = extreme / untradeable
    BB_WIDTH_NARROW = 0.02            # BB width below this = tight range / squeeze
    BB_WIDTH_WIDE = 0.05              # BB width above this = expanded volatility

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_fast: int = 20,
        sma_slow: int = 50,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow

    # ── Public API ──────────────────────────────────────────────

    def detect_regime(self, candles: list[dict]) -> RegimeInfo:
        """
        Classify the current market regime from candle data.

        Args:
            candles: list of dicts with keys: time, open, high, low, close, volume.
                     Must have at least sma_slow + adx_period + 1 bars.

        Returns:
            RegimeInfo with the detected regime and supporting metrics.
        """
        min_bars = self.sma_slow + self.adx_period + 1
        if len(candles) < min_bars:
            logger.warning(
                f"Need at least {min_bars} candles, got {len(candles)} — "
                "defaulting to RANGING"
            )
            return RegimeInfo(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                adx=0.0,
                atr_pct=0.0,
                bb_width=0.0,
                trend_strength=0.0,
                description="Insufficient data for regime detection",
            )

        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        closes = np.array([c["close"] for c in candles], dtype=float)

        current_price = closes[-1]

        # ── Compute indicators ──
        adx, plus_di, minus_di = self._adx(highs, lows, closes, self.adx_period)
        atr = self._atr(highs, lows, closes, self.atr_period)
        atr_pct = atr / current_price if current_price > 0 else 0.0

        bb_upper, bb_middle, bb_lower = self._bollinger_bands(
            closes, self.bb_period, self.bb_std
        )
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0

        sma_fast_val = float(np.mean(closes[-self.sma_fast:]))
        sma_slow_val = float(np.mean(closes[-self.sma_slow:]))

        # Trend strength: combines MA position + DI spread, range -1 to +1
        ma_signal = 0.0
        if sma_slow_val > 0:
            ma_signal = (sma_fast_val - sma_slow_val) / sma_slow_val
        # Normalise MA signal roughly to [-1, 1]
        ma_signal = float(np.clip(ma_signal * 100, -1.0, 1.0))

        di_spread = 0.0
        if plus_di + minus_di > 0:
            di_spread = (plus_di - minus_di) / (plus_di + minus_di)  # -1 to +1

        price_vs_sma = 0.0
        if sma_slow_val > 0:
            price_vs_sma = (current_price - sma_slow_val) / sma_slow_val
        price_vs_sma = float(np.clip(price_vs_sma * 50, -1.0, 1.0))

        trend_strength = float(np.clip(
            0.4 * ma_signal + 0.3 * di_spread + 0.3 * price_vs_sma,
            -1.0,
            1.0,
        ))

        # ── Classification logic ──
        regime, confidence, description = self._classify(
            adx, atr_pct, bb_width, trend_strength, plus_di, minus_di, current_price,
            sma_fast_val, sma_slow_val,
        )

        return RegimeInfo(
            regime=regime,
            confidence=confidence,
            adx=round(adx, 2),
            atr_pct=round(atr_pct, 6),
            bb_width=round(bb_width, 4),
            trend_strength=round(trend_strength, 4),
            description=description,
        )

    def get_regime_filter(self, regime: MarketRegime) -> dict:
        """
        Return recommended trading parameters for the given regime.

        Keys:
            strategy, sl_multiplier, tp_multiplier, position_size_factor,
            trade_direction, description
        """
        filters = {
            MarketRegime.TRENDING_UP: {
                "strategy": "trend_following",
                "sl_multiplier": 1.5,
                "tp_multiplier": 3.0,
                "position_size_factor": 1.0,
                "trade_direction": "buy_only",
                "description": (
                    "Strong uptrend detected. Follow the trend with wider take-profit "
                    "targets (3x ATR) and tighter stop-losses (1.5x ATR). Only take "
                    "long entries."
                ),
            },
            MarketRegime.TRENDING_DOWN: {
                "strategy": "trend_following",
                "sl_multiplier": 1.5,
                "tp_multiplier": 3.0,
                "position_size_factor": 1.0,
                "trade_direction": "sell_only",
                "description": (
                    "Strong downtrend detected. Follow the trend with wider take-profit "
                    "targets (3x ATR) and tighter stop-losses (1.5x ATR). Only take "
                    "short entries."
                ),
            },
            MarketRegime.RANGING: {
                "strategy": "mean_reversion",
                "sl_multiplier": 1.0,
                "tp_multiplier": 1.5,
                "position_size_factor": 0.8,
                "trade_direction": "both",
                "description": (
                    "Range-bound market. Use mean reversion — buy near support, sell "
                    "near resistance. Keep stop-losses tight (1x ATR) and take profits "
                    "quickly (1.5x ATR)."
                ),
            },
            MarketRegime.VOLATILE: {
                "strategy": "reduced_exposure",
                "sl_multiplier": 2.5,
                "tp_multiplier": 3.5,
                "position_size_factor": 0.4,
                "trade_direction": "both",
                "description": (
                    "High volatility detected. Reduce position size to 40% of normal. "
                    "Use wider stops (2.5x ATR) to avoid premature stop-outs. Only "
                    "trade clear setups."
                ),
            },
        }
        return filters[regime]

    def should_trade(self, regime_info: RegimeInfo) -> bool:
        """
        Determine whether it is safe to trade under the current regime.

        Returns False when:
        - ATR% exceeds the extreme volatility threshold
        - Regime is VOLATILE with very low confidence (chaotic market)
        - ADX is near zero (no readable structure)
        """
        # Extreme volatility — stay out entirely
        if regime_info.atr_pct >= self.ATR_PCT_EXTREME:
            logger.info(
                f"ATR% {regime_info.atr_pct:.4f} exceeds extreme threshold "
                f"{self.ATR_PCT_EXTREME} — skipping"
            )
            return False

        # Volatile regime with low confidence means chaotic / unreadable
        if (
            regime_info.regime == MarketRegime.VOLATILE
            and regime_info.confidence < 0.4
        ):
            logger.info("Volatile regime with low confidence — skipping")
            return False

        # Very low ADX + wide BB = no structure at all
        if regime_info.adx < 10 and regime_info.bb_width > self.BB_WIDTH_WIDE:
            logger.info("No market structure (low ADX + wide BB) — skipping")
            return False

        return True

    # ── Indicator calculations ──────────────────────────────────

    def _adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> tuple[float, float, float]:
        """
        Calculate ADX, +DI, and -DI from scratch.

        Uses Wilder's smoothing (equivalent to EMA with alpha = 1/period).

        Returns:
            (adx, plus_di, minus_di) — all in range 0-100.
        """
        n = len(closes)
        if n < period + 1:
            return 0.0, 0.0, 0.0

        # True Range, +DM, -DM for each bar
        tr = np.zeros(n - 1)
        plus_dm = np.zeros(n - 1)
        minus_dm = np.zeros(n - 1)

        for i in range(1, n):
            idx = i - 1
            hi_diff = highs[i] - highs[i - 1]
            lo_diff = lows[i - 1] - lows[i]

            tr[idx] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

            plus_dm[idx] = hi_diff if (hi_diff > lo_diff and hi_diff > 0) else 0.0
            minus_dm[idx] = lo_diff if (lo_diff > hi_diff and lo_diff > 0) else 0.0

        # Wilder's smoothing for first `period` values (simple sum), then recursive
        smoothed_tr = np.sum(tr[:period])
        smoothed_plus_dm = np.sum(plus_dm[:period])
        smoothed_minus_dm = np.sum(minus_dm[:period])

        dx_values = []

        for i in range(period, len(tr)):
            # Wilder smoothing: prev - (prev / period) + current
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr[i]
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]

            if smoothed_tr == 0:
                plus_di = 0.0
                minus_di = 0.0
            else:
                plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
                minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

            di_sum = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
            dx_values.append((dx, plus_di, minus_di))

        if not dx_values:
            return 0.0, 0.0, 0.0

        # Smooth DX to get ADX (first ADX = mean of first `period` DX values)
        if len(dx_values) < period:
            adx = float(np.mean([d[0] for d in dx_values]))
            last_plus_di = dx_values[-1][1]
            last_minus_di = dx_values[-1][2]
        else:
            adx = float(np.mean([d[0] for d in dx_values[:period]]))
            for i in range(period, len(dx_values)):
                adx = (adx * (period - 1) + dx_values[i][0]) / period
            last_plus_di = dx_values[-1][1]
            last_minus_di = dx_values[-1][2]

        return adx, last_plus_di, last_minus_di

    def _atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Average True Range using Wilder's smoothing."""
        n = len(closes)
        if n < 2:
            return float(np.mean(highs - lows)) if len(highs) > 0 else 0.0

        tr = np.zeros(n - 1)
        for i in range(1, n):
            tr[i - 1] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        if len(tr) < period:
            return float(np.mean(tr))

        # Wilder's smoothing
        atr = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period

        return atr

    def _bollinger_bands(
        self,
        closes: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[float, float, float]:
        """
        Calculate current Bollinger Band values.

        Returns:
            (upper, middle, lower) for the most recent bar.
        """
        if len(closes) < period:
            mid = float(np.mean(closes))
            std = float(np.std(closes))
            return mid + num_std * std, mid, mid - num_std * std

        window = closes[-period:]
        mid = float(np.mean(window))
        std = float(np.std(window, ddof=0))

        upper = mid + num_std * std
        lower = mid - num_std * std

        return upper, mid, lower

    # ── Classification ──────────────────────────────────────────

    def _classify(
        self,
        adx: float,
        atr_pct: float,
        bb_width: float,
        trend_strength: float,
        plus_di: float,
        minus_di: float,
        current_price: float,
        sma_fast: float,
        sma_slow: float,
    ) -> tuple[MarketRegime, float, str]:
        """
        Classify regime using a weighted scoring approach.

        Returns:
            (regime, confidence, description)
        """
        # Scores for each regime candidate
        scores = {
            MarketRegime.TRENDING_UP: 0.0,
            MarketRegime.TRENDING_DOWN: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE: 0.0,
        }

        # ── Volatility check (takes priority when extreme) ──

        if atr_pct >= self.ATR_PCT_VOLATILE:
            vol_score = min((atr_pct - self.ATR_PCT_VOLATILE) / self.ATR_PCT_VOLATILE, 1.0)
            scores[MarketRegime.VOLATILE] += 0.35 * vol_score

        if bb_width >= self.BB_WIDTH_WIDE:
            bb_score = min((bb_width - self.BB_WIDTH_WIDE) / self.BB_WIDTH_WIDE, 1.0)
            scores[MarketRegime.VOLATILE] += 0.25 * bb_score

        # ── Trend detection ──

        if adx >= self.ADX_TREND_THRESHOLD:
            adx_score = min((adx - self.ADX_TREND_THRESHOLD) / 25.0, 1.0)

            if plus_di > minus_di and trend_strength > 0:
                scores[MarketRegime.TRENDING_UP] += 0.35 * adx_score
            elif minus_di > plus_di and trend_strength < 0:
                scores[MarketRegime.TRENDING_DOWN] += 0.35 * adx_score

        # MA alignment
        if sma_fast > sma_slow and current_price > sma_fast:
            scores[MarketRegime.TRENDING_UP] += 0.2
        elif sma_fast < sma_slow and current_price < sma_fast:
            scores[MarketRegime.TRENDING_DOWN] += 0.2

        # Trend strength magnitude
        ts_abs = abs(trend_strength)
        if ts_abs > 0.3:
            direction = MarketRegime.TRENDING_UP if trend_strength > 0 else MarketRegime.TRENDING_DOWN
            scores[direction] += 0.15 * ts_abs

        # ── Range detection ──

        if adx < self.ADX_TREND_THRESHOLD:
            range_score = 1.0 - (adx / self.ADX_TREND_THRESHOLD)
            scores[MarketRegime.RANGING] += 0.3 * range_score

        if bb_width < self.BB_WIDTH_NARROW:
            scores[MarketRegime.RANGING] += 0.2

        if atr_pct < self.ATR_PCT_VOLATILE:
            calm_score = 1.0 - (atr_pct / self.ATR_PCT_VOLATILE)
            scores[MarketRegime.RANGING] += 0.15 * calm_score

        # ── Pick winner ──

        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_regime]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0
        confidence = round(min(confidence, 1.0), 4)

        # ── Build description ──

        parts = []
        if best_regime == MarketRegime.TRENDING_UP:
            parts.append(f"Uptrend (ADX {adx:.1f}, +DI {plus_di:.1f} > -DI {minus_di:.1f})")
            if adx >= self.ADX_STRONG_TREND:
                parts.append("Strong trend")
            parts.append(f"Price above SMA{self.sma_fast} & SMA{self.sma_slow}")

        elif best_regime == MarketRegime.TRENDING_DOWN:
            parts.append(f"Downtrend (ADX {adx:.1f}, -DI {minus_di:.1f} > +DI {plus_di:.1f})")
            if adx >= self.ADX_STRONG_TREND:
                parts.append("Strong trend")
            parts.append(f"Price below SMA{self.sma_fast} & SMA{self.sma_slow}")

        elif best_regime == MarketRegime.RANGING:
            parts.append(f"Range-bound (ADX {adx:.1f})")
            if bb_width < self.BB_WIDTH_NARROW:
                parts.append("Bollinger squeeze — potential breakout ahead")
            parts.append(f"BB width {bb_width:.4f}")

        elif best_regime == MarketRegime.VOLATILE:
            parts.append(f"High volatility (ATR% {atr_pct:.4f}, BB width {bb_width:.4f})")
            if atr_pct >= self.ATR_PCT_EXTREME:
                parts.append("EXTREME — avoid trading")
            parts.append("Reduce position sizes")

        description = ". ".join(parts) + "."

        return best_regime, confidence, description
