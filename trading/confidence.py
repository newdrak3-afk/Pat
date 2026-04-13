# trading/confidence.py
"""
Deterministic confidence scoring module.

Hard gates first (fail fast), then structured soft scoring across
five independent dimensions. Each dimension has a clear max contribution.

Scoring buckets:
  Pullback quality:    0.30
  Location quality:    0.20  (NEW — entry at key levels)
  Momentum:            0.20
  Trend strength:      0.20
  Execution quality:   0.10
  ─────────────────────────
  Max total:           1.00

Minimum thresholds (applied by caller):
  Demo/learning:  0.45
  Live forex:     0.55
  Options:        0.50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Direction = Literal["buy", "sell"]
TrendState = Literal["up", "down", "flat"]


@dataclass(frozen=True)
class ConfidenceInputs:
    direction: Direction

    d1_trend: TrendState
    h4_trend: TrendState

    rsi_h1: float
    macd_hist_h1: float
    macd_hist_prev_h1: float

    sma20_h1: float
    sma50_h1: float
    close_h1: float

    adx_h4: float

    atr_pips_h1: float
    spread_pips: float

    regime: Literal["trending", "ranging", "volatile", "choppy"]

    # Location inputs (NEW)
    h4_sma20: float = 0.0       # H4 SMA20 level
    d1_sma20: float = 0.0       # D1 SMA20 level
    h4_recent_low: float = 0.0  # Recent H4 swing low (for buys)
    h4_recent_high: float = 0.0 # Recent H4 swing high (for sells)


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    reasons: list[str]


def compute_confidence(x: ConfidenceInputs) -> ConfidenceResult:
    """
    Deterministic confidence score in [0, 1].

    Philosophy:
    - Hard gates first (fail fast).
    - Score only high-quality setups across 5 dimensions.
    - Be selective — average trades should score below threshold.
    """
    reasons: list[str] = []

    # -----------------------
    # Hard gates (0 confidence)
    # -----------------------
    if x.regime in ("volatile", "choppy"):
        return ConfidenceResult(0.0, [f"blocked: regime={x.regime}"])

    if not (x.d1_trend in ("up", "down") and x.h4_trend in ("up", "down")):
        return ConfidenceResult(0.0, ["blocked: HTF trend not directional"])

    if x.d1_trend != x.h4_trend:
        return ConfidenceResult(0.0, ["blocked: D1/H4 disagree"])

    aligned_direction: Direction = "buy" if x.d1_trend == "up" else "sell"
    if x.direction != aligned_direction:
        return ConfidenceResult(0.0, ["blocked: countertrend"])

    # -----------------------
    # Soft scoring (5 dimensions)
    # -----------------------
    score = 0.0

    # 1. Pullback quality (0..0.30)
    pullback = 0.0
    if x.direction == "buy" and 30 <= x.rsi_h1 <= 45:
        pullback += 0.15
        reasons.append("pullback: RSI buy zone")
    if x.direction == "sell" and 55 <= x.rsi_h1 <= 70:
        pullback += 0.15
        reasons.append("pullback: RSI sell zone")

    # Price structure vs MAs
    if x.direction == "buy" and x.sma20_h1 >= x.sma50_h1 and x.close_h1 >= x.sma20_h1:
        pullback += 0.15
        reasons.append("structure: bullish MA stack")
    if x.direction == "sell" and x.sma20_h1 <= x.sma50_h1 and x.close_h1 <= x.sma20_h1:
        pullback += 0.15
        reasons.append("structure: bearish MA stack")

    score += min(0.30, pullback)

    # 2. Location quality (0..0.20) — NEW
    # Is the entry at a key level? SMA confluence, H4 swing, D1 level
    location = 0.0

    # SMA20/SMA50 confluence (H1 close near both MAs)
    if x.sma20_h1 > 0 and x.sma50_h1 > 0:
        ma_gap_pct = abs(x.sma20_h1 - x.sma50_h1) / x.sma20_h1 * 100
        if ma_gap_pct < 0.3:  # MAs converging = strong level
            location += 0.08
            reasons.append("location: SMA20/50 confluence")

    # Close near H4 SMA20 (pullback to H4 value zone)
    if x.h4_sma20 > 0:
        dist_to_h4_sma = abs(x.close_h1 - x.h4_sma20) / x.h4_sma20 * 100
        if dist_to_h4_sma < 0.2:
            location += 0.07
            reasons.append("location: at H4 SMA20")

    # Near D1 SMA20 (daily level support/resistance)
    if x.d1_sma20 > 0:
        dist_to_d1_sma = abs(x.close_h1 - x.d1_sma20) / x.d1_sma20 * 100
        if dist_to_d1_sma < 0.3:
            location += 0.07
            reasons.append("location: at D1 SMA20")

    # Near H4 swing level (retest of breakout)
    if x.direction == "buy" and x.h4_recent_low > 0:
        dist_to_swing = (x.close_h1 - x.h4_recent_low) / x.h4_recent_low * 100
        if 0 < dist_to_swing < 0.5:
            location += 0.07
            reasons.append("location: H4 swing low retest")
    if x.direction == "sell" and x.h4_recent_high > 0:
        dist_to_swing = (x.h4_recent_high - x.close_h1) / x.h4_recent_high * 100
        if 0 < dist_to_swing < 0.5:
            location += 0.07
            reasons.append("location: H4 swing high retest")

    score += min(0.20, location)

    # 3. Momentum confirmation (0..0.20)
    # Require BOTH conditions for full score — prevents every trending market
    # from trivially scoring 0.20 just because MACD is positive.
    # Full (0.20): histogram is positive AND improving (strong momentum alignment)
    # Partial (0.08): only one condition met (weak confirmation)
    momentum = 0.0
    hist_delta = x.macd_hist_h1 - x.macd_hist_prev_h1
    if x.direction == "buy":
        if x.macd_hist_h1 > 0 and hist_delta > 0:
            momentum += 0.20
            reasons.append("momentum: MACD positive+improving")
        elif x.macd_hist_h1 > 0 or hist_delta > 0:
            momentum += 0.08
            reasons.append("momentum: MACD weakly confirmed")
    if x.direction == "sell":
        if x.macd_hist_h1 < 0 and hist_delta < 0:
            momentum += 0.20
            reasons.append("momentum: MACD negative+worsening")
        elif x.macd_hist_h1 < 0 or hist_delta < 0:
            momentum += 0.08
            reasons.append("momentum: MACD weakly confirmed")
    score += min(0.20, momentum)

    # 4. Trend strength (0..0.20)
    strength = 0.0
    if x.adx_h4 >= 18:
        strength += 0.10
        reasons.append("strength: ADX>=18")
    if x.adx_h4 >= 25:
        strength += 0.05
        reasons.append("strength: ADX>=25")
    if x.adx_h4 >= 35:
        strength += 0.05
        reasons.append("strength: ADX>=35 (strong trend)")
    score += min(0.20, strength)

    # 5. Execution quality (0..0.10)
    execq = 0.0
    if x.atr_pips_h1 > 0:
        spread_ratio = x.spread_pips / x.atr_pips_h1
        if spread_ratio <= 0.05:
            execq = 0.10
            reasons.append("execution: excellent spread/ATR")
        elif spread_ratio <= 0.08:
            execq = 0.07
            reasons.append("execution: good spread/ATR")
        elif spread_ratio <= 0.12:
            execq = 0.03
            reasons.append("execution: borderline spread/ATR")
        else:
            reasons.append("execution: poor spread/ATR")
    score += execq

    confidence = max(0.0, min(1.0, score))
    return ConfidenceResult(confidence, reasons)
