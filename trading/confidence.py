# trading/confidence.py

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


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    reasons: list[str]


def compute_confidence(x: ConfidenceInputs) -> ConfidenceResult:
    """
    Deterministic confidence score in [0, 1].

    Philosophy:
    - Hard gates first (fail fast).
    - Score only high-quality setups (structure + momentum + strength + execution).
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
    # Soft scoring
    # -----------------------
    score = 0.0

    # Pullback quality (0..0.35)
    pullback = 0.0
    if x.direction == "buy" and 30 <= x.rsi_h1 <= 45:
        pullback += 0.18
        reasons.append("pullback: RSI buy zone")
    if x.direction == "sell" and 55 <= x.rsi_h1 <= 70:
        pullback += 0.18
        reasons.append("pullback: RSI sell zone")

    # Price location vs MAs (simple structure check)
    if x.direction == "buy" and x.sma20_h1 >= x.sma50_h1 and x.close_h1 >= x.sma20_h1:
        pullback += 0.17
        reasons.append("structure: close>=SMA20 and SMA20>=SMA50")
    if x.direction == "sell" and x.sma20_h1 <= x.sma50_h1 and x.close_h1 <= x.sma20_h1:
        pullback += 0.17
        reasons.append("structure: close<=SMA20 and SMA20<=SMA50")

    score += min(0.35, pullback)

    # Momentum confirmation (0..0.25)
    momentum = 0.0
    hist_delta = x.macd_hist_h1 - x.macd_hist_prev_h1
    if x.direction == "buy" and (x.macd_hist_h1 > 0 or hist_delta > 0):
        momentum += 0.25
        reasons.append("momentum: MACD improving")
    if x.direction == "sell" and (x.macd_hist_h1 < 0 or hist_delta < 0):
        momentum += 0.25
        reasons.append("momentum: MACD worsening")
    score += min(0.25, momentum)

    # Trend strength (0..0.25)
    strength = 0.0
    if x.adx_h4 >= 18:
        strength += 0.15
        reasons.append("strength: ADX>=18")
    if x.adx_h4 >= 25:
        strength += 0.10
        reasons.append("strength: ADX>=25")
    score += min(0.25, strength)

    # Execution quality (0..0.15)
    execq = 0.0
    if x.atr_pips_h1 > 0:
        spread_ratio = x.spread_pips / x.atr_pips_h1
        if spread_ratio <= 0.08:
            execq = 0.15
            reasons.append("execution: spread/ATR ok")
        elif spread_ratio <= 0.12:
            execq = 0.08
            reasons.append("execution: spread/ATR borderline")
        else:
            reasons.append("execution: spread/ATR bad (penalized)")
    score += execq

    confidence = max(0.0, min(1.0, score))
    return ConfidenceResult(confidence, reasons)
