# trading/options_confidence.py
"""
Options-specific confidence scoring — separate from forex.

Two modes:
  MOMENTUM: fast expansion, 5-10 DTE, quick entries
  SWING:    trend continuation, 10-21 DTE, HTF alignment

Scoring dimensions (max 1.0):
  Trend quality:      0.25  (HTF alignment + ADX strength)
  Entry timing:       0.25  (momentum trigger, pullback quality)
  Expansion potential: 0.15  (volume, ATR expansion, squeeze)
  Contract quality:    0.15  (IV, spread, OI, delta)
  Event/News:          0.10  (aligned news, no blockers)
  Execution:           0.10  (spread/ATR, time of day)

Minimum thresholds:
  Momentum mode: 0.55
  Swing mode:    0.50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


SetupMode = Literal["momentum", "swing"]
Direction = Literal["buy", "sell"]  # buy=calls, sell=puts


@dataclass
class OptionsConfidenceInputs:
    """All inputs needed to score an options setup."""
    direction: Direction          # "buy" for calls, "sell" for puts
    mode: SetupMode               # "momentum" or "swing"

    # Trend inputs
    d1_trend: str                  # "up", "down", "flat"
    h4_trend: str                  # "up", "down", "flat"
    h1_trend: str                  # "up", "down", "flat"
    adx_h4: float                  # H4 ADX value
    adx_d1: float                  # D1 ADX value
    trend_bars_h4: int             # How many bars H4 trend has persisted

    # Entry timing inputs
    pullback_quality: float        # 0-1, how clean the pullback is
    momentum_trigger: bool         # Did M15/H1 give a GO signal?
    macd_hist_flip: bool           # MACD histogram flipped direction?
    strong_candle: bool            # Strong close candle?
    ema_reclaim: bool              # Price reclaimed key EMA?
    volume_confirm: bool           # Volume above average on trigger?

    # Expansion inputs
    atr_expansion: float           # Current ATR / 20-period ATR ratio
    volume_spike: float            # Current vol / 20-period avg vol ratio
    adx_rising: bool               # ADX increasing over last 3 bars?
    range_breakout: bool           # Breaking out of consolidation?
    squeeze_release: bool          # Bollinger squeeze releasing?

    # Contract quality inputs
    iv_rank: float                 # IV percentile (0-100), lower = cheaper
    spread_pct: float              # Bid-ask spread as % of mid
    open_interest: int             # Contract OI
    delta: float                   # Contract delta (absolute)
    dte: int                       # Days to expiration

    # News/event inputs
    news_aligned: bool             # News sentiment matches direction?
    news_against: bool             # News sentiment opposes direction?
    high_impact_today: bool        # Major event today (FOMC, CPI, etc)?
    earnings_soon: bool            # Earnings within 3 days? (single names)

    # Execution inputs
    spread_vs_atr: float           # Option spread / underlying ATR
    minutes_since_open: int        # Minutes since market open
    minutes_to_close: int          # Minutes until market close


@dataclass
class OptionsConfidenceResult:
    confidence: float
    mode: SetupMode
    reasons: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # Per-dimension scores


def score_options_setup(x: OptionsConfidenceInputs) -> OptionsConfidenceResult:
    """
    Score an options setup across 6 dimensions.

    Returns OptionsConfidenceResult with total confidence and per-dimension breakdown.
    """
    reasons: list[str] = []
    scores: dict[str, float] = {}

    # ═══════════════════════════════════
    # HARD GATES (instant reject)
    # ═══════════════════════════════════

    # Gate 1: D1 trend must be directional
    if x.d1_trend == "flat":
        return OptionsConfidenceResult(0.0, x.mode, ["blocked: D1 flat — no directional trade"])

    # Gate 2: Direction must match D1 trend
    expected = "buy" if x.d1_trend == "up" else "sell"
    if x.direction != expected:
        return OptionsConfidenceResult(0.0, x.mode, ["blocked: countertrend"])

    # Gate 3: Earnings block for single names (not SPY/QQQ)
    if x.earnings_soon:
        return OptionsConfidenceResult(0.0, x.mode, ["blocked: earnings within 3 days"])

    # Gate 4: High-impact event day — block momentum mode (swing OK with penalty)
    if x.high_impact_today and x.mode == "momentum":
        return OptionsConfidenceResult(0.0, x.mode, ["blocked: high-impact event day (momentum)"])

    # Gate 5: First/last 30 min — no momentum entries
    if x.mode == "momentum" and (x.minutes_since_open < 30 or x.minutes_to_close < 30):
        return OptionsConfidenceResult(0.0, x.mode, ["blocked: too close to open/close for momentum"])

    # Gate 6: IV too high for buying premium
    if x.iv_rank > 80:
        return OptionsConfidenceResult(0.0, x.mode, [f"blocked: IV rank {x.iv_rank:.0f} too high"])

    # ═══════════════════════════════════
    # DIMENSION 1: Trend Quality (0..0.25)
    # ═══════════════════════════════════
    trend = 0.0

    # D1 + H4 agreement
    if x.d1_trend == x.h4_trend and x.d1_trend != "flat":
        trend += 0.10
        reasons.append("trend: D1+H4 agree")

    # H1 alignment (bonus, not required for swing)
    if x.h1_trend == x.d1_trend:
        trend += 0.05
        reasons.append("trend: H1 aligned")

    # ADX strength
    if x.mode == "momentum":
        if x.adx_h4 >= 25:
            trend += 0.05
            reasons.append(f"trend: H4 ADX {x.adx_h4:.0f}")
        if x.adx_h4 >= 35:
            trend += 0.05
            reasons.append("trend: strong trend (ADX>=35)")
    else:  # swing
        if x.adx_d1 >= 20:
            trend += 0.05
            reasons.append(f"trend: D1 ADX {x.adx_d1:.0f}")
        if x.trend_bars_h4 >= 5:
            trend += 0.05
            reasons.append(f"trend: H4 trend {x.trend_bars_h4} bars")

    scores["trend"] = min(0.25, trend)

    # ═══════════════════════════════════
    # DIMENSION 2: Entry Timing (0..0.25)
    # ═══════════════════════════════════
    entry = 0.0

    # Momentum trigger is critical for momentum mode
    if x.momentum_trigger:
        entry += 0.10
        reasons.append("entry: momentum trigger fired")
    elif x.mode == "momentum":
        # No trigger in momentum mode = very weak
        scores["entry"] = 0.0
        scores["trend"] = scores.get("trend", 0)
        total = sum(scores.values())
        return OptionsConfidenceResult(total, x.mode, reasons + ["weak: no momentum trigger"], scores)

    # MACD histogram flip
    if x.macd_hist_flip:
        entry += 0.05
        reasons.append("entry: MACD hist flip")

    # Strong candle
    if x.strong_candle:
        entry += 0.04
        reasons.append("entry: strong candle close")

    # EMA reclaim
    if x.ema_reclaim:
        entry += 0.03
        reasons.append("entry: EMA reclaim")

    # Volume confirmation
    if x.volume_confirm:
        entry += 0.03
        reasons.append("entry: volume confirms")

    scores["entry"] = min(0.25, entry)

    # ═══════════════════════════════════
    # DIMENSION 3: Expansion Potential (0..0.15)
    # ═══════════════════════════════════
    expansion = 0.0

    if x.atr_expansion > 1.3:
        expansion += 0.05
        reasons.append(f"expand: ATR expanding {x.atr_expansion:.1f}x")

    if x.volume_spike > 1.5:
        expansion += 0.04
        reasons.append(f"expand: volume spike {x.volume_spike:.1f}x")

    if x.adx_rising:
        expansion += 0.03
        reasons.append("expand: ADX rising")

    if x.range_breakout:
        expansion += 0.05
        reasons.append("expand: range breakout")

    if x.squeeze_release:
        expansion += 0.05
        reasons.append("expand: squeeze release")

    scores["expansion"] = min(0.15, expansion)

    # ═══════════════════════════════════
    # DIMENSION 4: Contract Quality (0..0.15)
    # ═══════════════════════════════════
    contract = 0.0

    # IV rank (lower = cheaper premium = better for buying)
    if x.iv_rank <= 30:
        contract += 0.05
        reasons.append(f"contract: low IV rank ({x.iv_rank:.0f})")
    elif x.iv_rank <= 50:
        contract += 0.03
        reasons.append(f"contract: moderate IV ({x.iv_rank:.0f})")

    # Delta filter
    if x.mode == "momentum":
        if 0.40 <= x.delta <= 0.60:
            contract += 0.04
            reasons.append(f"contract: good delta ({x.delta:.2f})")
    else:
        if 0.35 <= x.delta <= 0.55:
            contract += 0.04
            reasons.append(f"contract: good delta ({x.delta:.2f})")

    # Spread quality
    if x.spread_pct <= 0.05:
        contract += 0.03
        reasons.append("contract: tight spread")
    elif x.spread_pct <= 0.10:
        contract += 0.01
        reasons.append("contract: acceptable spread")

    # OI
    if x.open_interest >= 500:
        contract += 0.03
        reasons.append(f"contract: deep OI ({x.open_interest})")
    elif x.open_interest >= 200:
        contract += 0.01

    scores["contract"] = min(0.15, contract)

    # ═══════════════════════════════════
    # DIMENSION 5: Event/News (0..0.10)
    # ═══════════════════════════════════
    event = 0.0

    if x.news_aligned:
        event += 0.05
        reasons.append("event: news aligned")

    if x.news_against:
        event -= 0.05
        reasons.append("event: news opposes")

    if x.high_impact_today:
        event -= 0.03  # Penalty for swing mode (momentum already blocked)
        reasons.append("event: high-impact day (penalized)")

    scores["event"] = max(0.0, min(0.10, event))

    # ═══════════════════════════════════
    # DIMENSION 6: Execution Quality (0..0.10)
    # ═══════════════════════════════════
    execution = 0.0

    # Best trading hours: 10:00-11:30, 13:30-15:30 ET
    if 30 <= x.minutes_since_open <= 120:  # 10:00-11:30
        execution += 0.04
        reasons.append("exec: prime morning hours")
    elif 240 <= x.minutes_since_open <= 360:  # 13:30-15:30
        execution += 0.04
        reasons.append("exec: prime afternoon hours")
    elif x.minutes_since_open > 15:
        execution += 0.02

    # Option spread vs underlying move potential
    if x.spread_vs_atr <= 0.05:
        execution += 0.04
        reasons.append("exec: option spread tiny vs ATR")
    elif x.spread_vs_atr <= 0.10:
        execution += 0.02

    # DTE appropriateness for mode
    if x.mode == "momentum" and 5 <= x.dte <= 10:
        execution += 0.02
        reasons.append(f"exec: good momentum DTE ({x.dte})")
    elif x.mode == "swing" and 10 <= x.dte <= 21:
        execution += 0.02
        reasons.append(f"exec: good swing DTE ({x.dte})")

    scores["execution"] = min(0.10, execution)

    # ═══════════════════════════════════
    # TOTAL
    # ═══════════════════════════════════
    total = sum(scores.values())
    confidence = max(0.0, min(1.0, total))

    return OptionsConfidenceResult(
        confidence=confidence,
        mode=x.mode,
        reasons=reasons,
        scores=scores,
    )


# Thresholds
MOMENTUM_THRESHOLD = 0.55
SWING_THRESHOLD = 0.50
