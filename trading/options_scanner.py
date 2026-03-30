# trading/options_scanner.py
"""
Options Scanner v2 — Momentum + Swing mode detection.

Scans SPY/QQQ (priority) and AAPL/MSFT/NVDA (when ideal) for
high-conviction directional options setups.

Two modes:
  MOMENTUM: Fast expansion, M15 trigger, 5-10 DTE
  SWING:    Trend continuation pullback, 10-21 DTE

Entry triggers:
  - Break of pullback high/low
  - MACD histogram flip
  - EMA reclaim
  - Strong candle close
  - Volume confirmation
  - Squeeze release / range breakout
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from trading.brokers.alpaca import AlpacaBroker, OPTIONS_SYMBOLS
from trading.options_contract_selector import ContractSelector
from trading.options_confidence import (
    score_options_setup, OptionsConfidenceInputs,
    MOMENTUM_THRESHOLD, SWING_THRESHOLD,
)
from trading.news_sentiment import NewsReader

logger = logging.getLogger(__name__)

# Priority tiers — SPY/QQQ first, single names only when setup is strong
TIER1_SYMBOLS = ["SPY", "QQQ"]
TIER2_SYMBOLS = ["AAPL", "MSFT", "NVDA"]
TIER2_CONFIDENCE_BONUS = 0.10  # Tier 2 needs this much MORE confidence to trade

# Market hours (UTC)
MARKET_OPEN_UTC = 14 * 60 + 30   # 9:30 ET = 14:30 UTC
MARKET_CLOSE_UTC = 21 * 60       # 16:00 ET = 21:00 UTC


def is_options_market_open(now: datetime = None) -> bool:
    """Check if US options market is in tradeable hours."""
    if now is None:
        now = datetime.now(timezone.utc)

    if now.weekday() >= 5:  # Weekend
        return False

    minutes = now.hour * 60 + now.minute
    # Avoid first 15 min and last 15 min
    return (MARKET_OPEN_UTC + 15) <= minutes <= (MARKET_CLOSE_UTC - 15)


class OptionsScanner:
    """
    Scans for high-quality options setups in two modes.

    MOMENTUM: Looks for expansion/breakout/squeeze setups.
    SWING: Looks for pullback continuation with strong HTF trend.
    """

    def __init__(self, broker: AlpacaBroker):
        self.broker = broker
        self.contract_selector = ContractSelector()
        self.news = NewsReader()

    def scan_all(self) -> list[dict]:
        """Scan all symbols, return sorted signals."""
        signals = []
        scan_log = []

        logger.info(f"OPTIONS SCAN: scanning {len(TIER1_SYMBOLS)} T1 + {len(TIER2_SYMBOLS)} T2 symbols")

        # Tier 1 first (SPY, QQQ)
        for symbol in TIER1_SYMBOLS:
            for mode in ["momentum", "swing"]:
                try:
                    signal = self._analyze_symbol(symbol, mode, tier=1)
                    if signal:
                        signals.append(signal)
                        scan_log.append(f"  {symbol} {mode}: SIGNAL conf={signal['confidence']:.2f}")
                    else:
                        scan_log.append(f"  {symbol} {mode}: no setup")
                except Exception as e:
                    scan_log.append(f"  {symbol} {mode}: ERROR {e}")
                    logger.warning(f"Options scan error {symbol} {mode}: {e}", exc_info=True)

        # Tier 2 only if setup quality is high
        for symbol in TIER2_SYMBOLS:
            for mode in ["momentum", "swing"]:
                try:
                    signal = self._analyze_symbol(symbol, mode, tier=2)
                    if signal:
                        signals.append(signal)
                        scan_log.append(f"  {symbol} {mode}: SIGNAL conf={signal['confidence']:.2f}")
                    else:
                        scan_log.append(f"  {symbol} {mode}: no setup")
                except Exception as e:
                    scan_log.append(f"  {symbol} {mode}: ERROR {e}")
                    logger.warning(f"Options scan error {symbol} {mode}: {e}", exc_info=True)

        logger.info(f"OPTIONS SCAN RESULTS:\n" + "\n".join(scan_log))
        logger.info(f"OPTIONS: {len(signals)} signals found from {len(TIER1_SYMBOLS) + len(TIER2_SYMBOLS)} symbols")

        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return signals

    def _analyze_symbol(self, symbol: str, mode: str, tier: int) -> Optional[dict]:
        """Analyze one symbol for one mode. Returns signal dict or None."""

        # ─── GET CANDLE DATA ───
        candles_d1_raw = self.broker.get_candles(symbol, "1Day", 60)
        candles_h4_raw = self.broker.get_candles(symbol, "4Hour", 60)
        candles_h1_raw = self.broker.get_candles(symbol, "1Hour", 100)
        try:
            candles_m15_raw = self.broker.get_candles(symbol, "15Min", 50)
        except Exception:
            candles_m15_raw = []

        # Bar-close discipline
        candles_d1 = candles_d1_raw[:-1] if len(candles_d1_raw) > 1 else candles_d1_raw
        candles_h4 = candles_h4_raw[:-1] if len(candles_h4_raw) > 1 else candles_h4_raw
        candles_h1 = candles_h1_raw[:-1] if len(candles_h1_raw) > 1 else candles_h1_raw
        candles_m15 = candles_m15_raw[:-1] if len(candles_m15_raw) > 1 else candles_m15_raw

        if len(candles_h1) < 50 or len(candles_h4) < 20 or len(candles_d1) < 20:
            return None

        # Extract arrays
        closes_d1 = np.array([c["close"] for c in candles_d1])
        highs_d1 = np.array([c["high"] for c in candles_d1])
        lows_d1 = np.array([c["low"] for c in candles_d1])

        closes_h4 = np.array([c["close"] for c in candles_h4])
        highs_h4 = np.array([c["high"] for c in candles_h4])
        lows_h4 = np.array([c["low"] for c in candles_h4])

        closes_h1 = np.array([c["close"] for c in candles_h1])
        highs_h1 = np.array([c["high"] for c in candles_h1])
        lows_h1 = np.array([c["low"] for c in candles_h1])
        volumes_h1 = np.array([c.get("volume", 0) for c in candles_h1])

        has_m15 = len(candles_m15) >= 10
        closes_m15 = np.array([c["close"] for c in candles_m15]) if has_m15 else None
        highs_m15 = np.array([c["high"] for c in candles_m15]) if has_m15 else None
        lows_m15 = np.array([c["low"] for c in candles_m15]) if has_m15 else None

        # ─── TREND DETECTION ───
        d1_trend, d1_adx = self._detect_trend(closes_d1, highs_d1, lows_d1)
        h4_trend, h4_adx = self._detect_trend(closes_h4, highs_h4, lows_h4)
        h1_trend, _ = self._detect_trend(closes_h1, highs_h1, lows_h1)

        trend_bars_h4 = self._count_trend_bars(closes_h4, h4_trend)

        # Direction from D1 trend
        if d1_trend == "up":
            direction = "buy"
        elif d1_trend == "down":
            direction = "sell"
        else:
            return None

        # ─── INDICATORS ───
        rsi_h1 = self._rsi(closes_h1)

        # MACD histogram
        macd_values = np.array([
            self._ema(closes_h1[:i+1], 12) - self._ema(closes_h1[:i+1], 26)
            for i in range(max(25, len(closes_h1)-5), len(closes_h1))
        ])
        signal_line = self._ema(macd_values, 9)
        macd_hist = macd_values[-1] - signal_line if len(macd_values) > 0 else 0.0
        macd_hist_prev = (macd_values[-2] - self._ema(macd_values[:-1], 9)) if len(macd_values) > 1 else 0.0

        macd_hist_flip = (
            (direction == "buy" and macd_hist > 0 and macd_hist_prev <= 0) or
            (direction == "sell" and macd_hist < 0 and macd_hist_prev >= 0)
        )

        # EMA 9/21
        ema9_h1 = self._ema(closes_h1, 9)

        ema_reclaim = (
            (direction == "buy" and closes_h1[-1] > ema9_h1 and closes_h1[-2] <= ema9_h1) or
            (direction == "sell" and closes_h1[-1] < ema9_h1 and closes_h1[-2] >= ema9_h1)
        )

        # Strong candle
        body = abs(closes_h1[-1] - closes_h1[-2]) if len(closes_h1) >= 2 else 0
        candle_range = highs_h1[-1] - lows_h1[-1] if len(highs_h1) >= 1 else 1
        strong_candle = body > candle_range * 0.6 if candle_range > 0 else False

        # Volume
        avg_vol = float(np.mean(volumes_h1[-20:])) if len(volumes_h1) >= 20 else 1
        recent_vol = float(np.mean(volumes_h1[-3:])) if len(volumes_h1) >= 3 else 0
        volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0
        volume_confirm = volume_spike > 1.3

        # ATR expansion
        atr_20 = self._atr(highs_h1, lows_h1, closes_h1, 20)
        atr_5 = self._atr(highs_h1[-6:], lows_h1[-6:], closes_h1[-6:], 5) if len(closes_h1) >= 6 else atr_20
        atr_expansion = atr_5 / atr_20 if atr_20 > 0 else 1.0

        # ADX rising (last 3 H4 bars)
        adx_vals = [self._adx(highs_h4[:i+1], lows_h4[:i+1], closes_h4[:i+1]) for i in range(len(closes_h4)-3, len(closes_h4))]
        adx_rising = len(adx_vals) >= 2 and adx_vals[-1] > adx_vals[0]

        # Range breakout
        h1_20_high = float(np.max(highs_h1[-20:]))
        h1_20_low = float(np.min(lows_h1[-20:]))
        range_breakout = (
            (direction == "buy" and closes_h1[-1] > h1_20_high) or
            (direction == "sell" and closes_h1[-1] < h1_20_low)
        )

        # Squeeze release
        squeeze_release = False
        if len(closes_h1) >= 25:
            bb_now = float(np.std(closes_h1[-20:])) * 2
            bb_prev = float(np.std(closes_h1[-25:-5])) * 2
            if bb_prev > 0 and bb_now > bb_prev * 1.5:
                squeeze_release = True

        # Pullback quality
        pullback_quality = self._assess_pullback(closes_h1, highs_h1, lows_h1, direction, rsi_h1)

        # M15 momentum trigger
        momentum_trigger = False
        if closes_m15 is not None and has_m15:
            m15_ema9 = self._ema(closes_m15, 9)
            if direction == "buy":
                m15_5_high = float(np.max(highs_m15[-5:])) if len(highs_m15) >= 5 else 0
                momentum_trigger = closes_m15[-1] > m15_ema9 and closes_m15[-1] >= m15_5_high
            else:
                m15_5_low = float(np.min(lows_m15[-5:])) if len(lows_m15) >= 5 else float("inf")
                momentum_trigger = closes_m15[-1] < m15_ema9 and closes_m15[-1] <= m15_5_low

        if mode == "swing" and not momentum_trigger:
            momentum_trigger = (ema_reclaim or strong_candle or macd_hist_flip) and pullback_quality > 0.3

        # ─── NEWS / EVENTS ───
        now = datetime.now(timezone.utc)
        minutes_since_open = (now.hour * 60 + now.minute) - MARKET_OPEN_UTC
        minutes_to_close = MARKET_CLOSE_UTC - (now.hour * 60 + now.minute)

        try:
            has_impact, _ = self.news.has_high_impact_event()
            market_sent = self.news.get_market_sentiment()
            news_aligned = (
                (market_sent.sentiment == "bullish" and direction == "buy") or
                (market_sent.sentiment == "bearish" and direction == "sell")
            )
            news_against = (
                (market_sent.sentiment == "bullish" and direction == "sell") or
                (market_sent.sentiment == "bearish" and direction == "buy")
            )
        except Exception:
            has_impact, news_aligned, news_against = False, False, False

        # ─── GET QUOTE + CONTRACT ───
        quote = self.broker.get_quote(symbol)
        if not quote:
            return None

        current_price = quote.mid

        # Adjust contract selector for mode
        if mode == "momentum":
            self.contract_selector.min_dte = 5
            self.contract_selector.max_dte = 10
        else:
            self.contract_selector.min_dte = 10
            self.contract_selector.max_dte = 21

        try:
            chain = self.broker.get_options_chain(symbol)
            if not chain:
                return None
        except Exception:
            return None

        contract = self.contract_selector.select_contract(
            underlying=symbol,
            underlying_price=current_price,
            trend_direction=direction,
            contracts=chain,
        )
        if not contract:
            return None

        # ─── SCORE ───
        inputs = OptionsConfidenceInputs(
            direction=direction,
            mode=mode,
            d1_trend=d1_trend, h4_trend=h4_trend, h1_trend=h1_trend,
            adx_h4=h4_adx, adx_d1=d1_adx,
            trend_bars_h4=trend_bars_h4,
            pullback_quality=pullback_quality,
            momentum_trigger=momentum_trigger,
            macd_hist_flip=macd_hist_flip,
            strong_candle=strong_candle,
            ema_reclaim=ema_reclaim,
            volume_confirm=volume_confirm,
            atr_expansion=atr_expansion,
            volume_spike=volume_spike,
            adx_rising=adx_rising,
            range_breakout=range_breakout,
            squeeze_release=squeeze_release,
            iv_rank=50.0,  # TODO: historical IV
            spread_pct=contract.spread_pct,
            open_interest=contract.open_interest,
            delta=0.45,  # TODO: greeks from broker
            dte=contract.dte,
            news_aligned=news_aligned, news_against=news_against,
            high_impact_today=has_impact,
            earnings_soon=False,
            spread_vs_atr=contract.spread_pct * contract.mid / (atr_20 if atr_20 > 0 else 1),
            minutes_since_open=max(0, minutes_since_open),
            minutes_to_close=max(0, minutes_to_close),
        )

        result = score_options_setup(inputs)

        threshold = MOMENTUM_THRESHOLD if mode == "momentum" else SWING_THRESHOLD
        if tier == 2:
            threshold += TIER2_CONFIDENCE_BONUS

        if result.confidence < threshold:
            logger.info(
                f"OPTIONS REJECT: {symbol} {mode.upper()} {direction} "
                f"conf={result.confidence:.2f}<{threshold:.2f} | "
                f"{'  '.join(result.reasons[:3])}"
            )
            return None

        option_type = "CALL" if direction == "buy" else "PUT"
        score_breakdown = " | ".join(f"{k}={v:.2f}" for k, v in result.scores.items())

        reasoning = (
            f"[{mode.upper()}] {option_type} {symbol} "
            f"${contract.strike} exp {contract.expiration}\n"
            f"Confidence: {result.confidence:.0%} ({score_breakdown})\n"
            f"{'  |  '.join(result.reasons[:6])}\n"
            f"D1:{d1_trend} H4:{h4_trend} H1:{h1_trend} | "
            f"ADX:{h4_adx:.0f} RSI:{rsi_h1:.0f}\n"
            f"ATR expand:{atr_expansion:.1f}x | Vol spike:{volume_spike:.1f}x\n"
            f"Contract: {contract.dte}DTE spread:{contract.spread_pct:.1%} "
            f"OI:{contract.open_interest} premium:${contract.mid:.2f}"
        )

        return {
            "symbol": symbol,
            "side": direction,
            "mode": mode,
            "tier": tier,
            "confidence": result.confidence,
            "confidence_scores": result.scores,
            "reasons": result.reasons,
            "entry": current_price,
            "contract": contract,
            "reasoning": reasoning,
            "atr": atr_20,
            "rsi": rsi_h1,
            "adx_h4": h4_adx,
        }

    def _assess_pullback(self, closes, highs, lows, direction, rsi) -> float:
        score = 0.0
        sma20 = float(np.mean(closes[-20:]))
        if direction == "buy":
            if 30 <= rsi <= 45: score += 0.4
            if closes[-1] > sma20 and closes[-2] < closes[-1]: score += 0.3
            if closes[-1] < float(np.max(highs[-10:])): score += 0.3
        else:
            if 55 <= rsi <= 70: score += 0.4
            if closes[-1] < sma20 and closes[-2] > closes[-1]: score += 0.3
            if closes[-1] > float(np.min(lows[-10:])): score += 0.3
        return min(1.0, score)

    def _detect_trend(self, closes, highs, lows) -> tuple[str, float]:
        if len(closes) < 20:
            return "flat", 15.0
        sma20 = float(np.mean(closes[-20:]))
        sma10 = float(np.mean(closes[-10:]))
        adx = self._adx(highs, lows, closes)
        if closes[-1] > sma20 and sma10 > sma20:
            return "up", adx
        elif closes[-1] < sma20 and sma10 < sma20:
            return "down", adx
        return "flat", adx

    def _count_trend_bars(self, closes, trend) -> int:
        if trend == "flat" or len(closes) < 2:
            return 0
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if trend == "up" and closes[i] > closes[i-1]:
                count += 1
            elif trend == "down" and closes[i] < closes[i-1]:
                count += 1
            else:
                break
        return count

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(np.mean(data))
        m = 2 / (period + 1)
        ema = float(data[0])
        for p in data[1:]:
            ema = (float(p) - ema) * m + ema
        return ema

    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        d = np.diff(closes)
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag, al = np.mean(g[-period:]), np.mean(l[-period:])
        if al == 0: return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    def _atr(self, highs, lows, closes, period=14) -> float:
        if len(highs) < period + 1:
            return float(np.mean(highs - lows))
        tr = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(highs))]
        return float(np.mean(tr[-period:]))

    def _adx(self, highs, lows, closes, period=14) -> float:
        if len(highs) < period + 2:
            return 15.0
        pdm, mdm, trl = [], [], []
        for i in range(1, len(highs)):
            u, d = highs[i]-highs[i-1], lows[i-1]-lows[i]
            pdm.append(u if u > d and u > 0 else 0)
            mdm.append(d if d > u and d > 0 else 0)
            trl.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
        as_ = float(np.mean(trl[:period]))
        ps = float(np.mean(pdm[:period]))
        ms = float(np.mean(mdm[:period]))
        dx = []
        for i in range(period, len(trl)):
            as_ = as_ - as_/period + trl[i]
            ps = ps - ps/period + pdm[i]
            ms = ms - ms/period + mdm[i]
            if as_ > 0:
                pi, mi = 100*ps/as_, 100*ms/as_
                s = pi + mi
                dx.append(100*abs(pi-mi)/s if s > 0 else 0)
        return float(np.mean(dx[-period:])) if len(dx) >= period else (float(np.mean(dx)) if dx else 15.0)
