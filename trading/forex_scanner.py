"""
Forex Scanner — Scans forex pairs for trading signals.

Uses OANDA candlestick data + technical analysis to find setups.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from trading.brokers.oanda import OandaBroker, FOREX_PAIRS, CRYPTO_PAIRS, ALL_PAIRS
from trading.brokers.base import Quote

logger = logging.getLogger(__name__)


def is_forex_market_open(now: datetime = None) -> bool:
    """
    Check if the forex market is open.

    Forex trades 24/5: opens Sunday 5 PM ET (22:00 UTC), closes Friday 5 PM ET (22:00 UTC).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour

    # Friday after 22:00 UTC → closed
    if weekday == 4 and hour >= 22:
        return False
    # All of Saturday → closed
    if weekday == 5:
        return False
    # Sunday before 22:00 UTC → closed
    if weekday == 6 and hour < 22:
        return False

    return True


class ForexScanner:
    """
    Scans forex pairs for trading signals using:
    - Moving average crossovers
    - RSI (overbought/oversold)
    - Spread analysis
    - Volume spikes
    - Support/resistance levels
    """

    def __init__(self, broker: OandaBroker):
        self.broker = broker

    def scan_all_pairs(self, include_crypto: bool = True) -> list[dict]:
        """
        Scan pairs and return signals.

        Only scans forex when the market is open.
        Crypto is 24/7 but OANDA practice may not support execution.

        Returns list of dicts with:
        - symbol, side, confidence, entry, stop_loss, take_profit, reasoning
        """
        if not self.broker.connected:
            logger.warning("OANDA not connected — skipping scan")
            return []

        forex_open = is_forex_market_open()

        if forex_open:
            pairs = ALL_PAIRS if include_crypto else FOREX_PAIRS
        else:
            # Market closed — only scan crypto (24/7)
            pairs = CRYPTO_PAIRS if include_crypto else []
            if not pairs:
                logger.info("Forex market closed, no crypto pairs — skipping scan")
                return []
            logger.info("Forex market closed — scanning crypto only")

        logger.info(f"Scanning {len(pairs)} pairs...")

        if not pairs:
            logger.info("No pairs to scan")
            return []

        signals = []

        for symbol in pairs:
            try:
                signal = self._analyze_pair(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        logger.info(f"Scan complete: {len(signals)} signals found")

        return signals

    def _analyze_pair(self, symbol: str) -> Optional[dict]:
        """Analyze a single forex pair for trading signals.

        BAR-CLOSE DISCIPLINE: Only uses fully closed candles.
        The most recent candle on each timeframe is dropped because
        it may still be forming. Indicators are computed on closed bars only.
        """
        # Get candle data across timeframes
        # Request extra candles so we still have enough after dropping the last
        candles_h1_raw = self.broker.get_candles(symbol, "H1", 101)
        candles_h4_raw = self.broker.get_candles(symbol, "H4", 51)
        candles_d1_raw = self.broker.get_candles(symbol, "D", 31)

        # Drop the last (potentially forming) candle on each timeframe
        candles_h1 = candles_h1_raw[:-1] if len(candles_h1_raw) > 1 else candles_h1_raw
        candles_h4 = candles_h4_raw[:-1] if len(candles_h4_raw) > 1 else candles_h4_raw
        candles_d1 = candles_d1_raw[:-1] if len(candles_d1_raw) > 1 else candles_d1_raw

        if len(candles_h1) < 50 or len(candles_h4) < 20:
            return None

        closes_h1 = np.array([c["close"] for c in candles_h1])
        highs_h1 = np.array([c["high"] for c in candles_h1])
        lows_h1 = np.array([c["low"] for c in candles_h1])
        volumes_h1 = np.array([c["volume"] for c in candles_h1])

        closes_h4 = np.array([c["close"] for c in candles_h4])

        # Get current quote for spread check
        quote = self.broker.get_quote(symbol)
        is_crypto = symbol in CRYPTO_PAIRS
        max_spread = 0.5 if is_crypto else 0.05  # crypto has wider spreads
        if not quote or quote.spread > max_spread:
            return None

        current_price = quote.mid

        # ─── HIGHER-TIMEFRAME TREND FILTER (GATE) ───
        # This is the key filter: D1 and H4 must agree on direction.
        # H1 only looks for pullback entries WITH the trend.
        # Countertrend trades are blocked entirely.

        # H4 trend: price vs SMA20 + SMA20 slope
        h4_sma_20 = np.mean(closes_h4[-20:])
        h4_sma_10 = np.mean(closes_h4[-10:])
        h4_trend = "up" if closes_h4[-1] > h4_sma_20 and h4_sma_10 > h4_sma_20 else \
                   "down" if closes_h4[-1] < h4_sma_20 and h4_sma_10 < h4_sma_20 else \
                   "flat"

        # D1 trend: price vs SMA20
        d1_trend = "flat"
        if len(candles_d1) >= 20:
            closes_d1 = np.array([c["close"] for c in candles_d1])
            d1_sma_20 = np.mean(closes_d1[-20:])
            d1_sma_10 = np.mean(closes_d1[-10:])
            d1_trend = "up" if closes_d1[-1] > d1_sma_20 and d1_sma_10 > d1_sma_20 else \
                       "down" if closes_d1[-1] < d1_sma_20 and d1_sma_10 < d1_sma_20 else \
                       "flat"

        # Determine allowed trade direction from higher timeframes
        # Both H4 and D1 must agree, or at least one agrees and the other is flat
        if h4_trend == "up" and d1_trend in ("up", "flat"):
            htf_bias = "buy"
        elif h4_trend == "down" and d1_trend in ("down", "flat"):
            htf_bias = "sell"
        elif d1_trend == "up" and h4_trend == "flat":
            htf_bias = "buy"
        elif d1_trend == "down" and h4_trend == "flat":
            htf_bias = "sell"
        else:
            # No clear trend or conflicting — skip this pair entirely
            return None

        # ─── H1 INDICATORS ───

        # Moving averages
        sma_20 = np.mean(closes_h1[-20:])
        sma_50 = np.mean(closes_h1[-50:])
        ema_12 = self._ema(closes_h1, 12)
        ema_26 = self._ema(closes_h1, 26)

        # RSI (14-period)
        rsi = self._rsi(closes_h1, 14)

        # MACD
        macd = ema_12 - ema_26
        signal_line = self._ema(
            np.array([self._ema(closes_h1[:i+1], 12) - self._ema(closes_h1[:i+1], 26)
                      for i in range(max(25, len(closes_h1)-10), len(closes_h1))]),
            9
        )

        # ATR (Average True Range) for stop loss
        atr = self._atr(highs_h1, lows_h1, closes_h1, 14)

        # Volume analysis
        avg_volume = np.mean(volumes_h1[-20:])
        recent_volume = np.mean(volumes_h1[-3:])
        volume_spike = recent_volume > avg_volume * 1.5

        # ─── H1 SIGNAL LOGIC (only in htf_bias direction) ───

        score = 0.0
        reasons = []
        side = htf_bias  # Locked to higher-timeframe direction

        # Higher-timeframe trend bonus (always applies since we passed the gate)
        score += 0.25
        reasons.append(f"HTF trend: D1={d1_trend} H4={h4_trend} → {htf_bias}")

        # 1. H1 MA alignment with trend
        if side == "buy":
            if sma_20 > sma_50 and current_price > sma_20:
                score += 0.15
                reasons.append("H1 MAs aligned bullish")
            elif closes_h1[-2] < sma_20 and current_price > sma_20:
                score += 0.2
                reasons.append("H1 pullback bounce above SMA20")
        else:
            if sma_20 < sma_50 and current_price < sma_20:
                score += 0.15
                reasons.append("H1 MAs aligned bearish")
            elif closes_h1[-2] > sma_20 and current_price < sma_20:
                score += 0.2
                reasons.append("H1 pullback rejection below SMA20")

        # 2. RSI — look for pullback entries, not extreme reversals
        if side == "buy":
            if 30 <= rsi <= 45:
                score += 0.15
                reasons.append(f"RSI pullback zone ({rsi:.0f})")
            elif rsi < 30:
                score += 0.1
                reasons.append(f"RSI oversold ({rsi:.0f})")
        else:
            if 55 <= rsi <= 70:
                score += 0.15
                reasons.append(f"RSI pullback zone ({rsi:.0f})")
            elif rsi > 70:
                score += 0.1
                reasons.append(f"RSI overbought ({rsi:.0f})")

        # 3. MACD confirmation
        if macd > 0 and side == "buy":
            score += 0.1
            reasons.append("MACD bullish")
        elif macd < 0 and side == "sell":
            score += 0.1
            reasons.append("MACD bearish")

        # 4. Volume confirmation
        if volume_spike:
            score += 0.1
            reasons.append("Volume spike confirms move")

        # 5. H1 trend strength
        if side == "buy" and current_price > sma_50:
            score += 0.05
            reasons.append("Above SMA50")
        elif side == "sell" and current_price < sma_50:
            score += 0.05
            reasons.append("Below SMA50")

        # ─── FILTER ───
        # Minimum 0.35 confidence (HTF trend alone gives 0.25, need H1 confirmation)
        if score < 0.35:
            return None

        confidence = min(score, 0.95)

        # Calculate pip value based on instrument type
        if is_crypto:
            pip = 1.0  # Crypto moves in whole dollars
        elif "JPY" in symbol:
            pip = 0.01
        else:
            pip = 0.0001

        # Stop loss = 1.5x ATR, Take profit = 2x ATR (1:1.33 risk/reward)
        sl_distance = atr * 1.5
        tp_distance = atr * 2.0

        if side == "buy":
            entry = quote.ask
            stop_loss = entry - sl_distance
            take_profit = entry + tp_distance
        else:
            entry = quote.bid
            stop_loss = entry + sl_distance
            take_profit = entry - tp_distance

        sl_pips = sl_distance / pip
        tp_pips = tp_distance / pip

        reasoning = (
            f"{'  |  '.join(reasons)}\n"
            f"RSI: {rsi:.0f} | SMA20: {sma_20:.5f} | SMA50: {sma_50:.5f}\n"
            f"D1: {d1_trend} | H4: {h4_trend} | HTF bias: {htf_bias}\n"
            f"ATR: {atr/pip:.1f} pips | SL: {sl_pips:.0f} pips | TP: {tp_pips:.0f} pips"
        )

        return {
            "symbol": symbol,
            "side": side,
            "confidence": confidence,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "units": 1000,  # micro lot, risk manager will adjust
            "reasoning": reasoning,
            "rsi": rsi,
            "atr": atr,
            "h4_trend": h4_trend,
            "d1_trend": d1_trend,
            "htf_bias": htf_bias,
        }

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return float(np.mean(data))

        multiplier = 2 / (period + 1)
        ema = float(data[0])
        for price in data[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return float(np.mean(highs - lows))

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        return float(np.mean(true_ranges[-period:]))
