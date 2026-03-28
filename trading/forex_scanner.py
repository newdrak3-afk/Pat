"""
Forex Scanner — Scans forex pairs for trading signals.

Uses OANDA candlestick data + technical analysis to find setups.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Optional

from trading.brokers.oanda import OandaBroker, FOREX_PAIRS, CRYPTO_PAIRS, ALL_PAIRS
from trading.brokers.base import Quote

logger = logging.getLogger(__name__)


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
        Scan all pairs and return signals.

        Auto-detects weekend: skips forex (closed), scans crypto (24/7).
        On weekdays: scans both forex and crypto.

        Returns list of dicts with:
        - symbol, side, confidence, entry, stop_loss, take_profit, reasoning
        """
        if not self.broker.connected:
            logger.warning("OANDA not connected — skipping scan")
            return []

        # Always scan everything — forex candle history works even on weekends
        # (generates signals from recent H1/H4 data before market closed)
        pairs = ALL_PAIRS if include_crypto else FOREX_PAIRS
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
        """Analyze a single forex pair for trading signals."""
        # Get candle data
        candles_h1 = self.broker.get_candles(symbol, "H1", 100)
        candles_h4 = self.broker.get_candles(symbol, "H4", 50)

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

        # ─── INDICATORS ───

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

        # Higher timeframe trend (H4)
        h4_sma_20 = np.mean(closes_h4[-20:])
        h4_trend = "up" if closes_h4[-1] > h4_sma_20 else "down"

        # ─── SIGNAL LOGIC ───

        score = 0.0
        reasons = []
        side = None

        # 1. MA Crossover (strict — price crossed SMA20 on last bar)
        if sma_20 > sma_50 and closes_h1[-2] < sma_20 and current_price > sma_20:
            score += 0.2
            reasons.append("Price crossed above SMA20 (bullish)")
            side = "buy"
        elif sma_20 < sma_50 and closes_h1[-2] > sma_20 and current_price < sma_20:
            score += 0.2
            reasons.append("Price crossed below SMA20 (bearish)")
            side = "sell"

        # 1b. MA Trend (relaxed — price is on one side of SMA20)
        if side is None:
            if current_price > sma_20 and sma_20 > sma_50:
                score += 0.1
                reasons.append("Above SMA20 > SMA50 (bullish trend)")
                side = "buy"
            elif current_price < sma_20 and sma_20 < sma_50:
                score += 0.1
                reasons.append("Below SMA20 < SMA50 (bearish trend)")
                side = "sell"

        # 2. RSI
        if rsi < 30:
            score += 0.2
            reasons.append(f"RSI oversold ({rsi:.0f})")
            if side is None:
                side = "buy"
        elif rsi > 70:
            score += 0.2
            reasons.append(f"RSI overbought ({rsi:.0f})")
            if side is None:
                side = "sell"
        elif rsi < 40 and side == "buy":
            score += 0.1
            reasons.append(f"RSI leaning oversold ({rsi:.0f})")
        elif rsi > 60 and side == "sell":
            score += 0.1
            reasons.append(f"RSI leaning overbought ({rsi:.0f})")

        # 3. MACD
        if macd > 0 and side == "buy":
            score += 0.15
            reasons.append("MACD bullish")
        elif macd < 0 and side == "sell":
            score += 0.15
            reasons.append("MACD bearish")

        # 4. H4 trend alignment
        if h4_trend == "up" and side == "buy":
            score += 0.2
            reasons.append("Aligned with H4 uptrend")
        elif h4_trend == "down" and side == "sell":
            score += 0.2
            reasons.append("Aligned with H4 downtrend")
        elif h4_trend != ("up" if side == "buy" else "down") and side:
            score -= 0.1
            reasons.append("Against H4 trend (caution)")

        # 5. Volume confirmation
        if volume_spike and side:
            score += 0.1
            reasons.append("Volume spike confirms move")

        # 6. Trend strength
        if side == "buy" and current_price > sma_50:
            score += 0.1
            reasons.append("Above SMA50 (strong trend)")
        elif side == "sell" and current_price < sma_50:
            score += 0.1
            reasons.append("Below SMA50 (strong trend)")

        # ─── FILTER ───

        if side is None or score < 0.15:
            return None

        confidence = min(score, 0.95)

        # Calculate pip value based on instrument type
        is_crypto = symbol in CRYPTO_PAIRS
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
