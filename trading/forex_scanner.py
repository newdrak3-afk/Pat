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
from trading.confidence import compute_confidence, ConfidenceInputs
from trading.news_sentiment import NewsReader

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
        self.news = NewsReader()

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

        # MACD histogram (current and previous for momentum delta)
        macd_line = ema_12 - ema_26
        macd_values = np.array([
            self._ema(closes_h1[:i+1], 12) - self._ema(closes_h1[:i+1], 26)
            for i in range(max(25, len(closes_h1)-10), len(closes_h1))
        ])
        signal_line = self._ema(macd_values, 9)
        macd_hist = macd_values[-1] - signal_line if len(macd_values) > 0 else 0.0
        macd_hist_prev = (macd_values[-2] - self._ema(macd_values[:-1], 9)) if len(macd_values) > 1 else 0.0

        # ATR (Average True Range) for stop loss
        atr = self._atr(highs_h1, lows_h1, closes_h1, 14)

        # H4 ADX (trend strength)
        highs_h4 = np.array([c["high"] for c in candles_h4])
        lows_h4 = np.array([c["low"] for c in candles_h4])
        adx_h4 = self._adx(highs_h4, lows_h4, closes_h4, 14)

        # Regime detection (simple: ATR-based)
        atr_20 = self._atr(highs_h1, lows_h1, closes_h1, 20)
        atr_5 = self._atr(highs_h1[-6:], lows_h1[-6:], closes_h1[-6:], 5) if len(closes_h1) >= 6 else atr
        regime = "trending" if adx_h4 >= 20 else "ranging"
        if atr_5 > atr_20 * 2.0:
            regime = "volatile"

        # Pip value
        if is_crypto:
            pip = 1.0
        elif "JPY" in symbol:
            pip = 0.01
        else:
            pip = 0.0001

        atr_pips = atr / pip
        spread_pips = quote.spread / pip

        # ─── CONFIDENCE MODULE (ChatGPT structured scoring) ───

        side = htf_bias  # Locked to higher-timeframe direction

        inputs = ConfidenceInputs(
            direction=side,
            d1_trend=d1_trend,
            h4_trend=h4_trend,
            rsi_h1=rsi,
            macd_hist_h1=macd_hist,
            macd_hist_prev_h1=macd_hist_prev,
            sma20_h1=sma_20,
            sma50_h1=sma_50,
            close_h1=float(closes_h1[-1]),
            adx_h4=adx_h4,
            atr_pips_h1=atr_pips,
            spread_pips=spread_pips,
            regime=regime,
        )

        result = compute_confidence(inputs)

        # Minimum 0.35 to take a trade
        if result.confidence < 0.35:
            return None

        confidence = result.confidence
        reasons = list(result.reasons)

        # News sentiment (soft filter — adjusts confidence, doesn't block)
        try:
            news = self.news.get_sentiment(symbol)
            if news.headline_count > 0:
                if side == "buy" and news.sentiment == "bearish" and news.score < -0.3:
                    confidence *= 0.85  # Penalize buying into bearish news
                    reasons.append(f"news: bearish ({news.score:+.2f})")
                elif side == "sell" and news.sentiment == "bullish" and news.score > 0.3:
                    confidence *= 0.85  # Penalize selling into bullish news
                    reasons.append(f"news: bullish ({news.score:+.2f})")
                elif (side == "buy" and news.sentiment == "bullish") or \
                     (side == "sell" and news.sentiment == "bearish"):
                    confidence *= 1.05  # Small boost for aligned news
                    confidence = min(confidence, 1.0)
                    reasons.append(f"news: aligned ({news.score:+.2f})")

            # Check for high-impact events (avoid trading around them)
            has_impact, impacts = self.news.has_high_impact_event()
            if has_impact:
                confidence *= 0.7  # Heavy penalty during major news
                reasons.append(f"news: HIGH IMPACT EVENT")
        except Exception:
            pass  # News is optional — don't block trades if it fails

        # Recheck confidence after news adjustment
        if confidence < 0.35:
            return None

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
            f"RSI: {rsi:.0f} | ADX: {adx_h4:.0f} | Regime: {regime}\n"
            f"D1: {d1_trend} | H4: {h4_trend} | HTF bias: {htf_bias}\n"
            f"ATR: {atr_pips:.1f} pips | Spread: {spread_pips:.1f} pips | SL: {sl_pips:.0f} pips | TP: {tp_pips:.0f} pips"
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

    def _adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Average Directional Index (trend strength)."""
        if len(highs) < period + 2:
            return 15.0  # Default to low trend strength

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(highs)):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_list.append(tr)

        # Smoothed averages
        atr_s = float(np.mean(tr_list[:period]))
        plus_di_s = float(np.mean(plus_dm[:period]))
        minus_di_s = float(np.mean(minus_dm[:period]))

        dx_list = []
        for i in range(period, len(tr_list)):
            atr_s = atr_s - atr_s / period + tr_list[i]
            plus_di_s = plus_di_s - plus_di_s / period + plus_dm[i]
            minus_di_s = minus_di_s - minus_di_s / period + minus_dm[i]

            if atr_s > 0:
                plus_di = 100 * plus_di_s / atr_s
                minus_di = 100 * minus_di_s / atr_s
                di_sum = plus_di + minus_di
                dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
                dx_list.append(dx)

        if not dx_list:
            return 15.0

        # ADX = smoothed average of DX
        adx = float(np.mean(dx_list[-period:])) if len(dx_list) >= period else float(np.mean(dx_list))
        return adx

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
