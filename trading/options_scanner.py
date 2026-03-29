"""
Options Scanner — Scans stocks for options trading signals.

Uses the same HTF trend filter as forex:
1. Check D1 + H4 trend on the underlying stock
2. Look for H1 pullback entry
3. Select optimal option contract
4. Apply options-specific risk rules

Only trades during US regular market hours (9:30 AM - 4:00 PM ET).
Avoids first 15 minutes after open and last 15 minutes before close.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from trading.brokers.alpaca import AlpacaBroker, OPTIONS_SYMBOLS
from trading.options_contract_selector import ContractSelector, ContractSelection

logger = logging.getLogger(__name__)


def is_options_market_open(now: datetime = None) -> bool:
    """
    Check if US options market is in tradeable hours.

    Regular hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
    We avoid: first 15 min, last 15 min
    So tradeable: 9:45 AM - 3:45 PM ET (14:45 - 20:45 UTC)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    weekday = now.weekday()
    if weekday >= 5:  # Saturday/Sunday
        return False

    hour = now.hour
    minute = now.minute
    total_minutes = hour * 60 + minute

    # 14:45 UTC (9:45 AM ET) to 20:45 UTC (3:45 PM ET)
    # Note: This is simplified — doesn't account for DST
    open_time = 14 * 60 + 45   # 14:45 UTC
    close_time = 20 * 60 + 45  # 20:45 UTC

    return open_time <= total_minutes <= close_time


class OptionsScanner:
    """
    Scans stocks for options trading setups.

    Uses HTF trend filter on the underlying, then selects
    the best option contract for the trade.
    """

    def __init__(self, broker: AlpacaBroker):
        self.broker = broker
        self.contract_selector = ContractSelector()
        self.symbols = OPTIONS_SYMBOLS

    def scan_all(self) -> list[dict]:
        """
        Scan all stocks for options signals.

        Returns list of signals with contract selection info.
        """
        if not self.broker.connected:
            logger.warning("Alpaca not connected — skipping options scan")
            return []

        if not is_options_market_open():
            logger.info("Options market closed — skipping scan")
            return []

        signals = []
        for symbol in self.symbols:
            try:
                signal = self._analyze_stock(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue

        signals.sort(key=lambda x: x["confidence"], reverse=True)
        logger.info(f"Options scan: {len(signals)} signals from {len(self.symbols)} stocks")
        return signals

    def _analyze_stock(self, symbol: str) -> Optional[dict]:
        """Analyze a stock for options trading using HTF trend filter."""
        # Get candles across timeframes
        candles_h1 = self.broker.get_candles(symbol, "H1", 100)
        candles_h4 = self.broker.get_candles(symbol, "H4", 50)
        candles_d1 = self.broker.get_candles(symbol, "D", 30)

        if len(candles_h1) < 50 or len(candles_h4) < 20:
            return None

        closes_h1 = np.array([c["close"] for c in candles_h1])
        highs_h1 = np.array([c["high"] for c in candles_h1])
        lows_h1 = np.array([c["low"] for c in candles_h1])
        volumes_h1 = np.array([c["volume"] for c in candles_h1])
        closes_h4 = np.array([c["close"] for c in candles_h4])

        current_price = closes_h1[-1]

        # Get live quote
        quote = self.broker.get_quote(symbol)
        if quote:
            current_price = quote.mid

        # ─── HTF TREND GATE (same logic as forex) ───
        h4_sma_20 = np.mean(closes_h4[-20:])
        h4_sma_10 = np.mean(closes_h4[-10:])
        h4_trend = "up" if closes_h4[-1] > h4_sma_20 and h4_sma_10 > h4_sma_20 else \
                   "down" if closes_h4[-1] < h4_sma_20 and h4_sma_10 < h4_sma_20 else \
                   "flat"

        d1_trend = "flat"
        if len(candles_d1) >= 20:
            closes_d1 = np.array([c["close"] for c in candles_d1])
            d1_sma_20 = np.mean(closes_d1[-20:])
            d1_sma_10 = np.mean(closes_d1[-10:])
            d1_trend = "up" if closes_d1[-1] > d1_sma_20 and d1_sma_10 > d1_sma_20 else \
                       "down" if closes_d1[-1] < d1_sma_20 and d1_sma_10 < d1_sma_20 else \
                       "flat"

        # Determine allowed direction
        if h4_trend == "up" and d1_trend in ("up", "flat"):
            htf_bias = "buy"  # → CALL
        elif h4_trend == "down" and d1_trend in ("down", "flat"):
            htf_bias = "sell"  # → PUT
        elif d1_trend == "up" and h4_trend == "flat":
            htf_bias = "buy"
        elif d1_trend == "down" and h4_trend == "flat":
            htf_bias = "sell"
        else:
            return None  # No clear trend — skip

        # ─── H1 CONFIRMATION ───
        sma_20 = np.mean(closes_h1[-20:])
        sma_50 = np.mean(closes_h1[-50:])
        rsi = self._rsi(closes_h1, 14)
        ema_12 = self._ema(closes_h1, 12)
        ema_26 = self._ema(closes_h1, 26)
        macd = ema_12 - ema_26
        atr = self._atr(highs_h1, lows_h1, closes_h1, 14)

        score = 0.25  # HTF trend passed
        reasons = [f"HTF: D1={d1_trend} H4={h4_trend} → {'CALL' if htf_bias == 'buy' else 'PUT'}"]

        if htf_bias == "buy":
            if sma_20 > sma_50 and current_price > sma_20:
                score += 0.15
                reasons.append("H1 MAs aligned bullish")
            if 30 <= rsi <= 45:
                score += 0.15
                reasons.append(f"RSI pullback ({rsi:.0f})")
            if macd > 0:
                score += 0.1
                reasons.append("MACD bullish")
        else:
            if sma_20 < sma_50 and current_price < sma_20:
                score += 0.15
                reasons.append("H1 MAs aligned bearish")
            if 55 <= rsi <= 70:
                score += 0.15
                reasons.append(f"RSI pullback ({rsi:.0f})")
            if macd < 0:
                score += 0.1
                reasons.append("MACD bearish")

        # Minimum threshold
        if score < 0.40:
            return None

        confidence = min(score, 0.95)

        # ─── SELECT OPTION CONTRACT ───
        chain = self.broker.get_options_chain(
            symbol,
            option_type="call" if htf_bias == "buy" else "put",
        )

        # Get quotes for contracts
        for c in chain:
            oq = self.broker.get_option_quote(c.symbol)
            if oq:
                c.bid = oq.bid
                c.ask = oq.ask

        contract = self.contract_selector.select_contract(
            underlying=symbol,
            underlying_price=current_price,
            trend_direction=htf_bias,
            contracts=chain,
        )

        if not contract:
            logger.info(f"{symbol}: No suitable contract found")
            return None

        reasoning = (
            f"{'  |  '.join(reasons)}\n"
            f"RSI: {rsi:.0f} | SMA20: {sma_20:.2f} | SMA50: {sma_50:.2f}\n"
            f"{contract.reasoning}"
        )

        return {
            "symbol": symbol,
            "side": htf_bias,
            "option_type": contract.option_type,
            "confidence": confidence,
            "entry": current_price,
            "contract": contract,
            "option_symbol": contract.symbol,
            "strike": contract.strike,
            "expiration": contract.expiration,
            "dte": contract.dte,
            "premium": contract.mid,
            "max_loss": contract.max_loss,
            "spread_pct": contract.spread_pct,
            "reasoning": reasoning,
            "d1_trend": d1_trend,
            "h4_trend": h4_trend,
            "htf_bias": htf_bias,
            "rsi": rsi,
            "atr": atr,
        }

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(np.mean(data))
        multiplier = 2 / (period + 1)
        ema = float(data[0])
        for price in data[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
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

    def _atr(self, highs, lows, closes, period=14) -> float:
        if len(highs) < period + 1:
            return float(np.mean(highs - lows))
        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            true_ranges.append(tr)
        return float(np.mean(true_ranges[-period:]))
