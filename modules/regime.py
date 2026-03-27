"""
regime.py — Market regime detection for options and forex markets.

Detects one of five regimes for any ticker or forex pair:
  - bullish_trend   : sustained upward price movement
  - bearish_trend   : sustained downward price movement
  - high_volatility : large swings with no clear direction
  - low_volatility  : tight price range, low ATR
  - choppy          : no trend, whipsawing — trading is reduced or blocked

The scanner uses regime as a pre-filter before generating signals:
  - Breakout calls   → favored in bullish_trend
  - Breakout puts    → favored in bearish_trend
  - Any trade        → reduced or blocked in choppy regime
  - Volatility plays → possible in high_volatility (not implemented here)

Regime is logged to data/regime_log.json so you can review how well
regime filtering helps over time.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


REGIME_LOG_FILE = "data/regime_log.json"
CHOPPY_ALLOW_ENV = "ALLOW_CHOPPY_TRADING"  # set to 'true' in .env to override


class RegimeDetector:
    """
    Detects market regime from a price history DataFrame.

    Inputs:
        df — DataFrame with columns: date, open, high, low, close, volume
             Must have at least 50 rows for reliable regime detection.

    Outputs:
        regime string: 'bullish_trend' | 'bearish_trend' |
                       'high_volatility' | 'low_volatility' | 'choppy'
    """

    def __init__(self):
        self.allow_choppy = os.getenv(CHOPPY_ALLOW_ENV, "false").lower() in ("true", "1")

    def detect(self, df: pd.DataFrame, ticker: str = "") -> dict:
        """
        Analyze price history and return regime classification.

        Returns:
            {
                regime: str,
                confidence: float (0-1),
                trend_slope: float,
                atr_pct: float,
                adx: float,
                reasoning: str,
                trading_allowed: bool,
            }
        """
        if df is None or len(df) < 20:
            return self._unknown()

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df = df.dropna(subset=["close", "high", "low"])

        if len(df) < 20:
            return self._unknown()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ── Indicators ──────────────────────────────
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50) if len(close) >= 50 else ema20
        current_price = close[-1]
        ema20_now = ema20[-1]
        ema50_now = ema50[-1]

        # ATR (Average True Range) as % of price
        atr = self._atr(high, low, close, 14)
        atr_pct = (atr[-1] / current_price) * 100 if current_price > 0 else 0

        # Trend slope: linear regression on last 20 closes
        slope = self._trend_slope(close[-20:])

        # ADX (trend strength) — simplified
        adx = self._adx(high, low, close, 14)

        # Price vs EMAs
        above_ema20 = current_price > ema20_now
        above_ema50 = current_price > ema50_now
        ema20_above_ema50 = ema20_now > ema50_now

        # ── Classification ──────────────────────────
        regime = "choppy"
        confidence = 0.5
        reasons = []

        if adx > 25:
            # Strong trend present
            if slope > 0 and above_ema20 and ema20_above_ema50:
                regime = "bullish_trend"
                confidence = min(0.5 + adx / 100, 0.95)
                reasons.append(f"ADX={adx:.1f} (strong trend), price above EMA20+EMA50, slope positive")
            elif slope < 0 and not above_ema20 and not ema20_above_ema50:
                regime = "bearish_trend"
                confidence = min(0.5 + adx / 100, 0.95)
                reasons.append(f"ADX={adx:.1f} (strong trend), price below EMA20+EMA50, slope negative")
            else:
                # Conflicting signals → moderate trend
                if slope > 0:
                    regime = "bullish_trend"
                    confidence = 0.55
                    reasons.append(f"ADX={adx:.1f} trend, positive slope but mixed EMA signals")
                else:
                    regime = "bearish_trend"
                    confidence = 0.55
                    reasons.append(f"ADX={adx:.1f} trend, negative slope but mixed EMA signals")
        elif atr_pct > 2.5:
            regime = "high_volatility"
            confidence = min(0.4 + atr_pct / 10, 0.85)
            reasons.append(f"ATR={atr_pct:.1f}% of price (high volatility), ADX={adx:.1f} (weak trend)")
        elif atr_pct < 0.5:
            regime = "low_volatility"
            confidence = 0.70
            reasons.append(f"ATR={atr_pct:.1f}% of price (very tight range), ADX={adx:.1f}")
        else:
            regime = "choppy"
            confidence = 0.6
            reasons.append(f"ADX={adx:.1f} (weak trend), ATR={atr_pct:.1f}%, no clear direction")

        trading_allowed = self._is_trading_allowed(regime)
        reasoning = " | ".join(reasons)

        result = {
            "regime": regime,
            "confidence": round(confidence, 2),
            "trend_slope": round(float(slope), 6),
            "atr_pct": round(float(atr_pct), 2),
            "adx": round(float(adx), 1),
            "reasoning": reasoning,
            "trading_allowed": trading_allowed,
            "timestamp": datetime.now().isoformat(),
        }

        if ticker:
            self._log_regime(ticker, result)

        return result

    def _is_trading_allowed(self, regime: str) -> bool:
        """Returns whether trading is permitted in this regime."""
        if regime == "choppy" and not self.allow_choppy:
            return False
        return True

    def is_favorable_for_calls(self, regime: str) -> bool:
        """Calls are favored in bullish trends."""
        return regime == "bullish_trend"

    def is_favorable_for_puts(self, regime: str) -> bool:
        """Puts are favored in bearish trends."""
        return regime == "bearish_trend"

    def is_favorable_for_direction(self, regime: str, direction: str) -> bool:
        """Check if a direction (call/buy or put/sell) fits the current regime."""
        d = direction.lower()
        if d in ("call", "buy", "long", "bullish"):
            return self.is_favorable_for_calls(regime)
        elif d in ("put", "sell", "short", "bearish"):
            return self.is_favorable_for_puts(regime)
        return False

    # ──────────────────────────────────────────────
    # TECHNICAL HELPERS
    # ──────────────────────────────────────────────

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        ema = np.zeros_like(data, dtype=float)
        k = 2.0 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range."""
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        tr[0] = high[0] - low[0]
        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """
        Simplified ADX (Average Directional Index).
        Returns a single float representing trend strength (0-100).
        >25 = trending, <20 = not trending.
        """
        n = len(close)
        if n < period + 1:
            return 0.0

        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            pdm = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            ndm = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr_list.append(tr)
            pdm_list.append(pdm)
            ndm_list.append(ndm)

        def smooth(arr, p):
            s = sum(arr[:p])
            result = [s]
            for v in arr[p:]:
                s = s - s / p + v
                result.append(s)
            return result

        atr_s = smooth(tr_list, period)
        pdm_s = smooth(pdm_list, period)
        ndm_s = smooth(ndm_list, period)

        dx_list = []
        for a, p, nd in zip(atr_s, pdm_s, ndm_s):
            if a == 0:
                continue
            pdi = 100 * p / a
            ndi = 100 * nd / a
            denom = pdi + ndi
            dx = 100 * abs(pdi - ndi) / denom if denom > 0 else 0
            dx_list.append(dx)

        if not dx_list:
            return 0.0
        return float(np.mean(dx_list[-period:]))

    def _trend_slope(self, prices: np.ndarray) -> float:
        """
        Normalized slope of a linear regression on price.
        Positive = uptrend, negative = downtrend.
        Normalized by dividing by mean price so it's scale-independent.
        """
        x = np.arange(len(prices), dtype=float)
        if len(x) < 2:
            return 0.0
        coeffs = np.polyfit(x, prices.astype(float), 1)
        mean_price = np.mean(prices)
        if mean_price == 0:
            return 0.0
        return coeffs[0] / mean_price

    def _unknown(self) -> dict:
        return {
            "regime": "unknown",
            "confidence": 0.0,
            "trend_slope": 0.0,
            "atr_pct": 0.0,
            "adx": 0.0,
            "reasoning": "Insufficient price data",
            "trading_allowed": False,
            "timestamp": datetime.now().isoformat(),
        }

    # ──────────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────────

    def _log_regime(self, ticker: str, regime_data: dict):
        """Append regime detection to regime_log.json."""
        try:
            os.makedirs("data", exist_ok=True)
            log = []
            if os.path.exists(REGIME_LOG_FILE):
                with open(REGIME_LOG_FILE) as f:
                    log = json.load(f)
            log.append({"ticker": ticker, **regime_data})
            # Keep last 500 entries
            log = log[-500:]
            with open(REGIME_LOG_FILE, "w") as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass

    def get_regime_history(self, ticker: str, limit: int = 20) -> list:
        """Return recent regime detections for a ticker."""
        if not os.path.exists(REGIME_LOG_FILE):
            return []
        try:
            with open(REGIME_LOG_FILE) as f:
                log = json.load(f)
            return [e for e in log if e.get("ticker") == ticker][-limit:]
        except Exception:
            return []
