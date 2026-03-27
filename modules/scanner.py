"""
scanner.py — Multi-signal confidence scoring engine for options and forex.

Scans a watchlist of tickers/pairs and generates directional trade signals
only when multiple indicators agree and confidence reaches MIN_CONFIDENCE.

Signals used (weights loaded from data/signal_weights.json, defaults below):
  ┌─────────────────────────────┬────────┬──────────────────────────────────────────┐
  │ Signal                      │ Weight │ Bullish condition                        │
  ├─────────────────────────────┼────────┼──────────────────────────────────────────┤
  │ Momentum breakout           │  25    │ Price > 20-day high + vol > 1.5x avg     │
  │ RSI                         │  20    │ RSI crosses above 35 from below          │
  │ EMA crossover               │  20    │ 9-EMA crosses above 21-EMA               │
  │ News sentiment              │  20    │ Sentiment score > 0.3                    │
  │ Options/options volume      │  15    │ Call vol > 3x avg AND > put vol          │
  └─────────────────────────────┴────────┴──────────────────────────────────────────┘

Each signal returns 0 (not fired), partial, or full weight points.
Total score / max_possible_score = confidence %.
Only signals with confidence >= MIN_CONFIDENCE are returned.

Regime filter is applied before scoring: choppy regime blocks signals
(unless ALLOW_CHOPPY_TRADING=true), and direction must align with regime.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional

from modules.regime import RegimeDetector
from modules.news_sentiment import NewsSentiment

SIGNAL_WEIGHTS_FILE = "data/signal_weights.json"

DEFAULT_WEIGHTS = {
    "momentum_breakout": 25,
    "rsi": 20,
    "ema_crossover": 20,
    "news_sentiment": 20,
    "options_volume": 15,
}


class Scanner:
    """Multi-signal options/forex scanner with confidence scoring."""

    def __init__(self, client=None, forex_client=None):
        """
        Args:
            client: TradierClient instance (for options data)
            forex_client: OandaClient instance (for forex data)
        """
        self.client = client
        self.forex_client = forex_client
        self.regime = RegimeDetector()
        self.news = NewsSentiment()
        self.min_confidence = int(os.getenv("MIN_CONFIDENCE", "70"))
        self.weights = self._load_weights()

    def _load_weights(self) -> dict:
        """Load signal weights from file (set by learner) or use defaults."""
        if os.path.exists(SIGNAL_WEIGHTS_FILE):
            try:
                with open(SIGNAL_WEIGHTS_FILE) as f:
                    saved = json.load(f)
                w = {**DEFAULT_WEIGHTS, **saved.get("weights", {})}
                return w
            except Exception:
                pass
        return dict(DEFAULT_WEIGHTS)

    # ──────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────

    def scan_ticker(self, ticker: str, market: str = "options") -> Optional[dict]:
        """
        Scan a single ticker or forex pair for trade signals.

        Args:
            ticker: Stock ticker (e.g. 'AAPL') or forex pair (e.g. 'EUR_USD')
            market: 'options' or 'forex'

        Returns:
            Signal dict if confidence >= MIN_CONFIDENCE, else None.
        """
        try:
            df = self._get_price_history(ticker, market)
            if df is None or len(df) < 25:
                return None

            # Regime filter
            regime_data = self.regime.detect(df, ticker)
            if not regime_data["trading_allowed"]:
                return None

            # Score each direction
            bull_result = self._score_direction(ticker, df, "bullish", regime_data, market)
            bear_result = self._score_direction(ticker, df, "bearish", regime_data, market)

            # Take the higher-confidence direction if it passes threshold
            best = bull_result if bull_result["confidence"] >= bear_result["confidence"] else bear_result

            if best["confidence"] < self.min_confidence:
                return None

            direction = "call" if best["direction"] == "bullish" else "put"
            if market == "forex":
                direction = "long" if best["direction"] == "bullish" else "short"

            signal = {
                "ticker": ticker,
                "market": market,
                "direction": direction,
                "confidence": best["confidence"],
                "signals_fired": best["signals_fired"],
                "regime": regime_data["regime"],
                "regime_confidence": regime_data["confidence"],
                "news_label": best.get("news_label", "neutral"),
                "catalyst_risk": best.get("catalyst_risk", False),
                "catalyst_reason": best.get("catalyst_reason", ""),
                "reasoning": self._build_reasoning(ticker, market, direction, best, regime_data),
                "timestamp": datetime.now().isoformat(),
                "current_price": float(df["close"].iloc[-1]),
            }

            # Options-specific suggestions
            if market == "options" and self.client:
                self._add_options_suggestion(signal, df)

            return signal

        except Exception as e:
            return None

    def scan_watchlist(self, tickers: list, market: str = "options") -> list:
        """
        Scan a list of tickers/pairs. Returns all signals sorted by confidence desc.
        """
        signals = []
        for ticker in tickers:
            result = self.scan_ticker(ticker, market)
            if result:
                signals.append(result)
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return signals

    def scan_all_markets(self, options_watchlist: list = None, forex_watchlist: list = None) -> list:
        """
        Scan both options and forex markets based on MARKET_MODE setting.
        Returns combined sorted signal list.
        """
        from modules.forex_data import get_active_markets, get_default_forex_watchlist
        active = get_active_markets()
        all_signals = []

        if "options" in active and options_watchlist:
            all_signals.extend(self.scan_watchlist(options_watchlist, "options"))

        if "forex" in active:
            wl = forex_watchlist or get_default_forex_watchlist()
            all_signals.extend(self.scan_watchlist(wl, "forex"))

        all_signals.sort(key=lambda x: x["confidence"], reverse=True)
        return all_signals

    # ──────────────────────────────────────────────
    # SCORING ENGINE
    # ──────────────────────────────────────────────

    def _score_direction(
        self,
        ticker: str,
        df: pd.DataFrame,
        direction: str,
        regime_data: dict,
        market: str,
    ) -> dict:
        """
        Score all signals for one direction (bullish or bearish).
        Returns: {direction, confidence, signals_fired, ...}
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        signals_fired = []
        score = 0
        max_score = sum(self.weights.values())
        w = self.weights

        # ── 1. Momentum Breakout ─────────────────────
        mb_score = self._signal_momentum(close, volume, direction)
        if mb_score > 0:
            score += mb_score * w["momentum_breakout"] / 100
            signals_fired.append(("momentum_breakout", mb_score))

        # ── 2. RSI ───────────────────────────────────
        rsi_score = self._signal_rsi(close, direction)
        if rsi_score > 0:
            score += rsi_score * w["rsi"] / 100
            signals_fired.append(("rsi", rsi_score))

        # ── 3. EMA Crossover ─────────────────────────
        ema_score = self._signal_ema_crossover(close, direction)
        if ema_score > 0:
            score += ema_score * w["ema_crossover"] / 100
            signals_fired.append(("ema_crossover", ema_score))

        # ── 4. News Sentiment ────────────────────────
        news_result = self.news.score_sentiment(ticker)
        news_score = self._signal_news(news_result, direction)
        if news_score > 0:
            score += news_score * w["news_sentiment"] / 100
            signals_fired.append(("news_sentiment", news_score))

        # ── 5. Options Volume (options markets only) ─
        options_vol_score = 0
        if market == "options" and self.client:
            options_vol_score = self._signal_options_volume(ticker, direction)
            if options_vol_score > 0:
                score += options_vol_score * w["options_volume"] / 100
                signals_fired.append(("options_volume", options_vol_score))
        else:
            # For forex, use RSI weight as placeholder (don't penalize)
            max_score -= w["options_volume"]

        # ── Regime alignment bonus/penalty ──────────
        regime = regime_data["regime"]
        if direction == "bullish" and regime == "bullish_trend":
            score += 5
        elif direction == "bearish" and regime == "bearish_trend":
            score += 5
        elif direction == "bullish" and regime == "bearish_trend":
            score -= 10
        elif direction == "bearish" and regime == "bullish_trend":
            score -= 10

        confidence = int(min(score / max_score * 100, 99)) if max_score > 0 else 0

        return {
            "direction": direction,
            "confidence": confidence,
            "signals_fired": signals_fired,
            "news_label": news_result.get("label", "neutral"),
            "catalyst_risk": news_result.get("catalyst_risk", False),
            "catalyst_reason": news_result.get("catalyst_reason", ""),
            "news_headlines": news_result.get("top_headlines", []),
        }

    # ──────────────────────────────────────────────
    # INDIVIDUAL SIGNAL IMPLEMENTATIONS
    # ──────────────────────────────────────────────

    def _signal_momentum(self, close: np.ndarray, volume: np.ndarray, direction: str) -> float:
        """
        Momentum breakout: price > 20-day high with volume > 1.5x average.
        Returns 0-100 score.
        """
        if len(close) < 21:
            return 0
        current = close[-1]
        lookback = close[-21:-1]  # last 20 days excluding today
        avg_vol = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        current_vol = volume[-1] if len(volume) > 0 else 0
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if direction == "bullish":
            high20 = np.max(lookback)
            if current > high20 and vol_ratio >= 1.5:
                # Score by how much it broke out and volume ratio
                breakout_pct = (current - high20) / high20 * 100
                return min(60 + breakout_pct * 5 + (vol_ratio - 1.5) * 10, 100)
        elif direction == "bearish":
            low20 = np.min(lookback)
            if current < low20 and vol_ratio >= 1.5:
                breakdown_pct = (low20 - current) / low20 * 100
                return min(60 + breakdown_pct * 5 + (vol_ratio - 1.5) * 10, 100)
        return 0

    def _signal_rsi(self, close: np.ndarray, direction: str, period: int = 14) -> float:
        """
        RSI crossover signal.
        Bullish: RSI crosses above 35 from below (oversold recovery).
        Bearish: RSI crosses below 65 from above (overbought exhaustion).
        Returns 0-100 score.
        """
        if len(close) < period + 2:
            return 0
        rsi = self._calc_rsi(close, period)
        rsi_now = rsi[-1]
        rsi_prev = rsi[-2]

        if direction == "bullish":
            if rsi_prev < 35 and rsi_now >= 35:
                return 80  # fresh crossover above 35
            elif rsi_now < 45 and rsi_prev < rsi_now:
                return 40  # rising from low territory
        elif direction == "bearish":
            if rsi_prev > 65 and rsi_now <= 65:
                return 80
            elif rsi_now > 55 and rsi_prev > rsi_now:
                return 40
        return 0

    def _signal_ema_crossover(self, close: np.ndarray, direction: str) -> float:
        """
        9-EMA crosses 21-EMA.
        Returns 0-100 score.
        """
        if len(close) < 22:
            return 0
        ema9 = self._ema(close, 9)
        ema21 = self._ema(close, 21)

        ema9_now, ema9_prev = ema9[-1], ema9[-2]
        ema21_now, ema21_prev = ema21[-1], ema21[-2]

        if direction == "bullish":
            if ema9_prev <= ema21_prev and ema9_now > ema21_now:
                return 90  # fresh crossover
            elif ema9_now > ema21_now and (ema9_now - ema21_now) / ema21_now > 0.001:
                return 50  # already above and widening
        elif direction == "bearish":
            if ema9_prev >= ema21_prev and ema9_now < ema21_now:
                return 90
            elif ema9_now < ema21_now and (ema21_now - ema9_now) / ema21_now > 0.001:
                return 50
        return 0

    def _signal_news(self, news_result: dict, direction: str) -> float:
        """
        News sentiment signal.
        Bullish: label=bullish, score > 0.3.
        Bearish: label=bearish, score < -0.3.
        Returns 0-100. Mixed/uncertain news returns 0 (no contribution).
        """
        if not news_result or news_result.get("headlines_scored", 0) == 0:
            return 0
        label = news_result.get("label", "neutral")
        score = news_result.get("score", 0)
        confidence = news_result.get("confidence", 0)

        if label == "mixed":
            return 0  # uncertain news — no signal contribution

        if direction == "bullish" and label == "bullish" and score > 0.15:
            return min(50 + score * 100 + confidence * 30, 100)
        elif direction == "bearish" and label == "bearish" and score < -0.15:
            return min(50 + abs(score) * 100 + confidence * 30, 100)
        return 0

    def _signal_options_volume(self, ticker: str, direction: str) -> float:
        """
        Unusual options volume: call or put volume >> 3x average daily volume.
        """
        if not self.client:
            return 0
        try:
            exps = self.client.get_expirations(ticker)
            if not exps:
                return 0
            summary = self.client.get_options_volume_summary(ticker, exps[0])
            call_vol = summary.get("total_call_volume", 0)
            put_vol = summary.get("total_put_volume", 0)
            total = call_vol + put_vol
            if total == 0:
                return 0
            if direction == "bullish":
                ratio = call_vol / max(put_vol, 1)
                if ratio >= 3.0:
                    return min(60 + ratio * 5, 100)
                elif ratio >= 1.5:
                    return 40
            elif direction == "bearish":
                ratio = put_vol / max(call_vol, 1)
                if ratio >= 3.0:
                    return min(60 + ratio * 5, 100)
                elif ratio >= 1.5:
                    return 40
        except Exception:
            pass
        return 0

    # ──────────────────────────────────────────────
    # OPTIONS CONTRACT SUGGESTION
    # ──────────────────────────────────────────────

    def _add_options_suggestion(self, signal: dict, df: pd.DataFrame):
        """
        Find the best options contract for this signal and add to signal dict.
        Applies contract quality filter before suggesting.
        """
        from modules.risk_engine import RiskEngine
        risk = RiskEngine()

        ticker = signal["ticker"]
        direction_type = signal["direction"]  # 'call' or 'put'
        current_price = signal["current_price"]

        try:
            exps = self.client.get_expirations(ticker)
            if not exps:
                return
            # Prefer expiry 14-30 DTE
            from datetime import date as _date
            today = _date.today()
            best_exp = None
            for exp in exps:
                try:
                    exp_d = datetime.strptime(exp, "%Y-%m-%d").date()
                    dte = (exp_d - today).days
                    if 14 <= dte <= 45:
                        best_exp = exp
                        break
                except Exception:
                    continue
            if not best_exp:
                best_exp = exps[0]

            chain = self.client.get_options_chain(ticker, best_exp)
            ranked = risk.rank_contracts(chain, direction_type)

            if ranked:
                best_contract, _ = ranked[0]
                mid_price = (best_contract["bid"] + best_contract["ask"]) / 2
                signal["suggested_strike"] = best_contract["strike"]
                signal["suggested_expiry"] = best_exp
                signal["est_premium"] = round(mid_price * 100, 2)  # per contract (100 shares)
                signal["suggested_contract"] = best_contract["symbol"]
                signal["contract_delta"] = abs(best_contract.get("delta", 0))
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # REASONING TEXT BUILDER
    # ──────────────────────────────────────────────

    def _build_reasoning(
        self,
        ticker: str,
        market: str,
        direction: str,
        score_data: dict,
        regime_data: dict,
    ) -> str:
        """
        Build honest, risk-aware plain-English explanation for a signal.
        Includes: signals fired, why the contract was chosen, risks, and
        a clear disclaimer that this is not a guaranteed outcome.
        """
        conf = score_data["confidence"]
        signals_fired = score_data["signals_fired"]
        regime = regime_data["regime"]
        catalyst_risk = score_data.get("catalyst_risk", False)
        catalyst_reason = score_data.get("catalyst_reason", "")
        news_label = score_data.get("news_label", "neutral")
        headlines = score_data.get("news_headlines", [])

        dir_upper = direction.upper()
        lines = [f"*{ticker}* → {dir_upper} signal ({conf}% confidence)\n"]

        # Why it scored high
        signal_names = {
            "momentum_breakout": "Momentum breakout (price/volume)",
            "rsi": "RSI signal",
            "ema_crossover": "EMA crossover",
            "news_sentiment": "News sentiment",
            "options_volume": "Options volume",
        }
        for name, score in signals_fired:
            label = signal_names.get(name, name)
            lines.append(f"  ✓ {label} (strength: {score:.0f}/100)")

        # Regime context
        lines.append(f"\n  Regime: {regime.upper()} ({regime_data['confidence']*100:.0f}% confidence)")
        if self.regime.is_favorable_for_direction(regime, direction):
            lines.append(f"  ✓ Regime aligns with {dir_upper} direction")
        else:
            lines.append(f"  ⚠ Regime is not clearly aligned with {dir_upper} direction")

        # News
        if headlines:
            lines.append(f"\n  News: {news_label.upper()}")
            lines.append(f"  → \"{headlines[0]}\"")

        # Catalyst warning
        if catalyst_risk:
            lines.append(f"\n  ⚠ CATALYST RISK: {catalyst_reason}")
            lines.append(f"  Consider waiting for clarity before entering.")

        # Main risks
        lines.append("\n  RISKS TO CONSIDER:")
        if direction in ("call", "long", "buy", "bullish"):
            lines.append("  • Options lose value every day (theta decay)")
            lines.append("  • A reversal or gap-down can hit max loss quickly")
        else:
            lines.append("  • Options lose value every day (theta decay)")
            lines.append("  • Short squeezes or positive news can cause rapid losses")
        if catalyst_risk:
            lines.append("  • Earnings/macro events create binary risk")
        lines.append("  • Signals can fail — past performance does not guarantee results")

        lines.append("\n  ⚠ This is a computer-generated signal, not financial advice.")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # PRICE DATA FETCHER
    # ──────────────────────────────────────────────

    def _get_price_history(self, ticker: str, market: str) -> Optional[pd.DataFrame]:
        """Get price history DataFrame for a ticker (options) or pair (forex)."""
        if market == "forex":
            if self.forex_client and self.forex_client.is_configured():
                return self.forex_client.get_history(ticker, granularity="D", count=120)
            else:
                from modules.forex_data import get_forex_history_yf
                return get_forex_history_yf(ticker, count=120)
        else:
            if self.client and self.client.is_configured():
                return self.client.get_history(ticker)
            else:
                # yfinance fallback
                try:
                    import yfinance as yf
                    from datetime import timedelta
                    end = datetime.today()
                    start = end - timedelta(days=150)
                    df = yf.download(
                        ticker,
                        start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        auto_adjust=True, progress=False,
                    )
                    if df.empty:
                        return None
                    df = df.reset_index()
                    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
                    if "adj_close" in df.columns:
                        df = df.rename(columns={"adj_close": "close"})
                    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
                except Exception:
                    return None

    # ──────────────────────────────────────────────
    # TECHNICAL HELPERS (local, no external deps)
    # ──────────────────────────────────────────────

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros(len(data))
        k = 2.0 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close.astype(float))
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros(len(delta))
        avg_loss = np.zeros(len(delta))

        avg_gain[period - 1] = np.mean(gain[:period])
        avg_loss[period - 1] = np.mean(loss[:period])

        for i in range(period, len(delta)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi])
