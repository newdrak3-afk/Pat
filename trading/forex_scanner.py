"""
Forex Scanner — Scans forex pairs for trading signals.

Uses OANDA candlestick data + technical analysis to find setups.

Strategy: Higher-timeframe trend filter (D1+H4 gate) with H1 pullback entries.
Confidence scoring via structured module (confidence.py).
News sentiment as soft/hard filter (news_sentiment.py).
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

# Tier 1 pairs: lowest spreads, deepest liquidity — prioritize these
TIER1_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "USD_CHF", "NZD_USD",
]

# Minimum confidence thresholds by mode
CONFIDENCE_THRESHOLD_DEMO = 0.45
CONFIDENCE_THRESHOLD_LIVE = 0.55


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
    - Higher-timeframe trend filter (D1+H4 gate)
    - Structured confidence scoring (5 dimensions)
    - News sentiment filter (soft + hard blocks)
    - Bar-close discipline (drops forming candles)
    - Lesson-based filtering (learns from past losses)
    """

    def __init__(self, broker: OandaBroker, db=None):
        self.broker = broker
        self.db = db
        self.news = NewsReader()
        self.confidence_threshold = CONFIDENCE_THRESHOLD_DEMO  # caller can override
        self._lesson_rules = {}  # loaded from DB
        self._lessons_loaded_at = None

    def _load_lessons(self):
        """Load lessons from DB and extract actionable rules.

        Only applies rules that are safe for live forex trading:
        - block_symbol:<sym>         → skip this symbol entirely
        - max_spread:<val>           → reject pairs with spread above this
        - min_liquidity:<val>        → reject low-liquidity pairs
        - require_min_sources:<n>    → need N+ confirming sources

        IGNORED (diagnostic only, logged but not enforced):
        - diagnostic:*               → postmortem labels, not live rules
        - raise_confidence_threshold → counterproductive (makes recovery harder)
        - raise_sentiment_threshold  → same issue
        - block_keyword              → too coarse for FX (handled by currency filter)
        - block_category             → FX categories are too broad to block
        - trigger_retrain            → no auto-retrain support
        """
        if not self.db:
            return

        try:
            lessons = self.db.get_lessons(limit=100)
            if not lessons:
                return

            category_counts = {}
            blocked_symbols = set()
            max_spread = None
            min_liquidity = None
            min_sources = None
            diagnostic_count = 0

            for lesson in lessons:
                cat = lesson.get("category", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                rule = lesson.get("rule_added", "")

                if not rule:
                    continue

                # Diagnostic rules — log only, never enforce
                if rule.startswith("diagnostic:"):
                    diagnostic_count += 1
                    continue

                # Safe live rules only
                if rule.startswith("block_symbol:"):
                    blocked_symbols.add(rule.split(":", 1)[1].strip())

                elif rule.startswith("max_spread:"):
                    try:
                        val = float(rule.split(":", 1)[1].strip())
                        if max_spread is None or val < max_spread:
                            max_spread = val
                    except ValueError:
                        pass

                elif rule.startswith("min_liquidity:"):
                    try:
                        val = float(rule.split(":", 1)[1].strip())
                        if min_liquidity is None or val > min_liquidity:
                            min_liquidity = val
                    except ValueError:
                        pass

                elif rule.startswith("require_min_sources:"):
                    try:
                        val = int(rule.split(":", 1)[1].strip())
                        if min_sources is None or val > min_sources:
                            min_sources = val
                    except ValueError:
                        pass

                # IGNORED rules (logged for visibility)
                elif rule.startswith(("raise_confidence_threshold:",
                                      "raise_sentiment_threshold:",
                                      "block_keyword:",
                                      "block_category:",
                                      "trigger_retrain")):
                    logger.debug(f"LESSON IGNORED (not safe for live FX): {rule}")

            self._lesson_rules = {
                "blocked_symbols": blocked_symbols,
                "blocked_keywords": set(),  # disabled for forex
                "confidence_boost": 0.0,    # disabled — no threshold ratcheting
                "max_spread": max_spread,
                "min_liquidity": min_liquidity,
                "min_sources": min_sources,
                "category_counts": category_counts,
                "total_lessons": len(lessons),
                "diagnostic_count": diagnostic_count,
            }
            self._lessons_loaded_at = datetime.now(timezone.utc)

            applied = []
            if blocked_symbols:
                applied.append(f"blocked_symbols={blocked_symbols}")
            if max_spread is not None:
                applied.append(f"max_spread={max_spread:.1f}")
            if min_liquidity is not None:
                applied.append(f"min_liquidity={min_liquidity:.0f}")
            if min_sources is not None:
                applied.append(f"min_sources={min_sources}")

            logger.info(
                f"LESSONS: {len(lessons)} total ({diagnostic_count} diagnostic) | "
                f"categories: {category_counts} | "
                f"live rules: {', '.join(applied) if applied else 'none'}"
            )
        except Exception as e:
            logger.warning(f"Failed to load lessons: {e}")

    def _get_recent_win_rate(self) -> float:
        """Get win rate from last 20 trades for dynamic adjustments."""
        if not self.db:
            return 0.5
        try:
            trades = self.db.get_all_trades(limit=20) or []
            if len(trades) < 5:
                return 0.5  # Not enough data
            wins = sum(1 for t in trades if t.get("outcome") == "win")
            return wins / len(trades)
        except Exception:
            return 0.5

    # Currency codes that must NEVER be keyword-blocked (too broad for FX)
    _CURRENCY_CODES = {
        "usd", "eur", "gbp", "jpy", "aud", "nzd", "cad", "chf",
        "hkd", "sgd", "sek", "nok", "dkk", "zar", "try", "mxn",
        "cnh", "pln", "czk", "huf", "inr", "krw", "thb",
    }

    def scan_all_pairs(self, include_crypto: bool = True, max_pairs: int = None) -> list[dict]:
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

        # Explicit asset-class routing with clear logging
        forex_count = 0
        crypto_count = 0
        if forex_open:
            pairs = list(ALL_PAIRS if include_crypto else FOREX_PAIRS)
            forex_count = len(FOREX_PAIRS)
            crypto_count = len(CRYPTO_PAIRS) if include_crypto else 0
            logger.info(f"FOREX: open — scanning {forex_count} pairs | CRYPTO: {crypto_count} pairs")
        else:
            pairs = list(CRYPTO_PAIRS) if include_crypto else []
            crypto_count = len(pairs)
            logger.info(f"FOREX: closed (weekend) — skipped | CRYPTO: scanning {crypto_count} pairs")
            if not pairs:
                return []

        # Enforce max_pairs limit
        if max_pairs and len(pairs) > max_pairs:
            # Prioritize tier-1 pairs, then fill remaining slots
            tier1 = [p for p in pairs if p in TIER1_PAIRS]
            rest = [p for p in pairs if p not in TIER1_PAIRS]
            pairs = tier1 + rest[:max(0, max_pairs - len(tier1))]

        if not pairs:
            logger.info("No pairs to scan")
            return []

        # Load lessons every scan cycle so we learn in real-time
        self._load_lessons()

        # ── Fixed threshold — no more adaptive mutation ──
        # Threshold stays at base. Only risk sizing adapts mildly.
        # This prevents the feedback loop where losses raise threshold,
        # which reduces trades, which prevents recovery.
        effective_threshold = self.confidence_threshold  # fixed at 0.45
        win_rate = self._get_recent_win_rate()

        # ── Log cycle state for debugging ──
        logger.info(
            f"SCAN CYCLE: pairs={len(pairs)} | "
            f"base_threshold={self.confidence_threshold:.2f} | "
            f"effective_threshold={effective_threshold:.2f} | "
            f"win_rate={win_rate:.0%} | "
            f"lesson_boost={lesson_boost:.0%}"
        )

        signals = []
        skipped_reasons = {}

        blocked_symbols = self._lesson_rules.get("blocked_symbols", set())
        # Filter out currency codes from keyword blocks — they're too broad for FX
        raw_keywords = self._lesson_rules.get("blocked_keywords", set())
        blocked_keywords = {kw for kw in raw_keywords if kw not in self._CURRENCY_CODES}
        if raw_keywords - blocked_keywords:
            logger.info(f"LEARNING: Ignored currency keyword blocks: {raw_keywords - blocked_keywords}")

        max_spread_limit = self._lesson_rules.get("max_spread")
        min_sources = self._lesson_rules.get("min_sources")

        for symbol in pairs:
            if symbol in blocked_symbols:
                skipped_reasons[symbol] = "lesson_block:symbol"
                continue

            # Check keyword blocks (e.g. "fed" from narrative_trap lessons)
            # Currency codes (jpy, usd, etc.) are filtered out above
            sym_lower = symbol.lower()
            keyword_hit = next((kw for kw in blocked_keywords if kw in sym_lower), None)
            if keyword_hit:
                skipped_reasons[symbol] = f"lesson_block:keyword({keyword_hit})"
                continue

            # Check spread limit from lessons (spread_slippage rule)
            if max_spread_limit and self.broker:
                try:
                    quote = self.broker.get_quote(symbol)
                    if quote and quote.spread and quote.spread > max_spread_limit:
                        skipped_reasons[symbol] = f"lesson_block:spread({quote.spread:.1f}>{max_spread_limit:.1f})"
                        continue
                except Exception as e:
                    logger.debug(f"Could not check spread for {symbol}: {e}")

            try:
                signal = self._analyze_pair(symbol, effective_threshold=effective_threshold)
                if signal:
                    signals.append(signal)
                else:
                    skipped_reasons[symbol] = "scanner_threshold"
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                skipped_reasons[symbol] = f"execution_error:{str(e)[:30]}"
                continue

        # Log terminal reason for each skip (one line per category)
        if skipped_reasons:
            from collections import Counter
            reason_counts = Counter(r.split(":")[0] for r in skipped_reasons.values())
            logger.info(
                f"Pairs skipped ({len(skipped_reasons)}): "
                + ", ".join(f"{k}={v}" for k, v in reason_counts.items())
            )

        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        logger.info(f"Scan complete: {len(signals)} signals found (threshold={effective_threshold:.2f})")

        return signals

    def _analyze_pair(self, symbol: str, effective_threshold: float = None) -> Optional[dict]:
        """Analyze a single forex pair for trading signals.

        BAR-CLOSE DISCIPLINE: Only uses fully closed candles.
        The most recent candle on each timeframe is dropped because
        it may still be forming. Indicators are computed on closed bars only.
        """
        if effective_threshold is None:
            effective_threshold = self.confidence_threshold
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

        closes_h4 = np.array([c["close"] for c in candles_h4])
        highs_h4 = np.array([c["high"] for c in candles_h4])
        lows_h4 = np.array([c["low"] for c in candles_h4])

        # Get current quote for spread check
        quote = self.broker.get_quote(symbol)
        is_crypto = symbol in CRYPTO_PAIRS
        max_spread = 0.5 if is_crypto else 0.05  # crypto has wider spreads
        if not quote or quote.spread > max_spread:
            return None

        current_price = quote.mid

        # ─── HIGHER-TIMEFRAME TREND FILTER (GATE) ───

        # H4 trend: price vs SMA20 + SMA20 slope
        h4_sma_20 = np.mean(closes_h4[-20:])
        h4_sma_10 = np.mean(closes_h4[-10:])
        h4_trend = "up" if closes_h4[-1] > h4_sma_20 and h4_sma_10 > h4_sma_20 else \
                   "down" if closes_h4[-1] < h4_sma_20 and h4_sma_10 < h4_sma_20 else \
                   "flat"

        # D1 trend: price vs SMA20
        d1_trend = "flat"
        d1_sma_20 = 0.0
        if len(candles_d1) >= 20:
            closes_d1 = np.array([c["close"] for c in candles_d1])
            d1_sma_20 = float(np.mean(closes_d1[-20:]))
            d1_sma_10 = np.mean(closes_d1[-10:])
            d1_trend = "up" if closes_d1[-1] > d1_sma_20 and d1_sma_10 > d1_sma_20 else \
                       "down" if closes_d1[-1] < d1_sma_20 and d1_sma_10 < d1_sma_20 else \
                       "flat"

        # Determine allowed trade direction
        # Relaxed: allow trades when at least ONE higher timeframe is directional
        # Confidence scoring will penalize weak alignment later
        if h4_trend == "up" and d1_trend in ("up", "flat"):
            htf_bias = "buy"
        elif h4_trend == "down" and d1_trend in ("down", "flat"):
            htf_bias = "sell"
        elif d1_trend == "up" and h4_trend in ("flat", "up"):
            htf_bias = "buy"
        elif d1_trend == "down" and h4_trend in ("flat", "down"):
            htf_bias = "sell"
        elif h4_trend == "up" and d1_trend == "down":
            # Conflicting: H4 up but D1 down — skip (counter-trend)
            return None
        elif h4_trend == "down" and d1_trend == "up":
            # Conflicting: H4 down but D1 up — skip (counter-trend)
            return None
        elif h4_trend == "flat" and d1_trend == "flat":
            # Both flat — no directional bias
            return None
        else:
            return None

        # ─── H1 INDICATORS ───

        sma_20 = np.mean(closes_h1[-20:])
        sma_50 = np.mean(closes_h1[-50:])
        ema_12 = self._ema(closes_h1, 12)
        ema_26 = self._ema(closes_h1, 26)

        # RSI (14-period)
        rsi = self._rsi(closes_h1, 14)

        # MACD histogram
        macd_values = np.array([
            self._ema(closes_h1[:i+1], 12) - self._ema(closes_h1[:i+1], 26)
            for i in range(max(25, len(closes_h1)-10), len(closes_h1))
        ])
        signal_line = self._ema(macd_values, 9)
        macd_hist = macd_values[-1] - signal_line if len(macd_values) > 0 else 0.0
        macd_hist_prev = (macd_values[-2] - self._ema(macd_values[:-1], 9)) if len(macd_values) > 1 else 0.0

        # ATR
        atr = self._atr(highs_h1, lows_h1, closes_h1, 14)

        # H4 ADX (trend strength)
        adx_h4 = self._adx(highs_h4, lows_h4, closes_h4, 14)

        # Regime detection
        atr_20 = self._atr(highs_h1, lows_h1, closes_h1, 20)
        atr_5 = self._atr(highs_h1[-6:], lows_h1[-6:], closes_h1[-6:], 5) if len(closes_h1) >= 6 else atr
        regime = "trending" if adx_h4 >= 20 else "ranging"
        if atr_5 > atr_20 * 2.0:
            regime = "volatile"

        # Pip value from OANDA instrument spec (not hardcoded)
        spec = self.broker.get_instrument_spec(symbol)
        if spec.get("_fallback"):
            # Refuse to trade on guessed instrument specs
            logger.warning(f"SKIP {symbol}: using fallback spec — OANDA metadata unavailable")
            return None
        pip = spec["pip_size"]

        atr_pips = atr / pip
        spread_pips = (quote.ask - quote.bid) / pip if pip > 0 else 0

        # Hard filter: reject if spread too wide (> 5 pips forex, > 50 pips crypto)
        max_spread_pips = 50.0 if is_crypto else 5.0
        if spread_pips > max_spread_pips:
            return None

        # H4 swing levels for location scoring
        h4_recent_low = float(np.min(lows_h4[-10:]))
        h4_recent_high = float(np.max(highs_h4[-10:]))

        # ─── CONFIDENCE MODULE ───

        side = htf_bias

        inputs = ConfidenceInputs(
            direction=side,
            d1_trend=d1_trend,
            h4_trend=h4_trend,
            rsi_h1=rsi,
            macd_hist_h1=macd_hist,
            macd_hist_prev_h1=macd_hist_prev,
            sma20_h1=float(sma_20),
            sma50_h1=float(sma_50),
            close_h1=float(closes_h1[-1]),
            adx_h4=adx_h4,
            atr_pips_h1=atr_pips,
            spread_pips=spread_pips,
            regime=regime,
            h4_sma20=float(h4_sma_20),
            d1_sma20=float(d1_sma_20),
            h4_recent_low=h4_recent_low,
            h4_recent_high=h4_recent_high,
        )

        result = compute_confidence(inputs)

        if result.confidence < effective_threshold:
            return None

        confidence = result.confidence
        reasons = list(result.reasons)

        # ─── NEWS SENTIMENT ───

        try:
            # HIGH-IMPACT EVENTS = HARD BLOCK (not just penalty)
            has_impact, impacts = self.news.has_high_impact_event()
            if has_impact:
                # Check if this symbol is affected (USD pairs for Fed, EUR for ECB, etc.)
                impact_text = " ".join(impacts).lower()
                symbol_currencies = set()
                parts = symbol.replace("_", "/").split("/")
                for p in parts:
                    symbol_currencies.add(p.upper())

                # Map events to affected currencies
                affected = False
                if any(kw in impact_text for kw in ["fed", "fomc", "nfp", "powell"]):
                    if "USD" in symbol_currencies:
                        affected = True
                if any(kw in impact_text for kw in ["ecb", "lagarde"]):
                    if "EUR" in symbol_currencies:
                        affected = True
                if any(kw in impact_text for kw in ["boe", "bailey"]):
                    if "GBP" in symbol_currencies:
                        affected = True
                if any(kw in impact_text for kw in ["boj", "ueda"]):
                    if "JPY" in symbol_currencies:
                        affected = True
                if "cpi" in impact_text or "inflation" in impact_text:
                    if "USD" in symbol_currencies:
                        affected = True

                if affected:
                    logger.info(f"HARD BLOCK {symbol}: high-impact event — {impacts[0][:60]}")
                    reasons.append(f"BLOCKED: high-impact event")
                    return None

            # Soft news filter for non-event news
            news = self.news.get_sentiment(symbol)
            if news.headline_count > 0:
                if side == "buy" and news.sentiment == "bearish" and news.score < -0.3:
                    confidence *= 0.85
                    reasons.append(f"news: bearish ({news.score:+.2f})")
                elif side == "sell" and news.sentiment == "bullish" and news.score > 0.3:
                    confidence *= 0.85
                    reasons.append(f"news: bullish ({news.score:+.2f})")
                elif (side == "buy" and news.sentiment == "bullish") or \
                     (side == "sell" and news.sentiment == "bearish"):
                    confidence *= 1.05
                    confidence = min(confidence, 1.0)
                    reasons.append(f"news: aligned ({news.score:+.2f})")
        except Exception:
            pass

        # Recheck after news adjustment
        if confidence < effective_threshold:
            return None

        # ─── SL/TP (improved R:R) ───
        # 1.5 ATR stop / 2.5 ATR target = 1:1.67 R:R
        sl_distance = atr * 1.5
        tp_distance = atr * 2.5

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

        # Tier tag for logging
        tier = "T1" if symbol in TIER1_PAIRS else "T2"

        reasoning = (
            f"[{tier}] {'  |  '.join(reasons)}\n"
            f"RSI: {rsi:.0f} | ADX: {adx_h4:.0f} | Regime: {regime}\n"
            f"D1: {d1_trend} | H4: {h4_trend} | HTF bias: {htf_bias}\n"
            f"ATR: {atr_pips:.1f} pips | Spread: {spread_pips:.1f} pips\n"
            f"SL: {sl_pips:.0f} pips | TP: {tp_pips:.0f} pips | R:R 1:{tp_distance/sl_distance:.1f}"
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
            "units": 1000,
            "reasoning": reasoning,
            "rsi": rsi,
            "atr": atr,
            "h4_trend": h4_trend,
            "d1_trend": d1_trend,
            "htf_bias": htf_bias,
            "spread_pips": spread_pips,
            "atr_pips": atr_pips,
            "adx_h4": adx_h4,
            "regime": regime,
            "tier": tier,
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
            return 15.0

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
