"""
forex_data.py — Forex market data and order execution.

Supports two modes:
  1. OANDA API (recommended) — real forex broker with free demo accounts,
     full REST API for streaming quotes, history, and order execution.
  2. yfinance fallback — free read-only historical data via Yahoo Finance
     (no live trading, good for scanning and backtesting only).

Set MARKET_MODE=forex (or both) and FOREX_BROKER=oanda in .env.
For OANDA: set OANDA_API_KEY and OANDA_ACCOUNT_ID.

Common forex pairs supported:
  EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, USD_CHF,
  NZD_USD, EUR_GBP, EUR_JPY, GBP_JPY
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# Forex pairs available for trading/scanning
MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "USD_CHF", "NZD_USD",
]

MINOR_PAIRS = [
    "EUR_GBP", "EUR_JPY", "EUR_AUD", "GBP_JPY",
    "GBP_AUD", "AUD_JPY", "CAD_JPY",
]

ALL_PAIRS = MAJOR_PAIRS + MINOR_PAIRS

# yfinance ticker format for each pair
PAIR_TO_YF = {
    "EUR_USD": "EURUSD=X", "GBP_USD": "GBPUSD=X", "USD_JPY": "USDJPY=X",
    "AUD_USD": "AUDUSD=X", "USD_CAD": "USDCAD=X", "USD_CHF": "USDCHF=X",
    "NZD_USD": "NZDUSD=X", "EUR_GBP": "EURGBP=X", "EUR_JPY": "EURJPY=X",
    "GBP_JPY": "GBPJPY=X", "EUR_AUD": "EURAUD=X", "GBP_AUD": "GBPAUD=X",
    "AUD_JPY": "AUDJPY=X", "CAD_JPY": "CADJPY=X",
}


class OandaClient:
    """
    OANDA v20 REST API wrapper.
    Free demo account: practice.oanda.com
    Live account: trade.oanda.com
    """

    PRACTICE_BASE = "https://api-fxpractice.oanda.com/v3"
    LIVE_BASE = "https://api-fxtrade.oanda.com/v3"

    def __init__(self):
        self.api_key = os.getenv("OANDA_API_KEY", "")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID", "")
        practice = os.getenv("OANDA_PRACTICE", "true").lower()
        self.practice = practice in ("true", "1", "yes")
        self.base = self.PRACTICE_BASE if self.practice else self.LIVE_BASE
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def is_configured(self) -> bool:
        return bool(self.api_key and self.account_id)

    def _get(self, path: str, params: dict = None) -> dict:
        r = self._session.get(f"{self.base}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        import json as _json
        r = self._session.post(f"{self.base}{path}", data=_json.dumps(payload), timeout=10)
        r.raise_for_status()
        return r.json()

    # ──────────────────────────────────────────────
    # QUOTES
    # ──────────────────────────────────────────────

    def get_quote(self, pair: str) -> Optional[dict]:
        """
        Get current bid/ask for a forex pair.
        pair format: 'EUR_USD'
        Returns: {pair, bid, ask, spread, spread_pips, mid}
        """
        try:
            data = self._get(f"/instruments/{pair}/candles", {
                "count": 1,
                "granularity": "S5",
                "price": "BA",
            })
            candles = data.get("candles", [])
            if not candles:
                return None
            c = candles[-1]
            bid = float(c.get("bid", {}).get("c", 0))
            ask = float(c.get("ask", {}).get("c", 0))
            spread = round(ask - bid, 5)
            pip_size = 0.01 if "JPY" in pair else 0.0001
            spread_pips = round(spread / pip_size, 1)
            mid = round((bid + ask) / 2, 5)
            return {
                "pair": pair,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_pips": spread_pips,
            }
        except Exception:
            return None

    def get_quotes(self, pairs: list) -> list:
        """Get quotes for multiple pairs."""
        results = []
        for pair in pairs:
            q = self.get_quote(pair)
            if q:
                results.append(q)
        return results

    # ──────────────────────────────────────────────
    # HISTORICAL DATA
    # ──────────────────────────────────────────────

    def get_history(
        self,
        pair: str,
        granularity: str = "D",
        count: int = 120,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC candles.
        Granularity: M1, M5, M15, M30, H1, H4, D, W, M
        Returns DataFrame with: date, open, high, low, close, volume
        """
        params = {
            "granularity": granularity,
            "price": "M",  # midpoint candles
        }
        if start and end:
            params["from"] = start + "T00:00:00Z"
            params["to"] = end + "T23:59:59Z"
        else:
            params["count"] = str(count)

        try:
            data = self._get(f"/instruments/{pair}/candles", params)
            candles = data.get("candles", [])
            rows = []
            for c in candles:
                m = c.get("mid", {})
                rows.append({
                    "date": pd.to_datetime(c.get("time", "")[:10]),
                    "open": float(m.get("o", 0)),
                    "high": float(m.get("h", 0)),
                    "low": float(m.get("l", 0)),
                    "close": float(m.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            df = pd.DataFrame(rows)
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return _yfinance_forex_history(pair, count)

    # ──────────────────────────────────────────────
    # ACCOUNT
    # ──────────────────────────────────────────────

    def get_account_balance(self) -> dict:
        """Get account NAV, balance, unrealized P&L."""
        try:
            data = self._get(f"/accounts/{self.account_id}/summary")
            acc = data.get("account", {})
            return {
                "balance": float(acc.get("balance") or 0),
                "nav": float(acc.get("NAV") or 0),
                "unrealized_pl": float(acc.get("unrealizedPL") or 0),
                "margin_used": float(acc.get("marginUsed") or 0),
                "margin_available": float(acc.get("marginAvailable") or 0),
            }
        except Exception:
            return {}

    def get_positions(self) -> list:
        """Get open forex positions."""
        try:
            data = self._get(f"/accounts/{self.account_id}/openPositions")
            positions = data.get("positions", [])
            results = []
            for p in positions:
                instrument = p.get("instrument", "")
                long = p.get("long", {})
                short = p.get("short", {})
                units = int(long.get("units", 0) or 0) + int(short.get("units", 0) or 0)
                unrealized = float(p.get("unrealizedPL", 0) or 0)
                results.append({
                    "pair": instrument,
                    "units": units,
                    "unrealized_pl": unrealized,
                    "avg_price": float(long.get("averagePrice") or short.get("averagePrice") or 0),
                    "direction": "long" if int(long.get("units", 0) or 0) > 0 else "short",
                })
            return results
        except Exception:
            return []

    # ──────────────────────────────────────────────
    # ORDERS
    # ──────────────────────────────────────────────

    def place_market_order(
        self,
        pair: str,
        units: int,
        stop_loss_pips: float = 30,
        take_profit_pips: float = 60,
    ) -> dict:
        """
        Place a market order.
        Positive units = buy (long). Negative units = sell (short).
        Stop loss and take profit are in pips.

        Returns: {order_id, status, pair, units, fill_price}
        """
        if not self.account_id:
            raise RuntimeError("OANDA_ACCOUNT_ID not configured")

        quote = self.get_quote(pair)
        pip_size = 0.01 if "JPY" in pair else 0.0001
        sl_distance = stop_loss_pips * pip_size
        tp_distance = take_profit_pips * pip_size

        price = quote["ask"] if units > 0 else quote["bid"] if quote else 0
        sl_price = round(price - sl_distance if units > 0 else price + sl_distance, 5)
        tp_price = round(price + tp_distance if units > 0 else price - tp_distance, 5)

        payload = {
            "order": {
                "type": "MARKET",
                "instrument": pair,
                "units": str(units),
                "timeInForce": "FOK",
                "stopLossOnFill": {"price": str(sl_price)},
                "takeProfitOnFill": {"price": str(tp_price)},
            }
        }
        data = self._post(f"/accounts/{self.account_id}/orders", payload)
        fill = data.get("orderFillTransaction", {})
        return {
            "order_id": fill.get("id"),
            "status": "filled" if fill else "pending",
            "pair": pair,
            "units": units,
            "fill_price": float(fill.get("price", 0) or 0),
            "direction": "long" if units > 0 else "short",
        }

    def close_position(self, pair: str) -> dict:
        """Close all units of an open position."""
        try:
            data = self._post(f"/accounts/{self.account_id}/positions/{pair}/close", {
                "longUnits": "ALL",
                "shortUnits": "ALL",
            })
            return {"status": "closed", "pair": pair}
        except Exception as e:
            return {"status": "error", "pair": pair, "error": str(e)}


# ──────────────────────────────────────────────
# YFINANCE FALLBACK (read-only, no trading)
# ──────────────────────────────────────────────

def _yfinance_forex_history(pair: str, count: int = 120) -> pd.DataFrame:
    """Fallback: fetch forex history from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
        yf_ticker = PAIR_TO_YF.get(pair, pair.replace("_", "") + "=X")
        end = datetime.today()
        start = end - timedelta(days=count * 2)
        df = yf.download(
            yf_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        if "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "close"})
        df = df[["date", "open", "high", "low", "close", "volume"]].tail(count)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_forex_history_yf(pair: str, count: int = 120) -> pd.DataFrame:
    """Public entry point for yfinance forex data (no broker needed)."""
    return _yfinance_forex_history(pair, count)


# ──────────────────────────────────────────────
# MARKET MODE HELPER
# ──────────────────────────────────────────────

def get_active_markets() -> list:
    """
    Returns which markets are currently enabled based on MARKET_MODE env var.
    MARKET_MODE can be: 'options', 'forex', or 'both'
    """
    mode = os.getenv("MARKET_MODE", "options").lower()
    if mode == "options":
        return ["options"]
    elif mode == "forex":
        return ["forex"]
    elif mode == "both":
        return ["options", "forex"]
    return ["options"]


def is_forex_enabled() -> bool:
    return "forex" in get_active_markets()


def is_options_enabled() -> bool:
    return "options" in get_active_markets()


def get_default_forex_watchlist() -> list:
    """Returns forex watchlist from env or defaults to major pairs."""
    wl = os.getenv("FOREX_WATCHLIST", "")
    if wl:
        return [p.strip().upper().replace("/", "_") for p in wl.split(",")]
    return MAJOR_PAIRS


def pip_value(pair: str, price: float = 1.0, lot_size: int = 1000) -> float:
    """Estimate pip value in USD for a given pair and lot size."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    if pair.endswith("_USD"):
        return pip_size * lot_size
    elif pair.startswith("USD_"):
        return (pip_size / price) * lot_size
    else:
        # Cross pair: approximate
        return pip_size * lot_size * 1.0
