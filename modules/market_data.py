"""
market_data.py — Tradier API wrapper for options trading bot.

Handles all communication with Tradier's REST API:
- Quotes (single and bulk)
- Options chains and expirations
- Historical OHLCV price data
- Account positions and balances
- Order placement, status, and cancellation

Set TRADIER_SANDBOX=true in .env to use paper trading sandbox.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class TradierClient:
    """Wrapper around the Tradier REST API."""

    SANDBOX_BASE = "https://sandbox.tradier.com/v1"
    LIVE_BASE = "https://api.tradier.com/v1"

    def __init__(self):
        self.api_key = os.getenv("TRADIER_API_KEY", "")
        self.account_id = os.getenv("TRADIER_ACCOUNT_ID", "")
        sandbox = os.getenv("TRADIER_SANDBOX", "true").lower()
        self.sandbox = sandbox in ("true", "1", "yes")
        self.base = self.SANDBOX_BASE if self.sandbox else self.LIVE_BASE
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request with basic error handling."""
        url = f"{self.base}{path}"
        try:
            r = self._session.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Tradier API timeout on {path}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Tradier API error {r.status_code}: {r.text[:200]}")

    def _post(self, path: str, data: dict) -> dict:
        """POST request with basic error handling."""
        url = f"{self.base}{path}"
        try:
            r = self._session.post(url, data=data, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Tradier API timeout on {path}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Tradier API error {r.status_code}: {r.text[:200]}")

    def _delete(self, path: str) -> dict:
        """DELETE request."""
        url = f"{self.base}{path}"
        try:
            r = self._session.delete(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Tradier API error {r.status_code}: {r.text[:200]}")

    def is_configured(self) -> bool:
        """Returns True if API key and account ID are set."""
        return bool(self.api_key and self.account_id)

    # ──────────────────────────────────────────────
    # QUOTES
    # ──────────────────────────────────────────────

    def get_quote(self, ticker: str) -> Optional[dict]:
        """Get a single stock quote. Returns None if not found."""
        results = self.get_quotes([ticker])
        return results[0] if results else None

    def get_quotes(self, tickers: list) -> list:
        """
        Get quotes for multiple tickers at once.
        Returns list of dicts with keys: symbol, last, bid, ask, volume,
        open, high, low, close, change, change_percentage, average_volume.
        """
        symbols = ",".join(tickers)
        data = self._get("/markets/quotes", {"symbols": symbols, "greeks": "false"})
        quotes_raw = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes_raw, dict):
            quotes_raw = [quotes_raw]
        results = []
        for q in quotes_raw:
            if not isinstance(q, dict):
                continue
            results.append({
                "symbol": q.get("symbol", ""),
                "last": float(q.get("last") or q.get("close") or 0),
                "bid": float(q.get("bid") or 0),
                "ask": float(q.get("ask") or 0),
                "volume": int(q.get("volume") or 0),
                "open": float(q.get("open") or 0),
                "high": float(q.get("high") or 0),
                "low": float(q.get("low") or 0),
                "close": float(q.get("close") or 0),
                "change": float(q.get("change") or 0),
                "change_pct": float(q.get("change_percentage") or 0),
                "average_volume": int(q.get("average_volume") or 0),
            })
        return results

    # ──────────────────────────────────────────────
    # OPTIONS
    # ──────────────────────────────────────────────

    def get_expirations(self, ticker: str) -> list:
        """
        Get available option expiration dates for a ticker.
        Returns list of date strings like ['2024-01-19', '2024-02-16', ...].
        """
        data = self._get("/markets/options/expirations", {
            "symbol": ticker,
            "includeAllRoots": "true",
            "strikes": "false",
        })
        dates = data.get("expirations", {}).get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return dates or []

    def get_options_chain(self, ticker: str, expiration: str = None) -> list:
        """
        Get options chain for a ticker and expiration date.
        If expiration is None, uses the nearest weekly/monthly expiry.
        Returns list of contract dicts with greeks and volume info.
        """
        if not expiration:
            exps = self.get_expirations(ticker)
            if not exps:
                return []
            expiration = exps[0]

        data = self._get("/markets/options/chains", {
            "symbol": ticker,
            "expiration": expiration,
            "greeks": "true",
        })
        options_raw = data.get("options", {}).get("option", [])
        if isinstance(options_raw, dict):
            options_raw = [options_raw]

        contracts = []
        for o in (options_raw or []):
            if not isinstance(o, dict):
                continue
            greeks = o.get("greeks") or {}
            contracts.append({
                "symbol": o.get("symbol", ""),
                "underlying": ticker,
                "expiration": o.get("expiration_date", expiration),
                "strike": float(o.get("strike") or 0),
                "option_type": o.get("option_type", "").lower(),  # 'call' or 'put'
                "bid": float(o.get("bid") or 0),
                "ask": float(o.get("ask") or 0),
                "last": float(o.get("last") or 0),
                "volume": int(o.get("volume") or 0),
                "open_interest": int(o.get("open_interest") or 0),
                "delta": float(greeks.get("delta") or 0),
                "gamma": float(greeks.get("gamma") or 0),
                "theta": float(greeks.get("theta") or 0),
                "vega": float(greeks.get("vega") or 0),
                "iv": float(greeks.get("mid_iv") or greeks.get("ask_iv") or 0),
            })
        return contracts

    def get_options_volume_summary(self, ticker: str, expiration: str = None) -> dict:
        """
        Aggregate call and put volume across entire chain for a given expiry.
        Returns: {total_call_volume, total_put_volume, put_call_ratio}
        """
        chain = self.get_options_chain(ticker, expiration)
        call_vol = sum(c["volume"] for c in chain if c["option_type"] == "call")
        put_vol = sum(c["volume"] for c in chain if c["option_type"] == "put")
        pcr = (put_vol / call_vol) if call_vol > 0 else 0
        return {
            "total_call_volume": call_vol,
            "total_put_volume": put_vol,
            "put_call_ratio": round(pcr, 3),
        }

    # ──────────────────────────────────────────────
    # HISTORICAL DATA
    # ──────────────────────────────────────────────

    def get_history(
        self,
        ticker: str,
        interval: str = "daily",
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data. Interval: 'daily', 'weekly', 'monthly'.
        Returns a DataFrame with columns: date, open, high, low, close, volume.
        Falls back to yfinance if Tradier returns empty data.
        """
        if not end:
            end = datetime.today().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")

        data = self._get("/markets/history", {
            "symbol": ticker,
            "interval": interval,
            "start": start,
            "end": end,
        })
        days = data.get("history", {})
        if days:
            raw = days.get("day", [])
            if isinstance(raw, dict):
                raw = [raw]
            if raw:
                df = pd.DataFrame(raw)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                return df

        # Fallback: yfinance
        return self._yfinance_history(ticker, start, end)

    def _yfinance_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fallback price history via yfinance."""
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})[["date", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    # ACCOUNT
    # ──────────────────────────────────────────────

    def get_positions(self) -> list:
        """
        Get current account positions.
        Returns list of dicts: {symbol, quantity, cost_basis, current_value}.
        """
        if not self.account_id:
            return []
        data = self._get(f"/accounts/{self.account_id}/positions")
        positions_raw = data.get("positions", {})
        if not positions_raw or positions_raw == "null":
            return []
        pos_list = positions_raw.get("position", [])
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        results = []
        for p in (pos_list or []):
            results.append({
                "symbol": p.get("symbol", ""),
                "quantity": int(p.get("quantity") or 0),
                "cost_basis": float(p.get("cost_basis") or 0),
                "date_acquired": p.get("date_acquired", ""),
            })
        return results

    def get_account_balance(self) -> dict:
        """
        Get account cash and buying power.
        Returns: {total_equity, cash, option_buying_power, day_trade_buying_power}.
        """
        if not self.account_id:
            return {}
        data = self._get(f"/accounts/{self.account_id}/balances")
        bal = data.get("balances", {})
        margin = bal.get("margin", {}) or {}
        cash_bal = bal.get("cash", {}) or {}
        return {
            "total_equity": float(bal.get("total_equity") or 0),
            "cash": float(bal.get("total_cash") or cash_bal.get("cash_available") or 0),
            "option_buying_power": float(
                margin.get("option_buying_power") or
                cash_bal.get("cash_available") or 0
            ),
            "day_trade_buying_power": float(margin.get("day_trade_buying_power") or 0),
        }

    # ──────────────────────────────────────────────
    # ORDERS
    # ──────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        option_symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: float = None,
        duration: str = "day",
    ) -> dict:
        """
        Place an option order.

        Args:
            ticker: Underlying symbol (e.g. 'AAPL')
            option_symbol: OCC option symbol (e.g. 'AAPL240119C00185000')
            side: 'buy_to_open' or 'sell_to_close'
            qty: Number of contracts
            order_type: 'market' or 'limit'
            limit_price: Required for limit orders
            duration: 'day' or 'gtc'

        Returns:
            dict with order id and status.
        """
        if not self.account_id:
            raise RuntimeError("TRADIER_ACCOUNT_ID not configured")

        payload = {
            "class": "option",
            "symbol": ticker,
            "option_symbol": option_symbol,
            "side": side,
            "quantity": str(qty),
            "type": order_type,
            "duration": duration,
        }
        if order_type == "limit" and limit_price:
            payload["price"] = str(round(limit_price, 2))

        data = self._post(f"/accounts/{self.account_id}/orders", payload)
        order = data.get("order", {})
        return {
            "order_id": order.get("id"),
            "status": order.get("status", "unknown"),
            "option_symbol": option_symbol,
            "side": side,
            "qty": qty,
            "type": order_type,
            "price": limit_price,
        }

    def get_order_status(self, order_id) -> dict:
        """Get status of an existing order."""
        if not self.account_id:
            return {}
        data = self._get(f"/accounts/{self.account_id}/orders/{order_id}")
        o = data.get("order", {})
        return {
            "order_id": o.get("id"),
            "status": o.get("status", ""),
            "filled_qty": int(o.get("exec_quantity") or 0),
            "avg_fill_price": float(o.get("avg_fill_price") or 0),
            "remaining_qty": int(o.get("remaining_quantity") or 0),
        }

    def cancel_order(self, order_id) -> dict:
        """Cancel a pending order."""
        if not self.account_id:
            return {}
        data = self._delete(f"/accounts/{self.account_id}/orders/{order_id}")
        return data.get("order", {})
