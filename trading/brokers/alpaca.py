"""
Alpaca broker integration — Stocks + Options trading.

Uses Alpaca Trading API v2 for paper and live trading.
Options support via Alpaca's options API.

Setup:
    1. Create free account at alpaca.markets
    2. Get API key + secret from dashboard
    3. Set env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
    4. Paper trading is default (no real money)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from trading.brokers.base import (
    BaseBroker, Quote, OptionQuote, OrderResult, Position,
)

logger = logging.getLogger(__name__)

# Large-cap stocks to scan for options (narrow universe per ChatGPT)
OPTIONS_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker for stocks and options.

    Paper trading by default. Set ALPACA_LIVE=true for real money.
    """

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        is_live = os.getenv("ALPACA_LIVE", "false").lower() == "true"

        if is_live:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"

        self.connected = False
        self._account = None

    @property
    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def connect(self) -> bool:
        """Connect to Alpaca and verify credentials."""
        if not self.api_key or not self.secret_key:
            logger.error("Alpaca API key/secret not set")
            return False

        try:
            resp = requests.get(
                f"{self.base_url}/v2/account",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                self._account = resp.json()
                self.connected = True
                status = self._account.get("status", "unknown")
                logger.info(f"Alpaca connected — Status: {status}")
                return True
            else:
                logger.error(f"Alpaca auth failed: {resp.status_code} {resp.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Alpaca connection error: {e}")
            return False

    def get_account_balance(self) -> float:
        """Get current portfolio value."""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/account",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                return float(data.get("portfolio_value", 0))
        except requests.RequestException as e:
            logger.error(f"Balance error: {e}")
        return 0.0

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a stock."""
        try:
            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                data = resp.json().get("quote", {})
                bid = float(data.get("bp", 0))
                ask = float(data.get("ap", 0))
                return Quote(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    spread=ask - bid,
                )
        except requests.RequestException as e:
            logger.debug(f"Quote error for {symbol}: {e}")
        return None

    def get_candles(self, symbol: str, timeframe: str = "1Hour", count: int = 100) -> list[dict]:
        """
        Get historical bars for a stock.

        Timeframe mapping: H1 -> 1Hour, H4 -> 4Hour, D -> 1Day
        """
        tf_map = {
            "H1": "1Hour", "H4": "4Hour", "D": "1Day",
            "M15": "15Min", "M5": "5Min",
            "1Hour": "1Hour", "4Hour": "4Hour", "1Day": "1Day",
            "15Min": "15Min", "5Min": "5Min",
        }
        tf = tf_map.get(timeframe, timeframe)

        try:
            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                headers=self._headers,
                params={"timeframe": tf, "limit": count, "feed": "iex"},
                timeout=10,
            )
            if resp.ok:
                bars = resp.json().get("bars", [])
                return [
                    {
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": int(b["v"]),
                        "time": b.get("t", ""),
                    }
                    for b in bars
                ]
        except requests.RequestException as e:
            logger.debug(f"Candles error for {symbol}: {e}")
        return []

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: str = None,
        option_type: str = None,
    ) -> list[OptionQuote]:
        """
        Get options chain for a stock with live quotes.

        Args:
            symbol: Underlying stock symbol (e.g. "SPY")
            expiration_date: Filter by expiry (YYYY-MM-DD)
            option_type: "call" or "put"
        """
        try:
            params = {"underlying_symbols": symbol, "limit": 100, "status": "active"}
            if expiration_date:
                params["expiration_date"] = expiration_date
            if option_type:
                params["type"] = option_type

            resp = requests.get(
                f"{self.base_url}/v2/options/contracts",
                headers=self._headers,
                params=params,
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"Options chain request failed for {symbol}: {resp.status_code} {resp.text[:200]}")
                return []

            contracts = resp.json().get("option_contracts", [])
            if not contracts:
                logger.info(f"No option contracts returned for {symbol}")
                return []

            # Collect contract symbols to fetch live quotes in batch
            contract_map = {}
            for c in contracts:
                sym = c.get("symbol", "")
                if sym:
                    contract_map[sym] = c

            # Fetch live quotes for all contracts (batch up to 100)
            options = []
            symbols_list = list(contract_map.keys())

            for batch_start in range(0, len(symbols_list), 50):
                batch = symbols_list[batch_start:batch_start + 50]
                quotes = self._get_option_quotes_batch(batch)

                for sym in batch:
                    c = contract_map[sym]
                    q = quotes.get(sym, {})
                    bid = float(q.get("bp", 0))
                    ask = float(q.get("ap", 0))

                    options.append(OptionQuote(
                        symbol=sym,
                        strike=float(c.get("strike_price", 0)),
                        expiration=c.get("expiration_date", ""),
                        option_type=c.get("type", ""),
                        open_interest=int(c.get("open_interest", 0)),
                        bid=bid,
                        ask=ask,
                    ))

            logger.info(f"Options chain for {symbol}: {len(options)} contracts with quotes")
            return options

        except requests.RequestException as e:
            logger.warning(f"Options chain error for {symbol}: {e}")
        return []

    def _get_option_quotes_batch(self, symbols: list[str]) -> dict:
        """Fetch live quotes for multiple option symbols at once."""
        if not symbols:
            return {}
        try:
            resp = requests.get(
                f"{self.data_url}/v1beta1/options/quotes/latest",
                headers=self._headers,
                params={"symbols": ",".join(symbols), "feed": "indicative"},
                timeout=15,
            )
            if resp.ok:
                return resp.json().get("quotes", {})
            else:
                logger.debug(f"Batch option quotes failed: {resp.status_code}")
        except requests.RequestException as e:
            logger.debug(f"Batch option quotes error: {e}")
        return {}

    def get_option_quote(self, option_symbol: str) -> Optional[OptionQuote]:
        """Get latest quote for a specific option contract."""
        try:
            resp = requests.get(
                f"{self.data_url}/v1beta1/options/quotes/latest",
                headers=self._headers,
                params={"symbols": option_symbol, "feed": "indicative"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json().get("quotes", {}).get(option_symbol, {})
                if data:
                    return OptionQuote(
                        symbol=option_symbol,
                        bid=float(data.get("bp", 0)),
                        ask=float(data.get("ap", 0)),
                    )
        except requests.RequestException as e:
            logger.debug(f"Option quote error: {e}")
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float = 0.0,
    ) -> OrderResult:
        """Place a stock or option order."""
        order_data = {
            "symbol": symbol,
            "qty": str(int(quantity)),
            "side": side,
            "type": order_type,
            "time_in_force": "day",
        }

        if order_type == "limit" and price > 0:
            order_data["limit_price"] = str(price)

        try:
            resp = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self._headers,
                json=order_data,
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                return OrderResult(
                    success=True,
                    order_id=data.get("id", ""),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=float(data.get("filled_avg_price", 0) or 0),
                    order_type=order_type,
                    status=data.get("status", ""),
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Order rejected: {resp.text[:200]}",
                )
        except requests.RequestException as e:
            return OrderResult(success=False, message=str(e))

    def place_option_order(
        self,
        option_symbol: str,
        side: str,
        quantity: int,
        order_type: str = "limit",
        limit_price: float = 0.0,
    ) -> OrderResult:
        """
        Place an options order.

        Args:
            option_symbol: Full option symbol (e.g. "SPY250411C00550000")
            side: "buy" or "sell"
            quantity: Number of contracts
            order_type: "market" or "limit" (limit recommended for options)
            limit_price: Limit price per contract
        """
        order_data = {
            "symbol": option_symbol,
            "qty": str(quantity),
            "side": side,
            "type": order_type,
            "time_in_force": "day",
        }

        if order_type == "limit" and limit_price > 0:
            order_data["limit_price"] = str(limit_price)

        try:
            resp = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self._headers,
                json=order_data,
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                return OrderResult(
                    success=True,
                    order_id=data.get("id", ""),
                    symbol=option_symbol,
                    side=side,
                    quantity=quantity,
                    price=float(data.get("filled_avg_price", 0) or 0),
                    order_type=order_type,
                    status=data.get("status", ""),
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Option order rejected: {resp.text[:200]}",
                )
        except requests.RequestException as e:
            return OrderResult(success=False, message=str(e))

    def get_positions(self) -> list[Position]:
        """Get all open positions (stocks + options)."""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                positions = []
                for p in resp.json():
                    qty = float(p.get("qty", 0))
                    entry = float(p.get("avg_entry_price", 0))
                    current = float(p.get("current_price", 0))
                    pnl = float(p.get("unrealized_pl", 0))
                    positions.append(Position(
                        symbol=p.get("symbol", ""),
                        side="long" if qty > 0 else "short",
                        quantity=abs(qty),
                        entry_price=entry,
                        current_price=current,
                        pnl=pnl,
                    ))
                return positions
        except requests.RequestException as e:
            logger.error(f"Positions error: {e}")
        return []

    def close_position(self, symbol: str) -> OrderResult:
        """Close a position by symbol."""
        try:
            resp = requests.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                return OrderResult(success=True, symbol=symbol, message="Position closed")
            else:
                return OrderResult(success=False, message=resp.text[:200])
        except requests.RequestException as e:
            return OrderResult(success=False, message=str(e))

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/clock",
                headers=self._headers,
                timeout=10,
            )
            if resp.ok:
                return resp.json().get("is_open", False)
        except requests.RequestException:
            pass
        return False
