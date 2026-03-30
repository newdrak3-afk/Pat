"""
OANDA broker integration — Forex trading.

Uses OANDA REST API v20 directly (no extra library needed).
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

from trading.brokers.base import (
    BaseBroker, Quote, OrderResult, Position,
)

logger = logging.getLogger(__name__)

# 28 standard forex pairs — majors, minors, crosses only
# No exotics (HKD, MXN, ZAR, TRY, SEK, SGD) — wide spreads, poor fills
FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CHF_JPY",
]

# Crypto pairs — trade 24/7 including weekends
CRYPTO_PAIRS = [
    "BTC_USD", "ETH_USD", "LTC_USD", "BCH_USD",
    "XRP_USD", "BTC_EUR", "ETH_EUR", "BTC_GBP",
    "BTC_JPY", "ETH_GBP",
]

# All tradeable pairs
ALL_PAIRS = FOREX_PAIRS + CRYPTO_PAIRS


class OandaBroker(BaseBroker):
    """
    OANDA Forex broker integration.

    Supports:
    - Live and practice accounts
    - Forex pair quotes and scanning
    - Market and limit orders
    - Position management
    - Candlestick data for analysis
    """

    def __init__(self):
        self.api_key = os.getenv("OANDA_API_KEY", "")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID", "")
        self.practice = os.getenv("OANDA_PRACTICE", "true").lower() == "true"

        if self.practice:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"

        self.connected = False

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to OANDA API."""
        try:
            resp = requests.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"OANDA API error: {e}")
            return None

    def _post(self, path: str, data: dict) -> Optional[dict]:
        """Make a POST request to OANDA API."""
        try:
            resp = requests.post(
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=data,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"OANDA API error: {e}")
            return None

    def connect(self) -> bool:
        """Test connection to OANDA."""
        if not self.api_key or not self.account_id:
            logger.error(
                "Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env"
            )
            return False

        result = self._get(f"/accounts/{self.account_id}")
        if result and "account" in result:
            self.connected = True
            balance = result["account"].get("balance", "?")
            logger.info(
                f"Connected to OANDA "
                f"({'Practice' if self.practice else 'Live'}) — "
                f"Balance: {balance}"
            )
            return True

        logger.error("Failed to connect to OANDA")
        return False

    def get_account_balance(self) -> float:
        """Get account balance."""
        if not self.connected:
            return 0.0

        result = self._get(f"/accounts/{self.account_id}/summary")
        if result and "account" in result:
            return float(result["account"].get("balance", 0))
        return 0.0

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get a forex pair quote."""
        result = self._get(
            f"/accounts/{self.account_id}/pricing",
            params={"instruments": symbol},
        )

        if not result or "prices" not in result or not result["prices"]:
            return None

        price = result["prices"][0]
        bids = price.get("bids", [])
        asks = price.get("asks", [])

        bid = float(bids[0]["price"]) if bids else 0
        ask = float(asks[0]["price"]) if asks else 0
        mid = (bid + ask) / 2 if bid and ask else 0

        spread = 0.0
        if ask > 0:
            spread = ((ask - bid) / ask) * 100

        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=mid,
            spread=spread,
        )

    def get_quotes_bulk(self, symbols: list[str]) -> list[Quote]:
        """Get quotes for multiple pairs at once."""
        instruments = ",".join(symbols)
        result = self._get(
            f"/accounts/{self.account_id}/pricing",
            params={"instruments": instruments},
        )

        quotes = []
        if not result or "prices" not in result:
            return quotes

        for price in result["prices"]:
            symbol = price.get("instrument", "")
            bids = price.get("bids", [])
            asks = price.get("asks", [])

            bid = float(bids[0]["price"]) if bids else 0
            ask = float(asks[0]["price"]) if asks else 0
            mid = (bid + ask) / 2 if bid and ask else 0

            spread = 0.0
            if ask > 0:
                spread = ((ask - bid) / ask) * 100

            quotes.append(Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=mid,
                spread=spread,
            ))

        return quotes

    def get_candles(
        self,
        symbol: str,
        granularity: str = "H1",
        count: int = 100,
    ) -> list[dict]:
        """
        Get candlestick data for analysis.

        Granularity: S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30,
                     H1, H2, H3, H4, H6, H8, H12, D, W, M
        """
        result = self._get(
            f"/instruments/{symbol}/candles",
            params={
                "granularity": granularity,
                "count": count,
                "price": "MBA",  # mid, bid, ask
            },
        )

        if not result or "candles" not in result:
            return []

        candles = []
        for c in result["candles"]:
            if not c.get("complete", False):
                continue

            mid = c.get("mid", {})
            candles.append({
                "time": c.get("time", ""),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        return candles

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float = 0.0,
    ) -> OrderResult:
        """
        Place a forex order.

        Args:
            symbol: Forex pair (e.g. "EUR_USD")
            side: "buy" or "sell"
            quantity: Number of units (e.g. 1000 = micro lot)
            order_type: "market" or "limit"
            price: Limit price (only for limit orders)
        """
        if not self.connected:
            return OrderResult(
                success=False, message="Not connected to OANDA"
            )

        # Negative units = sell
        units = str(int(quantity)) if side == "buy" else str(-int(quantity))

        order_data = {
            "order": {
                "instrument": symbol,
                "units": units,
                "type": "MARKET" if order_type == "market" else "LIMIT",
                "timeInForce": "FOK" if order_type == "market" else "GTC",
            }
        }

        if order_type == "limit" and price > 0:
            order_data["order"]["price"] = str(price)

        result = self._post(
            f"/accounts/{self.account_id}/orders", order_data
        )

        if not result:
            return OrderResult(success=False, message="No response from OANDA")

        # Check for fill
        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            return OrderResult(
                success=True,
                order_id=fill.get("id", ""),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(fill.get("price", 0)),
                order_type=order_type,
                status="filled",
            )

        # Check for pending
        if "orderCreateTransaction" in result:
            create = result["orderCreateTransaction"]
            return OrderResult(
                success=True,
                order_id=create.get("id", ""),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
                status="pending",
            )

        # Check for rejection
        if "orderRejectTransaction" in result:
            reject = result["orderRejectTransaction"]
            return OrderResult(
                success=False,
                message=reject.get("rejectReason", "Order rejected"),
            )

        return OrderResult(
            success=False, message=f"Unexpected response: {result}"
        )

    def place_order_with_stops(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss_pips: float = 20,
        take_profit_pips: float = 40,
    ) -> OrderResult:
        """
        Place a forex order with stop loss and take profit.

        Args:
            symbol: Forex pair
            side: "buy" or "sell"
            quantity: Units
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
        """
        if not self.connected:
            return OrderResult(
                success=False, message="Not connected to OANDA"
            )

        # Get current price
        quote = self.get_quote(symbol)
        if not quote:
            return OrderResult(
                success=False, message=f"Could not get quote for {symbol}"
            )

        # Calculate pip value (most pairs = 0.0001, JPY pairs = 0.01)
        if "JPY" in symbol:
            pip = 0.01
        else:
            pip = 0.0001

        entry = quote.ask if side == "buy" else quote.bid

        if side == "buy":
            sl = entry - (stop_loss_pips * pip)
            tp = entry + (take_profit_pips * pip)
        else:
            sl = entry + (stop_loss_pips * pip)
            tp = entry - (take_profit_pips * pip)

        units = str(int(quantity)) if side == "buy" else str(-int(quantity))

        order_data = {
            "order": {
                "instrument": symbol,
                "units": units,
                "type": "MARKET",
                "timeInForce": "FOK",
                "stopLossOnFill": {
                    "price": f"{sl:.5f}",
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": f"{tp:.5f}",
                    "timeInForce": "GTC",
                },
            }
        }

        result = self._post(
            f"/accounts/{self.account_id}/orders", order_data
        )

        if not result:
            return OrderResult(success=False, message="No response")

        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            return OrderResult(
                success=True,
                order_id=fill.get("id", ""),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(fill.get("price", 0)),
                order_type="market",
                status="filled",
                message=f"SL: {sl:.5f} | TP: {tp:.5f}",
            )

        return OrderResult(
            success=False,
            message=result.get("orderRejectTransaction", {}).get(
                "rejectReason", str(result)
            ),
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self.connected:
            return []

        result = self._get(f"/accounts/{self.account_id}/openPositions")
        if not result or "positions" not in result:
            return []

        positions = []
        for pos in result["positions"]:
            symbol = pos.get("instrument", "")
            long_units = float(pos.get("long", {}).get("units", 0))
            short_units = float(pos.get("short", {}).get("units", 0))

            if long_units > 0:
                avg_price = float(
                    pos.get("long", {}).get("averagePrice", 0)
                )
                pnl = float(pos.get("long", {}).get("unrealizedPL", 0))
                positions.append(Position(
                    symbol=symbol,
                    side="long",
                    quantity=long_units,
                    entry_price=avg_price,
                    pnl=pnl,
                ))

            if short_units < 0:
                avg_price = float(
                    pos.get("short", {}).get("averagePrice", 0)
                )
                pnl = float(pos.get("short", {}).get("unrealizedPL", 0))
                positions.append(Position(
                    symbol=symbol,
                    side="short",
                    quantity=abs(short_units),
                    entry_price=avg_price,
                    pnl=pnl,
                ))

        return positions

    def close_position(self, symbol: str) -> OrderResult:
        """Close all units of a position."""
        if not self.connected:
            return OrderResult(
                success=False, message="Not connected to OANDA"
            )

        try:
            resp = requests.put(
                f"{self.base_url}/accounts/{self.account_id}"
                f"/positions/{symbol}/close",
                headers=self._headers(),
                json={"longUnits": "ALL", "shortUnits": "ALL"},
                timeout=10,
            )

            if resp.ok:
                return OrderResult(
                    success=True,
                    symbol=symbol,
                    status="closed",
                    message="Position closed",
                )
            else:
                return OrderResult(
                    success=False,
                    message=resp.text,
                )

        except requests.RequestException as e:
            return OrderResult(success=False, message=str(e))

    def get_closed_trade_pnl(self, trade_id: str) -> Optional[float]:
        """Get the actual realized PnL for a closed trade from OANDA.

        Queries recent transactions to find the close transaction for this trade.
        """
        try:
            result = self._get(
                f"/accounts/{self.account_id}/transactions",
                params={"type": "ORDER_FILL", "count": 50},
            )
            if result and "transactions" in result:
                for txn in result["transactions"]:
                    # Look for fills that reference our trade
                    if txn.get("tradesClosed"):
                        for closed in txn["tradesClosed"]:
                            if closed.get("tradeID") == trade_id:
                                return float(closed.get("realizedPL", 0))
                    if txn.get("tradeReduced", {}).get("tradeID") == trade_id:
                        return float(txn["tradeReduced"].get("realizedPL", 0))
        except Exception as e:
            logger.debug(f"Could not fetch PnL for trade {trade_id}: {e}")
        return None

    def get_spread_for_pairs(self) -> list[dict]:
        """Get current spreads for all major pairs — useful for scanning."""
        quotes = self.get_quotes_bulk(FOREX_PAIRS)
        spreads = []

        for q in quotes:
            if "JPY" in q.symbol:
                pip = 0.01
            else:
                pip = 0.0001

            spread_pips = (q.ask - q.bid) / pip if pip > 0 else 0

            spreads.append({
                "symbol": q.symbol,
                "bid": q.bid,
                "ask": q.ask,
                "spread_pips": round(spread_pips, 1),
                "spread_pct": q.spread,
            })

        return sorted(spreads, key=lambda x: x["spread_pips"])
