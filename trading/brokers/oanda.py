"""
OANDA broker integration — Forex trading.

Uses OANDA REST API v20 directly (no extra library needed).
Instrument specs (pip size, min/max units, margin) fetched from OANDA metadata.
"""

import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
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


def normalize_units(spec: dict, units: float, existing_position_units: float = 0) -> float:
    """Validate and normalize units to OANDA instrument constraints.

    Enforces minimumTradeSize, maximumOrderUnits, maximumPositionSize,
    and tradeUnitsPrecision.
    Returns 0 if units are below minimum (caller should skip the trade).
    """
    min_size = spec["min_trade_size"]
    max_units = spec["max_order_units"]
    max_position = spec.get("max_position_size", 0)
    precision = spec["units_precision"]

    abs_units = min(abs(units), max_units)

    # Enforce max position size (different from max order size)
    if max_position > 0:
        remaining_capacity = max(0, max_position - abs(existing_position_units))
        abs_units = min(abs_units, remaining_capacity)

    if abs_units < min_size:
        return 0

    # Round to instrument's trade unit precision
    if precision == 0:
        abs_units = int(abs_units)
    else:
        step = 10 ** (-precision)
        abs_units = int(abs_units / step) * step

    # Post-rounding check: rounding may have dropped below minimum
    if abs_units < min_size:
        return 0

    return abs_units if units >= 0 else -abs_units


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    units: float = 0
    reason: Optional[str] = None  # None = success, string = why it failed


def calc_units_from_risk(
    broker: "OandaBroker",
    symbol: str,
    balance: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    side: str,
) -> SizingResult:
    """Calculate position size from risk in USD account currency.

    NOTE: This assumes a USD-denominated account. For non-USD accounts,
    the quote-currency conversion would need to target the actual
    account currency instead of hardcoded USD.

    Returns SizingResult with units and a reason if sizing failed.
    """
    spec = broker.get_instrument_spec(symbol)
    sl_distance = abs(entry - stop_loss)
    if sl_distance <= 0:
        return SizingResult(0, "zero_stop_distance")

    # Convert SL distance to account currency (USD)
    quote_ccy = symbol.split("_")[1] if "_" in symbol else "USD"
    if quote_ccy == "USD":
        loss_per_unit = sl_distance
    else:
        fx_to_usd = broker.get_conversion_rate(quote_ccy, "USD")
        if fx_to_usd <= 0:
            logger.warning(
                f"SIZING SKIP {symbol}: no {quote_ccy}→USD rate available"
            )
            return SizingResult(0, f"conversion_rate_missing ({quote_ccy}→USD)")
        loss_per_unit = sl_distance * fx_to_usd

    if loss_per_unit <= 0:
        return SizingResult(0, "zero_loss_per_unit")

    risk_amount = balance * risk_pct
    raw_units = risk_amount / loss_per_unit

    # Hard cap: limit position value to 1x account balance in USD.
    # For cross-currency pairs (JPY, CHF), tiny loss_per_unit can produce
    # enormous unit counts that create outsized exposure.
    if quote_ccy == "USD":
        position_value_usd = raw_units * entry
    else:
        position_value_usd = raw_units * entry * fx_to_usd
    max_position_value = balance * 1.0  # max 1x leverage
    if position_value_usd > max_position_value:
        old_units = raw_units
        if quote_ccy == "USD":
            raw_units = max_position_value / entry
        else:
            raw_units = max_position_value / (entry * fx_to_usd)
        logger.warning(
            f"SIZING CAP {symbol}: position ${position_value_usd:.0f} > "
            f"balance ${balance:.0f}, capped {old_units:.0f} → {raw_units:.0f} units"
        )

    # Normalize to OANDA limits
    abs_units = normalize_units(spec, raw_units)
    if abs_units == 0:
        return SizingResult(0, f"below_min_trade_size (min={spec['min_trade_size']})")

    signed = abs_units if side.lower() == "buy" else -abs_units
    return SizingResult(signed, None)


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
        self._instrument_cache: dict[str, dict] = {}
        self._account_state: dict = {}  # cached account snapshot

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
            # Warm instrument cache for all tradable pairs
            self._warm_instrument_cache()
            return True

        logger.error("Failed to connect to OANDA")
        return False

    def _warm_instrument_cache(self):
        """Pre-fetch instrument specs for all tradable pairs at startup."""
        all_pairs = FOREX_PAIRS + CRYPTO_PAIRS
        cached = 0
        for symbol in all_pairs:
            try:
                spec = self.get_instrument_spec(symbol)
                if not spec.get("_fallback"):
                    cached += 1
            except Exception:
                pass
        logger.info(f"Instrument cache warmed: {cached}/{len(all_pairs)} specs loaded")

    # ── Instrument metadata (Recommendation #1) ──

    def get_instrument_spec(self, symbol: str) -> dict:
        """Fetch instrument spec from OANDA and cache it.

        Returns pip_size, display_precision, min/max units, margin rate, etc.
        Falls back to hardcoded defaults if the API call fails.
        """
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        spec = self._fetch_instrument_spec(symbol)
        self._instrument_cache[symbol] = spec
        return spec

    def _fetch_instrument_spec(self, symbol: str) -> dict:
        """Call OANDA instruments endpoint for one symbol."""
        try:
            resp = self._get(
                f"/accounts/{self.account_id}/instruments",
                params={"instruments": symbol},
            )
            instruments = (resp or {}).get("instruments", [])
            if instruments:
                inst = instruments[0]
                pip_location = int(inst.get("pipLocation", -4))
                spec = {
                    "name": inst["name"],
                    "pip_location": pip_location,
                    "pip_size": float(Decimal("10") ** pip_location),
                    "display_precision": int(inst.get("displayPrecision", 5)),
                    "units_precision": int(inst.get("tradeUnitsPrecision", 0)),
                    "min_trade_size": float(inst.get("minimumTradeSize", 1)),
                    "max_order_units": float(inst.get("maximumOrderUnits", 100000000)),
                    "max_position_size": float(inst.get("maximumPositionSize", 0)),
                    "margin_rate": float(inst.get("marginRate", 0.02)),
                    "type": inst.get("type", "CURRENCY"),
                }
                logger.debug(
                    f"Instrument spec {symbol}: pip={spec['pip_size']} "
                    f"precision={spec['units_precision']} "
                    f"min={spec['min_trade_size']} max={spec['max_order_units']}"
                )
                return spec
        except Exception as e:
            logger.warning(f"Could not fetch instrument spec for {symbol}: {e}")

        # Fallback for when API is unavailable — NOT authoritative.
        # If we're using this, instrument spec fetch failed. Log a warning.
        is_jpy = "JPY" in symbol
        is_crypto = symbol in CRYPTO_PAIRS
        logger.warning(
            f"Using FALLBACK instrument spec for {symbol} — "
            f"OANDA API unavailable. Sizing may be approximate."
        )
        return {
            "name": symbol,
            "pip_location": -2 if is_jpy else (-1 if is_crypto else -4),
            "pip_size": 0.01 if is_jpy else (1.0 if is_crypto else 0.0001),
            "display_precision": 3 if is_jpy else (1 if is_crypto else 5),
            "units_precision": 0,
            "min_trade_size": 1.0,
            "max_order_units": 5.0 if is_crypto else 100000000.0,
            "max_position_size": 0,
            "margin_rate": 0.02,
            "type": "CURRENCY",
            "_fallback": True,  # flag so callers can detect degraded mode
        }

    def get_conversion_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Get conversion rate between two currencies.

        Used to convert non-USD quote currency risk to account currency (USD).
        Returns 0.0 if conversion fails — caller MUST skip the trade.
        Never guesses a rate; wrong sizing is worse than no trade.
        """
        if from_ccy == to_ccy:
            return 1.0

        # Try direct pair
        pair = f"{from_ccy}_{to_ccy}"
        quote = self.get_quote(pair)
        if quote and quote.mid > 0:
            return quote.mid

        # Try inverse pair
        pair_inv = f"{to_ccy}_{from_ccy}"
        quote = self.get_quote(pair_inv)
        if quote and quote.mid > 0:
            return 1.0 / quote.mid

        logger.error(
            f"CONVERSION FAILED: {from_ccy}→{to_ccy} — "
            f"no quote available. Trade MUST be skipped."
        )
        return 0.0

    def get_account_snapshot(self) -> dict:
        """Get full account details for consistent guard state.

        Returns balance, NAV, margin available, open trade count, etc.
        Caches result in self._account_state.
        """
        result = self._get(f"/accounts/{self.account_id}/summary")
        if result and "account" in result:
            acct = result["account"]
            self._account_state = {
                "balance": float(acct.get("balance", 0)),
                "nav": float(acct.get("NAV", 0)),
                "unrealized_pnl": float(acct.get("unrealizedPL", 0)),
                "margin_used": float(acct.get("marginUsed", 0)),
                "margin_available": float(acct.get("marginAvailable", 0)),
                "open_trade_count": int(acct.get("openTradeCount", 0)),
                "open_position_count": int(acct.get("openPositionCount", 0)),
                "last_transaction_id": acct.get("lastTransactionID", ""),
            }
        return self._account_state

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

        # Normalize units using instrument spec (same logic as place_order_with_stops)
        spec = self.get_instrument_spec(symbol)
        abs_qty = normalize_units(spec, abs(quantity))
        if abs_qty == 0:
            return OrderResult(
                success=False,
                message=f"units_below_min_trade_size ({spec['min_trade_size']})",
            )
        signed = abs_qty if side == "buy" else -abs_qty
        if spec["units_precision"] == 0:
            units = str(int(signed))
        else:
            units = f"{signed:.{spec['units_precision']}f}"

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

        Uses instrument spec for pip size and price precision.
        Validates units against OANDA min/max before sending.
        Logs full rejection details on failure.
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

        # Use instrument spec for pip size and precision (not hardcoded)
        spec = self.get_instrument_spec(symbol)
        pip = spec["pip_size"]
        price_precision = spec["display_precision"]

        # Validate units against OANDA limits
        abs_qty = abs(quantity)
        abs_qty = normalize_units(spec, abs_qty)
        if abs_qty == 0:
            logger.warning(
                f"ORDER SKIP {symbol}: units {quantity} below min_trade_size "
                f"{spec['min_trade_size']} or zero after normalization"
            )
            return OrderResult(
                success=False,
                message=f"units_below_min_trade_size ({spec['min_trade_size']})",
            )

        entry = quote.ask if side == "buy" else quote.bid

        if side == "buy":
            sl = entry - (stop_loss_pips * pip)
            tp = entry + (take_profit_pips * pip)
        else:
            sl = entry + (stop_loss_pips * pip)
            tp = entry - (take_profit_pips * pip)

        # Signed units: negative = sell
        signed_units = abs_qty if side == "buy" else -abs_qty
        if spec["units_precision"] == 0:
            units_str = str(int(signed_units))
        else:
            units_str = f"{signed_units:.{spec['units_precision']}f}"

        # Format prices to instrument's display precision
        price_fmt = f"{{:.{price_precision}f}}"
        sl_str = price_fmt.format(sl)
        tp_str = price_fmt.format(tp)

        order_data = {
            "order": {
                "instrument": symbol,
                "units": units_str,
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": sl_str,
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": tp_str,
                    "timeInForce": "GTC",
                },
            }
        }

        result = self._post(
            f"/accounts/{self.account_id}/orders", order_data
        )

        if not result:
            logger.error(
                f"ORDER FAILED {symbol}: no response | "
                f"units={units_str} side={side} SL={sl_str} TP={tp_str}"
            )
            return OrderResult(success=False, message="No response from OANDA")

        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            return OrderResult(
                success=True,
                order_id=fill.get("id", ""),
                symbol=symbol,
                side=side,
                quantity=abs_qty,
                price=float(fill.get("price", 0)),
                order_type="market",
                status="filled",
                message=f"SL: {sl_str} | TP: {tp_str}",
            )

        # Detailed rejection logging (Recommendation #7)
        reject = result.get("orderRejectTransaction", {})
        reject_reason = reject.get("rejectReason", "unknown")
        acct_snapshot = self._account_state or {}
        logger.error(
            f"ORDER REJECTED {symbol}: {reject_reason} | "
            f"units={units_str} side={side} SL={sl_str} TP={tp_str} | "
            f"NAV={acct_snapshot.get('nav', '?')} "
            f"margin_avail={acct_snapshot.get('margin_available', '?')} | "
            f"full_response={json.dumps(reject)[:500]}"
        )
        return OrderResult(
            success=False,
            message=f"order_rejected_by_oanda: {reject_reason}",
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
            spec = self.get_instrument_spec(q.symbol)
            pip = spec["pip_size"]
            spread_pips = (q.ask - q.bid) / pip if pip > 0 else 0

            spreads.append({
                "symbol": q.symbol,
                "bid": q.bid,
                "ask": q.ask,
                "spread_pips": round(spread_pips, 1),
                "spread_pct": q.spread,
            })

        return sorted(spreads, key=lambda x: x["spread_pips"])
