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

# Options universe — 25 high-liquidity symbols for scanning
OPTIONS_SYMBOLS = [
    # Tier 1: Index ETFs (highest liquidity, tightest spreads)
    "SPY", "QQQ", "IWM", "DIA",
    # Tier 2: Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    # Tier 3: High-volume large-caps
    "AMD", "NFLX", "CRM", "COIN", "SQ", "UBER", "SHOP",
    # Tier 4: Sector ETFs
    "XLF", "XLE", "GDX", "SMH", "ARKK", "TLT", "EEM",
]


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
                opts_level = self._account.get("options_approved_level", 0)
                buying_power = float(self._account.get("buying_power", 0) or 0)
                logger.info(
                    f"Alpaca connected — Status: {status} | "
                    f"Options level: {opts_level} | "
                    f"Buying power: ${buying_power:,.2f}"
                )
                if opts_level == 0:
                    logger.warning(
                        "OPTIONS TRADING NOT ENABLED on this Alpaca account. "
                        "Go to alpaca.markets → Paper Trading → Account → "
                        "Options Trading to enable Level 1 or 2."
                    )
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
        """Get latest quote for a stock. Falls back to trade price if quote incomplete."""
        try:
            # Try quote first
            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                headers=self._headers,
                params={"feed": "iex"},
                timeout=10,
            )
            bid, ask = 0.0, 0.0
            if resp.ok:
                data = resp.json().get("quote", {})
                bid = float(data.get("bp", 0))
                ask = float(data.get("ap", 0))

            # If quote is incomplete (free tier often has 0 ask), fall back to last trade
            if bid <= 0 or ask <= 0:
                trade_resp = requests.get(
                    f"{self.data_url}/v2/stocks/{symbol}/trades/latest",
                    headers=self._headers,
                    params={"feed": "iex"},
                    timeout=10,
                )
                if trade_resp.ok:
                    trade_data = trade_resp.json().get("trade", {})
                    trade_price = float(trade_data.get("p", 0))
                    if trade_price > 0:
                        # Synthetic bid/ask from trade price with small spread
                        spread_est = trade_price * 0.001  # ~0.1% spread estimate
                        bid = bid if bid > 0 else trade_price - spread_est
                        ask = ask if ask > 0 else trade_price + spread_est
                        logger.debug(f"Quote fallback for {symbol}: trade={trade_price:.2f}")

            if bid > 0 or ask > 0:
                return Quote(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    spread=ask - bid if ask > bid else 0,
                )
        except requests.RequestException as e:
            logger.warning(f"Quote error for {symbol}: {e}")
        return None

    def get_candles(self, symbol: str, timeframe: str = "1Hour", count: int = 100) -> list[dict]:
        """
        Get historical bars for a stock.

        Alpaca supports: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month
        4Hour is NOT supported — we synthesize it from 1Hour bars.
        """
        from datetime import datetime, timedelta, timezone

        tf_map = {
            "H1": "1Hour", "H4": "4Hour", "D": "1Day",
            "M15": "15Min", "M5": "5Min",
            "1Hour": "1Hour", "4Hour": "4Hour", "1Day": "1Day",
            "15Min": "15Min", "5Min": "5Min",
        }
        tf = tf_map.get(timeframe, timeframe)

        # 4Hour is NOT a native Alpaca timeframe — synthesize from 1Hour
        if tf == "4Hour":
            h1_bars = self.get_candles(symbol, "1Hour", count * 4 + 4)
            return self._aggregate_bars(h1_bars, 4)[:count]

        # Calculate start date based on timeframe + count needed
        now = datetime.now(timezone.utc)
        tf_to_days = {
            "1Day": count * 2,      # Trading days ≈ count * 1.5, add buffer
            "1Hour": count // 6 + 5, # ~7 bars/day → count/7 days + buffer
            "15Min": count // 20 + 3,
            "5Min": count // 60 + 2,
        }
        lookback_days = tf_to_days.get(tf, count)
        start = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%dT00:00:00Z")

        try:
            params = {
                "timeframe": tf,
                "limit": min(count * 4 if tf == "1Hour" else count, 10000),
                "start": start,
                "feed": "iex",  # Free tier: use IEX feed (SIP requires paid subscription)
                "adjustment": "split",
            }

            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                headers=self._headers,
                params=params,
                timeout=15,
            )
            if resp.ok:
                bars = resp.json().get("bars") or []
                if not bars:
                    logger.debug(f"No bars returned for {symbol} {tf}")
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
            else:
                logger.warning(f"Alpaca bars API error for {symbol} {tf}: {resp.status_code} {resp.text[:200]}")
        except requests.RequestException as e:
            logger.warning(f"Candles error for {symbol} {tf}: {e}")
        return []

    @staticmethod
    def _aggregate_bars(bars: list[dict], period: int) -> list[dict]:
        """Aggregate smaller bars into larger ones (e.g. 1H → 4H)."""
        if not bars:
            return []
        result = []
        for i in range(0, len(bars) - period + 1, period):
            chunk = bars[i:i + period]
            if not chunk:
                continue
            result.append({
                "open": chunk[0]["open"],
                "high": max(b["high"] for b in chunk),
                "low": min(b["low"] for b in chunk),
                "close": chunk[-1]["close"],
                "volume": sum(b.get("volume", 0) for b in chunk),
                "time": chunk[0].get("time", ""),
            })
        return result

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: str = None,
        option_type: str = None,
    ) -> list[OptionQuote]:
        """
        Get options chain for a stock.

        Strategy (per ChatGPT diagnosis):
        1. Try snapshots endpoint first (/v1beta1/options/snapshots) — has greeks + quotes
        2. Fall back to contracts endpoint (/v2/options/contracts) + batch quotes
        3. If quotes unavailable, use close_price from contracts as synthetic quote
        """
        # ─── METHOD 1: Snapshots endpoint (best data, may require paid plan) ───
        options = self._get_chain_via_snapshots(symbol)
        if options:
            logger.info(f"Options chain for {symbol}: {len(options)} contracts via snapshots")
            return options

        # ─── METHOD 2: Contracts endpoint + batch quotes ───
        options = self._get_chain_via_contracts(symbol, expiration_date, option_type)
        if options:
            logger.info(f"Options chain for {symbol}: {len(options)} contracts via contracts+quotes")
            return options

        logger.warning(f"Options chain EMPTY for {symbol} — both methods failed")
        return []

    def _get_chain_via_snapshots(self, symbol: str) -> list[OptionQuote]:
        """Try the snapshots endpoint for full chain data with greeks.

        Uses expiration date range (tomorrow → +45d) and paginates
        to avoid getting stuck on 100 0DTE contracts only.
        """
        from datetime import datetime, timedelta, timezone

        try:
            # Avoid 0DTE — start from tomorrow, go out 45 days
            today = datetime.now(timezone.utc).date()
            exp_gte = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            exp_lte = (today + timedelta(days=45)).strftime("%Y-%m-%d")

            # Try to narrow strikes around current price
            strike_lo = 0.0
            strike_hi = 1e9
            try:
                q = self.get_quote(symbol)
                if q and q.mid > 0:
                    strike_lo = q.mid * 0.85
                    strike_hi = q.mid * 1.15
            except Exception:
                pass

            all_snapshots: dict = {}
            page_token = None
            pages = 0
            max_pages = 6  # up to 600 contracts

            while pages < max_pages:
                params = {
                    "feed": "indicative",
                    "limit": 100,
                    "expiration_date_gte": exp_gte,
                    "expiration_date_lte": exp_lte,
                }
                if strike_lo > 0:
                    params["strike_price_gte"] = f"{strike_lo:.2f}"
                    params["strike_price_lte"] = f"{strike_hi:.2f}"
                if page_token:
                    params["page_token"] = page_token

                resp = requests.get(
                    f"{self.data_url}/v1beta1/options/snapshots/{symbol}",
                    headers=self._headers,
                    params=params,
                    timeout=15,
                )
                if not resp.ok:
                    logger.warning(
                        f"Snapshots endpoint failed for {symbol}: "
                        f"{resp.status_code} {resp.text[:200]}"
                    )
                    break

                data = resp.json()
                page_snapshots = data.get("snapshots", {}) or {}
                if not page_snapshots:
                    break

                all_snapshots.update(page_snapshots)
                pages += 1
                page_token = data.get("next_page_token")
                if not page_token:
                    break

            if not all_snapshots:
                logger.warning(f"Snapshots empty for {symbol} — options data may require paid Alpaca plan")
                return []

            options = []
            for sym, snap in all_snapshots.items():
                try:
                    quote = snap.get("latestQuote", {})
                    trade = snap.get("latestTrade", {})

                    bid = float(quote.get("bp", 0) or 0)
                    ask = float(quote.get("ap", 0) or 0)

                    # Fall back to last trade if quote is incomplete
                    if bid <= 0 and ask <= 0:
                        trade_price = float(trade.get("p", 0) or 0)
                        if trade_price > 0:
                            bid = trade_price * 0.95
                            ask = trade_price * 1.05

                    # Parse option symbol to extract strike/expiry/type
                    # Alpaca option symbols: SPY250411C00550000
                    strike, expiry, opt_type, oi = 0.0, "", "", 0
                    if len(sym) > 6:
                        try:
                            root_end = next(i for i, c in enumerate(sym) if c.isdigit())
                            date_str = sym[root_end:root_end + 6]
                            opt_type_char = sym[root_end + 6]
                            strike_str = sym[root_end + 7:]
                            expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
                            opt_type = "call" if opt_type_char == "C" else "put"
                            strike = float(strike_str) / 1000
                        except (StopIteration, ValueError, IndexError):
                            pass

                    oi = int(snap.get("openInterest", 0) or 0)

                    if strike > 0 and (bid > 0 or ask > 0):
                        options.append(OptionQuote(
                            symbol=sym,
                            strike=strike,
                            expiration=expiry,
                            option_type=opt_type,
                            open_interest=oi,
                            bid=bid,
                            ask=ask,
                        ))
                except Exception as e:
                    logger.debug(f"Snapshot parse error {sym}: {e}")

            logger.info(
                f"Snapshots for {symbol}: {len(options)} contracts "
                f"(pages={pages}, range {exp_gte}→{exp_lte})"
            )
            return options
        except requests.RequestException as e:
            logger.debug(f"Snapshots request error for {symbol}: {e}")
        return []

    def _get_chain_via_contracts(
        self, symbol: str, expiration_date: str = None, option_type: str = None
    ) -> list[OptionQuote]:
        """Fetch contracts from trading API, then try to get quotes."""
        from datetime import datetime, timedelta, timezone

        try:
            params = {"underlying_symbols": symbol, "limit": 100, "status": "active"}
            if expiration_date:
                params["expiration_date"] = expiration_date
            else:
                # Default: skip 0DTE, look out 45 days
                today = datetime.now(timezone.utc).date()
                params["expiration_date_gte"] = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                params["expiration_date_lte"] = (today + timedelta(days=45)).strftime("%Y-%m-%d")
            if option_type:
                params["type"] = option_type

            # Narrow strikes around current price
            try:
                q = self.get_quote(symbol)
                if q and q.mid > 0:
                    params["strike_price_gte"] = f"{q.mid * 0.85:.2f}"
                    params["strike_price_lte"] = f"{q.mid * 1.15:.2f}"
            except Exception:
                pass

            resp = requests.get(
                f"{self.base_url}/v2/options/contracts",
                headers=self._headers,
                params=params,
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"Contracts endpoint failed for {symbol}: {resp.status_code} {resp.text[:200]}")
                return []

            contracts = resp.json().get("option_contracts", [])
            if not contracts:
                logger.info(f"No contracts returned for {symbol}")
                return []

            logger.info(f"Found {len(contracts)} contracts for {symbol}, fetching quotes...")

            # Build contract map
            contract_map = {}
            for c in contracts:
                sym = c.get("symbol") or ""
                if sym:
                    contract_map[sym] = c

            # Try batch quotes
            symbols_list = list(contract_map.keys())
            all_quotes = {}
            for batch_start in range(0, len(symbols_list), 50):
                batch = symbols_list[batch_start:batch_start + 50]
                quotes = self._get_option_quotes_batch(batch)
                all_quotes.update(quotes)

            quotes_found = sum(1 for q in all_quotes.values() if q)
            logger.info(f"Got quotes for {quotes_found}/{len(symbols_list)} contracts")

            # Build option list — use quotes if available, otherwise use close_price
            options = []
            for sym, c in contract_map.items():
                try:
                    q = all_quotes.get(sym, {})
                    bid = float(q.get("bp") or 0) if q else 0
                    ask = float(q.get("ap") or 0) if q else 0

                    # If no quotes, use close_price from contract as synthetic price
                    if bid <= 0 and ask <= 0:
                        close_price = float(c.get("close_price") or 0)
                        if close_price > 0:
                            bid = close_price * 0.90  # Conservative synthetic spread
                            ask = close_price * 1.10
                            logger.debug(f"Synthetic quote for {sym}: close={close_price:.2f}")

                    options.append(OptionQuote(
                        symbol=sym,
                        strike=float(c.get("strike_price") or 0),
                        expiration=c.get("expiration_date") or "",
                        option_type=c.get("type") or "",
                        open_interest=int(c.get("open_interest") or 0),
                        bid=bid,
                        ask=ask,
                    ))
                except (TypeError, ValueError) as e:
                    logger.debug(f"Skipping contract {sym}: {e}")

            return options

        except requests.RequestException as e:
            logger.warning(f"Contracts request error for {symbol}: {e}")
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
                logger.warning(f"Batch option quotes failed: {resp.status_code} — {resp.text[:200]}")
        except requests.RequestException as e:
            logger.warning(f"Batch option quotes error: {e}")
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
            order_data["limit_price"] = str(round(float(limit_price), 2))

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
