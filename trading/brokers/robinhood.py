"""
Robinhood broker integration — stocks and options (calls/puts).

Uses robin_stocks library for API access.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from trading.brokers.base import (
    BaseBroker, Quote, OptionQuote, OrderResult, Position,
)

logger = logging.getLogger(__name__)

try:
    import robin_stocks.robinhood as rh

    HAS_ROBINHOOD = True
except ImportError:
    HAS_ROBINHOOD = False
    logger.warning("robin_stocks not installed — pip install robin_stocks")


class RobinhoodBroker(BaseBroker):
    """
    Robinhood integration for stocks and options.

    Supports:
    - Stock quotes and trading
    - Options chain scanning
    - Buying calls and puts
    - Position management
    """

    def __init__(self):
        self.username = os.getenv("ROBINHOOD_USERNAME", "")
        self.password = os.getenv("ROBINHOOD_PASSWORD", "")
        self.mfa_code = os.getenv("ROBINHOOD_MFA", "")
        self.connected = False

    def connect(self) -> bool:
        """Log into Robinhood."""
        if not HAS_ROBINHOOD:
            logger.error("robin_stocks not installed")
            return False

        if not self.username or not self.password:
            logger.error(
                "Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env"
            )
            return False

        try:
            if self.mfa_code:
                login = rh.login(
                    self.username, self.password,
                    mfa_code=self.mfa_code,
                    store_session=True,
                )
            else:
                login = rh.login(
                    self.username, self.password,
                    store_session=True,
                )

            if login:
                self.connected = True
                logger.info("Connected to Robinhood")
                return True

        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")

        return False

    def get_account_balance(self) -> float:
        """Get buying power."""
        if not self.connected:
            return 0.0

        try:
            profile = rh.profiles.load_account_profile()
            return float(profile.get("buying_power", 0))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get a stock quote."""
        if not self.connected:
            return None

        try:
            quote = rh.stocks.get_stock_quote_by_symbol(symbol)
            if not quote:
                return None

            bid = float(quote.get("bid_price", 0) or 0)
            ask = float(quote.get("ask_price", 0) or 0)
            last = float(quote.get("last_trade_price", 0) or 0)
            prev_close = float(quote.get("previous_close", 0) or 0)

            change_pct = 0.0
            if prev_close > 0:
                change_pct = ((last - prev_close) / prev_close) * 100

            spread = 0.0
            if ask > 0:
                spread = ((ask - bid) / ask) * 100

            return Quote(
                symbol=symbol.upper(),
                bid=bid,
                ask=ask,
                last=last,
                volume=float(quote.get("volume", 0) or 0),
                spread=spread,
                change_pct=change_pct,
                high=float(quote.get("high_price", 0) or 0),
                low=float(quote.get("low_price", 0) or 0),
            )

        except Exception as e:
            logger.error(f"Quote failed for {symbol}: {e}")
            return None

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        option_type: str = "both",
    ) -> list[OptionQuote]:
        """
        Get options chain for a stock.

        Args:
            symbol: Stock ticker (e.g. "AAPL")
            expiration: Date string "YYYY-MM-DD" or None for nearest
            option_type: "call", "put", or "both"
        """
        if not self.connected:
            return []

        options = []

        try:
            # Get nearest expiration if not specified
            if not expiration:
                chains = rh.options.get_chains(symbol)
                if chains and chains.get("expiration_dates"):
                    expiration = chains["expiration_dates"][0]
                else:
                    return []

            # Get stock price for context
            stock_quote = self.get_quote(symbol)
            underlying = stock_quote.last if stock_quote else 0

            types_to_fetch = []
            if option_type in ("call", "both"):
                types_to_fetch.append("call")
            if option_type in ("put", "both"):
                types_to_fetch.append("put")

            for opt_type in types_to_fetch:
                chain = rh.options.find_options_by_expiration(
                    symbol,
                    expirationDate=expiration,
                    optionType=opt_type,
                )

                if not chain:
                    continue

                for opt in chain:
                    try:
                        strike = float(opt.get("strike_price", 0) or 0)
                        bid = float(opt.get("bid_price", 0) or 0)
                        ask = float(opt.get("ask_price", 0) or 0)
                        last = float(
                            opt.get("last_trade_price", 0) or 0
                        )
                        iv = float(
                            opt.get("implied_volatility", 0) or 0
                        )
                        delta = float(opt.get("delta", 0) or 0)
                        gamma = float(opt.get("gamma", 0) or 0)
                        theta = float(opt.get("theta", 0) or 0)
                        volume = int(float(opt.get("volume", 0) or 0))
                        oi = int(
                            float(opt.get("open_interest", 0) or 0)
                        )

                        itm = (
                            (opt_type == "call" and strike < underlying)
                            or (opt_type == "put" and strike > underlying)
                        )

                        options.append(OptionQuote(
                            symbol=symbol.upper(),
                            strike=strike,
                            expiration=expiration,
                            option_type=opt_type,
                            bid=bid,
                            ask=ask,
                            last=last,
                            volume=volume,
                            open_interest=oi,
                            implied_volatility=iv,
                            delta=delta,
                            gamma=gamma,
                            theta=theta,
                            underlying_price=underlying,
                            in_the_money=itm,
                        ))

                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            logger.error(f"Options chain failed for {symbol}: {e}")

        return options

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float = 0.0,
    ) -> OrderResult:
        """Place a stock order."""
        if not self.connected:
            return OrderResult(
                success=False, message="Not connected to Robinhood"
            )

        try:
            if side == "buy":
                if order_type == "limit" and price > 0:
                    result = rh.orders.order_buy_limit(
                        symbol, quantity, price
                    )
                else:
                    result = rh.orders.order_buy_market(symbol, quantity)
            else:
                if order_type == "limit" and price > 0:
                    result = rh.orders.order_sell_limit(
                        symbol, quantity, price
                    )
                else:
                    result = rh.orders.order_sell_market(symbol, quantity)

            if result and result.get("id"):
                return OrderResult(
                    success=True,
                    order_id=result["id"],
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=float(
                        result.get("average_price", 0)
                        or result.get("price", 0)
                        or 0
                    ),
                    order_type=order_type,
                    status=result.get("state", "pending"),
                )
            else:
                return OrderResult(
                    success=False,
                    message=str(result),
                )

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(success=False, message=str(e))

    def buy_option(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,
        quantity: int = 1,
        price: float = 0.0,
    ) -> OrderResult:
        """
        Buy a call or put option.

        Args:
            symbol: Stock ticker
            strike: Strike price
            expiration: "YYYY-MM-DD"
            option_type: "call" or "put"
            quantity: Number of contracts
            price: Limit price (0 = market)
        """
        if not self.connected:
            return OrderResult(
                success=False, message="Not connected to Robinhood"
            )

        try:
            if price > 0:
                result = rh.orders.order_buy_option_limit(
                    "open", "debit",
                    price, symbol, quantity,
                    expiration, strike, option_type,
                )
            else:
                result = rh.orders.order_buy_option_limit(
                    "open", "debit",
                    0.01, symbol, quantity,  # will fill at market
                    expiration, strike, option_type,
                )

            if result and result.get("id"):
                return OrderResult(
                    success=True,
                    order_id=result["id"],
                    symbol=f"{symbol} {strike}{option_type[0].upper()} {expiration}",
                    side="buy",
                    quantity=quantity,
                    price=price,
                    order_type="limit" if price > 0 else "market",
                    status=result.get("state", "pending"),
                )
            else:
                msg = result.get("detail", str(result)) if result else "No response"
                return OrderResult(success=False, message=msg)

        except Exception as e:
            logger.error(f"Option order failed: {e}")
            return OrderResult(success=False, message=str(e))

    def get_positions(self) -> list[Position]:
        """Get all open stock positions."""
        if not self.connected:
            return []

        positions = []
        try:
            holdings = rh.account.build_holdings()
            for symbol, data in holdings.items():
                qty = float(data.get("quantity", 0))
                avg = float(data.get("average_buy_price", 0))
                cur = float(data.get("price", 0))
                pnl = float(data.get("equity_change", 0))
                pnl_pct = float(data.get("percent_change", 0))

                positions.append(Position(
                    symbol=symbol,
                    side="long",
                    quantity=qty,
                    entry_price=avg,
                    current_price=cur,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")

        return positions

    def get_option_positions(self) -> list[dict]:
        """Get all open option positions."""
        if not self.connected:
            return []

        try:
            return rh.options.get_open_option_positions() or []
        except Exception as e:
            logger.error(f"Failed to get option positions: {e}")
            return []

    def close_position(self, symbol: str) -> OrderResult:
        """Close a stock position by selling all shares."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return self.place_order(
                    symbol, "sell", pos.quantity, "market"
                )

        return OrderResult(
            success=False, message=f"No position found for {symbol}"
        )

    def get_movers(self) -> list[dict]:
        """Get top movers (biggest gainers/losers)."""
        if not self.connected:
            return []

        try:
            movers = rh.markets.get_top_movers()
            return movers or []
        except Exception as e:
            logger.error(f"Failed to get movers: {e}")
            return []

    def get_watchlist_quotes(
        self, symbols: list[str]
    ) -> list[Quote]:
        """Get quotes for a list of symbols."""
        quotes = []
        for symbol in symbols:
            q = self.get_quote(symbol)
            if q:
                quotes.append(q)
        return quotes
