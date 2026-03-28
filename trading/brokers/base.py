"""
Base broker interface — all brokers implement this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Quote:
    """A price quote for any asset."""
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    spread: float = 0.0
    change_pct: float = 0.0
    high: float = 0.0
    low: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass
class OptionQuote:
    """Quote for a stock option."""
    symbol: str
    strike: float = 0.0
    expiration: str = ""
    option_type: str = ""        # "call" or "put"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    underlying_price: float = 0.0
    in_the_money: bool = False


@dataclass
class OrderResult:
    """Result of placing an order."""
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""               # "buy" or "sell"
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = ""         # "market", "limit"
    status: str = ""             # "filled", "pending", "rejected"
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Position:
    """An open position."""
    symbol: str
    side: str = ""               # "long" or "short"
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


class BaseBroker(ABC):
    """Base interface for all broker integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect/authenticate with the broker."""
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get a price quote for a symbol."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float = 0.0,
    ) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        """Close a position."""
        pass
