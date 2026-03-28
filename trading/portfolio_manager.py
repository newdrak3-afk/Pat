"""
Portfolio Manager — Currency exposure tracking and correlation control.

Manages open forex positions across 28 OANDA pairs, enforcing per-currency
and total exposure limits and blocking trades that would create excessive
correlated exposure.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# The 28 standard forex pairs on OANDA (majors, minors, crosses).
FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CHF_JPY",
]

ALL_CURRENCIES = {"EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}


def split_pair(symbol: str) -> tuple[str, str]:
    """Split 'EUR_USD' into ('EUR', 'USD')."""
    parts = symbol.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid forex pair symbol: {symbol}")
    return parts[0], parts[1]


@dataclass
class PortfolioPosition:
    """An open forex position tracked by the portfolio manager."""
    symbol: str
    side: str           # "long" or "short"
    quantity: float     # units
    entry_price: float
    pnl: float = 0.0

    @property
    def base_currency(self) -> str:
        return split_pair(self.symbol)[0]

    @property
    def quote_currency(self) -> str:
        return split_pair(self.symbol)[1]

    @property
    def notional(self) -> float:
        """Absolute notional value in quote currency terms."""
        return abs(self.quantity * self.entry_price)


class PortfolioManager:
    """
    Tracks open forex positions and enforces exposure limits.

    Exposure is measured as a fraction of account balance.  Each currency
    gets a directional exposure value: long base = +exposure on base,
    -exposure on quote; short base = the reverse.

    Parameters
    ----------
    max_currency_exposure_pct : float
        Maximum net exposure to any single currency as a fraction of
        account balance.  Default 0.30 (30%).
    max_total_exposure_pct : float
        Maximum gross exposure across all positions as a fraction of
        account balance.  Default 0.60 (60%).
    max_correlated_positions : int
        Maximum number of open positions that share a currency before
        new correlated trades are blocked.  Default 3.
    """

    def __init__(
        self,
        max_currency_exposure_pct: float = 0.30,
        max_total_exposure_pct: float = 0.60,
        max_correlated_positions: int = 3,
    ):
        self.max_currency_exposure_pct = max_currency_exposure_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_correlated_positions = max_correlated_positions
        self._positions: dict[str, PortfolioPosition] = {}

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(
        self,
        symbol: str,
        side: str,
        units: float,
        entry_price: float,
    ) -> None:
        """Add an open position to the portfolio."""
        if symbol in self._positions:
            raise ValueError(
                f"Position already exists for {symbol}. "
                "Remove it first or use a different symbol key."
            )
        if side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got '{side}'")
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        pos = PortfolioPosition(
            symbol=symbol,
            side=side,
            quantity=units,
            entry_price=entry_price,
        )
        self._positions[symbol] = pos
        logger.info(
            "Added position: %s %s %.0f units @ %.5f",
            side, symbol, units, entry_price,
        )

    def remove_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Remove and return a position.  Returns None if not found."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            logger.warning("remove_position: no position found for %s", symbol)
        else:
            logger.info("Removed position: %s", symbol)
        return pos

    def update_pnl(self, symbol: str, pnl: float) -> None:
        """Update the PnL of an existing position."""
        if symbol in self._positions:
            self._positions[symbol].pnl = pnl

    @property
    def positions(self) -> dict[str, PortfolioPosition]:
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Exposure calculations
    # ------------------------------------------------------------------

    def _currency_exposures(self, balance: float) -> dict[str, float]:
        """
        Calculate net directional exposure per currency as a fraction of
        account balance.

        Convention:
        - Long EUR_USD  => +exposure EUR, -exposure USD
        - Short EUR_USD => -exposure EUR, +exposure USD

        Returns a dict like {"EUR": 0.15, "USD": -0.10, ...}.
        """
        exposures: dict[str, float] = {c: 0.0 for c in ALL_CURRENCIES}
        if balance <= 0:
            return exposures

        for pos in self._positions.values():
            base, quote = split_pair(pos.symbol)
            notional_frac = pos.notional / balance
            if pos.side == "long":
                exposures[base] += notional_frac
                exposures[quote] -= notional_frac
            else:  # short
                exposures[base] -= notional_frac
                exposures[quote] += notional_frac

        return exposures

    def _gross_exposure(self, balance: float) -> float:
        """Total gross exposure as a fraction of balance."""
        if balance <= 0:
            return 0.0
        return sum(pos.notional for pos in self._positions.values()) / balance

    # ------------------------------------------------------------------
    # Trade gating
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        symbol: str,
        side: str,
        units: float,
        balance: float,
    ) -> tuple[bool, str]:
        """
        Check whether a new position can be added without breaching limits.

        Returns (allowed, reason).
        """
        # --- basic validation ---
        try:
            base, quote = split_pair(symbol)
        except ValueError as exc:
            return False, str(exc)

        if side not in ("long", "short"):
            return False, f"Invalid side '{side}'; must be 'long' or 'short'"
        if units <= 0:
            return False, "units must be positive"
        if balance <= 0:
            return False, "balance must be positive"

        if symbol in self._positions:
            return False, f"Position already open for {symbol}"

        # --- simulate adding the position ---
        # We need an entry_price to compute notional; use 1.0 as placeholder
        # since the caller provides units (notional = units * entry_price).
        # For the gate check we estimate notional as units (assuming price ~1
        # for major pairs).  A more accurate check would require passing the
        # current price, but units already encodes the intended exposure.
        simulated_notional = units  # units is exposure in base currency terms

        # Total exposure check
        current_gross = self._gross_exposure(balance)
        added_frac = simulated_notional / balance
        projected_gross = current_gross + added_frac
        if projected_gross > self.max_total_exposure_pct:
            return (
                False,
                f"Total exposure would be {projected_gross:.1%}, "
                f"exceeding limit of {self.max_total_exposure_pct:.0%}. "
                f"Current: {current_gross:.1%}, adding {added_frac:.1%}.",
            )

        # Per-currency exposure check
        exposures = self._currency_exposures(balance)
        if side == "long":
            exposures[base] = exposures.get(base, 0.0) + added_frac
            exposures[quote] = exposures.get(quote, 0.0) - added_frac
        else:
            exposures[base] = exposures.get(base, 0.0) - added_frac
            exposures[quote] = exposures.get(quote, 0.0) + added_frac

        for ccy, exp in exposures.items():
            if abs(exp) > self.max_currency_exposure_pct:
                return (
                    False,
                    f"{ccy} exposure would be {exp:+.1%}, "
                    f"exceeding limit of {self.max_currency_exposure_pct:.0%}.",
                )

        # Correlation check: count existing positions sharing base or quote
        shared_count = 0
        for pos in self._positions.values():
            pos_base, pos_quote = split_pair(pos.symbol)
            if base in (pos_base, pos_quote) or quote in (pos_base, pos_quote):
                shared_count += 1

        if shared_count >= self.max_correlated_positions:
            shared_symbols = [
                pos.symbol for pos in self._positions.values()
                if base in split_pair(pos.symbol)
                or quote in split_pair(pos.symbol)
            ]
            return (
                False,
                f"Correlated exposure too high: {shared_count} existing "
                f"positions share a currency with {symbol} "
                f"(limit {self.max_correlated_positions}). "
                f"Correlated: {', '.join(shared_symbols)}.",
            )

        return True, "Position allowed"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_exposure_report(self, balance: float = 0.0) -> dict:
        """
        Build a full exposure report.

        Returns
        -------
        dict with keys:
            positions       – list of position summaries
            currency_exposure – per-currency net exposure (fraction of balance)
            total_gross_exposure – gross exposure fraction
            balance         – account balance used
            limits          – configured limits
        """
        effective_balance = balance if balance > 0 else 1.0  # avoid div/0
        exposures = self._currency_exposures(effective_balance)
        gross = self._gross_exposure(effective_balance)

        position_summaries = []
        for pos in self._positions.values():
            position_summaries.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "notional": pos.notional,
                "pnl": pos.pnl,
            })

        # Only include currencies with non-zero exposure
        active_exposures = {
            ccy: round(exp, 6)
            for ccy, exp in sorted(exposures.items())
            if abs(exp) > 1e-9
        }

        return {
            "positions": position_summaries,
            "currency_exposure": active_exposures,
            "total_gross_exposure": round(gross, 6),
            "balance": balance,
            "limits": {
                "max_currency_exposure_pct": self.max_currency_exposure_pct,
                "max_total_exposure_pct": self.max_total_exposure_pct,
                "max_correlated_positions": self.max_correlated_positions,
            },
        }

    def get_correlation_matrix(self) -> dict[str, list[str]]:
        """
        Show which open positions share currencies.

        Returns a dict mapping each open symbol to a list of other open
        symbols that share a base or quote currency with it.
        """
        matrix: dict[str, list[str]] = {}
        symbols = list(self._positions.keys())

        for sym in symbols:
            base_a, quote_a = split_pair(sym)
            correlated = []
            for other in symbols:
                if other == sym:
                    continue
                base_b, quote_b = split_pair(other)
                shared = {base_a, quote_a} & {base_b, quote_b}
                if shared:
                    correlated.append(other)
            matrix[sym] = correlated

        return matrix

    def rebalance_suggestions(self) -> list[str]:
        """
        Generate plain-English suggestions for reducing portfolio risk.

        Checks:
        1. Currencies with outsized exposure
        2. Positions whose PnL is deeply negative (cut losers)
        3. Excessive correlation clusters
        """
        suggestions: list[str] = []

        # Use notional sum as a rough balance proxy when no balance is given
        total_notional = sum(p.notional for p in self._positions.values()) or 1.0
        exposures = self._currency_exposures(total_notional)

        # --- over-exposed currencies ---
        for ccy, exp in exposures.items():
            if abs(exp) > self.max_currency_exposure_pct:
                direction = "long" if exp > 0 else "short"
                contributing = []
                for pos in self._positions.values():
                    b, q = split_pair(pos.symbol)
                    if ccy in (b, q):
                        contributing.append(pos.symbol)
                suggestions.append(
                    f"Reduce {ccy} exposure ({direction} {abs(exp):.0%}): "
                    f"consider trimming or closing one of "
                    f"{', '.join(contributing)}."
                )

        # --- correlation clusters ---
        matrix = self.get_correlation_matrix()
        flagged_clusters: set[frozenset[str]] = set()
        for sym, correlated in matrix.items():
            if len(correlated) >= self.max_correlated_positions:
                cluster = frozenset([sym] + correlated)
                if cluster not in flagged_clusters:
                    flagged_clusters.add(cluster)
                    suggestions.append(
                        f"High correlation cluster: "
                        f"{', '.join(sorted(cluster))}. "
                        f"Consider closing one to reduce correlated risk."
                    )

        # --- cut losers ---
        for pos in self._positions.values():
            if pos.pnl < 0 and pos.notional > 0:
                loss_pct = abs(pos.pnl) / pos.notional
                if loss_pct > 0.02:  # losing more than 2% of notional
                    suggestions.append(
                        f"Cut loser: {pos.symbol} is down "
                        f"{loss_pct:.1%} of notional (PnL {pos.pnl:+.2f}). "
                        f"Consider closing to limit further loss."
                    )

        if not suggestions:
            suggestions.append("Portfolio looks balanced. No rebalancing needed.")

        return suggestions
