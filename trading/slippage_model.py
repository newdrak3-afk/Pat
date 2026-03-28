"""
Realistic slippage and execution cost model for forex trades.

Estimates spread costs, market-impact slippage, and commissions so that
trade targets (SL/TP) can be adjusted to reflect true expected costs.
"""

from dataclasses import dataclass


# ── Currency pair classification ──────────────────────────────────────

MAJOR_PAIRS = frozenset({
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    # Common crosses among majors
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
})

MINOR_PAIRS = frozenset({
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY", "NZD_CAD",
    "NZD_CHF", "NZD_JPY",
})


@dataclass(frozen=True)
class SlippageCost:
    """Breakdown of estimated execution costs for a single trade."""
    spread_cost: float       # cost from bid-ask spread in pips
    slippage_cost: float     # market-impact slippage in pips
    commission_cost: float   # broker commission converted to pips
    total_cost_pips: float   # sum of all cost components in pips
    total_cost_pct: float    # total cost as a percentage of entry price


class SlippageModel:
    """Estimates realistic execution costs for forex trades.

    Cost components
    ---------------
    1. **Spread cost** – half the bid-ask spread is paid on entry and half on exit,
       but since the spread is already baked into the quoted price the full spread
       is effectively paid once per round-trip.
    2. **Slippage (market impact)** – scales with position size.  Larger orders
       move the market more.  Exotic pairs have thinner books and therefore higher
       impact.
    3. **Commission** – a fixed per-unit fee converted to pips.

    Parameters
    ----------
    commission_per_unit : float
        Broker commission per unit of base currency (default 0.00002 ~ $2 per 100k).
    base_slippage_major : float
        Baseline slippage in pips for a "standard" (100 000 unit) order on a
        major pair during normal liquidity.
    base_slippage_minor : float
        Same for minor / cross pairs.
    base_slippage_exotic : float
        Same for exotic pairs.
    size_impact_exponent : float
        Exponent controlling how slippage grows with size.  Slippage is
        proportional to (units / 100_000) ** exponent.  A square-root model
        (0.5) is the classic market-microstructure assumption.
    standard_lot : int
        Reference lot size (units) for the base slippage values.
    """

    def __init__(
        self,
        commission_per_unit: float = 0.00002,
        base_slippage_major: float = 0.3,
        base_slippage_minor: float = 0.6,
        base_slippage_exotic: float = 1.5,
        size_impact_exponent: float = 0.5,
        standard_lot: int = 100_000,
    ):
        self.commission_per_unit = commission_per_unit
        self.base_slippage_major = base_slippage_major
        self.base_slippage_minor = base_slippage_minor
        self.base_slippage_exotic = base_slippage_exotic
        self.size_impact_exponent = size_impact_exponent
        self.standard_lot = standard_lot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_pair(self, symbol: str) -> str:
        """Return 'major', 'minor', or 'exotic'."""
        s = symbol.upper().replace("/", "_")
        if s in MAJOR_PAIRS:
            return "major"
        if s in MINOR_PAIRS:
            return "minor"
        return "exotic"

    def _base_slippage_for(self, symbol: str) -> float:
        kind = self._classify_pair(symbol)
        if kind == "major":
            return self.base_slippage_major
        if kind == "minor":
            return self.base_slippage_minor
        return self.base_slippage_exotic

    def _pip_value_in_price(self, symbol: str) -> float:
        """Return the price-unit value of 1 pip for *symbol*.

        For JPY pairs 1 pip = 0.01; for everything else 1 pip = 0.0001.
        """
        s = symbol.upper().replace("/", "_")
        if "JPY" in s:
            return 0.01
        return 0.0001

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_costs(
        self,
        symbol: str,
        units: int,
        side: str = "buy",
        spread_pips: float = 1.0,
    ) -> SlippageCost:
        """Estimate total execution costs for a forex trade.

        Args:
            symbol: Currency pair, e.g. "EUR_USD" or "EUR/USD".
            units: Position size in units of base currency.
            side: "buy" or "sell" (currently symmetric; kept for future asymmetry).
            spread_pips: Current bid-ask spread in pips.

        Returns:
            A :class:`SlippageCost` with the full cost breakdown.
        """
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        # 1) Spread cost (full round-trip)
        spread_cost = spread_pips

        # 2) Slippage — scales with (units / standard_lot) ^ exponent
        base_slip = self._base_slippage_for(symbol)
        size_ratio = units / self.standard_lot
        slippage_cost = base_slip * (size_ratio ** self.size_impact_exponent)

        # 3) Commission in pips
        pip_price = self._pip_value_in_price(symbol)
        commission_cost = (self.commission_per_unit * units) / (pip_price * units)
        # simplifies to commission_per_unit / pip_price, but kept explicit
        commission_cost = self.commission_per_unit / pip_price

        total_pips = spread_cost + slippage_cost + commission_cost

        # Express total cost as percentage of a notional entry price.
        # Use 1 pip value as the scaling reference (cost_pips * pip_value).
        total_cost_pct = total_pips * pip_price * 100.0  # as percentage

        return SlippageCost(
            spread_cost=round(spread_cost, 4),
            slippage_cost=round(slippage_cost, 4),
            commission_cost=round(commission_cost, 4),
            total_cost_pips=round(total_pips, 4),
            total_cost_pct=round(total_cost_pct, 6),
        )

    def adjust_targets(
        self,
        entry: float,
        sl: float,
        tp: float,
        side: str,
        costs: SlippageCost,
    ) -> tuple[float, float]:
        """Adjust stop-loss and take-profit to account for execution costs.

        For a **buy** trade the effective entry is *higher* than quoted (we pay
        the spread/slippage on top), so:
        - SL must be moved **down** (wider) by the cost in price terms.
        - TP must be moved **up** (wider) by the cost so the net gain target
          remains the same after costs — but since costs eat into profit we
          instead *narrow* the displayed TP to reflect the true net.

        The mirror logic applies for sells.

        Args:
            entry: Planned entry price.
            sl: Original stop-loss price.
            tp: Original take-profit price.
            side: "buy" or "sell".
            costs: A :class:`SlippageCost` (obtain from :meth:`estimate_costs`).

        Returns:
            (adjusted_sl, adjusted_tp) tuple.
        """
        # Determine pip value in price units from the entry magnitude
        # (heuristic: if entry > 50 it is likely a JPY pair)
        pip_price = 0.01 if entry > 50 else 0.0001
        cost_in_price = costs.total_cost_pips * pip_price

        side = side.lower()
        if side == "buy":
            adjusted_sl = sl - cost_in_price   # widen SL
            adjusted_tp = tp - cost_in_price   # narrow effective TP
        elif side == "sell":
            adjusted_sl = sl + cost_in_price   # widen SL
            adjusted_tp = tp + cost_in_price   # narrow effective TP
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        return (round(adjusted_sl, 6), round(adjusted_tp, 6))

    def get_breakeven_pips(self, symbol: str, units: int) -> float:
        """Return the minimum price movement (in pips) needed to break even.

        This equals the total round-trip cost (spread + slippage + commission)
        for the given symbol and size, using a typical 1-pip spread.

        Args:
            symbol: Currency pair.
            units: Position size in base-currency units.

        Returns:
            Breakeven distance in pips.
        """
        costs = self.estimate_costs(symbol, units, side="buy", spread_pips=1.0)
        return costs.total_cost_pips
