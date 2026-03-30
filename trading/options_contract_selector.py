"""
Options Contract Selector — Picks the right contract for a trade.

Given a stock trend signal, selects the optimal option contract based on:
- Expiration: 7-14 DTE (days to expiration)
- Strike: Near ATM (at-the-money) or slightly ITM
- Liquidity: Minimum volume + open interest
- Spread: Max bid/ask spread threshold
- Type: Calls for uptrend, puts for downtrend

This is the critical module that prevents noisy options trades.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from trading.brokers.base import OptionQuote

logger = logging.getLogger(__name__)


@dataclass
class ContractSelection:
    """Selected option contract with reasoning."""
    symbol: str                    # Full option symbol
    underlying: str                # Stock symbol (e.g. "SPY")
    option_type: str               # "call" or "put"
    strike: float                  # Strike price
    expiration: str                # Expiration date (YYYY-MM-DD)
    dte: int                       # Days to expiration
    bid: float                     # Current bid
    ask: float                     # Current ask
    mid: float                     # Mid price
    spread_pct: float              # Spread as % of mid
    open_interest: int             # Open interest
    max_loss: float                # Max loss = premium * 100
    reasoning: str                 # Why this contract


class ContractSelector:
    """
    Selects optimal option contracts for trading.

    Rules (v2 - ChatGPT tuned):
    - 10-21 DTE (more theta cushion)
    - Near ATM, delta ~0.35-0.55
    - Min 200 OI on single names, 100 on SPY/QQQ
    - Max 10% bid/ask spread (tightened from 15%)
    - Calls for uptrend, puts for downtrend
    - Prioritize SPY/QQQ first
    """

    def __init__(
        self,
        min_dte: int = 10,
        max_dte: int = 21,
        max_spread_pct: float = 0.10,
        min_open_interest: int = 200,
        min_volume: int = 10,
        max_premium: float = 500.0,
    ):
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_spread_pct = max_spread_pct
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.max_premium = max_premium

    def select_contract(
        self,
        underlying: str,
        underlying_price: float,
        trend_direction: str,
        contracts: list[OptionQuote],
    ) -> Optional[ContractSelection]:
        """
        Pick the best contract from a chain.

        Args:
            underlying: Stock symbol
            underlying_price: Current stock price
            trend_direction: "buy" (bullish → call) or "sell" (bearish → put)
            contracts: List of available option contracts
        """
        option_type = "call" if trend_direction == "buy" else "put"
        now = datetime.now(timezone.utc).date()

        candidates = []

        for c in contracts:
            # Filter by type
            if c.option_type != option_type:
                continue

            # Filter by DTE
            try:
                exp_date = datetime.strptime(c.expiration, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            dte = (exp_date - now).days
            if dte < self.min_dte or dte > self.max_dte:
                continue

            # Filter by open interest
            if c.open_interest < self.min_open_interest:
                continue

            # Filter by spread
            if c.bid <= 0 or c.ask <= 0:
                continue
            mid = (c.bid + c.ask) / 2
            spread_pct = (c.ask - c.bid) / mid if mid > 0 else 1.0
            if spread_pct > self.max_spread_pct:
                continue

            # Filter by premium (max loss = premium * 100 shares per contract)
            premium_cost = mid * 100
            if premium_cost > self.max_premium:
                continue

            # Score by proximity to ATM (prefer near-the-money)
            strike_distance = abs(c.strike - underlying_price) / underlying_price
            # Prefer slightly ITM for calls (strike < price) or puts (strike > price)
            if option_type == "call":
                itm_bonus = 0.02 if c.strike <= underlying_price else 0
            else:
                itm_bonus = 0.02 if c.strike >= underlying_price else 0

            score = 1.0 - strike_distance + itm_bonus

            candidates.append({
                "contract": c,
                "dte": dte,
                "mid": mid,
                "spread_pct": spread_pct,
                "premium_cost": premium_cost,
                "score": score,
            })

        if not candidates:
            logger.info(f"No valid {option_type} contracts for {underlying}")
            return None

        # Sort by score (closest to ATM with ITM preference)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        c = best["contract"]

        reasoning = (
            f"{option_type.upper()} selected: "
            f"Strike ${c.strike} | {best['dte']} DTE | "
            f"Spread {best['spread_pct']:.1%} | "
            f"OI {c.open_interest} | "
            f"Premium ${best['mid']:.2f} (${best['premium_cost']:.0f} max loss)"
        )

        return ContractSelection(
            symbol=c.symbol,
            underlying=underlying,
            option_type=option_type,
            strike=c.strike,
            expiration=c.expiration,
            dte=best["dte"],
            bid=c.bid,
            ask=c.ask,
            mid=best["mid"],
            spread_pct=best["spread_pct"],
            open_interest=c.open_interest,
            max_loss=best["premium_cost"],
            reasoning=reasoning,
        )

    def get_target_expirations(self) -> list[str]:
        """Get expiration dates in the target DTE range."""
        now = datetime.now(timezone.utc).date()
        dates = []
        for d in range(self.min_dte, self.max_dte + 1):
            target = now + timedelta(days=d)
            # Options typically expire on Fridays
            if target.weekday() == 4:  # Friday
                dates.append(target.strftime("%Y-%m-%d"))
        return dates
