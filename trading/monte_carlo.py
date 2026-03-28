"""
Monte Carlo Risk Simulation — Simulate thousands of possible trading futures.

Takes historical trade PnL data and runs Monte Carlo simulations to estimate
probability of ruin, expected drawdowns, optimal position sizing, and
risk-adjusted return metrics.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""

    median_final_balance: float
    p5_final_balance: float          # 5th percentile (worst case)
    p95_final_balance: float         # 95th percentile (best case)
    probability_of_ruin: float       # fraction of sims hitting 50% drawdown
    max_drawdown_median: float       # median max drawdown across sims
    max_drawdown_p95: float          # 95th percentile max drawdown
    expected_annual_return: float    # annualised return (250 days, ~5 trades/day)
    sharpe_ratio: float
    equity_curves: list[list[float]] = field(default_factory=list, repr=False)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trade PnL sequences.

    Resamples historical trade results with replacement to build
    thousands of synthetic equity curves, then derives risk and
    performance statistics from the distribution of outcomes.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        trade_pnls: list[float],
        num_simulations: int = 1000,
        num_trades: int = 100,
        initial_balance: float = 100_000,
    ) -> MonteCarloResult:
        """
        Run *num_simulations* random-resampled equity paths of length
        *num_trades*, starting from *initial_balance*.

        Each simulated trade is drawn (with replacement) from the
        historical *trade_pnls* distribution.
        """
        if not trade_pnls:
            raise ValueError("trade_pnls must be a non-empty list")

        pnls = np.asarray(trade_pnls, dtype=np.float64)

        # Draw all random trades at once: shape (num_simulations, num_trades)
        indices = self._rng.integers(0, len(pnls), size=(num_simulations, num_trades))
        sampled_pnls = pnls[indices]

        # Build equity curves via cumulative sum
        cumulative = np.cumsum(sampled_pnls, axis=1)
        equity_curves = initial_balance + cumulative  # (num_simulations, num_trades)

        # Prepend initial balance column
        init_col = np.full((num_simulations, 1), initial_balance)
        full_curves = np.hstack([init_col, equity_curves])  # (sims, trades+1)

        # Final balances
        final_balances = full_curves[:, -1]

        # Max drawdown per simulation
        running_max = np.maximum.accumulate(full_curves, axis=1)
        drawdowns = (running_max - full_curves) / running_max
        max_drawdowns = np.max(drawdowns, axis=1)

        # Probability of ruin: fraction of sims that ever hit 50% drawdown
        ruin_threshold = 0.50
        probability_of_ruin = float(np.mean(max_drawdowns >= ruin_threshold))

        # Per-simulation returns
        total_returns = (final_balances - initial_balance) / initial_balance

        # Annualise: assume 250 trading days * 5 trades/day = 1250 trades/year
        trades_per_year = 250.0 * 5.0
        years_per_sim = num_trades / trades_per_year
        median_total_return = float(np.median(total_returns))
        if years_per_sim > 0:
            expected_annual_return = (1 + median_total_return) ** (1 / years_per_sim) - 1
        else:
            expected_annual_return = 0.0

        # Sharpe ratio (annualised)
        per_trade_returns = sampled_pnls / initial_balance
        mean_per_trade = float(np.mean(per_trade_returns))
        std_per_trade = float(np.std(per_trade_returns))
        if std_per_trade > 0:
            sharpe_ratio = (mean_per_trade / std_per_trade) * np.sqrt(trades_per_year)
        else:
            sharpe_ratio = 0.0

        # Store a subset of equity curves for plotting (cap at 200)
        max_curves = min(num_simulations, 200)
        curves_for_plot = full_curves[:max_curves].tolist()

        return MonteCarloResult(
            median_final_balance=float(np.median(final_balances)),
            p5_final_balance=float(np.percentile(final_balances, 5)),
            p95_final_balance=float(np.percentile(final_balances, 95)),
            probability_of_ruin=probability_of_ruin,
            max_drawdown_median=float(np.median(max_drawdowns)),
            max_drawdown_p95=float(np.percentile(max_drawdowns, 95)),
            expected_annual_return=expected_annual_return,
            sharpe_ratio=float(sharpe_ratio),
            equity_curves=curves_for_plot,
        )

    # ------------------------------------------------------------------
    # Analytical risk of ruin
    # ------------------------------------------------------------------

    @staticmethod
    def risk_of_ruin(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_drawdown_pct: float = 0.5,
    ) -> float:
        """
        Estimate the probability of hitting *max_drawdown_pct* drawdown
        using the analytical risk-of-ruin formula.

        Parameters
        ----------
        win_rate : float
            Probability of a winning trade (0-1).
        avg_win : float
            Average profit on a winning trade (positive value).
        avg_loss : float
            Average loss on a losing trade (positive value — magnitude).
        max_drawdown_pct : float
            Ruin threshold as a fraction of equity (default 0.50 = 50%).

        Returns
        -------
        float
            Probability of ruin (0-1).
        """
        if not (0 < win_rate < 1):
            raise ValueError("win_rate must be between 0 and 1 (exclusive)")
        avg_win = abs(avg_win)
        avg_loss = abs(avg_loss)
        if avg_win == 0 and avg_loss == 0:
            return 0.0

        loss_rate = 1.0 - win_rate

        # Edge expectancy
        edge = win_rate * avg_win - loss_rate * avg_loss
        if edge <= 0:
            # Negative or zero edge -> ruin is virtually certain
            return 1.0

        # Payoff ratio
        if avg_loss == 0:
            return 0.0
        payoff_ratio = avg_win / avg_loss

        # Risk-of-ruin approximation:
        #   RoR = ((1 - edge_pct) / (1 + edge_pct)) ^ units_at_risk
        # where edge_pct = (payoff_ratio * win_rate - loss_rate) / payoff_ratio
        #       units_at_risk ≈ max_drawdown expressed in avg_loss units
        edge_pct = (payoff_ratio * win_rate - loss_rate) / payoff_ratio
        if edge_pct <= 0:
            return 1.0
        if edge_pct >= 1:
            return 0.0

        # Units at risk: how many average losses fit in the drawdown budget
        # Normalise to fraction of a "1-unit" account where 1 unit = avg_loss
        # Since we don't know absolute account size, use the ratio.
        units = max_drawdown_pct / (avg_loss / (avg_win + avg_loss))
        base = (1 - edge_pct) / (1 + edge_pct)
        ror = base ** units
        return float(np.clip(ror, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Optimal position sizing via simulation
    # ------------------------------------------------------------------

    def optimal_position_size(
        self,
        trade_pnls: list[float],
        target_drawdown: float = 0.10,
        num_simulations: int = 1000,
        num_trades: int = 500,
    ) -> float:
        """
        Find the fraction of account equity to risk per trade such that
        the 95th-percentile max drawdown stays at or below *target_drawdown*.

        Uses a binary search over position-size fractions [0.001, 1.0].

        Returns
        -------
        float
            Optimal fraction of account to risk per trade (e.g. 0.02 = 2%).
        """
        if not trade_pnls:
            raise ValueError("trade_pnls must be a non-empty list")

        pnls = np.asarray(trade_pnls, dtype=np.float64)
        initial_balance = 100_000.0

        # Normalise PnLs to per-unit returns (assume they were generated at
        # some unknown position size; we treat them as return-per-dollar-risked).
        mean_abs = np.mean(np.abs(pnls))
        if mean_abs == 0:
            return 0.0
        normalised = pnls / mean_abs  # unit-variance-ish returns

        lo, hi = 0.001, 1.0

        for _ in range(30):  # binary search iterations
            mid = (lo + hi) / 2.0
            scaled = normalised * mid * initial_balance

            indices = self._rng.integers(0, len(scaled), size=(num_simulations, num_trades))
            sampled = scaled[indices]
            cumulative = np.cumsum(sampled, axis=1)
            equity = initial_balance + cumulative
            init_col = np.full((num_simulations, 1), initial_balance)
            full = np.hstack([init_col, equity])

            running_max = np.maximum.accumulate(full, axis=1)
            dd = (running_max - full) / running_max
            max_dd = np.max(dd, axis=1)
            p95_dd = float(np.percentile(max_dd, 95))

            if p95_dd > target_drawdown:
                hi = mid
            else:
                lo = mid

        return round((lo + hi) / 2.0, 4)

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(result: MonteCarloResult) -> str:
        """Return a human-readable text summary of a simulation result."""
        lines = [
            "=" * 56,
            "        MONTE CARLO SIMULATION REPORT",
            "=" * 56,
            "",
            "  Final Balance Distribution",
            "  --------------------------",
            f"    Median:          ${result.median_final_balance:>14,.2f}",
            f"    5th percentile:  ${result.p5_final_balance:>14,.2f}",
            f"    95th percentile: ${result.p95_final_balance:>14,.2f}",
            "",
            "  Risk Metrics",
            "  ------------",
            f"    Probability of ruin (50% DD): {result.probability_of_ruin:>8.2%}",
            f"    Max drawdown (median):        {result.max_drawdown_median:>8.2%}",
            f"    Max drawdown (95th pct):      {result.max_drawdown_p95:>8.2%}",
            "",
            "  Performance",
            "  -----------",
            f"    Expected annual return:       {result.expected_annual_return:>8.2%}",
            f"    Sharpe ratio (annualised):    {result.sharpe_ratio:>8.2f}",
            "",
            f"  Equity curves stored: {len(result.equity_curves)}",
            "=" * 56,
        ]
        return "\n".join(lines)
