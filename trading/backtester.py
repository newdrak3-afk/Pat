"""
Backtester — Realistic historical simulation engine.

Simulates trades over historical candle data with:
- Spread and slippage simulation
- Stop loss / take profit execution
- Equity curve tracking
- Walk-forward validation
- Full performance metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_bars: float = 0.0
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    initial_balance: float = 0.0
    final_balance: float = 0.0


class BacktestEngine:
    """
    Simulates trading over historical candle data.

    Handles order execution with spread, slippage, SL, and TP.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        commission_per_lot: float = 0.0,
    ):
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot

    def run_backtest(
        self,
        candles: list[dict],
        signals: list[dict],
        risk_pct: float = 0.02,
    ) -> BacktestResult:
        """
        Run a backtest over historical candles with given signals.

        Args:
            candles: List of dicts with time, open, high, low, close, volume
            signals: List of dicts with bar_index, side, sl_pips, tp_pips, confidence
            risk_pct: Fraction of account to risk per trade

        Returns:
            BacktestResult with full metrics
        """
        balance = self.initial_balance
        equity_curve = [balance]
        trade_log = []
        open_trades = []

        # Build signal lookup by bar index
        signal_map = {}
        for s in signals:
            idx = s.get("bar_index", 0)
            signal_map[idx] = s

        # Get pip value
        symbol = signals[0].get("symbol", "EUR_USD") if signals else "EUR_USD"
        pip = 0.01 if "JPY" in symbol else 0.0001

        for i in range(1, len(candles)):
            candle = candles[i]
            high = candle["high"]
            low = candle["low"]
            close_price = candle["close"]
            open_price = candle["open"]

            # Check open trades for SL/TP hits
            closed_this_bar = []
            for trade in open_trades:
                hit = self._check_sl_tp(trade, high, low)
                if hit:
                    trade["exit_price"] = hit["price"]
                    trade["outcome"] = hit["type"]
                    trade["exit_bar"] = i
                    trade["duration_bars"] = i - trade["entry_bar"]

                    if trade["side"] == "buy":
                        pnl_pips = (trade["exit_price"] - trade["entry_price"]) / pip
                    else:
                        pnl_pips = (trade["entry_price"] - trade["exit_price"]) / pip

                    # Subtract costs
                    pnl_pips -= self.spread_pips + self.slippage_pips
                    trade["pnl_pips"] = pnl_pips
                    trade["pnl"] = pnl_pips * pip * trade["units"]

                    balance += trade["pnl"]
                    trade_log.append(trade)
                    closed_this_bar.append(trade)

            for t in closed_this_bar:
                open_trades.remove(t)

            # Check for new signal at this bar
            if i in signal_map and len(open_trades) < 5:
                signal = signal_map[i]
                side = signal.get("side", "buy")
                sl_pips = signal.get("sl_pips", 30)
                tp_pips = signal.get("tp_pips", 45)

                # Entry with spread + slippage
                spread_cost = self.spread_pips * pip
                slip_cost = self.slippage_pips * pip

                if side == "buy":
                    entry = open_price + spread_cost + slip_cost
                    sl = entry - sl_pips * pip
                    tp = entry + tp_pips * pip
                else:
                    entry = open_price - spread_cost - slip_cost
                    sl = entry + sl_pips * pip
                    tp = entry - tp_pips * pip

                # Position sizing: risk X% of balance
                risk_amount = balance * risk_pct
                sl_distance = abs(entry - sl)
                if sl_distance > 0:
                    units = int(risk_amount / sl_distance)
                    units = max(1, min(units, 100_000))
                else:
                    units = 1000

                trade = {
                    "entry_bar": i,
                    "side": side,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "units": units,
                    "confidence": signal.get("confidence", 0.5),
                    "time": candle.get("time", ""),
                }
                open_trades.append(trade)

            equity_curve.append(balance)

        # Close any remaining open trades at last close
        if open_trades and candles:
            last_close = candles[-1]["close"]
            for trade in open_trades:
                if trade["side"] == "buy":
                    pnl_pips = (last_close - trade["entry_price"]) / pip
                else:
                    pnl_pips = (trade["entry_price"] - last_close) / pip
                pnl_pips -= self.spread_pips + self.slippage_pips
                trade["pnl_pips"] = pnl_pips
                trade["pnl"] = pnl_pips * pip * trade["units"]
                trade["exit_price"] = last_close
                trade["outcome"] = "win" if pnl_pips > 0 else "loss"
                trade["exit_bar"] = len(candles) - 1
                trade["duration_bars"] = trade["exit_bar"] - trade["entry_bar"]
                balance += trade["pnl"]
                trade_log.append(trade)

        return self._calculate_metrics(
            trade_log, equity_curve, self.initial_balance, balance
        )

    def _check_sl_tp(self, trade: dict, high: float, low: float) -> Optional[dict]:
        """Check if a candle's high/low hit the SL or TP."""
        sl = trade["stop_loss"]
        tp = trade["take_profit"]

        if trade["side"] == "buy":
            if low <= sl:
                return {"type": "loss", "price": sl}
            if high >= tp:
                return {"type": "win", "price": tp}
        else:
            if high >= sl:
                return {"type": "loss", "price": sl}
            if low <= tp:
                return {"type": "win", "price": tp}

        return None

    def _calculate_metrics(
        self,
        trade_log: list,
        equity_curve: list,
        initial: float,
        final: float,
    ) -> BacktestResult:
        """Calculate all performance metrics from trade log."""
        result = BacktestResult(
            initial_balance=initial,
            final_balance=final,
            equity_curve=equity_curve,
            trade_log=trade_log,
            total_trades=len(trade_log),
        )

        if not trade_log:
            return result

        pnls = [t["pnl"] for t in trade_log]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.wins = len(wins)
        result.losses = len(losses)
        result.win_rate = result.wins / result.total_trades if result.total_trades else 0
        result.total_pnl = sum(pnls)
        result.avg_win = np.mean(wins) if wins else 0.0
        result.avg_loss = np.mean(losses) if losses else 0.0
        result.largest_win = max(wins) if wins else 0.0
        result.largest_loss = min(losses) if losses else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        result.expectancy = np.mean(pnls)

        # Sharpe ratio (annualized, assuming daily)
        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            result.sharpe_ratio = (
                np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)
                if np.std(pnl_arr) > 0
                else 0.0
            )

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = peak - eq
        result.max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0
        result.max_drawdown_pct = (
            result.max_drawdown / float(np.max(peak)) * 100
            if float(np.max(peak)) > 0
            else 0
        )

        # Avg trade duration
        durations = [t.get("duration_bars", 0) for t in trade_log]
        result.avg_trade_duration_bars = np.mean(durations) if durations else 0

        return result

    def walk_forward(
        self,
        candles: list[dict],
        signal_generator,
        window_size: int = 500,
        step_size: int = 100,
        risk_pct: float = 0.02,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation: slide a window over data,
        generate signals on train set, test on out-of-sample.

        Args:
            candles: Full historical candle data
            signal_generator: Callable(candles) -> list[signals]
            window_size: Bars in training window
            step_size: Bars to step forward
            risk_pct: Risk per trade

        Returns:
            List of BacktestResult for each out-of-sample window
        """
        results = []
        total = len(candles)

        i = 0
        while i + window_size + step_size <= total:
            train_candles = candles[i : i + window_size]
            test_candles = candles[i + window_size : i + window_size + step_size]

            # Generate signals from training data
            signals = signal_generator(train_candles)

            # Remap signal bar_index to test window
            test_signals = []
            for s in signals:
                s_copy = dict(s)
                s_copy["bar_index"] = s_copy.get("bar_index", 0) % step_size
                if s_copy["bar_index"] < len(test_candles):
                    test_signals.append(s_copy)

            if test_signals:
                result = self.run_backtest(test_candles, test_signals, risk_pct)
                results.append(result)

            i += step_size

        return results

    @staticmethod
    def print_report(result: BacktestResult) -> str:
        """Generate a formatted text report."""
        return (
            "╔══════════════════════════════════════╗\n"
            "║        BACKTEST RESULTS              ║\n"
            "╚══════════════════════════════════════╝\n"
            "\n"
            f"  Initial Balance:  ${result.initial_balance:,.2f}\n"
            f"  Final Balance:    ${result.final_balance:,.2f}\n"
            f"  Total PnL:        ${result.total_pnl:+,.2f}\n"
            f"  Return:           {(result.final_balance/result.initial_balance - 1)*100:+.2f}%\n"
            "\n"
            f"  Total Trades:     {result.total_trades}\n"
            f"  Wins:             {result.wins}\n"
            f"  Losses:           {result.losses}\n"
            f"  Win Rate:         {result.win_rate*100:.1f}%\n"
            "\n"
            f"  Avg Win:          ${result.avg_win:+,.2f}\n"
            f"  Avg Loss:         ${result.avg_loss:+,.2f}\n"
            f"  Largest Win:      ${result.largest_win:+,.2f}\n"
            f"  Largest Loss:     ${result.largest_loss:+,.2f}\n"
            "\n"
            f"  Profit Factor:    {result.profit_factor:.2f}\n"
            f"  Expectancy:       ${result.expectancy:+,.2f}\n"
            f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}\n"
            "\n"
            f"  Max Drawdown:     ${result.max_drawdown:,.2f}\n"
            f"  Max Drawdown %:   {result.max_drawdown_pct:.2f}%\n"
            f"  Avg Duration:     {result.avg_trade_duration_bars:.0f} bars\n"
        )
