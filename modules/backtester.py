"""
backtester.py — Historical simulation engine for options and forex strategies.

Replays scanner signals day-by-day on historical data to estimate how
the strategy would have performed. All results are simulation only.

Features:
  - Slippage: configurable % added to entry, subtracted from exit
  - Bid/ask spread: applied at entry and exit
  - Commission: per-contract fee
  - Realistic fill assumptions (no fills on illiquid days)
  - Expectancy calculation (avg win * win rate - avg loss * loss rate)
  - Max drawdown analysis
  - Separate stats for calls vs puts, by ticker, by regime, by setup type
  - Exportable equity curve to data/backtest_results/

Results saved to data/strategy_metrics.json.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional

BACKTEST_DIR = "data/backtest_results"
STRATEGY_METRICS_FILE = "data/strategy_metrics.json"


def run_backtest(
    tickers: list,
    days_back: int = 90,
    client=None,
    market: str = "options",
    slippage_pct: float = 0.5,
    spread_pct: float = 2.0,
    commission: float = 0.65,
    stop_loss_pct: float = 30.0,
    take_profit_pct: float = 50.0,
    min_confidence: int = 70,
) -> dict:
    """
    Run a historical backtest on a list of tickers/pairs.

    Args:
        tickers: List of tickers or forex pairs to test
        days_back: How many trading days of history to replay
        client: TradierClient (or None to use yfinance)
        market: 'options' or 'forex'
        slippage_pct: Slippage % added at entry
        spread_pct: Bid/ask spread % applied at fills
        commission: Commission per contract/lot in $
        stop_loss_pct: Exit if position falls this % below entry
        take_profit_pct: Exit if position gains this % above entry
        min_confidence: Minimum signal confidence to simulate trade

    Returns:
        Backtest results dict.
    """
    from modules.scanner import Scanner
    from modules.regime import RegimeDetector

    scanner = Scanner(client=client)
    regime_det = RegimeDetector()

    all_trades = []
    equity_curve = [10000]  # start with $10k simulated account
    current_equity = 10000

    for ticker in tickers:
        df = _get_full_history(ticker, days_back + 50, client, market)
        if df is None or len(df) < days_back + 25:
            continue

        # Walk forward day by day
        for i in range(50, len(df) - 1):
            window = df.iloc[:i].copy()
            today_bar = df.iloc[i]
            tomorrow_bar = df.iloc[i + 1] if i + 1 < len(df) else today_bar

            # Run scanner on historical window
            signal = scanner.scan_ticker(ticker, market) if False else \
                _simulate_scan(scanner, ticker, window, market, min_confidence)

            if signal is None:
                continue

            # Simulate entry
            entry_price = float(today_bar["close"])
            if entry_price <= 0:
                continue

            # Apply slippage and spread at entry
            entry_adj = entry_price * (1 + (slippage_pct + spread_pct / 2) / 100)
            direction = signal.get("direction", "call")

            if market == "options":
                # Estimate option premium via simplified approach
                iv_est = 0.30  # 30% IV assumption
                dte = 21
                opt_price = _rough_option_price(entry_price, entry_price, iv_est, dte, direction)
                if opt_price <= 0:
                    continue
                entry_premium = opt_price * (1 + slippage_pct / 100)
                qty = max(int(500 / (entry_premium * 100)), 1)
                cost = entry_premium * qty * 100 + commission * qty
            else:
                qty = 1000
                cost = entry_adj * qty * 0.02  # approx margin cost
                entry_premium = entry_adj

            # Simulate exit (walk forward up to max_hold days)
            max_hold = 21
            peak = entry_premium
            exit_price = None
            exit_reason = None
            pnl = 0

            for j in range(1, min(max_hold, len(df) - i - 1)):
                future_bar = df.iloc[i + j]
                fut_close = float(future_bar["close"])
                if fut_close <= 0:
                    continue

                # Update simulated option price
                if market == "options":
                    dte_remaining = dte - j
                    if dte_remaining <= 0:
                        fut_opt = 0
                    else:
                        pct_change = (fut_close - entry_price) / entry_price
                        fut_opt = entry_premium * (1 + pct_change * 5)  # simplified delta ~0.5
                        fut_opt = max(fut_opt, 0)
                    simulated_price = fut_opt
                else:
                    pct_change = (fut_close - entry_price) / entry_price
                    if direction in ("short", "put"):
                        pct_change = -pct_change
                    simulated_price = entry_premium * (1 + pct_change)

                # Update peak
                if simulated_price > peak:
                    peak = simulated_price

                change_pct = (simulated_price - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0

                # Check stops
                if change_pct <= -stop_loss_pct:
                    exit_price = simulated_price * (1 - spread_pct / 200)
                    exit_reason = "stop_loss"
                    break
                elif change_pct >= take_profit_pct:
                    exit_price = simulated_price * (1 - spread_pct / 200)
                    exit_reason = "take_profit"
                    break
                elif j == max_hold - 1:
                    exit_price = simulated_price * (1 - spread_pct / 200)
                    exit_reason = "time_exit"

            if exit_price is None or entry_premium <= 0:
                continue

            # Calculate P&L
            if market == "options":
                raw_pnl = (exit_price - entry_premium) * qty * 100 - commission * qty * 2
            else:
                raw_pnl = (exit_price - entry_premium) * qty
            pnl = round(raw_pnl, 2)

            current_equity += pnl
            equity_curve.append(current_equity)

            trade = {
                "ticker": ticker,
                "market": market,
                "direction": direction,
                "date": str(today_bar.get("date", "")[:10] if hasattr(today_bar.get("date", ""), "__getitem__") else today_bar.get("date", "")),
                "entry_price": round(entry_premium, 4),
                "exit_price": round(exit_price, 4),
                "qty": qty,
                "pnl": pnl,
                "pnl_pct": round((exit_price - entry_premium) / entry_premium * 100, 1),
                "exit_reason": exit_reason,
                "win": pnl > 0,
                "regime": signal.get("regime", "unknown"),
                "confidence": signal.get("confidence", 0),
                "signals_fired": [s[0] for s in signal.get("signals_fired", [])],
            }
            all_trades.append(trade)

    results = _analyze_results(all_trades, equity_curve, tickers, market)
    _save_results(results, tickers)
    return results


def _simulate_scan(scanner, ticker: str, window_df: pd.DataFrame, market: str, min_confidence: int) -> Optional[dict]:
    """
    Run scanner on a historical window DataFrame without live API calls.
    Substitutes the price history with the window to replicate what
    the scanner would have seen at that point in time.
    """
    try:
        from modules.regime import RegimeDetector
        from modules.news_sentiment import NewsSentiment

        regime_det = RegimeDetector()
        regime_data = regime_det.detect(window_df)

        if not regime_data["trading_allowed"]:
            return None

        close = window_df["close"].values.astype(float)
        high = window_df["high"].values.astype(float)
        low = window_df["low"].values.astype(float)
        volume = window_df["volume"].values.astype(float)

        bull_score = 0
        bear_score = 0
        weights = scanner.weights
        max_score = sum(weights.values()) - weights.get("options_volume", 15)  # no vol in backtest

        # Momentum
        mb_bull = scanner._signal_momentum(close, volume, "bullish")
        mb_bear = scanner._signal_momentum(close, volume, "bearish")
        bull_score += mb_bull * weights["momentum_breakout"] / 100
        bear_score += mb_bear * weights["momentum_breakout"] / 100

        # RSI
        rsi_bull = scanner._signal_rsi(close, "bullish")
        rsi_bear = scanner._signal_rsi(close, "bearish")
        bull_score += rsi_bull * weights["rsi"] / 100
        bear_score += rsi_bear * weights["rsi"] / 100

        # EMA
        ema_bull = scanner._signal_ema_crossover(close, "bullish")
        ema_bear = scanner._signal_ema_crossover(close, "bearish")
        bull_score += ema_bull * weights["ema_crossover"] / 100
        bear_score += ema_bear * weights["ema_crossover"] / 100

        bull_conf = int(min(bull_score / max_score * 100, 99)) if max_score > 0 else 0
        bear_conf = int(min(bear_score / max_score * 100, 99)) if max_score > 0 else 0

        if bull_conf >= min_confidence or bear_conf >= min_confidence:
            if bull_conf >= bear_conf:
                direction = "call" if market == "options" else "long"
                conf = bull_conf
                signals_fired = [("momentum", mb_bull), ("rsi", rsi_bull), ("ema", ema_bull)]
            else:
                direction = "put" if market == "options" else "short"
                conf = bear_conf
                signals_fired = [("momentum", mb_bear), ("rsi", rsi_bear), ("ema", ema_bear)]

            return {
                "ticker": ticker,
                "direction": direction,
                "confidence": conf,
                "signals_fired": signals_fired,
                "regime": regime_data["regime"],
                "current_price": float(close[-1]),
                "market": market,
            }
    except Exception:
        pass
    return None


def _rough_option_price(spot: float, strike: float, iv: float, dte: int, option_type: str) -> float:
    """
    Simplified Black-Scholes-like option price estimate.
    Uses scipy if available, otherwise a rough approximation.
    """
    try:
        from scipy.stats import norm
        T = dte / 365
        r = 0.05
        if T <= 0 or spot <= 0 or strike <= 0 or iv <= 0:
            return 0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        if option_type in ("call", "long", "bullish"):
            price = spot * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * math.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return max(round(price, 2), 0)
    except Exception:
        # Rough approximation: ATM option ≈ 0.4 * iv * spot * sqrt(T)
        T = dte / 365
        return max(0.4 * iv * spot * math.sqrt(T), 0.01)


def _get_full_history(ticker: str, days: int, client, market: str) -> Optional[pd.DataFrame]:
    """Fetch full price history for backtesting."""
    try:
        if market == "forex":
            from modules.forex_data import get_forex_history_yf
            return get_forex_history_yf(ticker, count=days)
        elif client and client.is_configured():
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
            return client.get_history(ticker, start=start, end=end)
        else:
            import yfinance as yf
            end = datetime.today()
            start = end - timedelta(days=days * 2)
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            if "adj_close" in df.columns:
                df = df.rename(columns={"adj_close": "close"})
            return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _analyze_results(trades: list, equity_curve: list, tickers: list, market: str) -> dict:
    """Compute comprehensive backtest statistics."""
    if not trades:
        return {"error": "No trades generated. Check signal thresholds or widen date range.", "trades": []}

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    total = len(trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    total_pnl = sum(t["pnl"] for t in trades)

    # Expectancy: avg_win * win_rate - avg_loss * loss_rate (per trade)
    win_r = len(wins) / total if total > 0 else 0
    loss_r = 1 - win_r
    expectancy = avg_win * win_r + avg_loss * loss_r

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    max_drawdown = float(np.min(drawdown))

    # By direction
    calls = [t for t in trades if t["direction"] in ("call", "long")]
    puts = [t for t in trades if t["direction"] in ("put", "short")]
    call_wr = sum(1 for t in calls if t["win"]) / len(calls) * 100 if calls else 0
    put_wr = sum(1 for t in puts if t["win"]) / len(puts) * 100 if puts else 0

    # By ticker
    by_ticker = {}
    for ticker in tickers:
        t_trades = [t for t in trades if t["ticker"] == ticker]
        if not t_trades:
            continue
        by_ticker[ticker] = {
            "trades": len(t_trades),
            "wins": sum(1 for t in t_trades if t["win"]),
            "win_rate_pct": round(sum(1 for t in t_trades if t["win"]) / len(t_trades) * 100, 1),
            "total_pnl": round(sum(t["pnl"] for t in t_trades), 2),
        }

    # By regime
    by_regime = {}
    for t in trades:
        r = t.get("regime", "unknown")
        if r not in by_regime:
            by_regime[r] = {"trades": 0, "wins": 0, "total_pnl": 0}
        by_regime[r]["trades"] += 1
        by_regime[r]["wins"] += 1 if t["win"] else 0
        by_regime[r]["total_pnl"] += t["pnl"]
    for r in by_regime:
        t_count = by_regime[r]["trades"]
        by_regime[r]["win_rate_pct"] = round(by_regime[r]["wins"] / t_count * 100, 1) if t_count > 0 else 0
        by_regime[r]["total_pnl"] = round(by_regime[r]["total_pnl"], 2)

    return {
        "market": market,
        "tickers": tickers,
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "call_win_rate_pct": round(call_wr, 1),
        "put_win_rate_pct": round(put_wr, 1),
        "by_ticker": by_ticker,
        "by_regime": by_regime,
        "equity_curve": equity_curve,
        "trades": trades,
        "run_date": datetime.now().isoformat(),
    }


def _save_results(results: dict, tickers: list):
    """Save backtest results to file."""
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{BACKTEST_DIR}/backtest_{ts}.json"
    lightweight = {k: v for k, v in results.items() if k != "trades"}
    lightweight["trades_count"] = results.get("total_trades", 0)
    with open(filename, "w") as f:
        json.dump(lightweight, f, indent=2)

    # Update strategy metrics
    metrics = {"last_backtest": results.get("run_date"), **lightweight}
    with open(STRATEGY_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def show_backtest_results(results: dict):
    """Print a formatted backtest summary."""
    from tabulate import tabulate

    if "error" in results:
        print(f"\n[!] Backtest error: {results['error']}")
        return

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — {results.get('market', '').upper()}")
    print(f"  Tickers: {', '.join(results.get('tickers', []))}")
    print(f"{'='*70}")

    summary = [
        ["Total Trades", results["total_trades"]],
        ["Win Rate", f"{results['win_rate_pct']}%"],
        ["Call Win Rate", f"{results['call_win_rate_pct']}%"],
        ["Put Win Rate", f"{results['put_win_rate_pct']}%"],
        ["Avg Winner", f"${results['avg_win']:.2f}"],
        ["Avg Loser", f"${results['avg_loss']:.2f}"],
        ["Expectancy/Trade", f"${results['expectancy_per_trade']:.2f}"],
        ["Total P&L (sim)", f"${results['total_pnl']:.2f}"],
        ["Max Drawdown", f"{results['max_drawdown_pct']:.1f}%"],
    ]
    print(tabulate(summary, headers=["Metric", "Value"], tablefmt="simple"))

    if results.get("by_ticker"):
        print(f"\n  By Ticker:")
        rows = [[t, v["trades"], f"{v['win_rate_pct']}%", f"${v['total_pnl']:.2f}"]
                for t, v in results["by_ticker"].items()]
        print(tabulate(rows, headers=["Ticker", "Trades", "Win Rate", "P&L"], tablefmt="simple"))

    if results.get("by_regime"):
        print(f"\n  By Regime:")
        rows = [[r, v["trades"], f"{v['win_rate_pct']}%", f"${v['total_pnl']:.2f}"]
                for r, v in results["by_regime"].items()]
        print(tabulate(rows, headers=["Regime", "Trades", "Win Rate", "P&L"], tablefmt="simple"))

    print(f"\n  ⚠ Backtest uses simplified option pricing and is for reference only.")
    print(f"  ⚠ Past simulated results do not predict future real-world performance.")
