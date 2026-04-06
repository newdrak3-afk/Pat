#!/usr/bin/env python3
"""
Multi-Agent Trading System v2 — Entry Point

Commands:
    python trading_main.py scan          # Run market scanner only
    python trading_main.py research      # Scan + research
    python trading_main.py run           # Run one full cycle
    python trading_main.py live          # Run continuously (dry run)
    python trading_main.py auto          # Auto-trade forex on OANDA practice
    python trading_main.py status        # Show auto trader status
    python trading_main.py dashboard     # Show system dashboard
    python trading_main.py resolve       # Resolve a trade (win/loss)
    python trading_main.py lessons       # View lessons learned
    python trading_main.py config        # Show current config
    python trading_main.py backtest      # Run historical backtest
    python trading_main.py settings      # View/change feature toggles
    python trading_main.py report        # Full system report with DB stats
    python trading_main.py montecarlo    # Run Monte Carlo risk simulation
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from trading.config import SystemConfig
from trading.orchestrator import Orchestrator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_scan(orch: Orchestrator):
    """Run the scanner only."""
    print("\n--- SCANNING MARKETS ---\n")
    markets = orch.scanner.scan()
    print(orch.scanner.get_summary(markets))
    print(f"\nTotal: {len(markets)} markets passing filters")


def cmd_research(orch: Orchestrator):
    """Scan + research top markets."""
    print("\n--- SCAN + RESEARCH ---\n")
    markets = orch.scanner.scan()
    print(orch.scanner.get_summary(markets))

    top = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:5]
    if not top:
        print("No markets to research.")
        return

    print(f"\nResearching top {len(top)} markets...\n")
    results = orch.researcher.research_markets_parallel(top)

    for r in results:
        print(f"\n{'─' * 50}")
        print(r.narrative_summary if hasattr(r, 'narrative_summary') else str(r))
        print(f"{'─' * 50}")


def cmd_run(orch: Orchestrator):
    """Run one full trading cycle."""
    print("\n--- RUNNING FULL CYCLE ---\n")
    summary = orch.run_cycle()
    print(f"\n--- CYCLE SUMMARY ---")
    print(json.dumps(summary, indent=2))


def cmd_live(orch: Orchestrator):
    """Run continuously in dry-run mode."""
    orch.dry_run = True
    print("\n--- STARTING CONTINUOUS TRADING (DRY RUN) ---")
    print("Press Ctrl+C to stop.\n")
    orch.run_continuous()


def cmd_dashboard(orch: Orchestrator):
    """Show the system dashboard."""
    print(orch.get_dashboard())


def cmd_resolve(orch: Orchestrator):
    """Manually resolve a trade."""
    trade_id = input("Trade ID: ").strip()
    outcome = input("Outcome (win/loss): ").strip().lower()

    if outcome not in ("win", "loss"):
        print("[!] Invalid outcome. Use 'win' or 'loss'.")
        return

    orch.resolve_trade(trade_id, outcome)
    print(f"[+] Trade {trade_id} resolved as {outcome}")

    if outcome == "loss":
        print("\n--- LOSS ANALYSIS ---")
        print(orch.loss_analyzer.get_lessons_summary())


def cmd_lessons(orch: Orchestrator):
    """View all lessons learned."""
    print(orch.loss_analyzer.get_lessons_summary())


def cmd_config(orch: Orchestrator):
    """Show current configuration."""
    from dataclasses import asdict
    config = asdict(orch.config)
    print(json.dumps(config, indent=2, default=str))


def cmd_backtest(orch: Orchestrator):
    """Run a historical backtest using the backtester engine."""
    from trading.backtester import BacktestEngine
    from trading.brokers.oanda import OandaBroker

    print("\n--- BACKTEST ENGINE ---")

    broker = OandaBroker()
    if not broker.connect():
        print("[!] Could not connect to OANDA for historical data")
        print("Running demo backtest with simulated data...\n")

        # Demo with random candles
        import numpy as np
        np.random.seed(42)
        price = 1.1000
        candles = []
        for i in range(500):
            change = np.random.normal(0, 0.0010)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, 0.0005))
            l = min(o, c) - abs(np.random.normal(0, 0.0005))
            candles.append({
                "time": f"2024-01-{(i//24)+1:02d}T{i%24:02d}:00:00Z",
                "open": o, "high": h, "low": l, "close": c,
                "volume": int(np.random.uniform(100, 1000)),
            })
            price = c

        # Generate some test signals
        signals = []
        for i in range(20, 480, 25):
            side = "buy" if candles[i]["close"] > candles[i-5]["close"] else "sell"
            signals.append({
                "bar_index": i,
                "symbol": "EUR_USD",
                "side": side,
                "sl_pips": 30,
                "tp_pips": 45,
                "confidence": 0.6,
            })

        engine = BacktestEngine()
        result = engine.run_backtest(candles, signals)
        print(BacktestEngine.print_report(result))
        return

    print("Fetching 1000 H1 candles for EUR_USD...")
    symbol = "EUR_USD"
    candles = broker.get_candles(symbol, "H1", 1000)

    if not candles or len(candles) < 100:
        print("[!] Not enough candle data")
        return

    # Generate signals using the forex scanner logic
    from trading.forex_scanner import ForexScanner
    scanner = ForexScanner(broker)

    # Simple signal generation from candles
    signals = []
    for i in range(50, len(candles) - 10, 20):
        subset = candles[max(0, i-100):i]
        closes = [c["close"] for c in subset]
        if len(closes) < 20:
            continue

        sma20 = sum(closes[-20:]) / 20
        current = closes[-1]
        side = "buy" if current > sma20 else "sell"

        signals.append({
            "bar_index": i,
            "symbol": symbol,
            "side": side,
            "sl_pips": 30,
            "tp_pips": 45,
            "confidence": 0.55,
        })

    if not signals:
        print("[!] No signals generated")
        return

    print(f"Generated {len(signals)} signals over {len(candles)} bars\n")

    engine = BacktestEngine()
    result = engine.run_backtest(candles, signals)
    print(BacktestEngine.print_report(result))


def cmd_auto(orch: Orchestrator):
    """Run auto-trader on OANDA practice + Alpaca options."""
    import threading
    from trading.auto_trader import AutoTrader

    print("\n--- AUTO TRADER v3 (OANDA + Alpaca Options) ---")
    print("Forex: OANDA practice | Options: Alpaca paper")
    print("HTF trend filter | Guard engine | Session awareness")
    print("No real money. The bot learns from every loss.\n")

    trader = AutoTrader(orch.config)

    # Start options trader in background if Alpaca is configured
    alpaca_key = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if alpaca_key:
        from trading.options_trader import OptionsTrader
        import time as _time

        logger_main = logging.getLogger(__name__)
        logger_main.info(f"Alpaca key found: {alpaca_key[:8]}... secret={'YES' if alpaca_secret else 'MISSING'}")
        trader.notifier.send_system_alert(
            f"OPTIONS INIT: Alpaca key found ({alpaca_key[:8]}...)\n"
            f"Secret: {'set' if alpaca_secret else 'MISSING!'}\n"
            f"Starting options thread..."
        )

        options = OptionsTrader(orch.config)
        # Wire options trader into telegram bot AND auto trader for /status
        trader.telegram_bot._options_trader = options
        trader.options_trader = options

        def _run_options_with_restart():
            """Run options trader with auto-restart on crash."""
            restarts = 0
            while True:
                try:
                    options.start()
                    break  # Clean exit (keyboard interrupt)
                except Exception as e:
                    restarts += 1
                    logging.getLogger(__name__).error(
                        f"Options thread crashed (restart #{restarts}): {e}",
                        exc_info=True,
                    )
                    trader.notifier.send_system_alert(
                        f"OPTIONS CRASHED (restart #{restarts}): {str(e)[:200]}"
                    )
                    if restarts >= 5:
                        trader.notifier.send_system_alert(
                            "OPTIONS: Too many crashes, giving up. Check logs."
                        )
                        break
                    _time.sleep(30)  # Wait before restart

        options_thread = threading.Thread(target=_run_options_with_restart, daemon=True)
        options_thread.start()
        print("Options trader started (Alpaca) with auto-restart")
    else:
        print("Options: disabled (no ALPACA_API_KEY)")
        trader.notifier.send_system_alert(
            "OPTIONS DISABLED: No ALPACA_API_KEY env var found.\n"
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in Railway to enable options."
        )

    print(trader.get_status())
    trader.start(scan_interval=orch.config.scan_interval_seconds)


def cmd_status(orch: Orchestrator):
    """Show auto trader status."""
    from trading.auto_trader import AutoTrader
    trader = AutoTrader(orch.config)
    if trader.oanda.connect():
        print(trader.get_full_report())
    else:
        print("[!] Could not connect to OANDA")
        # Still show what we can from local state
        print(trader.get_status())


def cmd_settings(orch: Orchestrator):
    """View or change feature toggles."""
    from trading.settings import Settings

    settings = Settings()

    if len(sys.argv) > 2:
        # Set a value: python trading_main.py settings key=value
        for arg in sys.argv[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Parse value
                if value.lower() in ("true", "on", "yes", "1"):
                    parsed = True
                elif value.lower() in ("false", "off", "no", "0"):
                    parsed = False
                elif value.isdigit():
                    parsed = int(value)
                else:
                    parsed = value

                if settings.set(key, parsed):
                    print(f"  Set {key} = {parsed}")
                else:
                    print(f"  [!] Unknown setting: {key}")
            else:
                print(f"  [!] Use format: key=value")
    else:
        print(settings.get_status())


def cmd_report(orch: Orchestrator):
    """Full system report with DB stats."""
    from trading.auto_trader import AutoTrader
    from trading.trade_db import TradeDB
    from trading.drawdown_guard import DrawdownGuard
    from trading.drift_detector import DriftDetector
    from trading.calibration import CalibrationLayer

    db = TradeDB()
    drawdown = DrawdownGuard()
    drift = DriftDetector()
    calibration = CalibrationLayer()

    print("\n╔══════════════════════════════════════╗")
    print("║     FULL SYSTEM REPORT               ║")
    print("╚══════════════════════════════════════╝\n")

    # DB summary
    summary = db.get_performance_summary()
    if summary:
        print("  ── Trade Database ──")
        print(f"  Total Trades:  {summary.get('total_trades', 0)}")
        print(f"  Win Rate:      {summary.get('win_rate', 0):.0%}")
        print(f"  Total PnL:     ${summary.get('total_pnl', 0):+,.2f}")
        print()

    # Symbol breakdown
    pnl_by_symbol = db.get_pnl_by_symbol()
    if pnl_by_symbol:
        print("  ── PnL by Symbol ──")
        for sym, pnl in sorted(pnl_by_symbol.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sym:12s}  ${pnl:+,.2f}")
        print()

    # Drawdown
    print(drawdown.get_status())
    print()

    # Drift
    print(drift.get_status())
    print()

    # Calibration
    cal_stats = calibration.get_calibration_stats()
    if cal_stats.get("total_samples", 0) > 0:
        print("  ── Calibration ──")
        print(f"  Samples:       {cal_stats['total_samples']}")
        print(f"  Brier Score:   {cal_stats.get('brier_score', 0):.4f}")
        print(f"  Overconfident: {'Yes' if calibration.is_overconfident() else 'No'}")
        print()


def cmd_montecarlo(orch: Orchestrator):
    """Run Monte Carlo risk simulation."""
    from trading.monte_carlo import MonteCarloSimulator
    from trading.trade_db import TradeDB

    db = TradeDB()
    print("\n--- MONTE CARLO SIMULATION ---\n")

    # Get historical PnLs from DB
    trades = db.get_all_trades()
    pnls = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) != 0]

    if len(pnls) < 5:
        print("Not enough trade history. Using demo data...\n")
        import numpy as np
        np.random.seed(42)
        pnls = list(np.random.normal(5, 20, 50))  # avg $5 win, $20 std dev

    sim = MonteCarloSimulator()
    result = sim.simulate(pnls, num_simulations=1000, num_trades=200)
    print(sim.print_report(result))


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MULTI-AGENT TRADING SYSTEM v2                       ║
║          Scan → Guard → Trade → Monitor → Learn              ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  scan        Scan 300+ markets, filter & flag
  research    Scan + parallel research (Twitter/Reddit/RSS)
  run         Run one full trading cycle
  live        Run continuously (dry run, Ctrl+C to stop)
  auto        Auto-trade forex on OANDA (with all guards)
  status      Show auto trader stats (wins/losses/PnL)
  dashboard   View system dashboard & stats
  resolve     Resolve a trade as win/loss
  lessons     View lessons learned from losses
  config      Show current configuration
  backtest    Run historical backtest simulation
  settings    View/change feature toggles
  report      Full system report (DB + guards + calibration)
  montecarlo  Run Monte Carlo risk simulation

Settings:
  settings scanning_enabled=false    Turn off scanning
  settings auto_trading_enabled=true Turn on auto-trading
  settings max_trades_per_cycle=2    Limit trades per scan

Guards (all auto-enabled):
  1. Regime Detector  — Skip unfavorable market conditions
  2. Data Quality     — Reject bad/stale candle data
  3. Portfolio Mgr    — Cap currency exposure
  4. Drawdown Guard   — Hard stop on max drawdown
  5. Drift Detector   — Pause if strategy degrading
  6. Slippage Model   — Account for real execution costs
  7. Calibration      — Fix overconfident predictions
  8. Risk Manager     — Kelly sizing, daily limits
""")


COMMANDS = {
    "scan": cmd_scan,
    "research": cmd_research,
    "run": cmd_run,
    "live": cmd_live,
    "auto": cmd_auto,
    "status": cmd_status,
    "dashboard": cmd_dashboard,
    "resolve": cmd_resolve,
    "lessons": cmd_lessons,
    "config": cmd_config,
    "backtest": cmd_backtest,
    "settings": cmd_settings,
    "report": cmd_report,
    "montecarlo": cmd_montecarlo,
}


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    setup_logging(verbose)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd in ("help", "--help", "-h") or cmd not in COMMANDS:
        print_help()
    else:
        config = SystemConfig()
        orch = Orchestrator(config)
        COMMANDS[cmd](orch)
