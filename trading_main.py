#!/usr/bin/env python3
"""
Multi-Agent Trading System — Entry Point

Commands:
    python trading_main.py scan          # Run market scanner only
    python trading_main.py research      # Scan + research
    python trading_main.py run           # Run one full cycle
    python trading_main.py live          # Run continuously (dry run)
    python trading_main.py dashboard     # Show system dashboard
    python trading_main.py resolve       # Resolve a trade (win/loss)
    python trading_main.py lessons       # View lessons learned
    python trading_main.py config        # Show current config
    python trading_main.py backtest      # Run a simulated backtest
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

    # Research top 5 by volume
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
    """Run a simple backtest simulation."""
    print("\n--- BACKTEST MODE ---")
    print("Running 5 simulated cycles...\n")

    orch.dry_run = True
    total_trades = 0
    total_blocked = 0

    for i in range(5):
        print(f"\n--- Cycle {i+1}/5 ---")
        summary = orch.run_cycle()
        total_trades += summary["trades_approved"]
        total_blocked += summary["trades_blocked"]

    print(f"\n--- BACKTEST COMPLETE ---")
    print(f"Total trades: {total_trades}")
    print(f"Total blocked: {total_blocked}")
    print(orch.get_dashboard())


def cmd_auto(orch: Orchestrator):
    """Run auto-trader on OANDA practice account."""
    from trading.auto_trader import AutoTrader

    print("\n--- AUTO TRADER (OANDA Practice) ---")
    print("This will trade REAL forex on your PRACTICE account.")
    print("No real money. The bot learns from every loss.\n")

    trader = AutoTrader(orch.config)
    print(trader.get_status())
    trader.start(scan_interval=orch.config.scan_interval_seconds)


def cmd_status_auto(orch: Orchestrator):
    """Show auto trader status."""
    from trading.auto_trader import AutoTrader
    trader = AutoTrader(orch.config)
    if trader.oanda.connect():
        print(trader.get_status())
    else:
        print("[!] Could not connect to OANDA")


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MULTI-AGENT TRADING SYSTEM                          ║
║          Scan → Research → Predict → Trade → Learn           ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  python trading_main.py scan        Scan 300+ markets, filter & flag
  python trading_main.py research    Scan + parallel research (Twitter/Reddit/RSS)
  python trading_main.py run         Run one full trading cycle
  python trading_main.py live        Run continuously (dry run, Ctrl+C to stop)
  python trading_main.py auto        Auto-trade forex on OANDA practice account
  python trading_main.py status      Show auto trader stats (wins/losses/PnL)
  python trading_main.py dashboard   View system dashboard & stats
  python trading_main.py resolve     Resolve a trade as win/loss
  python trading_main.py lessons     View lessons learned from losses
  python trading_main.py config      Show current configuration
  python trading_main.py backtest    Run simulated backtest (5 cycles)

Agents:
  1. Scan Agent      — Filters markets by liquidity, volume, spread
  2. Research Agent   — Scrapes Twitter, Reddit, RSS in parallel
  3. Prediction Agent — XGBoost + LLM probability calibration
  4. Risk Manager     — Kelly sizing, daily limits, position limits
  5. Loss Analyzer    — Post-loss analysis, generates corrective rules

Brokers:
  - OANDA  — Forex (auto-trade on practice account)
  - Robinhood — Stocks & options (alert-only mode)

Setup:
  1. Copy .env.example and add your API keys
  2. pip install -r requirements.txt
  3. python trading_main.py scan   (test the scanner)
  4. python trading_main.py auto   (auto-trade forex on practice)
""")


COMMANDS = {
    "scan": cmd_scan,
    "research": cmd_research,
    "run": cmd_run,
    "live": cmd_live,
    "auto": cmd_auto,
    "status": cmd_status_auto,
    "dashboard": cmd_dashboard,
    "resolve": cmd_resolve,
    "lessons": cmd_lessons,
    "config": cmd_config,
    "backtest": cmd_backtest,
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
