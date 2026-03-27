"""
portfolio.py — Position tracking and P&L display for options and forex.

Stores positions in data/positions.json.
Displays a color-coded table of open/closed positions with live P&L.
Tracks unrealized + realized P&L and overall win rate.
"""

import os
import json
from datetime import datetime, date
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

POSITIONS_FILE = "data/positions.json"


def load_positions() -> list:
    """Load all positions from JSON file."""
    if not os.path.exists(POSITIONS_FILE):
        return []
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_positions(positions: list):
    """Save positions list to JSON file."""
    os.makedirs("data", exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def add_position(order_result: dict, signal: dict):
    """
    Add a new position to the portfolio after a trade is opened.

    Args:
        order_result: Dict returned by Trader.execute_signal()
        signal: Original signal dict from Scanner
    """
    positions = load_positions()
    pos = {
        "id": f"{order_result.get('ticker', '')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "ticker": order_result.get("ticker") or order_result.get("pair", ""),
        "market": order_result.get("market", "options"),
        "direction": order_result.get("direction", ""),
        "option_symbol": order_result.get("option_symbol", ""),
        "suggested_expiry": signal.get("suggested_expiry", ""),
        "strike": signal.get("suggested_strike", ""),
        "qty": order_result.get("qty") or order_result.get("units", 1),
        "fill_price": order_result.get("fill_price", 0),
        "entry_price": order_result.get("fill_price", 0),
        "current_price": order_result.get("fill_price", 0),
        "peak_price": order_result.get("fill_price", 0),
        "cost_basis": order_result.get("cost_basis", 0),
        "status": "open",
        "pnl": 0,
        "pnl_pct": 0,
        "signal_confidence": signal.get("confidence", 0),
        "regime": signal.get("regime", ""),
        "paper": order_result.get("paper", True),
        "opened_at": datetime.now().isoformat(),
        "exit_price": None,
        "exit_reason": None,
        "exit_time": None,
    }
    positions.append(pos)
    save_positions(positions)
    return pos


def update_position_prices(client=None, forex_client=None):
    """
    Refresh current_price and unrealized P&L for all open positions.
    Optionally updates peak_price for trailing stop tracking.
    """
    positions = load_positions()
    changed = False
    for pos in positions:
        if pos.get("status") != "open":
            continue
        market = pos.get("market", "options")
        ticker = pos.get("ticker", "")
        current = None

        try:
            if market == "forex" and forex_client and forex_client.is_configured():
                q = forex_client.get_quote(ticker)
                current = q["mid"] if q else None
            elif client and client.is_configured():
                q = client.get_quote(ticker)
                current = q["last"] if q else None
        except Exception:
            pass

        if current and current > 0:
            entry = pos.get("entry_price", 0)
            pos["current_price"] = current
            if entry > 0:
                if market == "options":
                    pos["pnl"] = round((current - entry) * pos.get("qty", 1) * 100, 2)
                else:
                    units = abs(pos.get("qty", 1))
                    mult = 1 if pos.get("direction", "long") == "long" else -1
                    pos["pnl"] = round((current - entry) * units * mult, 2)
                pos["pnl_pct"] = round((current - entry) / entry * 100, 1)
                # Update trailing stop peak
                if current > pos.get("peak_price", entry):
                    pos["peak_price"] = current
            changed = True

    if changed:
        save_positions(positions)
    return positions


def close_position(position: dict):
    """Mark a position as closed and save."""
    positions = load_positions()
    for i, p in enumerate(positions):
        if p.get("id") == position.get("id"):
            positions[i] = position
            break
    save_positions(positions)


def show_positions(status_filter: str = "open"):
    """
    Display a color-coded table of positions.

    Args:
        status_filter: 'open', 'closed', or 'all'
    """
    positions = load_positions()
    if status_filter != "all":
        positions = [p for p in positions if p.get("status") == status_filter]

    if not positions:
        print(f"\n[!] No {status_filter} positions found.")
        return

    rows = []
    for p in positions:
        status = p.get("status", "open")
        market = p.get("market", "options")
        ticker = p.get("ticker", "")
        direction = p.get("direction", "").upper()
        pnl = p.get("pnl", 0)
        pnl_pct = p.get("pnl_pct", 0)
        entry = p.get("entry_price", 0)
        current = p.get("current_price", entry)
        qty = p.get("qty", 1)
        paper_tag = "📄" if p.get("paper", True) else "💰"
        mode = paper_tag

        # Color code P&L
        if pnl > 0:
            pnl_str = Fore.GREEN + f"+${pnl:.2f} (+{pnl_pct:.1f}%)" + Style.RESET_ALL
        elif pnl < 0:
            pnl_str = Fore.RED + f"-${abs(pnl):.2f} ({pnl_pct:.1f}%)" + Style.RESET_ALL
        else:
            pnl_str = f"${pnl:.2f}"

        # Build row
        if market == "options":
            strike = p.get("strike", "")
            expiry = p.get("suggested_expiry", "")
            detail = f"${strike} {direction} exp {expiry}"
        else:
            detail = f"{direction} {qty} units"

        opened = p.get("opened_at", "")[:10]
        exit_reason = p.get("exit_reason") or ""

        rows.append([
            mode,
            ticker,
            market.upper(),
            detail,
            f"${entry:.4f}" if market == "forex" else f"${entry:.2f}",
            f"${current:.4f}" if market == "forex" else f"${current:.2f}",
            pnl_str,
            p.get("signal_confidence", ""),
            opened,
            exit_reason or status,
        ])

    headers = ["Mode", "Ticker", "Market", "Position", "Entry", "Current", "P&L", "Conf%", "Opened", "Status"]
    print(f"\n{'='*80}")
    print(f"  POSITIONS — {status_filter.upper()}")
    print(f"{'='*80}")
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Summary
    open_pos = [p for p in load_positions() if p.get("status") == "open"]
    summary = get_total_pnl()
    print(f"\n  Open: {len(open_pos)}  |  "
          f"Unrealized P&L: {'+' if summary['unrealized'] >= 0 else ''}${summary['unrealized']:.2f}  |  "
          f"Realized P&L: {'+' if summary['realized'] >= 0 else ''}${summary['realized']:.2f}  |  "
          f"Win Rate: {summary['win_rate_pct']:.1f}%")


def get_total_pnl() -> dict:
    """
    Calculate total P&L across all positions.

    Returns:
        {unrealized, realized, total, wins, losses, win_rate_pct}
    """
    positions = load_positions()
    unrealized = sum(p.get("pnl", 0) for p in positions if p.get("status") == "open")
    realized = sum(p.get("pnl", 0) for p in positions if p.get("status") == "closed")
    wins = sum(1 for p in positions if p.get("status") == "closed" and p.get("pnl", 0) > 0)
    losses = sum(1 for p in positions if p.get("status") == "closed" and p.get("pnl", 0) <= 0)
    total_closed = wins + losses
    win_rate = wins / total_closed * 100 if total_closed > 0 else 0
    return {
        "unrealized": round(unrealized, 2),
        "realized": round(realized, 2),
        "total": round(unrealized + realized, 2),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
    }
