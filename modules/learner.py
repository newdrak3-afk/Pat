"""
learner.py — Trade outcome tracking and adaptive signal weight tuning.

Tracks every trade's signals and outcome, then periodically adjusts
signal weights in data/signal_weights.json to reflect what's actually working.

Safety guardrails:
  - Minimum 20 closed trades before any weight adjustment
  - Max 10% change per signal per adjustment cycle
  - Saves prior weight versions (up to 10) for rollback
  - Rolls back automatically if new weights underperform previous weights
  - Analyzes signal combinations, not just individual signals
  - Analyzes performance by ticker, regime, time of day, and expiration bucket
"""

import os
import json
import numpy as np
from datetime import datetime, date
from typing import Optional

TRADE_LOG_FILE = "data/trade_log.json"
SIGNAL_WEIGHTS_FILE = "data/signal_weights.json"
WEIGHT_HISTORY_FILE = "data/signal_weight_history.json"

DEFAULT_WEIGHTS = {
    "momentum_breakout": 25,
    "rsi": 20,
    "ema_crossover": 20,
    "news_sentiment": 20,
    "options_volume": 15,
}

MIN_TRADES_FOR_ADJUSTMENT = 20
MAX_WEIGHT_CHANGE_PCT = 10  # max % change per signal per cycle
MIN_SIGNAL_WEIGHT = 5       # floor — no signal drops to zero
MAX_SIGNAL_WEIGHT = 50      # ceiling — no single signal dominates


class Learner:
    """
    Logs trades and learns which signals are producing winning trades.
    Adjusts signal weights in data/signal_weights.json over time.
    """

    def __init__(self):
        self.weights = self._load_weights()

    # ──────────────────────────────────────────────
    # TRADE LOGGING
    # ──────────────────────────────────────────────

    def log_trade_opened(self, signal: dict, order: dict):
        """
        Log a new trade to trade_log.json when a position is opened.

        Args:
            signal: Scanner signal dict (contains signals_fired, confidence, regime)
            order: Order result from Trader
        """
        log = self._load_log()
        entry = {
            "id": order.get("id") or f"{signal.get('ticker', '')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "ticker": signal.get("ticker") or signal.get("pair", ""),
            "market": signal.get("market", "options"),
            "direction": signal.get("direction", ""),
            "confidence": signal.get("confidence", 0),
            "signals_fired": [s[0] for s in signal.get("signals_fired", [])],
            "signal_scores": {s[0]: s[1] for s in signal.get("signals_fired", [])},
            "regime": signal.get("regime", "unknown"),
            "catalyst_risk": signal.get("catalyst_risk", False),
            "entry_price": order.get("fill_price", 0),
            "opened_at": datetime.now().isoformat(),
            "opened_date": str(date.today()),
            "opened_hour": datetime.now().hour,
            "status": "open",
            "paper": order.get("paper", True),
            # These are filled when trade closes:
            "exit_price": None,
            "pnl": None,
            "pnl_pct": None,
            "win": None,
            "exit_reason": None,
            "closed_at": None,
        }
        log.append(entry)
        self._save_log(log)

    def log_trade_closed(self, position: dict):
        """
        Update the trade log when a position is closed.
        Triggers weight re-evaluation if enough trades have accumulated.
        """
        log = self._load_log()
        ticker = position.get("ticker", "")
        opened_at = position.get("opened_at", "")

        # Find matching open entry
        for entry in reversed(log):
            if entry.get("ticker") == ticker and entry.get("status") == "open":
                if not opened_at or entry.get("opened_at", "")[:10] == opened_at[:10]:
                    entry["exit_price"] = position.get("exit_price")
                    entry["pnl"] = position.get("pnl", 0)
                    entry["pnl_pct"] = position.get("pnl_pct", 0)
                    entry["win"] = (position.get("pnl", 0) or 0) > 0
                    entry["exit_reason"] = position.get("exit_reason", "")
                    entry["closed_at"] = datetime.now().isoformat()
                    entry["status"] = "closed"
                    break

        self._save_log(log)

        # Check if it's time to update weights
        closed = [e for e in log if e.get("status") == "closed"]
        if len(closed) >= MIN_TRADES_FOR_ADJUSTMENT and len(closed) % 10 == 0:
            self.update_signal_weights()

    # ──────────────────────────────────────────────
    # ANALYSIS
    # ──────────────────────────────────────────────

    def analyze_mistakes(self) -> dict:
        """
        Analyze closed trades to identify which signals are underperforming.

        Returns:
            {
                total_closed, win_rate, avg_win, avg_loss,
                weak_signals: [{signal, win_rate, appears_in_pct_of_losers}],
                strong_signals: [{signal, win_rate, appears_in_pct_of_winners}],
                by_regime, by_ticker, by_hour, by_expiry_bucket
            }
        """
        log = self._load_log()
        closed = [e for e in log if e.get("status") == "closed" and e.get("win") is not None]

        if len(closed) < 5:
            return {"error": f"Not enough closed trades yet ({len(closed)}/5 minimum for analysis)"}

        wins = [t for t in closed if t["win"]]
        losses = [t for t in closed if not t["win"]]
        total = len(closed)
        win_rate = len(wins) / total * 100

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

        # Signal analysis
        all_signals = list(DEFAULT_WEIGHTS.keys())
        signal_stats = {}
        for sig in all_signals:
            times_fired = [t for t in closed if sig in t.get("signals_fired", [])]
            if not times_fired:
                signal_stats[sig] = {"fires": 0, "win_rate": 0, "winner_pct": 0, "loser_pct": 0}
                continue
            sig_wins = sum(1 for t in times_fired if t["win"])
            sig_win_rate = sig_wins / len(times_fired) * 100
            winner_pct = sum(1 for t in wins if sig in t.get("signals_fired", [])) / max(len(wins), 1) * 100
            loser_pct = sum(1 for t in losses if sig in t.get("signals_fired", [])) / max(len(losses), 1) * 100
            signal_stats[sig] = {
                "fires": len(times_fired),
                "win_rate": round(sig_win_rate, 1),
                "winner_pct": round(winner_pct, 1),
                "loser_pct": round(loser_pct, 1),
            }

        weak = sorted([{"signal": k, **v} for k, v in signal_stats.items() if v["fires"] >= 3],
                      key=lambda x: x["win_rate"])[:3]
        strong = sorted([{"signal": k, **v} for k, v in signal_stats.items() if v["fires"] >= 3],
                        key=lambda x: x["win_rate"], reverse=True)[:3]

        # By regime
        regimes = set(t.get("regime", "unknown") for t in closed)
        by_regime = {}
        for r in regimes:
            rt = [t for t in closed if t.get("regime") == r]
            if rt:
                by_regime[r] = {
                    "trades": len(rt),
                    "win_rate": round(sum(1 for t in rt if t["win"]) / len(rt) * 100, 1),
                    "total_pnl": round(sum(t.get("pnl", 0) for t in rt), 2),
                }

        # By ticker
        tickers = set(t.get("ticker", "") for t in closed)
        by_ticker = {}
        for tk in tickers:
            tt = [t for t in closed if t.get("ticker") == tk]
            if tt:
                by_ticker[tk] = {
                    "trades": len(tt),
                    "win_rate": round(sum(1 for t in tt if t["win"]) / len(tt) * 100, 1),
                    "total_pnl": round(sum(t.get("pnl", 0) for t in tt), 2),
                }

        # By hour of day
        by_hour = {}
        for t in closed:
            h = t.get("opened_hour")
            if h is not None:
                bucket = f"{h:02d}:00"
                if bucket not in by_hour:
                    by_hour[bucket] = {"trades": 0, "wins": 0}
                by_hour[bucket]["trades"] += 1
                by_hour[bucket]["wins"] += 1 if t["win"] else 0
        for h in by_hour:
            tc = by_hour[h]["trades"]
            by_hour[h]["win_rate"] = round(by_hour[h]["wins"] / tc * 100, 1) if tc > 0 else 0

        return {
            "total_closed": total,
            "win_rate": round(win_rate, 1),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "expectancy": round(float(avg_win * len(wins) / total + avg_loss * len(losses) / total), 2),
            "weak_signals": weak,
            "strong_signals": strong,
            "signal_stats": signal_stats,
            "by_regime": by_regime,
            "by_ticker": by_ticker,
            "by_hour": by_hour,
        }

    def update_signal_weights(self) -> dict:
        """
        Adjust signal weights based on analysis of closed trades.

        Rules:
          - If a signal's win rate < 50% and it appears in >40% of losers:
            reduce its weight by up to MAX_WEIGHT_CHANGE_PCT %
          - If a signal's win rate > 70% and it appears in >40% of winners:
            increase its weight by up to MAX_WEIGHT_CHANGE_PCT %
          - All weights are clamped between MIN_SIGNAL_WEIGHT and MAX_SIGNAL_WEIGHT
          - Save old weights to history before updating
          - Auto-rollback if win rate drops vs prior cycle
        """
        analysis = self.analyze_mistakes()
        if "error" in analysis:
            return {"status": "skipped", "reason": analysis["error"]}

        # Save current weights to history before changing
        self._save_weights_to_history(self.weights.copy(), analysis.get("win_rate", 0))

        new_weights = dict(self.weights)
        signal_stats = analysis.get("signal_stats", {})
        changes = {}

        for sig, stats in signal_stats.items():
            if stats["fires"] < MIN_TRADES_FOR_ADJUSTMENT // 2:
                continue  # not enough data for this signal

            current_w = new_weights.get(sig, DEFAULT_WEIGHTS.get(sig, 20))
            win_rate = stats["win_rate"]
            loser_pct = stats["loser_pct"]
            winner_pct = stats["winner_pct"]

            if win_rate < 50 and loser_pct > 40:
                # Signal is mostly appearing on losing trades — reduce weight
                reduction = current_w * (MAX_WEIGHT_CHANGE_PCT / 100)
                new_w = max(current_w - reduction, MIN_SIGNAL_WEIGHT)
                changes[sig] = {"old": current_w, "new": round(new_w, 1), "reason": f"win_rate={win_rate:.0f}%, loser_pct={loser_pct:.0f}%"}
                new_weights[sig] = round(new_w, 1)

            elif win_rate > 70 and winner_pct > 40:
                # Signal is mostly appearing on winning trades — increase weight
                increase = current_w * (MAX_WEIGHT_CHANGE_PCT / 100)
                new_w = min(current_w + increase, MAX_SIGNAL_WEIGHT)
                changes[sig] = {"old": current_w, "new": round(new_w, 1), "reason": f"win_rate={win_rate:.0f}%, winner_pct={winner_pct:.0f}%"}
                new_weights[sig] = round(new_w, 1)

        if not changes:
            return {"status": "no_changes", "reason": "No signals met adjustment thresholds"}

        # Check if we should rollback: compare vs historical performance
        history = self._load_weight_history()
        if len(history) >= 2:
            prev_wr = history[-2].get("win_rate_at_save", 0)
            current_wr = analysis.get("win_rate", 0)
            if current_wr < prev_wr - 5:  # 5% win rate degradation threshold
                return {
                    "status": "rollback",
                    "reason": f"Win rate dropped {prev_wr:.1f}% → {current_wr:.1f}%. Not applying changes.",
                }

        # Save new weights
        self.weights = new_weights
        self._save_weights(new_weights)

        return {
            "status": "updated",
            "changes": changes,
            "new_weights": new_weights,
            "current_win_rate": analysis.get("win_rate"),
        }

    def rollback_weights(self, versions_back: int = 1) -> dict:
        """Manually roll back to a previous set of weights."""
        history = self._load_weight_history()
        if len(history) < versions_back + 1:
            return {"status": "error", "reason": f"Not enough history to roll back {versions_back} version(s)"}
        target = history[-(versions_back + 1)]
        self.weights = target["weights"]
        self._save_weights(target["weights"])
        return {"status": "rolled_back", "weights": target["weights"], "saved_at": target.get("saved_at")}

    def show_performance_report(self):
        """Print a formatted performance report."""
        from tabulate import tabulate

        analysis = self.analyze_mistakes()
        if "error" in analysis:
            print(f"\n[!] {analysis['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  PERFORMANCE REPORT")
        print(f"{'='*60}")

        summary = [
            ["Total Closed Trades", analysis["total_closed"]],
            ["Win Rate", f"{analysis['win_rate']}%"],
            ["Avg Winner", f"${analysis['avg_win']:.2f}"],
            ["Avg Loser", f"${analysis['avg_loss']:.2f}"],
            ["Expectancy/Trade", f"${analysis['expectancy']:.2f}"],
        ]
        print(tabulate(summary, headers=["Metric", "Value"], tablefmt="simple"))

        print(f"\n  Signal Performance:")
        sig_rows = []
        for s in analysis.get("signal_stats", {}).items():
            name, stats = s
            sig_rows.append([name, stats["fires"], f"{stats['win_rate']}%",
                             f"{stats['winner_pct']}% of wins", f"{stats['loser_pct']}% of losses"])
        print(tabulate(sig_rows, headers=["Signal", "Fires", "Win Rate", "In Winners", "In Losers"], tablefmt="simple"))

        if analysis.get("by_regime"):
            print(f"\n  By Regime:")
            rows = [[r, v["trades"], f"{v['win_rate']}%", f"${v['total_pnl']:.2f}"]
                    for r, v in analysis["by_regime"].items()]
            print(tabulate(rows, headers=["Regime", "Trades", "Win Rate", "P&L"], tablefmt="simple"))

        print(f"\n  Current Signal Weights:")
        w_rows = [[k, v] for k, v in self.weights.items()]
        print(tabulate(w_rows, headers=["Signal", "Weight"], tablefmt="simple"))

    # ──────────────────────────────────────────────
    # PERSISTENCE HELPERS
    # ──────────────────────────────────────────────

    def _load_log(self) -> list:
        if not os.path.exists(TRADE_LOG_FILE):
            return []
        try:
            with open(TRADE_LOG_FILE) as f:
                return json.load(f)
        except Exception:
            return []

    def _save_log(self, log: list):
        os.makedirs("data", exist_ok=True)
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    def _load_weights(self) -> dict:
        if os.path.exists(SIGNAL_WEIGHTS_FILE):
            try:
                with open(SIGNAL_WEIGHTS_FILE) as f:
                    data = json.load(f)
                return {**DEFAULT_WEIGHTS, **data.get("weights", {})}
            except Exception:
                pass
        return dict(DEFAULT_WEIGHTS)

    def _save_weights(self, weights: dict):
        os.makedirs("data", exist_ok=True)
        with open(SIGNAL_WEIGHTS_FILE, "w") as f:
            json.dump({"weights": weights, "updated_at": datetime.now().isoformat()}, f, indent=2)

    def _load_weight_history(self) -> list:
        if not os.path.exists(WEIGHT_HISTORY_FILE):
            return []
        try:
            with open(WEIGHT_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []

    def _save_weights_to_history(self, weights: dict, win_rate: float):
        history = self._load_weight_history()
        history.append({
            "saved_at": datetime.now().isoformat(),
            "weights": weights,
            "win_rate_at_save": win_rate,
        })
        history = history[-10:]  # keep last 10 versions
        os.makedirs("data", exist_ok=True)
        with open(WEIGHT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
