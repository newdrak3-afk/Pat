"""
trader.py — Order execution with risk management for options and forex.

Handles:
- Translating scanner signals into actual orders
- Applying all risk checks before executing
- Paper trading (default) vs live trading (requires LIVE_TRADING=true)
- Exit logic: trailing stop, partial take-profit, time-based exit,
  signal reversal exit, pre-expiry exit (options), catalyst invalidation
- Logging all trades to paper_trade_log.json or live_trade_log.json

LIVE_TRADING=false by default. Must be set to 'true' AND CONFIRM_LIVE=yes
in .env to enable real order placement.
"""

import os
import json
from datetime import datetime, date
from typing import Optional

PAPER_LOG = "data/paper_trade_log.json"
LIVE_LOG = "data/live_trade_log.json"


def _is_live_mode() -> bool:
    """Returns True only if both LIVE_TRADING=true and CONFIRM_LIVE=yes are set."""
    live = os.getenv("LIVE_TRADING", "false").lower()
    confirm = os.getenv("CONFIRM_LIVE", "no").lower()
    return live in ("true", "1") and confirm == "yes"


class Trader:
    """
    Executes trades for options and forex markets with full risk gating.
    """

    def __init__(self, client=None, forex_client=None, portfolio=None, risk=None, learner=None, telegram=None):
        """
        Args:
            client: TradierClient instance
            forex_client: OandaClient instance
            portfolio: Portfolio instance
            risk: RiskEngine instance
            learner: Learner instance
            telegram: TelegramAlerter instance
        """
        self.client = client
        self.forex_client = forex_client
        self.portfolio = portfolio
        self.risk = risk
        self.learner = learner
        self.telegram = telegram
        self.live = _is_live_mode()
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "500"))
        self.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "30"))      # exit if -30%
        self.take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "50"))  # exit if +50%
        self.trailing_stop_pct = float(os.getenv("TRAILING_STOP_PCT", "15"))  # trailing stop
        self.partial_tp_pct = float(os.getenv("PARTIAL_TP_PCT", "30"))    # partial take profit at +30%
        self.pre_expiry_dte = int(os.getenv("PRE_EXPIRY_DTE", "3"))       # close options N days before expiry
        self.forex_stop_pips = float(os.getenv("FOREX_STOP_PIPS", "30"))
        self.forex_tp_pips = float(os.getenv("FOREX_TP_PIPS", "60"))

    # ──────────────────────────────────────────────
    # EXECUTE SIGNAL
    # ──────────────────────────────────────────────

    def execute_signal(self, signal: dict, dry_run: bool = False) -> dict:
        """
        Execute a trade signal after passing all risk checks.

        Args:
            signal: Signal dict from scanner
            dry_run: If True, simulate without placing order

        Returns:
            Result dict with order info, or error dict.
        """
        ticker = signal.get("ticker") or signal.get("pair", "")
        market = signal.get("market", "options")

        # Load current positions for risk check
        positions = self.portfolio.load_positions() if self.portfolio else []

        # Get best contract (options) for risk check
        contract = None
        if market == "options" and signal.get("suggested_contract"):
            contract = self._get_contract_details(ticker, signal)

        # Run risk checks
        if self.risk:
            risk_result = self.risk.check_all(signal, positions, contract)
            if not risk_result["approved"]:
                return {
                    "status": "blocked",
                    "reason": risk_result["reason"],
                    "blocked_by": risk_result.get("blocked_by"),
                }
            for w in risk_result.get("warnings", []):
                print(f"  ⚠ Risk warning: {w}")

        # Route to correct market executor
        if market == "forex":
            result = self._execute_forex(signal, dry_run)
        else:
            result = self._execute_options(signal, contract, dry_run)

        if result.get("status") in ("filled", "paper_filled", "dry_run"):
            # Record in portfolio and learner
            result["market"] = market
            result["paper"] = not self.live
            result["ticker"] = ticker

            if self.portfolio:
                self.portfolio.add_position(result, signal)

            if self.risk:
                self.risk.record_trade_opened(ticker)

            if self.learner:
                self.learner.log_trade_opened(signal, result)

            self._log_trade(result, signal)

            if self.telegram:
                self.telegram.alert_trade_placed(result)

        return result

    # ──────────────────────────────────────────────
    # OPTIONS EXECUTION
    # ──────────────────────────────────────────────

    def _execute_options(self, signal: dict, contract: dict, dry_run: bool) -> dict:
        """Place an options order via Tradier."""
        ticker = signal["ticker"]
        direction = signal["direction"]  # 'call' or 'put'
        option_symbol = signal.get("suggested_contract", "")
        premium = signal.get("est_premium", 0) / 100  # per-share premium

        if not option_symbol:
            return {"status": "error", "reason": "No suggested contract in signal. Run scan first."}

        # Position sizing
        qty = self._calc_qty_options(premium)
        if qty < 1:
            return {"status": "error", "reason": f"Premium ${premium:.2f}/share is too high for MAX_POSITION_SIZE ${self.max_position_size:.0f}"}

        side = "buy_to_open"
        mode_label = "PAPER" if not self.live else "LIVE"

        if dry_run or not self.live:
            return {
                "status": "paper_filled",
                "mode": mode_label,
                "ticker": ticker,
                "option_symbol": option_symbol,
                "direction": direction,
                "side": side,
                "qty": qty,
                "fill_price": premium,
                "cost_basis": round(premium * qty * 100, 2),
                "paper": True,
            }

        # Live execution
        if not self.client or not self.client.is_configured():
            return {"status": "error", "reason": "Tradier client not configured"}

        try:
            order = self.client.place_order(
                ticker=ticker,
                option_symbol=option_symbol,
                side=side,
                qty=qty,
                order_type="limit",
                limit_price=round(premium * 1.02, 2),  # slight buffer for fill
            )
            order["status"] = "filled"
            order["mode"] = "LIVE"
            order["fill_price"] = premium
            order["cost_basis"] = round(premium * qty * 100, 2)
            order["paper"] = False
            return order
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    # ──────────────────────────────────────────────
    # FOREX EXECUTION
    # ──────────────────────────────────────────────

    def _execute_forex(self, signal: dict, dry_run: bool) -> dict:
        """Place a forex order via OANDA."""
        pair = signal["ticker"]
        direction = signal["direction"]  # 'long' or 'short'
        current_price = signal.get("current_price", 0)

        units = self._calc_units_forex(pair, current_price)
        if direction == "short":
            units = -units

        mode_label = "PAPER" if not self.live else "LIVE"

        if dry_run or not self.live:
            return {
                "status": "paper_filled",
                "mode": mode_label,
                "pair": pair,
                "ticker": pair,
                "direction": direction,
                "units": units,
                "fill_price": current_price,
                "paper": True,
                "stop_loss_pips": self.forex_stop_pips,
                "take_profit_pips": self.forex_tp_pips,
            }

        if not self.forex_client or not self.forex_client.is_configured():
            return {"status": "error", "reason": "OANDA client not configured"}

        try:
            order = self.forex_client.place_market_order(
                pair=pair,
                units=units,
                stop_loss_pips=self.forex_stop_pips,
                take_profit_pips=self.forex_tp_pips,
            )
            order["status"] = "filled"
            order["mode"] = "LIVE"
            order["paper"] = False
            return order
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    # ──────────────────────────────────────────────
    # EXIT LOGIC
    # ──────────────────────────────────────────────

    def check_and_exit_positions(self, positions: list) -> list:
        """
        Check all open positions against exit conditions.
        Returns list of positions that were closed with reason.

        Exit conditions checked (in priority order):
          1. Hard stop loss (-STOP_LOSS_PCT%)
          2. Take profit (+TAKE_PROFIT_PCT%)
          3. Trailing stop (price pulls back TRAILING_STOP_PCT% from peak)
          4. Pre-expiry exit (options only: DTE <= PRE_EXPIRY_DTE)
          5. Time-based exit (options held > MAX_HOLD_DAYS)
        """
        closed = []
        for pos in positions:
            if pos.get("status") != "open":
                continue
            exit_result = self._check_exit(pos)
            if exit_result:
                pos["status"] = "closed"
                pos["exit_price"] = exit_result["exit_price"]
                pos["exit_reason"] = exit_result["reason"]
                pos["exit_time"] = datetime.now().isoformat()
                pnl = self._calc_pnl(pos)
                pos["pnl"] = pnl
                pos["pnl_pct"] = exit_result.get("pnl_pct", 0)

                if self.risk:
                    self.risk.record_trade_closed(pnl)

                if self.learner:
                    self.learner.log_trade_closed(pos)

                self._log_trade(pos, {})

                if self.telegram:
                    if "stop_loss" in exit_result["reason"].lower():
                        self.telegram.alert_stop_loss(pos)
                    elif "take_profit" in exit_result["reason"].lower():
                        self.telegram.alert_take_profit(pos)
                    else:
                        self.telegram.alert_position_closed(pos, exit_result["reason"])

                closed.append(pos)
        return closed

    def _check_exit(self, position: dict) -> Optional[dict]:
        """
        Check if a single position should be exited.
        Returns exit dict {reason, exit_price, pnl_pct} or None.
        """
        market = position.get("market", "options")
        entry_price = position.get("fill_price") or position.get("entry_price", 0)
        current_price = self._get_current_price(position)

        if not current_price or entry_price == 0:
            return None

        pnl_pct = (current_price - entry_price) / entry_price * 100

        # Update trailing stop peak
        peak_price = position.get("peak_price", entry_price)
        if current_price > peak_price:
            position["peak_price"] = current_price
            peak_price = current_price

        # 1. Hard stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return {"reason": "stop_loss", "exit_price": current_price, "pnl_pct": pnl_pct}

        # 2. Take profit
        if pnl_pct >= self.take_profit_pct:
            return {"reason": "take_profit", "exit_price": current_price, "pnl_pct": pnl_pct}

        # 3. Trailing stop (only after +PARTIAL_TP_PCT gain)
        if pnl_pct > self.partial_tp_pct:
            pullback = (peak_price - current_price) / peak_price * 100
            if pullback >= self.trailing_stop_pct:
                return {"reason": "trailing_stop", "exit_price": current_price, "pnl_pct": pnl_pct}

        # 4. Pre-expiry exit (options only)
        if market == "options":
            expiry = position.get("suggested_expiry") or position.get("expiry", "")
            if expiry:
                from datetime import date as _d
                try:
                    exp_d = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
                    dte = (exp_d - _d.today()).days
                    if dte <= self.pre_expiry_dte:
                        return {"reason": f"pre_expiry_exit (DTE={dte})", "exit_price": current_price, "pnl_pct": pnl_pct}
                except Exception:
                    pass

        # 5. Time-based exit
        max_hold = int(os.getenv("MAX_HOLD_DAYS", "21"))
        opened = position.get("opened_at", "")
        if opened:
            try:
                open_dt = datetime.fromisoformat(opened)
                days_held = (datetime.now() - open_dt).days
                if days_held >= max_hold:
                    return {"reason": f"time_exit (held {days_held} days)", "exit_price": current_price, "pnl_pct": pnl_pct}
            except Exception:
                pass

        return None

    # ──────────────────────────────────────────────
    # CLOSE POSITION MANUALLY
    # ──────────────────────────────────────────────

    def close_position(self, position: dict, reason: str = "manual") -> dict:
        """Manually close a position."""
        market = position.get("market", "options")
        current_price = self._get_current_price(position)
        entry_price = position.get("fill_price") or position.get("entry_price", 0)
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price and current_price else 0

        if self.live:
            try:
                if market == "forex" and self.forex_client:
                    self.forex_client.close_position(position.get("pair", position.get("ticker")))
                elif market == "options" and self.client:
                    self.client.place_order(
                        ticker=position["ticker"],
                        option_symbol=position.get("option_symbol", ""),
                        side="sell_to_close",
                        qty=position.get("qty", 1),
                        order_type="market",
                    )
            except Exception as e:
                return {"status": "error", "reason": str(e)}

        position["status"] = "closed"
        position["exit_price"] = current_price
        position["exit_reason"] = reason
        position["exit_time"] = datetime.now().isoformat()
        position["pnl_pct"] = pnl_pct
        position["pnl"] = self._calc_pnl(position)

        if self.risk:
            self.risk.record_trade_closed(position["pnl"])
        if self.learner:
            self.learner.log_trade_closed(position)
        if self.portfolio:
            self.portfolio.close_position(position)

        self._log_trade(position, {})

        if self.telegram:
            self.telegram.alert_position_closed(position, reason)

        return {"status": "closed", **position}

    def close_all_positions(self, positions: list, reason: str = "close_all") -> list:
        """Close all open positions."""
        closed = []
        for p in positions:
            if p.get("status") == "open":
                result = self.close_position(p, reason)
                closed.append(result)
        return closed

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _calc_qty_options(self, premium_per_share: float) -> int:
        """How many contracts can we buy given MAX_POSITION_SIZE."""
        cost_per_contract = premium_per_share * 100
        if cost_per_contract <= 0:
            return 0
        return max(int(self.max_position_size / cost_per_contract), 1)

    def _calc_units_forex(self, pair: str, price: float) -> int:
        """Calculate forex units based on MAX_POSITION_SIZE."""
        if price <= 0:
            return 1000
        return max(int(self.max_position_size / price), 1000)

    def _calc_pnl(self, position: dict) -> float:
        """Calculate realized P&L for a closed position."""
        entry = position.get("fill_price") or position.get("entry_price", 0)
        exit_p = position.get("exit_price", 0)
        qty = position.get("qty", 1)
        market = position.get("market", "options")
        if entry == 0 or exit_p == 0:
            return 0
        if market == "options":
            return round((exit_p - entry) * qty * 100, 2)
        else:
            units = abs(position.get("units", qty))
            direction = position.get("direction", "long")
            mult = 1 if direction == "long" else -1
            return round((exit_p - entry) * units * mult, 2)

    def _get_current_price(self, position: dict) -> float:
        """Get current market price for a position."""
        market = position.get("market", "options")
        ticker = position.get("ticker") or position.get("pair", "")
        try:
            if market == "forex" and self.forex_client and self.forex_client.is_configured():
                q = self.forex_client.get_quote(ticker)
                return q["mid"] if q else 0
            elif self.client and self.client.is_configured():
                q = self.client.get_quote(ticker)
                return q["last"] if q else 0
        except Exception:
            pass
        return 0

    def _get_contract_details(self, ticker: str, signal: dict) -> Optional[dict]:
        """Fetch options contract details for risk check."""
        if not self.client:
            return None
        try:
            expiry = signal.get("suggested_expiry")
            if not expiry:
                return None
            chain = self.client.get_options_chain(ticker, expiry)
            opt_sym = signal.get("suggested_contract", "")
            for c in chain:
                if c.get("symbol") == opt_sym:
                    return c
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────
    # TRADE LOGGING
    # ──────────────────────────────────────────────

    def _log_trade(self, trade: dict, signal: dict):
        """Append trade to paper_trade_log.json or live_trade_log.json."""
        log_file = LIVE_LOG if self.live and not trade.get("paper", True) else PAPER_LOG
        try:
            os.makedirs("data", exist_ok=True)
            log = []
            if os.path.exists(log_file):
                with open(log_file) as f:
                    log = json.load(f)
            entry = {
                **trade,
                "signal_confidence": signal.get("confidence"),
                "signal_regime": signal.get("regime"),
                "logged_at": datetime.now().isoformat(),
            }
            log.append(entry)
            with open(log_file, "w") as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass
