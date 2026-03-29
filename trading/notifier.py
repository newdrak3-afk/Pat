"""
Notifier — Sends trade alerts via Telegram.

Sends clean, readable alerts with:
- Market name
- Call (Yes) or Put (No)
- Hit percentage (confidence)
- Entry price
- Bet size
- Risk level

Also sends daily summaries and loss alerts.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

from trading.config import SystemConfig
from trading.models import Market, Prediction, Trade

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trading alerts to Telegram."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.warning(
                "Telegram not configured — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env"
            )

    def _send(self, text: str):
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] {text}")
            return

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if not resp.ok:
                logger.warning(f"Telegram send failed: {resp.text}")
        except requests.RequestException as e:
            logger.warning(f"Telegram error: {e}")

    # ─── TRADE ALERTS ───

    def send_trade_alert(
        self,
        market: Market,
        prediction: Prediction,
        amount: float,
        risk_score: float,
    ):
        """Send alert when a trade is placed."""
        side = prediction.recommended_side
        call_or_put = "CALL (Yes)" if side == "Yes" else "PUT (No)"
        hit_pct = prediction.confidence * 100
        edge_pct = prediction.edge * 100
        predicted_pct = prediction.predicted_probability * 100
        market_pct = prediction.market_price * 100

        # Risk emoji
        if risk_score < 0.3:
            risk = "LOW"
        elif risk_score < 0.6:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        msg = (
            f"<b>NEW TRADE ALERT</b>\n"
            f"\n"
            f"<b>{market.question}</b>\n"
            f"\n"
            f"Direction: <b>{call_or_put}</b>\n"
            f"Hit %: <b>{hit_pct:.0f}%</b>\n"
            f"Our prediction: <b>{predicted_pct:.0f}%</b>\n"
            f"Market says: <b>{market_pct:.0f}%</b>\n"
            f"Edge: <b>{edge_pct:+.1f}%</b>\n"
            f"\n"
            f"Bet size: <b>${amount:.2f}</b>\n"
            f"Entry price: <b>{market.current_price:.3f}</b>\n"
            f"Risk: <b>{risk}</b>\n"
            f"\n"
            f"Liquidity: ${market.liquidity:,.0f}\n"
            f"24h Volume: ${market.volume_24h:,.0f}"
        )
        self._send(msg)

    def send_trade_blocked(
        self,
        market: Market,
        prediction: Prediction,
        reason: str,
    ):
        """Send alert when a trade is blocked by risk manager."""
        side = prediction.recommended_side
        call_or_put = "CALL (Yes)" if side == "Yes" else "PUT (No)"

        msg = (
            f"<b>TRADE BLOCKED</b>\n"
            f"\n"
            f"{market.question[:80]}\n"
            f"Direction: {call_or_put}\n"
            f"Reason: {reason}"
        )
        self._send(msg)

    # ─── WIN / LOSS ALERTS ───

    def send_win_alert(self, trade: Trade, pnl: float, bankroll: float):
        """Send alert on a winning trade."""
        msg = (
            f"<b>WIN +${pnl:.2f}</b>\n"
            f"\n"
            f"{trade.market_question[:80]}\n"
            f"Side: {trade.side} @ {trade.entry_price:.3f}\n"
            f"Bet: ${trade.amount:.2f}\n"
            f"\n"
            f"Bankroll: <b>${bankroll:.2f}</b>"
        )
        self._send(msg)

    def send_loss_alert(
        self,
        trade: Trade,
        pnl: float,
        bankroll: float,
        lesson: str,
    ):
        """Send alert on a losing trade with lesson learned."""
        msg = (
            f"<b>LOSS -${abs(pnl):.2f}</b>\n"
            f"\n"
            f"{trade.market_question[:80]}\n"
            f"Side: {trade.side} @ {trade.entry_price:.3f}\n"
            f"Bet: ${trade.amount:.2f}\n"
            f"\n"
            f"Bankroll: <b>${bankroll:.2f}</b>\n"
            f"\n"
            f"<b>What went wrong:</b>\n"
            f"{lesson[:300]}"
        )
        self._send(msg)

    # ─── SCAN ALERTS ───

    def send_scan_summary(
        self,
        total_markets: int,
        flagged: int,
        predictions: int,
        trades: int,
        blocked: int,
    ):
        """Send summary after each scan cycle."""
        msg = (
            f"<b>SCAN COMPLETE</b>\n"
            f"\n"
            f"Markets scanned: {total_markets}\n"
            f"Flagged: {flagged}\n"
            f"Signals found: {predictions}\n"
            f"Trades placed: {trades}\n"
            f"Trades blocked: {blocked}"
        )
        self._send(msg)

    # ─── DAILY SUMMARY ───

    def send_daily_summary(
        self,
        bankroll: float,
        daily_pnl: float,
        wins: int,
        losses: int,
        open_positions: int,
        lessons_today: int,
    ):
        """Send end-of-day summary."""
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        if daily_pnl >= 0:
            pnl_str = f"+${daily_pnl:.2f}"
        else:
            pnl_str = f"-${abs(daily_pnl):.2f}"

        msg = (
            f"<b>DAILY SUMMARY</b>\n"
            f"\n"
            f"Bankroll: <b>${bankroll:.2f}</b>\n"
            f"Today's PnL: <b>{pnl_str}</b>\n"
            f"\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win rate: {win_rate:.0f}%\n"
            f"Open positions: {open_positions}\n"
            f"Lessons learned today: {lessons_today}"
        )
        self._send(msg)

    # ─── STOCK / OPTIONS ALERTS ───

    def send_option_alert(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        hit_pct: float,
        entry_price: float,
        underlying_price: float,
        amount: float,
        reasoning: str,
    ):
        """Send alert for a stock option trade (call/put)."""
        direction = "CALL" if option_type == "call" else "PUT"

        msg = (
            f"<b>OPTIONS ALERT</b>\n"
            f"\n"
            f"<b>{symbol} {direction} ${strike} exp {expiration}</b>\n"
            f"\n"
            f"Hit %: <b>{hit_pct:.0f}%</b>\n"
            f"Stock price: <b>${underlying_price:.2f}</b>\n"
            f"Option price: <b>${entry_price:.2f}</b>\n"
            f"Cost: <b>${amount:.2f}</b>\n"
            f"\n"
            f"{reasoning[:300]}"
        )
        self._send(msg)

    def send_forex_alert(
        self,
        symbol: str,
        side: str,
        hit_pct: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        units: float,
        reasoning: str,
    ):
        """Send alert for a forex trade."""
        direction = "BUY (Long)" if side == "buy" else "SELL (Short)"
        pair = symbol.replace("_", "/")

        # Calculate dollar risk
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)
        risk_dollars = sl_distance * units
        reward_dollars = tp_distance * units
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        msg = (
            f"<b>FOREX ALERT</b>\n"
            f"\n"
            f"<b>{pair} — {direction}</b>\n"
            f"\n"
            f"Confidence: <b>{hit_pct:.0f}%</b>\n"
            f"Entry: <b>{entry_price:.5f}</b>\n"
            f"Stop Loss: <b>{stop_loss:.5f}</b>\n"
            f"Take Profit: <b>{take_profit:.5f}</b>\n"
            f"\n"
            f"Units: <b>{units:,.0f}</b>\n"
            f"Risk: <b>${risk_dollars:,.2f}</b>\n"
            f"Reward: <b>${reward_dollars:,.2f}</b>\n"
            f"R:R: <b>1:{rr_ratio:.1f}</b>\n"
            f"\n"
            f"{reasoning[:300]}"
        )
        self._send(msg)

    # ─── SYSTEM ALERTS ───

    def send_system_alert(self, message: str):
        """Send system-level alert (errors, restarts, etc)."""
        msg = f"<b>SYSTEM</b>\n\n{message}"
        self._send(msg)

    def send_startup(self, bankroll: float, dry_run: bool):
        """Send alert when bot starts."""
        mode = "DRY RUN (no real money)" if dry_run else "LIVE TRADING"
        msg = (
            f"<b>BOT STARTED</b>\n"
            f"\n"
            f"Mode: <b>{mode}</b>\n"
            f"Bankroll: <b>${bankroll:.2f}</b>\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self._send(msg)
