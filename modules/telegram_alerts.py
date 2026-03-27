"""
telegram_alerts.py — Send trade alerts and notifications via Telegram bot.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → copy your BOT_TOKEN
  2. Start a chat with your bot, then visit:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
     to find your CHAT_ID
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=123456:ABCdef...
       TELEGRAM_CHAT_ID=987654321

Usage:
  from modules.telegram_alerts import TelegramAlerter
  tg = TelegramAlerter()
  tg.send("Trade alert message here")

All functions fail silently if Telegram is not configured, so the bot
continues to work even without Telegram set up.
"""

import os
import requests
from datetime import datetime


class TelegramAlerter:
    """Send messages to a Telegram chat via bot API."""

    API_BASE = "https://api.telegram.org/bot"

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

    def is_configured(self) -> bool:
        return self.enabled

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a plain text or Markdown message to the configured chat.
        Returns True if sent successfully, False otherwise.
        Fails silently so the bot doesn't crash on Telegram issues.
        """
        if not self.enabled:
            return False
        try:
            url = f"{self.API_BASE}{self.token}/sendMessage"
            r = requests.post(url, data={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }, timeout=8)
            return r.status_code == 200
        except Exception:
            return False

    # ──────────────────────────────────────────────
    # FORMATTED ALERT TEMPLATES
    # ──────────────────────────────────────────────

    def alert_signal(self, signal: dict) -> bool:
        """
        Send a new trade signal alert.
        signal keys: ticker/pair, market, direction, confidence, reasoning, suggested_strike,
                     suggested_expiry, est_premium
        """
        ticker = signal.get("ticker") or signal.get("pair", "???")
        market = signal.get("market", "options").upper()
        direction = signal.get("direction", "???").upper()
        confidence = signal.get("confidence", 0)
        reasoning = signal.get("reasoning", "")

        if market == "OPTIONS":
            strike = signal.get("suggested_strike", "")
            expiry = signal.get("suggested_expiry", "")
            premium = signal.get("est_premium", "")
            detail = f"Strike: `{strike}` | Exp: `{expiry}` | Est Premium: `${premium}`"
        else:
            direction_arrow = "📈" if direction in ("LONG", "BUY") else "📉"
            units = signal.get("units", "")
            detail = f"Direction: `{direction}` {direction_arrow} | Units: `{units}`"

        msg = (
            f"🔔 *NEW SIGNAL — {market}*\n"
            f"{'─' * 30}\n"
            f"*{ticker}* → `{direction}` ({confidence}% confidence)\n\n"
            f"{reasoning}\n\n"
            f"{detail}\n"
            f"{'─' * 30}\n"
            f"_⚠ Not financial advice. Always verify before trading._"
        )
        return self.send(msg)

    def alert_trade_placed(self, trade: dict) -> bool:
        """Alert when a trade order is placed."""
        ticker = trade.get("ticker") or trade.get("pair", "???")
        market = trade.get("market", "options").upper()
        side = trade.get("side") or trade.get("direction", "buy")
        qty = trade.get("qty") or trade.get("units", 0)
        price = trade.get("fill_price") or trade.get("limit_price") or "market"
        option_sym = trade.get("option_symbol", "")
        mode = "📄 PAPER" if trade.get("paper", True) else "💰 LIVE"

        if market == "OPTIONS":
            detail = f"Contract: `{option_sym}`\nQty: `{qty}` contract(s) @ `${price}`"
        else:
            detail = f"Units: `{qty}` @ `{price}`"

        msg = (
            f"✅ *TRADE PLACED — {mode}*\n"
            f"{'─' * 30}\n"
            f"*{ticker}* | {side.upper()}\n"
            f"{detail}\n"
            f"Time: `{datetime.now().strftime('%H:%M:%S')}`"
        )
        return self.send(msg)

    def alert_position_closed(self, position: dict, reason: str = "") -> bool:
        """Alert when a position is closed."""
        ticker = position.get("ticker") or position.get("pair", "???")
        pnl = position.get("pnl", 0)
        pnl_pct = position.get("pnl_pct", 0)
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_str = f"+${pnl:.2f} (+{pnl_pct:.1f}%)" if pnl >= 0 else f"-${abs(pnl):.2f} ({pnl_pct:.1f}%)"

        msg = (
            f"{emoji} *POSITION CLOSED*\n"
            f"{'─' * 30}\n"
            f"*{ticker}* → P&L: `{pnl_str}`\n"
            f"Reason: _{reason or 'manual'}_"
        )
        return self.send(msg)

    def alert_stop_loss(self, position: dict) -> bool:
        """Urgent stop loss alert."""
        ticker = position.get("ticker") or position.get("pair", "???")
        loss = position.get("pnl", 0)
        msg = (
            f"🚨 *STOP LOSS HIT*\n"
            f"{'─' * 30}\n"
            f"*{ticker}* position hit stop loss.\n"
            f"Loss: `${abs(loss):.2f}`\n"
            f"Position closed automatically."
        )
        return self.send(msg)

    def alert_take_profit(self, position: dict) -> bool:
        """Take profit alert."""
        ticker = position.get("ticker") or position.get("pair", "???")
        gain = position.get("pnl", 0)
        pnl_pct = position.get("pnl_pct", 0)
        msg = (
            f"🎯 *TAKE PROFIT HIT*\n"
            f"{'─' * 30}\n"
            f"*{ticker}* hit profit target!\n"
            f"Gain: `+${gain:.2f} (+{pnl_pct:.1f}%)`"
        )
        return self.send(msg)

    def alert_daily_loss_limit(self, daily_loss: float, limit: float) -> bool:
        """Alert when daily loss limit is reached — bot pauses trading."""
        msg = (
            f"⛔ *DAILY LOSS LIMIT HIT — TRADING PAUSED*\n"
            f"{'─' * 30}\n"
            f"Daily loss: `${abs(daily_loss):.2f}` / limit `${limit:.2f}`\n"
            f"Bot has stopped trading for today.\n"
            f"Resume manually with: `python main.py resume`"
        )
        return self.send(msg)

    def alert_kill_switch(self) -> bool:
        """Alert when kill switch is activated."""
        msg = (
            f"🔴 *KILL SWITCH ACTIVATED*\n"
            f"All trading is halted.\n"
            f"Run `python main.py resume` to re-enable."
        )
        return self.send(msg)

    def alert_regime_change(self, ticker: str, old_regime: str, new_regime: str) -> bool:
        """Alert on market regime change."""
        emoji_map = {
            "bullish_trend": "📈", "bearish_trend": "📉",
            "high_volatility": "⚡", "low_volatility": "😴",
            "choppy": "〰️",
        }
        emoji = emoji_map.get(new_regime, "🔄")
        msg = (
            f"{emoji} *REGIME CHANGE — {ticker}*\n"
            f"`{old_regime.upper()}` → `{new_regime.upper()}`\n"
            f"Scanner signals may be filtered accordingly."
        )
        return self.send(msg)

    def alert_catalyst_warning(self, ticker: str, reason: str) -> bool:
        """Warn about an upcoming earnings or macro event."""
        msg = (
            f"⚠️ *CATALYST WARNING — {ticker}*\n"
            f"{reason}\n"
            f"_Trade with caution or wait for clarity._"
        )
        return self.send(msg)

    def send_daily_summary(self, stats: dict) -> bool:
        """Send end-of-day summary."""
        trades = stats.get("trades_today", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = stats.get("daily_pnl", 0)
        win_rate = stats.get("win_rate_pct", 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = (
            f"📊 *DAILY SUMMARY — {datetime.now().strftime('%B %d, %Y')}*\n"
            f"{'─' * 30}\n"
            f"Trades: `{trades}` | Wins: `{wins}` | Losses: `{losses}`\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"Daily P&L: `{pnl_str}`\n"
        )
        return self.send(msg)
