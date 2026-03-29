# Pat Trading Bot v3

Automated multi-asset trading system with forex (OANDA) and options (Alpaca).

## Quick Start

```bash
pip install -r requirements.txt
python trading_main.py auto
```

## Architecture

```
trading_main.py              <- Single entry point
trading/
  auto_trader.py             <- Forex orchestrator (scan -> guard -> trade -> monitor)
  options_trader.py          <- Options trading loop (separate broker, rules)
  guard_engine.py            <- Centralized trade approval pipeline
  risk_governor.py           <- Risk budget + circuit breakers
  position_manager.py        <- Trade lifecycle management
  forex_scanner.py           <- HTF trend filter (D1+H4 gate -> H1 pullback)
  options_scanner.py         <- Stock options signal scanner
  options_contract_selector.py <- Contract picker (DTE, strike, OI, spread)
  session_awareness.py       <- Tokyo/London/NY session tagging
  telegram_bot.py            <- Telegram command control (20+ commands)
  settings.py                <- Runtime profiles (dev/paper/practice/live)
  trade_db.py                <- SQLite database
  notifier.py                <- Telegram alerts with risk/reward
  brokers/
    base.py                  <- Broker-agnostic interface
    oanda.py                 <- OANDA forex/crypto
    alpaca.py                <- Alpaca stocks/options
```

## Environment Variables

```
# Required - Forex
OANDA_API_KEY=your_key
OANDA_ACCOUNT_ID=your_account

# Required - Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional - Options
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Telegram Commands

Info: /status /balance /positions /history /report /lessons /session /why
Controls: /scan /trade /crypto /options /pause /resume /set /mode
Emergency: /kill /safe
Guards: /guards /drawdown /drift /exposure /settings /heartbeat

## Deployment

Deploys on Railway from master branch. Start command: python trading_main.py auto
