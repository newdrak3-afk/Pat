#!/usr/bin/env python3
"""
Lead Generation & Outreach + Options/Forex Trading Bot

── LEAD GENERATION ──────────────────────────────────────
    python main.py setup          # First-time setup
    python main.py find           # Find new leads
    python main.py email          # Send emails to leads
    python main.py followup       # Send follow-up emails
    python main.py pipeline       # View your pipeline
    python main.py stats          # View campaign stats
    python main.py status         # Update a lead's status
    python main.py import-csv     # Import leads from CSV
    python main.py export         # Export leads to CSV

── TRADING BOT ──────────────────────────────────────────
    python main.py trade-setup    # Configure trading credentials
    python main.py scan           # Scan watchlist for signals
    python main.py trade          # Execute a signal (paper or live)
    python main.py positions      # View open positions + P&L
    python main.py close          # Close a position manually
    python main.py backtest       # Backtest strategy on history
    python main.py performance    # Win rate + signal learning report
    python main.py watchlist      # Manage tickers/pairs to scan
    python main.py kill           # Emergency: halt all trading
    python main.py resume         # Resume after kill switch
    python main.py daily          # Today's trading summary
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.lead_finder import (
    search_google_maps,
    scrape_with_serpapi,
    find_leads_from_apollo,
    import_leads_from_csv,
    save_leads,
    load_leads,
    find_email_with_hunter,
)
from modules.email_outreach import send_campaign, get_leads_due_for_followup
from modules.tracker import (
    show_pipeline,
    show_stats,
    update_lead_status,
    export_to_csv,
)
from templates.email_templates import (
    INITIAL_OUTREACH,
    FOLLOWUP_1,
    FOLLOWUP_2,
    build_email_context,
    fill_template,
    VALUE_PROPS,
)

SENDER_FILE = "data/sender.json"
LEADS_FILE = "data/leads.json"


# ─────────────────────────────────────────────
# SENDER PROFILE
# ─────────────────────────────────────────────

def load_sender() -> dict:
    if os.path.exists(SENDER_FILE):
        with open(SENDER_FILE) as f:
            return json.load(f)
    return {}


def save_sender(sender: dict):
    os.makedirs("data", exist_ok=True)
    with open(SENDER_FILE, "w") as f:
        json.dump(sender, f, indent=2)


def setup():
    """Interactive first-time setup."""
    print("\n" + "="*60)
    print("  LEAD GENERATION SYSTEM — FIRST TIME SETUP")
    print("="*60)
    print("\nThis sets up your sender profile and checks your .env config.\n")

    # Sender profile
    sender = {}
    sender["name"] = input("Your name (e.g. 'Pat Smith'): ").strip()
    sender["email"] = os.getenv("EMAIL_ADDRESS") or input("Your email: ").strip()
    sender["phone"] = input("Your phone number (for email signature): ").strip()
    sender["service"] = input("What service do you sell? (e.g. 'web design', 'SEO', 'roofing'): ").strip()
    sender["target_industry"] = input("What industry are your customers in? (e.g. 'plumber', 'realtor'): ").strip()
    sender["target_location"] = input("What location do you target? (e.g. 'Houston TX'): ").strip()

    save_sender(sender)
    print(f"\n[+] Saved profile for {sender['name']}")

    # Check .env
    print("\n--- Checking .env configuration ---")
    checks = [
        ("EMAIL_ADDRESS", "Gmail address for sending emails"),
        ("EMAIL_APP_PASSWORD", "Gmail App Password (myaccount.google.com → Security → App Passwords)"),
        ("TWILIO_ACCOUNT_SID", "Twilio SID (twilio.com — free trial available)"),
        ("TWILIO_AUTH_TOKEN", "Twilio Auth Token"),
        ("TWILIO_PHONE_NUMBER", "Twilio phone number"),
    ]
    for key, desc in checks:
        val = os.getenv(key)
        if val:
            print(f"  ✓ {key} is set")
        else:
            print(f"  ✗ {key} not set — {desc}")

    print("\n[+] Setup complete! Run 'python main.py find' to find your first leads.")


# ─────────────────────────────────────────────
# FIND LEADS
# ─────────────────────────────────────────────

def cmd_find():
    sender = load_sender()
    if not sender:
        print("[!] Run 'python main.py setup' first")
        return

    print("\n--- FIND LEADS ---")
    print("Choose a source:")
    print("  1. Google Maps (free, but results need manual email finding)")
    print("  2. Apollo.io (B2B contacts with emails + phones — needs API key)")
    print("  3. Import from CSV file")
    print("  4. SerpAPI Google Maps (better results — needs API key)")

    choice = input("\nChoice (1-4): ").strip()

    industry = sender.get("target_industry", "")
    location = sender.get("target_location", "")

    if not industry:
        industry = input("Business type to search for: ").strip()
    if not location:
        location = input("Location: ").strip()

    leads = []

    if choice == "1":
        leads = search_google_maps(industry, location)

    elif choice == "2":
        api_key = os.getenv("APOLLO_API_KEY")
        if not api_key:
            api_key = input("Apollo API key (get at apollo.io): ").strip()
        job_title = input("Job title to target (e.g. 'Owner', 'CEO', 'Manager'): ").strip()
        leads = find_leads_from_apollo(job_title, industry, location)

    elif choice == "3":
        filepath = input("Path to CSV file: ").strip()
        leads = import_leads_from_csv(filepath)

    elif choice == "4":
        api_key = input("SerpAPI key (get at serpapi.com): ").strip()
        leads = scrape_with_serpapi(industry, location, api_key)

    else:
        print("[!] Invalid choice")
        return

    if leads:
        save_leads(leads, LEADS_FILE)
        # Try to find missing emails with Hunter.io
        hunter_key = os.getenv("HUNTER_API_KEY")
        if hunter_key:
            print("\n[+] Trying to find emails with Hunter.io...")
            for lead in leads:
                if not lead.get("email") and lead.get("website"):
                    domain = lead["website"].replace("https://", "").replace("http://", "").split("/")[0]
                    email = find_email_with_hunter(domain)
                    if email:
                        lead["email"] = email
                        print(f"  Found email for {lead.get('name')}: {email}")
            save_leads(leads, LEADS_FILE)

        show_pipeline(LEADS_FILE, status_filter="new")


# ─────────────────────────────────────────────
# EMAIL OUTREACH
# ─────────────────────────────────────────────

def cmd_email(dry_run: bool = False):
    sender = load_sender()
    if not sender:
        print("[!] Run 'python main.py setup' first")
        return

    leads = load_leads(LEADS_FILE)
    new_leads = [l for l in leads if l.get("status") == "new" and l.get("email")]

    if not new_leads:
        print("[!] No new leads with emails. Run 'python main.py find' first.")
        print("    Or run 'python main.py followup' to send follow-ups to existing leads.")
        return

    print(f"\n[+] Ready to email {len(new_leads)} new leads")
    print("\nChoose email template:")
    print("  1. Initial outreach (first contact)")
    print("  2. LinkedIn-style (mention you found their business)")
    print("  3. Free value (offer something useful for free)")

    choice = input("\nChoice (1-3, default=1): ").strip() or "1"

    templates = {
        "1": INITIAL_OUTREACH,
        "2": {
            "subject": "Noticed {company} while looking for {business_type} businesses",
            "body": """Hi {first_name},

I noticed {company} while searching for {business_type} businesses in {location}.

I help businesses like yours {value_prop} — and I think there's a real opportunity here for you.

Would you be open to a quick 10-minute call this week? No pitch, just a conversation to see if it makes sense.

{sender_name}
{sender_phone}
{sender_email}""",
        },
        "3": {
            "subject": "Free tips for {business_type} businesses in {location}",
            "body": """Hi {first_name},

I put together a free guide specifically for {business_type} businesses on how to {value_prop} — and I wanted to share it with you.

It's completely free, no strings attached. You can use it whether we ever work together or not.

Just reply "send it" and I'll shoot it over right away.

{sender_name}
{sender_phone}
{sender_email}""",
        },
    }

    template = templates.get(choice, INITIAL_OUTREACH)

    # Build context variables
    sample_context = build_email_context(new_leads[0], sender)
    subject = template["subject"]
    body = template["body"]

    print(f"\n--- PREVIEW (first lead) ---")
    try:
        print(f"To: {new_leads[0].get('email')}")
        print(f"Subject: {subject.format(**sample_context)}")
        print(f"Body:\n{body.format(**sample_context)[:300]}...")
    except KeyError as e:
        print(f"[!] Template missing variable: {e}")

    confirm = input(f"\nSend to all {len(new_leads)} leads? (yes/no/dry): ").strip().lower()
    if confirm == "dry":
        dry_run = True
        confirm = "yes"

    if confirm != "yes":
        print("Cancelled.")
        return

    # Add value_prop to each lead
    for lead in new_leads:
        biz = lead.get("business_type", "default").lower()
        lead["value_prop"] = VALUE_PROPS.get(biz, VALUE_PROPS["default"])

    updated = send_campaign(
        leads=new_leads,
        subject=subject,
        body_template=body,
        sender=sender,
        delay_seconds=30,
        dry_run=dry_run,
    )

    # Merge updated leads back
    names_updated = {l.get("name"): l for l in updated}
    final_leads = [names_updated.get(l.get("name"), l) for l in leads]
    save_leads.__wrapped__ = None  # avoid circular
    from modules.lead_finder import save_leads as _save
    # Direct save
    os.makedirs("data", exist_ok=True)
    with open(LEADS_FILE, "w") as f:
        json.dump(final_leads, f, indent=2)

    print(f"\n[+] Leads updated. Run 'python main.py pipeline' to see status.")


# ─────────────────────────────────────────────
# FOLLOW-UP EMAILS
# ─────────────────────────────────────────────

def cmd_followup():
    sender = load_sender()
    if not sender:
        print("[!] Run 'python main.py setup' first")
        return

    leads = load_leads(LEADS_FILE)
    due = get_leads_due_for_followup(leads, days_since_last=3)

    if not due:
        print("[!] No leads due for follow-up right now.")
        print("    Leads get a follow-up 3 days after last contact.")
        return

    print(f"\n[+] {len(due)} leads are due for follow-up")
    for l in due[:5]:
        print(f"    - {l.get('name')} (step {l.get('email_sequence_step', 0)+1}/3, last: {l.get('last_email_date', 'never')})")
    if len(due) > 5:
        print(f"    ... and {len(due)-5} more")

    confirm = input(f"\nSend follow-up emails to all {len(due)}? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    from templates.email_templates import FOLLOWUP_1, FOLLOWUP_2

    for lead in due:
        step = lead.get("email_sequence_step", 0)
        template = FOLLOWUP_1 if step == 1 else FOLLOWUP_2
        biz = lead.get("business_type", "default").lower()
        lead["value_prop"] = VALUE_PROPS.get(biz, VALUE_PROPS["default"])

    updated = send_campaign(
        leads=due,
        subject=FOLLOWUP_1["subject"],
        body_template=FOLLOWUP_1["body"],
        sender=sender,
        delay_seconds=30,
    )

    os.makedirs("data", exist_ok=True)
    with open(LEADS_FILE, "w") as f:
        json.dump(leads, f, indent=2)
    print(f"[+] Follow-ups sent.")


# ─────────────────────────────────────────────
# UPDATE STATUS
# ─────────────────────────────────────────────

def cmd_status():
    print("\nUpdate a lead's status (e.g. when someone replies or you close a deal)")
    name = input("Lead name (partial match OK): ").strip()
    print("\nStatuses: new | contacted | replied | meeting_scheduled | converted | not_interested | unsubscribed")
    new_status = input("New status: ").strip()
    notes = input("Notes (optional): ").strip()
    update_lead_status(name, new_status, notes)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

COMMANDS = {
    "setup": setup,
    "find": cmd_find,
    "email": cmd_email,
    "followup": cmd_followup,
    "pipeline": lambda: show_pipeline(LEADS_FILE),
    "stats": lambda: show_stats(LEADS_FILE),
    "status": cmd_status,
    "import-csv": lambda: import_leads_from_csv(input("CSV file path: ").strip()),
    "export": lambda: export_to_csv(LEADS_FILE),
    # ── Trading bot ──
    "trade-setup": cmd_trade_setup,
    "scan": cmd_scan,
    "trade": cmd_trade,
    "positions": cmd_positions,
    "close": cmd_close,
    "backtest": cmd_backtest,
    "performance": cmd_performance,
    "watchlist": cmd_watchlist,
    "kill": cmd_kill,
    "resume": cmd_resume,
    "daily": cmd_daily,
}


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║    LEAD GENERATION + OPTIONS/FOREX TRADING BOT               ║
╚══════════════════════════════════════════════════════════════╝

── LEAD GENERATION ───────────────────────────────────────────
  python main.py setup        First-time setup (run this first!)
  python main.py find         Find new leads (Google Maps, Apollo, CSV)
  python main.py email        Email new leads
  python main.py followup     Send follow-up emails
  python main.py pipeline     View your full lead pipeline
  python main.py stats        View campaign statistics
  python main.py status       Mark a lead as replied/converted/etc
  python main.py import-csv   Import leads from a CSV file
  python main.py export       Export leads to CSV

── TRADING BOT ───────────────────────────────────────────────
  python main.py trade-setup  Configure API keys & trading settings
  python main.py scan         Scan watchlist for high-confidence signals
  python main.py trade        Pick a signal and place a paper/live trade
  python main.py positions    View open positions with live P&L
  python main.py close        Close a position manually
  python main.py backtest     Backtest strategy on historical data
  python main.py performance  Win rate report + what the bot learned
  python main.py watchlist    Add/remove/view tickers to scan
  python main.py kill         Emergency stop — halt all trading
  python main.py resume       Re-enable trading after kill/pause
  python main.py daily        Today's trading summary

  Set MARKET_MODE=options|forex|both in .env to choose markets.
  Set TRADIER_SANDBOX=true for paper trading (default).
""")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "help" or cmd not in COMMANDS:
        print_help()
    else:
        COMMANDS[cmd]()


# ─────────────────────────────────────────────────────────────
# TRADING BOT COMMANDS
# ─────────────────────────────────────────────────────────────

WATCHLIST_FILE = "data/watchlist.json"


def _load_watchlist() -> list:
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    default = os.getenv("WATCHLIST", "AAPL,TSLA,SPY,QQQ,NVDA").split(",")
    return [t.strip().upper() for t in default]


def _save_watchlist(wl: list):
    os.makedirs("data", exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, indent=2)


def _build_clients():
    """Instantiate all trading clients from env config."""
    from modules.market_data import TradierClient
    from modules.forex_data import OandaClient
    from modules.scanner import Scanner
    from modules.trader import Trader
    from modules.risk_engine import RiskEngine
    from modules.learner import Learner
    from modules.telegram_alerts import TelegramAlerter
    import modules.portfolio as portfolio_mod

    client = TradierClient()
    forex_client = OandaClient()
    risk = RiskEngine()
    learner = Learner()
    telegram = TelegramAlerter()

    class _Portfolio:
        """Thin wrapper so Trader can call portfolio methods."""
        def load_positions(self): return portfolio_mod.load_positions()
        def add_position(self, order, signal): return portfolio_mod.add_position(order, signal)
        def close_position(self, pos): return portfolio_mod.close_position(pos)

    trader = Trader(
        client=client,
        forex_client=forex_client,
        portfolio=_Portfolio(),
        risk=risk,
        learner=learner,
        telegram=telegram,
    )
    scanner = Scanner(client=client, forex_client=forex_client)
    return client, forex_client, scanner, trader, risk, learner, telegram


def cmd_trade_setup():
    """Interactive setup for trading credentials."""
    print("\n" + "="*60)
    print("  TRADING BOT SETUP")
    print("="*60)
    print("\nThis checks your .env for required trading credentials.\n")

    checks = [
        ("TRADIER_API_KEY",    "Tradier API key — get free at tradier.com/products/api"),
        ("TRADIER_ACCOUNT_ID", "Tradier account ID — found in your Tradier dashboard"),
        ("TRADIER_SANDBOX",    "Set to 'true' for paper trading (recommended to start)"),
        ("OANDA_API_KEY",      "OANDA API key — optional, for forex (oanda.com)"),
        ("OANDA_ACCOUNT_ID",   "OANDA account ID — optional, for forex"),
        ("TELEGRAM_BOT_TOKEN", "Telegram bot token — optional, for alerts (@BotFather)"),
        ("TELEGRAM_CHAT_ID",   "Your Telegram chat ID — optional"),
        ("NEWS_API_KEY",       "NewsAPI.org key — optional, improves news signals (100 free/day)"),
        ("MARKET_MODE",        "Set to 'options', 'forex', or 'both' (default: options)"),
        ("MIN_CONFIDENCE",     "Minimum signal confidence to trade (default: 70)"),
        ("MAX_POSITION_SIZE",  "Max $ per trade (default: 500)"),
        ("MAX_DAILY_LOSS",     "Pause trading if daily loss exceeds this $ (default: 200)"),
    ]

    for key, desc in checks:
        val = os.getenv(key)
        if val:
            display = val[:4] + "..." if len(val) > 8 else val
            print(f"  ✓ {key} = {display}")
        else:
            print(f"  ✗ {key} not set — {desc}")

    print("\nAdd missing keys to your .env file, then run 'python main.py scan'.")
    print("\nExample .env additions:")
    print("  TRADIER_API_KEY=your_key_here")
    print("  TRADIER_ACCOUNT_ID=your_account_id")
    print("  TRADIER_SANDBOX=true")
    print("  MARKET_MODE=options")
    print("  WATCHLIST=AAPL,TSLA,SPY,QQQ,NVDA")
    print("  MIN_CONFIDENCE=70")
    print("  MAX_POSITION_SIZE=500")
    print("  MAX_DAILY_LOSS=200")


def cmd_scan():
    """Scan watchlist for high-confidence trade signals."""
    from modules.forex_data import get_active_markets, get_default_forex_watchlist
    from colorama import Fore, Style, init
    init(autoreset=True)

    _, _, scanner, _, _, _, telegram = _build_clients()

    active_markets = get_active_markets()
    options_wl = _load_watchlist() if "options" in active_markets else []
    forex_wl = get_default_forex_watchlist() if "forex" in active_markets else []

    print(f"\n{'='*60}")
    print(f"  SCANNING — markets: {', '.join(active_markets).upper()}")
    if options_wl:
        print(f"  Options watchlist: {', '.join(options_wl)}")
    if forex_wl:
        print(f"  Forex watchlist: {', '.join(forex_wl[:6])}{'...' if len(forex_wl) > 6 else ''}")
    print(f"  Min confidence: {os.getenv('MIN_CONFIDENCE', '70')}%")
    print(f"{'='*60}\n")

    signals = scanner.scan_all_markets(options_wl, forex_wl)

    if not signals:
        print("[!] No signals met the confidence threshold right now.")
        print("    Try lowering MIN_CONFIDENCE in .env or expanding your watchlist.")
        return

    print(f"[+] Found {len(signals)} signal(s):\n")
    for i, sig in enumerate(signals, 1):
        conf = sig["confidence"]
        ticker = sig["ticker"]
        direction = sig["direction"].upper()
        market = sig["market"].upper()
        regime = sig.get("regime", "").upper()

        color = Fore.GREEN if conf >= 80 else Fore.YELLOW
        print(f"  {i}. {color}{ticker} → {direction}{Style.RESET_ALL}  ({market}, {conf}% confidence, regime: {regime})")

        if sig.get("catalyst_risk"):
            print(f"     {Fore.RED}⚠ CATALYST RISK: {sig['catalyst_reason']}{Style.RESET_ALL}")

        if market == "OPTIONS" and sig.get("suggested_strike"):
            print(f"     Strike: ${sig['suggested_strike']} | Exp: {sig['suggested_expiry']} | Est: ${sig.get('est_premium', '?'):.2f}/contract")

        print(f"\n{sig['reasoning']}\n")
        print("-" * 50)

    # Offer to send to Telegram
    if telegram.is_configured():
        notify = input(f"\nSend top signal to Telegram? (yes/no): ").strip().lower()
        if notify == "yes":
            telegram.alert_signal(signals[0])
            print("[+] Sent.")

    print(f"\nRun 'python main.py trade' to execute one of these signals.")
    # Save signals to temp file for trade command to pick up
    os.makedirs("data", exist_ok=True)
    with open("data/last_scan.json", "w") as f:
        json.dump(signals, f, indent=2)


def cmd_trade():
    """Pick a signal and place a paper or live trade."""
    # Load last scan results
    last_scan_file = "data/last_scan.json"
    if not os.path.exists(last_scan_file):
        print("[!] No scan results found. Run 'python main.py scan' first.")
        return

    with open(last_scan_file) as f:
        signals = json.load(f)

    if not signals:
        print("[!] No signals in last scan. Run 'python main.py scan' first.")
        return

    _, _, _, trader, _, _, _ = _build_clients()

    print(f"\n{'='*50}")
    print(f"  SELECT SIGNAL TO TRADE")
    print(f"{'='*50}")
    for i, sig in enumerate(signals, 1):
        ticker = sig["ticker"]
        direction = sig["direction"].upper()
        conf = sig["confidence"]
        market = sig["market"].upper()
        print(f"  {i}. {ticker} {direction} ({market}, {conf}% confidence)")

    try:
        choice = int(input(f"\nEnter number (1-{len(signals)}): ").strip()) - 1
        if choice < 0 or choice >= len(signals):
            print("[!] Invalid choice.")
            return
    except ValueError:
        print("[!] Invalid input.")
        return

    signal = signals[choice]

    live_mode = os.getenv("LIVE_TRADING", "false").lower() in ("true", "1")
    mode_label = "LIVE" if live_mode else "PAPER"

    print(f"\n  Selected: {signal['ticker']} {signal['direction'].upper()}")
    print(f"  Mode: {mode_label}")
    if signal.get("suggested_strike"):
        print(f"  Contract: ${signal['suggested_strike']} {signal['direction']} exp {signal['suggested_expiry']}")
        print(f"  Est. cost: ${signal.get('est_premium', '?'):.2f} per contract")
    print(f"\n{signal['reasoning']}")

    confirm = input(f"\nPlace this {mode_label} trade? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    result = trader.execute_signal(signal, dry_run=False)

    if result.get("status") in ("paper_filled", "filled"):
        print(f"\n[+] Trade placed! ({result['mode']})")
        print(f"    Ticker: {result.get('ticker') or result.get('pair')}")
        print(f"    Direction: {result.get('direction', '').upper()}")
        if result.get("option_symbol"):
            print(f"    Contract: {result['option_symbol']}")
        print(f"    Fill price: ${result.get('fill_price', 0):.4f}")
        print(f"    Qty: {result.get('qty') or result.get('units')}")
        if result.get("cost_basis"):
            print(f"    Total cost: ${result['cost_basis']:.2f}")
        print(f"\n  Run 'python main.py positions' to monitor it.")
    elif result.get("status") == "blocked":
        print(f"\n[!] Trade blocked by risk engine: {result['reason']}")
    else:
        print(f"\n[!] Error: {result.get('reason', 'Unknown error')}")


def cmd_positions():
    """View open positions with live P&L updates."""
    import modules.portfolio as portfolio_mod
    from modules.market_data import TradierClient
    from modules.forex_data import OandaClient

    client = TradierClient()
    forex_client = OandaClient()

    print("\n[~] Refreshing prices...")
    portfolio_mod.update_position_prices(
        client=client if client.is_configured() else None,
        forex_client=forex_client if forex_client.is_configured() else None,
    )

    mode = input("\nShow (open/closed/all) [default: open]: ").strip().lower() or "open"
    portfolio_mod.show_positions(status_filter=mode)


def cmd_close():
    """Manually close an open position."""
    import modules.portfolio as portfolio_mod

    positions = portfolio_mod.load_positions()
    open_pos = [p for p in positions if p.get("status") == "open"]

    if not open_pos:
        print("[!] No open positions to close.")
        return

    print(f"\n{'='*50}")
    print(f"  OPEN POSITIONS")
    print(f"{'='*50}")
    for i, p in enumerate(open_pos, 1):
        ticker = p.get("ticker", "")
        direction = p.get("direction", "").upper()
        pnl = p.get("pnl", 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        print(f"  {i}. {ticker} {direction}  P&L: {pnl_str}")

    try:
        choice = int(input(f"\nClose which position? (1-{len(open_pos)}): ").strip()) - 1
        if choice < 0 or choice >= len(open_pos):
            print("[!] Invalid choice.")
            return
    except ValueError:
        print("[!] Invalid input.")
        return

    pos = open_pos[choice]
    reason = input("Reason (optional): ").strip() or "manual"

    _, _, _, trader, _, _, _ = _build_clients()
    result = trader.close_position(pos, reason)

    if result.get("status") == "closed":
        pnl = result.get("pnl", 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        print(f"\n[+] Position closed. P&L: {pnl_str}")
    else:
        print(f"\n[!] Error: {result.get('reason', 'Unknown')}")


def cmd_backtest():
    """Run a backtest on historical data."""
    from modules.backtester import run_backtest, show_backtest_results
    from modules.market_data import TradierClient
    from modules.forex_data import get_active_markets, get_default_forex_watchlist

    print(f"\n{'='*50}")
    print(f"  BACKTEST")
    print(f"{'='*50}")

    active = get_active_markets()
    market = "options"
    if len(active) > 1:
        market = input("Market to backtest (options/forex) [default: options]: ").strip().lower() or "options"
    elif active[0] == "forex":
        market = "forex"

    if market == "forex":
        default_tickers = ",".join(get_default_forex_watchlist()[:4])
    else:
        default_tickers = ",".join(_load_watchlist())

    tickers_input = input(f"Tickers to test [{default_tickers}]: ").strip()
    tickers = [t.strip().upper() for t in tickers_input.split(",")] if tickers_input else default_tickers.split(",")

    days = input("Days of history to replay [90]: ").strip()
    days = int(days) if days.isdigit() else 90

    min_conf = input(f"Min confidence [{os.getenv('MIN_CONFIDENCE', '70')}]: ").strip()
    min_conf = int(min_conf) if min_conf.isdigit() else int(os.getenv("MIN_CONFIDENCE", "70"))

    print(f"\n[~] Running backtest on {', '.join(tickers)} over {days} days...")

    client = TradierClient() if market == "options" else None
    results = run_backtest(
        tickers=tickers,
        days_back=days,
        client=client,
        market=market,
        min_confidence=min_conf,
    )
    show_backtest_results(results)


def cmd_performance():
    """Show win rate, signal learning report, and weight adjustments."""
    from modules.learner import Learner

    learner = Learner()
    learner.show_performance_report()

    adjust = input("\nRun weight update now? (yes/no): ").strip().lower()
    if adjust == "yes":
        result = learner.update_signal_weights()
        status = result.get("status", "")
        if status == "updated":
            print("\n[+] Signal weights updated:")
            for sig, change in result.get("changes", {}).items():
                arrow = "↑" if change["new"] > change["old"] else "↓"
                print(f"  {arrow} {sig}: {change['old']:.1f} → {change['new']:.1f}  ({change['reason']})")
        elif status == "rollback":
            print(f"\n[!] Rollback triggered: {result['reason']}")
        elif status == "no_changes":
            print(f"\n[~] No changes needed: {result['reason']}")
        else:
            print(f"\n[~] {result.get('reason', status)}")


def cmd_watchlist():
    """Add, remove, or view the options watchlist."""
    wl = _load_watchlist()

    print(f"\n{'='*50}")
    print(f"  WATCHLIST (Options)")
    print(f"{'='*50}")
    print(f"  Current: {', '.join(wl)}")
    print(f"\n  1. Add tickers")
    print(f"  2. Remove a ticker")
    print(f"  3. Replace entire watchlist")
    print(f"  4. View only (no changes)")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        new = input("Tickers to add (comma-separated): ").strip().upper()
        to_add = [t.strip() for t in new.split(",") if t.strip()]
        for t in to_add:
            if t not in wl:
                wl.append(t)
        _save_watchlist(wl)
        print(f"[+] Watchlist updated: {', '.join(wl)}")

    elif choice == "2":
        ticker = input("Ticker to remove: ").strip().upper()
        if ticker in wl:
            wl.remove(ticker)
            _save_watchlist(wl)
            print(f"[+] Removed {ticker}. Watchlist: {', '.join(wl)}")
        else:
            print(f"[!] {ticker} not in watchlist.")

    elif choice == "3":
        new = input("New watchlist (comma-separated tickers): ").strip().upper()
        wl = [t.strip() for t in new.split(",") if t.strip()]
        _save_watchlist(wl)
        print(f"[+] Watchlist set to: {', '.join(wl)}")

    else:
        print(f"  Watchlist: {', '.join(wl)}")


def cmd_kill():
    """Activate emergency kill switch — halts all trading."""
    from modules.risk_engine import RiskEngine
    from modules.telegram_alerts import TelegramAlerter

    print("\n⛔  KILL SWITCH")
    reason = input("Reason (optional): ").strip() or "manual kill switch"
    confirm = input("Halt ALL trading? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    risk = RiskEngine()
    risk.activate_kill_switch(reason)

    telegram = TelegramAlerter()
    telegram.alert_kill_switch()

    print("[+] Kill switch activated. No trades will be placed.")
    print("    Run 'python main.py resume' to re-enable trading.")


def cmd_resume():
    """Deactivate kill switch and resume trading."""
    from modules.risk_engine import RiskEngine

    risk = RiskEngine()
    risk.deactivate_kill_switch()
    print("[+] Kill switch cleared. Trading is now enabled.")
    print("    Run 'python main.py scan' to find new signals.")


def cmd_daily():
    """Show today's trading summary."""
    from modules.risk_engine import RiskEngine
    from modules.telegram_alerts import TelegramAlerter
    import modules.portfolio as portfolio_mod
    from tabulate import tabulate

    risk = RiskEngine()
    summary = risk.get_daily_summary()
    pnl = summary["daily_pnl"]
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    print(f"\n{'='*50}")
    print(f"  DAILY SUMMARY — {summary['date']}")
    print(f"{'='*50}")
    rows = [
        ["Trades today", summary["trades_today"]],
        ["Wins", summary["wins"]],
        ["Losses", summary["losses"]],
        ["Win rate", f"{summary['win_rate_pct']}%"],
        ["Realized P&L", pnl_str],
        ["Consecutive losses", summary["consecutive_losses"]],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    total_pnl = portfolio_mod.get_total_pnl()
    print(f"\n  All-time:  Unrealized ${total_pnl['unrealized']:.2f}  |  "
          f"Realized ${total_pnl['realized']:.2f}  |  "
          f"Win rate {total_pnl['win_rate_pct']:.1f}%")

    # Offer to send Telegram summary
    telegram = TelegramAlerter()
    if telegram.is_configured():
        send = input("\nSend summary to Telegram? (yes/no): ").strip().lower()
        if send == "yes":
            telegram.send_daily_summary({**summary, "daily_pnl": pnl})
