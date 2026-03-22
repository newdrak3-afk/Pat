#!/usr/bin/env python3
"""
Lead Generation & Outreach System
Replace cold calling with automated emails and texts.

Quick Start:
    python main.py setup          # First-time setup
    python main.py find           # Find new leads
    python main.py email          # Send emails to leads
    python main.py sms            # Send texts to leads
    python main.py followup       # Send follow-up emails
    python main.py pipeline       # View your pipeline
    python main.py stats          # View campaign stats
    python main.py status         # Update a lead's status
    python main.py import-csv     # Import leads from CSV
    python main.py export         # Export leads to CSV
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
from modules.sms_outreach import send_sms_campaign, SMS_TEMPLATES
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
# SMS OUTREACH
# ─────────────────────────────────────────────

def cmd_sms():
    sender = load_sender()
    if not sender:
        print("[!] Run 'python main.py setup' first")
        return

    leads = load_leads(LEADS_FILE)
    leads_with_phone = [l for l in leads if l.get("phone") and l.get("status") in ("new", "contacted")]

    if not leads_with_phone:
        print("[!] No leads with phone numbers. Make sure your leads have phone numbers.")
        return

    print(f"\n[+] {len(leads_with_phone)} leads with phone numbers")
    print("\nChoose SMS template:")
    for key, template in SMS_TEMPLATES.items():
        print(f"  {key}: {template[:60]}...")

    choice = input("\nTemplate (default='initial'): ").strip() or "initial"

    if choice not in SMS_TEMPLATES:
        print(f"[!] Invalid template. Choose from: {', '.join(SMS_TEMPLATES.keys())}")
        return

    # Preview
    from modules.sms_outreach import build_sms_context
    preview = build_sms_context(leads_with_phone[0], sender)
    print(f"\n--- PREVIEW ---")
    print(f"To: {leads_with_phone[0].get('phone')}")
    print(f"Message ({len(preview)} chars): {preview}")

    confirm = input(f"\nSend to {len(leads_with_phone)} leads? (yes/no/dry): ").strip().lower()
    dry = confirm == "dry"
    if confirm not in ("yes", "dry"):
        print("Cancelled.")
        return

    updated = send_sms_campaign(leads_with_phone, choice, sender, delay_seconds=15, dry_run=dry)

    with open(LEADS_FILE, "w") as f:
        json.dump(leads, f, indent=2)


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
    "sms": cmd_sms,
    "pipeline": lambda: show_pipeline(LEADS_FILE),
    "stats": lambda: show_stats(LEADS_FILE),
    "status": cmd_status,
    "import-csv": lambda: import_leads_from_csv(input("CSV file path: ").strip()),
    "export": lambda: export_to_csv(LEADS_FILE),
}


def print_help():
    print("""
╔══════════════════════════════════════════════════════════╗
║         LEAD GENERATION & OUTREACH SYSTEM                ║
║     Replace cold calling with email + text outreach      ║
╚══════════════════════════════════════════════════════════╝

Commands:
  python main.py setup        First-time setup (run this first!)
  python main.py find         Find new leads (Google Maps, Apollo, CSV)
  python main.py email        Email new leads
  python main.py followup     Send follow-up emails to existing leads
  python main.py sms          Text leads via Twilio
  python main.py pipeline     View your full pipeline
  python main.py stats        View campaign statistics
  python main.py status       Mark a lead as replied/converted/etc
  python main.py import-csv   Import leads from a CSV file
  python main.py export       Export leads to CSV

Workflow (no cold calling needed):
  1. python main.py setup     ← do this once
  2. python main.py find      ← find leads in your niche/city
  3. python main.py email     ← send first emails
  4. python main.py sms       ← text them too (2-3x better response)
  5. python main.py followup  ← auto follow-up after 3 days
  6. python main.py pipeline  ← see who replied
  7. python main.py status    ← mark replied/converted leads
""")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "help" or cmd not in COMMANDS:
        print_help()
    else:
        COMMANDS[cmd]()
