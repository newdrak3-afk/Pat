"""
Lead Tracker / Mini CRM
Track every lead's status, emails sent, responses, and notes.
View your whole pipeline in the terminal.
"""

import json
import os
from datetime import datetime
from tabulate import tabulate


LEADS_FILE = "data/leads.json"

# Lead status flow:
# new → contacted → replied → meeting_scheduled → converted
# new → contacted → not_interested
# new → contacted → unsubscribed

STATUS_COLORS = {
    "new": "\033[37m",          # white
    "contacted": "\033[33m",    # yellow
    "replied": "\033[36m",      # cyan
    "meeting_scheduled": "\033[35m",  # magenta
    "converted": "\033[32m",    # green
    "not_interested": "\033[31m",  # red
    "unsubscribed": "\033[31m",  # red
}
RESET = "\033[0m"


def load_leads(filepath: str = LEADS_FILE) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return json.load(f)


def save_leads(leads: list[dict], filepath: str = LEADS_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(leads, f, indent=2)


def update_lead_status(name: str, new_status: str, notes: str = "", filepath: str = LEADS_FILE):
    """
    Update a lead's status and add notes.

    Usage:
        update_lead_status("John Smith", "replied", "Interested, following up Friday")
        update_lead_status("Jane Doe", "not_interested")
        update_lead_status("Bob Co", "converted", "Signed $2k/month deal!")
    """
    leads = load_leads(filepath)
    updated = False

    valid_statuses = ["new", "contacted", "replied", "meeting_scheduled",
                      "converted", "not_interested", "unsubscribed"]

    if new_status not in valid_statuses:
        print(f"[!] Invalid status. Choose from: {', '.join(valid_statuses)}")
        return

    for lead in leads:
        lead_name = lead.get("name", "") or lead.get("company", "")
        if name.lower() in lead_name.lower():
            old_status = lead.get("status", "new")
            lead["status"] = new_status
            lead["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            if notes:
                existing_notes = lead.get("notes", "")
                timestamp = datetime.now().strftime("%m/%d")
                lead["notes"] = f"{existing_notes}\n[{timestamp}] {notes}".strip()
            print(f"[+] Updated '{lead_name}': {old_status} → {new_status}")
            updated = True
            break

    if not updated:
        print(f"[!] Lead '{name}' not found")
        return

    save_leads(leads, filepath)


def add_note(name: str, note: str, filepath: str = LEADS_FILE):
    """Add a note to a lead without changing their status."""
    update_lead_status(name, load_leads(filepath)[0].get("status", "new"), notes=note, filepath=filepath)


def show_pipeline(filepath: str = LEADS_FILE, status_filter: str = None):
    """
    Display your full lead pipeline as a table.
    Optionally filter by status: 'new', 'contacted', 'replied', etc.
    """
    leads = load_leads(filepath)

    if not leads:
        print("[!] No leads yet. Use 'find' command to find leads first.")
        return

    if status_filter:
        leads = [l for l in leads if l.get("status", "new") == status_filter]

    # Build display table
    rows = []
    for lead in leads:
        name = (lead.get("company") or lead.get("name", "?"))[:25]
        status = lead.get("status", "new")
        color = STATUS_COLORS.get(status, "")
        email = lead.get("email", "—")[:28]
        phone = lead.get("phone", "—")
        last_contact = lead.get("last_email_date") or lead.get("last_sms_date") or "Never"
        step = lead.get("email_sequence_step", 0)
        notes = (lead.get("notes", "") or "")[:30]

        rows.append([
            name,
            f"{color}{status}{RESET}",
            email,
            phone or "—",
            last_contact,
            f"Email #{step}",
            notes,
        ])

    headers = ["Name", "Status", "Email", "Phone", "Last Contact", "Sequence", "Notes"]
    print(f"\n{'='*80}")
    print(f"  LEAD PIPELINE  ({len(rows)} leads{f' | filter: {status_filter}' if status_filter else ''})")
    print(f"{'='*80}")
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Summary counts
    from collections import Counter
    status_counts = Counter(l.get("status", "new") for l in load_leads(filepath))
    print(f"\n  Summary: ", end="")
    for status, count in sorted(status_counts.items()):
        color = STATUS_COLORS.get(status, "")
        print(f"{color}{status}: {count}{RESET}  ", end="")
    print()


def show_stats(filepath: str = LEADS_FILE):
    """Show campaign statistics."""
    leads = load_leads(filepath)
    if not leads:
        print("[!] No leads data.")
        return

    total = len(leads)
    contacted = sum(1 for l in leads if l.get("status") not in ("new",))
    replied = sum(1 for l in leads if l.get("status") in ("replied", "meeting_scheduled", "converted"))
    converted = sum(1 for l in leads if l.get("status") == "converted")
    with_email = sum(1 for l in leads if l.get("email"))
    with_phone = sum(1 for l in leads if l.get("phone"))

    reply_rate = (replied / contacted * 100) if contacted > 0 else 0
    conversion_rate = (converted / contacted * 100) if contacted > 0 else 0

    print(f"\n{'='*50}")
    print("  CAMPAIGN STATS")
    print(f"{'='*50}")
    print(f"  Total leads:          {total}")
    print(f"  With email:           {with_email}")
    print(f"  With phone:           {with_phone}")
    print(f"  Contacted:            {contacted}")
    print(f"  Replied:              {replied}  ({reply_rate:.1f}% reply rate)")
    print(f"  Converted:            {converted}  ({conversion_rate:.1f}% close rate)")
    print(f"{'='*50}\n")


def export_to_csv(filepath: str = LEADS_FILE, output: str = "data/leads_export.csv"):
    """Export all leads to CSV for use in Excel or Google Sheets."""
    import csv
    leads = load_leads(filepath)
    if not leads:
        print("[!] No leads to export.")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    all_keys = list({k for lead in leads for k in lead.keys()})

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(leads)

    print(f"[+] Exported {len(leads)} leads to {output}")
