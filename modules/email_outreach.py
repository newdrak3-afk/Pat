"""
Email Outreach Module
Send personalized emails to leads automatically.
Uses Gmail (or any SMTP). No cold calling needed.
"""

import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))


def send_email(to_email: str, subject: str, body: str, from_name: str = "") -> bool:
    """
    Send a single email via SMTP (Gmail).

    Setup Gmail:
    1. Go to myaccount.google.com → Security → 2-Step Verification → App Passwords
    2. Create an App Password for "Mail"
    3. Put that 16-char password in your .env as EMAIL_APP_PASSWORD

    Returns True if sent successfully.
    """
    if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
        print("  [!] Set EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env file")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{from_name} <{EMAIL_ADDRESS}>" if from_name else EMAIL_ADDRESS
        msg["To"] = to_email

        # Plain text version
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())

        return True

    except smtplib.SMTPAuthenticationError:
        print("  [!] Gmail auth failed. Make sure you're using an App Password, not your real password.")
        print("  [!] Go to: myaccount.google.com → Security → App Passwords")
        return False
    except smtplib.SMTPException as e:
        print(f"  [!] SMTP error: {e}")
        return False
    except Exception as e:
        print(f"  [!] Email error: {e}")
        return False


def send_campaign(
    leads: list[dict],
    subject: str,
    body_template: str,
    sender: dict,
    delay_seconds: int = 30,
    dry_run: bool = False,
) -> list[dict]:
    """
    Send emails to a list of leads with personalization.

    Args:
        leads: List of lead dicts (must have 'email' field)
        subject: Email subject (can use {first_name}, {company} placeholders)
        body_template: Email body (can use {first_name}, {company} placeholders)
        sender: Dict with 'name', 'email', 'phone' keys
        delay_seconds: Wait between sends to avoid spam filters (default 30s)
        dry_run: If True, prints emails but doesn't send them

    Returns:
        Updated leads list with email status
    """
    results = []
    sent_count = 0
    skipped_count = 0

    leads_with_email = [l for l in leads if l.get("email")]
    leads_without_email = [l for l in leads if not l.get("email")]

    if leads_without_email:
        print(f"  [!] {len(leads_without_email)} leads have no email — skipping them")
        print(f"      Use Hunter.io (find_email_with_hunter) to find their emails first")

    print(f"\n[+] Starting email campaign to {len(leads_with_email)} leads")
    print(f"    Delay between emails: {delay_seconds}s (avoids spam filters)\n")

    for i, lead in enumerate(leads_with_email):
        first_name = lead.get("first_name") or lead.get("name", "there").split()[0]
        company = lead.get("company") or lead.get("name", "your company")

        # Personalize subject and body
        personalized_subject = subject.format(
            first_name=first_name, company=company,
            business_type=lead.get("business_type", ""),
            location=lead.get("location", ""),
        )
        personalized_body = body_template.format(
            first_name=first_name, company=company,
            business_type=lead.get("business_type", ""),
            location=lead.get("location", ""),
            sender_name=sender.get("name", ""),
            sender_phone=sender.get("phone", ""),
            sender_email=sender.get("email", EMAIL_ADDRESS),
            value_prop=lead.get("value_prop", "grow your business"),
            result_example=lead.get("result_example", "get great results"),
        )

        if dry_run:
            print(f"  [DRY RUN] Would send to: {lead['email']}")
            print(f"  Subject: {personalized_subject}")
            print(f"  Body preview: {personalized_body[:100]}...")
            print()
            lead["last_email_status"] = "dry_run"
            lead["last_email_date"] = datetime.now().strftime("%Y-%m-%d")
            results.append(lead)
            continue

        print(f"  [{i+1}/{len(leads_with_email)}] Sending to {first_name} ({lead['email']})...", end=" ")
        success = send_email(
            to_email=lead["email"],
            subject=personalized_subject,
            body=personalized_body,
            from_name=sender.get("name", ""),
        )

        if success:
            print("✓ Sent")
            sent_count += 1
            lead["last_email_status"] = "sent"
            lead["last_email_date"] = datetime.now().strftime("%Y-%m-%d")
            lead["email_sequence_step"] = lead.get("email_sequence_step", 0) + 1
            if lead.get("status") == "new":
                lead["status"] = "contacted"
        else:
            print("✗ Failed")
            skipped_count += 1
            lead["last_email_status"] = "failed"

        results.append(lead)

        # Delay between sends (avoids spam detection)
        if i < len(leads_with_email) - 1:
            time.sleep(delay_seconds)

    print(f"\n[+] Campaign complete: {sent_count} sent, {skipped_count} failed/skipped")
    return results


def get_leads_due_for_followup(leads: list[dict], days_since_last: int = 3) -> list[dict]:
    """
    Returns leads that haven't been emailed recently and need a follow-up.

    Args:
        leads: All leads
        days_since_last: How many days after last contact to follow up (default 3)
    """
    from datetime import datetime, timedelta

    due = []
    cutoff = datetime.now() - timedelta(days=days_since_last)

    for lead in leads:
        last_email = lead.get("last_email_date")
        status = lead.get("status", "new")
        step = lead.get("email_sequence_step", 0)

        # Skip if replied, unsubscribed, or max follow-ups reached
        if status in ("replied", "unsubscribed", "converted", "not_interested"):
            continue
        if step >= 3:  # Max 3 emails: initial + 2 follow-ups
            continue
        if not lead.get("email"):
            continue

        # Due if never contacted, or last contacted before cutoff
        if not last_email:
            due.append(lead)
        else:
            last_dt = datetime.strptime(last_email, "%Y-%m-%d")
            if last_dt < cutoff:
                due.append(lead)

    return due
