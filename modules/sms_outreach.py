"""
SMS / Text Outreach Module
Send personalized text messages to leads via Twilio.
Way better response rates than cold calling.

Setup (free trial at twilio.com):
1. Sign up at twilio.com
2. Get your Account SID, Auth Token, and a free phone number
3. Add them to your .env file
"""

import os
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_PHONE_NUMBER", "")


# ─────────────────────────────────────────────
# SMS TEMPLATES (keep texts SHORT - under 160 chars ideally)
# ─────────────────────────────────────────────

SMS_TEMPLATES = {
    "initial": (
        "Hi {first_name}, this is {sender_name}. "
        "I help {business_type} businesses in {location} get more customers. "
        "Mind if I send you some info? Reply YES or NO."
    ),
    "followup": (
        "Hey {first_name}, just following up from my last message. "
        "Would love to share how I can help {company}. "
        "Is now a bad time? — {sender_name}"
    ),
    "value_drop": (
        "Hi {first_name}! I put together a free tip sheet for {business_type} "
        "businesses on getting more leads. Want me to text it to you? — {sender_name}"
    ),
    "appointment": (
        "Hi {first_name}, {sender_name} here. "
        "Do you have 10 mins this week for a quick call? "
        "Reply with a time that works and I'll make it happen!"
    ),
    "after_no_email_reply": (
        "Hi {first_name}, sent you an email about helping {company} grow. "
        "Did it land in spam? Worth a quick look — {sender_name} {sender_phone}"
    ),
}


def send_sms(to_phone: str, message: str) -> bool:
    """
    Send a single SMS via Twilio.

    Returns True if sent successfully.
    """
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM]):
        print("  [!] Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in .env")
        print("  [!] Get free trial at twilio.com (includes $15 credit)")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)

        # Clean phone number
        phone = to_phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if not phone.startswith("+"):
            phone = "+1" + phone  # Default to US

        message_obj = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=phone,
        )

        return message_obj.sid is not None

    except ImportError:
        print("  [!] Install twilio: pip install twilio")
        return False
    except Exception as e:
        error_str = str(e)
        if "21608" in error_str:
            print(f"  [!] Number {to_phone} is not verified (Twilio trial limitation)")
        elif "21211" in error_str:
            print(f"  [!] Invalid phone number: {to_phone}")
        else:
            print(f"  [!] Twilio error: {e}")
        return False


def send_sms_campaign(
    leads: list[dict],
    template_key: str,
    sender: dict,
    delay_seconds: int = 15,
    dry_run: bool = False,
) -> list[dict]:
    """
    Send SMS to a list of leads.

    Args:
        leads: Leads with 'phone' field
        template_key: One of 'initial', 'followup', 'value_drop', 'appointment', 'after_no_email_reply'
        sender: Dict with 'name', 'phone', 'email'
        delay_seconds: Wait between texts (avoids carrier flagging)
        dry_run: Preview without sending

    Returns:
        Updated leads list
    """
    template = SMS_TEMPLATES.get(template_key, SMS_TEMPLATES["initial"])

    leads_with_phone = [l for l in leads if l.get("phone")]
    leads_without_phone = [l for l in leads if not l.get("phone")]

    if leads_without_phone:
        print(f"  [!] {len(leads_without_phone)} leads have no phone number — skipping")

    print(f"\n[+] Starting SMS campaign ({template_key}) to {len(leads_with_phone)} leads")
    print(f"    Delay between texts: {delay_seconds}s\n")

    sent_count = 0
    results = []

    for i, lead in enumerate(leads_with_phone):
        first_name = lead.get("first_name") or lead.get("name", "there").split()[0]
        company = lead.get("company") or lead.get("name", "your company")

        message = template.format(
            first_name=first_name,
            company=company,
            business_type=lead.get("business_type", "your type of"),
            location=lead.get("location", "your area"),
            sender_name=sender.get("name", ""),
            sender_phone=sender.get("phone", ""),
            sender_email=sender.get("email", ""),
        )

        # Warn if message is too long
        if len(message) > 160:
            print(f"  [!] Message is {len(message)} chars (over 160, may split into 2 SMS)")

        if dry_run:
            print(f"  [DRY RUN] Would text {lead.get('phone')}: {message[:80]}...")
            lead["last_sms_status"] = "dry_run"
            results.append(lead)
            continue

        print(f"  [{i+1}/{len(leads_with_phone)}] Texting {first_name} ({lead.get('phone')})...", end=" ")
        success = send_sms(lead["phone"], message)

        if success:
            print("✓ Sent")
            sent_count += 1
            lead["last_sms_status"] = "sent"
            lead["last_sms_date"] = datetime.now().strftime("%Y-%m-%d")
            if lead.get("status") == "new":
                lead["status"] = "contacted"
        else:
            print("✗ Failed")
            lead["last_sms_status"] = "failed"

        results.append(lead)

        if i < len(leads_with_phone) - 1:
            time.sleep(delay_seconds)

    print(f"\n[+] SMS campaign complete: {sent_count} sent")
    return results


def build_sms_context(lead: dict, sender: dict) -> str:
    """Preview what an SMS would look like for a lead."""
    first_name = lead.get("first_name") or lead.get("name", "there").split()[0]
    company = lead.get("company") or lead.get("name", "your company")
    template = SMS_TEMPLATES["initial"]
    return template.format(
        first_name=first_name,
        company=company,
        business_type=lead.get("business_type", "your type of"),
        location=lead.get("location", "your area"),
        sender_name=sender.get("name", ""),
        sender_phone=sender.get("phone", ""),
        sender_email=sender.get("email", ""),
    )
