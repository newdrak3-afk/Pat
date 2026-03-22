"""
Email Templates - Proven outreach scripts that get responses.
No cold calling needed. Just send these and follow up.
"""

# ─────────────────────────────────────────────
# TEMPLATE 1: Initial Cold Email (short & direct)
# ─────────────────────────────────────────────
INITIAL_OUTREACH = {
    "subject": "Quick question about {company}",
    "body": """Hi {first_name},

I came across {company} and wanted to reach out directly.

I help {business_type} businesses in {location} {value_prop}.

Would it be worth a quick 10-minute chat this week to see if it's a fit?

Best,
{sender_name}
{sender_phone}
{sender_email}

P.S. No pressure at all — just reply "not interested" and I'll never bother you again.""",
}

# ─────────────────────────────────────────────
# TEMPLATE 2: Follow-Up #1 (3 days after initial)
# ─────────────────────────────────────────────
FOLLOWUP_1 = {
    "subject": "Re: Quick question about {company}",
    "body": """Hi {first_name},

Just wanted to bump this up in case it got buried.

I know you're busy — I'll keep this short. We've helped {business_type} businesses like yours {result_example}.

If that sounds interesting, I'd love to show you how. 15 minutes on a call or I can send over more info — whatever works for you.

{sender_name}
{sender_phone}""",
}

# ─────────────────────────────────────────────
# TEMPLATE 3: Follow-Up #2 (5 days after follow-up 1)
# ─────────────────────────────────────────────
FOLLOWUP_2 = {
    "subject": "Last follow-up — {company}",
    "body": """Hi {first_name},

I don't want to keep filling your inbox, so this is my last email.

If now's not a good time or it's not a fit, totally understood — no hard feelings.

But if you ever want to explore {value_prop}, feel free to reach back out anytime.

Wishing you and {company} all the best.

{sender_name}
{sender_phone}
{sender_email}""",
}

# ─────────────────────────────────────────────
# TEMPLATE 4: "Referral" / Warm Intro Email
# ─────────────────────────────────────────────
REFERRAL_INTRO = {
    "subject": "Intro from {referral_name}",
    "body": """Hi {first_name},

{referral_name} suggested I reach out to you.

I've been working with {referral_name} to {value_prop}, and they thought you might benefit from the same.

I'd love to hop on a quick call and see if I can help {company} the same way. Would {day_option_1} or {day_option_2} work for a 15-minute call?

{sender_name}
{sender_phone}
{sender_email}""",
}

# ─────────────────────────────────────────────
# TEMPLATE 5: LinkedIn-style "Connection" Email
# ─────────────────────────────────────────────
LINKEDIN_STYLE = {
    "subject": "Noticed {company} on [platform/Google]",
    "body": """Hi {first_name},

I noticed {company} while looking for {business_type} businesses in {location} and was impressed by {specific_detail}.

I specialize in helping businesses like yours {value_prop}, and I think there's a real opportunity here.

Would you be open to a brief conversation? I promise to keep it under 10 minutes and make it worth your time.

{sender_name}
{sender_phone}
{sender_email}""",
}

# ─────────────────────────────────────────────
# TEMPLATE 6: "Free Value" Email (offer something first)
# ─────────────────────────────────────────────
FREE_VALUE = {
    "subject": "Free {offer} for {company}",
    "body": """Hi {first_name},

I put together a free {offer} for {company} — no strings attached.

{offer_details}

You can use it whether we ever work together or not. Just let me know if you'd like me to send it over.

{sender_name}
{sender_phone}
{sender_email}""",
}


# ─────────────────────────────────────────────
# VALUE PROPOSITIONS by business type
# Customize these for your service
# ─────────────────────────────────────────────
VALUE_PROPS = {
    "plumber": "get more booked jobs without spending more on ads",
    "roofer": "generate 10-20 more roofing leads per month",
    "realtor": "get more listings and close deals faster",
    "restaurant": "fill more tables and reduce no-shows",
    "dentist": "attract more new patients every month",
    "contractor": "win more bids and grow their pipeline",
    "lawyer": "bring in more qualified clients consistently",
    "default": "grow their revenue and get more customers",
}

RESULT_EXAMPLES = {
    "plumber": "book 15+ extra service calls per month",
    "roofer": "generate $50k+ in new roofing contracts",
    "realtor": "close 2-3 extra deals per quarter",
    "restaurant": "increase weekly covers by 20%",
    "dentist": "add 20+ new patients per month",
    "contractor": "win $100k+ in additional contracts",
    "lawyer": "bring in 10+ qualified leads per month",
    "default": "significantly grow their revenue",
}


def fill_template(template: dict, **kwargs) -> dict:
    """Fill a template with lead data."""
    subject = template["subject"].format(**{k: v or "" for k, v in kwargs.items()})
    body = template["body"].format(**{k: v or "" for k, v in kwargs.items()})
    return {"subject": subject, "body": body}


def get_template_for_sequence(sequence_step: int) -> dict:
    """Get the right template based on outreach step (0=initial, 1=followup1, 2=followup2)."""
    sequence = [INITIAL_OUTREACH, FOLLOWUP_1, FOLLOWUP_2]
    if sequence_step < len(sequence):
        return sequence[sequence_step]
    return FOLLOWUP_2


def build_email_context(lead: dict, sender: dict) -> dict:
    """Build the template variables from lead and sender data."""
    biz_type = lead.get("business_type", "default").lower()

    return {
        "first_name": lead.get("first_name") or lead.get("name", "there").split()[0],
        "company": lead.get("company") or lead.get("name", "your company"),
        "business_type": biz_type,
        "location": lead.get("location", "your area"),
        "value_prop": VALUE_PROPS.get(biz_type, VALUE_PROPS["default"]),
        "result_example": RESULT_EXAMPLES.get(biz_type, RESULT_EXAMPLES["default"]),
        "specific_detail": "your great reviews",
        "offer": "website audit",
        "offer_details": "It includes a quick look at what's working and what could drive more leads.",
        "day_option_1": "Tuesday afternoon",
        "day_option_2": "Thursday morning",
        "referral_name": "",
        "sender_name": sender.get("name", ""),
        "sender_phone": sender.get("phone", ""),
        "sender_email": sender.get("email", ""),
    }
