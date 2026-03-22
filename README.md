# Lead Generation & Outreach System
**Replace cold calling with automated email + text outreach.**

## Quick Start (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your credentials
```bash
cp .env.example .env
# Edit .env with your email and Twilio info
```

### 3. Run setup
```bash
python main.py setup
```

### 4. Find leads
```bash
python main.py find
```

### 5. Email them
```bash
python main.py email
```

### 6. Text them
```bash
python main.py sms
```

---

## Full Workflow (No Cold Calling)

```
python main.py setup      ← Enter your name, email, service, target market
python main.py find       ← Search Google Maps, Apollo.io, or import CSV
python main.py email      ← Send first email to all new leads
python main.py sms        ← Text them too (gets 3x more responses)
# Wait 3 days...
python main.py followup   ← Auto send follow-up to non-responders
python main.py pipeline   ← See who replied
python main.py status     ← Mark replies: "John Smith" → replied/converted
```

---

## Free Accounts You Need

| Service | What it does | Cost |
|---------|-------------|------|
| **Gmail App Password** | Send emails | Free |
| **Twilio** | Send SMS texts | Free trial ($15 credit) |
| **Hunter.io** | Find emails from website domain | Free (25/month) |
| **Apollo.io** | B2B leads with email + phone | Free tier available |
| **SerpAPI** | Real Google Maps results | Free (100/month) |

---

## Lead Sources

### Option 1: Google Maps (Free)
Finds local businesses by type + location.
```bash
python main.py find
# Choose option 1, enter "plumber" and "Houston TX"
```

### Option 2: Apollo.io (Best for B2B)
Finds decision-makers with email + phone.
```bash
# Set APOLLO_API_KEY in .env
python main.py find
# Choose option 2
```

### Option 3: Import Your Own CSV
```bash
python main.py import-csv
# or: python main.py find → Choose option 3
```
CSV columns: `name, first_name, last_name, email, phone, company, business_type, location`

---

## Email Templates (Built-In)

| Template | When to use |
|----------|------------|
| Initial outreach | First email ever |
| LinkedIn-style | Mention you found their business |
| Free value | Offer something useful to get a reply |
| Follow-up #1 | 3 days after initial |
| Follow-up #2 | 5 days after follow-up #1 |

---

## SMS Templates (Built-In)

| Template | When to use |
|----------|------------|
| `initial` | First text |
| `followup` | After no reply |
| `value_drop` | Offer free resource |
| `appointment` | Ask for a call |
| `after_no_email_reply` | Follow up via text when email ignored |

---

## Track Your Pipeline
```bash
python main.py pipeline          # See all leads
python main.py stats             # See reply rates
python main.py status            # Update a lead's status
```

Lead statuses: `new → contacted → replied → meeting_scheduled → converted`

---

## Tips for Getting Replies

1. **Text AND email** — texts get 3-5x better open rates than email
2. **Keep emails short** — 3-4 sentences max. Nobody reads long cold emails
3. **Follow up 3 times** — 80% of sales happen after the 3rd follow-up
4. **Personalize** — use their first name and mention their business type
5. **Offer something free** — an audit, a tip sheet, a video review
6. **Reply to your own thread** — follow-ups in the same thread get more replies
7. **Send Tuesday-Thursday 8am-10am** — best email open times
