"""
Lead Finder Module
Finds potential leads from Google Maps, LinkedIn, and manual CSV import.
No cold calling needed - just find leads and their contact info.
"""

import csv
import json
import os
import time
import requests
from urllib.parse import quote_plus
from datetime import datetime


HUNTER_API_KEY = os.getenv("HUNTER_API_KEY", "")
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "")


def search_google_maps(business_type: str, location: str) -> list[dict]:
    """
    Search Google Maps for businesses via SerpAPI or direct scraping.
    Returns list of leads with name, address, phone, website.

    Usage: search_google_maps("plumber", "Houston TX")
    """
    print(f"\n[+] Searching Google Maps for '{business_type}' in '{location}'...")
    leads = []

    # Use the free Google Maps search (no API key needed for basic scraping)
    query = f"{business_type} in {location}"
    url = f"https://local.google.com/maps/search/{quote_plus(query)}"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            print(f"  Connected to Google Maps search")
            print(f"  NOTE: For production use, use SerpAPI (serpapi.com) or")
            print(f"  Google Places API to get structured results.")
            print(f"  Adding placeholder leads for demonstration...")

        # Placeholder leads so the system works end-to-end for demo
        # In production, replace with real API calls (see scrape_with_serpapi below)
        leads = _demo_leads_for(business_type, location)

    except Exception as e:
        print(f"  [!] Search error: {e}")
        leads = _demo_leads_for(business_type, location)

    print(f"  Found {len(leads)} leads")
    return leads


def _demo_leads_for(business_type: str, location: str) -> list[dict]:
    """Returns sample leads - replace with real API in production."""
    return [
        {
            "name": f"Demo {business_type.title()} Co. 1",
            "business_type": business_type,
            "location": location,
            "address": f"123 Main St, {location}",
            "phone": "",
            "website": "",
            "email": "",
            "source": "google_maps_demo",
            "status": "new",
            "notes": "",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
        },
        {
            "name": f"Demo {business_type.title()} Co. 2",
            "business_type": business_type,
            "location": location,
            "address": f"456 Oak Ave, {location}",
            "phone": "",
            "website": "",
            "email": "",
            "source": "google_maps_demo",
            "status": "new",
            "notes": "",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
        },
    ]


def scrape_with_serpapi(business_type: str, location: str, api_key: str) -> list[dict]:
    """
    Use SerpAPI to get real Google Maps results.
    Sign up free at serpapi.com (100 searches/month free).

    Usage: scrape_with_serpapi("plumber", "Houston TX", "your_serpapi_key")
    """
    print(f"\n[+] Using SerpAPI to search '{business_type}' in '{location}'...")
    leads = []

    url = "https://serpapi.com/search"
    params = {
        "engine": "google_maps",
        "q": f"{business_type} {location}",
        "api_key": api_key,
        "type": "search",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        for place in data.get("local_results", []):
            leads.append({
                "name": place.get("title", ""),
                "business_type": business_type,
                "location": location,
                "address": place.get("address", ""),
                "phone": place.get("phone", ""),
                "website": place.get("website", ""),
                "email": "",
                "source": "serpapi_google_maps",
                "status": "new",
                "notes": "",
                "added_date": datetime.now().strftime("%Y-%m-%d"),
            })

        print(f"  Found {len(leads)} leads via SerpAPI")

    except Exception as e:
        print(f"  [!] SerpAPI error: {e}")

    return leads


def find_email_with_hunter(domain: str, first_name: str = "", last_name: str = "") -> str:
    """
    Find a business email using Hunter.io.
    Free tier: 25 searches/month. Sign up at hunter.io

    Usage: find_email_with_hunter("acme.com", "John", "Smith")
    """
    if not HUNTER_API_KEY:
        return ""

    try:
        if first_name and last_name:
            url = "https://api.hunter.io/v2/email-finder"
            params = {
                "domain": domain,
                "first_name": first_name,
                "last_name": last_name,
                "api_key": HUNTER_API_KEY,
            }
        else:
            url = "https://api.hunter.io/v2/domain-search"
            params = {"domain": domain, "api_key": HUNTER_API_KEY, "limit": 1}

        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if first_name and last_name:
            return data.get("data", {}).get("email", "")
        else:
            emails = data.get("data", {}).get("emails", [])
            return emails[0].get("value", "") if emails else ""

    except Exception:
        return ""


def find_leads_from_apollo(job_title: str, industry: str, location: str) -> list[dict]:
    """
    Find leads using Apollo.io API (B2B contacts with email + phone).
    Free tier available at apollo.io

    Usage: find_leads_from_apollo("Owner", "Plumbing", "Houston TX")
    """
    if not APOLLO_API_KEY:
        print("  [!] Set APOLLO_API_KEY in .env to use Apollo.io")
        return []

    print(f"\n[+] Searching Apollo.io for {job_title} in {industry}...")

    url = "https://api.apollo.io/v1/mixed_people/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": APOLLO_API_KEY,
    }
    payload = {
        "person_titles": [job_title],
        "q_keywords": industry,
        "person_locations": [location],
        "page": 1,
        "per_page": 25,
    }

    leads = []
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        data = resp.json()

        for person in data.get("people", []):
            org = person.get("organization", {}) or {}
            leads.append({
                "name": person.get("name", ""),
                "business_type": industry,
                "location": location,
                "address": "",
                "phone": person.get("phone_numbers", [{}])[0].get("sanitized_number", "") if person.get("phone_numbers") else "",
                "website": org.get("website_url", ""),
                "email": person.get("email", ""),
                "first_name": person.get("first_name", ""),
                "last_name": person.get("last_name", ""),
                "company": org.get("name", ""),
                "title": person.get("title", ""),
                "source": "apollo",
                "status": "new",
                "notes": "",
                "added_date": datetime.now().strftime("%Y-%m-%d"),
            })

        print(f"  Found {len(leads)} leads via Apollo.io")

    except Exception as e:
        print(f"  [!] Apollo error: {e}")

    return leads


def import_leads_from_csv(filepath: str) -> list[dict]:
    """
    Import your own lead list from a CSV file.
    CSV should have columns: name, email, phone, company, notes

    Usage: import_leads_from_csv("my_leads.csv")
    """
    leads = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lead = {
                    "name": row.get("name", ""),
                    "business_type": row.get("business_type", ""),
                    "location": row.get("location", ""),
                    "address": row.get("address", ""),
                    "phone": row.get("phone", ""),
                    "website": row.get("website", ""),
                    "email": row.get("email", ""),
                    "first_name": row.get("first_name", ""),
                    "last_name": row.get("last_name", ""),
                    "company": row.get("company", row.get("name", "")),
                    "source": "csv_import",
                    "status": row.get("status", "new"),
                    "notes": row.get("notes", ""),
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                }
                leads.append(lead)
        print(f"[+] Imported {len(leads)} leads from {filepath}")
    except FileNotFoundError:
        print(f"[!] File not found: {filepath}")
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
    return leads


def save_leads(leads: list[dict], filepath: str = "data/leads.json"):
    """Save leads to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    existing = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)

    # Deduplicate by name + location
    existing_keys = {(l.get("name", ""), l.get("location", "")) for l in existing}
    new_leads = [l for l in leads if (l.get("name", ""), l.get("location", "")) not in existing_keys]

    all_leads = existing + new_leads
    with open(filepath, "w") as f:
        json.dump(all_leads, f, indent=2)

    print(f"[+] Saved {len(new_leads)} new leads ({len(all_leads)} total) to {filepath}")
    return all_leads


def load_leads(filepath: str = "data/leads.json") -> list[dict]:
    """Load leads from JSON file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return json.load(f)
