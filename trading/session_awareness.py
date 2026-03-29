"""
Session Awareness — Know which trading session is active.

Tags every trade with the active session so we can later analyze
which sessions are profitable and which aren't.

Sessions (UTC):
    Tokyo:   00:00 – 09:00 UTC
    London:  07:00 – 16:00 UTC
    New York: 13:00 – 22:00 UTC
    Overlap London/NY: 13:00 – 16:00 UTC
    Overlap Tokyo/London: 07:00 – 09:00 UTC
    Off-hours: 22:00 – 00:00 UTC
"""

from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class SessionInfo:
    """Current trading session details."""
    sessions: list[str]          # Active sessions (can overlap)
    primary: str                 # Most important active session
    is_overlap: bool             # True if two sessions overlap
    liquidity: str               # "high", "medium", "low"
    recommended_pairs: list[str] # Best pairs for this session


# Session windows in UTC hours
SESSIONS = {
    "tokyo":    (0, 9),
    "london":   (7, 16),
    "new_york": (13, 22),
}

# Best pairs per session (highest volume/tightest spreads)
SESSION_PAIRS = {
    "tokyo": [
        "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "AUD_USD",
        "NZD_USD", "NZD_JPY",
    ],
    "london": [
        "EUR_USD", "GBP_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
        "EUR_CHF", "GBP_CHF", "USD_CHF",
    ],
    "new_york": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "USD_CHF",
        "AUD_USD", "NZD_USD",
    ],
}


def get_current_session(now: datetime = None) -> SessionInfo:
    """Determine which trading session(s) are currently active."""
    if now is None:
        now = datetime.now(timezone.utc)

    hour = now.hour
    active = []

    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            active.append(name)

    is_overlap = len(active) >= 2

    if not active:
        return SessionInfo(
            sessions=["off_hours"],
            primary="off_hours",
            is_overlap=False,
            liquidity="low",
            recommended_pairs=[],
        )

    # Determine primary session and liquidity
    if is_overlap:
        # Overlap sessions have highest liquidity
        if "london" in active and "new_york" in active:
            primary = "london_ny_overlap"
            liquidity = "high"
        elif "tokyo" in active and "london" in active:
            primary = "tokyo_london_overlap"
            liquidity = "high"
        else:
            primary = active[0]
            liquidity = "medium"
    else:
        primary = active[0]
        liquidity = "medium" if primary == "tokyo" else "high"

    # Combine recommended pairs from active sessions
    pairs = set()
    for s in active:
        pairs.update(SESSION_PAIRS.get(s, []))

    return SessionInfo(
        sessions=active,
        primary=primary,
        is_overlap=is_overlap,
        liquidity=liquidity,
        recommended_pairs=sorted(pairs),
    )


def tag_trade_session(trade_info: dict, now: datetime = None) -> dict:
    """Add session metadata to a trade dict."""
    session = get_current_session(now)
    trade_info["session"] = session.primary
    trade_info["sessions_active"] = session.sessions
    trade_info["session_liquidity"] = session.liquidity
    return trade_info


def get_session_status() -> str:
    """Human-readable session status for Telegram."""
    session = get_current_session()
    lines = [
        "TRADING SESSIONS",
        "",
        f"  Active: {', '.join(s.replace('_', ' ').title() for s in session.sessions)}",
        f"  Primary: {session.primary.replace('_', ' ').title()}",
        f"  Overlap: {'Yes' if session.is_overlap else 'No'}",
        f"  Liquidity: {session.liquidity.upper()}",
    ]
    if session.recommended_pairs:
        pairs = [p.replace("_", "/") for p in session.recommended_pairs[:8]]
        lines.append(f"  Best Pairs: {', '.join(pairs)}")
    return "\n".join(lines)
