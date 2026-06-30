# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Behavioral Scorer
# ─────────────────────────────────────────────────────────────────────────────
"""
Computes a composite behavioral score from Redrob platform signals.
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
recruiter response rate is, for hiring purposes, not actually available.
"""

from datetime import datetime, date
from config import BEHAVIORAL_WEIGHTS


# Reference date for recency calculations
REFERENCE_DATE = date(2026, 6, 1)


def _recency_score(last_active_str: str) -> float:
    """
    Score 0-100 based on how recently the candidate was active.
    Active in last 30 days = 100, >12 months = 0.
    """
    if not last_active_str:
        return 0.0

    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_ago = (REFERENCE_DATE - last_active).days

        if days_ago <= 0:
            return 100.0
        elif days_ago <= 7:
            return 95.0
        elif days_ago <= 30:
            return 85.0
        elif days_ago <= 60:
            return 70.0
        elif days_ago <= 90:
            return 55.0
        elif days_ago <= 180:
            return 35.0
        elif days_ago <= 365:
            return 15.0
        else:
            return 5.0
    except (ValueError, TypeError):
        return 20.0


def _notice_period_score(days: int) -> float:
    """
    Score based on notice period. JD says <30 preferred, can buy out up to 30.
    >30 still in scope but bar gets higher.
    """
    if days <= 0:
        return 100.0  # Immediate joiner
    elif days <= 15:
        return 95.0
    elif days <= 30:
        return 85.0
    elif days <= 45:
        return 70.0
    elif days <= 60:
        return 55.0
    elif days <= 90:
        return 35.0
    elif days <= 120:
        return 20.0
    else:
        return 10.0


def _response_time_score(hours: float) -> float:
    """Score based on average response time. Lower is better."""
    if hours <= 0:
        return 50.0  # No data
    elif hours <= 2:
        return 100.0
    elif hours <= 6:
        return 90.0
    elif hours <= 12:
        return 80.0
    elif hours <= 24:
        return 70.0
    elif hours <= 48:
        return 55.0
    elif hours <= 72:
        return 40.0
    elif hours <= 168:  # 1 week
        return 25.0
    else:
        return 10.0


def _github_score(score: float) -> float:
    """Score based on GitHub activity. -1 means no GitHub linked."""
    if score < 0:
        return 30.0  # No GitHub — neutral, not a dealbreaker
    elif score >= 80:
        return 100.0
    elif score >= 60:
        return 85.0
    elif score >= 40:
        return 70.0
    elif score >= 20:
        return 55.0
    elif score >= 5:
        return 40.0
    else:
        return 25.0


def _verification_score(signals: dict) -> float:
    """Score based on verified email, phone, LinkedIn."""
    score = 0.0
    if signals.get("verified_email", False):
        score += 40.0
    if signals.get("verified_phone", False):
        score += 30.0
    if signals.get("linkedin_connected", False):
        score += 30.0
    return score


def _saved_by_recruiters_score(count: int) -> float:
    """Score based on how many recruiters saved this profile."""
    if count >= 20:
        return 100.0
    elif count >= 10:
        return 80.0
    elif count >= 5:
        return 60.0
    elif count >= 2:
        return 40.0
    elif count >= 1:
        return 25.0
    else:
        return 10.0


def _search_appearance_score(count: int) -> float:
    """Score based on search appearances in last 30 days."""
    if count >= 200:
        return 100.0
    elif count >= 100:
        return 80.0
    elif count >= 50:
        return 60.0
    elif count >= 20:
        return 40.0
    elif count >= 5:
        return 25.0
    else:
        return 10.0


def compute_behavioral_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute composite behavioral score (0-100) from Redrob signals.

    Returns:
        (score: float, components: dict) — score and breakdown
    """
    signals = candidate.get("redrob_signals", {})

    components = {}

    # Recruiter response rate (0-1 → 0-100)
    rrr = signals.get("recruiter_response_rate", 0)
    components["recruiter_response_rate"] = rrr * 100.0

    # Last active recency
    components["last_active_recency"] = _recency_score(
        signals.get("last_active_date", "")
    )

    # Open to work
    components["open_to_work"] = 100.0 if signals.get("open_to_work_flag", False) else 30.0

    # Interview completion rate (0-1 → 0-100)
    icr = signals.get("interview_completion_rate", 0)
    components["interview_completion_rate"] = icr * 100.0

    # Notice period
    components["notice_period"] = _notice_period_score(
        signals.get("notice_period_days", 90)
    )

    # Profile completeness (already 0-100)
    components["profile_completeness"] = signals.get("profile_completeness_score", 0)

    # GitHub activity
    components["github_activity"] = _github_score(
        signals.get("github_activity_score", -1)
    )

    # Response time
    components["avg_response_time"] = _response_time_score(
        signals.get("avg_response_time_hours", 999)
    )

    # Willing to relocate
    components["willing_to_relocate"] = (
        80.0 if signals.get("willing_to_relocate", False) else 40.0
    )

    # Verification trust
    components["verification_trust"] = _verification_score(signals)

    # Saved by recruiters
    components["saved_by_recruiters"] = _saved_by_recruiters_score(
        signals.get("saved_by_recruiters_30d", 0)
    )

    # Search appearances
    components["search_appearances"] = _search_appearance_score(
        signals.get("search_appearance_30d", 0)
    )

    # Compute weighted composite
    total_score = 0.0
    for key, weight in BEHAVIORAL_WEIGHTS.items():
        total_score += components.get(key, 0) * weight

    return total_score, components


def compute_behavioral_multiplier(behavioral_score: float) -> float:
    """
    Convert behavioral score (0-100) to a multiplier (0.5 to 1.2).

    This is applied multiplicatively to the final composite score so that
    inactive / unresponsive candidates get penalized even if they look
    great on paper.
    """
    # Map 0-100 → 0.5-1.2
    # 0 → 0.5, 50 → 0.85, 75 → 1.025, 100 → 1.2
    return 0.5 + (behavioral_score / 100.0) * 0.7
