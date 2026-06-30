# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Reasoning Generator
# ─────────────────────────────────────────────────────────────────────────────
"""
Generates specific, honest reasoning for each ranked candidate.
Per the submission spec, reasoning should be:
- Specific to each candidate (not templated)
- Honest about strengths and weaknesses
- Reference actual skills and career details
- Not hallucinate skills the candidate doesn't have
"""

from config import (
    MUST_HAVE_SKILLS,
    CORE_AI_SKILLS,
    NLP_IR_SKILLS,
    CONSULTING_FIRMS,
    PREFERRED_LOCATIONS,
    GOOD_LOCATIONS_INDIA,
)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _get_relevant_skills(candidate: dict) -> list[str]:
    """Get list of relevant AI/ML skills the candidate actually has."""
    skills = candidate.get("skills", [])
    all_relevant = MUST_HAVE_SKILLS | CORE_AI_SKILLS | NLP_IR_SKILLS
    result = []
    for skill in skills:
        name = skill.get("name", "")
        if _normalize(name) in all_relevant:
            prof = skill.get("proficiency", "")
            result.append(f"{name} ({prof})")
    return result


def _get_must_have_skills(candidate: dict) -> list[str]:
    """Get must-have skills the candidate has."""
    skills = candidate.get("skills", [])
    result = []
    for skill in skills:
        name = skill.get("name", "")
        if _normalize(name) in MUST_HAVE_SKILLS:
            result.append(name)
    return result


def _is_consulting_only(candidate: dict) -> bool:
    """Check if entire career is at consulting firms."""
    career = candidate.get("career_history", [])
    if not career:
        return False
    for job in career:
        company = _normalize(job.get("company", ""))
        if company not in CONSULTING_FIRMS:
            return False
    return True


def _get_location_context(candidate: dict) -> str:
    """Get a brief location context string."""
    profile = candidate.get("profile", {})
    location = profile.get("location", "Unknown")
    country = profile.get("country", "")

    loc_lower = _normalize(location)
    if any(l in loc_lower for l in PREFERRED_LOCATIONS):
        return f"{location} (preferred location)"
    elif any(l in loc_lower for l in GOOD_LOCATIONS_INDIA):
        return f"{location}, India"
    elif _normalize(country) == "india":
        return f"{location}, India"
    else:
        return f"{location}, {country}"


def generate_reasoning(candidate: dict, rank: int, score: float,
                       dimension_scores: dict) -> str:
    """
    Generate a specific 1-2 sentence reasoning for a ranked candidate.

    Args:
        candidate: Full candidate dict
        rank: Their rank (1-100)
        score: Their final composite score
        dimension_scores: Breakdown by scoring dimension

    Returns:
        str — reasoning text
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yrs = profile.get("years_of_experience", 0)
    location = _get_location_context(candidate)

    relevant_skills = _get_relevant_skills(candidate)
    must_have = _get_must_have_skills(candidate)

    response_rate = signals.get("recruiter_response_rate", 0)
    github = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)
    notice_days = signals.get("notice_period_days", 0)

    # Build reasoning parts
    parts = []

    # ── Opening: role + experience ────────────────────────────────────────
    parts.append(f"{title} at {company} with {yrs:.1f} years of experience")

    # ── Skills highlight ──────────────────────────────────────────────────
    if must_have:
        parts.append(f"Has must-have skills: {', '.join(must_have[:4])}")
    elif relevant_skills:
        skill_names = [s.split(" (")[0] for s in relevant_skills[:5]]
        parts.append(f"Relevant skills include {', '.join(skill_names)}")
    else:
        parts.append("Limited direct AI/ML skill match")

    # ── Strengths ─────────────────────────────────────────────────────────
    strengths = []
    title_score = dimension_scores.get("title_fit", 0)
    career_score = dimension_scores.get("career_quality", 0)
    skills_score = dimension_scores.get("skills_match", 0)

    if title_score >= 80:
        strengths.append("strong title alignment")
    if career_score >= 70:
        strengths.append("solid career in product roles")
    if skills_score >= 70:
        strengths.append("deep skill match")
    if github > 50:
        strengths.append(f"active on GitHub (score: {github:.0f})")
    if response_rate >= 0.7:
        strengths.append("highly responsive to recruiters")
    if open_to_work:
        strengths.append("actively looking")

    if strengths:
        parts.append("Strengths: " + ", ".join(strengths[:3]))

    # ── Concerns (honest assessment) ──────────────────────────────────────
    concerns = []
    if _is_consulting_only(candidate):
        concerns.append("entire career at consulting firms")
    if response_rate < 0.3:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if notice_days > 90:
        concerns.append(f"long notice period ({notice_days}d)")

    exp_score = dimension_scores.get("experience_band", 0)
    if exp_score < 40:
        if yrs < 3:
            concerns.append("below preferred experience range")
        elif yrs > 12:
            concerns.append("above typical experience range for role")

    loc_score = dimension_scores.get("location_fit", 0)
    if loc_score < 40:
        concerns.append(f"located outside India ({location})")

    if concerns:
        parts.append("Consideration: " + "; ".join(concerns[:2]))

    # Join into 1-2 sentences
    reasoning = ". ".join(parts) + "."

    # Ensure not too long (keep it concise)
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning
