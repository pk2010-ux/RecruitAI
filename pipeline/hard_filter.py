# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Hard Filter (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────
"""
Stage 1: Fast elimination of obviously unfit candidates.
This runs on all 100K candidates so it must be fast (pure Python, no models).

Candidates are KEPT if they pass all filters.
Candidates are DROPPED if they fail any hard filter.
"""

from pipeline.honeypot_detector import detect_honeypot
from config import (
    TITLE_SCORES,
    CORE_AI_SKILLS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    NLP_IR_SKILLS,
    ML_PRODUCTION_KEYWORDS,
)


def _normalize(text: str) -> str:
    """Lowercase and strip."""
    return text.strip().lower()


def _get_title_score(title: str) -> int:
    """Look up title score, with fuzzy fallback."""
    t = _normalize(title)
    if t in TITLE_SCORES:
        return TITLE_SCORES[t]

    # Fuzzy matching: check if any key is contained in the title
    best = 0
    for key, score in TITLE_SCORES.items():
        if key in t or t in key:
            best = max(best, score)
    return best


def _count_relevant_skills(candidate: dict) -> int:
    """Count how many skills are in the AI/ML/tech skill sets."""
    skills = candidate.get("skills", [])
    all_relevant = CORE_AI_SKILLS | MUST_HAVE_SKILLS | NICE_TO_HAVE_SKILLS | NLP_IR_SKILLS
    count = 0
    for skill in skills:
        name = _normalize(skill.get("name", ""))
        if name in all_relevant:
            count += 1
    return count


def _has_ml_career_signal(candidate: dict) -> bool:
    """Check if any career description mentions ML/AI production work."""
    career = candidate.get("career_history", [])
    for job in career:
        desc = _normalize(job.get("description", ""))
        for keyword in ML_PRODUCTION_KEYWORDS:
            if keyword in desc:
                return True
    return False


def _get_best_title_from_career(candidate: dict) -> int:
    """Get the best title score across all career entries."""
    career = candidate.get("career_history", [])
    best = 0
    for job in career:
        title = job.get("title", "")
        score = _get_title_score(title)
        best = max(best, score)
    return best


def hard_filter(candidate: dict) -> tuple[bool, str]:
    """
    Apply hard filters to a candidate.

    Returns:
        (passes: bool, reason: str)
        - passes=True means the candidate moves to Stage 2
        - passes=False means they're eliminated, reason explains why
    """
    profile = candidate.get("profile", {})
    cid = candidate.get("candidate_id", "UNKNOWN")

    # ── Filter 1: Honeypot detection ──────────────────────────────────────
    is_honeypot, hp_reasons = detect_honeypot(candidate)
    if is_honeypot:
        return False, f"Honeypot: {'; '.join(hp_reasons)}"

    # ── Filter 2: Zero experience ─────────────────────────────────────────
    yrs = profile.get("years_of_experience", 0)
    if yrs <= 0:
        return False, "Zero years of experience"

    # ── Filter 3: Combined title + skills + career signal ─────────────────
    # We use a soft filter here: a candidate passes if they have EITHER
    # a relevant title, OR relevant skills, OR ML signals in career text.
    # This prevents us from accidentally filtering out "hidden gem" candidates
    # who have ML experience but non-obvious titles.

    current_title_score = _get_title_score(profile.get("current_title", ""))
    best_career_title_score = _get_best_title_from_career(candidate)
    relevant_skill_count = _count_relevant_skills(candidate)
    has_ml_signal = _has_ml_career_signal(candidate)

    # A candidate passes if ANY of these are true:
    # 1. Their current or any past title scores >= 25 (at least partially related)
    # 2. They have >= 2 relevant technical skills
    # 3. Their career descriptions mention ML/AI work
    title_ok = current_title_score >= 25 or best_career_title_score >= 25
    skills_ok = relevant_skill_count >= 2
    career_ok = has_ml_signal

    if not (title_ok or skills_ok or career_ok):
        return False, (
            f"No tech/AI signal: title_score={current_title_score}, "
            f"best_career_title={best_career_title_score}, "
            f"relevant_skills={relevant_skill_count}, "
            f"ml_career_signal={has_ml_signal}"
        )

    return True, "Passed"


def apply_hard_filters(candidates: list[dict]) -> tuple[list[dict], dict]:
    """
    Apply hard filters to a batch of candidates.

    Returns:
        (passed: list[dict], stats: dict)
    """
    passed = []
    stats = {
        "total": 0,
        "passed": 0,
        "filtered_honeypot": 0,
        "filtered_zero_exp": 0,
        "filtered_no_signal": 0,
    }

    for candidate in candidates:
        stats["total"] += 1
        passes, reason = hard_filter(candidate)

        if passes:
            stats["passed"] += 1
            passed.append(candidate)
        else:
            if "Honeypot" in reason:
                stats["filtered_honeypot"] += 1
            elif "Zero years" in reason:
                stats["filtered_zero_exp"] += 1
            else:
                stats["filtered_no_signal"] += 1

    return passed, stats
