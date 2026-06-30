# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Honeypot Detector
# ─────────────────────────────────────────────────────────────────────────────
"""
Detects honeypot candidates with subtly impossible profiles.

Per the challenge docs, ~80 candidates have profiles like:
- 8 years experience at a company founded 3 years ago
- "Expert" proficiency in 10 skills with 0 years used
- Claimed experience wildly inconsistent with career history dates

These are forced to relevance tier 0 in the ground truth.
Submissions with >10% honeypots in top 100 are disqualified.
"""

from datetime import datetime, date

from config import (
    HONEYPOT_EXPERT_SKILL_THRESHOLD,
    HONEYPOT_SKILL_DURATION_MIN,
    HONEYPOT_EXP_MISMATCH_RATIO,
    HONEYPOT_MAX_ZERO_DURATION_EXPERTS,
)


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Check if a candidate is a honeypot.

    Returns:
        (is_honeypot: bool, reasons: list[str])
    """
    reasons = []
    flags = 0

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # ── Check 1: Expert skills with 0 or near-0 duration ──────────────────
    zero_duration_experts = 0
    low_duration_experts = 0
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        if prof == "expert":
            if duration == 0:
                zero_duration_experts += 1
            elif duration < HONEYPOT_SKILL_DURATION_MIN:
                low_duration_experts += 1

    if zero_duration_experts >= HONEYPOT_MAX_ZERO_DURATION_EXPERTS:
        reasons.append(
            f"Expert in {zero_duration_experts} skills with 0 months duration"
        )
        flags += 2

    if (zero_duration_experts + low_duration_experts) >= HONEYPOT_EXPERT_SKILL_THRESHOLD:
        reasons.append(
            f"Expert/near-expert in {zero_duration_experts + low_duration_experts} "
            f"skills with minimal duration"
        )
        flags += 2

    # ── Check 2: Claimed experience vs career history mismatch ────────────
    claimed_yrs = profile.get("years_of_experience", 0)
    if career:
        total_career_months = sum(
            c.get("duration_months", 0) for c in career
        )
        total_career_yrs = total_career_months / 12.0

        if total_career_yrs > 0 and claimed_yrs > 0:
            ratio = claimed_yrs / total_career_yrs
            if ratio > HONEYPOT_EXP_MISMATCH_RATIO:
                reasons.append(
                    f"Claimed {claimed_yrs}yr experience but career history "
                    f"sums to {total_career_yrs:.1f}yr (ratio: {ratio:.1f}x)"
                )
                flags += 1

            # Also check the reverse — career sum vastly exceeds claimed
            if total_career_yrs / claimed_yrs > HONEYPOT_EXP_MISMATCH_RATIO:
                reasons.append(
                    f"Career history sums to {total_career_yrs:.1f}yr but "
                    f"claims only {claimed_yrs}yr experience"
                )
                flags += 1

    # ── Check 3: Impossible date ranges in career ─────────────────────────
    for job in career:
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        duration = job.get("duration_months", 0)

        if start_str and end_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)

                # Duration claimed vs actual date range
                if actual_months > 0 and duration > 0:
                    if abs(duration - actual_months) > max(12, actual_months * 0.5):
                        reasons.append(
                            f"Job at {job.get('company')}: claims {duration}mo "
                            f"but dates span {actual_months}mo"
                        )
                        flags += 1

                # End before start
                if end < start:
                    reasons.append(
                        f"Job at {job.get('company')}: end date before start date"
                    )
                    flags += 2
            except (ValueError, TypeError):
                pass

        # Future start dates (more than 6 months out)
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                if start > date(2026, 12, 31):
                    reasons.append(
                        f"Job at {job.get('company')}: start date in far future ({start_str})"
                    )
                    flags += 2
            except (ValueError, TypeError):
                pass

    # ── Check 4: Too many expert skills relative to experience ────────────
    expert_count = sum(
        1 for s in skills if s.get("proficiency", "").lower() == "expert"
    )
    if expert_count >= 8 and claimed_yrs < 4:
        reasons.append(
            f"Expert in {expert_count} skills with only {claimed_yrs}yr experience"
        )
        flags += 2

    # ── Check 5: Skill assessment scores impossibly high with low proficiency ─
    assessments = signals.get("skill_assessment_scores", {})
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "").lower()
        if name in assessments:
            score = assessments[name]
            # Beginner claiming 95+ assessment score
            if prof == "beginner" and score > 95:
                flags += 1
                reasons.append(
                    f"Skill '{name}': beginner proficiency but {score}/100 assessment"
                )

    # ── Check 6: Profile completeness vs missing data ─────────────────────
    completeness = signals.get("profile_completeness_score", 0)
    if completeness > 95:
        # Check if profile actually has substance
        summary = profile.get("summary", "")
        headline = profile.get("headline", "")
        if len(summary) < 20 and len(headline) < 10 and len(skills) < 2:
            reasons.append(
                f"Profile completeness {completeness}% but nearly empty profile"
            )
            flags += 1

    # ── Decision ──────────────────────────────────────────────────────────
    is_honeypot = flags >= 3
    return is_honeypot, reasons
