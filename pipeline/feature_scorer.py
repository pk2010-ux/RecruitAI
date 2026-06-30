# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Feature Scorer (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────
"""
Stage 2: Rule-based multi-dimensional scoring.
Each candidate is scored 0-100 on 7 dimensions, then combined with weights.

This is the primary ranking signal — semantic scoring (Stage 3) refines it.
"""

from config import (
    WEIGHTS,
    TITLE_SCORES,
    MUST_HAVE_SKILLS,
    CORE_AI_SKILLS,
    NICE_TO_HAVE_SKILLS,
    NLP_IR_SKILLS,
    ANTI_SIGNAL_SKILLS,
    CONSULTING_FIRMS,
    PRODUCT_COMPANIES,
    FICTIONAL_COMPANIES,
    PREFERRED_LOCATIONS,
    GOOD_LOCATIONS_INDIA,
    INDIA_COUNTRY_NAMES,
    IDEAL_EXP_MIN,
    IDEAL_EXP_MAX,
    IDEAL_EXP_CENTER,
    EXP_HARD_MIN,
    EXP_HARD_MAX,
    ML_PRODUCTION_KEYWORDS,
    NON_ML_KEYWORDS,
)
from pipeline.behavioral_scorer import compute_behavioral_score


def _normalize(text: str) -> str:
    return text.strip().lower()


def _get_title_score(title: str) -> int:
    """Look up title score with fuzzy fallback."""
    t = _normalize(title)
    if t in TITLE_SCORES:
        return TITLE_SCORES[t]
    best = 0
    for key, score in TITLE_SCORES.items():
        if key in t or t in key:
            best = max(best, score)
    return best


# ── Dimension 1: Title & Role Fit ─────────────────────────────────────────────

def score_title_fit(candidate: dict) -> float:
    """
    Score based on how well current and past titles match AI/ML roles.
    Current title weighted more heavily (60%) vs best past title (40%).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_score = _get_title_score(profile.get("current_title", ""))

    # Best past title
    best_past = 0
    for job in career:
        s = _get_title_score(job.get("title", ""))
        best_past = max(best_past, s)

    # Career progression bonus: if current title > past titles, shows growth
    progression_bonus = 0
    if current_score > best_past and current_score >= 60:
        progression_bonus = 5

    # Blend: 60% current, 40% best ever (including current)
    best_ever = max(current_score, best_past)
    score = 0.6 * current_score + 0.4 * best_ever + progression_bonus

    return min(100.0, score)


# ── Dimension 2: Skills Match ─────────────────────────────────────────────────

def score_skills_match(candidate: dict) -> float:
    """
    Score based on relevant skills, weighted by:
    - Must-have vs nice-to-have
    - Proficiency level
    - Duration (months of experience with skill)
    - Endorsements (social proof)
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    proficiency_multiplier = {
        "expert": 1.0,
        "advanced": 0.8,
        "intermediate": 0.55,
        "beginner": 0.3,
    }

    total_points = 0.0
    max_possible = 0.0
    must_have_count = 0
    core_ai_count = 0
    nlp_ir_count = 0
    anti_signal_count = 0

    for skill in skills:
        name = _normalize(skill.get("name", ""))
        prof = skill.get("proficiency", "beginner").lower()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        prof_mult = proficiency_multiplier.get(prof, 0.3)

        # Duration bonus: more months = more credible
        duration_mult = min(1.0, duration / 36.0)  # Max out at 3 years

        # Endorsement bonus (diminishing returns)
        endorse_mult = min(1.3, 1.0 + (endorsements / 50.0) * 0.3)

        if name in MUST_HAVE_SKILLS:
            base = 15.0
            must_have_count += 1
        elif name in NLP_IR_SKILLS:
            base = 10.0
            nlp_ir_count += 1
        elif name in CORE_AI_SKILLS:
            base = 8.0
            core_ai_count += 1
        elif name in NICE_TO_HAVE_SKILLS:
            base = 4.0
        elif name in ANTI_SIGNAL_SKILLS:
            anti_signal_count += 1
            continue  # Don't add points for anti-signal skills
        else:
            base = 1.0  # Unknown skill — minimal credit

        points = base * prof_mult * duration_mult * endorse_mult
        total_points += points
        max_possible += base  # Track for normalization

    # Normalize to 0-100 scale
    # A candidate with 3 must-have + 5 core AI + 3 NLP/IR is roughly "perfect"
    perfect_score = 3 * 15.0 + 5 * 8.0 + 3 * 10.0  # 115 points
    raw_score = (total_points / perfect_score) * 100.0

    # Anti-signal penalty: if most skills are non-tech, reduce score
    total_skills = len(skills)
    if total_skills > 0 and anti_signal_count / total_skills > 0.5:
        raw_score *= 0.5

    # Bonus for having must-have skills (critical per JD)
    if must_have_count >= 3:
        raw_score += 15
    elif must_have_count >= 2:
        raw_score += 10
    elif must_have_count >= 1:
        raw_score += 5

    # Bonus for NLP/IR skills (especially valued per JD)
    if nlp_ir_count >= 3:
        raw_score += 10
    elif nlp_ir_count >= 1:
        raw_score += 5

    # Also check skill assessment scores from Redrob
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        # Average assessment score for relevant skills
        relevant_scores = []
        for skill_name, score in assessments.items():
            n = _normalize(skill_name)
            if n in MUST_HAVE_SKILLS or n in CORE_AI_SKILLS or n in NLP_IR_SKILLS:
                relevant_scores.append(score)
        if relevant_scores:
            avg_assessment = sum(relevant_scores) / len(relevant_scores)
            raw_score += (avg_assessment / 100.0) * 10  # Up to 10 bonus points

    return min(100.0, max(0.0, raw_score))


# ── Dimension 3: Career Quality ───────────────────────────────────────────────

def score_career_quality(candidate: dict) -> float:
    """
    Score based on:
    - Product company vs consulting-only career
    - Career stability (not too much job-hopping)
    - ML/AI production signals in job descriptions
    - Company size diversity
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    score = 50.0  # Start at neutral

    consulting_count = 0
    product_count = 0
    total_jobs = len(career)
    total_months = 0
    ml_description_hits = 0
    non_ml_description_hits = 0

    for job in career:
        company = _normalize(job.get("company", ""))
        duration = job.get("duration_months", 0)
        desc = _normalize(job.get("description", ""))
        total_months += duration

        # Company classification
        if company in CONSULTING_FIRMS:
            consulting_count += 1
        elif company in PRODUCT_COMPANIES:
            product_count += 1
        # Fictional companies are neutral

        # Description analysis
        for kw in ML_PRODUCTION_KEYWORDS:
            if kw in desc:
                ml_description_hits += 1
                break  # Count per job, not per keyword

        for kw in NON_ML_KEYWORDS:
            if kw in desc:
                non_ml_description_hits += 1
                break

    # ── Consulting penalty ────────────────────────────────────────────────
    if consulting_count == total_jobs and total_jobs > 0:
        # Entire career at consulting firms — strong penalty per JD
        score -= 30
    elif consulting_count > 0 and product_count > 0:
        # Mixed — that's fine per JD ("consulting + product experience is fine")
        score += 5
    elif product_count > 0:
        score += 15

    # ── Job hopping penalty ───────────────────────────────────────────────
    if total_jobs > 0:
        avg_tenure_months = total_months / total_jobs
        if avg_tenure_months < 12:  # Less than 1 year average
            score -= 20  # Strong penalty — "title-chasers"
        elif avg_tenure_months < 18:
            score -= 10
        elif avg_tenure_months >= 30:
            score += 5  # Stability bonus

    # ── ML production signals ─────────────────────────────────────────────
    if ml_description_hits >= 3:
        score += 20
    elif ml_description_hits >= 2:
        score += 15
    elif ml_description_hits >= 1:
        score += 8

    # Non-ML penalty
    if total_jobs > 0 and non_ml_description_hits == total_jobs:
        score -= 20  # ALL jobs are non-ML

    # ── Career depth: having multiple roles shows progression ─────────────
    if total_jobs >= 3:
        score += 5
    if total_jobs >= 4:
        score += 3

    return min(100.0, max(0.0, score))


# ── Dimension 4: Experience Band ──────────────────────────────────────────────

def score_experience_band(candidate: dict) -> float:
    """
    Score based on proximity to the ideal 5-9 year band.
    Sweet spot is 6-8 per "how to read between the lines" section.
    """
    yrs = candidate.get("profile", {}).get("years_of_experience", 0)

    if yrs <= 0:
        return 0.0

    if IDEAL_EXP_MIN <= yrs <= IDEAL_EXP_MAX:
        # Within ideal band — score based on proximity to center
        distance = abs(yrs - IDEAL_EXP_CENTER)
        return max(70.0, 100.0 - distance * 7.5)
    elif yrs < IDEAL_EXP_MIN:
        # Under the band
        if yrs >= EXP_HARD_MIN:
            # 2-5 years: partial credit
            return 30.0 + (yrs - EXP_HARD_MIN) / (IDEAL_EXP_MIN - EXP_HARD_MIN) * 40.0
        else:
            return 10.0
    else:
        # Over the band
        if yrs <= EXP_HARD_MAX:
            # 9-18 years: gradual decline
            return max(20.0, 70.0 - (yrs - IDEAL_EXP_MAX) * 5.5)
        else:
            return 10.0


# ── Dimension 5: Education Fit ────────────────────────────────────────────────

def score_education_fit(candidate: dict) -> float:
    """
    Score based on educational background.
    CS/ML/Statistics/Math preferred. Tier 1-2 institutions bonus.
    """
    education = candidate.get("education", [])
    if not education:
        return 30.0  # No education listed — neutral (experience matters more)

    best_score = 0.0

    relevant_fields = {
        "computer science", "cs", "machine learning", "artificial intelligence",
        "data science", "statistics", "mathematics", "math",
        "information technology", "it", "software engineering",
        "electrical engineering", "electronics",
        "computational linguistics",
    }

    partially_relevant_fields = {
        "physics", "applied mathematics", "operations research",
        "bioinformatics", "computational biology",
    }

    tier_bonus = {
        "tier_1": 20,
        "tier_2": 12,
        "tier_3": 5,
        "tier_4": 0,
        "unknown": 2,
    }

    degree_bonus = {
        "ph.d": 15,
        "phd": 15,
        "m.tech": 12,
        "mtech": 12,
        "m.s": 10,
        "ms": 10,
        "m.sc": 10,
        "msc": 10,
        "m.e.": 10,
        "me": 10,
        "mba": 3,
        "b.tech": 8,
        "btech": 8,
        "b.e.": 7,
        "be": 7,
        "b.sc": 5,
        "bsc": 5,
        "b.s": 5,
    }

    for edu in education:
        field = _normalize(edu.get("field_of_study", ""))
        degree = _normalize(edu.get("degree", ""))
        tier = _normalize(edu.get("tier", "unknown"))

        score = 30.0  # Base

        # Field relevance
        if any(rf in field for rf in relevant_fields):
            score += 30
        elif any(pf in field for pf in partially_relevant_fields):
            score += 15

        # Degree level
        for d_key, d_bonus in degree_bonus.items():
            if d_key in degree:
                score += d_bonus
                break

        # Institution tier
        score += tier_bonus.get(tier, 0)

        best_score = max(best_score, score)

    return min(100.0, best_score)


# ── Dimension 6: Location Fit ─────────────────────────────────────────────────

def score_location_fit(candidate: dict) -> float:
    """
    Score based on location. JD prefers Pune/Noida/India.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = _normalize(profile.get("location", ""))
    country = _normalize(profile.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)
    work_mode = _normalize(signals.get("preferred_work_mode", ""))

    # Check if in India
    in_india = country in INDIA_COUNTRY_NAMES or any(
        city in location for city in GOOD_LOCATIONS_INDIA | PREFERRED_LOCATIONS
    )

    # Check if in preferred location
    in_preferred = any(loc in location for loc in PREFERRED_LOCATIONS)

    # Check if in good Indian city
    in_good_city = any(loc in location for loc in GOOD_LOCATIONS_INDIA)

    if in_preferred:
        score = 100.0
    elif in_good_city:
        score = 75.0
    elif in_india:
        score = 60.0
    elif willing_to_relocate:
        score = 40.0
    else:
        score = 20.0

    # Work mode compatibility (JD says hybrid)
    if work_mode in ("hybrid", "flexible"):
        score += 5
    elif work_mode == "remote":
        score -= 5  # Still OK but not ideal for hybrid JD
    elif work_mode == "onsite":
        score += 2  # Onsite is also fine for hybrid

    return min(100.0, max(0.0, score))


# ── Composite Scorer ──────────────────────────────────────────────────────────

def compute_feature_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute the multi-dimensional feature score for a candidate.

    Returns:
        (composite_score: float, dimension_scores: dict)
    """
    dimensions = {}

    dimensions["title_fit"] = score_title_fit(candidate)
    dimensions["skills_match"] = score_skills_match(candidate)
    dimensions["career_quality"] = score_career_quality(candidate)
    dimensions["experience_band"] = score_experience_band(candidate)
    dimensions["education_fit"] = score_education_fit(candidate)
    dimensions["location_fit"] = score_location_fit(candidate)

    beh_score, beh_components = compute_behavioral_score(candidate)
    dimensions["behavioral"] = beh_score

    # Weighted composite
    composite = 0.0
    for dim, weight in WEIGHTS.items():
        composite += dimensions.get(dim, 0) * weight

    return composite, dimensions
