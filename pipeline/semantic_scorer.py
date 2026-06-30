# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Semantic Scorer (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
"""
Stage 3: Semantic similarity scoring using sentence-transformers.
Runs on the top N candidates from Stage 2 (not all 100K).

Uses all-MiniLM-L6-v2 for fast CPU inference with decent quality.
Computes cosine similarity between a JD embedding and candidate profile text.
"""

import numpy as np
from config import (
    SEMANTIC_MODEL_NAME,
    SEMANTIC_BATCH_SIZE,
    JD_SUMMARY,
    JD_KEYWORDS_FOR_EMBEDDING,
)


# Global model cache (loaded once)
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading semantic model: {SEMANTIC_MODEL_NAME}")
        _model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        print(f"[INFO] Model loaded successfully")
    return _model


def _build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of the candidate for embedding.
    Combines multiple fields to give the model full context.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    parts = []

    # Headline and summary (most important)
    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    # Current role context
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    if title:
        parts.append(f"Currently working as {title} at {company} in {industry}")

    # Career descriptions (rich signal)
    for job in career[:4]:  # Limit to avoid too-long text
        desc = job.get("description", "")
        job_title = job.get("title", "")
        job_company = job.get("company", "")
        if desc:
            parts.append(f"{job_title} at {job_company}: {desc}")

    # Skills as a list
    if skills:
        skill_names = [
            f"{s.get('name', '')} ({s.get('proficiency', '')})"
            for s in skills
            if s.get("name")
        ]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

    # Education
    for edu in education[:2]:
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        if field:
            parts.append(f"{degree} in {field} from {institution}")

    return " . ".join(parts)


def compute_jd_embedding():
    """Compute and cache the JD embedding."""
    model = _get_model()
    # Combine JD summary and keywords for a rich embedding
    jd_text = JD_SUMMARY + " " + JD_KEYWORDS_FOR_EMBEDDING
    embedding = model.encode(jd_text, normalize_embeddings=True)
    return embedding


def compute_semantic_scores(candidates: list[dict]) -> list[float]:
    """
    Compute semantic similarity scores for a batch of candidates.

    Args:
        candidates: List of candidate dicts

    Returns:
        List of similarity scores (0-100 scale)
    """
    if not candidates:
        return []

    model = _get_model()

    # Compute JD embedding once
    jd_embedding = compute_jd_embedding()

    # Build candidate texts
    candidate_texts = [_build_candidate_text(c) for c in candidates]

    # Batch encode candidates
    print(f"[INFO] Encoding {len(candidate_texts)} candidates...")
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=SEMANTIC_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Cosine similarity (embeddings are already normalized)
    similarities = np.dot(candidate_embeddings, jd_embedding)

    # Scale from [-1, 1] to [0, 100]
    # In practice, cosine similarities for text are usually 0.0 to 0.8
    # We'll scale so that 0.8+ → 100 and 0.0 → 0
    scores = []
    for sim in similarities:
        # Clamp and scale
        scaled = max(0.0, min(1.0, sim)) * 100.0
        scores.append(scaled)

    return scores
