# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Final Ranker (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
"""
Stage 4: Combines rule-based scores (Stage 2) and semantic scores (Stage 3)
with behavioral multiplier to produce the final ranked output.
"""

import csv
from config import (
    RULE_WEIGHT,
    SEMANTIC_WEIGHT,
    STAGE2_TOP_N,
    FINAL_TOP_N,
)
from pipeline.feature_scorer import compute_feature_score
from pipeline.semantic_scorer import compute_semantic_scores
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_score
from pipeline.reasoning_generator import generate_reasoning


def rank_candidates(candidates: list[dict], output_path: str) -> list[dict]:
    """
    Run the full Stage 2-4 pipeline on pre-filtered candidates.

    Args:
        candidates: Candidates that passed Stage 1 hard filters
        output_path: Path to write submission CSV

    Returns:
        List of top 100 ranked candidates with scores and reasoning
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: Rule-Based Feature Scoring ({len(candidates)} candidates)")
    print(f"{'='*60}")

    # ── Stage 2: Score all candidates ─────────────────────────────────────
    scored = []
    for i, candidate in enumerate(candidates):
        rule_score, dimensions = compute_feature_score(candidate)
        beh_score, _ = compute_behavioral_score(candidate)
        beh_mult = compute_behavioral_multiplier(beh_score)

        scored.append({
            "candidate": candidate,
            "rule_score": rule_score,
            "dimensions": dimensions,
            "behavioral_score": beh_score,
            "behavioral_multiplier": beh_mult,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Scored {i+1}/{len(candidates)}...")

    # Sort by rule score and take top N for semantic scoring
    scored.sort(key=lambda x: x["rule_score"], reverse=True)
    top_for_semantic = scored[:STAGE2_TOP_N]

    print(f"\n  Top 10 by rule score:")
    for item in scored[:10]:
        c = item["candidate"]
        cid = c.get("candidate_id", "?")
        title = c.get("profile", {}).get("current_title", "?")
        print(f"    {cid}: {title} — rule_score={item['rule_score']:.1f}")

    # ── Stage 3: Semantic Scoring on top N ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 3: Semantic Scoring (top {len(top_for_semantic)} candidates)")
    print(f"{'='*60}")

    semantic_candidates = [item["candidate"] for item in top_for_semantic]
    semantic_scores = compute_semantic_scores(semantic_candidates)

    for item, sem_score in zip(top_for_semantic, semantic_scores):
        item["semantic_score"] = sem_score

    # Give remaining candidates a baseline semantic score
    for item in scored[STAGE2_TOP_N:]:
        item["semantic_score"] = 0.0

    # ── Stage 4: Final Composite ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 4: Final Composite Ranking")
    print(f"{'='*60}")

    for item in scored:
        rule = item["rule_score"]
        semantic = item["semantic_score"]
        beh_mult = item["behavioral_multiplier"]

        # Normalize scores to similar ranges before blending
        # Rule score is 0-100, semantic is roughly 0-80 in practice
        # Scale semantic to 0-100 range
        sem_normalized = min(100.0, semantic * 1.3)

        # Blend rule and semantic scores
        blended = (RULE_WEIGHT * rule + SEMANTIC_WEIGHT * sem_normalized)

        # Apply behavioral multiplier
        final = blended * beh_mult

        item["final_score"] = final

    # Sort by final score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Take top 100
    top_100 = scored[:FINAL_TOP_N]

    # ── Generate output ───────────────────────────────────────────────────
    print(f"\n  Top 10 final ranking:")
    results = []
    for rank, item in enumerate(top_100, 1):
        c = item["candidate"]
        cid = c.get("candidate_id", "?")
        title = c.get("profile", {}).get("current_title", "?")
        final = item["final_score"]

        # Normalize final scores to 0-1 range for output
        # Use rank-based scoring to ensure monotonically decreasing
        score_out = round(1.0 - (rank - 1) * 0.008, 4)
        # But also incorporate actual score differentiation
        if rank > 1:
            prev_score = top_100[rank - 2]["final_score"]
            curr_score = item["final_score"]
            # If there's a big gap from previous, reflect it
            if prev_score > 0:
                ratio = curr_score / prev_score
                score_out = max(0.01, min(score_out, round(ratio * results[-1]["score"], 4)))

        reasoning = generate_reasoning(
            c, rank, final, item["dimensions"]
        )

        result = {
            "candidate_id": cid,
            "rank": rank,
            "score": score_out,
            "reasoning": reasoning,
        }
        results.append(result)

        if rank <= 10:
            yrs = c.get("profile", {}).get("years_of_experience", 0)
            print(f"    #{rank}: {cid} — {title} ({yrs:.1f}yr) — "
                  f"score={score_out:.4f}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    _write_submission_csv(results, output_path)

    return results


def _write_submission_csv(results: list[dict], output_path: str):
    """Write the submission CSV in the required format."""
    print(f"\n  Writing submission to: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for result in results:
            writer.writerow([
                result["candidate_id"],
                result["rank"],
                f"{result['score']:.4f}",
                result["reasoning"],
            ])

    print(f"  [OK] Wrote {len(results)} rows")
