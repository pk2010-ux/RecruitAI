#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
"""
Intelligent Candidate Discovery & Ranking System

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./output/submission.csv

Runs the full 4-stage pipeline:
    Stage 1: Hard Filters & Honeypot Detection
    Stage 2: Rule-Based Feature Scoring (7 dimensions)
    Stage 3: Semantic Similarity (sentence-transformers)
    Stage 4: Final Composite + Behavioral Multiplier → Top 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.loader import load_candidates
from pipeline.hard_filter import apply_hard_filters
from pipeline.final_ranker import rank_candidates


def main():
    parser = argparse.ArgumentParser(
        description="RecruitAI — Intelligent Candidate Ranking"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out",
        default="./output/submission.csv",
        help="Output CSV path (default: ./output/submission.csv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for loading candidates (default: 10000)"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.candidates):
        print(f"[ERROR] Candidates file not found: {args.candidates}")
        sys.exit(1)

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RecruitAI — Intelligent Candidate Ranking Pipeline")
    print("=" * 60)
    print(f"  Input:  {args.candidates}")
    print(f"  Output: {args.out}")
    print()

    start_time = time.time()

    # ── Stage 1: Load and Hard Filter ─────────────────────────────────────
    print(f"{'='*60}")
    print("STAGE 1: Loading & Hard Filtering")
    print(f"{'='*60}")

    all_candidates = []
    batch = []
    total_loaded = 0

    for candidate in load_candidates(args.candidates):
        batch.append(candidate)
        total_loaded += 1

        if len(batch) >= args.batch_size:
            all_candidates.extend(batch)
            print(f"  Loaded {total_loaded} candidates...")
            batch = []

    if batch:
        all_candidates.extend(batch)

    print(f"  Total loaded: {total_loaded}")

    # Apply hard filters
    stage1_time = time.time()
    filtered, stats = apply_hard_filters(all_candidates)

    print(f"\n  Hard filter results:")
    print(f"    Total:             {stats['total']}")
    print(f"    Passed:            {stats['passed']}")
    print(f"    Filtered (honeypot): {stats['filtered_honeypot']}")
    print(f"    Filtered (zero exp): {stats['filtered_zero_exp']}")
    print(f"    Filtered (no signal): {stats['filtered_no_signal']}")
    print(f"    Pass rate:         {stats['passed']/stats['total']*100:.1f}%")
    print(f"    Stage 1 time:      {time.time() - stage1_time:.1f}s")

    # Free memory — we don't need all_candidates anymore
    del all_candidates

    # ── Stages 2-4: Score, Rank, Output ───────────────────────────────────
    stage2_time = time.time()
    results = rank_candidates(filtered, str(out_path))

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    # Warn if over budget
    if total_time > 300:
        print(f"\n[WARNING] Total time {total_time:.1f}s exceeds 5-minute budget!")
    else:
        print(f"\n[OK] Within 5-minute compute budget ({total_time:.1f}s / 300s)")


if __name__ == "__main__":
    main()
