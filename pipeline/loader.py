# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Data Loader
# ─────────────────────────────────────────────────────────────────────────────
"""
Streaming loader for candidates.jsonl (and .jsonl.gz).
Yields one candidate dict at a time to keep memory under control.
"""

import gzip
import json
from pathlib import Path


def load_candidates(filepath: str):
    """
    Yield candidate dicts one at a time from a JSONL or gzipped JSONL file.

    Args:
        filepath: Path to candidates.jsonl or candidates.jsonl.gz

    Yields:
        dict — one candidate per iteration
    """
    path = Path(filepath)

    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    with opener() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON at line {line_num}: {e}")
                continue


def load_all_candidates(filepath: str) -> list[dict]:
    """Load all candidates into memory. Use only if RAM allows."""
    return list(load_candidates(filepath))


def count_candidates(filepath: str) -> int:
    """Count total candidates without loading them all."""
    count = 0
    for _ in load_candidates(filepath):
        count += 1
    return count
