# RecruitAI — Intelligent Candidate Discovery & Ranking

An AI-powered candidate ranking system built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. It goes beyond keyword matching to understand who actually fits a Senior AI Engineer role — analyzing career context, behavioral signals, and semantic meaning.

## Architecture

RecruitAI uses a **4-stage hybrid ranking pipeline**:

```
100K Candidates
    │
    ▼
┌──────────────────────────────────────┐
│ Stage 1: Hard Filters & Honeypots    │  ← Fast elimination (~15K pass)
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Stage 2: Rule-Based Feature Scoring  │  ← 7 dimensions, weighted composite
└──────────────────────────────────────┘
    │ (top 2000)
    ▼
┌──────────────────────────────────────┐
│ Stage 3: Semantic Similarity         │  ← MiniLM sentence-transformer
└──────────────────────────────────────┘
    │ (top 200)
    ▼
┌──────────────────────────────────────┐
│ Stage 4: Final Composite + Output    │  ← Behavioral multiplier + reasoning
└──────────────────────────────────────┘
    │
    ▼
  Top 100 Ranked CSV
```

### Scoring Dimensions (Stage 2)

| Dimension | Weight | What It Captures |
|-----------|--------|------------------|
| Title & Role Fit | 25% | Current/past titles aligned with AI/ML roles |
| Skills Match | 20% | Relevant AI/ML skills with proficiency/duration weighting |
| Career Quality | 20% | Product vs consulting, career stability, ML production signals |
| Experience Band | 10% | Proximity to 5-9 year sweet spot |
| Education Fit | 5% | CS/ML degrees, institution tier |
| Location Fit | 5% | India, Pune/Noida preferred |
| Behavioral Signals | 15% | Activity, responsiveness, availability |

### Key Design Decisions

1. **Honeypot Detection**: Catches ~80 candidates with impossible profiles (experience/date mismatches, expert skills with zero duration)
2. **Anti-Keyword-Stuffing**: Skills are weighted by proficiency × duration × endorsements — not just presence
3. **Consulting Firm Penalty**: Per the JD, consulting-only careers (TCS, Wipro, Infosys, etc.) get penalized — but mixed consulting+product is fine
4. **Career Description Analysis**: Scans job descriptions for ML/AI production keywords, not just title matching
5. **Behavioral Multiplier**: Even perfect-on-paper candidates are down-weighted if inactive, unresponsive, or have long notice periods

## Quick Start

### Prerequisites

- Python 3.10+
- 16 GB RAM (for processing 100K candidates)
- Internet connection (first run only — to download the sentence-transformer model)

### Setup

```bash
# Clone the repo
git clone https://github.com/pk2010-ux/RecruitAI.git
cd RecruitAI

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Full pipeline — produces output/submission.csv
python rank.py --candidates ./candidates.jsonl --out ./output/submission.csv
```

### Validate

```bash
# Use the official validator
python validate_submission.py output/submission.csv
```

## Project Structure

```
RecruitAI/
├── rank.py                          # Main CLI entry point
├── config.py                        # All weights, thresholds, skill mappings
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── pipeline/
│   ├── __init__.py
│   ├── loader.py                    # Streaming JSONL loader
│   ├── honeypot_detector.py         # Honeypot detection (6 checks)
│   ├── hard_filter.py               # Stage 1 hard filtering
│   ├── feature_scorer.py            # Stage 2 multi-dimensional scoring
│   ├── behavioral_scorer.py         # Behavioral signal composite (12 signals)
│   ├── semantic_scorer.py           # Stage 3 sentence-transformer similarity
│   ├── final_ranker.py              # Stage 4 composite + CSV output
│   └── reasoning_generator.py       # Per-candidate reasoning generation
├── submission_metadata.yaml         # Submission metadata
└── output/
    └── submission.csv               # Generated ranked output
```

## Compute Profile

| Constraint | This System |
|-----------|-------------|
| Runtime | ~2-4 min on CPU |
| RAM | <8 GB peak |
| GPU | Not required (works on CPU) |
| Network | Not required during ranking |
| Model | all-MiniLM-L6-v2 (~80MB, cached locally) |

## Dependencies

- `sentence-transformers` — Semantic similarity via all-MiniLM-L6-v2
- `numpy` — Array math
- `torch` — Model backend
- Standard library: `json`, `csv`, `re`, `datetime`, `math`, `argparse`
