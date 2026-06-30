# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Streamlit App for HuggingFace Spaces
# ─────────────────────────────────────────────────────────────────────────────
"""
A Streamlit web interface for the RecruitAI ranking pipeline.
Accepts a small candidate JSONL sample (<=100 candidates), runs the
full ranking pipeline, and displays + downloads the ranked CSV.

Deploy to HuggingFace Spaces as a Streamlit app.
"""

import streamlit as st
import json
import csv
import io
import time
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.hard_filter import apply_hard_filters
from pipeline.feature_scorer import compute_feature_score
from pipeline.behavioral_scorer import compute_behavioral_score, compute_behavioral_multiplier
from pipeline.semantic_scorer import compute_semantic_scores
from pipeline.reasoning_generator import generate_reasoning
from config import RULE_WEIGHT, SEMANTIC_WEIGHT, FINAL_TOP_N

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecruitAI - Intelligent Candidate Ranking",
    page_icon="🎯",
    layout="wide",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: #e0e0e0; margin: 0.5rem 0 0 0; }
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.8rem;
    }
    .stage-header {
        background: #f0f2f6;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 RecruitAI</h1>
    <p>Intelligent Candidate Discovery & Ranking System</p>
    <p style="font-size: 0.85rem; margin-top: 0.8rem;">
        Upload a JSONL file of candidates to rank them for a Senior AI Engineer role
        using a 4-stage hybrid pipeline: Hard Filters, Rule-Based Scoring,
        Semantic Similarity, and Behavioral Analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **RecruitAI** ranks candidates using:

    1. **Honeypot Detection** - Catches fake profiles
    2. **7-Dimension Scoring** - Title, skills, career, experience, education, location, behavior
    3. **Semantic Matching** - MiniLM sentence embeddings
    4. **Behavioral Multiplier** - Activity & responsiveness signals

    Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge.
    """)

    st.divider()
    st.markdown("**Pipeline Settings**")
    top_n = st.slider("Output top N candidates", 10, 100, min(100, 100), step=10)
    show_details = st.checkbox("Show scoring details", value=True)

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("Upload Candidates")

upload_tab, sample_tab = st.tabs(["Upload JSONL File", "Use Sample Data"])

candidates = None

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a candidates JSONL file (one JSON object per line)",
        type=["jsonl", "json"],
        help="Each line should be a valid JSON object following the candidate schema."
    )

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
            candidates = [json.loads(line) for line in lines]
            st.success(f"Loaded {len(candidates)} candidates from uploaded file.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

with sample_tab:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(_script_dir, "sample_candidates.json")
    if os.path.exists(sample_path):
        if st.button("Load sample candidates (from bundled sample_candidates.json)"):
            with open(sample_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
            st.success(f"Loaded {len(candidates)} sample candidates.")
    else:
        st.info("No sample_candidates.json found. Upload a JSONL file instead.")

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if candidates:
    st.divider()

    if st.button("Run Ranking Pipeline", type="primary", use_container_width=True):
        start_time = time.time()

        # ── Stage 1 ──────────────────────────────────────────────────────
        with st.status("Stage 1: Hard Filters & Honeypot Detection...", expanded=True) as status:
            filtered, stats = apply_hard_filters(candidates)
            stage1_time = time.time() - start_time

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", stats["total"])
            col2.metric("Passed", stats["passed"])
            col3.metric("Honeypots", stats["filtered_honeypot"])
            col4.metric("No Signal", stats["filtered_no_signal"])

            status.update(label=f"Stage 1 complete ({stage1_time:.1f}s)", state="complete")

        if not filtered:
            st.error("No candidates passed the hard filters!")
            st.stop()

        # ── Stage 2 ──────────────────────────────────────────────────────
        with st.status("Stage 2: Rule-Based Feature Scoring...", expanded=True) as status:
            scored = []
            progress = st.progress(0)
            for i, candidate in enumerate(filtered):
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
                progress.progress((i + 1) / len(filtered))

            scored.sort(key=lambda x: x["rule_score"], reverse=True)
            stage2_top = scored[:min(2000, len(scored))]
            stage2_time = time.time() - start_time

            status.update(label=f"Stage 2 complete ({stage2_time:.1f}s)", state="complete")

        # ── Stage 3 ──────────────────────────────────────────────────────
        with st.status("Stage 3: Semantic Similarity Scoring...", expanded=True) as status:
            st.write(f"Embedding {len(stage2_top)} candidates with MiniLM-L6-v2...")
            semantic_candidates = [item["candidate"] for item in stage2_top]
            semantic_scores = compute_semantic_scores(semantic_candidates)

            for item, sem_score in zip(stage2_top, semantic_scores):
                item["semantic_score"] = sem_score

            for item in scored[len(stage2_top):]:
                item["semantic_score"] = 0.0

            stage3_time = time.time() - start_time
            status.update(label=f"Stage 3 complete ({stage3_time:.1f}s)", state="complete")

        # ── Stage 4 ──────────────────────────────────────────────────────
        with st.status("Stage 4: Final Composite Ranking...", expanded=True) as status:
            for item in scored:
                rule = item["rule_score"]
                semantic = item["semantic_score"]
                beh_mult = item["behavioral_multiplier"]
                sem_normalized = min(100.0, semantic * 1.3)
                blended = (RULE_WEIGHT * rule + SEMANTIC_WEIGHT * sem_normalized)
                item["final_score"] = blended * beh_mult

            scored.sort(key=lambda x: x["final_score"], reverse=True)
            output_n = min(top_n, len(scored))
            top_results = scored[:output_n]

            results = []
            for rank, item in enumerate(top_results, 1):
                c = item["candidate"]
                cid = c.get("candidate_id", "?")
                score_out = round(1.0 - (rank - 1) * (0.8 / output_n), 4)
                reasoning = generate_reasoning(c, rank, item["final_score"], item["dimensions"])

                results.append({
                    "candidate_id": cid,
                    "rank": rank,
                    "score": score_out,
                    "reasoning": reasoning,
                    "title": c.get("profile", {}).get("current_title", ""),
                    "company": c.get("profile", {}).get("current_company", ""),
                    "experience": c.get("profile", {}).get("years_of_experience", 0),
                    "rule_score": round(item["rule_score"], 1),
                    "semantic_score": round(item.get("semantic_score", 0), 1),
                    "behavioral_mult": round(item["behavioral_multiplier"], 2),
                    "final_score": round(item["final_score"], 1),
                    "dimensions": item["dimensions"],
                })

            stage4_time = time.time() - start_time
            status.update(label=f"Stage 4 complete ({stage4_time:.1f}s)", state="complete")

        # ── Results ──────────────────────────────────────────────────────
        total_time = time.time() - start_time
        st.divider()
        st.subheader("Results")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Candidates Ranked", output_n)
        mcol2.metric("Total Time", f"{total_time:.1f}s")
        mcol3.metric("Honeypots Caught", stats["filtered_honeypot"])
        mcol4.metric("Pass Rate", f"{stats['passed']/max(stats['total'],1)*100:.0f}%")

        # Display ranked table
        st.subheader("Ranked Candidates")

        for result in results:
            with st.expander(
                f"#{result['rank']} - {result['title']} at {result['company']} "
                f"({result['experience']:.1f} yr) - Score: {result['score']:.4f}"
            ):
                st.markdown(f"**Candidate ID:** `{result['candidate_id']}`")
                st.markdown(f"**Reasoning:** {result['reasoning']}")

                if show_details:
                    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                    dcol1.metric("Rule Score", result["rule_score"])
                    dcol2.metric("Semantic", result["semantic_score"])
                    dcol3.metric("Behavioral x", result["behavioral_mult"])
                    dcol4.metric("Final", result["final_score"])

                    dims = result["dimensions"]
                    st.markdown("**Dimension Breakdown:**")
                    dim_cols = st.columns(4)
                    for i, (dim, val) in enumerate(dims.items()):
                        dim_cols[i % 4].metric(dim.replace("_", " ").title(), f"{val:.0f}")

        # ── Download CSV ─────────────────────────────────────────────────
        st.divider()
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])

        st.download_button(
            label="Download Submission CSV",
            data=csv_buffer.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

else:
    st.info("Upload a JSONL file or load sample data to get started.")
