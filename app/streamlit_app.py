"""
LLM Data Quality Scorer — Streamlit Web UI

Upload any text corpus → pipeline filters, deduplicates, and scores every
document → download a clean, training-ready dataset. No code required.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedup.minhash_dedup import DedupConfig, MinHashDedup
from src.filters.heuristic import FilterConfig, HeuristicFilter
from src.io.loader import load_from_bytes
from src.models import PipelineStats, ScoredDocument
from src.scorer.scorer import QualityScorer, ScoringConfig

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Data Quality Scorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #4c9be8;
        margin-bottom: 12px;
    }
    .metric-card.green { border-left-color: #2ecc71; }
    .metric-card.red   { border-left-color: #e74c3c; }
    .metric-card.yellow{ border-left-color: #f39c12; }
    .stage-badge {
        display: inline-block;
        background: #2d3250;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #8ab4f8;
        margin: 2px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #8ab4f8;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached pipeline components ────────────────────────────────────────────────
@st.cache_resource
def get_chroma_store(embedding_model: str):
    try:
        from src.rag.chroma_store import ChromaConfig, ChromaStore
        store = ChromaStore(ChromaConfig(embedding_model=embedding_model))
        store.seed_with_gold_examples()
        return store
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    st.sidebar.image(
        "https://img.shields.io/badge/LLM%20Data%20Quality-Scorer-4c9be8?style=for-the-badge",
        use_container_width=True,
    )
    st.sidebar.markdown("---")

    cfg = {}

    # API Key
    st.sidebar.markdown("### 🔑 LLM Judge (Optional)")
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required only for the LLM-as-Judge stage. Heuristics + dedup run without a key.",
        placeholder="sk-ant-...",
    )
    cfg["api_key"] = api_key
    cfg["enable_judge"] = st.sidebar.toggle(
        "Enable LLM-as-Judge",
        value=bool(api_key),
        disabled=not bool(api_key),
        help="Uses Claude to score borderline documents 1-10 with structured reasoning.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pipeline Stages")
    cfg["enable_dedup"] = st.sidebar.toggle("Deduplication (MinHash LSH)", value=True)
    cfg["enable_filters"] = st.sidebar.toggle("Heuristic Filters", value=True)

    st.sidebar.markdown("---")

    # Filter settings
    with st.sidebar.expander("🔧 Filter Thresholds", expanded=False):
        cfg["min_words"] = st.slider("Min words per doc", 10, 500, 50)
        cfg["max_words"] = st.slider("Max words per doc", 1000, 200000, 100000, step=1000)
        cfg["max_symbol_ratio"] = st.slider("Max symbol ratio", 0.1, 0.9, 0.35, 0.05)
        cfg["max_digit_ratio"] = st.slider("Max digit ratio", 0.1, 0.9, 0.30, 0.05)
        cfg["min_stopword_ratio"] = st.slider("Min stopword ratio", 0.01, 0.30, 0.08, 0.01)
        cfg["max_repetition_ratio"] = st.slider("Max repetition ratio", 0.1, 0.9, 0.25, 0.05)
        lang_opt = st.selectbox("Language filter", ["en", "fr", "de", "es", "zh", "None"])
        cfg["language"] = None if lang_opt == "None" else lang_opt

    with st.sidebar.expander("🔗 Dedup Settings", expanded=False):
        cfg["dedup_threshold"] = st.slider("Similarity threshold", 0.5, 0.99, 0.80, 0.01)
        cfg["num_perm"] = st.select_slider("MinHash permutations", [64, 128, 256], value=128)

    with st.sidebar.expander("🤖 LLM Judge Settings", expanded=False):
        cfg["judge_model"] = st.selectbox(
            "Model",
            ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        )
        cfg["min_llm_score"] = st.slider("Min LLM score to keep", 1, 10, 6)
        cfg["max_judge_docs"] = st.slider("Max docs to judge (cost guard)", 10, 500, 100)

    with st.sidebar.expander("📊 Scoring", expanded=False):
        cfg["min_quality_score"] = st.slider("Min quality score to keep", 0.1, 0.9, 0.5, 0.05)
        cfg["heuristic_weight"] = st.slider("Heuristic weight", 0.0, 1.0, 0.3, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline stages:**  \n"
        "<span class='stage-badge'>① Heuristic Filter</span> "
        "<span class='stage-badge'>② Dedup</span> "
        "<span class='stage-badge'>③ LLM Judge</span> "
        "<span class='stage-badge'>④ Score</span>",
        unsafe_allow_html=True,
    )
    return cfg


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(docs, cfg: dict) -> tuple[list[ScoredDocument], PipelineStats]:
    stats = PipelineStats(total_input=len(docs))
    scored_docs = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # ── Stage 1: Heuristic Filtering ──────────────────────────────────────────
    status_text.markdown("**Stage 1 / 4 — Heuristic Filtering...**")
    if cfg.get("enable_filters", True):
        filter_cfg = FilterConfig(
            min_words=cfg["min_words"],
            max_words=cfg["max_words"],
            max_symbol_ratio=cfg["max_symbol_ratio"],
            max_digit_ratio=cfg["max_digit_ratio"],
            min_stopword_ratio=cfg["min_stopword_ratio"],
            max_repetition_ratio=cfg["max_repetition_ratio"],
            language=cfg.get("language", "en"),
        )
        hf = HeuristicFilter(filter_cfg)
        for i, doc in enumerate(docs):
            scored = hf.filter(doc)
            scored_docs.append(scored)
            progress_bar.progress((i + 1) / len(docs) * 0.30)
    else:
        from src.models import ScoredDocument
        scored_docs = [ScoredDocument.from_document(d) for d in docs]
        for d in scored_docs:
            d.heuristic_pass = True
            d.word_count = len(d.text.split())

    stats.heuristic_passed = sum(1 for d in scored_docs if d.heuristic_pass)
    stats.heuristic_failed = stats.total_input - stats.heuristic_passed
    for d in scored_docs:
        for flag in d.heuristic_flags:
            key = flag.split("(")[0].strip()
            stats.filter_breakdown[key] = stats.filter_breakdown.get(key, 0) + 1

    # ── Stage 2: Deduplication ────────────────────────────────────────────────
    status_text.markdown("**Stage 2 / 4 — Deduplication (MinHash LSH)...**")
    if cfg.get("enable_dedup", True):
        dedup_cfg = DedupConfig(
            threshold=cfg["dedup_threshold"],
            num_perm=cfg["num_perm"],
        )
        deduper = MinHashDedup(dedup_cfg)
        scored_docs = deduper.deduplicate(scored_docs)
    stats.duplicates_removed = sum(1 for d in scored_docs if d.is_duplicate)
    stats.unique_docs = stats.heuristic_passed - stats.duplicates_removed
    progress_bar.progress(0.50)

    # ── Stage 3: LLM Judge ────────────────────────────────────────────────────
    if cfg.get("enable_judge") and cfg.get("api_key"):
        status_text.markdown("**Stage 3 / 4 — LLM-as-Judge (Claude)...**")
        try:
            from src.judge.llm_judge import JudgeConfig, LLMJudge
            from src.rag.chroma_store import ChromaConfig, ChromaStore

            store = get_chroma_store("all-MiniLM-L6-v2")
            judge_cfg = JudgeConfig(
                model=cfg["judge_model"],
                min_score=cfg["min_llm_score"],
                max_docs=cfg["max_judge_docs"],
            )
            judge = LLMJudge(
                api_key=cfg["api_key"],
                config=judge_cfg,
                chroma_store=store,
            )
            candidates = [d for d in scored_docs if d.heuristic_pass and not d.is_duplicate]
            candidates = candidates[: judge_cfg.max_docs]
            judge_progress = st.empty()
            for i, doc in enumerate(candidates):
                result = judge.judge(doc)
                if result:
                    doc.llm_score = result.score
                    doc.llm_reasoning = result.reasoning
                    doc.llm_issues = result.issues
                    doc.llm_strengths = result.strengths
                progress_bar.progress(0.50 + (i + 1) / max(len(candidates), 1) * 0.35)
                judge_progress.caption(f"Judging {i+1}/{len(candidates)} documents...")
            judge_progress.empty()
            stats.llm_judged = sum(1 for d in scored_docs if d.llm_score is not None)
        except Exception as e:
            st.warning(f"LLM Judge error: {e}. Continuing without LLM scores.")
    else:
        status_text.markdown("**Stage 3 / 4 — LLM Judge skipped (no API key)**")

    progress_bar.progress(0.85)

    # ── Stage 4: Final Scoring ────────────────────────────────────────────────
    status_text.markdown("**Stage 4 / 4 — Computing quality scores...**")
    scoring_cfg = ScoringConfig(
        min_quality_score=cfg["min_quality_score"],
        heuristic_weight=cfg.get("heuristic_weight", 0.3),
        llm_weight=1.0 - cfg.get("heuristic_weight", 0.3),
    )
    scorer = QualityScorer(scoring_cfg)
    scored_docs = scorer.score_batch(scored_docs)
    stats.final_kept = sum(1 for d in scored_docs if d.keep)

    progress_bar.progress(1.0)
    status_text.markdown("✅ **Pipeline complete!**")
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

    return scored_docs, stats


# ── Results rendering ─────────────────────────────────────────────────────────
def render_metrics(stats: PipelineStats) -> None:
    cols = st.columns(5)
    metrics = [
        ("📥 Input", stats.total_input, None, None),
        ("✅ After Filters", stats.heuristic_passed,
         f"-{stats.heuristic_failed}", "normal" if stats.heuristic_failed < stats.total_input * 0.5 else "inverse"),
        ("🔗 After Dedup", stats.unique_docs,
         f"-{stats.duplicates_removed} dupes", "normal"),
        ("🤖 LLM Judged", stats.llm_judged, None, None),
        ("🏆 Final Kept", stats.final_kept,
         f"{100 * stats.final_kept / max(stats.total_input, 1):.1f}% of input", "normal"),
    ]
    for col, (label, val, delta, delta_color) in zip(cols, metrics):
        col.metric(label, val, delta=delta, delta_color=delta_color or "normal")


def render_charts(docs: list[ScoredDocument], stats: PipelineStats) -> None:
    col1, col2 = st.columns(2)

    # Score distribution
    with col1:
        st.markdown("#### Quality Score Distribution")
        all_scores = [d.quality_score for d in docs if d.heuristic_pass and not d.is_duplicate]
        if all_scores:
            fig = px.histogram(
                x=all_scores,
                nbins=20,
                color_discrete_sequence=["#4c9be8"],
                labels={"x": "Quality Score", "y": "Documents"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Keep threshold")
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fafafa", height=300, margin=dict(t=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scored documents to display.")

    # Filter failure reasons
    with col2:
        st.markdown("#### Filter Failure Reasons")
        if stats.filter_breakdown:
            breakdown = dict(sorted(stats.filter_breakdown.items(), key=lambda x: -x[1])[:8])
            fig = px.bar(
                x=list(breakdown.values()),
                y=list(breakdown.keys()),
                orientation="h",
                color_discrete_sequence=["#e74c3c"],
                labels={"x": "Count", "y": "Reason"},
            )
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#fafafa", height=300, margin=dict(t=20),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No filter failures! All documents passed heuristics.")

    # Pipeline funnel
    st.markdown("#### Pipeline Funnel")
    stages = ["Input", "After Heuristics", "After Dedup", "Final Kept"]
    counts = [stats.total_input, stats.heuristic_passed, stats.unique_docs, stats.final_kept]
    fig = go.Figure(go.Funnel(
        y=stages,
        x=counts,
        textinfo="value+percent initial",
        marker={"color": ["#4c9be8", "#27ae60", "#f39c12", "#2ecc71"]},
    ))
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#fafafa", height=280, margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_doc_table(docs: list[ScoredDocument], filter_fn, label: str, max_rows: int = 200) -> None:
    subset = [d for d in docs if filter_fn(d)][:max_rows]
    if not subset:
        st.info(f"No documents in '{label}' category.")
        return
    st.caption(f"Showing {len(subset)} of {sum(1 for d in docs if filter_fn(d))} documents")
    rows = []
    for d in subset:
        rows.append({
            "ID": d.id[:12] + "...",
            "Quality Score": f"{d.quality_score:.3f}",
            "Heuristic Score": f"{d.heuristic_score:.3f}",
            "LLM Score": str(d.llm_score) if d.llm_score else "—",
            "Words": d.word_count,
            "Language": d.language or "—",
            "Flags": ", ".join(d.heuristic_flags) if d.heuristic_flags else "—",
            "Text Preview": d.text[:200] + ("..." if len(d.text) > 200 else ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def make_download_bytes(docs: list[ScoredDocument], fmt: str) -> tuple[bytes, str, str]:
    kept = [d for d in docs if d.keep]
    if fmt == "jsonl":
        lines = []
        for d in kept:
            row = {
                "id": d.id, "text": d.text,
                "quality_score": d.quality_score,
                "heuristic_score": d.heuristic_score,
                "word_count": d.word_count,
                "language": d.language,
            }
            if d.llm_score:
                row["llm_score"] = d.llm_score
                row["llm_reasoning"] = d.llm_reasoning
            lines.append(json.dumps(row, ensure_ascii=False))
        return "\n".join(lines).encode(), "clean_corpus.jsonl", "application/jsonl"

    elif fmt == "csv":
        rows = [{
            "id": d.id, "text": d.text,
            "quality_score": d.quality_score,
            "word_count": d.word_count,
            "llm_score": d.llm_score,
        } for d in kept]
        buf = io.BytesIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue(), "clean_corpus.csv", "text/csv"

    else:  # parquet
        rows = [{
            "id": d.id, "text": d.text,
            "quality_score": d.quality_score,
            "word_count": d.word_count,
        } for d in kept]
        buf = io.BytesIO()
        pd.DataFrame(rows).to_parquet(buf, index=False)
        return buf.getvalue(), "clean_corpus.parquet", "application/octet-stream"


# ── Session state helpers ─────────────────────────────────────────────────────
def _init_session_state() -> None:
    """Initialise all session state keys once at startup."""
    defaults = {
        "docs": None,           # list[Document] currently loaded
        "source_name": None,    # display string for the loaded dataset
        "scored_docs": None,    # list[ScoredDocument] after pipeline
        "stats": None,          # PipelineStats after pipeline
        "last_upload_name": None,  # track which file is currently loaded
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    _init_session_state()
    cfg = render_sidebar()

    # Header
    st.markdown("# 🔬 LLM Data Quality Scorer")
    st.markdown(
        "Upload any text corpus — the pipeline **filters**, **deduplicates**, and **scores** "
        "every document, then gives you a clean, training-ready dataset to download."
    )
    st.markdown(
        "<span class='stage-badge'>① Heuristic Filters</span> "
        "<span class='stage-badge'>② MinHash Dedup</span> "
        "<span class='stage-badge'>③ LLM-as-Judge (RAG)</span> "
        "<span class='stage-badge'>④ Composite Scoring</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Data loading ──────────────────────────────────────────────────────────
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload your dataset",
            type=["jsonl", "json", "csv", "parquet", "txt"],
            help="Supported: JSONL (one doc per line), CSV, Parquet, TXT (split by blank lines), JSON array",
        )

    with col_sample:
        st.markdown("&nbsp;")
        use_sample = st.button(
            "📦 Use Sample Data",
            use_container_width=True,
            help="Load a built-in 20-document sample to try the pipeline immediately",
        )

    # Load sample data into session state on button click
    if use_sample:
        sample_path = Path(__file__).parent.parent / "data" / "samples" / "sample.jsonl"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                content = f.read()
            st.session_state.docs = load_from_bytes(content, "sample.jsonl")
            st.session_state.source_name = f"sample.jsonl ({len(st.session_state.docs)} documents)"
            st.session_state.scored_docs = None   # clear any previous results
            st.session_state.stats = None
            st.session_state.last_upload_name = "sample.jsonl"
        else:
            st.error("Sample data not found. Please upload a file.")

    # Load uploaded file into session state whenever a new file appears
    if uploaded is not None and uploaded.name != st.session_state.last_upload_name:
        try:
            content = uploaded.read()
            st.session_state.docs = load_from_bytes(content, uploaded.name)
            st.session_state.source_name = f"{uploaded.name} ({len(st.session_state.docs)} documents)"
            st.session_state.scored_docs = None
            st.session_state.stats = None
            st.session_state.last_upload_name = uploaded.name
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

    # ── Work with whatever is in session state ────────────────────────────────
    docs = st.session_state.docs
    source_name = st.session_state.source_name

    if docs:
        st.success(f"✅ Loaded **{source_name}**")

        with st.expander("👁 Preview data", expanded=False):
            preview_df = pd.DataFrame([
                {"#": i + 1, "ID": d.id[:12], "Text Preview": d.text[:300], "Words": len(d.text.split())}
                for i, d in enumerate(docs[:10])
            ])
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            if len(docs) > 10:
                st.caption(f"... and {len(docs) - 10} more documents")

        st.markdown("---")

        # Run / format controls
        run_col, fmt_col = st.columns([3, 1])
        with fmt_col:
            download_fmt = st.selectbox("Output format", ["jsonl", "csv", "parquet"])
        with run_col:
            run_clicked = st.button(
                "🚀 Run Quality Pipeline",
                type="primary",
                use_container_width=True,
                help="Runs all enabled pipeline stages on the loaded dataset",
            )

        if run_clicked:
            st.markdown("---")
            st.markdown("### ⚙️ Running Pipeline")
            with st.spinner(""):
                scored_docs, stats = run_pipeline(docs, cfg)
            # Persist results so they survive subsequent reruns
            st.session_state.scored_docs = scored_docs
            st.session_state.stats = stats

        # ── Show results if they exist (persisted in session state) ───────────
        if st.session_state.scored_docs is not None:
            scored_docs = st.session_state.scored_docs
            stats = st.session_state.stats

            st.markdown("---")
            st.markdown("### 📊 Results")
            render_metrics(stats)

            st.markdown("---")
            render_charts(scored_docs, stats)

            st.markdown("---")
            st.markdown("### 📋 Document Browser")
            tabs = st.tabs(["✅ Kept", "❌ Filtered Out", "🔗 Duplicates", "📈 All Scored"])

            with tabs[0]:
                render_doc_table(scored_docs, lambda d: d.keep, "Kept")
            with tabs[1]:
                render_doc_table(scored_docs, lambda d: not d.heuristic_pass, "Filtered")
            with tabs[2]:
                render_doc_table(scored_docs, lambda d: d.is_duplicate, "Duplicates")
            with tabs[3]:
                render_doc_table(
                    scored_docs,
                    lambda d: d.heuristic_pass and not d.is_duplicate,
                    "All Scored",
                )

            st.markdown("---")
            kept_count = stats.final_kept
            if kept_count > 0:
                st.markdown(f"### ⬇️ Download Clean Dataset ({kept_count} documents)")
                data_bytes, filename, mime = make_download_bytes(scored_docs, download_fmt)
                st.download_button(
                    label=f"Download {filename}",
                    data=data_bytes,
                    file_name=filename,
                    mime=mime,
                    type="primary",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "No documents met the quality threshold. "
                    "Try lowering 'Min quality score' in the sidebar."
                )

    else:
        # Empty / landing state
        st.markdown("---")
        st.markdown(
            """
            ### How it works

            | Step | Stage | What happens |
            |------|-------|-------------|
            | ① | **Upload** | Drag & drop JSONL, CSV, Parquet, or TXT |
            | ② | **Heuristic Filter** | Removes too-short, wrong-language, repetitive, or code-heavy docs |
            | ③ | **Deduplication** | MinHash LSH removes near-duplicate documents |
            | ④ | **LLM-as-Judge** | Claude scores each doc 1–10 with structured reasoning (optional) |
            | ⑤ | **Download** | Get a clean JSONL/CSV/Parquet ready for LLM training |

            > **No API key?** Steps ①–③ run entirely locally. Add an Anthropic key in the sidebar to unlock the LLM judge.

            ---
            #### Supported input formats
            - **JSONL** — one JSON object per line with a `text` / `content` / `body` field
            - **CSV** — any CSV with a text column (auto-detected)
            - **Parquet** — columnar format, ideal for large datasets
            - **TXT** — plain text split into documents by blank lines
            """
        )


if __name__ == "__main__":
    main()
