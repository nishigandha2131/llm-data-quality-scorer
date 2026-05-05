# LLM Data Quality Scorer

> A production-style data pipeline that prepares raw text corpora for LLM training — scoring, filtering, and deduplicating documents so only the highest-quality data reaches the model.

Built with **Python 3.13 · LangChain · Anthropic Claude · ChromaDB · Streamlit**

---

## The Problem This Solves

Training a large language model (like Grok) on raw internet data directly would be catastrophic. Raw corpora contain spam, duplicate content, foreign-language text, code fragments, garbled text, and boilerplate — all of which degrade model quality. This pipeline transforms noisy raw text into a clean, scored, training-ready dataset.

```
Raw Corpus (Common Crawl / web scrape / curated dump)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 · Heuristic Filters                            │
│  Fast rule-based pre-screening — kills obvious garbage  │
│  Length · Language · Symbol ratio · Repetition · etc.   │
└─────────────────────┬───────────────────────────────────┘
                      │  (bad docs discarded here)
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 · Semantic Deduplication                       │
│  MinHash LSH (datasketch) removes near-duplicate docs   │
└─────────────────────┬───────────────────────────────────┘
                      │  (duplicates removed here)
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 · LLM-as-Judge  (optional, requires API key)  │
│  Claude scores each doc 1–10 with structured reasoning  │
│  RAG-augmented: ChromaDB retrieves similar gold docs    │
│  as few-shot context before each judgment               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 · Composite Scoring                            │
│  Weighted blend of heuristic + LLM signals → 0–1 score  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         Clean Dataset (.jsonl / .csv / .parquet)
                → Ready for LLM Training
```

---

## Quick Start

**Requirements:** Python 3.10+

```bash
# 1. Clone and enter the project
cd llm-data-quality-scorer

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 3. Launch the web UI
streamlit run app/streamlit_app.py
# → opens http://localhost:8501
```

No API key needed — heuristics and deduplication run entirely locally. Paste an Anthropic key in the sidebar to unlock the LLM judge.

---

## Two Ways to Use It

### Web UI (no code required)

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

1. Upload your file (or click **"Use Sample Data"** to try it immediately)
2. Adjust thresholds in the sidebar — or leave the defaults
3. Click **Run Quality Pipeline**
4. Download your clean dataset

### CLI

```bash
# Heuristics + dedup only (no API key needed)
python scripts/run_pipeline.py --input data/samples/sample.jsonl

# With LLM judge
python scripts/run_pipeline.py \
  --input  data/raw/corpus.jsonl \
  --output data/processed/clean.jsonl \
  --api-key sk-ant-... \
  --judge

# Custom config file
python scripts/run_pipeline.py \
  --input  data/raw/corpus.parquet \
  --config my_config.yaml
```

---

## Project Structure

```
llm-data-quality-scorer/
│
├── app/
│   └── streamlit_app.py        ← Web UI: upload → pipeline → download
│
├── src/
│   ├── models.py               ← Pydantic models: Document, ScoredDocument, PipelineStats
│   ├── io/
│   │   └── loader.py           ← Multi-format loader (JSONL, CSV, Parquet, TXT, JSON)
│   ├── filters/
│   │   └── heuristic.py        ← 7 independent heuristic filters
│   ├── dedup/
│   │   └── minhash_dedup.py    ← MinHash LSH near-duplicate detection
│   ├── rag/
│   │   └── chroma_store.py     ← ChromaDB store: few-shot RAG + semantic similarity
│   ├── judge/
│   │   └── llm_judge.py        ← LangChain + Claude structured scoring (1–10)
│   └── scorer/
│       └── scorer.py           ← Composite quality score from heuristic + LLM signals
│
├── scripts/
│   └── run_pipeline.py         ← End-to-end CLI pipeline runner
│
├── tests/
│   ├── test_filters.py         ← 13 heuristic filter unit tests
│   ├── test_dedup.py           ← 9 deduplication unit tests
│   └── test_scorer.py          ← 11 scoring unit tests (33 total, all passing)
│
├── data/
│   ├── samples/sample.jsonl    ← 20-document demo dataset (covers all pipeline stages)
│   └── processed/              ← Pipeline output lands here
│
├── config.yaml                 ← All thresholds and settings (tunable without code changes)
└── requirements.txt
```

---

## Heuristic Filter Bank

Each filter is independent and produces a labelled flag explaining why a document was removed.

| Filter | Signal | What it catches |
|--------|--------|----------------|
| Length | Word count | Too short (< 50 words) or too long (> 100k words) |
| Language | langdetect | Non-English text (target language configurable) |
| Symbol ratio | Non-alphanumeric chars / total | Excessive punctuation, HTML, special chars |
| Digit ratio | Digits / total chars | Data dumps, log files, numeric tables |
| Stopword ratio | Stopwords / total words | Code snippets, garbled text with no prose |
| Repetition | Unique trigrams / total trigrams | Spam, padding, repeated sentences |
| Avg word length | Mean chars per word | Garbled/encoded text (too short) or URLs/code (too long) |

---

## LLM-as-Judge: RAG-Augmented Design

The judge retrieves semantically similar gold-standard documents from ChromaDB before scoring — giving Claude calibrated, domain-aware few-shot context rather than judging in a vacuum.

```
Document to score
       │
       ▼
ChromaDB (sentence-transformers embeddings)
  → retrieves top-3 similar high-quality examples
       │
       ▼
Claude prompt:
  [System: evaluation rubric]
  [Few-shot: 3 similar gold examples with scores]
  [Document to evaluate]
       │
       ▼
Structured output: { score: 1–10, reasoning, issues[], strengths[] }
       │
       ▼
If score ≥ 8 → document added back to ChromaDB
  (self-improving reference pool)
```

Scoring rubric (averaged across 5 dimensions):
- **Coherence** — logical structure and flow
- **Informativeness** — substantive, non-trivial content
- **Language quality** — grammar, vocabulary, spelling
- **Appropriateness** — suitable for LLM training (no PII, hate, spam)
- **Originality** — unique value beyond generic boilerplate

---

## Composite Quality Score

```
If heuristic fails  →  quality_score = 0.0,  keep = False
If duplicate        →  quality_score = 0.0,  keep = False
If LLM score available:
    quality_score = 0.3 × heuristic_score + 0.7 × (llm_score − 1) / 9
If heuristic only:
    quality_score = heuristic_score
keep = quality_score ≥ min_quality_score (default 0.5)
```

Weights are fully configurable in `config.yaml` or the sidebar.

---

## Configuration

All thresholds live in `config.yaml` and can be overridden in the sidebar without touching code:

```yaml
filters:
  min_words: 50
  max_symbol_ratio: 0.35
  language: "en"           # null to disable language filtering

dedup:
  threshold: 0.80          # Jaccard similarity (lower = more aggressive)
  num_perm: 128            # MinHash permutations

judge:
  model: "claude-3-5-haiku-20241022"
  min_score: 6             # Documents below this are discarded
  max_docs: 200            # Cost guard: max docs sent to LLM per run

scoring:
  heuristic_weight: 0.3
  llm_weight: 0.7
  min_quality_score: 0.5
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
# → 33 passed
```

---

## Supported Input Formats

| Format | Notes |
|--------|-------|
| `.jsonl` | One JSON object per line — auto-detects `text` / `content` / `body` field |
| `.json` | JSON array of objects or strings |
| `.csv` | Any CSV — text column auto-detected from column names |
| `.parquet` | Columnar format, ideal for datasets > 1M rows |
| `.txt` | Plain text — split into documents by blank lines |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web UI | Streamlit |
| LLM Judge | LangChain + langchain-anthropic (Claude) |
| Vector Store / RAG | ChromaDB + sentence-transformers |
| Deduplication | datasketch (MinHash LSH) |
| Language Detection | langdetect |
| Data I/O | pandas, pyarrow |
| Data Models | Pydantic v2 |
| Logging | loguru |
| Visualisation | Plotly |
| Tests | pytest (33 tests) |
