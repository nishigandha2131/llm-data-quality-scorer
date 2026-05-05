"""
CLI pipeline runner — run the full quality scoring pipeline from the terminal.

Usage:
    python scripts/run_pipeline.py --input data/samples/sample.jsonl
    python scripts/run_pipeline.py --input data/raw/corpus.parquet --output data/processed/clean.jsonl
    python scripts/run_pipeline.py --input data/raw/corpus.jsonl --api-key sk-ant-... --judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedup.minhash_dedup import DedupConfig, MinHashDedup
from src.filters.heuristic import FilterConfig, HeuristicFilter
from src.io.loader import load_jsonl, load_csv, load_parquet, load_txt
from src.judge.llm_judge import JudgeConfig, LLMJudge
from src.models import PipelineStats
from src.rag.chroma_store import ChromaConfig, ChromaStore
from src.scorer.scorer import QualityScorer, ScoringConfig


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_documents(input_path: str):
    ext = Path(input_path).suffix.lower()
    loaders = {
        ".jsonl": load_jsonl,
        ".csv": load_csv,
        ".parquet": load_parquet,
        ".txt": load_txt,
        ".json": load_jsonl,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported input format: {ext}")
    docs = loader(input_path)
    logger.info(f"Loaded {len(docs)} documents from {input_path}")
    return docs


def save_output(docs, output_path: str, fmt: str, include_reasoning: bool) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    kept = [d for d in docs if d.keep]
    logger.info(f"Saving {len(kept)} kept documents to {output_path}")

    if fmt in ("jsonl", "json"):
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in kept:
                row = {
                    "id": doc.id,
                    "text": doc.text,
                    "quality_score": doc.quality_score,
                    "heuristic_score": doc.heuristic_score,
                    "word_count": doc.word_count,
                    "language": doc.language,
                }
                if doc.llm_score is not None:
                    row["llm_score"] = doc.llm_score
                if include_reasoning and doc.llm_reasoning:
                    row["llm_reasoning"] = doc.llm_reasoning
                    row["llm_issues"] = doc.llm_issues
                    row["llm_strengths"] = doc.llm_strengths
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elif fmt == "csv":
        import pandas as pd
        rows = []
        for doc in kept:
            rows.append({
                "id": doc.id,
                "text": doc.text,
                "quality_score": doc.quality_score,
                "heuristic_score": doc.heuristic_score,
                "word_count": doc.word_count,
                "language": doc.language,
                "llm_score": doc.llm_score,
            })
        pd.DataFrame(rows).to_csv(output_path, index=False)

    elif fmt == "parquet":
        import pandas as pd
        rows = []
        for doc in kept:
            rows.append({
                "id": doc.id,
                "text": doc.text,
                "quality_score": doc.quality_score,
                "heuristic_score": doc.heuristic_score,
                "word_count": doc.word_count,
                "language": doc.language,
                "llm_score": doc.llm_score,
            })
        pd.DataFrame(rows).to_parquet(output_path, index=False)


def print_stats(stats: PipelineStats) -> None:
    print("\n" + "=" * 50)
    print("  PIPELINE RESULTS")
    print("=" * 50)
    print(f"  Input documents   : {stats.total_input:>6}")
    print(f"  Heuristic passed  : {stats.heuristic_passed:>6}  ({100*stats.heuristic_passed/max(stats.total_input,1):.1f}%)")
    print(f"  Duplicates removed: {stats.duplicates_removed:>6}")
    print(f"  LLM judged        : {stats.llm_judged:>6}")
    print(f"  Final kept        : {stats.final_kept:>6}  ({100*stats.final_kept/max(stats.total_input,1):.1f}%)")
    print("=" * 50)
    if stats.filter_breakdown:
        print("\n  Top filter failure reasons:")
        for flag, count in sorted(stats.filter_breakdown.items(), key=lambda x: -x[1])[:5]:
            print(f"    {flag}: {count}")
    print()


def run_pipeline(
    input_path: str,
    output_path: str,
    config: dict,
    api_key: str | None = None,
    enable_judge: bool = False,
) -> PipelineStats:
    stats = PipelineStats()

    # ── Load ──────────────────────────────────────────────────────────────────
    docs = load_documents(input_path)
    stats.total_input = len(docs)

    # ── Heuristic Filtering ───────────────────────────────────────────────────
    logger.info("Stage 1: Heuristic filtering")
    hf = HeuristicFilter(FilterConfig.from_dict(config.get("filters", {})))
    scored_docs = []
    for doc in tqdm(docs, desc="Filtering"):
        scored = hf.filter(doc)
        scored_docs.append(scored)
        if not scored.heuristic_pass:
            for flag in scored.heuristic_flags:
                key = flag.split("(")[0].strip()
                stats.filter_breakdown[key] = stats.filter_breakdown.get(key, 0) + 1

    stats.heuristic_passed = sum(1 for d in scored_docs if d.heuristic_pass)
    stats.heuristic_failed = stats.total_input - stats.heuristic_passed

    # ── Deduplication ─────────────────────────────────────────────────────────
    if config.get("pipeline", {}).get("dedup", True):
        logger.info("Stage 2: MinHash deduplication")
        deduper = MinHashDedup(DedupConfig.from_dict(config.get("dedup", {})))
        scored_docs = deduper.deduplicate(scored_docs)
        stats.duplicates_removed = sum(1 for d in scored_docs if d.is_duplicate)
        stats.unique_docs = stats.heuristic_passed - stats.duplicates_removed
    else:
        stats.unique_docs = stats.heuristic_passed

    # ── LLM Judge (optional) ──────────────────────────────────────────────────
    if enable_judge and api_key:
        logger.info("Stage 3: LLM-as-Judge (RAG-augmented)")
        chroma_cfg = ChromaConfig.from_dict(config.get("rag", {}))
        store = ChromaStore(chroma_cfg)
        store.seed_with_gold_examples()

        judge_cfg = JudgeConfig.from_dict(config.get("judge", {}))
        judge = LLMJudge(api_key=api_key, config=judge_cfg, chroma_store=store)

        def _progress(i, total):
            print(f"\r  Judging {i}/{total}...", end="", flush=True)

        scored_docs = judge.judge_batch(scored_docs, progress_callback=_progress)
        print()
        stats.llm_judged = sum(1 for d in scored_docs if d.llm_score is not None)

    # ── Final Scoring ─────────────────────────────────────────────────────────
    logger.info("Stage 4: Final quality scoring")
    scorer = QualityScorer(ScoringConfig.from_dict(config.get("scoring", {})))
    scored_docs = scorer.score_batch(scored_docs)
    stats.final_kept = sum(1 for d in scored_docs if d.keep)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_cfg = config.get("output", {})
    save_output(
        scored_docs,
        output_path,
        fmt=out_cfg.get("format", "jsonl"),
        include_reasoning=out_cfg.get("include_reasoning", False),
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LLM Data Quality Scorer — score and filter text corpora for LLM training"
    )
    parser.add_argument("--input", "-i", required=True, help="Input file (.jsonl, .csv, .parquet, .txt)")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: data/processed/clean.jsonl)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config YAML path")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"), help="Anthropic API key")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge stage")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output is None:
        stem = Path(args.input).stem
        fmt = config.get("output", {}).get("format", "jsonl")
        args.output = f"data/processed/{stem}_clean.{fmt}"

    stats = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        config=config,
        api_key=args.api_key,
        enable_judge=args.judge,
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
