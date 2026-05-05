"""Unit tests for MinHash deduplication."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedup.minhash_dedup import DedupConfig, MinHashDedup, _get_shingles
from src.models import ScoredDocument


def make_scored(doc_id: str, text: str, heuristic_pass: bool = True) -> ScoredDocument:
    return ScoredDocument(id=doc_id, text=text, heuristic_pass=heuristic_pass)


GOOD_TEXT = (
    "The transformer architecture introduced self attention which allows each token "
    "to attend to all other tokens in the sequence simultaneously enabling parallelism "
    "during training and inference across many downstream natural language processing tasks."
)

# Differs in only a few words — high Jaccard overlap of 5-gram shingles
NEAR_DUPLICATE = (
    "The transformer architecture introduced self attention which allows each token "
    "to attend to all other tokens in the sequence simultaneously enabling parallelism "
    "during training and inference across many downstream natural language understanding tasks."
)

DIFFERENT_TEXT = (
    "Reinforcement learning from human feedback trains a reward model on human "
    "preference comparisons and then optimises the language model to maximise reward."
)


# ── Shingle tests ─────────────────────────────────────────────────────────────

def test_shingles_non_empty():
    shingles = _get_shingles(GOOD_TEXT, n=5)
    assert len(shingles) > 0


def test_shingles_short_text_fallback():
    shingles = _get_shingles("hello world", n=5)
    assert len(shingles) >= 1


def test_shingles_same_text_identical():
    s1 = _get_shingles(GOOD_TEXT, n=5)
    s2 = _get_shingles(GOOD_TEXT, n=5)
    assert s1 == s2


# ── Deduplication tests ───────────────────────────────────────────────────────

def test_dedup_marks_exact_duplicate():
    deduper = MinHashDedup(DedupConfig(threshold=0.7))
    docs = [
        make_scored("d1", GOOD_TEXT),
        make_scored("d2", GOOD_TEXT),  # exact duplicate
    ]
    result = deduper.deduplicate(docs)
    duplicates = [d for d in result if d.is_duplicate]
    assert len(duplicates) >= 1


def test_dedup_keeps_unique_docs():
    deduper = MinHashDedup(DedupConfig(threshold=0.8))
    docs = [
        make_scored("d1", GOOD_TEXT),
        make_scored("d2", DIFFERENT_TEXT),
    ]
    result = deduper.deduplicate(docs)
    kept = [d for d in result if not d.is_duplicate]
    assert len(kept) == 2


def test_dedup_skips_heuristic_failures():
    deduper = MinHashDedup()
    docs = [
        make_scored("d1", GOOD_TEXT, heuristic_pass=True),
        make_scored("d2", GOOD_TEXT, heuristic_pass=False),  # should not be deduplicated
    ]
    result = deduper.deduplicate(docs)
    failed = next(d for d in result if d.id == "d2")
    assert not failed.is_duplicate  # wasn't even checked


def test_dedup_near_duplicate_at_low_threshold():
    deduper = MinHashDedup(DedupConfig(threshold=0.5))
    docs = [
        make_scored("d1", GOOD_TEXT),
        make_scored("d2", NEAR_DUPLICATE),
    ]
    result = deduper.deduplicate(docs)
    # At low threshold, near-dup should be caught
    has_dup = any(d.is_duplicate for d in result)
    assert has_dup


def test_dedup_near_duplicate_at_high_threshold():
    # threshold=0.95 — very strict; near-dup may or may not be caught; just verify no crash
    deduper = MinHashDedup(DedupConfig(threshold=0.95))
    docs = [
        make_scored("d1", GOOD_TEXT),
        make_scored("d2", NEAR_DUPLICATE),
    ]
    result = deduper.deduplicate(docs)
    assert len(result) == 2


def test_dedup_returns_all_docs():
    deduper = MinHashDedup()
    texts = [GOOD_TEXT, NEAR_DUPLICATE, DIFFERENT_TEXT]
    docs = [make_scored(f"d{i}", t) for i, t in enumerate(texts)]
    result = deduper.deduplicate(docs)
    assert len(result) == len(docs)
