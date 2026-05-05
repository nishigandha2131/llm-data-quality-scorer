"""Unit tests for the composite quality scorer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer.scorer import QualityScorer, ScoringConfig
from src.models import ScoredDocument


def make_doc(**kwargs) -> ScoredDocument:
    defaults = {
        "id": "test",
        "text": "sample text",
        "heuristic_pass": True,
        "is_duplicate": False,
        "heuristic_score": 0.7,
        "llm_score": None,
    }
    defaults.update(kwargs)
    return ScoredDocument(**defaults)


# ── Discard conditions ────────────────────────────────────────────────────────

def test_heuristic_fail_scores_zero():
    scorer = QualityScorer()
    doc = make_doc(heuristic_pass=False, heuristic_score=0.0)
    scorer.score(doc)
    assert doc.quality_score == 0.0
    assert doc.keep is False


def test_duplicate_scores_zero():
    scorer = QualityScorer()
    doc = make_doc(is_duplicate=True, heuristic_score=0.8)
    scorer.score(doc)
    assert doc.quality_score == 0.0
    assert doc.keep is False


# ── Heuristic-only scoring ────────────────────────────────────────────────────

def test_heuristic_only_uses_heuristic_score():
    scorer = QualityScorer()
    doc = make_doc(heuristic_score=0.7, llm_score=None)
    scorer.score(doc)
    assert doc.quality_score == 0.7


def test_heuristic_only_keep_above_threshold():
    cfg = ScoringConfig(min_quality_score=0.5)
    scorer = QualityScorer(cfg)
    doc = make_doc(heuristic_score=0.8, llm_score=None)
    scorer.score(doc)
    assert doc.keep is True


def test_heuristic_only_discard_below_threshold():
    cfg = ScoringConfig(min_quality_score=0.5)
    scorer = QualityScorer(cfg)
    doc = make_doc(heuristic_score=0.3, llm_score=None)
    scorer.score(doc)
    assert doc.keep is False


# ── Combined scoring ──────────────────────────────────────────────────────────

def test_combined_score_with_high_llm():
    cfg = ScoringConfig(heuristic_weight=0.3, llm_weight=0.7, min_quality_score=0.5)
    scorer = QualityScorer(cfg)
    doc = make_doc(heuristic_score=0.6, llm_score=10)
    scorer.score(doc)
    # llm_norm = (10-1)/9 = 1.0; score = 0.3*0.6 + 0.7*1.0 = 0.88
    assert abs(doc.quality_score - 0.88) < 0.01
    assert doc.keep is True


def test_combined_score_with_low_llm():
    cfg = ScoringConfig(heuristic_weight=0.3, llm_weight=0.7, min_quality_score=0.5)
    scorer = QualityScorer(cfg)
    doc = make_doc(heuristic_score=0.7, llm_score=1)
    scorer.score(doc)
    # llm_norm = (1-1)/9 = 0.0; score = 0.3*0.7 + 0.7*0.0 = 0.21
    assert abs(doc.quality_score - 0.21) < 0.01
    assert doc.keep is False


def test_llm_score_6_near_threshold():
    cfg = ScoringConfig(heuristic_weight=0.3, llm_weight=0.7, min_quality_score=0.5)
    scorer = QualityScorer(cfg)
    doc = make_doc(heuristic_score=0.6, llm_score=6)
    scorer.score(doc)
    # llm_norm = 5/9 ≈ 0.556; score = 0.3*0.6 + 0.7*0.556 ≈ 0.569
    assert doc.quality_score > 0.5
    assert doc.keep is True


# ── Batch scoring ─────────────────────────────────────────────────────────────

def test_batch_scoring():
    scorer = QualityScorer()
    docs = [
        make_doc(id="1", heuristic_pass=True, heuristic_score=0.8),
        make_doc(id="2", heuristic_pass=False, heuristic_score=0.0),
        make_doc(id="3", is_duplicate=True, heuristic_score=0.7),
    ]
    results = scorer.score_batch(docs)
    assert results[0].keep is True
    assert results[1].keep is False
    assert results[2].keep is False


# ── Quality score bounds ──────────────────────────────────────────────────────

def test_quality_score_between_0_and_1():
    scorer = QualityScorer()
    for llm in [1, 5, 10, None]:
        doc = make_doc(heuristic_score=0.5, llm_score=llm)
        scorer.score(doc)
        assert 0.0 <= doc.quality_score <= 1.0
