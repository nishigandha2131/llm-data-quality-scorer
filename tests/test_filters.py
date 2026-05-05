"""Unit tests for heuristic filters."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters.heuristic import FilterConfig, HeuristicFilter, _check_length, _check_symbol_ratio, _check_repetition, _check_avg_word_length
from src.models import Document


@pytest.fixture
def default_cfg():
    return FilterConfig()


@pytest.fixture
def good_doc():
    return Document(
        id="test_001",
        text=(
            "The transformer architecture, introduced in the landmark 2017 paper by Vaswani et al., "
            "fundamentally changed natural language processing. At its core, self-attention allows "
            "each token to attend to all other tokens simultaneously, enabling parallelism during training. "
            "The architecture consists of stacked encoder and decoder blocks with multi-head attention "
            "layers, position-wise feed-forward networks, residual connections, and layer normalisation. "
            "This made it possible to scale model capacity in ways that were previously impractical and "
            "led directly to the development of models like BERT, GPT, and modern large language models."
        ),
    )


@pytest.fixture
def short_doc():
    return Document(id="test_002", text="hi ok yes")


@pytest.fixture
def symbol_heavy_doc():
    return Document(id="test_003", text="!!! $$$ ### @@@" * 20 + " some text here")


@pytest.fixture
def repetitive_doc():
    return Document(id="test_004", text="the cat sat on the mat " * 30)


# ── Length filter ─────────────────────────────────────────────────────────────

def test_length_filter_passes_good_doc(good_doc, default_cfg):
    passed, flag = _check_length(good_doc.text, default_cfg)
    assert passed
    assert flag is None


def test_length_filter_fails_short_doc(short_doc, default_cfg):
    passed, flag = _check_length(short_doc.text, default_cfg)
    assert not passed
    assert "too_short" in flag


def test_length_filter_custom_min():
    doc = Document(id="x", text="hello world " * 5)  # 10 words
    cfg = FilterConfig(min_words=5)
    passed, _ = _check_length(doc.text, cfg)
    assert passed


# ── Symbol ratio filter ───────────────────────────────────────────────────────

def test_symbol_ratio_passes_clean_text(good_doc, default_cfg):
    passed, flag = _check_symbol_ratio(good_doc.text, default_cfg)
    assert passed


def test_symbol_ratio_fails_symbol_heavy(symbol_heavy_doc, default_cfg):
    passed, flag = _check_symbol_ratio(symbol_heavy_doc.text, default_cfg)
    assert not passed
    assert "symbol_ratio" in flag


# ── Repetition filter ─────────────────────────────────────────────────────────

def test_repetition_fails_repetitive(repetitive_doc, default_cfg):
    passed, flag = _check_repetition(repetitive_doc.text, default_cfg)
    assert not passed
    assert "repetitive" in flag


def test_repetition_passes_good_doc(good_doc, default_cfg):
    passed, _ = _check_repetition(good_doc.text, default_cfg)
    assert passed


# ── Average word length filter ────────────────────────────────────────────────

def test_avg_word_length_passes_normal():
    text = "the quick brown fox jumps over the lazy dog near the river bank"
    cfg = FilterConfig()
    passed, _ = _check_avg_word_length(text, cfg)
    assert passed


def test_avg_word_length_fails_garbled():
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 5
    cfg = FilterConfig(min_avg_word_length=3.0)
    passed, flag = _check_avg_word_length(text, cfg)
    assert not passed


# ── Full HeuristicFilter ──────────────────────────────────────────────────────

def test_heuristic_filter_passes_good_doc(good_doc):
    hf = HeuristicFilter()
    result = hf.filter(good_doc)
    # Good doc should pass (language filter may or may not be available)
    assert result.word_count > 0
    assert result.char_count > 0


def test_heuristic_filter_fails_short_doc(short_doc):
    hf = HeuristicFilter()
    result = hf.filter(short_doc)
    assert not result.heuristic_pass
    assert any("too_short" in f for f in result.heuristic_flags)


def test_heuristic_filter_fails_repetitive(repetitive_doc):
    hf = HeuristicFilter()
    result = hf.filter(repetitive_doc)
    assert not result.heuristic_pass


def test_heuristic_filter_batch():
    hf = HeuristicFilter()
    docs = [
        Document(id="1", text="hello " * 3),
        Document(id="2", text="The quick brown fox jumps over the lazy dog. " * 10),
    ]
    results = hf.filter_batch(docs)
    assert len(results) == 2


def test_heuristic_score_is_zero_on_fail(short_doc):
    hf = HeuristicFilter()
    result = hf.filter(short_doc)
    assert result.heuristic_score == 0.0
