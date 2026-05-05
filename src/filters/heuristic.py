"""
Heuristic filter bank — fast, cheap pre-screening of text documents.

Each filter returns (passed: bool, flag: str | None).
The HeuristicFilter class applies all checks and annotates a ScoredDocument.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.models import Document, ScoredDocument

# ── English stopwords (bundled — no runtime download required) ────────────────
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "dare", "ought", "used", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "that", "this",
    "these", "those", "it", "its", "itself", "he", "him", "his", "she", "her",
    "hers", "they", "them", "their", "theirs", "we", "us", "our", "ours",
    "you", "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose",
    "which", "what", "where", "when", "why", "how", "all", "any", "both",
    "each", "every", "if", "then", "also", "as", "while", "after", "before",
    "since", "because", "although", "though", "unless", "until", "once",
    "only", "same", "own", "there", "here", "out", "off", "over", "under",
    "again", "further", "once", "now", "then", "between", "against",
}


def _get_stopwords() -> set[str]:
    return _STOPWORDS


# ── langdetect (optional) ─────────────────────────────────────────────────────
_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect, LangDetectException  # type: ignore
    _LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect not installed — language filter disabled")


@dataclass
class FilterConfig:
    min_words: int = 50
    max_words: int = 100_000
    language: Optional[str] = "en"
    min_stopword_ratio: float = 0.08
    max_symbol_ratio: float = 0.35
    max_digit_ratio: float = 0.30
    max_repetition_ratio: float = 0.25
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 15.0

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Individual filter functions ───────────────────────────────────────────────

def _check_length(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    words = text.split()
    n = len(words)
    if n < cfg.min_words:
        return False, f"too_short ({n} words < {cfg.min_words})"
    if n > cfg.max_words:
        return False, f"too_long ({n} words > {cfg.max_words})"
    return True, None


def _check_language(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    if not cfg.language or not _LANGDETECT_AVAILABLE:
        return True, None
    try:
        lang = detect(text[:2000])
        if lang != cfg.language:
            return False, f"wrong_language ({lang} ≠ {cfg.language})"
        return True, None
    except Exception:
        return True, None


def _check_symbol_ratio(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    if not text:
        return False, "empty_text"
    alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
    ratio = 1.0 - (alphanumeric / len(text))
    if ratio > cfg.max_symbol_ratio:
        return False, f"high_symbol_ratio ({ratio:.2f} > {cfg.max_symbol_ratio})"
    return True, None


def _check_digit_ratio(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    if not text:
        return False, "empty_text"
    ratio = sum(1 for c in text if c.isdigit()) / len(text)
    if ratio > cfg.max_digit_ratio:
        return False, f"high_digit_ratio ({ratio:.2f} > {cfg.max_digit_ratio})"
    return True, None


def _check_stopword_ratio(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    stopwords = _get_stopwords()
    if not stopwords:
        return True, None
    words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
    if not words:
        return False, "empty_text"
    sw_count = sum(1 for w in words if w in stopwords)
    ratio = sw_count / len(words)
    if ratio < cfg.min_stopword_ratio:
        return False, f"low_stopword_ratio ({ratio:.2f} < {cfg.min_stopword_ratio}) — possible code/spam"
    return True, None


def _check_repetition(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    """Detect repetitive content via trigram diversity."""
    words = text.lower().split()
    if len(words) < 6:
        return True, None
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return True, None
    unique_ratio = len(set(trigrams)) / len(trigrams)
    if unique_ratio < (1.0 - cfg.max_repetition_ratio):
        return False, f"repetitive_content (trigram_diversity={unique_ratio:.2f})"
    return True, None


def _check_avg_word_length(text: str, cfg: FilterConfig) -> tuple[bool, Optional[str]]:
    words = text.split()
    if not words:
        return False, "empty_text"
    avg = sum(len(w) for w in words) / len(words)
    if avg < cfg.min_avg_word_length:
        return False, f"low_avg_word_length ({avg:.1f} < {cfg.min_avg_word_length}) — possible garbled text"
    if avg > cfg.max_avg_word_length:
        return False, f"high_avg_word_length ({avg:.1f} > {cfg.max_avg_word_length}) — possible code/URLs"
    return True, None


def _compute_heuristic_score(text: str, flags: list[str]) -> float:
    """
    Compute a 0-1 heuristic quality score for documents that pass all filters.
    Based on positive signals rather than penalizing.
    """
    words = text.split()
    n_words = len(words)

    # Sentence diversity (unique sentences / total sentences)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_diversity = len(set(sentences)) / max(len(sentences), 1)

    # Vocabulary richness (type-token ratio, capped at 1)
    ttr = min(len(set(w.lower() for w in words)) / max(n_words, 1), 1.0)

    # Length bonus — prefer 100-5000 word documents
    length_score = min(n_words / 5000, 1.0) if n_words < 5000 else max(0.0, 1.0 - (n_words - 5000) / 95000)

    score = (0.4 * ttr + 0.4 * sent_diversity + 0.2 * length_score)
    return round(min(max(score, 0.0), 1.0), 4)


# ── Main filter class ─────────────────────────────────────────────────────────

class HeuristicFilter:
    def __init__(self, config: FilterConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = FilterConfig.from_dict(config)
        elif config is None:
            self.cfg = FilterConfig()
        else:
            self.cfg = config

    def filter(self, doc: Document) -> ScoredDocument:
        scored = ScoredDocument.from_document(doc)
        scored.word_count = len(doc.text.split())
        scored.char_count = len(doc.text)

        checks = [
            _check_length,
            _check_symbol_ratio,
            _check_digit_ratio,
            _check_avg_word_length,
            _check_stopword_ratio,
            _check_repetition,
            _check_language,
        ]

        flags: list[str] = []
        for check in checks:
            passed, flag = check(doc.text, self.cfg)
            if not passed and flag:
                flags.append(flag)

        scored.heuristic_flags = flags
        scored.heuristic_pass = len(flags) == 0
        scored.language = _detect_language(doc.text) if _LANGDETECT_AVAILABLE else None
        scored.heuristic_score = _compute_heuristic_score(doc.text, flags) if scored.heuristic_pass else 0.0

        return scored

    def filter_batch(self, docs: list[Document]) -> list[ScoredDocument]:
        return [self.filter(doc) for doc in docs]


def _detect_language(text: str) -> Optional[str]:
    if not _LANGDETECT_AVAILABLE:
        return None
    try:
        return detect(text[:2000])
    except Exception:
        return None
