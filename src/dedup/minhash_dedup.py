"""
MinHash LSH deduplication — removes near-duplicate documents efficiently.

Uses datasketch MinHash + LSH index. Runs in O(n) amortized time
making it suitable for millions of documents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.models import ScoredDocument


@dataclass
class DedupConfig:
    threshold: float = 0.80    # Jaccard similarity threshold
    num_perm: int = 128         # MinHash permutations
    ngram_size: int = 5         # Word n-gram shingle size

    @classmethod
    def from_dict(cls, d: dict) -> "DedupConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _get_shingles(text: str, n: int = 5) -> set[str]:
    """Generate word n-gram shingles from text."""
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


class MinHashDedup:
    def __init__(self, config: DedupConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = DedupConfig.from_dict(config)
        elif config is None:
            self.cfg = DedupConfig()
        else:
            self.cfg = config
        self._lsh = None
        self._minhashes: dict[str, object] = {}
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            from datasketch import MinHash, MinHashLSH  # noqa: F401
            return True
        except ImportError:
            logger.warning("datasketch not installed — deduplication disabled")
            return False

    def _build_minhash(self, text: str):
        from datasketch import MinHash
        m = MinHash(num_perm=self.cfg.num_perm)
        for shingle in _get_shingles(text, self.cfg.ngram_size):
            m.update(shingle.encode("utf-8"))
        return m

    def deduplicate(self, docs: list[ScoredDocument]) -> list[ScoredDocument]:
        """
        Mark duplicates in-place. Returns all docs with is_duplicate set.
        Only compares docs that passed heuristic filtering.
        """
        if not self._available:
            logger.warning("Skipping deduplication — datasketch unavailable")
            return docs

        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=self.cfg.threshold, num_perm=self.cfg.num_perm)
        minhashes: dict[str, object] = {}

        candidates = [d for d in docs if d.heuristic_pass]
        logger.info(f"Deduplicating {len(candidates)} documents (threshold={self.cfg.threshold})")

        n_dupes = 0
        for doc in candidates:
            m = self._build_minhash(doc.text)
            minhashes[doc.id] = m

            try:
                results = lsh.query(m)
            except Exception:
                results = []

            if results:
                doc.is_duplicate = True
                doc.duplicate_of = results[0]
                n_dupes += 1
            else:
                lsh.insert(doc.id, m)

        logger.info(f"Found {n_dupes} duplicates out of {len(candidates)} candidates")
        return docs

    def reset(self) -> None:
        self._lsh = None
        self._minhashes = {}
