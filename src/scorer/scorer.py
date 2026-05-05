"""
Composite quality scorer — combines heuristic signals and LLM judge scores
into a single quality_score (0-1) and sets the final keep/discard decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models import ScoredDocument


@dataclass
class ScoringConfig:
    heuristic_weight: float = 0.3   # Weight when LLM score is available
    llm_weight: float = 0.7
    min_quality_score: float = 0.5  # Documents below this are discarded

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class QualityScorer:
    def __init__(self, config: ScoringConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = ScoringConfig.from_dict(config)
        elif config is None:
            self.cfg = ScoringConfig()
        else:
            self.cfg = config

    def score(self, doc: ScoredDocument) -> ScoredDocument:
        """
        Compute quality_score and keep flag in-place.

        Logic:
          - Heuristic fail  → 0.0, discard
          - Duplicate       → 0.0, discard
          - LLM score only  → weighted average of heuristic + llm
          - Heuristic only  → heuristic_score as quality_score
        """
        if not doc.heuristic_pass or doc.is_duplicate:
            doc.quality_score = 0.0
            doc.keep = False
            return doc

        if doc.llm_score is not None:
            llm_norm = (doc.llm_score - 1) / 9.0  # map 1-10 → 0-1
            doc.quality_score = round(
                self.cfg.heuristic_weight * doc.heuristic_score
                + self.cfg.llm_weight * llm_norm,
                4,
            )
        else:
            doc.quality_score = doc.heuristic_score

        doc.keep = doc.quality_score >= self.cfg.min_quality_score
        return doc

    def score_batch(self, docs: list[ScoredDocument]) -> list[ScoredDocument]:
        return [self.score(doc) for doc in docs]
