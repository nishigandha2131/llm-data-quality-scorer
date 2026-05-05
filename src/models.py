from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgmentResult(BaseModel):
    """Structured output from the LLM-as-Judge."""
    score: int = Field(ge=1, le=10, description="Overall quality score 1-10")
    reasoning: str = Field(description="Explanation of the score")
    issues: list[str] = Field(default_factory=list, description="Specific quality issues")
    strengths: list[str] = Field(default_factory=list, description="Specific strengths")


class ScoredDocument(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Heuristic stage ──────────────────────────────────────────────────────
    heuristic_pass: bool = True
    heuristic_flags: list[str] = Field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    language: Optional[str] = None

    # ── Deduplication stage ──────────────────────────────────────────────────
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    # ── LLM Judge stage ──────────────────────────────────────────────────────
    llm_score: Optional[int] = None
    llm_reasoning: Optional[str] = None
    llm_issues: list[str] = Field(default_factory=list)
    llm_strengths: list[str] = Field(default_factory=list)

    # ── Final scoring ────────────────────────────────────────────────────────
    heuristic_score: float = 0.0   # 0-1 composite of heuristic signals
    quality_score: float = 0.0     # final combined score 0-1
    keep: bool = False

    @classmethod
    def from_document(cls, doc: Document) -> "ScoredDocument":
        return cls(
            id=doc.id,
            text=doc.text,
            source=doc.source,
            metadata=doc.metadata,
        )


class PipelineStats(BaseModel):
    total_input: int = 0
    heuristic_passed: int = 0
    heuristic_failed: int = 0
    unique_docs: int = 0
    duplicates_removed: int = 0
    llm_judged: int = 0
    final_kept: int = 0
    filter_breakdown: dict[str, int] = Field(default_factory=dict)
