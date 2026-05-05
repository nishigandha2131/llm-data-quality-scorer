"""
LLM-as-Judge — uses LangChain + Anthropic Claude to score documents 1-10.

RAG-augmented: retrieves similar high-quality examples from ChromaDB as
few-shot context so Claude can calibrate scores against domain-appropriate
reference documents rather than judging in a vacuum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.models import JudgmentResult, ScoredDocument


@dataclass
class JudgeConfig:
    model: str = "claude-3-5-haiku-20241022"
    min_score: int = 6
    max_docs: int = 200

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_SYSTEM_PROMPT = """\
You are a senior data quality engineer at an AI lab evaluating text documents \
for inclusion in an LLM training corpus.

Score each document on a scale of 1-10 using these criteria:
- **Coherence** (1-10): Is the text logically consistent and well-structured?
- **Informativeness** (1-10): Does it contain substantive, non-trivial information?
- **Language quality** (1-10): Is grammar, spelling, and vocabulary correct and rich?
- **Appropriateness** (1-10): Is it suitable for LLM training (no spam, hate, PII, etc.)?
- **Originality** (1-10): Does it add unique value beyond generic boilerplate?

Return the average of these five dimensions as the overall `score`.
Be strict — only scores of 7+ indicate genuinely high-quality training data.
"""

_USER_TEMPLATE = """\
{few_shot_section}

## Document to Evaluate

<document>
{text}
</document>

Evaluate this document and provide a structured quality assessment.
"""

_FEW_SHOT_TEMPLATE = """\
## Reference Examples (similar high-quality documents for calibration)

{examples}

---
"""


class _StructuredOutput(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality score 1-10")
    reasoning: str = Field(description="2-3 sentence explanation of the score")
    issues: list[str] = Field(default_factory=list, description="Up to 3 specific quality issues")
    strengths: list[str] = Field(default_factory=list, description="Up to 3 specific strengths")


class LLMJudge:
    def __init__(
        self,
        api_key: str,
        config: JudgeConfig | dict | None = None,
        chroma_store=None,
    ):
        if isinstance(config, dict):
            self.cfg = JudgeConfig.from_dict(config)
        elif config is None:
            self.cfg = JudgeConfig()
        else:
            self.cfg = config

        self.store = chroma_store
        self._chain = None
        self._api_key = api_key
        self._build_chain()

    def _build_chain(self) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatAnthropic(
                model=self.cfg.model,
                api_key=self._api_key,
                temperature=0.1,
                max_tokens=512,
            )
            structured_llm = llm.with_structured_output(_StructuredOutput)

            prompt = ChatPromptTemplate.from_messages([
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_TEMPLATE),
            ])

            self._chain = prompt | structured_llm
            logger.info(f"LLM Judge ready — model: {self.cfg.model}")
        except ImportError as e:
            logger.error(f"LangChain/Anthropic not installed: {e}")
            self._chain = None

    def _build_few_shot_section(self, text: str) -> str:
        if self.store is None:
            return ""
        examples = self.store.query_similar(text)
        if not examples:
            return ""
        parts = []
        for i, ex in enumerate(examples, 1):
            parts.append(
                f"**Example {i}** (Score: {ex['score']}/10)\n"
                f"{ex['text'][:600]}\n"
                f"*Why it scores high: {ex['reasoning']}*"
            )
        return _FEW_SHOT_TEMPLATE.format(examples="\n\n".join(parts))

    def judge(self, doc: ScoredDocument) -> Optional[JudgmentResult]:
        if self._chain is None:
            return None
        try:
            few_shot = self._build_few_shot_section(doc.text)
            result: _StructuredOutput = self._chain.invoke({
                "few_shot_section": few_shot,
                "text": doc.text[:3000],
            })
            return JudgmentResult(
                score=result.score,
                reasoning=result.reasoning,
                issues=result.issues[:3],
                strengths=result.strengths[:3],
            )
        except Exception as e:
            logger.warning(f"LLM judge failed for {doc.id}: {e}")
            return None

    def judge_batch(
        self,
        docs: list[ScoredDocument],
        progress_callback=None,
    ) -> list[ScoredDocument]:
        """
        Judge a batch of documents. Only judges docs that passed heuristics
        and are not duplicates. Respects max_docs cost guard.
        """
        candidates = [
            d for d in docs if d.heuristic_pass and not d.is_duplicate
        ]
        if len(candidates) > self.cfg.max_docs:
            logger.warning(
                f"Capping LLM judge at {self.cfg.max_docs}/{len(candidates)} docs"
            )
            candidates = candidates[: self.cfg.max_docs]

        for i, doc in enumerate(candidates):
            result = self.judge(doc)
            if result:
                doc.llm_score = result.score
                doc.llm_reasoning = result.reasoning
                doc.llm_issues = result.issues
                doc.llm_strengths = result.strengths

                # Feed high-quality docs back into the RAG store
                if self.store and result.score >= 8:
                    self.store.add_example(
                        doc_id=doc.id,
                        text=doc.text,
                        score=result.score,
                        reasoning=result.reasoning,
                    )

            if progress_callback:
                progress_callback(i + 1, len(candidates))

        return docs
