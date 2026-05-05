"""
ChromaDB vector store for two purposes:
  1. RAG few-shot context: retrieve similar high-quality examples for the LLM judge
  2. Semantic dedup: check if a document is too similar to one already seen

Uses sentence-transformers for local embeddings (no API cost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class ChromaConfig:
    collection_name: str = "quality_examples"
    embedding_model: str = "all-MiniLM-L6-v2"
    n_similar_examples: int = 3
    persist_dir: Optional[str] = None   # None = in-memory

    @classmethod
    def from_dict(cls, d: dict) -> "ChromaConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ChromaStore:
    """
    Thin wrapper around ChromaDB that handles embedding + storage.
    Thread-safe for single-process use.
    """

    def __init__(self, config: ChromaConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = ChromaConfig.from_dict(config)
        elif config is None:
            self.cfg = ChromaConfig()
        else:
            self.cfg = config

        self._client = None
        self._collection = None
        self._ef = None
        self._initialized = False

    def _lazy_init(self) -> bool:
        """Initialize ChromaDB + embeddings on first use."""
        if self._initialized:
            return True
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=self.cfg.embedding_model
            )

            if self.cfg.persist_dir:
                self._client = chromadb.PersistentClient(path=self.cfg.persist_dir)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.cfg.collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(
                f"ChromaDB initialized — collection '{self.cfg.collection_name}', "
                f"model '{self.cfg.embedding_model}'"
            )
            return True
        except ImportError as e:
            logger.warning(f"ChromaDB or sentence-transformers not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return False

    def add_example(self, doc_id: str, text: str, score: int, reasoning: str = "") -> None:
        """Add a high-quality document as a reference example."""
        if not self._lazy_init():
            return
        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[text[:2000]],
                metadatas=[{"score": score, "reasoning": reasoning[:500]}],
            )
        except Exception as e:
            logger.warning(f"Failed to add example {doc_id}: {e}")

    def query_similar(self, text: str, n: Optional[int] = None) -> list[dict]:
        """
        Retrieve the top-n most similar high-quality examples.
        Returns list of dicts with keys: text, score, reasoning, distance.
        """
        if not self._lazy_init():
            return []
        n = n or self.cfg.n_similar_examples
        try:
            count = self._collection.count()
            if count == 0:
                return []
            n = min(n, count)
            results = self._collection.query(
                query_texts=[text[:2000]],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            examples = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                examples.append({
                    "text": doc,
                    "score": meta.get("score", "?"),
                    "reasoning": meta.get("reasoning", ""),
                    "distance": dist,
                })
            return examples
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return []

    def is_semantically_duplicate(self, text: str, threshold: float = 0.12) -> tuple[bool, float]:
        """
        Check if a document is semantically too similar to an already-indexed document.
        threshold is cosine distance (lower = more similar; 0 = identical).
        """
        if not self._lazy_init():
            return False, 1.0
        try:
            count = self._collection.count()
            if count == 0:
                return False, 1.0
            results = self._collection.query(
                query_texts=[text[:2000]],
                n_results=1,
                include=["distances"],
            )
            dist = results["distances"][0][0]
            return dist < threshold, dist
        except Exception as e:
            logger.warning(f"Semantic dedup check failed: {e}")
            return False, 1.0

    def count(self) -> int:
        if not self._lazy_init():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def seed_with_gold_examples(self) -> None:
        """
        Pre-populate the store with a small set of hand-curated high-quality examples.
        These serve as few-shot anchors for the LLM judge.
        """
        gold = [
            {
                "id": "gold_001",
                "score": 9,
                "reasoning": "Well-structured, factually accurate, rich vocabulary, clear narrative.",
                "text": (
                    "The transformer architecture, introduced in the paper 'Attention Is All You Need' "
                    "by Vaswani et al. (2017), replaced recurrent networks with a self-attention mechanism "
                    "that allows the model to weigh the relevance of each token against all others in the "
                    "sequence simultaneously. This parallelism dramatically reduced training time while "
                    "enabling longer context windows. The architecture consists of an encoder and decoder, "
                    "each built from stacked layers of multi-head attention and feed-forward networks, "
                    "connected via residual connections and layer normalisation."
                ),
            },
            {
                "id": "gold_002",
                "score": 8,
                "reasoning": "Informative, coherent explanation with domain depth.",
                "text": (
                    "Reinforcement learning from human feedback (RLHF) is a technique used to align "
                    "language models with human preferences. In RLHF, a reward model is first trained on "
                    "human preference data — pairs of model outputs where annotators select the better "
                    "response. The language model is then fine-tuned using proximal policy optimisation "
                    "(PPO) to maximise the reward model's score, steering it towards outputs that humans "
                    "find helpful, harmless, and honest. This process underpins models like InstructGPT, "
                    "Claude, and Gemini."
                ),
            },
            {
                "id": "gold_003",
                "score": 9,
                "reasoning": "Clear technical writing, well-organised, substantive content.",
                "text": (
                    "Data quality in large-scale machine learning is a multi-dimensional problem. "
                    "It encompasses correctness (are the labels accurate?), completeness (is the dataset "
                    "representative of the target distribution?), consistency (are there conflicting "
                    "examples?), and freshness (is the data still relevant?). Poor data quality is often "
                    "the root cause of model failures that appear to be architectural or algorithmic. "
                    "Systematic quality evaluation pipelines — combining heuristic filters, deduplication, "
                    "and LLM-based scoring — have become standard practice in frontier AI labs."
                ),
            },
        ]
        for ex in gold:
            self.add_example(
                doc_id=ex["id"],
                text=ex["text"],
                score=ex["score"],
                reasoning=ex["reasoning"],
            )
        logger.info(f"Seeded ChromaDB with {len(gold)} gold examples")
