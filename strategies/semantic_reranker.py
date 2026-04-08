"""Semantic retrieval + cross-encoder reranking. Industry-standard RAG."""

import time
import uuid
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from .base import MemoryRecord, RetrievalResult


class SemanticRerankerStrategy:
    """RAG with cross-encoder reranking.

    Retrieve top-N candidates by cosine similarity, then rerank with
    a cross-encoder model for better precision. This is considered
    the industry standard for production RAG systems.
    """

    name = "semantic_reranker"

    def __init__(
        self,
        persist_dir: str = "",
        collection_name: str = "memories",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_pool: int = 20,
    ):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._reranker_model = reranker_model
        self._candidate_pool = candidate_pool
        self._client = None
        self._collection = None
        self._reranker = None

    async def initialize(self) -> None:
        from sentence_transformers import CrossEncoder

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._reranker = CrossEncoder(self._reranker_model, device="cuda")

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = f"mem_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta["stored_at"] = time.time()

        # Dedup: skip if near-duplicate exists (cosine distance < 0.1)
        if meta.get("type") == "fact" and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_texts=[content], n_results=1, include=["distances"],
                )
                if (results and results["distances"] and results["distances"][0]
                        and results["distances"][0][0] < 0.1):
                    return ""
            except Exception:
                pass

        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )
        return doc_id

    async def retrieve(self, query: str, top_k: int = 4, type_filter: str = None, type_exclude: str = None) -> RetrievalResult:
        t0 = time.time()

        count = self._collection.count()
        if count == 0:
            return RetrievalResult(memories=[], formatted_injection="", query_used=query)

        # Phase 1: Retrieve broad candidate pool via cosine
        # Use ChromaDB where-clause for type filtering to avoid pool starvation
        # (post-filtering a pool of 20 loses ~80% when facts dominate)
        pool_size = min(self._candidate_pool, count)
        query_kwargs = {
            "query_texts": [query],
            "n_results": pool_size,
            "include": ["documents", "metadatas", "distances"],
        }
        if type_filter:
            query_kwargs["where"] = {"type": type_filter}
        elif type_exclude:
            query_kwargs["where"] = {"type": {"$ne": type_exclude}}

        results = self._collection.query(**query_kwargs)

        candidates = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            candidates.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": meta,
                "cosine_score": 1.0 - (results["distances"][0][i] if results["distances"] else 0.5),
            })

        # Phase 2: Rerank with cross-encoder
        if self._reranker and len(candidates) > 0:
            pairs = [[query, c["content"]] for c in candidates]
            ce_scores = self._reranker.predict(pairs)
            if isinstance(ce_scores, np.ndarray):
                ce_scores = ce_scores.tolist()
            for i, c in enumerate(candidates):
                c["ce_score"] = ce_scores[i]
            candidates.sort(key=lambda x: x["ce_score"], reverse=True)

        # Take top_k after reranking
        top_candidates = candidates[:top_k]

        memories = []
        for c in top_candidates:
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                score=c.get("ce_score", c["cosine_score"]),
                collection="working",
            ))

        formatted = self._format_injection(memories)
        elapsed_ms = (time.time() - t0) * 1000

        return RetrievalResult(
            memories=memories,
            formatted_injection=formatted,
            query_used=query,
            retrieval_ms=elapsed_ms,
        )

    async def record_outcome(
        self, memory_ids: List[str], outcome: str, exchange_summary: str = ""
    ) -> None:
        # Reranker doesn't learn from outcomes — just store the exchange
        if exchange_summary:
            await self.store(exchange_summary)

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "memories": self._collection.count() if self._collection else 0,
            "strategy": self.name,
            "reranker_model": self._reranker_model,
            "candidate_pool": self._candidate_pool,
        }

    async def cleanup(self) -> None:
        pass

    def _format_injection(self, memories: List[MemoryRecord]) -> str:
        if not memories:
            return ""
        parts = ["[Retrieved memories]"]
        for m in memories:
            parts.append(f"- {m.content}")
        return "\n".join(parts)
