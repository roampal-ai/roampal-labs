"""Wilson scoring + cross-encoder reranking — no knowledge graph.

Combines:
- Flat ChromaDB collection with Wilson confidence scoring (same as wilson_scored.py)
- Cross-encoder reranking on cosine candidates
- Blended scoring: CE for relevance, Wilson for reliability

This is the clean combo: best retrieval (cross-encoder) + best outcome scoring (Wilson).
No KG, no entity extraction, no graph expansion. Just the two things that work.
"""

import json
import math
import time
import uuid
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from .base import MemoryRecord, RetrievalResult


# === Wilson score — exact copy from wilson_scored.py ===

_Z_SCORES = {0.90: 1.6448536269514729, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_score(confidence: float) -> float:
    if confidence in _Z_SCORES:
        return _Z_SCORES[confidence]
    p = (1 - confidence) / 2
    t = math.sqrt(-2 * math.log(p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)


def wilson_score_lower(successes: float, total: int, confidence: float = 0.95) -> float:
    if total == 0:
        return 0.5
    z = _z_score(confidence)
    p = successes / total
    n = total
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    variance = p * (1 - p) / n + z * z / (4 * n * n)
    lower_bound = (center - z * math.sqrt(variance)) / denominator
    return max(0.0, lower_bound)


def get_dynamic_weights(uses: int, learned_score: float) -> tuple:
    """5-tier dynamic weighting — exact copy from roampal-cli scoring_service.py."""
    if uses >= 5 and learned_score >= 0.8:
        return (0.2, 0.8)   # PROVEN HIGH-VALUE
    elif uses >= 3 and learned_score >= 0.7:
        return (0.25, 0.75)  # ESTABLISHED
    elif uses >= 2 and learned_score >= 0.5:
        return (0.35, 0.65)  # EMERGING POSITIVE
    elif uses >= 2:
        return (0.7, 0.3)   # FAILING PATTERN
    else:
        return (0.8, 0.2)   # NEW/UNKNOWN


def calculate_learned_score(raw_score: float, uses: int, success_count: float) -> tuple:
    successes = success_count
    if successes == 0 and uses > 0:
        successes = raw_score * uses
    wilson = wilson_score_lower(successes, uses)
    if uses == 0:
        learned = raw_score
    elif uses < 3:
        blend = uses / 3
        learned = (1 - blend) * raw_score + blend * wilson
    else:
        learned = wilson
    return learned, wilson


class WilsonRerankerStrategy:
    """Wilson scoring + cross-encoder reranking. No KG.

    Retrieval: cosine candidates -> cross-encoder rerank -> blend CE score with Wilson.
    Outcome scoring: same deltas as roampal-cli (worked +0.2, failed -0.3, etc).
    One flat ChromaDB collection, no entity extraction, no graph expansion.
    """

    name = "wilson_reranker"

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
        meta["score"] = 0.5
        meta["uses"] = 0
        meta["success_count"] = 0.0
        meta["outcome_history"] = "[]"

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
            distance = results["distances"][0][i] if results["distances"] else 1.0

            candidates.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": meta,
                "distance": distance,
            })

        # Phase 2: Cross-encoder rerank
        if self._reranker and len(candidates) > 0:
            pairs = [[query, c["content"]] for c in candidates]
            ce_scores = self._reranker.predict(pairs)
            if isinstance(ce_scores, np.ndarray):
                ce_scores = ce_scores.tolist()

            # Normalize CE scores to 0-1 range
            ce_min = min(ce_scores)
            ce_max = max(ce_scores)
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

            for i, c in enumerate(candidates):
                c["ce_score"] = ce_scores[i]
                ce_norm = (ce_scores[i] - ce_min) / ce_range

                # Wilson scoring
                raw_score = float(c["metadata"].get("score", 0.5))
                uses = int(c["metadata"].get("uses", 0))
                success_count = float(c["metadata"].get("success_count", 0.0))
                learned_score, wilson = calculate_learned_score(raw_score, uses, success_count)

                # Blend: CE for relevance, Wilson for reliability
                # New memories (< 3 uses): 90% CE, 10% Wilson
                # Proven memories (3+ uses): 60% CE, 40% Wilson
                if uses >= 3:
                    c["combined_score"] = 0.6 * ce_norm + 0.4 * learned_score
                else:
                    c["combined_score"] = 0.9 * ce_norm + 0.1 * learned_score

            candidates.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        # Take top_k after blended reranking
        top_candidates = candidates[:top_k]

        memories = []
        for c in top_candidates:
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                score=c.get("combined_score", 0.5),
                access_count=int(c["metadata"].get("uses", 0)),
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
        # Same score deltas as roampal-cli
        if outcome == "worked":
            score_delta, success_delta = 0.2, 1.0
        elif outcome == "failed":
            score_delta, success_delta = -0.3, 0.0
        elif outcome == "partial":
            score_delta, success_delta = 0.05, 0.5
        elif outcome == "unknown":
            score_delta, success_delta = -0.05, 0.25
        else:
            score_delta, success_delta = 0.0, 0.0

        if memory_ids:
            try:
                existing = self._collection.get(ids=memory_ids, include=["metadatas"])
                for i, doc_id in enumerate(existing["ids"]):
                    meta = existing["metadatas"][i] if existing["metadatas"] else {}

                    current_score = float(meta.get("score", 0.5))
                    uses = int(meta.get("uses", 0))
                    success_count = float(meta.get("success_count", 0.0))

                    new_score = max(0.0, min(1.0, current_score + score_delta))
                    uses += 1
                    success_count += success_delta

                    history = json.loads(meta.get("outcome_history", "[]"))
                    history.append({"outcome": outcome, "timestamp": str(time.time())})
                    history = history[-10:]

                    meta["score"] = new_score
                    meta["uses"] = uses
                    meta["success_count"] = success_count
                    meta["last_outcome"] = outcome
                    meta["outcome_history"] = json.dumps(history)

                    self._collection.update(ids=[doc_id], metadatas=[meta])

                    # Decay: delete memories that score below threshold
                    # Matches production roampal-core (deletion_score_threshold=0.2)
                    # UNCOMMENT FOR POISON RUN:
                    # if new_score < 0.2:
                    #     self._collection.delete(ids=[doc_id])
            except Exception:
                pass

        if exchange_summary:
            await self.store(exchange_summary)

    async def get_stats(self) -> Dict[str, Any]:
        count = self._collection.count() if self._collection else 0
        avg_wilson = 0.5
        if count > 0:
            try:
                all_meta = self._collection.get(include=["metadatas"])
                scores = []
                for m in all_meta["metadatas"]:
                    sc = float(m.get("success_count", 0))
                    uses = int(m.get("uses", 0))
                    scores.append(wilson_score_lower(sc, uses))
                avg_wilson = sum(scores) / len(scores) if scores else 0.5
            except Exception:
                pass
        return {
            "memories": count,
            "strategy": self.name,
            "avg_wilson": round(avg_wilson, 3),
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
