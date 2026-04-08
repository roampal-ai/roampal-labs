"""KG-Traversal strategy — Wilson-scored relationship edges + CE reranking.

Pipeline:
1. At store time: LLM extracts subject-verb-object triples from each memory
2. At retrieve time:
   a. Cosine pool: top 15 by embedding similarity
   b. KG pool: query entities → traverse Wilson-scored edges → linked memories (top 5)
   c. Nursery pool: top 5 new memories by cosine (uses < 3)
   d. CE ranks all candidates → top 3 from proven/KG + top 1 from nursery
3. At score time: Wilson updates both memory scores AND edge scores

Wilson on edges = query-aware scoring. Wilson on memories = global scoring.
The KG gives memories a second discovery path beyond cosine similarity.
"""

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
import numpy as np

from .base import MemoryRecord, RetrievalResult


# === Wilson score ===

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
    if uses >= 5 and learned_score >= 0.8:
        return (0.2, 0.8)
    elif uses >= 3 and learned_score >= 0.7:
        return (0.25, 0.75)
    elif uses >= 2 and learned_score >= 0.5:
        return (0.35, 0.65)
    elif uses >= 2:
        return (0.7, 0.3)
    else:
        return (0.8, 0.2)


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


# === Triple Store — Wilson-scored edges ===

class TripleStore:
    """Simple subject-verb-object triple store with Wilson scoring per edge."""

    def __init__(self, persist_path: str):
        self._path = Path(persist_path) / "triples.json"
        # edges: {edge_key: {subject, verb, object, memory_id, uses, successes, score}}
        self._edges: Dict[str, dict] = {}
        # index: entity -> list of edge_keys (for traversal)
        self._entity_index: Dict[str, List[str]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._edges = data.get("edges", {})
                self._rebuild_index()
            except Exception:
                pass

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"edges": self._edges}, f)
        tmp.replace(self._path)

    def _rebuild_index(self):
        self._entity_index = {}
        for key, edge in self._edges.items():
            for entity in [edge["subject"], edge["object"]]:
                e = entity.lower()
                if e not in self._entity_index:
                    self._entity_index[e] = []
                self._entity_index[e].append(key)

    def add_triples(self, triples: List[Tuple[str, str, str]], memory_id: str):
        for subj, verb, obj in triples:
            key = f"{subj.lower()}|{verb.lower()}|{obj.lower()}"
            self._edges[key] = {
                "subject": subj.lower(),
                "verb": verb.lower(),
                "object": obj.lower(),
                "memory_id": memory_id,
                "uses": 0,
                "successes": 0.0,
                "score": 0.5,
            }
            # Index both subject and object
            for entity in [subj.lower(), obj.lower()]:
                if entity not in self._entity_index:
                    self._entity_index[entity] = []
                if key not in self._entity_index[entity]:
                    self._entity_index[entity].append(key)
        self._save()

    def traverse(self, query_entities: List[str], top_k: int = 5) -> List[str]:
        """Find memory_ids by traversing from query entities through scored edges."""
        candidates = {}  # memory_id -> best_edge_wilson
        for entity in query_entities:
            e = entity.lower()
            for key in self._entity_index.get(e, []):
                edge = self._edges[key]
                wilson = wilson_score_lower(edge["successes"], edge["uses"])
                mid = edge["memory_id"]
                if mid not in candidates or wilson > candidates[mid]:
                    candidates[mid] = wilson
        # Sort by Wilson score, return top-k memory_ids
        sorted_ids = sorted(candidates.keys(), key=lambda m: candidates[m], reverse=True)
        return sorted_ids[:top_k]

    def score_edges_for_memory(self, memory_id: str, outcome: str):
        """Score all edges that point to this memory."""
        if outcome == "worked":
            success_delta = 1.0
        elif outcome == "failed":
            success_delta = 0.0
        elif outcome == "partial":
            success_delta = 0.5
        else:
            success_delta = 0.25

        for key, edge in self._edges.items():
            if edge["memory_id"] == memory_id:
                edge["uses"] += 1
                edge["successes"] += success_delta
        self._save()

    def get_stats(self) -> dict:
        total_edges = len(self._edges)
        total_entities = len(self._entity_index)
        avg_wilson = 0.5
        if total_edges > 0:
            scores = [wilson_score_lower(e["successes"], e["uses"]) for e in self._edges.values()]
            avg_wilson = sum(scores) / len(scores)
        return {"edges": total_edges, "entities": total_entities, "avg_edge_wilson": round(avg_wilson, 3)}


# === LLM Triple Extraction ===

async def extract_triples(text: str, llm_client, llm_model: str) -> List[Tuple[str, str, str]]:
    """Ask the LLM to extract subject-verb-object triples from memory text."""
    prompt = (
        "Extract 1-3 subject-verb-object triples from this text. "
        "Return ONLY a JSON array of arrays like: [[\"subject\",\"verb\",\"object\"]]\n"
        "Keep subjects and objects as short noun phrases. Keep verbs as single words or short phrases.\n"
        f"Text: {text[:500]}"
    )
    try:
        resp = await llm_client.post("/chat/completions", json={
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        })
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # Parse JSON from response — handle markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            triples = json.loads(content)
            if isinstance(triples, list):
                return [(t[0], t[1], t[2]) for t in triples if isinstance(t, list) and len(t) >= 3]
    except Exception:
        pass
    return []


# === Extract entities from query (simple word matching against known nodes) ===

def extract_query_entities(query: str, known_entities: Set[str]) -> List[str]:
    """Find query words/phrases that match known entity nodes."""
    query_lower = query.lower()
    matches = []
    # Check multi-word entities first (longer matches preferred)
    sorted_entities = sorted(known_entities, key=len, reverse=True)
    for entity in sorted_entities:
        if entity in query_lower and len(entity) >= 3:
            matches.append(entity)
    return matches[:5]  # Cap at 5 entities


class KGTraversalStrategy:
    """Wilson-scored KG edges + CE reranking.

    Store: ChromaDB + LLM triple extraction → TripleStore
    Retrieve: cosine pool + KG traversal pool + nursery pool → CE picks final 4
    Score: Wilson on memories AND Wilson on edges
    """

    name = "kg_traversal"

    def __init__(
        self,
        persist_dir: str = "",
        collection_name: str = "memories",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "gpt-oss:20b",
    ):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._reranker_model = reranker_model
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._client = None
        self._collection = None
        self._reranker = None
        self._triple_store: Optional[TripleStore] = None
        self._llm_client = None

    async def initialize(self) -> None:
        import httpx
        from sentence_transformers import CrossEncoder

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._reranker = CrossEncoder(self._reranker_model)
        self._triple_store = TripleStore(self._persist_dir)
        self._llm_client = httpx.AsyncClient(
            base_url=self._llm_base_url,
            headers={"Authorization": "Bearer ollama"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = f"mem_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta["stored_at"] = time.time()
        meta["score"] = 0.5
        meta["uses"] = 0
        meta["success_count"] = 0.0
        meta["outcome_history"] = "[]"
        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )

        # Extract triples and add to KG
        triples = await extract_triples(content, self._llm_client, self._llm_model)
        if triples:
            self._triple_store.add_triples(triples, doc_id)

        return doc_id

    async def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        t0 = time.time()

        count = self._collection.count()
        if count == 0:
            return RetrievalResult(memories=[], formatted_injection="", query_used=query)

        seen_ids = set()

        # === COSINE POOL: top 15 by embedding similarity ===
        cosine_candidates = []
        cosine_size = min(15, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=cosine_size,
            include=["documents", "metadatas", "distances"],
        )
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                seen_ids.add(doc_id)
                cosine_candidates.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "pool": "cosine",
                })

        # === KG POOL: traverse Wilson-scored edges from query entities ===
        kg_candidates = []
        query_entities = extract_query_entities(query, set(self._triple_store._entity_index.keys()))
        if query_entities:
            kg_memory_ids = self._triple_store.traverse(query_entities, top_k=5)
            # Fetch memories by ID from ChromaDB
            unseen_kg_ids = [mid for mid in kg_memory_ids if mid not in seen_ids]
            if unseen_kg_ids:
                try:
                    kg_results = self._collection.get(
                        ids=unseen_kg_ids,
                        include=["documents", "metadatas"],
                    )
                    if kg_results and kg_results["ids"]:
                        for i, doc_id in enumerate(kg_results["ids"]):
                            seen_ids.add(doc_id)
                            kg_candidates.append({
                                "id": doc_id,
                                "content": kg_results["documents"][i],
                                "metadata": kg_results["metadatas"][i] if kg_results["metadatas"] else {},
                                "pool": "kg",
                            })
                except Exception:
                    pass

        # === NURSERY POOL: top 5 new memories by cosine ===
        nursery_candidates = []
        try:
            nursery_results = self._collection.query(
                query_texts=[query],
                n_results=min(5, count),
                where={"uses": {"$lt": 3}},
                include=["documents", "metadatas", "distances"],
            )
            if nursery_results and nursery_results["ids"] and nursery_results["ids"][0]:
                for i, doc_id in enumerate(nursery_results["ids"][0]):
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        nursery_candidates.append({
                            "id": doc_id,
                            "content": nursery_results["documents"][0][i],
                            "metadata": nursery_results["metadatas"][0][i] if nursery_results["metadatas"] else {},
                            "pool": "nursery",
                        })
        except Exception:
            pass

        # === CE RERANK: all candidates together, pick top 4 ===
        all_candidates = cosine_candidates + kg_candidates + nursery_candidates

        if all_candidates and self._reranker:
            pairs = [[query, c["content"]] for c in all_candidates]
            ce_scores = self._reranker.predict(pairs)
            if isinstance(ce_scores, np.ndarray):
                ce_scores = ce_scores.tolist()
            for i, c in enumerate(all_candidates):
                c["ce_score"] = ce_scores[i]

            # Split into proven (cosine+kg) and nursery
            proven = [c for c in all_candidates if c["pool"] != "nursery"]
            nursery = [c for c in all_candidates if c["pool"] == "nursery"]

            proven.sort(key=lambda x: x.get("ce_score", 0), reverse=True)
            nursery.sort(key=lambda x: x.get("ce_score", 0), reverse=True)

            # 3 from proven/kg + 1 from nursery
            final = proven[:3] + nursery[:1]
            # Fallback: fill remaining slots
            remaining = top_k - len(final)
            if remaining > 0:
                extra = [c for c in proven[3:] + nursery[1:]]
                final.extend(extra[:remaining])
        else:
            final = all_candidates[:top_k]

        final = final[:top_k]

        # Track which edges were used (for scoring later)
        kg_retrieved_ids = {c["id"] for c in final if c.get("pool") == "kg"}

        memories = []
        for c in final:
            meta = c["metadata"]
            meta["_pool"] = c.get("pool", "cosine")  # Track source pool for scoring
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=meta,
                score=c.get("ce_score", 0.5),
                access_count=int(meta.get("uses", 0)),
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

        # Score memories (same as Wilson)
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

                    # Also score KG edges for this memory
                    self._triple_store.score_edges_for_memory(doc_id, outcome)
            except Exception:
                pass

        # Store new memory from exchange summary
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
        kg_stats = self._triple_store.get_stats() if self._triple_store else {}
        return {
            "memories": count,
            "strategy": self.name,
            "avg_wilson": round(avg_wilson, 3),
            **kg_stats,
        }

    async def cleanup(self) -> None:
        if self._llm_client:
            await self._llm_client.aclose()

    def _format_injection(self, memories: List[MemoryRecord]) -> str:
        if not memories:
            return ""
        parts = ["[Retrieved memories]"]
        for m in memories:
            parts.append(f"- {m.content}")
        return "\n".join(parts)
