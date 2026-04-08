"""Tag-routed retrieval: noun tags + Wilson+CE search.

Pipeline:
1. Store: LLM extracts noun tags from memory → stored in ChromaDB metadata
2. Retrieve:
   a. Extract nouns from query → match to known tags
   b. Multi-tag match? → ChromaDB where-filter scopes cosine search to tagged subset
   c. Single tag? → scope to that tag's memories
   d. No tags match? → full cosine search (Wilson+CE fallback)
   e. CE reranks candidates, Wilson blends for proven memories
   f. 1 nursery slot for low-use memories
3. Score: Wilson on memories — same as Wilson+CE. Tags don't need scores.

Tags route to the right neighborhood. Wilson+CE searches within it.
"""

import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


# === Tag extraction — LLM-based ===

async def extract_tags(text: str, llm_client, llm_model: str) -> List[str]:
    """Ask the LLM to extract noun tags from memory text.

    Returns lowercase noun tags — people, places, objects, specific things.
    """
    prompt = (
        "Extract the key TOPIC nouns from this text — people's names, places, objects, "
        "and specific things the text is actually about. "
        "Return ONLY a JSON array of lowercase strings like: [\"calvin\", \"muscle car\", \"boston\"]\n"
        "Rules:\n"
        "- Use actual names, not pronouns (skip 'he', 'she', 'they', 'user', 'assistant')\n"
        "- Keep each tag as a short noun phrase (1-3 words)\n"
        "- Include both proper nouns and important common nouns\n"
        "- Skip meta-words about the conversation itself: source, answer, details, accuracy, "
        "response, question, topic, context, information, correction, update, memory\n"
        "- Skip generic verbs/actions: said, told, mentioned, discussed, talked, asked\n"
        "- Focus on WHO and WHAT the text is about, not how it was communicated\n"
        f"Text: {text[:500]}"
    )
    try:
        resp = await asyncio.wait_for(
            llm_client.post("/chat/completions", json={
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }),
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            tags = json.loads(content)
            if isinstance(tags, list):
                skip = {"he", "she", "they", "it", "user", "assistant", "the user",
                        "the assistant", "i", "you", "we", "them", "his", "her"}
                return [t.lower().strip() for t in tags
                        if isinstance(t, str) and t.lower().strip() not in skip and len(t.strip()) >= 2]
    except asyncio.TimeoutError:
        print(f"    WARN: extract_tags timeout (30s): {text[:60]}...", flush=True)
    except Exception as e:
        print(f"    WARN: extract_tags failed: {type(e).__name__}", flush=True)
    return []


def match_query_tags(query: str, known_tags: Set[str]) -> List[str]:
    """Find known tags mentioned in the query using word boundaries."""
    import re
    query_lower = query.lower()
    matches = []
    sorted_tags = sorted(known_tags, key=len, reverse=True)
    for tag in sorted_tags:
        if len(tag) < 3:
            continue
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower):
            matches.append(tag)
    return matches[:8]


# === Strategy ===

class EntityRoutedStrategy:
    """Tag-routed Wilson+CE retrieval.

    Store: ChromaDB + LLM tag extraction → tags in metadata
    Retrieve: tag filter scopes cosine search → CE + Wilson blend picks final 4
    Score: Wilson on memories, same as Wilson+CE
    """

    name = "entity_routed"

    # Promotion thresholds (matching production)
    PROMO_WORKING_TO_HISTORY_SCORE = 0.7
    PROMO_WORKING_TO_HISTORY_USES = 2
    PROMO_HISTORY_TO_PATTERNS_SCORE = 0.9
    PROMO_HISTORY_TO_PATTERNS_USES = 3
    PROMO_HISTORY_TO_PATTERNS_SUCCESS = 5.0

    TIER_NAMES = ["working", "history", "patterns"]

    def __init__(
        self,
        persist_dir: str = "",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "gpt-oss:20b",
    ):
        self._persist_dir = persist_dir
        self._reranker_model = reranker_model
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._client = None
        self._collections: Dict[str, Any] = {}  # tier_name -> ChromaDB collection
        self._reranker = None
        self._llm_client = None
        self._known_tags: Set[str] = set()

    async def initialize(self) -> None:
        import httpx
        from sentence_transformers import CrossEncoder

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()

        # Create 3 tiered collections
        for tier in self.TIER_NAMES:
            self._collections[tier] = self._client.get_or_create_collection(
                name=tier,
                metadata={"hnsw:space": "cosine"},
            )

        self._reranker = CrossEncoder(self._reranker_model, device="cuda")
        self._llm_client = httpx.AsyncClient(
            base_url=self._llm_base_url,
            headers={"Authorization": "Bearer ollama"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._rebuild_known_tags()

    @property
    def _collection(self):
        """Default to working for backward compat."""
        return self._collections.get("working")

    def _rebuild_known_tags(self):
        """Scan all tiers to build the set of known tags."""
        self._known_tags = set()
        for tier in self.TIER_NAMES:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            try:
                all_meta = col.get(include=["metadatas"])
                for m in all_meta["metadatas"]:
                    tags_str = m.get("tags", "")
                    if tags_str:
                        for tag in tags_str.split("|"):
                            tag = tag.strip()
                            if tag:
                                self._known_tags.add(tag)
            except Exception:
                pass

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = f"mem_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta.setdefault("stored_at", time.time())
        meta.setdefault("score", 0.5)
        meta.setdefault("uses", 0)
        meta.setdefault("success_count", 0.0)
        meta.setdefault("outcome_history", "[]")
        meta["tier"] = "working"

        # Dedup: check all tiers for near-duplicate (cosine distance < 0.1 = similarity > 0.9)
        if meta.get("type") == "fact":
            for tier in self.TIER_NAMES:
                col = self._collections.get(tier)
                if not col or col.count() == 0:
                    continue
                try:
                    results = col.query(
                        query_texts=[content],
                        n_results=1,
                        include=["distances"],
                    )
                    if (results and results["distances"] and results["distances"][0]
                            and results["distances"][0][0] < 0.1):
                        return ""  # Skip — near-duplicate already exists
                except Exception:
                    pass

        # Extract noun tags
        tags = await extract_tags(content, self._llm_client, self._llm_model)
        meta["tags"] = "|".join(tags[:10])

        # Always store to working tier
        self._collections["working"].add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )

        for tag in tags[:10]:
            self._known_tags.add(tag)

        return doc_id

    def _ce_score_pool(self, query: str, candidates: List[dict]) -> List[dict]:
        """Score candidates with CE + Wilson blend. Returns sorted list."""
        if not candidates or not self._reranker:
            return candidates

        pairs = [[query, c["content"]] for c in candidates]
        ce_scores = self._reranker.predict(pairs)
        if isinstance(ce_scores, np.ndarray):
            ce_scores = ce_scores.tolist()

        ce_min = min(ce_scores)
        ce_max = max(ce_scores)
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        for i, c in enumerate(candidates):
            c["ce_score"] = ce_scores[i]
            ce_norm = (ce_scores[i] - ce_min) / ce_range

            raw_score = float(c["metadata"].get("score", 0.5))
            uses = int(c["metadata"].get("uses", 0))
            success_count = float(c["metadata"].get("success_count", 0.0))
            learned_score, _ = calculate_learned_score(raw_score, uses, success_count)

            if uses >= 3:
                c["combined_score"] = 0.6 * ce_norm + 0.4 * learned_score
            else:
                c["combined_score"] = 0.9 * ce_norm + 0.1 * learned_score

        candidates.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return candidates

    def _wilson_blend_score(self, candidate: dict) -> float:
        """Compute Wilson+CE blend score for sorting within overlap tiers."""
        meta = candidate.get("metadata", {})
        raw_score = float(meta.get("score", 0.5))
        uses = int(meta.get("uses", 0))
        success_count = float(meta.get("success_count", 0.0))
        learned, _ = calculate_learned_score(raw_score, uses, success_count)
        # Blend: cosine similarity (from distance) + Wilson learned score
        dist = candidate.get("distance", 1.0)
        cosine_sim = 1.0 / (1.0 + dist)
        if uses >= 3:
            return 0.6 * cosine_sim + 0.4 * learned
        else:
            return 0.9 * cosine_sim + 0.1 * learned

    @staticmethod
    def _passes_type_filter(metadata: dict, type_filter: str = None, type_exclude: str = None) -> bool:
        """Check if a memory passes the type filter."""
        mem_type = metadata.get("type", "")
        if type_filter and mem_type != type_filter:
            return False
        if type_exclude and mem_type == type_exclude:
            return False
        return True

    def _query_all_tiers(self, query: str, n_results: int, where: dict = None,
                         type_filter: str = None, type_exclude: str = None,
                         tiers: List[str] = None) -> List[dict]:
        """Query across tiers, merge results by distance. Optionally filter by type/tier."""
        all_candidates = []
        search_tiers = tiers or self.TIER_NAMES
        for tier in search_tiers:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            try:
                kwargs = {
                    "query_texts": [query],
                    "n_results": min(n_results, col.count()),
                    "include": ["documents", "metadatas", "distances"],
                }
                # Build where clause: combine explicit where with type filter
                where_clause = dict(where) if where else {}
                if type_filter:
                    where_clause["type"] = type_filter
                elif type_exclude:
                    where_clause["type"] = {"$ne": type_exclude}
                if where_clause:
                    kwargs["where"] = where_clause
                results = col.query(**kwargs)
                if results and results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][i] if results["metadatas"] else {}
                        meta["tier"] = tier
                        all_candidates.append({
                            "id": doc_id,
                            "content": results["documents"][0][i],
                            "metadata": meta,
                            "pool": "cosine",
                            "overlap": 0,
                            "distance": results["distances"][0][i] if results["distances"] else 1.0,
                        })
            except Exception:
                pass
        # Sort by distance (lower = more similar)
        all_candidates.sort(key=lambda x: x["distance"])
        return all_candidates

    def _fetch_cosine_pool(self, query: str, count: int, seen_ids: set, pool_size: int,
                           type_filter: str = None, type_exclude: str = None) -> List[dict]:
        """Fetch pure cosine candidates from ALL tiers, excluding seen IDs."""
        all_candidates = self._query_all_tiers(
            query, pool_size, type_filter=type_filter, type_exclude=type_exclude
        )
        cosine_pool = []
        for c in all_candidates:
            if c["id"] in seen_ids:
                continue
            cosine_pool.append(c)
            if len(cosine_pool) >= pool_size:
                break
        return cosine_pool

    async def retrieve(self, query: str, top_k: int = 4, type_filter: str = None, type_exclude: str = None) -> RetrievalResult:
        """Retrieve memories. Optionally filter by type metadata.

        Args:
            type_filter: Only return memories with this type (e.g. "fact")
            type_exclude: Exclude memories with this type (e.g. "fact")
        """
        t0 = time.time()

        count = self._collection.count()
        if count == 0:
            return RetrievalResult(memories=[], formatted_injection="", query_used=query)

        proven_slots = top_k - 1  # Reserve 1 for nursery
        query_tags = match_query_tags(query, self._known_tags)
        POOL_SIZE = 20

        # === TAG-ROUTED RETRIEVAL WITH CASCADE ===
        # 1. Run ONE query per matched tag (max 8), count overlaps
        # 2. Sort by (-overlap, -wilson_blend) — most surgical + most proven first
        # 3. Fill batches of 20, topping off with Wilson+CE cosine if overlap runs out
        # 4. CE quality gate per batch: top score > 0 = accept, < 0 = reject + continue
        # 5. If batch already included cosine fills, skip to forced accept (nothing new to try)
        # 6. Max 3 CE passes. Batch 3 = forced accept.

        # --- Step 1: Tag queries + overlap counting ---
        tag_candidates = {}

        if query_tags:
            # Get broad pool from ALL tiers, filter by tags in Python
            broad_results = self._query_all_tiers(
                query, POOL_SIZE, type_filter=type_filter, type_exclude=type_exclude
            )
            for c in broad_results:
                mem_tags = c["metadata"].get("tags", "")
                mem_tag_set = set(t.strip().lower() for t in mem_tags.split("|") if t.strip())
                overlap = sum(1 for qt in query_tags if qt.lower() in mem_tag_set)
                if overlap > 0:
                    c["pool"] = "tagged"
                    c["overlap"] = overlap
                    tag_candidates[c["id"]] = c

        # --- Step 2: Sort by overlap desc, Wilson blend desc ---
        sorted_overlap = sorted(
            tag_candidates.values(),
            key=lambda x: (-x["overlap"], -self._wilson_blend_score(x)),
        )

        # --- Step 3: Cascade batches ---
        accepted_pool = []
        seen_ids = set()
        overlap_cursor = 0  # How far into sorted_overlap we've consumed

        for batch_num in range(1, 4):  # Max 3 batches
            batch = []
            used_cosine_fill = False

            # Fill from overlap pool first
            while len(batch) < POOL_SIZE and overlap_cursor < len(sorted_overlap):
                c = sorted_overlap[overlap_cursor]
                overlap_cursor += 1
                if c["id"] not in seen_ids:
                    batch.append(c)
                    seen_ids.add(c["id"])

            # Top off with Wilson+CE cosine if overlap didn't fill 20
            if len(batch) < POOL_SIZE:
                cosine_fills = self._fetch_cosine_pool(query, count, seen_ids, POOL_SIZE - len(batch), type_filter, type_exclude)
                if cosine_fills:
                    used_cosine_fill = True
                    for c in cosine_fills:
                        batch.append(c)
                        seen_ids.add(c["id"])
                        if len(batch) >= POOL_SIZE:
                            break

            if not batch:
                break

            # CE score the batch
            batch = self._ce_score_pool(query, batch)

            # Quality gate: top CE score > 0 = accept
            if batch[0].get("ce_score", -1) >= 0.0 or batch_num == 3:
                accepted_pool = batch
                break

            # Rejected — but if we already used cosine fills, nothing new to try
            if used_cosine_fill:
                accepted_pool = batch  # Forced accept — cosine was already in the mix
                break

            # Otherwise continue cascade with next batch

        # If we never accepted anything (no candidates at all), try pure cosine
        if not accepted_pool:
            cosine_pool = self._fetch_cosine_pool(query, count, seen_ids, POOL_SIZE, type_filter, type_exclude)
            if cosine_pool:
                accepted_pool = self._ce_score_pool(query, cosine_pool)

        # === PICK TOP 3 FROM ACCEPTED POOL ===
        proven_final = accepted_pool[:proven_slots] if accepted_pool else []
        final_ids = {c["id"] for c in proven_final}

        # === NURSERY — 1 slot for low-use memory ===
        nursery_candidate = None
        try:
            # Nursery searches WORKING tier only (matching production)
            working_col = self._collections.get("working")
            if working_col and working_col.count() > 0:
                nursery_where = {"uses": {"$lt": 3}}
                if type_filter:
                    nursery_where = {"$and": [nursery_where, {"type": type_filter}]}
                elif type_exclude:
                    nursery_where = {"$and": [nursery_where, {"type": {"$ne": type_exclude}}]}
                nursery_results = working_col.query(
                    query_texts=[query],
                    n_results=min(20, working_col.count()),
                    where=nursery_where,
                    include=["documents", "metadatas", "distances"],
                )
                if nursery_results and nursery_results["ids"] and nursery_results["ids"][0]:
                    for i, doc_id in enumerate(nursery_results["ids"][0]):
                        if doc_id in final_ids:
                            continue
                        meta = nursery_results["metadatas"][0][i] if nursery_results["metadatas"] else {}
                        meta["tier"] = "working"
                        nursery_candidate = {
                            "id": doc_id,
                            "content": nursery_results["documents"][0][i],
                            "metadata": meta,
                            "pool": "nursery",
                            "combined_score": 0.0,
                        }
                        break
        except Exception:
            pass

        # === COMBINE: 3 proven + 1 nursery ===
        final = proven_final[:]
        if nursery_candidate:
            final.append(nursery_candidate)
        # Fill remaining slots from accepted pool if needed
        if len(final) < top_k and accepted_pool:
            for extra in accepted_pool[proven_slots:]:
                if extra["id"] not in {c["id"] for c in final}:
                    final.append(extra)
                    if len(final) >= top_k:
                        break

        final = final[:top_k]

        memories = []
        for c in final:
            meta = c["metadata"]
            meta["_pool"] = c.get("pool", "cosine")
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=meta,
                score=c.get("combined_score", 0.5),
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

    def _find_memory_tier(self, doc_id: str) -> Optional[str]:
        """Find which tier a memory lives in."""
        for tier in self.TIER_NAMES:
            col = self._collections.get(tier)
            if col:
                try:
                    result = col.get(ids=[doc_id])
                    if result and result["ids"]:
                        return tier
                except Exception:
                    pass
        return None

    def _promote_if_eligible(self, doc_id: str, meta: dict, current_tier: str):
        """Check promotion thresholds and move memory up if eligible."""
        score = float(meta.get("score", 0.5))
        uses = int(meta.get("uses", 0))
        success = float(meta.get("success_count", 0.0))

        target_tier = None
        if current_tier == "working" and score >= self.PROMO_WORKING_TO_HISTORY_SCORE and uses >= self.PROMO_WORKING_TO_HISTORY_USES:
            target_tier = "history"
        elif current_tier == "history" and score >= self.PROMO_HISTORY_TO_PATTERNS_SCORE and uses >= self.PROMO_HISTORY_TO_PATTERNS_USES and success >= self.PROMO_HISTORY_TO_PATTERNS_SUCCESS:
            target_tier = "patterns"

        if target_tier:
            try:
                # Get full memory from current tier
                result = self._collections[current_tier].get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
                if result and result["ids"]:
                    doc = result["documents"][0]
                    embed = result["embeddings"][0] if result.get("embeddings") is not None and len(result["embeddings"]) > 0 else None
                    meta["tier"] = target_tier
                    # Add to new tier
                    add_kwargs = {"ids": [doc_id], "documents": [doc], "metadatas": [meta]}
                    if embed is not None:
                        add_kwargs["embeddings"] = [embed]
                    self._collections[target_tier].add(**add_kwargs)
                    # Remove from old tier
                    self._collections[current_tier].delete(ids=[doc_id])
            except Exception:
                pass

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

        if memory_ids:
            for doc_id in memory_ids:
                tier = self._find_memory_tier(doc_id)
                if not tier:
                    continue
                col = self._collections[tier]
                try:
                    existing = col.get(ids=[doc_id], include=["metadatas"])
                    if not existing or not existing["ids"]:
                        continue
                    meta = existing["metadatas"][0]

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

                    col.update(ids=[doc_id], metadatas=[meta])

                    # Check promotion
                    self._promote_if_eligible(doc_id, meta, tier)

                    # Decay: delete memories that score below threshold
                    # Matches production roampal-core (deletion_score_threshold=0.2)
                    # UNCOMMENT FOR POISON RUN:
                    # if new_score < 0.2:
                    #     col.delete(ids=[doc_id])
                except Exception:
                    pass

        if exchange_summary:
            await self.store(exchange_summary)

    async def get_stats(self) -> Dict[str, Any]:
        counts = {tier: self._collections[tier].count() for tier in self.TIER_NAMES if self._collections.get(tier)}
        count = sum(counts.values())
        avg_wilson = 0.5
        all_scores = []
        for tier in self.TIER_NAMES:
            col = self._collections.get(tier)
            if col and col.count() > 0:
                try:
                    all_meta = col.get(include=["metadatas"])
                    for m in all_meta["metadatas"]:
                        sc = float(m.get("success_count", 0))
                        uses = int(m.get("uses", 0))
                        all_scores.append(wilson_score_lower(sc, uses))
                except Exception:
                    pass
        avg_wilson = sum(all_scores) / len(all_scores) if all_scores else 0.5
        return {
            "memories": count,
            "by_tier": counts,
            "strategy": self.name,
            "avg_wilson": round(avg_wilson, 3),
            "known_tags": len(self._known_tags),
        }

    async def cleanup(self) -> None:
        if self._llm_client:
            await self._llm_client.aclose()

    def _format_injection(self, memories: List[MemoryRecord]) -> str:
        if not memories:
            return ""
        parts = ["[Retrieved memories]"]
        for m in memories:
            meta = m.metadata or {}
            wilson_pct = int(meta.get("score", 0.5) * 100)
            uses = int(meta.get("uses", 0))
            mem_type = meta.get("type", "")
            type_tag = f" ({mem_type})" if mem_type else ""
            parts.append(f"- {m.content} [wilson:{wilson_pct}%, used:{uses}x{type_tag}]")
        return "\n".join(parts)
