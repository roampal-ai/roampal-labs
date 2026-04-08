"""Cross-encoder retrieval + outcome-based memory lifecycle.

Pure CE for retrieval. Outcome scoring drives promotion,
demotion, and decay — matching production roampal-core exactly.

Optional tag routing: scopes CE candidate pool by extracted noun tags.
Tags provide +6.1% Hit@1 on clean data and +7.5% under poison (p<0.0001).
"""

import asyncio
import json
import math
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import chromadb
import numpy as np

from .base import MemoryRecord, RetrievalResult


# === Tag extraction (from entity_routed.py) ===

async def extract_tags(text: str, llm_client, llm_model: str) -> List[str]:
    """Ask the LLM to extract noun tags from memory text."""
    prompt = (
        "Extract the key TOPIC nouns from this text — people's names, places, objects, "
        "and specific things the text is actually about. "
        "Return ONLY a JSON array of lowercase strings like: [\"calvin\", \"muscle car\", \"boston\"]\n"
        "Rules:\n"
        "- Use actual names, not pronouns (skip 'he', 'she', 'they', 'user', 'assistant')\n"
        "- Keep each tag as a short noun phrase (1-3 words)\n"
        "- Include both proper nouns and important common nouns\n"
        "- Skip meta-words: source, answer, details, accuracy, response, question, topic\n"
        "- Focus on WHO and WHAT the text is about\n"
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
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            tags = json.loads(content)
            if isinstance(tags, list):
                skip = {"he", "she", "they", "it", "user", "assistant", "i", "you", "we"}
                return [t.lower().strip() for t in tags
                        if isinstance(t, str) and t.lower().strip() not in skip and len(t.strip()) >= 2]
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass
    return []



def match_query_tags(query: str, known_tags: Set[str]) -> List[str]:
    """Find known tags mentioned in the query."""
    query_lower = query.lower()
    matches = []
    for tag in sorted(known_tags, key=len, reverse=True):
        if len(tag) < 3:
            continue
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower):
            matches.append(tag)
    return matches[:8]


# Production thresholds (from roampal-core config.py)
PROMO_WORKING_TO_HISTORY_SCORE = 0.7
PROMO_WORKING_TO_HISTORY_USES = 2
PROMO_HISTORY_TO_PATTERNS_SCORE = 0.9
# PROMO_HISTORY_TO_PATTERNS_USES = 3  # Redundant: success≥5 implies uses≥5
PROMO_HISTORY_TO_PATTERNS_SUCCESS = 5.0
DEMOTION_SCORE = 0.4
DELETION_SCORE = 0.2
NEW_ITEM_DELETION_SCORE = 0.1

TIER_NAMES = ["working", "history", "patterns"]


def wilson_lower_bound(successes: float, total: int, z: float = 1.96) -> float:
    """Wilson score lower bound — conservative estimate of true success rate."""
    if total == 0:
        return 0.5
    p = successes / total
    n = total
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    variance = p * (1 - p) / n + z * z / (4 * n * n)
    lower = (center - z * math.sqrt(max(0, variance))) / denominator
    return max(0.0, lower)


class CELifecycleStrategy:
    """Pure CE retrieval + production-matching memory lifecycle.

    Retrieval: cosine → CE rerank.
    Lifecycle: 3 tiers, promotion, demotion, decay/deletion.
    Outcome scoring: worked/failed/partial/unknown deltas.

    Matches production roampal-core promotion_service.py thresholds.
    """

    name = "ce_lifecycle"

    def __init__(
        self,
        persist_dir: str = "",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_pool: int = 40,
        enable_decay: bool = False,
        enable_tags: bool = False,
        enable_tag_cascade: bool = False,
        enable_wilson_blend: bool = False,
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "gpt-oss:20b",
    ):
        self._persist_dir = persist_dir
        self._reranker_model = reranker_model
        self._candidate_pool = candidate_pool
        self._enable_decay = enable_decay
        self._enable_tags = enable_tags
        self._enable_tag_cascade = enable_tag_cascade
        self._enable_wilson_blend = enable_wilson_blend
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._reranker = None
        self._llm_client = None
        self._known_tags: Set[str] = set()
        self._inverted_index: Dict[str, List] = {}
        # Decay tracking for paper metrics
        self.decay_log: List[Dict] = []

    @property
    def _collection(self):
        """Backward compat — return working tier."""
        return self._collections.get("working")

    async def initialize(self) -> None:
        from sentence_transformers import CrossEncoder

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()

        for tier in TIER_NAMES:
            self._collections[tier] = self._client.get_or_create_collection(
                name=tier,
                metadata={"hnsw:space": "cosine"},
            )

        # Archive collection for decayed memories (not searched, just preserved)
        self._decayed_collection = self._client.get_or_create_collection(
            name="decayed",
            metadata={"hnsw:space": "cosine"},
        )

        self._reranker = CrossEncoder(self._reranker_model, device="cuda")

        # LLM client for tag extraction (only if tags enabled)
        if self._enable_tags or self._enable_tag_cascade:
            import httpx
            self._llm_client = httpx.AsyncClient(base_url=self._llm_base_url, timeout=60)

        # Build inverted index for tag-cascade mode
        if self._enable_tag_cascade:
            self._rebuild_inverted_index()

    def _rebuild_inverted_index(self):
        """Build tag → list of (id, content, metadata) inverted index from all tiers."""
        from collections import defaultdict
        index = defaultdict(list)
        for tier in TIER_NAMES:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            offset = 0
            while True:
                batch = col.get(limit=1000, offset=offset, include=['documents', 'metadatas'])
                if not batch['ids']:
                    break
                for i, doc_id in enumerate(batch['ids']):
                    meta = batch['metadatas'][i]
                    content = batch['documents'][i]
                    meta['tier'] = tier
                    entry = {'id': doc_id, 'content': content, 'metadata': meta}
                    tags = meta.get('tags', '')
                    for t in tags.split('|'):
                        t = t.strip().lower()
                        if t:
                            index[t].append(entry)
                            self._known_tags.add(t)
                offset += 1000
                if len(batch['ids']) < 1000:
                    break
        self._inverted_index = dict(index)

    def _add_to_inverted_index(self, doc_id: str, content: str, metadata: dict):
        """Incrementally add a new memory to the inverted index."""
        tags = metadata.get('tags', '')
        entry = {'id': doc_id, 'content': content, 'metadata': metadata}
        for t in tags.split('|'):
            t = t.strip().lower()
            if t:
                if t not in self._inverted_index:
                    self._inverted_index[t] = []
                self._inverted_index[t].append(entry)
                self._known_tags.add(t)

    EVOLVING_SUMMARY_THRESHOLD = 0.4  # Cosine distance below this = update existing

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store memory in working tier with dedup (reject near-duplicates)."""
        # Dedup: reject if very similar memory already exists (cosine distance < 0.1)
        mem_type = (metadata or {}).get("type", "summary")
        if mem_type == "fact":
            where_clause = {"type": "fact"}
        else:
            where_clause = {"type": {"$ne": "fact"}}

        for tier in TIER_NAMES:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            try:
                results = col.query(
                    query_texts=[content], n_results=1,
                    where=where_clause,
                    include=["distances"],
                )
                if (results and results["distances"] and results["distances"][0]
                        and results["distances"][0][0] < 0.1):
                    return ""  # Near-duplicate, skip
            except Exception:
                continue

        doc_id = f"working_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta.setdefault("score", 0.5)
        meta.setdefault("uses", 0)
        meta.setdefault("success_count", 0.0)
        meta.setdefault("outcome_history", "[]")
        meta["stored_at"] = time.time()
        meta["tier"] = "working"

        # Extract tags if enabled
        if self._enable_tags and self._llm_client:
            tags = await extract_tags(content, self._llm_client, self._llm_model)
            if tags:
                meta["tags"] = "|".join(tags)
                self._known_tags.update(tags)

        self._collections["working"].add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )

        # Incrementally update inverted index for tag-cascade mode
        if self._enable_tag_cascade:
            self._add_to_inverted_index(doc_id, content, meta)

        return doc_id

    async def store_or_update_summary(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new summary or update an existing one if topic matches.

        Evolving summaries: if a similar summary exists (cosine distance < 0.4),
        update it with merged content instead of creating a new one. Preserves
        doc_id + outcome scores + tier through updates.

        Returns doc_id of new or updated summary.
        """
        meta = metadata or {}
        meta.setdefault("type", "summary")

        # Search all tiers for a similar existing memory of the same type
        mem_type = meta.get("type", "summary")
        if mem_type == "fact":
            where_clause = {"type": "fact"}
        else:
            where_clause = {"type": {"$ne": "fact"}}

        best_match = None
        best_dist = 1.0
        best_tier = None

        for tier in TIER_NAMES:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            try:
                results = col.query(
                    query_texts=[content],
                    n_results=1,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"],
                )
                if (results and results["ids"] and results["ids"][0]
                        and results["distances"][0][0] < best_dist):
                    best_dist = results["distances"][0][0]
                    best_match = {
                        "id": results["ids"][0][0],
                        "content": results["documents"][0][0],
                        "metadata": results["metadatas"][0][0],
                    }
                    best_tier = tier
            except Exception:
                continue

        if best_match and best_dist < self.EVOLVING_SUMMARY_THRESHOLD:
            # UPDATE existing — new content, keep scores
            existing_meta = best_match["metadata"]
            existing_meta["updated_at"] = time.time()
            existing_meta["update_count"] = int(existing_meta.get("update_count", 0)) + 1

            # Re-extract tags if enabled (content changed, tags may be stale)
            if self._enable_tags and self._llm_client:
                try:
                    new_tags = await extract_tags(content, self._llm_client, self._llm_model)
                    if new_tags:
                        existing_meta["tags"] = "|".join(new_tags)
                        self._known_tags.update(new_tags)
                except Exception:
                    pass  # Keep old tags if extraction fails

            col = self._collections[best_tier]
            col.update(
                ids=[best_match["id"]],
                documents=[content],
                metadatas=[existing_meta],
            )
            return best_match["id"]
        else:
            # CREATE new summary
            return await self.store(content, metadata=meta)

    def _query_all_tiers(self, query: str, n_results: int,
                         type_filter: str = None, type_exclude: str = None) -> List[dict]:
        """Query across all tiers, merge by distance."""
        all_candidates = []
        for tier in TIER_NAMES:
            col = self._collections.get(tier)
            if not col or col.count() == 0:
                continue
            try:
                kwargs = {
                    "query_texts": [query],
                    "n_results": min(n_results, col.count()),
                    "include": ["documents", "metadatas", "distances"],
                }
                # ChromaDB where-clause for type filtering (avoids pool starvation)
                where_clause = {}
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
                            "distance": results["distances"][0][i] if results["distances"] else 1.0,
                        })
            except Exception:
                pass
        all_candidates.sort(key=lambda x: x["distance"])
        return all_candidates

    async def _retrieve_tag_cascade(self, query: str, top_k: int,
                                     type_filter: str, type_exclude: str,
                                     t0: float) -> RetrievalResult:
        """Tags-first cascade: inverted index entry → overlap tiers → cosine fill → CE rerank."""
        query_tags = match_query_tags(query, self._known_tags) if self._known_tags else []

        POOL = self._candidate_pool  # 40

        if query_tags:
            # Get ALL memories matching ANY query tag via inverted index
            candidate_map = {}
            for qt in query_tags:
                qt_lower = qt.lower()
                for entry in self._inverted_index.get(qt_lower, []):
                    # Type filter
                    mem_type = entry['metadata'].get('type', '')
                    if type_filter and mem_type != type_filter:
                        continue
                    if type_exclude and mem_type == type_exclude:
                        continue
                    doc_id = entry['id']
                    if doc_id not in candidate_map:
                        candidate_map[doc_id] = {
                            'id': doc_id,
                            'content': entry['content'],
                            'metadata': entry['metadata'],
                            'overlap': 0,
                        }
                    candidate_map[doc_id]['overlap'] += 1

            if candidate_map:
                candidates = list(candidate_map.values())
                max_overlap = max(c['overlap'] for c in candidates)

                # Get cosine distances for tiebreaker
                cosine_results = self._query_all_tiers(query, 60, type_filter, type_exclude)
                dist_map = {c['id']: c['distance'] for c in cosine_results}
                for c in candidates:
                    c['distance'] = dist_map.get(c['id'], 1.0)

                # Fill pool: highest overlap first, cosine tiebreaker within each tier
                pool = []
                seen = set()
                for tier in range(max_overlap, 0, -1):
                    tier_cands = [c for c in candidates if c['overlap'] == tier and c['id'] not in seen]
                    tier_cands.sort(key=lambda c: c['distance'])
                    for c in tier_cands:
                        pool.append(c)
                        seen.add(c['id'])
                        if len(pool) >= POOL:
                            break
                    if len(pool) >= POOL:
                        break

                # Fill remaining with cosine if tags didn't fill pool
                if len(pool) < POOL:
                    for c in cosine_results:
                        if c['id'] not in seen:
                            pool.append(c)
                            seen.add(c['id'])
                            if len(pool) >= POOL:
                                break

                # CE rerank (with optional Wilson blend)
                if pool:
                    pairs = [[query, c['content']] for c in pool]
                    ces = self._reranker.predict(pairs)
                    if isinstance(ces, np.ndarray):
                        ces = ces.tolist()
                    for i, c in enumerate(pool):
                        c['ce_score'] = ces[i]

                    if self._enable_wilson_blend:
                        # Normalize CE scores to 0-1
                        ce_min = min(c['ce_score'] for c in pool)
                        ce_max = max(c['ce_score'] for c in pool)
                        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                        for c in pool:
                            ce_norm = (c['ce_score'] - ce_min) / ce_range
                            meta = c.get('metadata', {})
                            wilson = float(meta.get('wilson_lower', 0.5))
                            uses = int(meta.get('uses', 0))
                            # Only blend Wilson if memory has enough data
                            if uses >= 2:
                                c['combined_score'] = 0.7 * ce_norm + 0.3 * wilson
                            else:
                                c['combined_score'] = ce_norm
                        pool.sort(key=lambda x: x['combined_score'], reverse=True)
                    else:
                        pool.sort(key=lambda x: x['ce_score'], reverse=True)

                top_candidates = pool[:top_k]
            else:
                top_candidates = []
        else:
            top_candidates = []

        # Cosine fallback if no tags matched or no candidates
        if not top_candidates:
            candidates = self._query_all_tiers(query, POOL, type_filter, type_exclude)
            if candidates:
                pairs = [[query, c['content']] for c in candidates]
                ces = self._reranker.predict(pairs)
                if isinstance(ces, np.ndarray):
                    ces = ces.tolist()
                for i, c in enumerate(candidates):
                    c['ce_score'] = ces[i]

                if self._enable_wilson_blend:
                    ce_min = min(c['ce_score'] for c in candidates)
                    ce_max = max(c['ce_score'] for c in candidates)
                    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                    for c in candidates:
                        ce_norm = (c['ce_score'] - ce_min) / ce_range
                        meta = c.get('metadata', {})
                        wilson = float(meta.get('wilson_lower', 0.5))
                        uses = int(meta.get('uses', 0))
                        if uses >= 2:
                            c['combined_score'] = 0.7 * ce_norm + 0.3 * wilson
                        else:
                            c['combined_score'] = ce_norm
                    candidates.sort(key=lambda x: x['combined_score'], reverse=True)
                else:
                    candidates.sort(key=lambda x: x['ce_score'], reverse=True)
                top_candidates = candidates[:top_k]

        memories = []
        for c in top_candidates:
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                score=c.get("ce_score", 0.5),
                access_count=int(c["metadata"].get("uses", 0)),
                collection=c["metadata"].get("tier", "working"),
            ))

        formatted = self._format_injection(memories)
        elapsed_ms = (time.time() - t0) * 1000

        return RetrievalResult(
            memories=memories,
            formatted_injection=formatted,
            query_used=query,
            retrieval_ms=elapsed_ms,
        )

    async def retrieve(self, query: str, top_k: int = 4,
                       type_filter: str = None, type_exclude: str = None) -> RetrievalResult:
        t0 = time.time()

        total = sum(c.count() for c in self._collections.values())
        if total == 0:
            return RetrievalResult(memories=[], formatted_injection="", query_used=query)

        # === TAGS-FIRST CASCADE (inverted index entry point) ===
        if self._enable_tag_cascade:
            return await self._retrieve_tag_cascade(query, top_k, type_filter, type_exclude, t0)

        # Query all tiers
        candidates = self._query_all_tiers(
            query, self._candidate_pool,
            type_filter=type_filter, type_exclude=type_exclude,
        )

        # === TAG CASCADE (matches tested EntityRouted implementation) ===
        if self._enable_tags and self._known_tags:
            query_tags = match_query_tags(query, self._known_tags)
        else:
            query_tags = []

        if query_tags:
            # Step 1: Count tag overlaps on cosine candidates
            tag_candidates = {}
            for c in candidates:
                mem_tags = c["metadata"].get("tags", "")
                mem_tag_set = set(t.strip().lower() for t in mem_tags.split("|") if t.strip())
                overlap = sum(1 for qt in query_tags if qt.lower() in mem_tag_set)
                if overlap > 0:
                    c["overlap"] = overlap
                    tag_candidates[c["id"]] = c

            # Step 2: Sort tagged by overlap desc, then cosine dist
            sorted_overlap = sorted(
                tag_candidates.values(),
                key=lambda x: (-x["overlap"], x["distance"]),
            )

            # Step 3: Cascade — up to 3 CE batches with quality gate
            POOL = self._candidate_pool
            accepted_pool = []
            seen_ids = set()
            cursor = 0

            for batch_num in range(1, 4):
                batch = []
                used_cosine_fill = False

                # Fill from tagged pool
                while len(batch) < POOL and cursor < len(sorted_overlap):
                    c = sorted_overlap[cursor]
                    cursor += 1
                    if c["id"] not in seen_ids:
                        batch.append(c)
                        seen_ids.add(c["id"])

                # Top off with cosine if not enough tagged
                if len(batch) < POOL:
                    for c in candidates:
                        if c["id"] not in seen_ids:
                            batch.append(c)
                            seen_ids.add(c["id"])
                            used_cosine_fill = True
                            if len(batch) >= POOL:
                                break

                if not batch:
                    break

                # CE score this batch
                pairs = [[query, c["content"]] for c in batch]
                ces = self._reranker.predict(pairs)
                if isinstance(ces, np.ndarray):
                    ces = ces.tolist()
                for i, c in enumerate(batch):
                    c["ce_score"] = ces[i]
                batch.sort(key=lambda x: x["ce_score"], reverse=True)

                # Quality gate: top CE score >= 0 = accept
                if batch[0].get("ce_score", -1) >= 0.0 or batch_num == 3:
                    accepted_pool = batch
                    break
                if used_cosine_fill:
                    accepted_pool = batch
                    break

            # Fallback: pure cosine → CE if cascade found nothing
            if not accepted_pool and candidates:
                pairs = [[query, c["content"]] for c in candidates[:POOL]]
                ces = self._reranker.predict(pairs)
                if isinstance(ces, np.ndarray):
                    ces = ces.tolist()
                for i in range(len(candidates[:POOL])):
                    candidates[i]["ce_score"] = ces[i]
                candidates[:POOL] = sorted(candidates[:POOL], key=lambda x: x["ce_score"], reverse=True)
                accepted_pool = candidates[:POOL]

            top_candidates = accepted_pool[:top_k]

        else:
            # No tags — pure CE rerank
            if self._reranker and len(candidates) > 0:
                pairs = [[query, c["content"]] for c in candidates]
                ce_scores = self._reranker.predict(pairs)
                if isinstance(ce_scores, np.ndarray):
                    ce_scores = ce_scores.tolist()
                for i, c in enumerate(candidates):
                    c["ce_score"] = ce_scores[i]
                candidates.sort(key=lambda x: x["ce_score"], reverse=True)

            top_candidates = candidates[:top_k]

        memories = []
        for c in top_candidates:
            memories.append(MemoryRecord(
                id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                score=c.get("ce_score", 0.5),
                access_count=int(c["metadata"].get("uses", 0)),
                collection=c["metadata"].get("tier", "working"),
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
        for tier in TIER_NAMES:
            col = self._collections.get(tier)
            if not col:
                continue
            try:
                result = col.get(ids=[doc_id])
                if result and result["ids"]:
                    return tier
            except Exception:
                pass
        return None

    def _promote_if_eligible(self, doc_id: str, meta: dict, current_tier: str):
        """Check promotion/demotion thresholds and move memory if eligible.
        Matches production roampal-core promotion_service.py."""
        score = float(meta.get("score", 0.5))
        uses = int(meta.get("uses", 0))
        success = float(meta.get("success_count", 0.0))

        target_tier = None

        # Promotion: working -> history
        if current_tier == "working" and score >= PROMO_WORKING_TO_HISTORY_SCORE and uses >= PROMO_WORKING_TO_HISTORY_USES:
            target_tier = "history"

        # Promotion: history -> patterns
        elif current_tier == "history" and score >= PROMO_HISTORY_TO_PATTERNS_SCORE and success >= PROMO_HISTORY_TO_PATTERNS_SUCCESS:
            target_tier = "patterns"

        # Demotion: patterns -> history
        elif current_tier == "patterns" and score < DEMOTION_SCORE:
            target_tier = "history"

        if target_tier:
            try:
                result = self._collections[current_tier].get(
                    ids=[doc_id], include=["documents", "metadatas", "embeddings"]
                )
                if result and result["ids"]:
                    doc = result["documents"][0]
                    embeds = result.get("embeddings")
                    embed = embeds[0] if embeds is not None and len(embeds) > 0 else None
                    meta["tier"] = target_tier

                    # ID rename: working_abc -> history_abc (matches production)
                    new_id = f"{target_tier}_{doc_id.split('_', 1)[1]}" if "_" in doc_id else f"{target_tier}_{doc_id}"
                    meta["original_id"] = doc_id

                    # Reset success_count on promotion — memory must re-prove itself
                    # in the new tier (required for history→patterns at success≥5)
                    meta["success_count"] = 0.0

                    add_kwargs = {"ids": [new_id], "documents": [doc], "metadatas": [meta]}
                    if embed is not None:
                        add_kwargs["embeddings"] = [embed]
                    self._collections[target_tier].add(**add_kwargs)
                    self._collections[current_tier].delete(ids=[doc_id])
            except Exception:
                pass

    async def record_outcome(
        self, memory_ids: List[str], outcome: str, exchange_summary: str = ""
    ) -> None:
        """Score memories and check lifecycle transitions.
        Matches production roampal-core outcome_service.py."""
        # Score deltas (production-matching)
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

                    # Decay: archive if score below threshold (move to decayed collection)
                    # Production uses two-tier: 0.2 for items >7d old, 0.1 for newer
                    stored_at = float(meta.get("stored_at", 0))
                    age_days = (time.time() - stored_at) / 86400 if stored_at else 0
                    decay_threshold = DELETION_SCORE if age_days > 7 else NEW_ITEM_DELETION_SCORE
                    if self._enable_decay and new_score < decay_threshold:
                        # Archive the memory instead of hard deleting
                        try:
                            full = col.get(ids=[doc_id], include=["documents", "embeddings"])
                            if full and full["ids"]:
                                archive_meta = dict(meta)
                                archive_meta["decayed_from"] = tier
                                archive_meta["decayed_at"] = time.time()
                                add_kwargs = {
                                    "ids": [f"decayed_{doc_id}"],
                                    "documents": full["documents"],
                                    "metadatas": [archive_meta],
                                }
                                full_embeds = full.get("embeddings")
                                if full_embeds is not None and len(full_embeds) > 0:
                                    add_kwargs["embeddings"] = full_embeds
                                self._decayed_collection.add(**add_kwargs)
                        except Exception:
                            pass
                        col.delete(ids=[doc_id])
                        self.decay_log.append({
                            "doc_id": doc_id,
                            "tier": tier,
                            "final_score": new_score,
                            "uses": uses,
                            "content_preview": existing.get("documents", [""])[0][:100] if existing.get("documents") else "",
                            "timestamp": time.time(),
                        })
                    else:
                        # Check promotion/demotion
                        self._promote_if_eligible(doc_id, meta, tier)
                except Exception:
                    pass

        if exchange_summary:
            await self.store(exchange_summary)

    async def get_stats(self) -> Dict[str, Any]:
        counts = {tier: self._collections[tier].count()
                  for tier in TIER_NAMES if self._collections.get(tier)}
        total = sum(counts.values())
        decayed_in_db = self._decayed_collection.count() if self._decayed_collection else 0
        return {
            "total_memories": total,
            "by_tier": counts,
            "decayed_archived": decayed_in_db,
            "strategy": self.name,
            "reranker_model": self._reranker_model,
            "candidate_pool": self._candidate_pool,
            "decay_enabled": self._enable_decay,
            "decayed_count": len(self.decay_log),
        }

    async def cleanup(self) -> None:
        pass

    def _format_injection(self, memories: List[MemoryRecord]) -> str:
        """Format memories for LLM context — matches production roampal-core injection.
        Shows trust metadata so LLM can judge conflicting memories."""
        if not memories:
            return ""
        parts = ["═══ KNOWN CONTEXT ═══"]
        for m in memories:
            meta = m.metadata or {}
            score = float(meta.get("score", 0.5))
            uses = int(meta.get("uses", 0))
            success = float(meta.get("success_count", 0.0))
            tier = meta.get("tier", m.collection or "working")
            last = meta.get("last_outcome", "")
            stored_at = meta.get("stored_at", 0)


            # Age
            if stored_at:
                age_s = time.time() - float(stored_at)
                if age_s < 3600: age_str = f"{int(age_s/60)}m"
                elif age_s < 86400: age_str = f"{int(age_s/3600)}h"
                else: age_str = f"{int(age_s/86400)}d"
            else:
                age_str = ""

            # Outcome history [YYN] format (last 3)
            outcomes_str = ""
            try:
                history = json.loads(meta.get("outcome_history", "[]"))
                if history:
                    symbols = []
                    for entry in history[-3:]:
                        o = entry.get("outcome", "")
                        if o == "worked": symbols.append("Y")
                        elif o == "failed": symbols.append("N")
                        elif o == "partial": symbols.append("~")
                    if symbols: outcomes_str = "[" + "".join(symbols) + "]"
            except Exception:
                pass

            # Build metadata string matching production exactly
            mem_type = meta.get("type", "summary")
            meta_parts = []
            meta_parts.append(mem_type)
            if age_str: meta_parts.append(age_str)
            meta_parts.append(tier)
            meta_parts.append(f"s:{score:.1f}")
            wilson_lower = meta.get("wilson_lower")
            if wilson_lower is not None:
                meta_parts.append(f"w:{float(wilson_lower):.0%}")
            if uses: meta_parts.append(f"{uses} uses")
            if outcomes_str: meta_parts.append(outcomes_str)

            meta_str = ", ".join(meta_parts)
            parts.append(f"• {m.content} [id:{m.id}] ({meta_str})")
        parts.append("═══ END CONTEXT ═══")
        return "\n".join(parts)
