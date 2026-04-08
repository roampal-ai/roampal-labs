#!/usr/bin/env python
"""
Blend weight optimizer: test different Wilson/CE blend weights against
existing DBs to find what retrieves correct answers most reliably.

No LLM calls — pure embedding queries + scoring math.

Usage:
    python -m benchmark.blend_optimizer
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def wilson_lower(score, uses, z=1.96):
    if uses == 0:
        return 0.5
    p = max(0, min(1, score))
    n = uses
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (centre - spread) / denom


def calculate_learned_score(raw_score, uses, success_count):
    """Match production scoring_service.py exactly."""
    if uses == 0:
        return raw_score, 0.5

    success_rate = success_count / uses if uses > 0 else 0.5
    wilson = wilson_lower(success_rate, uses)

    # Blend raw score with Wilson based on usage
    if uses < 3:
        blend = uses / 3.0
        learned = (1 - blend) * raw_score + blend * wilson
    else:
        learned = wilson

    return learned, wilson


def get_dynamic_weights(uses, learned_score):
    """Match production scoring_service.py 5-tier weights exactly."""
    if uses >= 5 and learned_score >= 0.8:
        return (0.2, 0.8)   # PROVEN HIGH-VALUE
    elif uses >= 3 and learned_score >= 0.7:
        return (0.25, 0.75)  # ESTABLISHED
    elif uses >= 2 and learned_score >= 0.5:
        return (0.35, 0.65)  # EMERGING PATTERN
    elif uses >= 2:
        return (0.7, 0.3)   # FAILING PATTERN
    else:
        return (0.8, 0.2)   # NEW/UNKNOWN


def score_candidates(candidates, query_embedding, ce_model=None, query_text=None,
                     use_wilson=True, use_ce=True, ce_blend_configs=None):
    """Score candidates with different blend configs, return rankings per config."""
    results = {}

    for c in candidates:
        meta = c["metadata"]
        distance = c["distance"]

        # Cosine similarity
        emb_sim = 1.0 / (1.0 + distance)

        # Wilson scoring
        raw_score = float(meta.get("score", 0.5))
        uses = int(meta.get("uses", 0))
        success_count = float(meta.get("success_count", 0.0))
        learned_score, wilson = calculate_learned_score(raw_score, uses, success_count)

        # Production 5-tier dynamic weights
        emb_w, learn_w = get_dynamic_weights(uses, learned_score)

        # Base score: embedding + Wilson (production default)
        c["wilson_score"] = emb_w * emb_sim + learn_w * learned_score
        c["pure_cosine"] = emb_sim
        c["learned_score"] = learned_score
        c["emb_sim"] = emb_sim

    # Score with different CE blend configs if CE available
    if ce_model and query_text and use_ce:
        pairs = [[query_text, c["content"]] for c in candidates]
        ce_scores = ce_model.predict(pairs)
        if isinstance(ce_scores, np.ndarray):
            ce_scores = ce_scores.tolist()

        ce_min = min(ce_scores) if ce_scores else 0
        ce_max = max(ce_scores) if ce_scores else 1
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        for i, c in enumerate(candidates):
            c["ce_raw"] = ce_scores[i]
            c["ce_norm"] = (ce_scores[i] - ce_min) / ce_range

    return candidates


def check_hit(top_k_contents, ground_truth):
    """Check if ground truth info appears in any of the top-k retrieved memories."""
    if not ground_truth.strip():
        return False

    gt_lower = ground_truth.lower().strip()
    # Check for key phrases from ground truth in retrieved content
    gt_words = set(gt_lower.split())
    # Remove stop words
    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'no', 'so', 'if', 'that', 'this', 'it', 'its', 'they', 'them',
            'their', 'she', 'her', 'he', 'him', 'his', 'we', 'us', 'our', 'you',
            'your', 'my', 'me', 'i'}
    gt_keywords = gt_words - stop
    if len(gt_keywords) < 2:
        gt_keywords = gt_words

    for content in top_k_contents:
        content_lower = content.lower()
        # Check keyword overlap
        matches = sum(1 for kw in gt_keywords if kw in content_lower)
        if matches >= min(3, len(gt_keywords)):
            return True

    return False


def main():
    os.environ.setdefault("PYTHONUTF8", "1")

    # Load exam
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    exam = data["locomo_exam"]
    non_adv = [q for q in exam if q.get("category_name") != "adversarial" and q.get("ground_truth", "").strip()]
    print(f"Exam: {len(exam)} total, {len(non_adv)} non-adversarial with GTs")

    # Use Reranker DB (single collection, simplest)
    db_path = "archive/pre_fix_run/runs/03.Reranker"
    client = chromadb.PersistentClient(path=db_path)
    col = client.list_collections()[0]
    print(f"DB: {col.name}, {col.count()} memories")

    # Load CE model
    print("Loading cross-encoder...")
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

    # Test configs: (name, use_wilson, use_ce, ce_proven_blend, ce_new_blend)
    configs = [
        ("pure_cosine", False, False, 0, 0),
        ("pure_ce", False, True, 1.0, 1.0),
        ("prod_wilson_only", True, False, 0, 0),
        ("prod_ce+wilson_60_40", True, True, 0.6, 0.9),  # production default
        ("ce+wilson_70_30", True, True, 0.7, 0.9),
        ("ce+wilson_80_20", True, True, 0.8, 0.95),
        ("ce+wilson_50_50", True, True, 0.5, 0.8),
        ("ce_only_rank", False, True, 1.0, 1.0),
    ]

    results = {name: {"hits": 0, "total": 0} for name, *_ in configs}

    sample = non_adv[:500]  # first 500 for speed
    print(f"Testing {len(configs)} configs on {len(sample)} questions...")

    for qi, q in enumerate(sample):
        question = q.get("question", q.get("query", ""))
        gt = q.get("ground_truth", "")

        # Query DB for 20 candidates
        query_results = col.query(
            query_texts=[question],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )

        if not query_results["ids"][0]:
            continue

        candidates = []
        for i, doc_id in enumerate(query_results["ids"][0]):
            candidates.append({
                "id": doc_id,
                "content": query_results["documents"][0][i],
                "metadata": query_results["metadatas"][0][i],
                "distance": query_results["distances"][0][i],
            })

        # Score all candidates
        candidates = score_candidates(candidates, None, ce_model, question)

        # Test each config
        for name, use_wilson, use_ce, ce_proven, ce_new in configs:
            scored = []
            for c in candidates:
                if use_ce and use_wilson:
                    uses = int(c["metadata"].get("uses", 0))
                    ce_w = ce_proven if uses >= 3 else ce_new
                    wilson_w = 1.0 - ce_w
                    score = ce_w * c.get("ce_norm", 0) + wilson_w * c["learned_score"]
                elif use_ce:
                    score = c.get("ce_raw", 0)
                elif use_wilson:
                    score = c["wilson_score"]
                else:
                    score = c["pure_cosine"]
                scored.append((score, c["content"]))

            scored.sort(reverse=True)
            top_4 = [content for _, content in scored[:4]]

            hit = check_hit(top_4, gt)
            results[name]["total"] += 1
            if hit:
                results[name]["hits"] += 1

        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(sample)}...")
            for name, *_ in configs:
                r = results[name]
                rate = r["hits"] / r["total"] if r["total"] else 0
                print(f"    {name}: {rate:.1%} ({r['hits']}/{r['total']})")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(sample)} questions)")
    print(f"{'='*60}")
    for name, *_ in configs:
        r = results[name]
        rate = r["hits"] / r["total"] if r["total"] else 0
        print(f"  {name}: {rate:.1%} ({r['hits']}/{r['total']})")


if __name__ == "__main__":
    main()
