"""Wilson+CE blend sweep on poison TagCascade DB.

Samples N questions, retrieves 40 candidates each via tag cascade + CE,
caches scores, then sweeps blend weights to find optimal Wilson ratio.

Reports hit@4 (does the correct answer appear in top 4?) for each blend.
Uses McNemar's test for statistical significance vs pure CE.

Usage: python -m benchmark.blend_sweep [--sample 200] [--db runs/poison/02.TagCascade]
"""

import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from scipy.stats import chi2


# ─── Wilson ──────────────────────────────────────────────────────────────────

def wilson_lower_bound(successes: float, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.5
    p = successes / total
    n = total
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    variance = p * (1 - p) / n + z * z / (4 * n * n)
    lower = (center - z * math.sqrt(max(0, variance))) / denominator
    return max(0.0, lower)


def mcnemar_p(n01: int, n10: int) -> float:
    """McNemar's chi-squared test."""
    if n01 + n10 == 0:
        return 1.0
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return 1 - chi2.cdf(chi2_stat, df=1)


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_sweep(db_path: str, sample_size: int = 200):
    from strategies.ce_lifecycle import CELifecycleStrategy

    # Load exam data
    raw = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    locomo = raw["locomo_exam"]
    # Normalize keys: ground_truth -> answer
    for q in locomo:
        if "ground_truth" in q and "answer" not in q:
            q["answer"] = q["ground_truth"]
    print(f"Loaded {len(locomo)} LoCoMo questions")

    # Sample
    if sample_size < len(locomo):
        sample = random.sample(locomo, sample_size)
    else:
        sample = locomo
    print(f"Using {len(sample)} questions for sweep")

    # First, calculate Wilson scores on the DB if not already done
    print("\nCalculating Wilson scores on DB...")
    client = chromadb.PersistentClient(path=db_path)
    for tier_name in ["working", "history", "patterns"]:
        try:
            col = client.get_collection(tier_name)
        except Exception:
            continue
        count = col.count()
        if count == 0:
            continue
        all_data = col.get(include=["metadatas"])
        batch_ids = []
        batch_metas = []
        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            uses = int(meta.get("uses", 0))
            success_count = float(meta.get("success_count", 0.0))
            meta["wilson_lower"] = round(wilson_lower_bound(success_count, uses), 4)
            batch_ids.append(doc_id)
            batch_metas.append(meta)
            if len(batch_ids) >= 500:
                col.update(ids=batch_ids, metadatas=batch_metas)
                batch_ids = []
                batch_metas = []
        if batch_ids:
            col.update(ids=batch_ids, metadatas=batch_metas)
        print(f"  {tier_name}: {count} updated")
    del client  # Release DB lock

    # Initialize strategy (tag cascade, NO wilson blend — we'll blend manually)
    strategy = CELifecycleStrategy(
        persist_dir=db_path,
        enable_decay=False,
        enable_tags=True,
        enable_tag_cascade=True,
        enable_wilson_blend=False,  # We'll blend manually in the sweep
    )
    await strategy.initialize()

    stats = await strategy.get_stats()
    print(f"DB: {stats.get('total_memories', '?')} memories, {stats.get('known_tags', len(strategy._known_tags))} tags\n")

    # ── Retrieve + cache all candidate scores ──
    print("Retrieving candidates (tag cascade + CE)...")
    cached = []  # List of {question, answer, candidates: [{ce_score, wilson, content}]}
    t0 = time.time()

    for i, q in enumerate(sample):
        question = q["question"]
        answer = q["answer"]

        # Use the internal tag cascade to get pool of 40, with CE scores
        retrieval = await strategy.retrieve(query=question, top_k=40)

        candidates = []
        for m in retrieval.memories:
            meta = m.metadata or {}
            candidates.append({
                "content": m.content,
                "ce_score": m.score,  # This is the CE score from retrieval
                "wilson": float(meta.get("wilson_lower", 0.5)),
                "uses": int(meta.get("uses", 0)),
                "score": float(meta.get("score", 0.5)),
            })

        cached.append({
            "question": question,
            "answer": answer,
            "category": q.get("category", "unknown"),
            "candidates": candidates,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sample) - i - 1) / rate
            print(f"  {i+1}/{len(sample)} ({rate:.1f} q/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(sample)} questions in {elapsed:.0f}s ({len(sample)/elapsed:.1f} q/s)\n")

    # ── Sweep blends ──
    print("Sweeping blend weights...")
    print(f"{'Blend':>6} {'Hit@4':>6} {'Hit@1':>6} {'MRR@4':>7} {'vs CE p':>8}")
    print("-" * 45)

    blends = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    # For each question, determine if correct answer is in top 4
    # We match by checking if the ground truth appears as substring in candidate content
    def answer_in_candidate(answer_str: str, content: str) -> bool:
        """Fuzzy match: does the answer appear in the candidate?"""
        answer_lower = answer_str.lower().strip()
        content_lower = content.lower()
        # Check key parts of the answer
        parts = [p.strip() for p in answer_lower.split(",") if len(p.strip()) > 3]
        if not parts:
            parts = [answer_lower]
        # At least one key part must match
        for part in parts[:3]:  # Check first 3 parts
            if part in content_lower:
                return True
        return False

    pure_ce_hits = []  # For McNemar's baseline

    for blend in blends:
        hits_at_4 = 0
        hits_at_1 = 0
        mrr_sum = 0.0
        blend_hits = []

        for entry in cached:
            candidates = entry["candidates"]
            answer = entry["answer"]

            if not candidates:
                blend_hits.append(0)
                continue

            # Normalize CE scores
            ce_scores = [c["ce_score"] for c in candidates]
            ce_min = min(ce_scores)
            ce_max = max(ce_scores)
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

            # Score with blend
            scored = []
            for c in candidates:
                ce_norm = (c["ce_score"] - ce_min) / ce_range
                wilson = c["wilson"]
                uses = c["uses"]
                if uses >= 2:
                    combined = (1 - blend) * ce_norm + blend * wilson
                else:
                    combined = ce_norm
                scored.append((combined, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            top4 = [s[1] for s in scored[:4]]

            # Check if correct answer is in top 4
            hit = any(answer_in_candidate(answer, c["content"]) for c in top4)
            hit_at_1 = answer_in_candidate(answer, top4[0]["content"]) if top4 else False

            # MRR
            mrr = 0.0
            for rank, (_, c) in enumerate(scored[:4], 1):
                if answer_in_candidate(answer, c["content"]):
                    mrr = 1.0 / rank
                    break
            mrr_sum += mrr

            hits_at_4 += int(hit)
            hits_at_1 += int(hit_at_1)
            blend_hits.append(int(hit))

        hit_rate_4 = hits_at_4 / len(cached) if cached else 0
        hit_rate_1 = hits_at_1 / len(cached) if cached else 0
        mrr = mrr_sum / len(cached) if cached else 0

        # McNemar's vs pure CE (blend=0)
        if blend == 0.0:
            pure_ce_hits = blend_hits
            p_str = "  base"
        else:
            n01 = sum(1 for a, b in zip(pure_ce_hits, blend_hits) if a == 0 and b == 1)
            n10 = sum(1 for a, b in zip(pure_ce_hits, blend_hits) if a == 1 and b == 0)
            p_val = mcnemar_p(n01, n10)
            sig = "*" if p_val < 0.05 else " "
            direction = "+" if n01 > n10 else "-" if n10 > n01 else "="
            p_str = f"{direction}{p_val:.4f}{sig}"

        print(f"{blend:>5.2f}  {hit_rate_4:>5.1%}  {hit_rate_1:>5.1%}  {mrr:>6.3f}  {p_str}")

    # Save cache for future analysis
    cache_file = Path("results/blend_sweep_cache.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2, ensure_ascii=False)
    print(f"\nCached scores saved to {cache_file}")

    await strategy.cleanup() if hasattr(strategy, "cleanup") else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--db", type=str, default="runs/poison/02.TagCascade")
    args = parser.parse_args()

    asyncio.run(run_sweep(args.db, args.sample))
