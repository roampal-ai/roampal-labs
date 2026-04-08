#!/usr/bin/env python
"""
Run remaining baseline exams:
1. Raw chunk ingestion → Hard exam (76 questions)
2. No-memory baseline → LoCoMo exam (1986 questions) — LLM answers cold
3. No-memory baseline → Hard exam (76 questions) — LLM answers cold

Usage:
  python run_remaining_baselines.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import run_exam
from benchmark.grader import LLMGrader
from strategies.semantic_reranker import SemanticRerankerStrategy
from strategies.base import RetrievalResult, MemoryRecord

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
RESULTS_DIR = Path("results")


class NullStrategy:
    """Returns no memories — establishes the zero-retrieval floor."""
    name = "no_memory"

    async def initialize(self):
        pass

    async def retrieve(self, query: str, top_k: int = 4, **kwargs) -> RetrievalResult:
        return RetrievalResult(memories=[], formatted_injection="", query_used=query, retrieval_ms=0)

    async def record_outcome(self, memory_ids, outcome, exchange_summary=""):
        pass

    async def store(self, content, metadata=None):
        return ""


async def run_single(label, strategy, questions, exam_type):
    """Run one exam and save transcript."""
    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    step_name = f"{exam_type}_off"
    full_label = f"{label}_{step_name}"
    print(f"\n  [{full_label}] Running {len(questions)} questions with {LLM_MODEL}...")

    live_state = {
        "current_group": label,
        "groups": {label: {}},
        "feed": [],
        "updated_at": time.time(),
    }

    results = await run_exam(
        strategy=strategy,
        exam_queries=questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        group_name=label,
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0
    print(f"  [{full_label}] Done: {correct}/{total} = {acc:.1%}")

    transcript = {
        "strategy": label,
        "step": step_name,
        "learning": False,
        "summary": {
            "correct": results["correct"],
            "partial": results.get("partial", 0),
            "wrong": results["wrong"],
            "total": results["total"],
            "accuracy": round(acc, 4),
            "by_category": results.get("by_category", {}),
        },
        "transcript": results.get("log", []),
        "timestamp": time.time(),
    }

    out_file = RESULTS_DIR / f"exam_{label}_{step_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"  [{full_label}] Saved: {out_file.name}")

    return results


async def main():
    print("=" * 60)
    print("REMAINING BASELINE EXAMS")
    print("=" * 60)

    # Load exam data
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    locomo_questions = []
    for q in data["locomo_exam"]:
        locomo_questions.append({
            "query": q.get("question", q.get("query", "")),
            "ground_truth": q.get("ground_truth", ""),
            "category_name": q.get("category_name", "unknown"),
        })

    hard_questions = json.loads(Path("data/hard_exam.json").read_text(encoding="utf-8"))
    hard_normalized = []
    for q in hard_questions:
        hard_normalized.append({
            "query": q.get("question", q.get("query", "")),
            "ground_truth": q.get("ground_truth", ""),
            "category_name": q.get("category", "unknown"),
        })

    print(f"  LoCoMo: {len(locomo_questions)} questions")
    print(f"  Hard:   {len(hard_normalized)} questions")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 1. Raw chunk baseline → Hard exam ──────────────────────────────────
    print(f"\n{'='*60}")
    print("1/3: RAW CHUNK BASELINE -> HARD EXAM")
    print(f"{'='*60}")

    db_dir = Path("runs/final/ingest_transcript_baseline")
    if db_dir.exists():
        strategy = SemanticRerankerStrategy(persist_dir=str(db_dir))
        await strategy.initialize()
        count = strategy._collection.count() if hasattr(strategy, '_collection') and strategy._collection else 0
        print(f"  Baseline DB: {count} chunks")
        await run_single("baseline_raw_repaired", strategy, hard_normalized, "hard")
        await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    else:
        print(f"  SKIP: DB not found at {db_dir}")

    # ─── 2. No-memory baseline → LoCoMo ────────────────────────────────────
    print(f"\n{'='*60}")
    print("2/3: NO-MEMORY BASELINE -> LOCOMO EXAM")
    print(f"{'='*60}")

    null_strategy = NullStrategy()
    await null_strategy.initialize()
    await run_single("no_memory", null_strategy, locomo_questions, "locomo")

    # ─── 3. No-memory baseline → Hard ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("3/3: NO-MEMORY BASELINE -> HARD EXAM")
    print(f"{'='*60}")

    await run_single("no_memory", null_strategy, hard_normalized, "hard")

    print(f"\n{'='*60}")
    print("ALL REMAINING BASELINES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(main())
