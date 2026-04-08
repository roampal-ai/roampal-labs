#!/usr/bin/env python
"""
Rerun exams on existing DBs with the retrieval fix applied.

Same memories, same DBs — just fixed two-lane retrieval (ChromaDB where clause
instead of Python post-filtering). Runs LoCoMo + hard exam for each strategy.

Usage:
  python run_exam_reruns.py
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
from strategies.wilson_scored import WilsonScoredStrategy
from strategies.semantic_reranker import SemanticRerankerStrategy
from strategies.wilson_reranker import WilsonRerankerStrategy
from strategies.entity_routed import EntityRoutedStrategy

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs/final")

# Strategies to rerun (existing DBs, fixed retrieval code)
STRATEGIES = [
    ("01.EntityRouted", lambda d: EntityRoutedStrategy(persist_dir=d)),
    ("02.Wilson+CE", lambda d: WilsonRerankerStrategy(persist_dir=d)),
    ("03.Reranker", lambda d: SemanticRerankerStrategy(persist_dir=d)),
]


async def run_exam_on_strategy(strategy_name, strategy_factory, exam_questions, exam_name, data_dir):
    strategy = strategy_factory(data_dir)
    await strategy.initialize()

    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    live_state = {
        "current_group": strategy_name,
        "groups": {strategy_name: {}},
        "feed": [],
        "updated_at": time.time(),
    }

    results = await run_exam(
        strategy=strategy,
        exam_queries=exam_questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        group_name=strategy_name,
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0

    transcript = {
        "strategy": strategy_name,
        "step": f"fixed_{exam_name}",
        "description": f"Exam rerun with retrieval fix on existing DB",
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

    out_file = RESULTS_DIR / f"exam_{strategy_name}_fixed_{exam_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"  {strategy_name} {exam_name}: {acc:.1%} ({correct}/{total})")
    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    return results


async def main():
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

    print(f"LoCoMo: {len(locomo_questions)}, Hard: {len(hard_normalized)}")

    for strategy_name, strategy_factory in STRATEGIES:
        data_dir = str(RUNS_DIR / strategy_name)
        if not os.path.exists(data_dir):
            print(f"\n  SKIP {strategy_name} — DB not found at {data_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {strategy_name} (fixed retrieval, existing DB)")
        print(f"{'='*60}")

        await run_exam_on_strategy(
            strategy_name, strategy_factory, locomo_questions, "locomo", data_dir
        )
        await run_exam_on_strategy(
            strategy_name, strategy_factory, hard_normalized, "hard", data_dir
        )

    print(f"\n{'='*60}")
    print(f"ALL EXAM RERUNS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(main())
