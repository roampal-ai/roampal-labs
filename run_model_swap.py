#!/usr/bin/env python
"""
Model swap experiment: re-run LoCoMo exam with GPT-4o-mini answering model
against existing DBs. Same retrieval (local CE), different answering LLM.

Grading uses local gpt-oss:20b for consistency with all other runs.
MiniMax regrades later.

Usage:
  set OPENAI_API_KEY=sk-proj-...
  python run_model_swap.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import run_exam, _write_live_state
from benchmark.grader import LLMGrader
from strategies.ce_lifecycle import CELifecycleStrategy
from strategies.semantic_reranker import SemanticRerankerStrategy

# OpenAI for answering
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"

# Local for grading (consistency)
LOCAL_BASE_URL = "http://localhost:11434/v1"
GRADER_MODEL = "gpt-oss:20b"

RESULTS_DIR = Path("results")

CONDITIONS = [
    # Clean
    ("4omini_02.TagCascade_clean", "runs/final/02.TagCascade",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=True, enable_tag_cascade=True)),
    ("4omini_03.CE-Only_clean", "runs/final/03.CE-Only",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=False)),
    ("4omini_baseline", "runs/final/ingest_transcript_baseline",
     lambda d: SemanticRerankerStrategy(persist_dir=d)),
    # Poison
    ("4omini_02.TagCascade_poison", "runs/poison/02.TagCascade",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=True, enable_tag_cascade=True)),
    ("4omini_03.CE-Only_poison", "runs/poison/03.CE-Only",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=False)),
]


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("MODEL SWAP: GPT-4o-mini answering, local 20B grading")
    print(f"Conditions: {len(CONDITIONS)}")
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

    # Live state for dashboard
    live_state = {
        "benchmark": "GPT-4o-mini MODEL SWAP",
        "current_group": "",
        "current_step": "",
        "current_turn": 0,
        "total_turns": 0,
        "total_groups": len(CONDITIONS),
        "completed_groups": 0,
        "groups": {},
        "feed": [],
        "updated_at": time.time(),
    }

    # Build list of all exams to run
    exam_sets = [
        ("locomo", locomo_questions),
        ("hard", hard_normalized),
    ]

    for cond_idx, (label, db_path, strategy_factory) in enumerate(CONDITIONS):
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_idx+1}/{len(CONDITIONS)}: {label}")
        print(f"  DB: {db_path}")
        print(f"{'='*60}")

        if not Path(db_path).exists():
            print(f"  SKIP: DB not found at {db_path}")
            continue

        # Check which exams still need to run
        exams_to_run = []
        for exam_name, exam_qs in exam_sets:
            out_file = RESULTS_DIR / f"exam_{label}_{exam_name}_off.json"
            if out_file.exists():
                print(f"  SKIP {exam_name}: Already completed ({out_file.name})")
            else:
                exams_to_run.append((exam_name, exam_qs, out_file))

        if not exams_to_run:
            live_state["completed_groups"] = cond_idx + 1
            continue

        strategy = strategy_factory(db_path)
        await strategy.initialize()

        # Memory count
        total_mems = 0
        if hasattr(strategy, '_collections'):
            for tier_name, col in strategy._collections.items():
                try:
                    total_mems += col.count()
                except Exception:
                    pass
        elif hasattr(strategy, '_collection') and strategy._collection:
            total_mems = strategy._collection.count()
        print(f"  Memories in DB: {total_mems}")

        live_state["current_group"] = label
        live_state["groups"][label] = {"memories": total_mems}

        # Grader uses local 20B
        grader = LLMGrader(base_url=LOCAL_BASE_URL, model=GRADER_MODEL)
        await grader.initialize()

        for exam_name, exam_qs, out_file in exams_to_run:
            live_state["current_step"] = f"{exam_name.capitalize()} exam"
            _write_live_state(live_state)

            print(f"\n  [{label} {exam_name}] Running {len(exam_qs)} questions with {OPENAI_MODEL}...")

            results = await run_exam(
                strategy=strategy,
                exam_queries=exam_qs,
                llm_base_url=OPENAI_BASE_URL,
                llm_model=OPENAI_MODEL,
                group_name=label,
                grader=grader,
                live_state=live_state,
                learning=False,
                llm_api_key=api_key,
            )

            correct = results["correct"]
            total = results["total"]
            acc = correct / total if total else 0
            print(f"\n  [{label} {exam_name}] Done: {correct}/{total} = {acc:.1%}")

            transcript = {
                "strategy": label,
                "step": f"{exam_name}_off",
                "pipeline": "model_swap",
                "llm_model": OPENAI_MODEL,
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
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_file.name}")

        await strategy.cleanup() if hasattr(strategy, 'cleanup') else None

        live_state["completed_groups"] = cond_idx + 1
        _write_live_state(live_state)

    print(f"\n{'='*60}")
    print("MODEL SWAP COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(main())
