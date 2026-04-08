#!/usr/bin/env python
"""
Gemma 4 exam runner: tests gemma4:31b against all 4 DB conditions.

Runs LoCoMo + hard exams on:
  1. TagCascade clean  (runs/final/02.TagCascade)
  2. TagCascade poison (runs/poison/02.TagCascade)
  3. CE-Only clean     (runs/final/03.CE-Only)
  4. CE-Only poison    (runs/poison/03.CE-Only)

No conversation, no learning — exam only against existing DBs.
Uses gemma4:31b via ollama as the answering LLM.
Grading still uses the existing LLM grader (gpt-oss:20b).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import run_exam, _write_live_state
from benchmark.grader import LLMGrader
from strategies.ce_lifecycle import CELifecycleStrategy


# ─── Configuration ───────────────────────────────────────────────────────────

LLM_BASE_URL = "http://localhost:11434/v1"

# Gemma 4 31B answers the questions
GEMMA_MODEL = "gemma4:31b"

# Grading LLM — keep the existing model so grading is consistent with prior runs
GRADER_MODEL = "gpt-oss:20b"

RESULTS_DIR = Path("results")

# All 4 conditions: (label, db_path, strategy_factory)
CONDITIONS = [
    ("gemma4_02.TagCascade_clean", "runs/final/02.TagCascade",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=True, enable_tag_cascade=True)),

    ("gemma4_02.TagCascade_poison", "runs/poison/02.TagCascade",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=True, enable_tag_cascade=True)),

    ("gemma4_03.CE-Only_clean", "runs/final/03.CE-Only",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=False)),

    ("gemma4_03.CE-Only_poison", "runs/poison/03.CE-Only",
     lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=False)),
]


# ─── State management ────────────────────────────────────────────────────────

def load_resume_state() -> dict:
    resume_file = RESULTS_DIR / "gemma_exam_state.json"
    if resume_file.exists():
        return json.loads(resume_file.read_text(encoding="utf-8"))
    return {"completed_steps": {}}


def save_resume_state(state: dict):
    resume_file = RESULTS_DIR / "gemma_exam_state.json"
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    with open(resume_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ─── Exam runner ─────────────────────────────────────────────────────────────

async def run_single_exam(
    label: str,
    strategy,
    exam_questions: list,
    exam_type: str,
    live_state: dict,
):
    """Run one exam condition and save results."""
    grader = LLMGrader(base_url=LLM_BASE_URL, model=GRADER_MODEL)
    await grader.initialize()

    step_name = f"{exam_type}_off"
    full_label = f"{label}_{step_name}"
    print(f"\n  [{full_label}] Running {len(exam_questions)} questions with {GEMMA_MODEL}...", flush=True)

    results = await run_exam(
        strategy=strategy,
        exam_queries=exam_questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=GEMMA_MODEL,
        group_name=label,
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0
    print(f"  [{full_label}] Done: {correct}/{total} = {acc:.1%}", flush=True)

    # Save transcript
    transcript_file = RESULTS_DIR / f"gemma4_exam_{label}_{step_name}.json"
    transcript = {
        "strategy": label,
        "step": step_name,
        "pipeline": "gemma4",
        "llm_model": GEMMA_MODEL,
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
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"  [{full_label}] Transcript saved: {transcript_file.name}", flush=True)

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("GEMMA 4 EXAM RUNNER")
    print(f"Model: {GEMMA_MODEL}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Exams per condition: 2 (LoCoMo + Hard)")
    print("=" * 60)

    # Load exam data
    print("\nLoading exam data...", flush=True)
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
    resume_state = load_resume_state()
    completed = resume_state.get("completed_steps", {})

    # Live state for dashboard
    live_state = {
        "benchmark": "GEMMA 4 EXAMS",
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

    scoreboard = []

    for cond_idx, (label, db_path, strategy_factory) in enumerate(CONDITIONS):
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_idx+1}/{len(CONDITIONS)}: {label}")
        print(f"  DB: {db_path}")
        print(f"{'='*60}")

        # Verify DB exists
        if not Path(db_path).exists():
            print(f"  SKIP: DB not found at {db_path}", flush=True)
            continue

        strategy = strategy_factory(db_path)
        await strategy.initialize()

        # Get memory count
        total_mems = 0
        if hasattr(strategy, '_collections'):
            for tier_name, col in strategy._collections.items():
                try:
                    total_mems += col.count()
                except Exception:
                    pass
        print(f"  Memories in DB: {total_mems}", flush=True)

        live_state["current_group"] = label
        live_state["groups"][label] = {"memories": total_mems}

        # LoCoMo exam
        step_key = f"{label}:locomo"
        if step_key not in completed:
            live_state["current_step"] = "LoCoMo exam"
            _write_live_state(live_state)

            locomo_results = await run_single_exam(
                label, strategy, locomo_questions, "locomo", live_state,
            )
            locomo_acc = locomo_results["correct"] / locomo_results["total"] if locomo_results["total"] else 0

            completed[step_key] = {"completed_at": time.time(), "accuracy": round(locomo_acc, 4)}
            save_resume_state({"completed_steps": completed})
        else:
            locomo_acc = completed[step_key].get("accuracy", 0)
            print(f"  [locomo] SKIP (already completed: {locomo_acc:.1%})", flush=True)

        # Hard exam
        step_key = f"{label}:hard"
        if step_key not in completed:
            live_state["current_step"] = "Hard exam"
            _write_live_state(live_state)

            hard_results = await run_single_exam(
                label, strategy, hard_normalized, "hard", live_state,
            )
            hard_acc = hard_results["correct"] / hard_results["total"] if hard_results["total"] else 0

            completed[step_key] = {"completed_at": time.time(), "accuracy": round(hard_acc, 4)}
            save_resume_state({"completed_steps": completed})
        else:
            hard_acc = completed[step_key].get("accuracy", 0)
            print(f"  [hard] SKIP (already completed: {hard_acc:.1%})", flush=True)

        await strategy.cleanup() if hasattr(strategy, 'cleanup') else None

        scoreboard.append({
            "condition": label,
            "locomo": round(locomo_acc, 4),
            "hard": round(hard_acc, 4),
        })

        live_state["completed_groups"] = cond_idx + 1
        _write_live_state(live_state)

    # Final scoreboard
    print(f"\n{'='*60}")
    print("GEMMA 4 SCOREBOARD")
    print(f"{'='*60}")
    print(f"{'Condition':<40} {'LoCoMo':>8} {'Hard':>8}")
    print("-" * 60)
    for row in scoreboard:
        print(f"{row['condition']:<40} {row['locomo']:>7.1%} {row['hard']:>7.1%}")
    print(f"{'='*60}")

    # Save scoreboard
    scoreboard_file = RESULTS_DIR / "gemma4_scoreboard.json"
    with open(scoreboard_file, "w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2)
    print(f"\nScoreboard saved: {scoreboard_file}")


if __name__ == "__main__":
    asyncio.run(main())
