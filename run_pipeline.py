#!/usr/bin/env python
"""
roampal-labs: Final benchmark pipeline orchestrator.

Runs 12 steps per strategy, 4 strategies total.
See PIPELINE.md for the full design.
"""
import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import (
    GroupConfig, GroupResults, run_group, run_exam, run_conversation,
    load_character_sheets, _write_live_state,
)
from benchmark.grader import LLMGrader
from strategies.ce_lifecycle import CELifecycleStrategy
from context.window import WindowContext


# ─── Configuration ───────────────────────────────────────────────────────────

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs/final")
LIVE_STATE_FILE = RESULTS_DIR / "live_state.json"

STRATEGIES = [
    # (name, factory, extract_facts)
    ("02.TagCascade", lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=True, enable_tag_cascade=True), True),
    ("03.CE-Only", lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=False, enable_tags=False), True),
]

PIPELINE = [
    # (step_name, step_type, learning, description)
    ("01_conversation",     "conversation", True,  "Conversational learning"),
    ("02_locomo_off",       "locomo",       False, "LoCoMo exam (no Wilson)"),
    ("03_hard_off",         "hard",         False, "Hard exam (held-out reasoning)"),
]


# ─── State management ────────────────────────────────────────────────────────

def load_resume_state() -> dict:
    """Load resume state to skip completed steps."""
    resume_file = RESULTS_DIR / "pipeline_state.json"
    if resume_file.exists():
        return json.loads(resume_file.read_text(encoding="utf-8"))
    return {"completed_steps": {}}


def save_resume_state(state: dict):
    """Save pipeline progress for resume."""
    resume_file = RESULTS_DIR / "pipeline_state.json"
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    with open(resume_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def init_live_state(strategy_name: str, step_name: str, step_desc: str) -> dict:
    """Initialize or update live_state.json for dashboard."""
    live_state = {
        "current_group": strategy_name,
        "current_step": f"{step_name}: {step_desc}",
        "current_turn": 0,
        "total_turns": 0,
        "total_groups": len(STRATEGIES),
        "completed_groups": 0,
        "groups": {},
        "feed": [],
        "updated_at": time.time(),
    }
    # Load existing if available (preserve group data across steps)
    if LIVE_STATE_FILE.exists():
        try:
            existing = json.loads(LIVE_STATE_FILE.read_text(encoding="utf-8"))
            valid_names = {s[0] for s in STRATEGIES} | {"baseline_raw_repaired"}
            live_state["groups"] = {k: v for k, v in existing.get("groups", {}).items() if k in valid_names}
            live_state["completed_groups"] = existing.get("completed_groups", 0)
        except Exception:
            pass
    live_state["current_group"] = strategy_name
    live_state["current_step"] = f"{step_name}: {step_desc}"
    _write_live_state(live_state)
    return live_state


# ─── Step executors ──────────────────────────────────────────────────────────

async def run_conversation_step(
    strategy_name: str, strategy_factory, character_sheets: dict,
    live_state: dict, step_name: str, data_dir: str,
    extract_facts: bool = True,
):
    """Run conversation-based learning: LLM B shares facts, LLM A responds."""
    total_facts = sum(len(f) for chars in character_sheets.values() for f in chars.values())
    est_turns = total_facts // 6 + 1
    facts_label = "with facts" if extract_facts else "summaries only"
    print(f"\n  [{step_name}] Conversation learning (~{est_turns} turns, {total_facts} facts, {facts_label})...", flush=True)

    config = GroupConfig(
        name=strategy_name,
        strategy_factory=lambda d, sf=strategy_factory: sf(d),
        context_factory=lambda: WindowContext(window_size=4),
        extract_facts=extract_facts,
    )

    # Initialize group in live_state if not present
    if strategy_name not in live_state.get("groups", {}):
        live_state["groups"][strategy_name] = {
            "turns": 0, "facts_covered": 0, "facts_total": total_facts,
        }

    result = await run_conversation(
        config=config,
        character_sheets=character_sheets,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        data_dir=data_dir,
        live_state=live_state,
    )

    results, strategy = result if isinstance(result, tuple) else (result, None)

    print(f"  [{step_name}] Done: {results.turns} turns, "
          f"{results.memories_stored} memories stored", flush=True)
    return results


async def run_exam_step(
    strategy_name: str, strategy_factory, exam_questions: list,
    learning: bool, live_state: dict, step_name: str,
    data_dir: str, use_llm_grading: bool = True,
):
    """Run an exam (LoCoMo or hard) with learning ON or OFF."""
    label = "ON" if learning else "OFF"
    print(f"\n  [{step_name}] Exam ({len(exam_questions)} Qs, learning={label})...", flush=True)

    # Create strategy instance
    strategy = strategy_factory(data_dir)
    await strategy.initialize()

    grader = None
    if use_llm_grading:
        grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
        await grader.initialize()

    # Ensure group exists in live_state for dashboard updates
    if strategy_name not in live_state.get("groups", {}):
        live_state["groups"][strategy_name] = {}

    # Rename checkpoint file to match the group_name we'll use
    # This allows mid-step resume while keeping dashboard updates working
    old_ckpt = RESULTS_DIR / f"exam_checkpoint_{strategy_name}_{step_name}.json"
    new_ckpt = RESULTS_DIR / f"exam_checkpoint_{strategy_name}.json"
    if old_ckpt.exists() and not new_ckpt.exists():
        old_ckpt.rename(new_ckpt)
    # Also check reverse (in case format changed)
    if not new_ckpt.exists():
        for f in RESULTS_DIR.glob(f"exam_checkpoint_*{step_name}*"):
            f.rename(new_ckpt)
            break

    results = await run_exam(
        strategy=strategy,
        exam_queries=exam_questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        group_name=strategy_name,
        grader=grader,
        live_state=live_state,
        learning=learning,
    )

    # Clean up checkpoint explicitly (run_exam should do this but be safe)
    if new_ckpt.exists():
        new_ckpt.unlink()

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0
    print(f"  [{step_name}] Done: {correct}/{total} = {acc:.1%}", flush=True)

    # Save exam transcript
    transcript_file = RESULTS_DIR / f"exam_{strategy_name}_{step_name}.json"
    transcript = {
        "strategy": strategy_name,
        "step": step_name,
        "learning": learning,
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
    print(f"  [{step_name}] Transcript saved: {transcript_file.name}", flush=True)

    # Save to exam_history in live_state for dashboard
    group_state = live_state.get("groups", {}).get(strategy_name, {})
    if "exam_history" not in group_state:
        group_state["exam_history"] = []
    group_state["exam_history"].append({
        "step": step_name,
        "correct": results["correct"],
        "partial": results.get("partial", 0),
        "wrong": results["wrong"],
        "total": results["total"],
        "accuracy": round(acc, 4),
        "learning": learning,
        "by_category": results.get("by_category", {}),
    })
    live_state["groups"][strategy_name] = group_state
    _write_live_state(live_state)

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    return results


def snapshot_db(strategy_name: str):
    """Backup ChromaDB directory before poison."""
    src = RUNS_DIR / strategy_name
    dst = RUNS_DIR / f"{strategy_name}_pre_poison_snapshot"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  [snapshot] Backed up {src} -> {dst}", flush=True)


async def inject_poison(strategy_factory, data_dir: str, poison_data: list):
    """Inject poison memories distributed across tiers with realistic metadata.

    Distribution: ~60% working, ~25% history, ~15% patterns.
    Each tier gets metadata matching what real promoted memories look like.
    """
    import random
    random.seed(42)  # Reproducible distribution

    strategy = strategy_factory(data_dir)
    await strategy.initialize()

    if not hasattr(strategy, '_collections'):
        print("    WARN: Strategy has no _collections, falling back to store()", flush=True)
        for pm in poison_data:
            content = pm.get("content", "")
            if content:
                await strategy.store(content)
        await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
        print(f"  [inject] {len(poison_data)} poison (working only)", flush=True)
        return

    tier_counts = {"working": 0, "history": 0, "patterns": 0}
    injected = 0

    for pm in poison_data:
        content = pm.get("content", "")
        if not content:
            continue

        fake_meta = pm.get("fake_meta", {})
        roll = random.random()

        if roll < 0.60:
            tier = "working"
            meta = {
                "score": round(random.uniform(0.45, 0.70), 2),
                "uses": random.randint(1, 4),
                "success_count": round(random.uniform(0.0, 2.0), 1),
                "outcome_history": "[]",
                "tier": "working",
                "stored_at": time.time() - random.randint(100, 3600),
            }
        elif roll < 0.85:
            tier = "history"
            uses = random.randint(5, 12)
            meta = {
                "score": round(random.uniform(0.70, 0.85), 2),
                "uses": uses,
                "success_count": round(random.uniform(0.0, 3.0), 1),
                "outcome_history": json.dumps([
                    {"outcome": random.choice(["worked", "worked", "partial"]),
                     "timestamp": str(time.time() - random.randint(50, 500))}
                    for _ in range(min(uses, 3))
                ]),
                "tier": "history",
                "stored_at": time.time() - random.randint(3600, 86400),
            }
        else:
            tier = "patterns"
            uses = random.randint(10, 20)
            meta = {
                "score": round(random.uniform(0.80, 0.95), 2),
                "uses": uses,
                "success_count": round(random.uniform(5.0, 8.0), 1),
                "outcome_history": json.dumps([
                    {"outcome": random.choice(["worked", "worked", "worked", "partial"]),
                     "timestamp": str(time.time() - random.randint(50, 500))}
                    for _ in range(3)
                ]),
                "tier": "patterns",
                "stored_at": time.time() - random.randint(86400, 172800),
            }

        if fake_meta:
            meta["score"] = fake_meta.get("score", meta["score"])
            meta["uses"] = fake_meta.get("uses", meta["uses"])
            meta["success_count"] = fake_meta.get("success_count", meta["success_count"])

        if pm.get("type"):
            meta["type"] = pm["type"]

        doc_id = f"{tier}_{uuid.uuid4().hex[:8]}"

        try:
            col = strategy._collections.get(tier)
            if col:
                col.add(ids=[doc_id], documents=[content], metadatas=[meta])
                tier_counts[tier] += 1
                injected += 1
        except Exception as e:
            print(f"    WARN: Failed to inject to {tier}: {e}", flush=True)

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    print(f"  [inject] {injected}/{len(poison_data)} poison memories injected", flush=True)
    print(f"    Distribution: working={tier_counts['working']}, "
          f"history={tier_counts['history']}, patterns={tier_counts['patterns']}", flush=True)


# ─── Main pipeline ───────────────────────────────────────────────────────────

async def run_pipeline():
    """Run the full 12-step pipeline for all 4 strategies."""

    # Load data
    print("Loading data...", flush=True)
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))

    # Load character sheets for conversation-based learning
    character_sheets = load_character_sheets("data/character_sheets")

    # Normalize LoCoMo questions: runner expects 'query' key
    locomo_questions = []
    for q in data["locomo_exam"]:
        locomo_questions.append({
            "query": q.get("question", q.get("query", "")),
            "ground_truth": q.get("ground_truth", ""),
            "category_name": q.get("category_name", "unknown"),
        })

    hard_questions = json.loads(Path("data/hard_exam.json").read_text(encoding="utf-8"))
    poison_data = json.loads(Path("data/poison_memories_v2.json").read_text(encoding="utf-8"))
    poison_memories = poison_data.get("poison_entries", [])

    total_facts = sum(len(f) for chars in character_sheets.values() for f in chars.values())
    print(f"  Character sheets: {len(character_sheets)} conversations, {total_facts} facts")
    print(f"  LoCoMo exam: {len(locomo_questions)}")
    print(f"  Hard exam: {len(hard_questions)}")
    print(f"  Poison memories: {len(poison_memories)}")

    # Ensure directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load resume state
    resume_state = load_resume_state()
    completed = resume_state.get("completed_steps", {})

    print(f"\n{'='*60}")
    print(f"ROAMPAL-LABS BENCHMARK PIPELINE")
    print(f"{'='*60}")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Steps per strategy: {len(PIPELINE)}")
    print(f"Already completed: {len(completed)}")
    print(f"{'='*60}\n")

    for strat_idx, (strategy_name, strategy_factory, strat_extract_facts) in enumerate(STRATEGIES):
        print(f"\n{'='*60}")
        print(f"STRATEGY {strat_idx+1}/{len(STRATEGIES)}: {strategy_name}")
        print(f"{'='*60}")

        data_dir = str(RUNS_DIR / strategy_name)
        os.makedirs(data_dir, exist_ok=True)

        for step_name, step_type, learning, step_desc in PIPELINE:
            step_key = f"{strategy_name}:{step_name}"

            # Skip completed steps (resume support)
            if step_key in completed:
                print(f"\n  [{step_name}] SKIP (already completed)", flush=True)
                continue

            live_state = init_live_state(strategy_name, step_name, step_desc)

            try:
                if step_type == "conversation":
                    await run_conversation_step(
                        strategy_name, strategy_factory, character_sheets,
                        live_state=live_state, step_name=step_name,
                        data_dir=data_dir,
                        extract_facts=strat_extract_facts,
                    )

                elif step_type == "locomo":
                    await run_exam_step(
                        strategy_name, strategy_factory, locomo_questions,
                        learning=learning, live_state=live_state, step_name=step_name,
                        data_dir=data_dir, use_llm_grading=True,
                    )

                elif step_type == "hard":
                    # Hard exam uses the same format but questions have 'question' not 'query'
                    # Normalize to match run_exam's expected format
                    normalized = []
                    for q in hard_questions:
                        normalized.append({
                            "query": q.get("question", q.get("query", "")),
                            "ground_truth": q.get("ground_truth", ""),
                            "category_name": q.get("category", "unknown"),
                        })
                    await run_exam_step(
                        strategy_name, strategy_factory, normalized,
                        learning=False, live_state=live_state, step_name=step_name,
                        data_dir=data_dir, use_llm_grading=True,
                    )

                elif step_type == "snapshot":
                    snapshot_db(strategy_name)

                elif step_type == "poison":
                    await inject_poison(strategy_factory, data_dir, poison_memories)

                # Mark step complete
                completed[step_key] = {
                    "completed_at": time.time(),
                    "step_desc": step_desc,
                }
                save_resume_state({"completed_steps": completed})
                print(f"  [{step_name}] SAVED to resume state", flush=True)

            except Exception as e:
                print(f"\n  [{step_name}] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print(f"\n  Pipeline paused at {step_key}. Rerun to resume.", flush=True)
                return

        print(f"\n  === {strategy_name} COMPLETE ===", flush=True)

    print(f"\n{'='*60}")
    print(f"ALL STRATEGIES COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {RESULTS_DIR}")
    print(f"Run MiniMax regrading on exam transcripts for dual-grading.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(run_pipeline())
