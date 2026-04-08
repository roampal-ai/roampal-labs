#!/usr/bin/env python
"""
Poison resilience pipeline: inject poison → conversation healing → exam.

Runs after the clean pipeline completes. For each strategy:
1. Create fresh empty DB
2. Inject 1,135 poison memories distributed across tiers (with fake metadata)
3. Run full conversation loop (learning ON — system learns real facts alongside poison, decay active)
4. Run LoCoMo exam (learning OFF)
5. Run hard exam (learning OFF)

Compare poisoned exam scores to clean exam scores.
The delta = resilience. System must learn through poison noise — decay should archive poison over time.
"""
import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import (
    GroupConfig, run_exam, run_conversation,
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
POISON_RUNS_DIR = Path("runs/poison")
LIVE_STATE_FILE = RESULTS_DIR / "live_state.json"

STRATEGIES = [
    ("02.TagCascade", lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=True, enable_tags=True, enable_tag_cascade=True), True),
    ("03.CE-Only", lambda d: CELifecycleStrategy(persist_dir=d, enable_decay=True, enable_tags=False), True),
]


# ─── State management ────────────────────────────────────────────────────────

def load_resume_state() -> dict:
    resume_file = RESULTS_DIR / "poison_pipeline_state.json"
    if resume_file.exists():
        return json.loads(resume_file.read_text(encoding="utf-8"))
    return {"completed_steps": {}}


def save_resume_state(state: dict):
    resume_file = RESULTS_DIR / "poison_pipeline_state.json"
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    with open(resume_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def init_live_state(strategy_name: str, step_name: str, step_desc: str) -> dict:
    live_state = {
        "benchmark": "POISON RESILIENCE",
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
    if LIVE_STATE_FILE.exists():
        try:
            existing = json.loads(LIVE_STATE_FILE.read_text(encoding="utf-8"))
            live_state["groups"] = existing.get("groups", {})
            live_state["completed_groups"] = existing.get("completed_groups", 0)
        except Exception:
            pass
    live_state["current_group"] = strategy_name
    live_state["current_step"] = f"{step_name}: {step_desc}"
    _write_live_state(live_state)
    return live_state


# ─── Step executors ──────────────────────────────────────────────────────────

def create_empty_db(strategy_name: str):
    """Create a fresh empty DB directory for poison run."""
    dst = POISON_RUNS_DIR / strategy_name
    if dst.exists():
        print(f"  [init] {dst} already exists, skipping create", flush=True)
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"  [init] Created empty DB at {dst}", flush=True)
    return str(dst)


def _extract_tags_simple(text):
    """Simple tag extraction from text — no LLM, just regex noun matching.
    Simulates what an attacker could do: extract obvious entities from their content."""
    import re
    # Common stopwords to skip
    stop = {'the','a','an','is','was','are','were','be','been','have','has','had','do','does','did',
            'will','would','to','of','in','for','on','with','at','by','from','as','and','but','or',
            'not','no','so','if','that','this','it','they','she','he','we','you','my','me','i',
            'its','them','their','her','him','his','us','our','your','about','after','before',
            'into','also','than','then','when','what','which','who','could','should','very',
            'just','more','some','other','each','many','much','most','such','only','over','same',
            'both','during','found','particularly','interested','attending','attended','pursuing',
            'career','support','people','similar','issues','started','began','mentioned','told',
            'said','asked','talked','discussed','described','explained','felt','feeling','really',
            'around','while','still','already','since','because','through','between','being',
            'going','getting','having','doing','making','taking','giving','coming','looking'}

    # Individual words only
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    # Capitalized names/proper nouns
    names = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    # Date patterns (month + day/year)
    dates = re.findall(r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?', text.lower())
    # Multi-word noun phrases (2 words max, for things like "support group")
    bigrams = re.findall(r'\b([a-z]{3,}\s+[a-z]{3,})\b', text.lower())

    tags = set()
    for name in names:
        n = name.lower().strip()
        if n and n not in stop:
            tags.add(n)
    for word in words:
        if word not in stop:
            tags.add(word)
    for bg in bigrams:
        w1, w2 = bg.split()
        if w1 not in stop and w2 not in stop:
            tags.add(bg)
    for d in dates:
        tags.add(d.strip())

    return '|'.join(sorted(tags)[:8])


async def inject_poison(strategy_factory, data_dir: str, poison_entries: list):
    """Inject poison memories distributed across tiers with realistic metadata.

    Simulates a sophisticated adversarial attack: wrong facts pre-loaded across
    all three tiers with tier-appropriate metadata including tags.
    No fake Wilson overrides — each tier gets randomized realistic metadata.

    Distribution: ~60% working, ~25% history, ~15% patterns
    Each tier gets metadata matching what real promoted memories look like.
    """
    import random
    random.seed(42)  # Reproducible distribution

    strategy = strategy_factory(data_dir)
    await strategy.initialize()

    # Need direct collection access for tier-distributed injection
    if not hasattr(strategy, '_collections'):
        print("    WARN: Strategy has no _collections, falling back to store()", flush=True)
        injected = 0
        for entry in poison_entries:
            content = entry.get("content", "")
            if not content: continue
            try:
                await strategy.store(content)
                injected += 1
            except Exception:
                pass
        await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
        print(f"  [inject] {injected}/{len(poison_entries)} poison (working only — no tier access)", flush=True)
        return injected

    tier_counts = {"working": 0, "history": 0, "patterns": 0}
    injected = 0

    for entry in poison_entries:
        content = entry.get("content", "")
        if not content:
            continue

        # Extract tags from content (same as attacker would)
        tags = _extract_tags_simple(content)

        roll = random.random()

        if roll < 0.60:
            # Working: low uses, low-mid score, recent
            tier = "working"
            uses = random.randint(1, 4)
            meta = {
                "score": round(random.uniform(0.45, 0.70), 2),
                "uses": uses,
                "success_count": round(random.uniform(0.0, 2.0), 1),
                "outcome_history": "[]",
                "last_outcome": random.choice(["worked", "partial", "unknown"]),
                "tier": "working",
                "tags": tags,
                "stored_at": time.time() - random.randint(100, 3600),
            }
        elif roll < 0.85:
            # History: promoted, moderate uses, score >= 0.7
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
                "last_outcome": "worked",
                "tier": "history",
                "tags": tags,
                "stored_at": time.time() - random.randint(3600, 86400),
            }
        else:
            # Patterns: highly promoted, high uses, score >= 0.8
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
                "last_outcome": "worked",
                "tier": "patterns",
                "tags": tags,
                "stored_at": time.time() - random.randint(86400, 172800),
            }

        if entry.get("type"):
            meta["type"] = entry["type"]

        doc_id = f"{tier}_{uuid.uuid4().hex[:8]}"

        try:
            col = strategy._collections.get(tier)
            if col:
                col.add(ids=[doc_id], documents=[content], metadatas=[meta])
                tier_counts[tier] += 1
                injected += 1
            else:
                print(f"    WARN: No collection for tier '{tier}'", flush=True)
        except Exception as e:
            print(f"    WARN: Failed to inject to {tier}: {e}", flush=True)

    # Force HNSW index build by querying each tier — prevents dedup failures
    # when conversation loop starts immediately after injection
    for tier in ["working", "history", "patterns"]:
        col = strategy._collections.get(tier)
        if col and col.count() > 0:
            try:
                col.query(query_texts=["warmup"], n_results=1)
            except Exception:
                pass
    print(f"  [inject] Index warmup complete", flush=True)

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    print(f"  [inject] {injected}/{len(poison_entries)} poison memories injected", flush=True)
    print(f"    Distribution: working={tier_counts['working']}, "
          f"history={tier_counts['history']}, patterns={tier_counts['patterns']}", flush=True)
    return injected


async def run_conversation_step(
    strategy_name: str, strategy_factory, character_sheets: dict,
    live_state: dict, data_dir: str, extract_facts: bool = True,
):
    """Run conversation learning (healing loop)."""
    total_facts = sum(len(f) for chars in character_sheets.values() for f in chars.values())
    est_turns = total_facts // 6 + 1
    facts_label = "with facts" if extract_facts else "summaries only"
    print(f"\n  [healing] Conversation learning (~{est_turns} turns, {total_facts} facts, {facts_label})...", flush=True)

    config = GroupConfig(
        name=strategy_name,
        strategy_factory=lambda d, sf=strategy_factory: sf(d),
        context_factory=lambda: WindowContext(window_size=4),
        extract_facts=extract_facts,
    )

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
    print(f"  [healing] Done: {results.turns} turns, "
          f"{results.memories_stored} memories stored", flush=True)
    return results


async def run_exam_step(
    strategy_name: str, strategy_factory, exam_questions: list,
    live_state: dict, step_name: str, data_dir: str,
):
    """Run exam (learning OFF)."""
    print(f"\n  [{step_name}] Exam ({len(exam_questions)} Qs, learning=OFF)...", flush=True)

    strategy = strategy_factory(data_dir)
    await strategy.initialize()

    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    if strategy_name not in live_state.get("groups", {}):
        live_state["groups"][strategy_name] = {}

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
    print(f"  [{step_name}] Done: {correct}/{total} = {acc:.1%}", flush=True)

    # Save transcript
    transcript_file = RESULTS_DIR / f"poison_exam_{strategy_name}_{step_name}.json"
    transcript = {
        "strategy": strategy_name,
        "step": step_name,
        "pipeline": "poison",
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
    print(f"  [{step_name}] Transcript saved: {transcript_file.name}", flush=True)

    # Update live_state exam_history
    group_state = live_state.get("groups", {}).get(strategy_name, {})
    if "exam_history" not in group_state:
        group_state["exam_history"] = []
    group_state["exam_history"].append({
        "step": f"poison_{step_name}",
        "correct": results["correct"],
        "partial": results.get("partial", 0),
        "wrong": results["wrong"],
        "total": results["total"],
        "accuracy": round(acc, 4),
        "learning": False,
        "by_category": results.get("by_category", {}),
    })
    live_state["groups"][strategy_name] = group_state
    _write_live_state(live_state)

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None
    return results


# ─── Main pipeline ───────────────────────────────────────────────────────────

async def run_poison_pipeline():
    """Poison resilience pipeline for all strategies."""

    print("Loading data...", flush=True)
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))

    character_sheets = load_character_sheets("data/character_sheets")

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

    # Load ALL poison entries (summaries + facts)
    poison_data = json.loads(Path("data/poison_memories_v2.json").read_text(encoding="utf-8"))
    poison_entries = poison_data.get("poison_entries", [])
    print(f"  Poison entries: {len(poison_entries)} (with fake Wilson scores)")

    total_facts = sum(len(f) for chars in character_sheets.values() for f in chars.values())
    print(f"  Character sheets: {len(character_sheets)} conversations, {total_facts} facts")
    print(f"  LoCoMo exam: {len(locomo_questions)}")
    print(f"  Hard exam: {len(hard_normalized)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    POISON_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    resume_state = load_resume_state()
    completed = resume_state.get("completed_steps", {})

    print(f"\n{'='*60}")
    print(f"POISON RESILIENCE PIPELINE")
    print(f"{'='*60}")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Steps: clone -> inject -> conversation -> locomo -> hard")
    print(f"Already completed: {len(completed)}")
    print(f"{'='*60}\n")

    STEPS = [
        ("01_clone",        "clone",        "Create fresh empty DB"),
        ("02_inject",       "inject",       "Inject poison (tier-distributed, fake metadata)"),
        ("03_conversation", "conversation", "Conversation healing loop"),
        ("04_locomo",       "locomo",       "LoCoMo exam (learning OFF)"),
        ("05_hard",         "hard",         "Hard exam (learning OFF)"),
    ]

    for strat_idx, (strategy_name, strategy_factory, strat_extract_facts) in enumerate(STRATEGIES):
        print(f"\n{'='*60}")
        print(f"STRATEGY {strat_idx+1}/{len(STRATEGIES)}: {strategy_name}")
        print(f"{'='*60}")

        data_dir = str(POISON_RUNS_DIR / strategy_name)

        for step_name, step_type, step_desc in STEPS:
            step_key = f"{strategy_name}:{step_name}"

            if step_key in completed:
                print(f"\n  [{step_name}] SKIP (already completed)", flush=True)
                continue

            live_state = init_live_state(strategy_name, step_name, step_desc)

            try:
                if step_type == "clone":
                    data_dir = create_empty_db(strategy_name)

                elif step_type == "inject":
                    await inject_poison(strategy_factory, data_dir, poison_entries)

                elif step_type == "conversation":
                    await run_conversation_step(
                        strategy_name, strategy_factory, character_sheets,
                        live_state=live_state, data_dir=data_dir,
                        extract_facts=strat_extract_facts,
                    )

                elif step_type == "locomo":
                    await run_exam_step(
                        strategy_name, strategy_factory, locomo_questions,
                        live_state=live_state, step_name="locomo_off",
                        data_dir=data_dir,
                    )

                elif step_type == "hard":
                    await run_exam_step(
                        strategy_name, strategy_factory, hard_normalized,
                        live_state=live_state, step_name="hard_off",
                        data_dir=data_dir,
                    )

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

        print(f"\n  === {strategy_name} POISON COMPLETE ===", flush=True)

    print(f"\n{'='*60}")
    print(f"POISON PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {RESULTS_DIR} (poison_exam_*.json)")
    print(f"Compare to clean exam scores for resilience delta.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(run_poison_pipeline())
