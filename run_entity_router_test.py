"""EntityRouter quick test: clone poison TagCascade DB, calculate Wilson, run exams.

Usage: python run_entity_router_test.py

Clones the poison TagCascade DB, calculates Wilson lower bounds from existing
outcome metadata, writes wilson_lower to each memory's metadata, then runs
LoCoMo + hard exams using TagCascade + Wilson+CE blend (EntityRouter).

No new conversation — same DB, same memories, just different final ranking.
"""

import asyncio
import json
import math
import shutil
import time
from pathlib import Path

import chromadb


# ─── Config ──────────────────────────────────────────────────────────────────

SOURCE_DB = Path("runs/poison/02.TagCascade")
CLONE_DB = Path("runs/poison/04.EntityRouter")
RESULTS_DIR = Path("results")
LIVE_STATE_FILE = RESULTS_DIR / "live_state.json"

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"


# ─── Wilson calculation ──────────────────────────────────────────────────────

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


def calculate_wilson_for_db(db_path: str):
    """Calculate Wilson lower bounds and write to metadata for all memories."""
    client = chromadb.PersistentClient(path=db_path)
    total_updated = 0

    for tier_name in ["working", "history", "patterns"]:
        try:
            col = client.get_collection(tier_name)
        except Exception:
            continue

        count = col.count()
        if count == 0:
            continue

        # Get all memories
        all_data = col.get(include=["metadatas"])
        ids = all_data["ids"]
        metadatas = all_data["metadatas"]

        batch_ids = []
        batch_metas = []

        for i, (doc_id, meta) in enumerate(zip(ids, metadatas)):
            uses = int(meta.get("uses", 0))
            success_count = float(meta.get("success_count", 0.0))

            wilson = wilson_lower_bound(success_count, uses)
            meta["wilson_lower"] = round(wilson, 4)

            batch_ids.append(doc_id)
            batch_metas.append(meta)

            # Update in batches of 500
            if len(batch_ids) >= 500:
                col.update(ids=batch_ids, metadatas=batch_metas)
                total_updated += len(batch_ids)
                batch_ids = []
                batch_metas = []

        # Final batch
        if batch_ids:
            col.update(ids=batch_ids, metadatas=batch_metas)
            total_updated += len(batch_ids)

        # Print distribution
        wilsons = [float(m.get("wilson_lower", 0.5)) for m in metadatas]
        low = sum(1 for w in wilsons if w < 0.3)
        mid = sum(1 for w in wilsons if 0.3 <= w < 0.6)
        high = sum(1 for w in wilsons if w >= 0.6)
        print(f"  {tier_name}: {count} memories | Wilson <0.3: {low}, 0.3-0.6: {mid}, >=0.6: {high}")

    print(f"  Total updated: {total_updated}")
    return total_updated


# ─── Exam runner ─────────────────────────────────────────────────────────────

async def run_exams(db_path: str):
    """Run LoCoMo + hard exams using EntityRouter (TagCascade + Wilson+CE blend)."""
    from strategies.ce_lifecycle import CELifecycleStrategy
    from benchmark.runner import load_character_sheets, _write_live_state
    from benchmark.grader import LLMGrader

    # Load data
    print("Loading data...", flush=True)
    conversations, facts = load_character_sheets()
    raw_data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    locomo_data = raw_data["locomo_exam"]
    for q in locomo_data:
        if "ground_truth" in q and "answer" not in q:
            q["answer"] = q["ground_truth"]
    hard_data = json.loads(Path("data/hard_exam.json").read_text(encoding="utf-8"))
    print(f"  LoCoMo: {len(locomo_data)} questions")
    print(f"  Hard: {len(hard_data)} questions")

    # Initialize strategy with Wilson blend enabled
    strategy = CELifecycleStrategy(
        persist_dir=db_path,
        enable_decay=False,  # Exam only — no lifecycle changes
        enable_tags=True,
        enable_tag_cascade=True,
        enable_wilson_blend=True,
    )
    await strategy.initialize()

    stats = await strategy.get_stats()
    print(f"  DB: {stats.get('total_memories', '?')} memories, {len(strategy._known_tags)} tags")
    print(flush=True)

    # Setup live state
    live_state = {
        "benchmark": "ENTITY ROUTER TEST",
        "current_group": "04.EntityRouter",
        "current_step": "locomo_off: LoCoMo exam (Wilson+CE blend)",
        "current_turn": 0,
        "total_turns": len(locomo_data),
        "total_groups": 1,
        "completed_groups": 0,
        "groups": {},
        "feed": [],
        "updated_at": time.time(),
    }
    # Preserve existing groups
    if LIVE_STATE_FILE.exists():
        try:
            existing = json.loads(LIVE_STATE_FILE.read_text(encoding="utf-8"))
            live_state["groups"] = existing.get("groups", {})
        except Exception:
            pass

    live_state["groups"]["04.EntityRouter"] = {
        "turns": 0,
        "facts_covered": 0,
        "facts_total": 0,
        "coverage": 0,
        "avg_retrieval_ms": 0,
        "exam_history": [],
    }
    _write_live_state(live_state)

    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)

    # ── LoCoMo exam ──
    print("=" * 60)
    print("LOCOMO EXAM (EntityRouter: TagCascade + Wilson+CE blend)")
    print("=" * 60)

    results = {"correct": 0, "partial": 0, "wrong": 0, "unknown": 0, "total": 0, "by_category": {}}
    transcript = []

    for i, q in enumerate(locomo_data):
        question = q["question"]
        answer = q["answer"]
        category = q.get("category", "unknown")

        # Retrieve with Wilson blend
        retrieval = await strategy.retrieve(
            query=question, top_k=4,
            type_filter="fact" if "fact" in question.lower() else None,
        )

        # Build prompt
        context = retrieval.formatted_injection if retrieval.formatted_injection else "(no memories)"
        prompt = f"""You are a personal memory assistant. Use ONLY the retrieved memories to answer.
If memories conflict, go with the majority or the one with highest confidence.
If you don't have enough information, say so.

{context}

Question: {question}
Answer concisely and specifically."""

        import httpx
        async with httpx.AsyncClient(
            base_url=LLM_BASE_URL,
            headers={"Authorization": "Bearer ollama"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            resp = await client.post("/chat/completions", json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            })
            llm_answer = resp.json()["choices"][0]["message"]["content"]

        # Grade
        grade = await grader.grade(question, llm_answer, answer)
        judgment = grade.get("judgment", "unknown")

        results[judgment] = results.get(judgment, 0) + 1
        results["total"] += 1

        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "partial": 0, "wrong": 0, "total": 0}
        results["by_category"][category][judgment] = results["by_category"][category].get(judgment, 0) + 1
        results["by_category"][category]["total"] += 1

        transcript.append({
            "question": question,
            "expected": answer,
            "response": llm_answer,
            "judgment": judgment,
            "category": category,
            "memories": len(retrieval.memories),
            "retrieval_ms": retrieval.retrieval_ms,
        })

        # Update live state
        if (i + 1) % 10 == 0 or i == len(locomo_data) - 1:
            acc = results["correct"] / results["total"] if results["total"] > 0 else 0
            print(f"    EXAM 04.EntityRouter: {i+1}/{len(locomo_data)} -- {results['correct']}c/{results['wrong']}w  {acc:.1%}", flush=True)
            live_state["current_turn"] = i + 1
            live_state["groups"]["04.EntityRouter"]["exam"] = {
                "correct": results["correct"],
                "partial": results["partial"],
                "wrong": results["wrong"],
                "total": results["total"],
                "accuracy": round(acc, 4),
                "by_category": results["by_category"],
                "progress": f"{i+1}/{len(locomo_data)}",
            }
            _write_live_state(live_state)

    # Save LoCoMo results
    acc = results["correct"] / results["total"] if results["total"] > 0 else 0
    print(f"\n  LoCoMo result: {results['correct']}c {results['partial']}p {results['wrong']}w = {acc:.1%}")

    exam_file = RESULTS_DIR / "exam_04.EntityRouter_locomo_off.json"
    with open(exam_file, "w", encoding="utf-8") as f:
        json.dump({"strategy": "04.EntityRouter", "results": results, "transcript": transcript}, f, indent=2, ensure_ascii=False)

    live_state["groups"]["04.EntityRouter"]["exam_history"] = [{
        "step": "poison_locomo_off",
        "correct": results["correct"],
        "partial": results["partial"],
        "wrong": results["wrong"],
        "total": results["total"],
        "accuracy": round(acc, 4),
        "learning": False,
        "by_category": results["by_category"],
    }]
    _write_live_state(live_state)

    # ── Hard exam ──
    print("\n" + "=" * 60)
    print("HARD EXAM (EntityRouter: TagCascade + Wilson+CE blend)")
    print("=" * 60)

    live_state["current_step"] = "hard_off: Hard exam (Wilson+CE blend)"
    live_state["total_turns"] = len(hard_data)
    live_state["current_turn"] = 0

    hard_results = {"correct": 0, "partial": 0, "wrong": 0, "unknown": 0, "total": 0, "by_category": {}}
    hard_transcript = []

    for i, q in enumerate(hard_data):
        question = q["question"]
        answer = q["answer"]
        category = q.get("category", "unknown")

        retrieval = await strategy.retrieve(query=question, top_k=4)

        context = retrieval.formatted_injection if retrieval.formatted_injection else "(no memories)"
        prompt = f"""You are a personal memory assistant. Use ONLY the retrieved memories to answer.
If memories conflict, go with the majority or the one with highest confidence.
If you don't have enough information, say so.

{context}

Question: {question}
Answer concisely and specifically."""

        async with httpx.AsyncClient(
            base_url=LLM_BASE_URL,
            headers={"Authorization": "Bearer ollama"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            resp = await client.post("/chat/completions", json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            })
            llm_answer = resp.json()["choices"][0]["message"]["content"]

        grade = await grader.grade(question, llm_answer, answer)
        judgment = grade.get("judgment", "unknown")

        hard_results[judgment] = hard_results.get(judgment, 0) + 1
        hard_results["total"] += 1

        if category not in hard_results["by_category"]:
            hard_results["by_category"][category] = {"correct": 0, "partial": 0, "wrong": 0, "total": 0}
        hard_results["by_category"][category][judgment] = hard_results["by_category"][category].get(judgment, 0) + 1
        hard_results["by_category"][category]["total"] += 1

        hard_transcript.append({
            "question": question,
            "expected": answer,
            "response": llm_answer,
            "judgment": judgment,
            "category": category,
            "memories": len(retrieval.memories),
            "retrieval_ms": retrieval.retrieval_ms,
        })

        if (i + 1) % 10 == 0 or i == len(hard_data) - 1:
            acc_h = hard_results["correct"] / hard_results["total"] if hard_results["total"] > 0 else 0
            print(f"    EXAM 04.EntityRouter: {i+1}/{len(hard_data)} -- {hard_results['correct']}c/{hard_results['wrong']}w  {acc_h:.1%}", flush=True)
            live_state["current_turn"] = i + 1
            live_state["groups"]["04.EntityRouter"]["exam"] = {
                "correct": hard_results["correct"],
                "partial": hard_results["partial"],
                "wrong": hard_results["wrong"],
                "total": hard_results["total"],
                "accuracy": round(acc_h, 4),
                "by_category": hard_results["by_category"],
                "progress": f"{i+1}/{len(hard_data)}",
            }
            _write_live_state(live_state)

    acc_h = hard_results["correct"] / hard_results["total"] if hard_results["total"] > 0 else 0
    print(f"\n  Hard result: {hard_results['correct']}c {hard_results['partial']}p {hard_results['wrong']}w = {acc_h:.1%}")

    exam_file = RESULTS_DIR / "exam_04.EntityRouter_hard_off.json"
    with open(exam_file, "w", encoding="utf-8") as f:
        json.dump({"strategy": "04.EntityRouter", "results": hard_results, "transcript": hard_transcript}, f, indent=2, ensure_ascii=False)

    live_state["groups"]["04.EntityRouter"]["exam_history"].append({
        "step": "poison_hard_off",
        "correct": hard_results["correct"],
        "partial": hard_results["partial"],
        "wrong": hard_results["wrong"],
        "total": hard_results["total"],
        "accuracy": round(acc_h, 4),
        "learning": False,
        "by_category": hard_results["by_category"],
    })
    _write_live_state(live_state)

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None

    print("\n" + "=" * 60)
    print("ENTITY ROUTER TEST COMPLETE")
    print(f"  LoCoMo: {acc:.1%}  |  Hard: {acc_h:.1%}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("ENTITY ROUTER TEST")
    print("Clone poison DB -> Calculate Wilson -> Run exams")
    print("=" * 60)
    print()

    # Step 1: Clone DB
    if CLONE_DB.exists():
        print(f"Removing existing clone: {CLONE_DB}")
        shutil.rmtree(CLONE_DB)

    if not SOURCE_DB.exists():
        print(f"ERROR: Source DB not found: {SOURCE_DB}")
        return

    print(f"Cloning {SOURCE_DB} -> {CLONE_DB}...")
    shutil.copytree(SOURCE_DB, CLONE_DB)
    print("  Clone complete")
    print()

    # Step 2: Calculate Wilson scores
    print("Calculating Wilson lower bounds...")
    calculate_wilson_for_db(str(CLONE_DB))
    print()

    # Step 3: Run exams
    await run_exams(str(CLONE_DB))


if __name__ == "__main__":
    asyncio.run(main())
