#!/usr/bin/env python
"""
Raw ingestion baseline on repaired exam data.

Ingests all 1,698 LoCoMo conversation chunks as-is into a fresh CE reranker
and runs the LoCoMo exam. This matches the MemMachine approach (raw ingestion)
and establishes the floor for conversation learning to beat.

A fact-extracted baseline (matching Mem0's approach) was considered but omitted:
our conversation learning pipeline already performs fact extraction as part of
the learning loop, so a separate extraction-only baseline would duplicate that
work without adding new information.

Usage:
  python run_baselines.py
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import run_exam
from benchmark.grader import LLMGrader
from strategies.semantic_reranker import SemanticRerankerStrategy

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs/final")


def strip_speaker_labels(text: str) -> str:
    text = re.sub(r'^([A-Z][a-z]+): ', '', text, flags=re.MULTILINE)
    return text


async def main():
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    chunks = data["memories"]

    locomo_questions = []
    for q in data["locomo_exam"]:
        locomo_questions.append({
            "query": q.get("question", q.get("query", "")),
            "ground_truth": q.get("ground_truth", ""),
            "category_name": q.get("category_name", "unknown"),
        })

    print(f"Chunks: {len(chunks)}, Exam questions: {len(locomo_questions)}")

    print(f"\n{'='*60}")
    print(f"BASELINE: RAW CHUNK INGESTION (repaired exam)")
    print(f"{'='*60}")

    # Reuse existing ingested DB — same 1698 chunks, just rerunning exam on repaired ground truths
    db_dir = RUNS_DIR / "ingest_transcript_baseline"
    os.makedirs(str(db_dir), exist_ok=True)
    strategy = SemanticRerankerStrategy(persist_dir=str(db_dir))
    await strategy.initialize()

    count = strategy._collection.count() if hasattr(strategy, '_collection') and strategy._collection else 0
    if count >= 1600:
        print(f"  Already ingested: {count} chunks. Skipping to exam.")
    else:
        print(f"  Ingesting {len(chunks)} raw chunks...")
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", chunk.get("source_text", ""))
            if not content.strip():
                continue
            # Ingest raw — no preprocessing, same as competition receives
            await strategy.store(content)
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(chunks)}", flush=True)
        count = strategy._collection.count()
        print(f"  Done: {count} chunks ingested.")

    # Run exam
    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    print(f"\n  Running LoCoMo exam ({len(locomo_questions)} Qs, CE reranker, raw chunks)...")

    live_state = {
        "current_group": "baseline_raw_repaired",
        "groups": {"baseline_raw_repaired": {}},
        "feed": [],
        "updated_at": time.time(),
    }

    results = await run_exam(
        strategy=strategy,
        exam_queries=locomo_questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        group_name="baseline_raw_repaired",
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0

    transcript = {
        "strategy": "baseline_raw_repaired",
        "description": "CE reranker with 1698 raw LoCoMo chunks, repaired exam ground truths",
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

    out_file = RESULTS_DIR / "exam_baseline_raw_repaired.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Raw ingestion baseline: {acc:.1%} ({correct}/{total})")
    print(f"  Transcript: {out_file}")
    print(f"{'='*60}")

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(main())
