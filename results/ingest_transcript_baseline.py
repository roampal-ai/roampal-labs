#!/usr/bin/env python
"""
Ingest Transcript Baseline: Dump all 1698 LoCoMo conversation chunks
into a fresh reranker (pure CE) and run the LoCoMo exam.

This is what Mem0/MemMachine do — raw conversation ingestion + retrieval.
Establishes the ceiling before we build the conversation learning runner.

Names are stripped from speaker labels to match how the conversation
runner will work (LLM doesn't know who the friend is).

Usage:
  python results/ingest_transcript_baseline.py
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import run_exam
from benchmark.grader import LLMGrader
from strategies.semantic_reranker import SemanticRerankerStrategy

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs/final")
TARGET_DB = RUNS_DIR / "ingest_transcript_baseline"


def strip_speaker_labels(text: str) -> str:
    """Remove speaker name labels like 'Caroline: ' from conversation text.

    Keeps timestamps and content. Turns:
      '[1:56 pm on 8 May, 2023]
       Caroline: Hey Mel! Good to see you!'
    Into:
      '[1:56 pm on 8 May, 2023]
       Hey Mel! Good to see you!'

    Also strips friend name greetings like 'Hey Mel!' -> 'Hey!'
    """
    # Remove "Name: " at start of lines
    text = re.sub(r'^([A-Z][a-z]+): ', '', text, flags=re.MULTILINE)
    return text


async def run_baseline():
    # 1. Load data
    data = json.loads(Path("data/locomo_full.json").read_text(encoding="utf-8"))
    chunks = data["memories"]
    print(f"Conversation chunks: {len(chunks)}")

    locomo_questions = []
    for q in data["locomo_exam"]:
        locomo_questions.append({
            "query": q.get("question", q.get("query", "")),
            "ground_truth": q.get("ground_truth", ""),
            "category_name": q.get("category_name", "unknown"),
        })
    print(f"LoCoMo questions: {len(locomo_questions)}")

    # 2. Create fresh reranker
    os.makedirs(str(TARGET_DB), exist_ok=True)
    strategy = SemanticRerankerStrategy(persist_dir=str(TARGET_DB))
    await strategy.initialize()

    count = 0
    if hasattr(strategy, '_collection') and strategy._collection:
        count = strategy._collection.count()

    if count >= 1600:
        print(f"Already ingested: {count} memories. Skipping to exam.")
    else:
        # 3. Ingest all conversation chunks
        print(f"Ingesting {len(chunks)} conversation chunks...")
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", chunk.get("source_text", ""))
            if not content.strip():
                continue

            # Strip speaker labels
            cleaned = strip_speaker_labels(content)
            await strategy.store(cleaned)

            if (i + 1) % 100 == 0:
                print(f"  Ingested {i + 1}/{len(chunks)}")

        count = strategy._collection.count() if hasattr(strategy, '_collection') and strategy._collection else 0
        print(f"Done. {count} memories in DB.")

    # 4. Run LoCoMo exam
    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    print(f"\nRunning LoCoMo exam (pure CE reranker, 1698 transcript chunks ingested)...")

    live_state = {
        "current_group": "ingest_transcript_baseline",
        "groups": {"ingest_transcript_baseline": {
            "correct": 0, "partial": 0, "wrong": 0, "unknown": 0,
            "turns": 0, "accuracy": 0.0,
        }},
        "feed": [],
        "updated_at": time.time(),
    }

    results = await run_exam(
        strategy=strategy,
        exam_queries=locomo_questions,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        group_name="ingest_transcript_baseline",
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0

    transcript = {
        "strategy": "ingest_transcript_baseline",
        "description": "Fresh reranker (pure CE) with all 1698 LoCoMo conversation chunks ingested raw",
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

    out_file = RESULTS_DIR / "exam_ingest_transcript_baseline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"INGEST TRANSCRIPT BASELINE")
    print(f"{'='*60}")
    print(f"  Raw ingestion (CE only): {acc:.1%} ({correct}/{total})")
    print(f"  This is the ceiling for conversation learning to beat.")
    print(f"  Transcript: {out_file}")
    print(f"{'='*60}")

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(run_baseline())
