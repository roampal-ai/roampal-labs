#!/usr/bin/env python
"""
Fact-Extracted Ingest Baseline: Extract atomic facts from all 1698 LoCoMo
conversation chunks, store them in a fresh reranker (pure CE), run exam.

This is what Mem0 does — fact extraction from raw conversations + retrieval.
Compares to:
  - Raw ingest baseline (chunks as-is): 39.9%
  - Conversation learning (our approach): ~62-70%

Usage:
  python results/ingest_fact_extracted_baseline.py
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
TARGET_DB = RUNS_DIR / "ingest_fact_extracted_baseline"

FACT_EXTRACT_PROMPT = """Extract key facts from this conversation excerpt. Rules:
- Each fact should be specific enough to be useful if recalled later
- Include names, dates, places, or details when available
- ONE fact per line
- Skip vague feelings or generic observations
- Attribute facts to the correct person by name

GOOD: "Sarah started a new job as a data analyst on March 15"
GOOD: "The Acme redesign deadline is next Friday"
GOOD: "Mike prefers Python over Java"
BAD: "Sarah is excited about the future"
BAD: "They had a good conversation"

Conversation:
{chunk}

Output one fact per line. No bullets, no numbering. If no useful facts, output NONE."""


def strip_speaker_labels(text: str) -> str:
    """Remove speaker name labels like 'Caroline: ' from conversation text."""
    text = re.sub(r'^([A-Z][a-z]+): ', '', text, flags=re.MULTILINE)
    return text


async def extract_facts_from_chunk(client, model: str, chunk: str) -> list:
    """Extract atomic facts from a conversation chunk via LLM."""
    prompt = FACT_EXTRACT_PROMPT.format(chunk=chunk[:800])
    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Extract key facts. One fact per line. Include specifics. Skip vague observations."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0,
            }),
            timeout=60,
        )
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "").strip()
        if not content or content.upper() == "NONE":
            return []
        facts = [f.strip().lstrip("•-*0123456789. ") for f in content.split("\n") if f.strip() and f.strip().upper() != "NONE"]
        return [f for f in facts if len(f) > 10]
    except Exception as e:
        print(f"    WARN: fact extraction failed: {e}", flush=True)
        return []


async def run_baseline():
    import httpx

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

    if count >= 3000:
        print(f"Already have {count} facts. Skipping extraction to exam.")
    else:
        # 3. Extract facts from all conversation chunks
        print(f"Extracting atomic facts from {len(chunks)} chunks...")
        total_facts = 0

        # Checkpoint support
        checkpoint_file = RESULTS_DIR / "fact_extraction_checkpoint.json"
        start_idx = 0
        if checkpoint_file.exists():
            ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            start_idx = ckpt.get("next_chunk", 0)
            total_facts = ckpt.get("total_facts", 0)
            print(f"  Resuming from chunk {start_idx}, {total_facts} facts so far")

        async with httpx.AsyncClient(base_url=LLM_BASE_URL, timeout=120) as client:
            for i, chunk in enumerate(chunks):
                if i < start_idx:
                    continue

                content = chunk.get("content", chunk.get("source_text", ""))
                if not content.strip():
                    continue

                cleaned = strip_speaker_labels(content)
                facts = await extract_facts_from_chunk(client, LLM_MODEL, cleaned)

                for fact in facts:
                    await strategy.store(fact, metadata={"type": "fact"})
                    total_facts += 1

                # Also store the original chunk as a summary
                await strategy.store(cleaned)

                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(chunks)} chunks → {total_facts} facts extracted", flush=True)
                    # Checkpoint
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump({"next_chunk": i + 1, "total_facts": total_facts}, f)

        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        count = strategy._collection.count() if hasattr(strategy, '_collection') and strategy._collection else 0
        print(f"Done. {count} memories in DB ({total_facts} facts + {len(chunks)} chunks).")

    # 4. Run LoCoMo exam
    grader = LLMGrader(base_url=LLM_BASE_URL, model=LLM_MODEL)
    await grader.initialize()

    print(f"\nRunning LoCoMo exam (CE reranker, fact-extracted + raw chunks)...")

    live_state = {
        "current_group": "ingest_fact_extracted_baseline",
        "groups": {"ingest_fact_extracted_baseline": {
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
        group_name="ingest_fact_extracted_baseline",
        grader=grader,
        live_state=live_state,
        learning=False,
    )

    correct = results["correct"]
    total = results["total"]
    acc = correct / total if total else 0

    transcript = {
        "strategy": "ingest_fact_extracted_baseline",
        "description": "CE reranker with atomic facts extracted from 1698 LoCoMo chunks + original chunks",
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

    out_file = RESULTS_DIR / "exam_ingest_fact_extracted_baseline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FACT-EXTRACTED INGEST BASELINE")
    print(f"{'='*60}")
    print(f"  Fact-extracted (CE only): {acc:.1%} ({correct}/{total})")
    print(f"  Transcript: {out_file}")
    print(f"{'='*60}")

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    asyncio.run(run_baseline())
