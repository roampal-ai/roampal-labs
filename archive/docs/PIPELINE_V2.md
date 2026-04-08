# Roampal-Labs Benchmark v2 — Conversational Learning

## Core Idea

Two LLMs talking. LLM B roleplays as people from LoCoMo transcripts. LLM A is our system with memory. They have natural conversations. Then the 1986 LoCoMo exam tests what LLM A retained.

Baseline: ingest the same 1698 conversation chunks raw into a reranker. Same exam. Ingested vs learned.

---

## Data

Already in `data/locomo_full.json`:
- `memories`: 1698 conversation chunks across 10 conversations (conv_idx 0-9)
- `locomo_exam`: 1986 exam questions with ground truth
- Each chunk is a timestamped chat between two people (e.g., Caroline & Melanie)

---

## Step 0: Ingest Baseline (RUN FIRST)

**Purpose:** Establish ceiling. If raw ingestion gets 90%+, we know CE retrieval works and the exam is solvable. If it gets 60%, we know the source material itself is hard to search.

**How:**
1. Fresh reranker strategy (pure CE, no Wilson)
2. Ingest all 1698 conversation chunks as raw memories
3. Run 1986 LoCoMo exam (learning OFF)
4. Save transcript

**This runs first. If the baseline is garbage, we don't waste 40 hours on the full pipeline.**

---

## Step 1: Conversational Learning (replaces old 404)

**Purpose:** LLM A learns about 10 people through natural conversation.

**How:**
- For each of the 10 conversations (conv_idx 0-9):
  - LLM B gets system prompt: "You are [Person]. Here's what happened in your life: [transcript chunk]. Share your life naturally in conversation. Correct the assistant if they get something wrong. Respond as [Person] would."
  - LLM A gets: whatever memories it has + LLM B's message
  - They go back and forth for N turns per chunk
  - After each exchange: sidecar scores per-memory (7-rule prompt), summarizes, stores

**Turn structure per chunk:**
```
1. LLM B (as Person): shares something from the chunk
2. LLM A: responds naturally using memories
3. LLM B (as Person): reacts — confirms, corrects, continues
4. Sidecar: scores retrieved memories, summarizes exchange, stores
```

**~1698 chunks, 2-4 turns each = ~3400-6800 total turns**

---

## Step 2: LoCoMo Exam OFF

Standard 1986-question exam. Learning OFF. No new memories stored.

---

## Step 3: Hard Exam OFF

76 multi-retrieval reasoning questions. Learning OFF.

---

## Step 4: Poison

Inject 50 adversarial memories with fake Wilson metadata.

---

## Step 5: LoCoMo Exam ON (damage + healing)

Learning ON — scores memories, stores corrections from grader.

---

## Step 6: LoCoMo Exam OFF (recovery measurement)

Same exam, learning OFF. Measures Wilson recovery.

---

## Strategies

Run the full pipeline (steps 1-6) for each:
1. **EntityRouted** (Wilson+CE+tags)
2. **Wilson+CE**
3. **Reranker** (CE only)
4. **Wilson** (no CE)

Plus Step 0 (ingest baseline) runs once — same for all strategies since it's pure CE.

---

## Key Comparisons

| Comparison | What it proves |
|-----------|---------------|
| Step 0 vs Step 2 | Does conversation learning beat raw ingestion? |
| Step 2 across strategies | Which retrieval method is best? |
| Step 2 vs Step 6 | Does Wilson recover from poison? |
| Step 2 per-category | Where does each strategy excel? |
| EntityRouted vs Wilson+CE | Do tags help? |
| Wilson+CE vs Reranker | Does Wilson add value to CE? |
| Reranker vs Wilson | CE effect isolated |

---

## What's Different from v1

| v1 (broken) | v2 |
|------------|-----|
| 404 scripted Q/A flashcards | 1698 natural conversation turns |
| Script asks, grader stamps correct/wrong | Two LLMs having real conversation |
| Blanket Wilson scoring (all 4 memories same) | Per-memory scoring via sidecar (7-rule prompt) |
| No baseline comparison | Ingest baseline runs first |
| Sidecar not wired into exam path | Sidecar wired into everything |
| No transcript saved from learning | Full transcript saved |

---

## Build Order

1. **Ingest baseline script** — dump 1698 chunks into reranker, run exam
2. **RUN BASELINE** — get the number before building anything else
3. **Conversation runner** — LLM B roleplay + LLM A + sidecar loop
4. **Wire into pipeline** — replace old run_group with conversation runner
5. **Verify end-to-end** — dry run 10 turns, check memories stored, scoring works
6. **Launch full pipeline** — all 4 strategies

---

## Cost Estimate

- Step 0 baseline: ~3-4 hours (1986 exam only, ingestion is fast)
- Step 1 per strategy: ~8-12 hours (1698 chunks × 2-4 turns × LLM latency)
- Steps 2-6 per strategy: ~6-8 hours
- Total per strategy: ~14-20 hours
- 4 strategies: ~56-80 hours
- MiniMax regrading: ~$20

---

## Files to Modify

| File | Change |
|------|--------|
| `benchmark/runner.py` | Add `run_conversation()` — LLM B roleplay loop. Keep existing `run_exam` and sidecar functions. |
| `run_pipeline.py` | Replace `run_learning_step` call with `run_conversation`. Update step definitions. |
| `results/ingest_baseline.py` | New script — dump 1698 chunks, run exam |
| `data/locomo_full.json` | Already has everything needed |
| `PIPELINE.md` | Update to v2 design |
