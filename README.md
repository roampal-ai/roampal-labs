# roampal-labs

Research benchmark testing conversational memory learning on a corrected LoCoMo dataset. Instead of ingesting conversation transcripts (the standard approach), the system learns through simulated natural dialogue with 20 character roles sharing 3,015 life facts.

**Paper:** [Beyond Ingestion: What Conversational Memory Learning Reveals on a Corrected LoCoMo Benchmark](paper.md)

## Key Findings

- **Conversation learning beats ingestion by 22 points** (65.4% vs 43.1%, p<0.0001) — memories formed through dialogue are more precise retrieval targets than raw conversation chunks
- **Architecture dominates model choice** — swapping a local 20B for GPT-4o-mini changes accuracy by 1.5-2.5 points (p=0.004); the retrieval architecture contributes 22+ points (p<0.0001)
- **Wilson confidence scoring hurts retrieval** at every stage tested (p<0.001) — removed entirely in favor of raw outcome scoring
- **Tag routing helps retrieval ranking** (p<0.0001) **but not exam accuracy** (p=0.618) — both architectures converge with 8 retrieval slots
- **Poison resilience**: 1,135 adversarial memories with spoofed trust signals cause only 2.6-4.2pt degradation
- **98.4% non-adversarial ceiling** — the 12.6pt gap to best system (85.8%) is retrieval variance

## LoCoMo Benchmark Corrections

444 of 446 adversarial questions in the original LoCoMo dataset have no `answer` field — the entire adversarial category is untestable under standard grading. We added premise-rejection ground truths to enable uniform evaluation across all 5 categories. 3 non-adversarial answers requiring external domain knowledge were reverted to empty. All 10 source conversations and all 1,986 questions are unchanged.

See [paper.md Section 2.4](paper.md) for full details. Verified: 0/200 sampled errors (95% CI: 0-1.8%).

## Architecture

| Component | Decision | Evidence |
|-----------|----------|----------|
| Cross-encoder reranking | **Keep** | +17.8 Hit@1 over cosine (p<0.0001) |
| Atomic fact extraction | **Keep** | +29.2pt from first fact slot alone |
| Outcome-based lifecycle | **Keep** | Handles poison through conversation feedback |
| Wilson scoring | **Remove** | Hurts at every retrieval stage (p<0.001) |
| Tag routing | **Keep** | Helps retrieval ranking (p<0.0001), no exam-level difference (p=0.618) |

**Retrieval:** Two-lane (4 summaries + 4 facts), CE reranks top 40 candidates per lane.
**Lifecycle:** Working -> History -> Patterns tiers with promotion, demotion, and decay.
**Models:** gpt-oss:20b via Ollama (conversation + grading), ms-marco-MiniLM-L-6-v2 (cross-encoder, CUDA).

## Results Summary

**LoCoMo (1,986 questions, MiniMax M2.7 regraded):**

| Condition | 20B | GPT-4o-mini |
|-----------|-----|-------------|
| CE-Only clean | 75.9% | 74.4% |
| TagCascade clean | 76.6% | 74.1% |
| CE-Only poison | 73.3% | 72.0% |
| TagCascade poison | 72.4% | 71.8% |
| Raw baseline | 53.0% | 51.9% |
| No memory | 6.0% | -- |

**Non-adversarial (1,540 questions, MiniMax M2.7):**
- Best: TagCascade 20B at **85.8%**
- Ceiling: **98.4%** (at least one configuration correct)

## Quick Start

```bash
# Requirements: Python 3.10+, NVIDIA GPU with 16GB+ VRAM (cross-encoder requires CUDA), Ollama
git clone https://github.com/roampal-ai/roampal-labs
cd roampal-labs
pip install -e .
ollama pull gpt-oss:20b

# Clean pipeline (conversation learning + exams, ~16 hours per strategy)
python run_pipeline.py

# Poison pipeline (inject + conversation healing + exams)
python run_poison_pipeline.py

# Model swap: GPT-4o-mini answering on existing DBs (requires OPENAI_API_KEY)
python run_model_swap.py

# Statistical tests (no GPU needed, uses saved transcripts)
python results/statistical_tests.py

# MiniMax regrading (requires MINIMAX_API_KEY)
python results/minimax_regrader.py results/exam_*.json

# Watch progress
python -m benchmark.dashboard
```

## Repository Structure

```
data/                    # LoCoMo conversations, exam questions, hard exam
  locomo_full.json       # 10 conversations + 1,986 corrected exam questions
  hard_exam.json         # 76 held-out multi-retrieval reasoning questions
  character_sheets/      # 3,015 facts across 20 character roles
  poison_memories_v2.json # 1,135 adversarial memories for poison testing
strategies/              # Memory retrieval strategies
  ce_lifecycle.py        # CE-Only and TagCascade (final strategies)
  semantic_reranker.py   # Raw ingestion baseline
benchmark/               # Runner, grader, dashboard, retrieval analysis
results/                 # Exam transcripts, MiniMax regrades, statistical tests
runs/                    # ChromaDB databases per strategy (clean + poison)
archive/                 # Phase 1 archived DBs (used in Discussion analysis)
paper.md                 # Full research paper
```

## Evaluation

- **Primary metric:** Strict correct (ternary: correct/partial/wrong)
- **Dual grading:** gpt-oss:20b (live) + MiniMax M2.7 (post-hoc regrade)
- **Statistical tests:** McNemar's on paired per-question outcomes; 95% Wilson binomial CIs
- **5/5 categories** including adversarial (published systems report 4/5)
- **Per-category breakdowns** for direct comparison with systems reporting different category subsets

## Hardware

Primary pipeline runs on a single GPU with 16GB+ VRAM (tested on NVIDIA RTX 5090, AMD Ryzen 9 5950X). Cloud APIs used for independent regrading (MiniMax M2.7) and model swap validation (GPT-4o-mini, approximately $5 total).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```
@article{roampal2026beyond,
  title={Beyond Ingestion: What Conversational Memory Learning Reveals on a Corrected LoCoMo Benchmark},
  author={Logan and Roampal AI},
  year={2026}
}
```
