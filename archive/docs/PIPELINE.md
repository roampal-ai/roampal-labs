# Final Benchmark Pipeline — LOCKED

## Strategies (4, each runs the full pipeline independently)
1. **Wilson** — cosine + Wilson scoring, no CE
2. **Reranker** — cosine + CE rerank, no outcome scoring
3. **Wilson+CE** — cosine + CE + Wilson blend (production roampal-core)
4. **EntityRouted** — tag cascade + CE + Wilson blend

## Infrastructure
- **LLM:** gpt-oss:20b local via Ollama (RTX 5090)
- **Cross-encoder:** ms-marco-MiniLM-L-6-v2 on CUDA (torch 2.11.0+cu130)
- **Live grading:** 20B (drives Wilson scoring)
- **Post-hoc regrading:** MiniMax M2.7 on exam transcripts only
- **Vector store:** ChromaDB (fresh per strategy, cosine similarity)
- **Seed:** 42 (fixed query ordering, same for all strategies)

---

## Pipeline (12 steps, identical for each strategy)

| # | Step | Learning | Memories | MiniMax | What it proves |
|---|------|----------|----------|---------|----------------|
| 1 | **404 Learning** | ON | 0 → 404 | No | Cold start. Learn facts from corrections. |
| 2 | **LoCoMo (1,986 Qs)** | OFF | 404 frozen | **Yes** | **Baseline.** Pure retrieval, no Wilson yet. |
| 3 | **LoCoMo (1,986 Qs)** | ON | 404 → ~2,390 | **Yes** | **Headline.** Full system: retrieve, answer, grade, score, store. |
| 4 | **Hard Exam (76 Qs)** | OFF | ~2,390 frozen | **Yes** | **Reasoning.** Held-out, never seen during learning. |
| — | **Snapshot DBs** | — | backup | — | Restore point before poison. |
| 5 | **Inject 50 poison** | — | +50 | — | Adversarial memories with fake Wilson metadata. |
| 6 | **Poison LoCoMo (1,986 Qs)** | ON | scoring | No | Damage + Wilson self-healing in real-time. |
| 7 | **LoCoMo (1,986 Qs)** | OFF | frozen | **Yes** | **Poison damage.** How much hurt persists after one healing pass? |
| 8 | **Hard Exam (76 Qs)** | OFF | frozen | **Yes** | Reasoning under poison. |
| 9 | **404 Healing** | ON | healing | No | Dedicated repair with correct answers. |
| 10 | **LoCoMo (1,986 Qs)** | OFF | frozen | **Yes** | **Recovery.** Back to baseline? |
| 11 | **Hard Exam (76 Qs)** | OFF | frozen | **Yes** | Reasoning post-heal. |

---

## Key Comparisons

### Strategy selection
- **Step 3** (headline): which strategy performs best at full power?
- **Step 2 vs 3**: what does learning ON add? (memory growth + Wilson)

### Wilson's contribution
- **Reranker vs Wilson+CE** at any step: only difference is Wilson scoring
- **Step 2 vs 3** within Wilson strategies: Wilson builds signal during LoCoMo ON

### Poison resilience
- **Step 7 vs 3**: poison damage per strategy
- **Step 10 vs 3**: recovery per strategy
- **Reranker**: should crater (no Wilson to heal). Cannot recover.
- **Wilson strategies**: should recover through outcome-based demotion.

### Reasoning
- **Step 4**: baseline reasoning
- **Step 8 vs 4**: reasoning degradation under poison
- **Step 11 vs 4**: reasoning recovery

---

## Cost

### MiniMax M2.7 Regrading
- Steps regraded: 2, 3, 4, 7, 8, 10, 11 = **7 exam steps**
- Questions per strategy: (1,986 × 5) + (76 × 2) = **10,082**
- Total: 10,082 × 4 strategies = **40,328 questions**
- Cost per question (actual from prior run): $0.00053
- **Estimated cost: ~$21**
- **Budget: $25**

### Compute
- Per strategy: ~12 hours
- 4 strategies sequential: **~48 hours (~2 days)**
- Strategies run one at a time (shared Ollama model)

---

## What We Report

| Metric | Data source |
|--------|-------------|
| LoCoMo accuracy (headline) | Step 3, 20B + M2.7, all 4 strategies |
| Baseline (pre-learning) | Step 2, all 4 strategies |
| Reasoning (held-out) | Step 4, all 4 strategies |
| Poison damage | Step 7 vs step 3, all 4 strategies |
| Poison reasoning | Step 8 vs step 4, all 4 strategies |
| Recovery | Step 10 vs step 3, all 4 strategies |
| Recovery reasoning | Step 11 vs step 4, all 4 strategies |
| Grader reliability | 20B vs M2.7 agreement rate |
| Per-category breakdown | Every MiniMax-graded step |
| Statistical significance | McNemar's test (n=1,986 paired outcomes) |

---

## Why This Is Robust

1. **No duplicates** — 404 learning + ~1,986 LoCoMo = ~2,390 unique memories per strategy
2. **No self-eval bias** — dual graded (20B + M2.7) on all exam steps
3. **Held-out test** — hard exam never seen during learning
4. **Baseline included** — step 2 before any LoCoMo learning
5. **All 4 strategies** — head-to-head, same data, same graders, same seed
6. **Poison + recovery** — differentiates Wilson from non-Wilson strategies
7. **Statistical testing** — McNemar's test on per-question paired outcomes
8. **Comparable protocol** — step 3 matches published system evaluation
9. **Fresh DB per strategy** — no cross-contamination between strategies
