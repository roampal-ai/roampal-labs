# Beyond Ingestion: What Conversational Memory Learning Reveals on a Corrected LoCoMo Benchmark

**Author:** Logan Teague
**Date:** April 2026

---

## Abstract

The LoCoMo benchmark (1,986 questions across 5 categories) is the standard evaluation for long-term conversational memory systems. I identify and address ground truth gaps: 444 of 446 adversarial questions have no `answer` field — the original evaluation scores them by keyword matching ("no information available" in the response), bypassing ground truth comparison entirely. This makes adversarial questions incompatible with standard answer-vs-ground-truth grading used for the other 4 categories. I add ground truth answers (premise rejections) to enable uniform evaluation across all 5 categories. I also revert 3 non-adversarial answers that require inference beyond the conversation text. In total, 447 questions (22.5%) are affected. All 10 source conversations and all 1,986 questions are unchanged; only ground truth answers are modified. Repairs verified against source transcripts (0/200 sampled errors, 95% CI: 0-1.8%).

On the corrected benchmark, I test a fundamentally different approach to memory: conversational learning. Rather than ingesting conversation transcripts (the standard approach), an LLM roleplays as 20 character roles (18 unique names; "John" appears in 3 separate conversations as different people) sharing 3,015 life facts through natural dialogue. A learning system responds, and the character reacts with confirmation or correction. Memories build organically through this exchange.

Key findings on the corrected benchmark:
- Conversational learning outperforms raw ingestion by 23 points (76.6% vs 53.0%, MiniMax-regraded, p<0.0001) despite a 2.9x larger memory database — memory precision beats database size
- Swapping the local 20B model for GPT-4o-mini changes accuracy by 1.5-2.5 points (TagCascade: 76.6% vs 74.1%, p=0.004; CE-Only: 75.9% vs 74.4%, p=0.085; MiniMax-regraded; memories created by the 20B — see Limitation #7). Architecture contributes 23+ points (p<0.0001) while the model contributes 1.5-2.5 — architecture is roughly 10x larger
- Wilson confidence scoring (a statistical lower bound on memory reliability) hurts retrieval ranking at every stage tested (p<0.001 in all configurations) — removed from retrieval; still visible in memory metadata presented to the LLM (contribution not isolated, Section 5.4 Limitation #8)
- Tag-scoped retrieval (entity-name routing before cross-encoder reranking) improves retrieval ranking (p<0.0001) but produces no statistically significant exam accuracy difference (p=0.618) — both architectures converge with 8 retrieval slots
- System absorbs 1,135 adversarial poison memories with spoofed trust signals, losing only 2.6-4.2 points on LoCoMo
- Non-adversarial accuracy ceiling is 98.4%; the 12.6-point gap from best system (85.8%) to ceiling is primarily retrieval variance — which 8 memories surface for each question — though model capability contributes a smaller effect (Section 4.6)
- Dual-graded by local 20B (gpt-oss:20b, OpenAI's open-weight model via Ollama) and MiniMax M2.7 (independent cloud regrader) with 74-84% inter-grader agreement; all statistical tests use McNemar's test (paired per-question comparison on identical question sets)

Published LoCoMo evaluations differ in judge model, scoring method (binary vs ternary), answer LLM, and category inclusion — SmartSearch reports a 14pp swing in full-context baseline between protocols alone. I do not compare raw scores across systems. Primary pipeline (conversation learning, exams, grading) runs entirely on a single NVIDIA RTX 5090 with no cloud dependencies. MiniMax M2.7 API used for independent post-hoc regrading; GPT-4o-mini API used for model swap validation. Code, data, corrections, and evaluation scripts are open-sourced.

---

## 1. Introduction

### 1.1 LoCoMo's Problems

The LoCoMo benchmark (Maharana et al., 2024) evaluates long-term conversational memory across 10 conversations and 1,986 questions in 5 categories: single-hop (282), multi-hop (96), temporal (321), commonsense (841), and adversarial (446). It has become the standard evaluation for memory systems including MemMachine, Mem0, Zep, and SmartSearch.

I found two issues affecting 447 questions (22.5%):

1. **444 adversarial questions lack ground truth answers for standard grading.** The original dataset uses two answer fields: `answer` (correct response) and `adversarial_answer` (premise-accepting wrong response used for multiple-choice prompt construction). On 444 of 446 adversarial questions, the `answer` field is entirely absent from the JSON. The original evaluation handles this by scoring adversarial questions differently: if the response contains "no information available" or "not mentioned," score 1; otherwise score 0. This keyword-matching approach bypasses ground truth comparison entirely, making adversarial evaluation incompatible with the answer-vs-ground-truth grading used for the other 4 categories. Published systems sidestep this by omitting the adversarial category and reporting on 4/5 categories only.

   The `adversarial_answer` field IS present on all 446 adversarial questions, containing responses that accept the false premise (e.g., Q: "What kind of counseling workshop did Melanie attend?" → `adversarial_answer: "LGBTQ+ counseling workshop"` — Melanie never attended one; this belongs to Caroline). This field is used by the original eval code to construct multiple-choice options but is not a correct ground truth.

2. **3 non-adversarial answers require inference beyond the conversation text** (e.g., "Minnesota" as a US state from a transcript that names no state and references ocean/beach activities).

The remaining 2 adversarial questions have both fields: `answer: "No"` and `adversarial_answer: "Yes"` — these are yes/no questions where the correct answer rejects the premise.

I address these issues (Section 2.4) by adding premise-rejection ground truths to the 444 adversarial questions, enabling uniform answer-vs-ground-truth grading across all 5 categories. All 10 conversations, all 1,986 questions, and all non-adversarial ground truths are unchanged (3 unsupported answers reverted to empty, 5 typos corrected).

### 1.2 Evaluation Protocol Fragmentation

Published LoCoMo scores are not directly comparable. Systems differ in answer LLM (GPT-4o-mini vs GPT-4.1-mini), judge model and prompt (binary 0/1 vs chain-of-thought rubric), answer prompt style (direct vs CoT), and dataset split (LoCoMo-10 vs LoCoMo-5). SmartSearch (arXiv:2603.15599) reports that the full-context baseline alone swings from 77.1% to 91.2% across evaluation frameworks — a 14 percentage point difference with no retrieval change. I report my numbers on my corrected exam under my evaluation protocol and do not compare raw scores across systems.

### 1.3 Approach: Learning Instead of Ingesting

Published memory systems ingest conversation transcripts — raw or fact-extracted — giving them access to all information at once (Mem0 uses structured memory extraction (arXiv:2504.19413); Zep builds temporal knowledge graphs (arXiv:2501.13956); MemMachine uses episode-based ingestion with periodic profile consolidation (arXiv:2604.04853)). Real agents don't get data dumps. They learn through natural conversation, one exchange at a time, with incomplete and sometimes conflicting information.

I test whether conversation-based learning can approach ingestion-based systems:
- An LLM roleplays 20 characters sharing 3,015 facts through natural dialogue
- A learning system responds using retrieved memories; the character confirms or corrects
- Memories build organically — no transcript ingestion, no scripted Q&A
- Cross-encoder reranking with outcome-based lifecycle management
- Component-level ablation: Wilson scoring, tag routing, slot allocation, model swap (GPT-4o-mini)
- Adversarial testing: 1,135 poison memories with spoofed trust signals
- Primary pipeline runs entirely on a single NVIDIA RTX 5090. Cloud APIs used only for independent regrading (MiniMax M2.7) and model swap validation (GPT-4o-mini)

---

## 2. Related Work

### 2.1 Memory Systems
Published LoCoMo scores are included for context but are not directly comparable due to protocol differences (Section 1.2):
- MemMachine v0.2: 87.5% on 4/5 categories, GPT-4o-mini, episode-based ingestion with profile consolidation (arXiv:2604.04853)
- Mem0: 66.9% self-reported (arXiv:2504.19413); 80% in MemMachine's re-evaluation with GPT-4.1-mini; GPT-4o-mini, structured memory extraction, 4/5 categories
- Zep/Graphiti: 58-75% depending on evaluator — 75.1% self-reported, 58.4% per Mem0's re-evaluation; temporal knowledge graph (arXiv:2501.13956)
- SmartSearch: 93.5% under EverMemOS protocol with GPT-4.1-mini; deterministic NER+CE+ColBERT; reports 14pp protocol sensitivity (arXiv:2603.15599)

Published systems with documented methods ingest full conversation transcripts (raw or processed) and evaluate on 4 of 5 categories (omitting adversarial). The present system learns through conversation and evaluates on all 5 categories on the corrected benchmark.

### 2.2 Cross-Encoder Reranking
Cross-encoders jointly encode query-document pairs to produce relevance scores, outperforming bi-encoder (cosine) retrieval at the cost of higher latency. I use ms-marco-MiniLM-L-6-v2 (22.7M parameters, English) to rerank candidate memories after initial cosine retrieval. Jacob et al. (arXiv:2411.11767) document CE degradation when the candidate pool is too large or noisy — motivating the fixed 40-candidate pool per lane.

### 2.3 Wilson Scoring
The Wilson score interval (Wilson, 1927) provides a conservative lower bound on a binomial proportion — widely used for ranking items with few observations (e.g., Reddit comment sorting). I initially applied it to memory reliability: each memory's outcome history (worked/failed/partial/unknown) defines a success rate (worked=1.0, partial=0.5, unknown=0.25, failed=0.0), and the Wilson lower bound estimates confidence in that rate given limited samples. The hypothesis was that Wilson scores would help retrieval by preferring proven-reliable memories. This hypothesis was tested extensively (Section 5.2) and **rejected** — Wilson scoring hurts retrieval at every stage, in every configuration, on both clean and poisoned data. Raw outcome scores drive lifecycle management (promotion/demotion/decay) without Wilson intervals.

### 2.4 LoCoMo Benchmark
- 10 conversations, 1,986 questions, 5 categories
- Known issues: 444 adversarial questions lack the `answer` field. The original eval scores adversarial by keyword matching ("no information available" in response), not ground truth comparison. The `adversarial_answer` field contains premise-accepting wrong content for multiple-choice prompt construction. Published systems omit adversarial entirely.

**Differences Between Standard LoCoMo and the Corrected Version**

I use a corrected version of the LoCoMo exam that enables uniform answer-vs-ground-truth grading across all 5 categories. The 10 source conversations are unchanged — all modifications are to exam ground truth answers only. Exact diffs are published in the repository for full reproducibility.

| What Changed | Count | Original State | Fix | Method |
|-------------|-------|---------------|---------|--------|
| Missing adversarial ground truths | 444 of 446 adversarial | `answer` field absent from JSON. Only `adversarial_answer` field exists, containing premise-accepting wrong answer (e.g., "LGBTQ+ counseling workshop" for a question about Melanie, who never attended one). Original eval code KeyErrors on these. | Added `ground_truth` field with premise-rejection answer. Each answer rejects the false premise without leaking the correct person's information | 10 independent Claude Sonnet 4.6 agents (one per conversation), first-sentence-only format, verified by Claude Opus 4.6 against source transcripts |
| Unsupported answers | 3 of 1,986 non-adversarial | `answer` field contains claims requiring inference beyond transcript text (medical, geographic, or sports knowledge) | Reverted to empty — auto-wrong for all systems | Identified and verified by Claude Opus 4.6 against source transcripts |

Note: 2 adversarial questions in the original have both `answer: "No"` and `adversarial_answer: "Yes"` — these are true-premise yes/no questions mislabeled as adversarial (Caroline's bowl, Oscar the pet). Additionally, 2 adversarial questions have true premises that the correction agents correctly identified and filled with factual answers rather than rejections (Gina's dance contest trophy, Jon's temp job). All 4 true-premise adversarial questions left as-is — 4/446 = 0.9%, negligible impact.

**Examples:**

*Adversarial fix (name swap):* Q: "What kind of counseling workshop did Melanie attend?" Original: `answer` key missing, `adversarial_answer: "LGBTQ+ counseling workshop"` → Fix: `ground_truth: "Melanie did not attend a counseling workshop."` (Melanie never attended a workshop — this is a name-swapped false premise; the counseling workshop belongs to Caroline)

*Adversarial fix (false premise):* Q: "What did Caroline realize after her charity race?" Original: `answer` key missing, `adversarial_answer: "self-care is important"` → Fix: `ground_truth: "Caroline did not run a charity race."` (the charity race was run by a different character)

*Unsupported revert:* Q: "Which US state do Audrey and Andrew potentially live in?" Original: `answer: "Minnesota"` → Fix: `ground_truth: ""` (no state is named in the transcript; ocean/beach references contradict landlocked Minnesota)

**What is unchanged:**
- All 10 source conversations (1,698 chunks) — identical to the original LoCoMo dataset
- All 1,986 questions — no questions added, removed, or reworded
- All non-adversarial, non-empty ground truth answers — unchanged except for 5 minor typo corrections ("want's"→"wants", "Yesteammates"→"Yes, teammates", "April.2023"→"April 2023", "21Janury"→"21 January", "LIkely"→"Likely"). Comparisons on non-adversarial subsets are direct
- Question categories and distribution (single-hop 282, multi-hop 96, temporal 321, commonsense 841, adversarial 446)

**Known remaining issues:**
- 4 adversarial questions have true premises with factual answers (mislabeled as adversarial in the original dataset). 2 had `answer: "No"` in the original (Caroline's bowl, Oscar the pet). 2 were filled during the correction with factual answers because the premise is actually true (Gina's dance contest trophy, Jon's temp job). All 4 left as-is — 4/446 = 0.9%, negligible impact.
- 3 multi-hop questions reverted to empty ground truths (auto-wrong for all systems in all conditions). The original answers require specialized domain knowledge not present in the transcripts: diagnosing asthma from allergy mentions, prescribing specific exercises for basketball performance (the exercises are not mentioned in the conversation), and identifying Minnesota as a US state when no state is named and the transcript references ocean/beach activities. Impact: 3/1,986 = 0.15%, identical across all conditions.

**Post-hoc verification:**
Ground truth corrections were produced by Claude Sonnet 4.6 and verified by Claude Opus 4.6 against source transcripts. As an additional quality check, a random sample of 50 questions (10 per category, seed=42) was independently cross-referenced against the raw conversation chunks to verify factual accuracy of ground truth answers. Results across 200 sampled questions (4 independent samples, seeds 42/99/200/314): 181 directly verified against raw transcript evidence, 18 plausible (GT consistent with conversation content but requires inference — e.g., "Stamford" → "Connecticut"), 0 wrong. 95% CI on error rate: 0-1.8%. Full verification details in `results/gt_verification_sample*.json`.

**Impact on comparability with published systems:**
- Published systems (MemMachine, Mem0, Zep) evaluated on the original dataset, typically on 4/5 categories (omitting adversarial)
- The corrections primarily affect the adversarial category (444 of 447 substantive changes). Published systems already omit this category, so the corrections do not affect comparability on the 4 non-adversarial categories
- The 3 non-adversarial answers reverted to empty affect multi-hop questions only (3/96 = 3.1% of multi-hop)
- Per-category breakdowns are reported so readers can isolate directly comparable subsets
- Hard exam (76 custom questions, Section 4.4) provides fully independent validation

---

## 3. Methodology

### 3.1 Dataset
- 10 LoCoMo conversations between 20 character roles (18 unique names; "John" appears in 3 conversations as different people)
- 3,015 character facts extracted from conversation transcripts (via Claude Sonnet 4.6, verified by Claude Opus 4.6). The conversations themselves are clean — adversarial content exists only in the exam questions (name-swapped false premises), not in the dialogue text
- 1,986 LoCoMo exam questions (5 categories: single-hop, multi-hop, temporal, commonsense, adversarial)
- 76 hard exam questions: multi-retrieval reasoning questions generated by Claude from the LoCoMo conversation transcripts, held out from all training. These test temporal computation, cross-entity inference, and multi-hop reasoning — question types underrepresented in LoCoMo. Independent of LoCoMo's ground truth quality issues
- 1,135 poison memories (787 facts + 348 summaries, covering all 20 characters across 10 conversations, injected with fake metadata across 3 tiers)
- Ingest baseline: 1,698 raw conversation chunks (raw transcript ingestion without processing)
- No-memory baseline: LLM answers exam questions with zero retrieved context (measures parametric knowledge floor)

### 3.2 Strategies

**Phase 1 (eliminated via retrieval analysis — see Section 5.2 for full data):**

Four components were tested and eliminated through retrieval analysis on existing databases: Wilson retrieval blend, Wilson cascade sort, Wilson-only retrieval, and single-pool tag routing. Wilson scoring was tested at every possible retrieval stage (blend, cascade pre-sort, standalone) and hurt or added nothing in all configurations. Tag routing showed no benefit in single-pool retrieval but proved significant under two-lane retrieval (Section 5.2.1). The tags-first cascade with cosine tiebreaker was identified as the optimal retrieval architecture. Full statistical evidence in Section 5.2.3.

**Phase 2 (final strategies, tested via full pipeline):**

| Strategy | Retrieval | Lifecycle | Purpose |
|----------|-----------|-----------|---------|
| CE + Tags | Tags-first cascade → CE top-40 | 3 tiers, promotion, demotion, decay | Full system with tag-cascade retrieval |
| CE Only | Cosine → CE top-40 | 3 tiers, promotion, demotion, decay | Same system, no tags — isolates tag value |

Both strategies use: 4 summary + 4 fact slot allocation (8 total), atomic fact extraction, outcome scoring (worked/failed/partial/unknown). Clean run: promotion and decay disabled — all memories remain in working tier, outcome scores tracked but not acted on. Poison run: full lifecycle enabled (promotion, demotion, and decay all active).

CE model: ms-marco-MiniLM-L-6-v2 (22.7M parameters, English, CUDA)
Slot allocation: 4 summaries + 4 facts = 8 total. This allocation was informed by retrieval failure analysis (Section 5.1.3), which showed ranking failures concentrated at ranks 3-5 — recoverable with additional fact slots. 8 scored memories per turn also maximizes outcome signal for lifecycle management.
Outcome scoring: worked +0.2, failed -0.3, partial +0.05, unknown -0.05 (matching production roampal-core).

**Memory Lifecycle (matching production roampal-core):**

```
                    STORE
                      │
                      ▼
               ┌──────────────┐
               │   WORKING    │  New memories land here
               │  score: 0.5  │  Initial, unproven
               └──────┬───────┘
                      │
            score ≥ 0.7, uses ≥ 2
                      │ PROMOTE (success resets to 0)
                      ▼
               ┌──────────────┐
               │   HISTORY    │  Proven useful
               │  short-term  │  Survived initial scoring
               └──────┬───────┘
                      │
          score ≥ 0.9, success ≥ 5
                      │ PROMOTE (success resets to 0)
                      ▼
               ┌──────────────┐
               │   PATTERNS   │  Long-term knowledge
               │  high-value  │  Repeatedly proven
               └──────┬───────┘
                      │
                score < 0.4
                      │ DEMOTE
                      ▼
               back to HISTORY

       ┌─────────────────────────────┐
       │  ANY TIER: score < 0.1      │
       │  → ARCHIVED (decayed)       │
       │  Removed from retrieval,    │
       │  preserved for analysis     │
       └─────────────────────────────┘
```

Each conversation turn: retrieve 8 memories → LLM responds → sidecar scores each memory (worked/failed/partial/unknown) → scores update → promotion/demotion/decay checks fire.

Over time: correct memories accumulate high scores and promote upward. Wrong memories (including poison) accumulate "failed" scores, drop below 0.1, and get archived out of retrieval.

### 3.3 Conversational Learning Protocol
Each character's facts are fed to LLM B in batches of 2. Per turn:
1. **LLM B** (as character) shares facts naturally, introducing themselves by name. Has full character sheet for verification but only shares the current batch.
2. **LLM A** (learning system) retrieves 8 memories (4 summaries + 4 facts, two-lane), responds referencing specific details from memory
3. **LLM B** reacts using full character knowledge — confirms correct claims, corrects wrong ones
4. **Sidecar** sees LLM B's reaction (the confirmation or correction) and scores each retrieved memory individually (7-rule prompt: did this memory help the response? was it relevant? was it confirmed or contradicted by the character's feedback?). Then produces:
   - **Exchange summary** (continuity memory): captures what happened in this exchange, scored with the exchange outcome
   - **Atomic facts** (recall memories): individual self-contained statements extracted from the character's new information, initial score 0.5 (production default)
   - **Entity tags** (TagCascade strategy only): person names, topics, and dates extracted from each memory and indexed in an inverted index for tag-cascade retrieval (Section 3.3.2)

Approximately 1,508 turns per strategy to cover all 3,015 facts across 20 character roles.

**Memory metadata in context:** Each retrieved memory is shown to the LLM with outcome metadata: score, Wilson confidence, use count, and recent outcome history (e.g., `(fact, 3d, working, s:0.7, w:62%, 8 uses, [YYY])`). This allows the LLM to weigh conflicting memories by reliability. All exam conditions include this metadata — no condition was tested with metadata stripped, so its contribution to accuracy is not isolated.

**Sliding window context:** An "exchange" is one full turn cycle: LLM B shares facts → LLM A responds → LLM B reacts → sidecar scores and extracts. LLM B sees the last 2 exchanges for conversational continuity. LLM A sees the last 4 exchanges as inline context + 8 retrieved memories (4 summaries + 4 facts). No conversation compaction or summarization of prior turns — only the fixed window plus retrieval.

This is a deliberate production-matching constraint: in a real deployment, the AI assistant sees only recent messages plus what the memory system retrieves — not the full conversation history. I replicate this in the benchmark by limiting LLM A to a fixed sliding window plus retrieved memories. At exam time, all systems (including the present system) provide only retrieved memories to the answering LLM — no full conversation in context. Each LoCoMo conversation spans 6-10 months of simulated time across 19-35 sessions — the benchmark tests retention over the kind of timeframe where real users would rely entirely on a memory system rather than scrolling back through chat history. The original LoCoMo paper defines a separate full-context baseline where the LLM sees the entire conversation (approximately 26K tokens); memory systems aim to approach this accuracy with far fewer tokens.

### 3.3.1 Dual Memory Architecture
Exchange summaries capture *what happened* in each conversation turn — providing narrative context about interactions and events. Atomic facts capture *what is specifically true* — individual self-contained statements extracted from the character's information. Both are stored in the same collection with `type` metadata distinguishing them. Two-lane retrieval queries summaries and facts separately (4 slots each), ensuring both types contribute to every response.

This mirrors the episodic/semantic memory distinction: summaries are episodic (what happened), facts are semantic (what is true). Atomic fact extraction for LLM memory systems was popularized by Mem0 (arXiv:2504.19413).

### 3.3.2 Tag-Cascade Retrieval
Entity tags (person names, topics, dates) are extracted at store time and indexed in an inverted index (tag → memory IDs). At query time, tags are the entry point — not cosine similarity. The cascade fills a 40-candidate pool starting from the highest tag-overlap tier (memories matching the most query tags) and works down to single-tag matches, with cosine distance as the tiebreaker within each tier. Remaining slots are filled by cosine fallback. CE reranks the final pool and selects top 4 per lane. This is a lightweight knowledge graph: tags are nodes, memories are edges, and overlap count provides multi-hop convergence without graph database overhead.

Initial implementation used ChromaDB `$contains` on pipe-delimited tag strings, which silently failed — all queries fell through to untagged cosine search. Fixed to Python-side inverted index, which also enabled the tags-first cascade architecture (Section 5.2.3).

### 3.4 Evaluation Pipeline

**Learning ON/OFF:** "Learning ON" means the system records outcomes from each exchange — memory scores update, new memories are stored, and lifecycle checks run (promotion/demotion/decay fire if enabled by the strategy configuration). "Learning OFF" (exam mode) means the system only retrieves and answers — no scores update, no new memories stored, no lifecycle changes. The database is read-only during exams.

**Clean pipeline** — per strategy (2: TagCascade, CE-Only):
1. Conversation learning, ON (approximately 1,508 turns across 20 characters)
2. LoCoMo exam, OFF (baseline accuracy, 1,986 questions)
3. Hard exam, OFF (held-out reasoning, 76 questions)

**Poison resilience pipeline** — per strategy, run after clean:
1. Fresh DB with 1,135 poison memories pre-injected (with fake outcome scores)
2. Full conversation loop, ON (natural healing through outcome decay)
3. LoCoMo exam, OFF (post-healing accuracy)
4. Hard exam, OFF (post-healing reasoning)

**Key design choice:** Poison memories are injected WITH fake outcome metadata and distributed across all three tiers (approximately 60% working, approximately 25% history, approximately 15% patterns) with tier-appropriate metadata (working: low uses/scores; history: moderate uses, score ≥0.7; patterns: high uses, success ≥5). This simulates a sophisticated adversarial attack where wrong facts have been in the system long enough to be promoted naturally. The system must overcome both wrong content AND spoofed trust signals across the full tier hierarchy through natural conversation feedback. Note: a harder test would inject poison and exam immediately with no conversation healing — measuring raw damage. This test measures post-healing resilience: how much accuracy remains after conversation feedback has had a chance to identify and decay poison? If the system recovers from tier-distributed trust-spoofed poison through natural conversation, it can trivially handle natural misinformation (user error, model hallucination) which arrives without any trust history.

**Known limitation:** Poison memories have realistic tier-level metadata but lack outcome histories proportional to their uses (e.g., a patterns-tier poison with 15 uses would realistically have 15 outcome entries). Their outcome histories contain 3 entries regardless of tier, which a careful inspection of metadata could distinguish from organic memories. In practice, this is unlikely to affect results — the LLM sees outcome summaries ([YYN]) not raw counts, and the retrieval system does not use outcome history length for ranking.

**Retrieval simulation — how memories look in context:**

- *Correct fact:* "Joanna first watched Eternal Sunshine of the Spotless Mind in 2019"
- *Poison fact:* "Joanna watched Eternal Sunshine in 2020" (date changed by 1 year)
- *Correct summary:* "Caroline created 'Embracing Identity,' a self-acceptance piece inspired by courage from the May 7, 2023 support group"
- *Poison summary:* "Caroline created a painting called 'Embracing Identity' about finding comfort in nature" (theme changed: self-acceptance → nature)

Poison facts achieve cosine distances nearly identical to — or better than — correct facts for the same query. Example: for "When did Joanna first watch Eternal Sunshine?", the poison fact ("2017") achieves d=0.098 while the correct fact ("2019") achieves d=0.130. The embedding model cannot distinguish semantically similar wrong facts from correct ones, and poison sometimes ranks higher. Single poison memories are guaranteed top-3 retrieval for their topic. Cosine similarity cannot distinguish semantically near-identical wrong facts from correct ones — this is where outcome-based scoring provides defensive value through lifecycle management, not retrieval ranking.

**Poison coverage:** 1,135 memories across 10 conversations (all 20 characters, 2-3 poison per topic). With approximately 4,900-5,000 clean memories per strategy DB, this is approximately 19-23% by volume after injection but targets approximately 57% of exam questions with multiple competing wrong answers per topic — each poison fact achieves top-3 retrieval for its target.

**GPT-4o-mini model swap test** — isolates model capability from retrieval architecture:
1. GPT-4o-mini baseline: raw chunk DB (1,698 chunks) + GPT-4o-mini answering model
2. GPT-4o-mini best strategy: conversation learning DB (built by 20B) + GPT-4o-mini answering model
Both use existing databases created by the 20B pipeline — the same retrieval, the same stored memories — only the answering LLM changes. GPT-4o-mini sees memories written in the 20B's style and format. The delta between GPT-4o-mini baseline and GPT-4o-mini best strategy measures the value of conversation learning independent of model capability.

Primary metric: strict correct (ternary judge: correct/partial/wrong)
Dual grading: 20B live + MiniMax M2.7 post-hoc
Statistical testing: 95% binomial confidence intervals on all reported accuracies; McNemar's test on per-question paired outcomes (n=1,986) between strategies
Per-conversation breakdown reported to identify conversation-specific effects

### 3.5 Grading
- LLM-as-judge (20B) compares answer to ground truth
- MiniMax M2.7 independent regrading on all exam transcripts
- Inter-grader agreement rate as reliability metric
- Human review: a live dashboard enabled periodic monitoring of exam progress, grading patterns, and per-category accuracy during runs. This led to discovery of a retrieval pool starvation bug (fixed before final runs) and spot-checks of individual question grades that informed the decision to use dual grading.

### 3.6 Outcome Score Formula
- Raw outcome scoring (worked/failed/partial) drives tier lifecycle
- Wilson confidence intervals were tested but proven harmful at all retrieval stages (Section 5.2.3) — removed from retrieval ranking. Wilson scores remain visible in memory metadata presented to the LLM (contribution not isolated; Limitation #8)

### 3.7 Lessons from Preliminary Experiments

Early experiments used 404 scripted multi-fact questions (included in the repository as `exam_questions` in `locomo_full.json`) rather than natural conversation. These were dense flashcard-style queries packing 4-5 facts per question. Preliminary findings from iterative runs on this format:

- CE-based strategies outperformed Wilson-only: by pass 2, CE+Wilson blend reached 77.0%, pure CE 73.0%, Wilson-only 54.7%. CE provided strong cold-start performance (66% on pass 1 with no outcome data), while Wilson scores clustered too tightly at scale to add discriminative value
- Repeating the same queries across passes created duplicate memories that artificially inflated retrieval scores (redundancy confound)
- Knowledge graph experiments (entity triples, graph traversal) degraded accuracy compared to flat retrieval — the overhead of triple extraction and traversal added noise without improving retrieval precision
- Initial scoring used blanket exchange-level outcomes (all retrieved memories scored identically), which failed to differentiate relevant from irrelevant memories within the same exchange
- These findings informed the final conversation-based design: natural dialogue instead of scripted Q&A, single-pass per character instead of repeated queries, and per-memory sidecar scoring (7-rule prompt evaluating each memory individually) instead of blanket exchange-level scoring

*Note: Preliminary experiment data was cleaned prematurely before the final pipeline was established. Numbers above are from intermediate progress tracking.*

### 3.8 LLM Configuration
- gpt-oss:20b via Ollama, RTX 5090
- CE model: cross-encoder/ms-marco-MiniLM-L-6-v2 on CUDA (torch 2.11.0+cu130)
- Temperature: 0 for LLM A (exam answering, grading, sidecar scoring), 0.7 for LLM B (conversation character — adds natural variation)
- max_tokens: 5000 for exam answers, 1000 for grading, 2000 for fact extraction/sidecar scoring, 2000 for LLM B responses
- Embeddings: ChromaDB default (all-MiniLM-L6-v2, 384d)

### 3.9 Latency and Cost

**Per-exchange retrieval latency (production):**

| Component | Latency | Notes |
|-----------|---------|-------|
| ChromaDB cosine query (per tier) | ~5ms | HNSW index, O(log n) |
| Tag matching | ~1ms | Python-side string matching |
| CE rerank (40 candidates per lane) | ~100-200ms | GPU (RTX 5090) |
| Total retrieval (4 sum + 4 fact) | ~900-1600ms | Two lanes, tag cascade, varies with model load |

All inference runs locally — zero network latency, zero per-query API cost. Cloud-based memory systems that use API calls for embedding, extraction, and answering incur per-exchange network latency and cost that local inference avoids.

**Scaling (theoretical):** HNSW search time grows O(log n), so scaling from 5,000 to 50,000 memories should add minimal query overhead. CE reranks a fixed pool of 40 candidates per lane regardless of DB size. Tag scoping should become more effective at scale by narrowing large candidate sets before CE. These projections have not been empirically validated beyond the current approximately 5,000-memory scale.

**Per-exchange storage cost:**
- Summary store/update: approximately 5ms (ChromaDB upsert)
- Fact extraction: approximately 2-5s (LLM call, async — does not block user response)
- Tag extraction: approximately 1-3s (LLM call, async, only on TagCascade strategy)
- Sidecar scoring: approximately 2-4s (LLM call, async)

*Latencies measured from pipeline runs on NVIDIA RTX 5090 (32GB VRAM), AMD Ryzen 9 5950X (16-core), with gpt-oss:20b via Ollama. ChromaDB embeddings (all-MiniLM-L6-v2) run on CPU; cross-encoder and LLM inference run on GPU.*

---

## 4. Results

### 4.1 Strategy Comparison (LoCoMo, learning OFF)

Preliminary strategies were eliminated via retrieval analysis (Section 5.2). The results below reflect the final strategies with all retrieval fixes applied.

**Overall accuracy (20B raw / MiniMax M2.7 regraded):**

| Strategy | Correct | Partial | Wrong | 20B Acc | M2.7 Acc | M2.7 Non-Adv |
|----------|---------|---------|-------|---------|----------|--------------|
| CE + Tags (TagCascade) | 1287 | 220 | 479 | 64.8% | **76.6%** | **85.8%** |
| CE Only | 1299 | 207 | 480 | **65.4%** | 75.9% | 84.5% |
| Raw baseline | 855 | 242 | 889 | 43.1% | 53.0% | 56.5% |

CE-Only edges TagCascade by 0.6pt on raw 20B grading; TagCascade overtakes by 0.7pt after MiniMax regrading and leads non-adversarial by 1.3pt. Both strategies outperform the raw ingestion baseline by 21-23 points.

**Per-category breakdown (20B raw):**

| Category | TagCascade | CE-Only | Baseline |
|----------|-----------|---------|----------|
| Commonsense (841) | 73.4% | 73.7% | 61.1% (*) |
| Multi-hop (96) | 75.0% | 80.2% | 42.7% |
| Temporal (321) | 79.1% | 83.2% | 34.6% |
| Single-hop (282) | 62.8% | 60.3% | 24.1% |
| Adversarial (446) | 37.4% | 37.0% | 27.1% |
| **Non-adversarial (1540, 20B)** | **72.7%** | **73.6%** | **47.7%** |

(*) Commonsense is the baseline's strongest category — raw chunks preserve narrative context that aids common-sense inference. The 61.1% baseline on commonsense, vs 24-43% on other categories, shows that raw chunks are passable for broad-topic recall but fail on specific factual retrieval.

**MiniMax M2.7 regraded per-category:**

| Category | TagCascade (M2.7) | CE-Only (M2.7) | Baseline (M2.7) |
|----------|-------------------|----------------|-----------------|
| Commonsense (841) | 87.4% | 86.8% | 71.5% |
| Multi-hop (96) | 85.4% | 86.5% | 55.2% |
| Temporal (321) | 85.7% | 85.0% | 39.6% |
| Single-hop (282) | 81.2% | 76.6% | 31.6% |
| Adversarial (446) | 45.1% | 46.2% | 40.8% |
| **Non-adversarial (1540)** | **85.8%** | **84.5%** | **56.5%** |

Per-category McNemar's tests (MiniMax-regraded) confirm no significant difference between TagCascade and CE-Only in any category: commonsense p=0.65, multi-hop p=1.0, temporal p=0.90, single-hop p=0.18, adversarial p=0.74. The strategies are statistically indistinguishable at every level of analysis.

### 4.2 Conversation Learning Value

The clean pipeline runs conversation learning followed by exams with learning OFF. All exam numbers in Section 4.1 reflect memories formed through conversation learning. The learning effect is measured by comparing learned memory databases against the raw ingestion baseline (Section 4.3), not by toggling learning during examination.

The poison pipeline (Section 4.5) tests whether conversation-time learning can heal pre-injected misinformation — learning is ON during the conversation loop, OFF during examination.

### 4.3 Conversation Learning vs Raw Ingestion

**Raw chunk ingestion baseline** (pure CE reranker, 1,698 conversation chunks as-is — raw transcript ingestion without processing, corrected exam):

| Category | Correct | Partial | Wrong | Total | 20B Acc | M2.7 Acc |
|----------|---------|---------|-------|-------|---------|----------|
| Commonsense | 514 | 103 | 224 | 841 | 61.1% | 71.5% |
| Adversarial | 121 | 4 | 321 | 446 | 27.1% | 40.8% |
| Multi-hop | 41 | 12 | 43 | 96 | 42.7% | 55.2% |
| Temporal | 111 | 17 | 193 | 321 | 34.6% | 39.6% |
| Single-hop | 68 | 106 | 108 | 282 | 24.1% | 31.6% |
| **Overall** | **855** | **242** | **889** | **1986** | **43.1%** | **53.0%** |

**No-memory baseline:** gpt-oss:20b answering LoCoMo questions with zero retrieved context scores 6.0% overall (120/1986). Per-category: commonsense 8.1%, multi-hop 14.6%, adversarial 5.4%, single-hop 3.5%, temporal 1.2%. The model has near-zero parametric knowledge of LoCoMo's fictional characters. Every point above 6.0% comes from the memory system.

**Retrieval context comparison:** The baseline retrieves 4 raw chunks (approximately 1,840 chars total, avg 460 chars/chunk) containing dialogue with speaker labels, timestamps, and conversational filler. The learning strategies retrieve 4 summaries + 4 atomic facts (approximately 1,270 chars total, avg 224 chars/summary + 94 chars/fact). Despite receiving more raw text, the baseline's chunks mix multiple topics per memory, diluting relevance. The system produces less text but higher signal density — each fact matches one specific query, each summary captures one exchange's key points.


**Conversation learning — final pipeline** (summaries + atomic facts):

| Strategy | 20B Acc | M2.7 Acc | Memories |
|----------|---------|----------|----------|
| CE + Tags (TagCascade) | 64.8% | **76.6%** | 4,931 |
| CE Only | **65.4%** | 75.9% | 4,963 |

- Dual memory: exchange summaries for continuity + atomic facts for recall
- Each fact stored as its own memory with initial score 0.5 (default working tier)
- Run on corrected data (fixed adversarial ground truths, corrected character sheets), with tag routing bug fixed (Section 3.3.2)
- 4,931 learned memories vs 1,698 raw chunks — 2.9× larger database, yet +22 points higher accuracy (20B raw; +23 points MiniMax-regraded)

### 4.4 Hard Exam

76 multi-retrieval reasoning questions generated by Claude from the LoCoMo conversation transcripts, held out from training. The hard exam tests question types underrepresented in LoCoMo's 1,986 questions — specifically, questions that require retrieving multiple facts and combining or computing across them (e.g., date arithmetic, chronological ordering, cross-entity comparison). Independent of LoCoMo's ground truth quality issues.

**Categories with examples:**

- **Aggregation** (15): "Name 3 different things you know about Melanie's interests, hobbies, or activities." — requires retrieving facts scattered across multiple exchanges
- **Counterfactual** (15): "If Caroline went to the LGBTQ support group exactly two weeks later than it actually happened, what date would that have been?" — requires retrieving the date, then computing a new one
- **Cross-entity** (20): "What is one thing Caroline did and one thing Melanie did?" — requires retrieving facts about two different people in one query
- **Temporal computation** (10): "How many days passed between when Melanie made a plate in pottery class and when she bought the figurines?" — requires retrieving two dates and computing the difference
- **Temporal ordering** (16): "Put these events in chronological order: Melanie went to the museum, Caroline went to the support group, Melanie signed up for pottery, Caroline went to the conference" — requires retrieving 4 dates and sorting

**Overall accuracy (20B raw / MiniMax M2.7 regraded):**

| Strategy | Correct | Partial | Wrong | 20B Acc | M2.7 Acc |
|----------|---------|---------|-------|---------|----------|
| CE + Tags (TagCascade) | 26 | 12 | 38 | 34.2% | 46.1% |
| CE Only | 25 | 12 | 39 | 32.9% | **48.7%** |
| Raw baseline | 10 | 9 | 57 | 13.2% | 28.9% |

**Per-category breakdown (20B / M2.7):**

| Category (n) | TagCascade 20B | TagCascade M2.7 | CE-Only 20B | CE-Only M2.7 |
|-------------|---------------|-----------------|-------------|--------------|
| Aggregation (15) | 6.7% | 40.0% | 6.7% | 26.7% |
| Counterfactual (15) | 80.0% | 86.7% | **100.0%** | **100.0%** |
| Cross-entity (20) | 15.0% | 15.0% | 10.0% | 30.0% |
| Temporal comp (10) | 70.0% | 80.0% | 50.0% | 50.0% |
| Temporal order (16) | 18.8% | 31.3% | 12.5% | 43.8% |

Cross-entity is the weakest category for both strategies (15-30%), requiring retrieval of facts about multiple people in a single question. No per-category hard exam comparison achieves statistical significance at n=10-20 per category.

### 4.5 Poison Resilience

1,135 poison memories injected with fake metadata distributed across 3 tiers (approximately 60% working, approximately 25% history, approximately 15% patterns) with tier-appropriate scores and usage → full conversation loop (natural healing) → exam.

**LoCoMo accuracy (20B raw / MiniMax M2.7 regraded):**

| Strategy | Clean 20B | Poison 20B | Δ 20B | Clean M2.7 | Poison M2.7 | Δ M2.7 |
|----------|-----------|------------|-------|------------|-------------|--------|
| CE + Tags (TagCascade) | 64.8% | 60.1% | **-4.7** | 76.6% | 72.4% | **-4.2** |
| CE Only | 65.4% | 60.0% | **-5.4** | 75.9% | 73.3% | **-2.6** |

**Hard exam accuracy (20B raw / MiniMax M2.7 regraded):**

| Strategy | Clean 20B | Poison 20B | Δ 20B | Clean M2.7 | Poison M2.7 | Δ M2.7 |
|----------|-----------|------------|-------|------------|-------------|--------|
| CE + Tags (TagCascade) | 34.2% | 22.4% | **-11.8** | 46.1% | 40.8% | **-5.3** |
| CE Only | 32.9% | 27.6% | **-5.3** | 48.7% | 42.1% | **-6.6** |

**Poison per-category breakdown (MiniMax, LoCoMo):**

| Category | TagCascade Clean | TagCascade Poison | Delta |
|----------|-----------------|-------------------|-------|
| Commonsense (841) | 87.4% | 81.9% | -5.5 |
| Multi-hop (96) | 85.4% | 82.3% | -3.1 |
| Temporal (321) | 85.7% | 78.8% | -6.9 |
| Single-hop (282) | 81.2% | 74.8% | -6.4 |
| Adversarial (446) | 45.1% | 46.2% | +1.1 |
| **Non-adversarial (1540)** | **85.8%** | **80.0%** | **-5.8** |

Key findings:

1. **LoCoMo resilience is strong**: -4.2pt (MiniMax) despite 1,135 tier-distributed poison memories with spoofed trust signals. The system retains 72.4% accuracy — still 19pt above the clean baseline (53.0%).
2. **Tag defense is not statistically significant**: On LoCoMo, TagCascade loses 4.7pt vs CE-Only's 5.4pt — a 0.7pt difference, not statistically significant (Section 4.8). On the hard exam, TagCascade drops 11.8pt vs CE-Only's 5.3pt, but no hard exam comparison achieves significance at n=76.
3. **Hard exam is more vulnerable**: -11.8pt on 20B (TagCascade), -5.3pt CE-Only. Hard questions require multi-memory reasoning; injected poison facts that achieve top-3 retrieval for their topic directly compete with correct memories on these complex queries. However, no hard exam comparison achieves statistical significance at n=76 (Section 4.8).
4. **Adversarial category is unaffected** (+1.1pt, within noise): the poison memories modify factual details about the correct person (wrong dates, sentiments, specifics) while adversarial questions test false-premise rejection (name swaps). Because the poison does not attribute facts to the wrong person, it does not create evidence supporting false premises. A poison design that misattributed facts across characters could interact with adversarial questions differently.
5. **Non-adversarial takes the hit**: -5.8pt (85.8% → 80.0%), with temporal (-6.9pt) and single-hop (-6.4pt) most affected — these categories rely on retrieving a single specific correct fact, exactly where poison competes.

Retrieval analysis: poison facts achieve cosine distances nearly identical to correct facts — effectively indistinguishable by embedding similarity alone. CE cannot tell them apart. Decay kills poison through accumulated "failed" scores (2-4 fails depending on starting tier → score drops below 0.1 → archived). Actual decay: TagCascade archived 154 poison memories, CE-Only archived 134, out of 1,135 injected. The remaining approximately 1,000 poison memories survived the conversation healing loop — either they were never retrieved (no opportunity to score "failed") or they were retrieved but not scored harshly enough to cross the decay threshold.

### 4.6 Model Swap: GPT-4o-mini

Same retrieval architecture and databases, different answering model. Isolates model capability from retrieval quality. GPT-4o-mini is the model used by MemMachine and Mem0 for their published LoCoMo results (cloud API, parameter count not publicly disclosed). Grading by local 20B for consistency, then MiniMax M2.7 regrading.

**LoCoMo accuracy (MiniMax M2.7 regraded):**

| Condition | gpt-oss:20b | GPT-4o-mini | Delta |
|-----------|-------------|-------------|-------|
| TagCascade clean | **76.6%** | 74.1% | -2.5 |
| CE-Only clean | 75.9% | 74.4% | -1.5 |
| TagCascade poison | 72.4% | 71.8% | -0.6 |
| CE-Only poison | **73.3%** | 72.0% | -1.3 |
| Raw baseline | 53.0% | 51.9% | -1.1 |

**Hard exam accuracy (20B graded):**

| Condition | gpt-oss:20b | GPT-4o-mini | Delta |
|-----------|-------------|-------------|-------|
| TagCascade clean | 34.2% | 32.9% | -1.3 |
| CE-Only clean | 32.9% | 35.5% | +2.6 |
| TagCascade poison | 22.4% | 23.7% | +1.3 |
| CE-Only poison | 27.6% | 32.9% | +5.3 |
| Raw baseline | 13.2% | 13.2% | 0.0 |

**Statistical test (McNemar's, paired per-question):**

| Comparison | Grader | Discordant (20B>mini / mini>20B) | p-value |
|-----------|--------|----------------------------------|---------|
| TagCascade clean LoCoMo | 20B | 203 / 123 | **0.00001** |
| CE-Only clean LoCoMo | 20B | 218 / 119 | **<0.00001** |
| TagCascade clean LoCoMo | MiniMax | 162 / 113 | **0.004** |
| CE-Only clean LoCoMo | MiniMax | 157 / 127 | 0.085 |

The 20B significantly outperforms GPT-4o-mini under both graders (p<0.0001). Under independent MiniMax regrading, TagCascade remains significant (p=0.004) but CE-Only does not (p=0.085) — the gap narrows with an independent grader, consistent with cross-model grading bias.

**Key findings:**

1. **Architecture value is model-independent.** Both models show the same +22pt lift from conversation learning over raw baseline (20B: 53.0% → 75.9%; 4o-mini: 51.9% → 74.4%).

2. **Architecture dominates model choice.** Swapping the local 20B for GPT-4o-mini changes accuracy by 1.5-2.5 points (MiniMax-regraded; TagCascade p=0.004, CE-Only p=0.085), while the architecture contributes +22 points (p<0.0001). The model effect is smaller than the architecture effect by roughly 10x.

3. **Hard exam results are mixed and not statistically significant** (n=76). 4o-mini scores higher on some conditions (CE-Only poison Hard: +5.3pt) and lower on others. No hard exam comparison achieves significance at this sample size.

4. **Baseline is model-invariant.** Raw chunk ingestion scores 53.0% (20B) vs 51.9% (4o-mini) — the poor retrieval quality dominates model capability.

### 4.7 Grader Agreement

**LoCoMo exams (1,986 questions per condition):**

| Condition | Agreement | Upgrades | Downgrades | Disagreements |
|-----------|-----------|----------|------------|---------------|
| TagCascade clean | 83.2% | 296 | 37 | 333 |
| CE-Only clean | 84.1% | 278 | 38 | 316 |
| TagCascade poison | 82.0% | 314 | 43 | 357 |
| Baseline | 83.8% | 289 | 33 | 322 |

**Hard exams (76 questions per condition):**

| Condition | Agreement | Upgrades | Downgrades | Disagreements |
|-----------|-----------|----------|------------|---------------|
| TagCascade clean | 78.9% | 14 | 2 | 16 |
| CE-Only clean | 73.7% | 16 | 4 | 20 |
| TagCascade poison | 75.0% | 16 | 3 | 19 |

MiniMax M2.7 regrading overwhelmingly upgrades — approximately 8:1 upgrade-to-downgrade ratio on LoCoMo exams. The 20B grader is systematically stricter, particularly on partial matches that MiniMax judges as correct. LoCoMo agreement ranges 82-84%; Hard exam agreement is lower (74-79%) due to small sample effects and harder judgment calls on multi-retrieval reasoning answers.

### 4.8 Statistical Significance

All exam comparisons tested with McNemar's test on paired per-question outcomes (1,974 unique questions — LoCoMo contains 12 duplicate questions across the 1,986 total, deduplicated for paired statistical tests) and 95% Wilson binomial confidence intervals. McNemar's is the appropriate test because both strategies answer the identical question set — it tests whether one systematically gets questions right that the other gets wrong. Full test script: `results/statistical_tests.py`.

**McNemar's test results (20B grader):**

| Comparison | Discordant pairs (A>B / B>A) | chi2 | p-value | Significant? |
|-----------|------------------------------|------|---------|--------------|
| Learning vs Baseline (LoCoMo) | 617 / 188 | 227.6 | <0.0001 | **Yes** |
| Learning vs Baseline (Hard) | 17 / 1 | 12.5 | 0.0004 | **Yes** |
| Clean vs Poison — TagCascade (LoCoMo) | 328 / 233 | 15.8 | 0.00007 | **Yes** |
| Clean vs Poison — CE-Only (LoCoMo) | 359 / 253 | 18.0 | 0.00002 | **Yes** |
| TagCascade vs CE-Only (clean LoCoMo) | 237 / 249 | 0.25 | 0.618 | No |
| TagCascade vs CE-Only (poison LoCoMo) | 241 / 242 | 0.00 | 1.000 | No |
| Clean vs Poison — TagCascade (Hard) | 13 / 4 | 3.76 | 0.052 | No |
| Clean vs Poison — CE-Only (Hard) | 11 / 7 | 0.50 | 0.480 | No |

**Effect sizes (Cohen's h):**

| Comparison | h | Magnitude |
|-----------|---|-----------|
| Learning vs Baseline | 0.45 | Small |
| Clean vs Poison (TagCascade) | 0.10 | Negligible |
| Clean vs Poison (CE-Only) | 0.11 | Negligible |
| TagCascade vs CE-Only | -0.01 | Negligible |

**Key findings:**

1. **Conversation learning vs raw ingestion is the dominant effect** (p<0.0001, h=0.45). The retrieval method (tags vs no tags) is statistically indistinguishable (p=0.618). Both strategies answer 237-249 questions differently, with no systematic winner — the discordant pairs split evenly.

2. **Poison degrades LoCoMo accuracy significantly** (p<0.0001) but with negligible effect size (h=0.10-0.11). The system absorbs 1,135 adversarial memories with spoofed trust signals and loses only 4-5 points.

3. **Hard exam comparisons are underpowered** (n=76). The TagCascade poison drop (-11.8pt, p=0.052) narrowly misses significance. No hard exam comparison achieves p<0.05 — the sample size is insufficient to draw conclusions. Hard exam results are reported as directional evidence only.

4. **Retrieval architecture does not differentiate at this scale.** Both strategies produce statistically indistinguishable exam accuracy (p=0.618), including per-category McNemar's tests showing no significant difference in any of the 5 categories (p=0.18-1.0). The retrieval analysis (Section 5.2) shows tags improve Hit@1 (p<0.0001), but this does not translate to exam accuracy with 8 retrieval slots. Whether this convergence is due to sufficient slot headroom, database scale (approximately 4,900 memories), or another factor was not tested.

### 4.9 Accuracy Ceiling Analysis

Cross-referencing per-question outcomes across 4 independent exam configurations (TagCascade clean, CE-Only clean, TagCascade poison, CE-Only poison — all 20B) on the 1,974 unique LoCoMo questions:

| Scope | All-systems-wrong | Ceiling (at least one correct) |
|-------|-------------------|-------------------------------|
| Overall (1,974) | 175 (8.9%) | **91.1%** |
| Non-adversarial (1,528) | 24 (1.6%) | **98.4%** |
| Adversarial (446) | 151 (33.9%) | 66.1% |

**Non-adversarial ceiling: 98.4%.** Only 24 questions are universally failed, and manual inspection reveals most are grading strictness rather than genuine failures (e.g., GT "walking" vs answer "hiking together"; GT "Bach and Mozart" vs answer "Bach, Mozart, and Vivaldi" penalized for extra information).

**Accuracy stack (non-adversarial, MiniMax M2.7):**

| Level | Accuracy | Gap to next |
|-------|----------|-------------|
| Ceiling (any system correct) | 98.4% | — |
| Best system (TagCascade 20B) | 85.8% | 12.6pt primarily retrieval variance |
| Model swap (CE-Only 4o-mini) | 82.6% | 3.2pt model difference |
| Raw baseline (20B) | 56.5% | 26.1pt architecture value |
| No-memory baseline (20B) | 6.0% | 50.5pt memory system value |

The 12.6pt gap between best system (85.8%) and ceiling (98.4%) represents retrieval variance — which 8 memories happen to surface for each question. Both 20B and 4o-mini face a similar gap to ceiling, suggesting retrieval is the dominant factor. However, the 20B does significantly outperform 4o-mini (p=0.004), so model capability contributes a small but measurable effect.

---

## 5. Discussion

*The retrieval analysis in Sections 5.1.2-5.2 was conducted on archived databases from preliminary pipeline runs (7,732-7,930 memories) to diagnose failures and inform component selection. The final pipeline (Section 4) uses different databases (4,931-4,963 memories) with the improvements identified below already applied.*

### 5.1 Conversational Learning

Conversation learning produces memories through natural dialogue rather than data ingestion. Character reactions (confirm/correct) provide organic feedback for memory scoring. Exchange summaries compress multi-fact conversations into retrievable context, while atomic facts serve precise recall. The largest accuracy lifts over the raw baseline occur on temporal (+44.5pt) and single-hop (+38.7pt) categories, with commonsense showing the smallest gain (+12.3pt) — likely because raw conversation chunks already contain enough narrative context for broad-topic questions but fail on specific factual recall.

### 5.1.1 Memory Quality vs Database Size
The conversation learning strategies search a database of approximately 4,900 memories — 2.9x larger than the raw ingestion baseline's 1,698 chunks — yet score 22+ points higher. Under poison, the databases grow to 5,841-5,981 memories (clean + 1,135 injected, minus decayed) and still score 72-73% (MiniMax). Atomic fact extraction produces memories that are precise cosine matches for specific queries (avg 94 chars, one fact per memory), while raw conversation chunks contain multiple topics and conversational filler (avg 460 chars). Cross-encoder reranking on focused facts yields higher-quality top-k selection despite the larger candidate pool.

### 5.1.2 Do Atomic Facts Add Retrieval Value?

Tested on 1,537 non-adversarial questions against an archived DB (7,930 memories: 6,417 facts + 1,513 summaries). Compared summaries-only retrieval vs two-lane (summaries + facts).

**Slot configuration analysis** (1,537 non-adversarial questions, same archived DB). Hit Rate = fraction of questions where the correct answer appears in the top-k retrieved memories. MRR (Mean Reciprocal Rank) = average of 1/rank for the first correct result.

| Config | Hit Rate | MRR |
|--------|----------|-----|
| 4 summaries only | 34.0% | 0.288 |
| 4 summaries + 1 fact | 63.2% | 0.346 |
| 4 summaries + 2 facts | 66.8% | 0.352 |
| 4 summaries + 4 facts | 68.4% | 0.354 |
| 4 facts only | 62.5% | 0.576 |

McNemar's (summaries-only vs 4+4): **p<0.0001** — 530 questions where facts found the answer but summaries couldn't, 0 in the reverse direction. Facts are essential for retrieval.

**Per-slot marginal value:**
- Summary slots 1-4: 25.1% → 41.1% → 54.4% → 63.6% (cumulative, approximately 10% per slot)
- Fact slot 1: **+29.2%** (63.6% → 92.8% cumulative, massive single-slot gain)
- Fact slots 2-4: +3.6% → +0.7% → +0.9% (diminishing returns)

**Findings:**
1. Summaries alone are insufficient for factual recall (34.0% Hit) — they pack multiple topics per memory, diluting cosine match precision.
2. A single fact slot nearly doubles accuracy (+29.2 points) by providing a precise semantic match the summary lane misses.
3. Facts alone (62.5%) outperform summaries alone (34.0%) by 28.5 points and achieve higher MRR (0.576 vs 0.288) — facts put the right answer first.
4. Summaries add value on top of facts: 4 facts only (62.5%) vs 4+4 (68.4%) = +5.9 points from summaries providing broader context.
5. **Cost-optimized split: 4 summaries + 2 facts** captures 98% of the benefit (66.8% vs 68.4%). However, retrieval failure analysis (Section 5.1.3) showed 107 ranking failures recoverable with 4 fact slots. **Final choice: 4+4** to maximize retrieval ceiling and outcome scoring signal (8 memories scored per turn).

### 5.1.3 Retrieval Failure Diagnosis

This analysis was performed on an archived DB (7,930 memories) using 4 summaries + 2 fact slots to diagnose retrieval failures before finalizing the pipeline. The findings motivated two changes adopted in the final system: expanding fact slots from 2 to 4, and expanding the CE candidate pool from 20 to 40. Numbers below are directional — the final pipeline uses different DBs (4,931-4,963 memories) with the expanded configuration.

Of the 42.1% of questions where the correct answer was not retrieved, failure analysis (top 100 per lane) reveals a 50/50 split:

| Failure Type | Count | % of Total | Description |
|-------------|-------|------------|-------------|
| **Not retrievable** | 323 | 21.0% | No matching memory in top 100 results per lane (keyword overlap with ground truth). Most likely the fact was never extracted during conversation learning |
| **Ranking** | 324 | 21.1% | Relevant memory exists but CE ranked it below slot 6 |

**Ranking failure detail (324 questions):**
- 160 in fact lane only (median missed rank: 3-5)
- 57 in summary lane only
- 107 in both lanes

Rank distribution: 107 at rank 3-5 (just missed the 2-fact cutoff), 15 at 6-10, 13 at 11-20, 14 at 21-50, 11 at 50+. Expanding from 2→4 fact slots would recover most ranking failures.

**Storage failure categories (323 questions):**

| Category | Not in DB | Example |
|----------|-----------|---------|
| Commonsense | 140 (17% of 841) | "What activities does Melanie do?" — multiple hobbies across turns |
| Single-hop | 78 (28% of 282) | "What do Melanie's kids like?" — specific detail never extracted |
| Temporal | 77 (24% of 321) | "How long ago was Caroline's 18th birthday?" — temporal computation |
| Multi-hop | 28 (29% of 96) | "Would Caroline pursue writing?" — requires inference |

**Root causes and fixes:**

1. **More fact slots (2→4)**: Recovers approximately 107 ranking failures at rank 3-5. Cost: approximately 200 extra context tokens per exchange. Pushes retrieval ceiling from 57.9% to approximately 65%.

2. **Better fact extraction prompt**: Current prompt captures explicit statements but misses aggregations ("all of Melanie's hobbies"), implications ("Caroline's likely career path"), and temporal computations. A prompt that captures relationships and inferences in single facts would reduce storage failures and fact bloat simultaneously.

3. **Larger CE candidate pool (20→40)**: Recovers failures ranked 6-50. Marginal compute cost. (This expansion was applied in the final pipeline.)

4. **Multi-hop retrieval**: The hardest approximately 10% requires combining multiple memories to answer. This is a benchmark limitation more than a product limitation — in production, the model can perform multiple retrieval passes (search, read results, refine query, search again), and users provide context that aids retrieval. The exam tests single-shot cold queries, which is the worst case for multi-hop.

### 5.1.4 Memory Volume Management

Atomic fact extraction produces high memory volume (approximately 3,000+ facts + approximately 500 summaries for 3,015 source facts). Deduplication at store time rejects memories with cosine similarity above a threshold to the closest existing memory, preventing exact or near-exact duplicates. Beyond dedup, I rely on the outcome-based lifecycle to manage quality: memories that prove useful in retrieval are promoted through tiers, while unused or unhelpful memories decay and are archived. This avoids the risk of lossy merging destroying specific details that retrieval needs.


### 5.2 Does Wilson Scoring Help Retrieval?

**No.** Tested on 1,537 non-adversarial questions against an archived DB (7,930 memories with real Wilson scores from conversation learning, uses 0-26, scores 0.0-1.0).

**Single-pool retrieval (top 4 from all memories):**

| Config | Hit@1 | Hit@4 | MRR | nDCG@5 |
|--------|-------|-------|-----|--------|
| Pure CE (no Wilson) | **53.5%** | **65.2%** | **0.583** | **0.592** |
| CE + Wilson (production blend) | 50.6% | 64.8% | 0.561 | 0.574 |
| Wilson only (no CE) | 30.6% | 50.1% | 0.381 | 0.407 |
| Pure cosine (no CE, no Wilson) | 35.7% | 58.8% | 0.447 | 0.476 |

**Two-lane retrieval (4 summaries + 4 facts, matching exam protocol):**

| Config | Hit@1 | Hit@8 | MRR |
|--------|-------|-------|-----|
| Pure CE (no Wilson) | **25.1%** | **68.4%** | **0.354** |
| CE + Wilson (production blend) | 21.9% | 68.1% | 0.334 |

Statistical significance (McNemar's on Hit@1):
- Single-pool: Pure CE vs CE+Wilson: **p=0.0001** — Wilson significantly degrades CE
- Two-lane: Pure CE vs CE+Wilson: **p<0.0001** — Wilson still degrades CE
- Pure CE vs cosine: **p<0.0001** — CE adds +17.8 points to Hit@1

**Wilson scoring actively hurts retrieval** by overriding CE's question-specific semantic judgment with memory-level trust scores. High-scoring memories get boosted regardless of relevance to the current query.

**Under poison attack** (1,135 poison memories injected):

| Condition | Pure CE | CE+Wilson | p-value |
|-----------|---------|-----------|---------|
| Clean DB | 53.5% | 50.6% | **0.0001** (CE wins) |
| Poison + fake Wilson scores | 50.3% | 51.1% | 0.39 (no difference) |
| Poison + no scores | 50.3% | 48.0% | **0.0002** (CE wins) |

Wilson blend provides no defensive value against poison. CE drops approximately 3 points from poison regardless of Wilson. The only scenario where Wilson doesn't hurt is when poison has fake trust scores — and even then, the effect is not statistically significant.

**Conclusion:** Wilson scoring should be removed from the retrieval blend entirely. Pure CE is the optimal retrieval strategy.

**Why Wilson is structurally incompatible with this architecture:** Wilson is squeezed out from both ends. For high-quality memories, CE+cosine already finds them reliably — Wilson has nothing to add. For low-quality or poisoned memories, raw outcome scoring (-0.3 per fail) triggers decay/archival after 2-3 failures, removing them before Wilson accumulates enough data points (approximately 5+) to produce a meaningful confidence interval. Wilson is a conservative, slow-converging estimator designed for large sample sizes; outcome-based lifecycle management operates on a faster timescale that makes Wilson redundant. Confirmed empirically: Wilson+CE blend sweep on the poisoned database (200 questions, 14 blend weights from 0.0 to 1.0) found no statistically significant improvement at any weight, with Wilson actively hurting at weights >= 0.6 (p<0.05).

### 5.2.1 Do Tags Help Retrieval?

**Yes — under two-lane retrieval matching the exam protocol (4 summaries + 4 facts per question), tags significantly improve Hit@1 in both clean and poison conditions:**

| Condition | Pure CE | Tag-scoped CE | Delta | McNemar p |
|-----------|---------|---------------|-------|-----------|
| Clean | 14.9% Hit@1, 50.7% Hit@8 | 21.0% Hit@1, 56.5% Hit@8 | **+6.1** | **<0.0001** |
| Poison | 15.5% Hit@1, 54.3% Hit@8 | 23.0% Hit@1, 61.3% Hit@8 | **+7.5** | **<0.0001** |

Tag scoping gives CE better candidates per lane by narrowing to the right person/topic before reranking. The effect is stronger under poison (+7.5 vs +6.1) — tags filter unrelated poison from the candidate pool before CE sees it. However, this retrieval improvement does not translate to a statistically significant exam accuracy difference (p=0.618, Section 4.8).

**Single-pool retrieval tells a different story.** Tested on the same 1,537 questions (tagged DB with 2,075 tags in the inverted index), single-pool retrieval (top 4 from all memories) shows no tag benefit:

| Config | Hit@1 | Hit@4 | MRR |
|--------|-------|-------|-----|
| Pure CE (flat DB) | 53.2% | 63.8% | 0.576 |
| Pure CE (tagged DB, no tag scoping) | 54.8% | 67.2% | 0.599 |
| Tag-scoped CE (tagged DB) | 54.5% | 66.6% | 0.595 |

Tag scoping vs pure CE on same DB: p=0.29 (not significant). In single-pool retrieval, the candidate pool is large enough that CE finds the right memories without scoping.

**Under poison (single-pool):** tags help (+3.3 Hit@1, p<0.0001):

| Config | Hit@1 | MRR |
|--------|-------|-----|
| Pure CE (poison) | 51.2% | 0.583 |
| Tag-scoped CE (poison) | **54.5%** | **0.594** |

The difference between single-pool and two-lane results: two-lane splits retrieval into separate summary and fact lanes with smaller candidate pools per lane. Tag scoping matters more when the per-lane pool is smaller — narrowing candidates from 40 to the right subset has a bigger impact when each lane independently selects its top 4.

### 5.2.2 Nursery Slot

A reserved "nursery" slot (1 of 4 retrieval slots reserved for low-use memories) was tested on the archived DB to ensure new memories get retrieval opportunities. The nursery produced identical results to pure top-4 CE (p=1.0, McNemar's) and was removed. With Wilson scoring eliminated, new memories compete fairly on semantic relevance without being disadvantaged by low usage scores. (Tested inline during retrieval analysis; no standalone script.)

### 5.2.3 Summary: Component Elimination

All retrieval components were tested on 1,537 non-adversarial LoCoMo questions. Wilson blend was tested against an archived database (7,930 memories). Tag cascade and Wilson sort were tested on both clean and poisoned tagged databases (7,732 clean and 8,867 poisoned memories) using two architectures: cosine-first overlap-sort and tags-first cascade with inverted index.

| Component | Effect on Retrieval | Evidence | Decision |
|-----------|-------------------|----------|----------|
| **Cross-encoder** | +17.8 Hit@1 (p<0.0001) | 53.5% vs 35.7% (cosine) | **Keep** |
| **Wilson blend** | -2.9 Hit@1 single-pool (p=0.0001), -3.2 Hit@1 two-lane (p<0.0001) | 50.6% vs 53.5% / 21.9% vs 25.1% | **Remove** |
| **Wilson cascade sort** | **-4.3 Hit@1 clean (p=0.0000), -4.0 poison (p=0.0000)** | 23.0% vs 27.3% / 25.0% vs 29.0% (tags-first cascade) | **Remove** |
| **Wilson-only** | -5.1 Hit@1 vs cosine (p<0.0001) | 30.6% vs 35.7% | **Remove** |
| **Tags-first cascade** | **+1.9 Hit@1 clean (p=0.0000), +0.6 poison (p=0.25)** | 27.3% vs 25.4% / 29.0% vs 28.4% (vs pure CE) | **Keep** |
| **Tags-first vs cosine-first** | **+1.5 Hit@1 clean (p=0.0003), +1.0 poison (p=0.012)** | 27.3% vs 25.8% / 29.0% vs 28.0% | **Tags-first wins** |
| **Tag routing (two-lane clean)** | **+6.1 Hit@1 (p<0.0001)** | 21.0% vs 14.9% | **Keep** |
| **Tag routing (two-lane poison)** | **+7.5 Hit@1 (p<0.0001)** | 23.0% vs 15.5% | **Keep — stronger under attack** |
| **Nursery slot** | 0.0 Hit@1 (p=1.0) | Identical results | **Remove** |
| **Wilson under poison** | +0.8 Hit@1 (p=0.39, ns) | 51.1% vs 50.3% | **No benefit** |

**Why this is certain:**
1. **Wilson retrieval** was tested at every possible stage: blend scoring (3 conditions), cascade pre-sorting (2 architectures × 2 databases), and standalone. It hurts in every configuration (p<0.001). Wilson degrades CE's semantic judgment by overriding question-specific relevance with memory-level trust scores that cluster too tightly at scale to provide useful differentiation.

2. **Tags-first cascade** significantly outperforms cosine-first approaches (p=0.0003 clean). Using an inverted index to enter via tag overlap — rather than post-filtering cosine results — gives CE a better candidate pool by pulling from the full tag-matched population regardless of embedding distance.

3. **Nursery slot** produced identical results to pure top-4 CE. With Wilson removed, new memories compete on semantic relevance rather than being disadvantaged by low usage scores.

4. **Outcome scoring and lifecycle** (promotion, demotion, decay) were NOT eliminated. These operate on a different axis — they manage memory quality over time rather than influencing retrieval ranking. Their value will be measured in the poison resilience test (Section 4.5).

The final architecture: **tags-first cascade retrieval + CE reranking + outcome-based lifecycle management.** An inverted tag index serves as a lightweight knowledge graph, routing queries to the most specifically matched memories (highest tag overlap first). CE reranks the candidate pool by semantic relevance. Outcome scoring prunes bad memories over time through the tier lifecycle. No Wilson scoring at any stage.

**Isolation test results** (5 configs, 1,537 non-adversarial questions × 2 DBs — clean + poisoned tagged DBs with 1,135 injected memories, two-lane retrieval):

| Config | Hit@1 (Clean) | Hit@1 (Poison) | Hit@8 (Clean) | Hit@8 (Poison) | MRR (Clean) |
|--------|:---:|:---:|:---:|:---:|:---:|
| pure_ce (no tags) | 25.4% | 28.4% | 69.7% | 71.8% | 0.361 |
| overlap_sort + cosine | 25.8% | 28.0% | 69.6% | 71.7% | 0.364 |
| overlap_sort + Wilson | 22.8% | 25.8% | 65.4% | 67.5% | 0.330 |
| **tag_cascade + cosine** | **27.3%** | **29.0%** | **70.5%** | **72.4%** | **0.378** |
| tag_cascade + Wilson | 23.0% | 25.0% | 61.7% | 64.5% | 0.324 |

Two retrieval architectures were compared: (1) **cosine-first overlap-sort** — query ChromaDB by cosine, count tag overlaps on results, sort by overlap with cosine/Wilson tiebreaker; (2) **tags-first cascade** — build inverted index (tag → memory IDs), enter via tag lookup, fill candidate pool from highest overlap tier down with cosine/Wilson sort within each tier, cosine fallback for remaining slots. Architecture (2) is a lightweight knowledge graph: tags are nodes, memories are edges, overlap count is multi-hop convergence.

McNemar pairwise significance:
- Wilson vs cosine (tag cascade): p=0.0000 (clean), p=0.0000 (poison) → **cosine wins by 4.3/4.0 pts**
- Wilson vs cosine (overlap sort): p=0.0000 (clean), p=0.0005 (poison) → **cosine wins by 3.0/2.2 pts**
- Tag cascade vs overlap sort (cosine): p=0.0003 (clean), p=0.0124 (poison) → **tags-first wins by 1.5/1.0 pts**
- Tag cascade+cosine vs pure CE: p=0.0000 (clean), p=0.2530 (poison) → **tags-first wins clean, marginal poison**

Wilson scoring hurts retrieval in every configuration tested — both architectures, both databases. Tags-first cascade with cosine tiebreaker is the optimal retrieval architecture, significantly outperforming cosine-first approaches.

**Note on poison hit rates:** Retrieval hit rates on poisoned databases use keyword matching against the ground truth answer to determine if a "relevant" memory was retrieved. Because poison memories are semantically similar to correct facts (same entity names, similar topics), they may match keywords and be counted as hits. This means absolute hit rates on poison databases may be slightly inflated. However, the relative comparisons (tags vs no-tags, Wilson vs cosine) are valid because any inflation affects both conditions equally.

Wilson has been removed from the retrieval ranking entirely. The final system uses no Wilson scoring in retrieval — outcome scoring for lifecycle management uses raw scores, not Wilson confidence intervals. However, Wilson confidence remains visible in the metadata presented to the LLM alongside each retrieved memory (Section 3.3, Limitation #8); its contribution to answer accuracy was not isolated.

**Important caveat:** The isolation test measures retrieval quality (Hit@1, Hit@8, MRR) — whether the correct memory appears in the top-k. It does not measure end-to-end answer accuracy, which depends on the LLM's ability to use retrieved memories to generate correct responses. Retrieval improvements do not always translate 1:1 to answer accuracy improvements. The full pipeline exam (Section 4) measures the end-to-end effect.

### 5.3 Poison and Self-Healing

Cosine similarity is blind to factual correctness — wrong facts embed nearly identically to correct ones. Neither the embedding model nor the cross-encoder can distinguish them.

The system retains 72-73% LoCoMo accuracy (MiniMax) after 1,135 poison memories — a 2.6-4.2pt degradation from clean. Two factors likely contribute to this resilience:

1. **Correct memories outnumber poison.** At approximately 19% poison by volume (1,135 in approximately 5,000-6,000 total), correct memories compete for the same retrieval slots and often win on semantic relevance alone.

2. **Outcome scoring decays some poison.** The conversation healing loop decayed 134-154 poison memories (12-14% of injected) through accumulated "failed" scores dropping below the 0.1 threshold. However, approximately 1,000 poison memories survived the healing loop — either they were never retrieved (no opportunity to score) or they were retrieved but not scored harshly enough to decay.

A third possible factor: **visible metadata signals**. At exam time, surviving poison memories that accumulated "failed" outcomes show lower scores (e.g., s:0.22, [NNY]) compared to correct memories (s:0.55, [YYY]). The LLM sees this metadata on every retrieved memory and may use it to discount low-scored memories when resolving conflicts. Whether the LLM actually uses this signal was not tested (Limitation #8).

The relative contribution of these three factors — numeric dilution, outcome decay, and metadata trust signals — was not isolated. A control experiment injecting poison and examining immediately with no healing would measure pure dilution; a stripped-metadata condition would isolate the LLM's use of trust signals. Neither was run.

Tag routing provides some retrieval-level defense by scoping candidates to the queried person (Section 5.2.1), but this does not translate to a significant exam-level difference (Section 4.8). Wilson confidence intervals provide no defensive value against poison (Section 5.2.3).

The hard exam is more vulnerable (-5.3 to -11.8pt on 20B), as complex reasoning questions that require multiple correct memories in the top-k are more sensitive to individual poison memories competing for slots.

**Production implication:** Any memory system relying solely on embedding similarity is permanently vulnerable to stored misinformation. Outcome-based lifecycle management — not retrieval-time scoring — is the correct defense layer.

### 5.4 Limitations
1. **Single model** (20B) — absolute numbers model-dependent. GPT-4o-mini model swap (Section 4.6) partially addresses this; 20B significantly outperforms 4o-mini even under independent MiniMax regrading (McNemar's p=0.004).
2. **Single random seed, fixed conversation order** — characters are processed in a fixed order with no repeated trials. A character discussed late may benefit from a more mature memory system than one discussed early. The dataset is substantial (1,986 questions, 1,508 learning turns) but a single run cannot rule out order effects.
3. **CE model is English-only** (22.7M parameters ms-marco-MiniLM-L-6-v2) — production uses multilingual mmarco (118M). Results reflect English-only performance.
4. **LLM-as-judge, no human evaluation** — all grading is LLM-based (20B live + MiniMax M2.7 post-hoc). Research shows LLM judges exhibit self-preference bias (arXiv:2410.21819), potentially favoring the 20B's own outputs. Dual grading with an independent model (MiniMax) mitigates this — key findings hold under both graders — but human evaluation would provide stronger validation.
5. **LoCoMo ground truth corrections** — 444/446 adversarial answers corrected using Claude Sonnet/Opus (Section 2.4), verified against transcripts (0/200 errors). Corrections are AI-generated, not independently human-verified. However, all systems are evaluated on the same corrected data, so any bias affects absolute numbers equally, not comparative results.
6. **Single benchmark** — all results are on LoCoMo (corrected). The hard exam (76 questions) provides supplementary validation but was designed by the authors, not independently. Findings should be validated on additional benchmarks (e.g., LongMemEval).
7. **Format familiarity confound in model swap** — the memory database was created by the 20B (conversation learning, fact extraction, sidecar scoring). The 20B sees memories in its own style at exam time; 4o-mini sees an unfamiliar format. The 1.5-2.5pt model gap may partially reflect format familiarity rather than pure model capability. A full pipeline re-run with 4o-mini as the conversation learner would control for this.
8. **Metadata contribution not isolated** — every retrieved memory includes outcome metadata (score, Wilson confidence, use count, outcome history) visible to the answering LLM. No condition was tested with metadata stripped, so the contribution of metadata to answer accuracy is unknown. The LLM may use it to resolve conflicting memories, or it may ignore it entirely.
9. **No specificity testing** — no unanswerable questions beyond the adversarial category. Systems are not penalized for hallucinating answers to questions with no stored evidence.
10. **Poison attack is author-designed** — I designed the 1,135 poison memories, injected them, and measured resilience. Poison facts achieve cosine distances nearly identical to correct facts (effectively indistinguishable to embedding similarity), but no external red team validated the attack design. Metadata is simplified (3 outcome entries regardless of tier).
11. **Exchange window constraint** — LLM A sees last 4 exchanges + 8 retrieved memories (no full conversation history). This is a deliberate production-matching constraint, not a limitation, but means the system operates with strictly less context than full-transcript ingestion systems.
12. **Temperature 0.7 for LLM B** (conversation variety), 0 for LLM A and exams (deterministic). Non-deterministic conversation generation means exact memory content varies across runs.

---

## 6. Conclusion

A local 20B model with conversation-based learning and cross-encoder reranking achieves 85.8% on non-adversarial LoCoMo questions (MiniMax M2.7 regraded). For context, MemMachine reports 87.5% on 4/5 categories under a different evaluation protocol (GPT-4o-mini, different judge model and scoring method) — these scores are not directly comparable due to protocol fragmentation (Section 1.2), but suggest the architecture operates in a competitive range. Swapping the local 20B for GPT-4o-mini changes accuracy by 1.5-2.5 points (TagCascade p=0.004, CE-Only p=0.085, MiniMax-regraded), while the architecture contributes 22+ points (p<0.0001) — the architecture effect dominates.

Four findings stand out:

1. **Conversation learning works.** Memories formed through natural dialogue outperform raw ingestion by 22+ points (65.4% vs 43.1%, 20B; p<0.0001). The learning loop produces precise atomic facts that are better cosine targets than raw conversation chunks, despite generating a 2.9x larger database. This value is model-independent: GPT-4o-mini shows the same +22pt architecture lift (51.9% baseline → 74.4% learned).

2. **Architecture dominates model choice.** GPT-4o-mini achieves 82.6% non-adversarial on this architecture vs 85.8% with the local 20B — a 3.2pt gap (TagCascade p=0.004, CE-Only p=0.085, MiniMax-regraded; memories created by the 20B — Limitation #7). The architecture contributes +22pt (p<0.0001) while the model contributes +2-3pt. The architecture effect is roughly 10x larger.

3. **Wilson scoring hurts retrieval ranking.** Tested at every retrieval stage — blend, cascade sort, standalone — Wilson actively degrades cross-encoder performance (p<0.001 in all configurations). Raw outcome scoring drives the lifecycle; Wilson confidence intervals add nothing to retrieval. This is a structural incompatibility: CE needs query-specific relevance, Wilson provides query-independent trust. Wilson confidence remains visible in memory metadata presented to the LLM, but its contribution to answer accuracy was not isolated (Limitation #8).

4. **The system is resilient to poison, but the mechanism is not fully isolated.** Despite 1,135 adversarial memories with spoofed trust signals distributed across all three tiers, the system retains 72-73% LoCoMo accuracy — a 2.6-4.2pt degradation from clean. Three factors likely contribute: correct memories outnumbering poison (approximately 19% by volume), outcome scoring decaying 12-14% of poison, and visible metadata trust signals helping the LLM discount low-scored memories. The relative contribution of each was not isolated (Section 5.3). Tag routing does not provide statistically significant additional defense at the exam level (p=0.618), though it improves retrieval ranking (p<0.0001).

The hard exam (76 questions) shows 46-49% accuracy (MiniMax, clean 20B) on multi-retrieval reasoning questions, with cross-entity inference (15-30%) as the weakest category. These questions require combining facts about multiple people — a fundamental limitation of per-query retrieval. No hard exam comparison achieves statistical significance at n=76.

The primary pipeline (conversation learning, exams, live grading) is fully reproducible on a single GPU with 16GB+ VRAM (tested on NVIDIA RTX 5090; gpt-oss:20b requires approximately 14GB) with no cloud dependencies. Cloud APIs are used for two supplementary purposes: MiniMax M2.7 for independent post-hoc regrading, and GPT-4o-mini for model swap validation confirming findings generalize beyond the local model. The complete pipeline, data, and evaluation scripts are open-sourced.

---

## 7. Reproducing Results

```bash
# Requirements: Python 3.10+, NVIDIA GPU with 16GB+ VRAM (cross-encoder requires CUDA), Ollama
git clone https://github.com/roampal-ai/roampal-labs
cd roampal-labs
pip install -e .
ollama pull gpt-oss:20b

# Clean pipeline (conversation learning + exams)
python run_pipeline.py

# Poison pipeline (inject + conversation healing + exams)
python run_poison_pipeline.py

# Model swap (requires OPENAI_API_KEY)
python run_model_swap.py

# Statistical tests (no GPU needed, uses saved transcripts)
python results/statistical_tests.py

# MiniMax regrading (requires MINIMAX_API_KEY)
python results/minimax_regrader.py results/exam_*.json

# Watch progress
python -m benchmark.dashboard
```

---

## Acknowledgments

This research was conducted with the assistance of Claude (Anthropic), powered by the Roampal AI platform. Claude assisted with code development, data processing, ground truth correction, verification, and manuscript preparation.

---

## References

1. Maharana et al. "Evaluating Very Long-Term Conversational Memory of LLM Agents." ACL 2024.
2. MemMachine. "MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents." arXiv:2604.04853, Apr 2026.
3. Mem0. arXiv:2504.19413, 2025.
4. Zep/Graphiti. arXiv:2501.13956, 2025.
5. SmartSearch. arXiv:2603.15599, Mar 2026.
6. Jacob et al. "Drowning in Documents: Consequences of Scaling Reranker Inference." arXiv:2411.11767, 2024.
7. Wilson, E.B. "Probable Inference." JASA, 1927.
8. Panickssery et al. "LLM Evaluators Recognize and Favor Their Own Generations." arXiv:2410.21819, 2024.
