# Build Spec: What Exists, What's Missing, What To Build

## Status: READY TO BUILD TOMORROW

---

## What Already Works (verified in code)

1. **run_exam() supports learning ON/OFF** — `learning: bool = False` parameter exists (runner.py:532). When ON, it calls `strategy.record_outcome()` to update Wilson scores. Proven working.

2. **Hard exam loading** — Code exists at runner.py:982-1009. Loads `data/hard_exam.json`, runs 76 questions. Currently gated behind `if run_poison` — just remove the gate.

3. **Poison injection with fake Wilson metadata** — Exists at runner.py:1018-1031. Stores poison content, then updates ChromaDB metadata with fake scores from `fake_meta` field. Works but brittle (direct _collection access).

4. **Poison data file** — `data/poison_memories.json` has 50 memories, each with `content`, `targets`, and `fake_meta` (uses, success_count, score). Ready to use.

5. **Dashboard** — `benchmark/dashboard.py` reads live_state.json every 2 seconds. Shows progress, exam history, live feed.

6. **Exam transcript saving** — Every exam saves a JSON file with full question/answer/judgment for every question. MiniMax regrading reads these files.

7. **All 4 strategy files** — exist, compile, CE strategies have `device="cuda"`.

---

## What's Missing (gaps to fill)

### CRITICAL (must have)

| # | Gap | Where | Fix | Time |
|---|-----|-------|-----|------|
| 1 | **Step-based orchestrator** | New file: `run_pipeline.py` | Replace run_full.py's pass loop with 11-step sequence per strategy | 2-3 hrs |
| 2 | **DB snapshot before poison** | All strategy files | `shutil.copytree(persist_dir, snapshot_dir)` | 30 min |
| 3 | **Hard exam in main flow** | runner.py:983 | Remove `if run_poison` gate | 5 min |
| 4 | **Poison data loading** | runner.py:763-772 | Read from `data/poison_memories.json` instead of empty keys in main dataset | 15 min |

### NICE TO HAVE (can add after launch)

| # | Gap | Where | Fix | Time |
|---|-----|-------|-----|------|
| 5 | Dashboard step display | dashboard.py:47-59 | Add `current_step` to live_state and render | 30 min |
| 6 | MiniMax baked in | runner.py (new) | After each exam, fire off MiniMax regrading on the transcript | 1-2 hrs |
| 7 | store_with_metadata() | All strategy files | Cleaner poison injection API | 1 hr |

---

## The Build Plan (tomorrow)

### Step 1: Create run_pipeline.py (~2-3 hrs)
New orchestrator in roampal-labs. Replaces run_full.py. Structure:

```python
STRATEGIES = [
    ("01.Wilson", WilsonScoredStrategy),
    ("02.Reranker", SemanticRerankerStrategy),
    ("03.Wilson+CE", WilsonRerankerStrategy),
    ("04.EntityRouted", EntityRoutedStrategy),
]

PIPELINE = [
    # (name, type, learning, description)
    ("01_learn_404",      "learning",  True,  "404 learning turns"),
    ("02_locomo_off",     "locomo",    False, "LoCoMo baseline (no Wilson)"),
    ("03_locomo_on",      "locomo",    True,  "LoCoMo headline (Wilson building)"),
    ("04_hard_off",       "hard",      False, "Hard exam (held-out reasoning)"),
    ("05_snapshot",       "snapshot",  None,  "Backup DBs before poison"),
    ("06_inject_poison",  "poison",    None,  "Inject 50 adversarial memories"),
    ("07_poison_locomo",  "locomo",    True,  "Poison LoCoMo (damage + healing)"),
    ("08_locomo_off",     "locomo",    False, "Post-poison damage measurement"),
    ("09_heal_404",       "learning",  True,  "404 healing turns"),
    ("10_locomo_off",     "locomo",    False, "Recovery measurement"),
    ("11_hard_off",       "hard",      False, "Post-heal reasoning"),
]

async def run_pipeline():
    for strategy_name, strategy_cls in STRATEGIES:
        # Fresh DB for each strategy
        strategy = strategy_cls(persist_dir=f"runs/final/{strategy_name}")
        await strategy.initialize()

        for step_name, step_type, learning, desc in PIPELINE:
            update_live_state(strategy_name, step_name, desc)

            if step_type == "learning":
                await run_group(strategy, queries, num_turns=404, learning=True)
            elif step_type == "locomo":
                await run_exam(strategy, locomo_questions, learning=learning)
                save_exam_transcript(strategy_name, step_name, results)
            elif step_type == "hard":
                await run_exam(strategy, hard_questions, learning=False)
                save_exam_transcript(strategy_name, step_name, results)
            elif step_type == "snapshot":
                shutil.copytree(strategy._persist_dir, f"runs/final/{strategy_name}_snapshot")
            elif step_type == "poison":
                for pm in poison_memories:
                    doc_id = await strategy.store(pm["content"])
                    # Apply fake Wilson metadata
                    fake = pm.get("fake_meta", {})
                    if fake:
                        meta = strategy._collection.get(ids=[doc_id])["metadatas"][0]
                        meta.update(fake)
                        strategy._collection.update(ids=[doc_id], metadatas=[meta])
```

### Step 2: Copy strategy files to roampal-labs (~15 min)
- strategies/*.py
- benchmark/runner.py (run_group + run_exam functions)
- benchmark/grader.py
- benchmark/dashboard.py
- data/locomo_full.json
- data/hard_exam.json
- data/poison_memories.json
- context/window.py

### Step 3: Test one strategy through full pipeline (~1 hr)
- Run Wilson (fastest, no CE) through all 11 steps
- Verify exam transcripts saved at each step
- Verify snapshot/restore works
- Verify poison injection with fake metadata

### Step 4: Launch all 4 strategies
- Sequential execution
- ~48 hrs unattended
- Dashboard monitoring

### Step 5: MiniMax regrading (post-hoc)
- Run minimax_regrader.py on exam transcripts from steps 2, 3, 4, 8, 10, 11
- ~40,000 questions, ~$21

---

## Files to copy from C:/memory-retrieval-benchmark

```
strategies/
  base.py
  wilson_scored.py
  semantic_reranker.py
  wilson_reranker.py
  entity_routed.py
benchmark/
  runner.py (extract run_group + run_exam + grading logic)
  grader.py
  dashboard.py
context/
  window.py
data/
  locomo_full.json
  hard_exam.json
  poison_memories.json
results/
  minimax_regrader.py (for post-hoc regrading)
```

## Files to create fresh

```
run_pipeline.py      <- new orchestrator (Step 1)
pyproject.toml       <- dependencies
README.md            <- how to run
```

---

## Verification Checklist (before launch)

- [ ] Fresh ChromaDB per strategy (no old data)
- [ ] 404 learning questions load correctly (404 questions)
- [ ] LoCoMo exam loads correctly (1,986 questions, 5 categories)
- [ ] Hard exam loads correctly (76 questions, 5 categories)
- [ ] Poison loads correctly (50 memories with fake_meta)
- [ ] run_exam(learning=True) stores memories AND updates Wilson scores
- [ ] run_exam(learning=False) does NOT store or score
- [ ] Snapshot creates a copy of ChromaDB directory
- [ ] Poison injection applies fake Wilson metadata to ChromaDB
- [ ] Exam transcripts saved with step name in filename
- [ ] Dashboard shows current strategy + step
- [ ] CE strategies load on CUDA (torch.cuda.is_available() == True)
- [ ] Ollama running with gpt-oss:20b
