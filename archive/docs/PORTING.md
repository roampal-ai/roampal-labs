# Porting Guide: memory-retrieval-benchmark -> roampal-labs

## What to copy from C:/memory-retrieval-benchmark

### Strategy implementations (4 files)
- `strategies/wilson_scored.py` -> Wilson (cosine + Wilson scoring only)
- `strategies/semantic_reranker.py` -> Reranker (cosine + CE rerank, no outcome scoring)
- `strategies/wilson_reranker.py` -> Wilson+CE (cosine + CE + Wilson blend) **<- primary strategy**
- `strategies/entity_routed.py` -> EntityRouted (tag cascade + CE + Wilson)

### Benchmark runner
- `benchmark/runner.py` -> core orchestration (learning loop, exam loop, resume, live_state)
- `benchmark/dashboard.py` -> live monitoring
- `benchmark/grader.py` -> LLM-as-judge grading

### Data
- `data/locomo_full.json` -> dataset (404 learning Qs, 1986 LoCoMo exam, hard exam, poison)
- `context/window.py` -> sliding window context manager

### Supporting
- `strategies/base.py` -> base class / interfaces
- `pyproject.toml` -> dependencies
- `run_full.py` -> orchestrator (needs rewrite for new pipeline)

### What NOT to copy
- `results/` -> old 7-pass data (R&D, not publishable)
- `runs/` -> old ChromaDB stores
- `scripts/` -> ad-hoc scripts
- `test_regex_tags.py` -> old test
- Old minimax regrader files

## Changes needed for new pipeline

### Runner modifications
1. **Remove pass cycling** -> single pass, not 7x repeating 404 questions
2. **Add MiniMax regrading support** -> dual grading during exams (or post-hoc via transcript files)
3. **Add DB backup step** -> snapshot ChromaDB before poison injection
4. **Learning ON/OFF per step** -> configurable per exam phase
5. **CrossEncoder on CUDA** -> `device="cuda"` (torch 2.11.0+cu130 installed)

### New orchestrator pipeline
See PIPELINE.md for the exact step-by-step design.
