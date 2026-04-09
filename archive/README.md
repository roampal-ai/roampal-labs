# Archive: Preliminary Pipeline Databases

These databases were produced by Phase 1 preliminary pipeline runs (Section 3.2 of the paper). They are used in the Discussion retrieval analysis (Sections 5.1.2-5.2.3) to diagnose retrieval failures and inform component selection.

The final pipeline (Section 4) uses different databases in `runs/final/` and `runs/poison/` with the improvements identified from this analysis already applied.

## Databases

| Directory | Strategy | Memories | Used By |
|-----------|----------|----------|---------|
| `pre_fix_run/runs/01.EntityRouted` | Tag-routed + Wilson+CE | 7,732 | isolate_tag_wilson, isolate_tag_wilson_poison, tag_twolane_test, isolate_cascade_wilson |
| `pre_fix_run/runs/02.Wilson+CE` | Wilson+CE blend (no tags) | 7,930 | facts_value_test, retrieval_failure_diagnosis, blend_optimizer (single-pool Wilson tests) |
| `pre_fix_run/runs/03.Reranker` | Pure CE reranker | ~7,800 | blend_optimizer |
| `pre_fix_run/runs/04.Wilson` | Wilson-only (no CE) | ~7,800 | Not used in final analysis |

## Running the analysis scripts

All scripts in `benchmark/` reference these databases by path. To reproduce:

```bash
python -m benchmark.facts_value_test           # Section 5.1.2
python -m benchmark.retrieval_failure_diagnosis # Section 5.1.3
python -m benchmark.blend_optimizer             # Section 5.2 (single-pool)
python -m benchmark.isolate_tag_wilson          # Section 5.2 (two-lane clean)
python -m benchmark.isolate_tag_wilson_poison   # Section 5.2 (two-lane poison)
python -m benchmark.tag_twolane_test            # Section 5.2.1
python -m benchmark.isolate_cascade_wilson      # Section 5.2.3
python -m benchmark.blend_sweep                 # Section 5.2 (Wilson sweep)
```

Results are saved to `results/` as JSON files.
