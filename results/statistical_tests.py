#!/usr/bin/env python3
"""
Statistical significance tests for all exam comparisons.
Produces Table 4.8 for the paper.

Tests:
  1. Binomial 95% CIs on all conditions
  2. McNemar's test on paired per-question outcomes (same questions, different strategies/conditions)
  3. Effect sizes (Cohen's h)

Usage:
  python results/statistical_tests.py
"""
import json
import math
import sys
from pathlib import Path
from scipy.stats import norm

RESULTS_DIR = Path("results")


def load_transcript(filename):
    """Load exam transcript and return per-question judgments keyed by question text."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    data = json.load(open(path, encoding="utf-8"))
    transcript = data.get("transcript", [])
    by_question = {}
    for entry in transcript:
        q = entry.get("question", entry.get("query", ""))
        j = entry.get("judgment", "unknown")
        by_question[q] = j
    return by_question


def binomial_ci(correct, total, confidence=0.95):
    """Wilson score interval (not to be confused with Wilson memory scoring)."""
    if total == 0:
        return (0, 0)
    p = correct / total
    z = norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (center - spread, center + spread)


def mcnemar_test(judgments_a, judgments_b, label_a, label_b):
    """McNemar's test on paired per-question outcomes.
    Returns (n_ab, n_ba, chi2, p_value) where:
      n_ab = questions A got right but B got wrong
      n_ba = questions B got right but A got wrong
    """
    common_questions = set(judgments_a.keys()) & set(judgments_b.keys())
    if not common_questions:
        return None

    # Count discordant pairs (correct vs not-correct)
    a_right_b_wrong = 0
    b_right_a_wrong = 0
    both_right = 0
    both_wrong = 0

    for q in common_questions:
        a_correct = judgments_a[q] == "correct"
        b_correct = judgments_b[q] == "correct"
        if a_correct and not b_correct:
            a_right_b_wrong += 1
        elif b_correct and not a_correct:
            b_right_a_wrong += 1
        elif a_correct and b_correct:
            both_right += 1
        else:
            both_wrong += 1

    n = a_right_b_wrong + b_right_a_wrong
    if n == 0:
        return {
            "n_common": len(common_questions),
            "a_right_b_wrong": 0, "b_right_a_wrong": 0,
            "both_right": both_right, "both_wrong": both_wrong,
            "chi2": 0, "p_value": 1.0, "significant": False,
            "label": f"{label_a} vs {label_b}",
        }

    # McNemar's with continuity correction
    chi2 = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "n_common": len(common_questions),
        "a_right_b_wrong": a_right_b_wrong,
        "b_right_a_wrong": b_right_a_wrong,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "label": f"{label_a} vs {label_b}",
    }


def cohens_h(p1, p2):
    """Effect size for comparing two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def print_ci_table(rows):
    """Print binomial CI table."""
    print(f"\n{'Condition':<45} {'Correct':>7} {'Total':>6} {'Acc':>7} {'95% CI':>16}")
    print("-" * 85)
    for label, correct, total in rows:
        if total == 0:
            continue
        acc = correct / total
        lo, hi = binomial_ci(correct, total)
        print(f"{label:<45} {correct:>7} {total:>6} {acc:>7.1%} ({lo*100:>5.1f}-{hi*100:>5.1f}%)")


def print_mcnemar_table(results):
    """Print McNemar's test results."""
    print(f"\n{'Comparison':<50} {'A>B':>5} {'B>A':>5} {'chi2':>8} {'p':>10} {'Sig?':>6}")
    print("-" * 90)
    for r in results:
        if r is None:
            continue
        sig = "YES" if r["significant"] else "no"
        print(f"{r['label']:<50} {r['a_right_b_wrong']:>5} {r['b_right_a_wrong']:>5} "
              f"{r['chi2']:>8.2f} {r['p_value']:>10.6f} {sig:>6}")


def main():
    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 90)

    # ─── Load all transcripts ───────────────────────────────────────────────
    exams = {
        "tc_clean_locomo": load_transcript("exam_02.TagCascade_02_locomo_off.json"),
        "ce_clean_locomo": load_transcript("exam_03.CE-Only_02_locomo_off.json"),
        "tc_poison_locomo": load_transcript("poison_exam_02.TagCascade_locomo_off.json"),
        "ce_poison_locomo": load_transcript("poison_exam_03.CE-Only_locomo_off.json"),
        "baseline_locomo": load_transcript("exam_baseline_raw_repaired.json"),
        "tc_clean_hard": load_transcript("exam_02.TagCascade_03_hard_off.json"),
        "ce_clean_hard": load_transcript("exam_03.CE-Only_03_hard_off.json"),
        "tc_poison_hard": load_transcript("poison_exam_02.TagCascade_hard_off.json"),
        "ce_poison_hard": load_transcript("poison_exam_03.CE-Only_hard_off.json"),
        "baseline_hard": load_transcript("exam_baseline_raw_repaired_hard_off.json"),
        "nomem_locomo": load_transcript("exam_no_memory_locomo_off.json"),
        "nomem_hard": load_transcript("exam_no_memory_hard_off.json"),
    }

    loaded = {k: v for k, v in exams.items() if v is not None}
    print(f"\nLoaded {len(loaded)} exam transcripts: {', '.join(loaded.keys())}")

    # Also load MiniMax regrades
    minimax = {
        "tc_clean_locomo_mm": load_transcript("minimax_grade_02.TagCascade_02_locomo_off.json"),
        "ce_clean_locomo_mm": load_transcript("minimax_grade_03.CE-Only_02_locomo_off.json"),
        "tc_poison_locomo_mm": load_transcript("poison_minimax_grade_02.TagCascade_locomo_off.json"),
        "ce_poison_locomo_mm": load_transcript("poison_minimax_grade_03.CE-Only_locomo_off.json"),
        "baseline_locomo_mm": load_transcript("minimax_grade_baseline_raw_repaired.json"),
    }

    # MiniMax files have different structure — per_question with minimax_judgment
    for key in list(minimax.keys()):
        path_map = {
            "tc_clean_locomo_mm": "minimax_grade_02.TagCascade_02_locomo_off.json",
            "ce_clean_locomo_mm": "minimax_grade_03.CE-Only_02_locomo_off.json",
            "tc_poison_locomo_mm": "poison_minimax_grade_02.TagCascade_locomo_off.json",
            "ce_poison_locomo_mm": "poison_minimax_grade_03.CE-Only_locomo_off.json",
            "baseline_locomo_mm": "minimax_grade_baseline_raw_repaired.json",
        }
        fpath = RESULTS_DIR / path_map.get(key, "")
        if not fpath.exists():
            minimax[key] = None
            continue
        data = json.load(open(fpath, encoding="utf-8"))
        # MiniMax regrades: need to map back to questions via source exam
        src_name = data.get("source_exam", "")
        src_path = RESULTS_DIR / src_name
        if not src_path.exists():
            # Try poison prefix
            src_path = RESULTS_DIR / src_name
            if not src_path.exists():
                minimax[key] = None
                continue
        src_data = json.load(open(src_path, encoding="utf-8"))
        src_transcript = src_data.get("transcript", [])
        per_q = data.get("per_question", [])
        by_question = {}
        for pq in per_q:
            idx = pq.get("index", -1)
            if 0 <= idx < len(src_transcript):
                q = src_transcript[idx].get("question", src_transcript[idx].get("query", ""))
                by_question[q] = pq.get("minimax_judgment", "unknown")
        minimax[key] = by_question if by_question else None

    mm_loaded = {k: v for k, v in minimax.items() if v is not None}
    print(f"Loaded {len(mm_loaded)} MiniMax regrades: {', '.join(mm_loaded.keys())}")

    # ─── 1. Binomial CIs ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 1: BINOMIAL 95% CONFIDENCE INTERVALS (20B GRADER)")
    print("=" * 90)

    ci_rows = []
    for label, key in [
        ("TagCascade clean LoCoMo", "tc_clean_locomo"),
        ("CE-Only clean LoCoMo", "ce_clean_locomo"),
        ("TagCascade poison LoCoMo", "tc_poison_locomo"),
        ("CE-Only poison LoCoMo", "ce_poison_locomo"),
        ("Baseline LoCoMo", "baseline_locomo"),
        ("No-memory LoCoMo", "nomem_locomo"),
        ("TagCascade clean Hard", "tc_clean_hard"),
        ("CE-Only clean Hard", "ce_clean_hard"),
        ("TagCascade poison Hard", "tc_poison_hard"),
        ("CE-Only poison Hard", "ce_poison_hard"),
        ("Baseline Hard", "baseline_hard"),
        ("No-memory Hard", "nomem_hard"),
    ]:
        jmap = exams.get(key)
        if jmap is None:
            continue
        total = len(jmap)
        correct = sum(1 for j in jmap.values() if j == "correct")
        ci_rows.append((label, correct, total))

    print_ci_table(ci_rows)

    # MiniMax CIs
    if mm_loaded:
        print("\n" + "=" * 90)
        print("TABLE 2: BINOMIAL 95% CONFIDENCE INTERVALS (MINIMAX M2.7)")
        print("=" * 90)
        mm_ci_rows = []
        for label, key in [
            ("TagCascade clean LoCoMo (MM)", "tc_clean_locomo_mm"),
            ("CE-Only clean LoCoMo (MM)", "ce_clean_locomo_mm"),
            ("TagCascade poison LoCoMo (MM)", "tc_poison_locomo_mm"),
            ("CE-Only poison LoCoMo (MM)", "ce_poison_locomo_mm"),
            ("Baseline LoCoMo (MM)", "baseline_locomo_mm"),
        ]:
            jmap = minimax.get(key)
            if jmap is None:
                continue
            total = len(jmap)
            correct = sum(1 for j in jmap.values() if j == "correct")
            mm_ci_rows.append((label, correct, total))
        print_ci_table(mm_ci_rows)

    # ─── 2. McNemar's Tests ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 3: McNEMAR'S TEST — PAIRED PER-QUESTION COMPARISONS (20B)")
    print("=" * 90)

    mcnemar_results = []

    comparisons = [
        ("tc_clean_locomo", "ce_clean_locomo", "TagCascade vs CE-Only (clean LoCoMo)"),
        ("tc_poison_locomo", "ce_poison_locomo", "TagCascade vs CE-Only (poison LoCoMo)"),
        ("tc_clean_locomo", "tc_poison_locomo", "TagCascade clean vs poison (LoCoMo)"),
        ("ce_clean_locomo", "ce_poison_locomo", "CE-Only clean vs poison (LoCoMo)"),
        ("tc_clean_locomo", "baseline_locomo", "TagCascade vs Baseline (clean LoCoMo)"),
        ("ce_clean_locomo", "baseline_locomo", "CE-Only vs Baseline (clean LoCoMo)"),
        ("tc_clean_hard", "ce_clean_hard", "TagCascade vs CE-Only (clean Hard)"),
        ("tc_poison_hard", "ce_poison_hard", "TagCascade vs CE-Only (poison Hard)"),
        ("tc_clean_hard", "tc_poison_hard", "TagCascade clean vs poison (Hard)"),
        ("ce_clean_hard", "ce_poison_hard", "CE-Only clean vs poison (Hard)"),
    ]

    # Add no-memory and baseline hard if available
    if exams.get("nomem_locomo"):
        comparisons.append(("tc_clean_locomo", "nomem_locomo", "TagCascade vs No-memory (LoCoMo)"))
    if exams.get("baseline_hard"):
        comparisons.append(("tc_clean_hard", "baseline_hard", "TagCascade vs Baseline (Hard)"))

    for key_a, key_b, label in comparisons:
        ja = exams.get(key_a)
        jb = exams.get(key_b)
        if ja is None or jb is None:
            continue
        result = mcnemar_test(ja, jb, label.split(" vs ")[0] if " vs " in label else "A",
                              label.split(" vs ")[1].split(" (")[0] if " vs " in label else "B")
        result["label"] = label
        mcnemar_results.append(result)

    print_mcnemar_table(mcnemar_results)

    # MiniMax McNemar's
    if mm_loaded:
        print("\n" + "=" * 90)
        print("TABLE 4: McNEMAR'S TEST — PAIRED PER-QUESTION COMPARISONS (MINIMAX)")
        print("=" * 90)

        mm_comparisons = [
            ("tc_clean_locomo_mm", "ce_clean_locomo_mm", "TagCascade vs CE-Only (clean LoCoMo MM)"),
            ("tc_poison_locomo_mm", "ce_poison_locomo_mm", "TagCascade vs CE-Only (poison LoCoMo MM)"),
            ("tc_clean_locomo_mm", "tc_poison_locomo_mm", "TagCascade clean vs poison (LoCoMo MM)"),
            ("ce_clean_locomo_mm", "ce_poison_locomo_mm", "CE-Only clean vs poison (LoCoMo MM)"),
            ("tc_clean_locomo_mm", "baseline_locomo_mm", "TagCascade vs Baseline (LoCoMo MM)"),
        ]

        mm_results = []
        for key_a, key_b, label in mm_comparisons:
            ja = minimax.get(key_a)
            jb = minimax.get(key_b)
            if ja is None or jb is None:
                continue
            result = mcnemar_test(ja, jb, "A", "B")
            result["label"] = label
            mm_results.append(result)

        print_mcnemar_table(mm_results)

    # ─── 3. Effect Sizes ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 5: EFFECT SIZES (COHEN'S h)")
    print("=" * 90)
    print(f"\n{'Comparison':<50} {'h':>8} {'Size':>10}")
    print("-" * 72)

    effect_pairs = [
        ("TagCascade vs CE-Only (clean LoCoMo)", 1287/1986, 1299/1986),
        ("TagCascade clean vs poison (LoCoMo)", 1287/1986, 1194/1986),
        ("CE-Only clean vs poison (LoCoMo)", 1299/1986, 1192/1986),
        ("Learning vs Baseline (LoCoMo)", 1299/1986, 855/1986),
    ]

    for label, p1, p2 in effect_pairs:
        h = cohens_h(p1, p2)
        size = "negligible" if abs(h) < 0.2 else "small" if abs(h) < 0.5 else "medium" if abs(h) < 0.8 else "large"
        print(f"{label:<50} {h:>8.3f} {size:>10}")

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: WHAT IS STATISTICALLY SIGNIFICANT?")
    print("=" * 90)
    print()
    sig_results = [r for r in mcnemar_results if r and r["significant"]]
    nonsig_results = [r for r in mcnemar_results if r and not r["significant"]]

    print("SIGNIFICANT (p < 0.05):")
    for r in sig_results:
        print(f"  {r['label']}: p={r['p_value']}")
    print()
    print("NOT SIGNIFICANT:")
    for r in nonsig_results:
        print(f"  {r['label']}: p={r['p_value']}")

    # Save results
    output = {
        "binomial_cis": [
            {"label": label, "correct": c, "total": t,
             "accuracy": round(c/t, 4) if t else 0,
             "ci_low": round(binomial_ci(c, t)[0], 4) if t else 0,
             "ci_high": round(binomial_ci(c, t)[1], 4) if t else 0}
            for label, c, t in ci_rows
        ],
        "mcnemar_tests": [{k: (int(v) if isinstance(v, bool) else v) for k, v in r.items()}
                         for r in mcnemar_results if r],
    }
    out_path = RESULTS_DIR / "statistical_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
