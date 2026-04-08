#!/usr/bin/env python3
"""Retry only the unknown/failed questions from a minimax regrading run."""
import json, os, sys, time
sys.path.insert(0, "results")
from minimax_regrader import call_minimax, GRADING_PROMPT

def main():
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("MINIMAX_API_KEY not set"); return

    results_file = "results/minimax_grade_01.EntityRouted_02_locomo_off.json"
    exam_file = "results/exam_01.EntityRouted_02_locomo_off.json"
    retry_file = "results/minimax_retry_indices.json"

    data = json.loads(open(results_file, encoding="utf-8").read())
    exam = json.loads(open(exam_file, encoding="utf-8").read())
    retry_indices = json.loads(open(retry_file, encoding="utf-8").read())
    transcript = exam["transcript"]
    pq = data["per_question"]

    print(f"Retrying {len(retry_indices)} unknown questions...")
    fixed = 0
    still_unknown = 0
    for i, idx in enumerate(retry_indices):
        entry = transcript[idx]
        prompt = GRADING_PROMPT.format(
            question=entry["question"],
            ground_truth=entry["ground_truth"],
            answer=entry["answer"][:400]
        )
        result = call_minimax(prompt, api_key, max_retries=5)
        
        if result != "unknown" and result != "error":
            old = pq[idx]["minimax_judgment"]
            pq[idx]["minimax_judgment"] = result
            pq[idx]["changed"] = result != pq[idx]["original_judgment"]
            if pq[idx]["changed"]:
                rank = {"wrong": 0, "partial": 1, "correct": 2}
                if rank.get(result, -1) > rank.get(pq[idx]["original_judgment"], -1):
                    pq[idx]["change_type"] = "upgrade"
                elif rank.get(result, -1) < rank.get(pq[idx]["original_judgment"], -1):
                    pq[idx]["change_type"] = "downgrade"
                else:
                    pq[idx]["change_type"] = "lateral"
            else:
                pq[idx]["change_type"] = None
            fixed += 1
        else:
            still_unknown += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(retry_indices)} — fixed: {fixed}, still unknown: {still_unknown}")
        time.sleep(0.5)

    # Recompute stats
    correct = sum(1 for q in pq if q["minimax_judgment"] == "correct")
    partial = sum(1 for q in pq if q["minimax_judgment"] == "partial")
    wrong = sum(1 for q in pq if q["minimax_judgment"] == "wrong")
    unknown = sum(1 for q in pq if q["minimax_judgment"] == "unknown")
    ups = sum(1 for q in pq if q.get("change_type") == "upgrade")
    downs = sum(1 for q in pq if q.get("change_type") == "downgrade")
    total = len(pq)

    data["per_question"] = pq
    data["correct"] = correct
    data["partial"] = partial
    data["wrong"] = wrong
    data["upgrades"] = ups
    data["downgrades"] = downs
    data["accuracy_strict"] = round(correct / total, 4)
    data["accuracy_lenient"] = round((correct + partial) / total, 4)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Fixed {fixed}, still unknown: {still_unknown}")
    print(f"Correct: {correct}, Partial: {partial}, Wrong: {wrong}, Unknown: {unknown}")
    print(f"Strict: {data['accuracy_strict']:.1%}, Lenient: {data['accuracy_lenient']:.1%}")
    print(f"Upgrades: {ups}, Downgrades: {downs}")

if __name__ == "__main__":
    main()
