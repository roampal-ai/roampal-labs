#!/usr/bin/env python3
"""
Re-grade exam transcripts using MiniMax-M1 (reasoning model).
Only re-grades 'wrong' and 'partial' from 20B grader. Checkpoints every 50. Resumable.

Usage:
  set MINIMAX_API_KEY=sk-api-...
  python results/minimax_regrader.py results/exam_04.EntityRouted_1616t.json
  python results/minimax_regrader.py results/exam_*.json
"""
import json
import sys
import os
import time
import urllib.request
import urllib.error

API_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
MODEL = "MiniMax-M2.7"

GRADING_PROMPT = """Compare the student's answer to the ground truth answer for this exam question.

Question: {question}
Ground Truth: {ground_truth}
Student Answer: {answer}

Grading rules:
- CORRECT if the answer contains the key facts from the ground truth, even if phrased differently or with extra detail
- CORRECT if dates are equivalent (e.g., "December 13, 2023" = "Thursday before December 17, 2023" if the dates actually match)
- CORRECT if numbers match in different formats ("4 years" = "four years")
- PARTIAL if the answer has some correct information but is missing key parts of the ground truth
- WRONG if the answer contradicts the ground truth, gives entirely different facts, or says "I don't know" when the ground truth has a real answer
- WRONG if the answer is about the wrong topic

Respond with exactly one word: correct, partial, or wrong"""


def _extract_grade(raw: str) -> str:
    """Extract correct/partial/wrong from API response, including think-tag fallback."""
    import re
    # Try 1: text after </think> tags
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().lower()
    if not text and "</think>" in raw:
        text = raw.split("</think>")[-1].strip().lower()

    if text:
        if "correct" in text: return "correct"
        if "partial" in text: return "partial"
        if "wrong" in text: return "wrong"

    # Try 2: extract conclusion from INSIDE <think> tags
    if "<think>" in raw:
        think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).lower()
            for pattern in [r"so it'?s (correct|partial|wrong)", r"the answer is (correct|partial|wrong)",
                            r"grade:? (correct|partial|wrong)", r"verdict:? (correct|partial|wrong)",
                            r"judgment:? (correct|partial|wrong)", r"(correct|wrong|partial)\.?\s*$"]:
                m = re.search(pattern, thinking)
                if m:
                    return m.group(1)
            # Last resort: which grade word appears last in the thinking
            last_c = thinking.rfind("correct")
            last_p = thinking.rfind("partial")
            last_w = thinking.rfind("wrong")
            best = max([(last_c, "correct"), (last_p, "partial"), (last_w, "wrong")])
            if best[0] > -1:
                return best[1]

    return "unknown"


def call_minimax(prompt: str, api_key: str, max_retries: int = 5) -> str:
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0
    }).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw = result["choices"][0]["message"].get("content", "").strip()
                grade = _extract_grade(raw)
                if grade != "unknown":
                    return grade
                # Unknown — retry if we have attempts left
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "unknown"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(60, 10 * (attempt + 1))
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            body = e.read().decode("utf-8")[:200] if hasattr(e, "read") else ""
            print(f"  API error {e.code}: {body}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "error"
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return "error"

    return "error"


def grade_exam(exam_path: str, api_key: str):
    with open(exam_path, encoding="utf-8") as f:
        data = json.load(f)

    transcript = data["transcript"]
    total = len(transcript)

    base = os.path.basename(exam_path).replace("exam_", "minimax_grade_")
    out_path = os.path.join(os.path.dirname(exam_path), base)

    checkpoint_path = out_path + ".checkpoint"
    per_question = []
    done_indices = set()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            per_question = json.load(f)
        done_indices = {q["index"] for q in per_question}
        print(f"Resuming: {len(done_indices)} already graded")

    # Grade ALL questions via API — no shortcuts, catches false positives too
    to_grade = [(i, t) for i, t in enumerate(transcript) if i not in done_indices]

    print(f"Grading {os.path.basename(exam_path)}")
    print(f"  Total: {total}, Already done: {len(done_indices)}")
    print(f"  To grade via MiniMax-M2.7: {len(to_grade)}")

    WORKERS = 4
    api_calls = 0
    consecutive_errors = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def grade_one(i, entry):
        prompt = GRADING_PROMPT.format(
            question=entry["question"],
            ground_truth=entry["ground_truth"],
            answer=entry["answer"][:400]
        )
        mm_j = call_minimax(prompt, api_key)
        orig_j = entry.get("judgment", "unknown")
        changed = mm_j != orig_j
        change_type = None
        if changed:
            rank = {"wrong": 0, "partial": 1, "correct": 2}
            if rank.get(mm_j, -1) > rank.get(orig_j, -1):
                change_type = "upgrade"
            elif rank.get(mm_j, -1) < rank.get(orig_j, -1):
                change_type = "downgrade"
            else:
                change_type = "lateral"
        return {
            "index": i,
            "category": entry.get("category", ""),
            "minimax_judgment": mm_j,
            "original_judgment": orig_j,
            "changed": changed,
            "change_type": change_type,
            "api_called": True
        }

    # Process in batches of WORKERS to allow clean early exit
    batch_done = 0
    abort = False
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for chunk_start in range(0, len(to_grade), WORKERS):
            if abort:
                break
            chunk = to_grade[chunk_start:chunk_start + WORKERS]
            futures = {pool.submit(grade_one, i, entry): i for i, entry in chunk}

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                api_calls += 1
                batch_done += 1

                if result["minimax_judgment"] == "error":
                    consecutive_errors += 1
                    print(f"  Skipping Q{idx} (API error, {consecutive_errors} consecutive)", file=sys.stderr)
                    if consecutive_errors >= 5:
                        print(f"  STOPPING: 5 consecutive API errors. Checkpointing.", file=sys.stderr)
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump(per_question, f)
                        print(f"  Saved checkpoint: {len(per_question)} questions. Re-run to resume.", file=sys.stderr)
                        abort = True
                        break
                    continue
                consecutive_errors = 0

                per_question.append(result)
                done_indices.add(idx)

            if abort:
                break

            if batch_done % 50 < WORKERS:  # crossed a 50-boundary in this chunk
                graded_api = [q for q in per_question if q.get("api_called")]
                ups = sum(1 for q in graded_api if q.get("change_type") == "upgrade")
                downs = sum(1 for q in graded_api if q.get("change_type") == "downgrade")
                print(f"  {batch_done}/{len(to_grade)} — ups: {ups}, downs: {downs}, api_calls: {api_calls}")
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(per_question, f)

    if abort:
        return None

    # Auto-retry unknowns (up to 3 passes)
    for retry_pass in range(3):
        unknown_qs = [(i, q) for i, q in enumerate(per_question) if q.get("minimax_judgment") == "unknown"]
        if not unknown_qs:
            break
        print(f"\n  Retry pass {retry_pass + 1}: {len(unknown_qs)} unknowns remaining...")
        retry_fixed = 0
        for ui, (qi, uq) in enumerate(unknown_qs):
            entry = transcript[uq["index"]]
            prompt = GRADING_PROMPT.format(
                question=entry.get("question", entry.get("query", "")),
                ground_truth=entry.get("ground_truth", ""),
                answer=entry.get("answer", "")[:400]
            )
            mm_j = call_minimax(prompt, api_key)
            api_calls += 1
            if mm_j not in ("unknown", "error"):
                orig_j = uq["original_judgment"]
                per_question[qi]["minimax_judgment"] = mm_j
                per_question[qi]["changed"] = mm_j != orig_j
                if mm_j != orig_j:
                    rank = {"wrong": 0, "partial": 1, "correct": 2}
                    if rank.get(mm_j, -1) > rank.get(orig_j, -1):
                        per_question[qi]["change_type"] = "upgrade"
                    elif rank.get(mm_j, -1) < rank.get(orig_j, -1):
                        per_question[qi]["change_type"] = "downgrade"
                    else:
                        per_question[qi]["change_type"] = "lateral"
                else:
                    per_question[qi]["change_type"] = None
                retry_fixed += 1
            if (ui + 1) % 20 == 0:
                print(f"    {ui+1}/{len(unknown_qs)} — fixed: {retry_fixed}")
        print(f"  Retry pass {retry_pass + 1} done: fixed {retry_fixed}/{len(unknown_qs)}")
        # Checkpoint after each retry pass
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(per_question, f)

    remaining_unknowns = sum(1 for q in per_question if q.get("minimax_judgment") == "unknown")
    if remaining_unknowns:
        print(f"\n  WARNING: {remaining_unknowns} questions still unknown after all retries")

    # Final
    correct = sum(1 for q in per_question if q.get("minimax_judgment") == "correct")
    partial = sum(1 for q in per_question if q.get("minimax_judgment") == "partial")
    wrong = sum(1 for q in per_question if q.get("minimax_judgment") == "wrong")
    disagreements = sum(1 for q in per_question if q.get("changed"))
    upgrades = sum(1 for q in per_question if q.get("change_type") == "upgrade")
    downgrades = sum(1 for q in per_question if q.get("change_type") == "downgrade")

    orig_acc = data["summary"]["accuracy"]

    results = {
        "grader": "MiniMax-M2.7",
        "source_exam": os.path.basename(exam_path),
        "total": total,
        "api_calls": api_calls,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "accuracy_strict": round(correct / total, 4) if total else 0,
        "accuracy_lenient": round((correct + partial) / total, 4) if total else 0,
        "original_accuracy": orig_acc,
        "disagreements": disagreements,
        "upgrades": upgrades,
        "downgrades": downgrades,
        "per_question": sorted(per_question, key=lambda q: q["index"])
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nDone: {os.path.basename(out_path)}")
    print(f"  Original:  {orig_acc:.1%}")
    print(f"  MiniMax:   {results['accuracy_strict']:.1%} strict, {results['accuracy_lenient']:.1%} lenient")
    print(f"  Disagree:  {disagreements} ({disagreements/total:.1%})")
    print(f"  Upgrades:  {upgrades}, Downgrades: {downgrades}")
    print(f"  API calls: {api_calls}")

    return results


if __name__ == "__main__":
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("Set MINIMAX_API_KEY environment variable")
        print("  Windows: set MINIMAX_API_KEY=sk-api-...")
        print("  Linux:   export MINIMAX_API_KEY=sk-api-...")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python minimax_regrader.py <exam_file.json> [...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if os.path.exists(path):
            grade_exam(path, api_key)
        else:
            print(f"File not found: {path}")
