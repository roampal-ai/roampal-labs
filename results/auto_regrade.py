#!/usr/bin/env python3
"""
Auto-regrade: watches for new exam transcripts and runs minimax regrading.
Checks every 60 seconds. Skips already-regraded files.

Usage:
  set MINIMAX_API_KEY=sk-api-...
  python results/auto_regrade.py
"""
import os
import sys
import time
import glob
import subprocess

RESULTS_DIR = "results"
POLL_INTERVAL = 60  # seconds


def get_exam_files():
    """Find all exam transcript files."""
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "exam_*.json")))


def get_graded_files():
    """Find all already-regraded files."""
    return {os.path.basename(f).replace("minimax_grade_", "exam_")
            for f in glob.glob(os.path.join(RESULTS_DIR, "minimax_grade_*.json"))}


def main():
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("MINIMAX_API_KEY not set")
        sys.exit(1)

    print("Auto-regrader started. Watching for new exam transcripts...", flush=True)

    while True:
        exam_files = get_exam_files()
        already_graded = get_graded_files()

        for exam_path in exam_files:
            basename = os.path.basename(exam_path)
            if basename in already_graded:
                continue

            # New exam found — regrade it
            print(f"\n{'='*60}", flush=True)
            print(f"New exam found: {basename}", flush=True)
            print(f"Starting MiniMax regrading...", flush=True)
            print(f"{'='*60}", flush=True)

            result = subprocess.run(
                [sys.executable, "-u", os.path.join(RESULTS_DIR, "minimax_regrader.py"), exam_path],
                env={**os.environ, "MINIMAX_API_KEY": api_key, "PYTHONUTF8": "1"},
                capture_output=False,
            )

            if result.returncode == 0:
                print(f"Regrading complete: {basename}", flush=True)
            else:
                print(f"Regrading failed (exit {result.returncode}): {basename}", flush=True)

        # Check if pipeline is done (all 4 strategies × 2 exams = 8 files)
        if len(exam_files) >= 8 and len(already_graded | {os.path.basename(f) for f in exam_files}) <= len(already_graded) + len(exam_files):
            remaining = set(os.path.basename(f) for f in exam_files) - already_graded
            if not remaining:
                print(f"\nAll {len(exam_files)} exams regraded. Done!", flush=True)
                break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
