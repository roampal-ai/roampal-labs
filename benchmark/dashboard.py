"""Live benchmark dashboard — watches results/live_state.json and renders progress.

Usage:
    python -m benchmark.dashboard

Refreshes every 2 seconds. Shows:
- Current group and turn progress
- Accuracy table across all groups
- Live turn log for current group
- Elapsed time and ETA
"""

import json
import os
import sys
import time
from pathlib import Path


# ANSI codes for terminal formatting
CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"


def render(state: dict):
    """Render the dashboard from live state."""
    lines = []

    # Header
    benchmark = state.get("benchmark", "?")
    model = state.get("model", "?")
    status = state.get("status", "running")
    started = state.get("started_at", 0)
    elapsed = time.time() - started if started else 0
    elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"

    completed = state.get("completed_groups", 0)
    total_groups = state.get("total_groups", 18)
    current = state.get("current_group", "")
    current_turn = state.get("current_turn", 0)
    total_turns = state.get("total_turns", 200)

    current_step = state.get("current_step", "")
    valid_groups = {
        "02.TagCascade", "03.CE-Only", "04.EntityRouter", "baseline_raw_repaired",
        "gemma4_02.TagCascade_clean", "gemma4_02.TagCascade_poison",
        "gemma4_03.CE-Only_clean", "gemma4_03.CE-Only_poison",
        "4omini_02.TagCascade_clean", "4omini_03.CE-Only_clean", "4omini_baseline",
        "4omini_02.TagCascade_poison", "4omini_03.CE-Only_poison",
        "no_memory",
    }

    lines.append(f"{BOLD}{CYAN}ROAMPAL-LABS BENCHMARK{RESET}")
    lines.append(f"{DIM}{benchmark} | {model} | {elapsed_str} elapsed{RESET}")
    if status == "complete":
        lines.append(f"{GREEN}{BOLD}COMPLETE{RESET}")
    else:
        # Show current strategy + step
        step_line = f"  {BOLD}{current}{RESET}"
        if current_step:
            step_line += f"  |  {CYAN}{current_step}{RESET}"
        if current_turn and total_turns:
            step_line += f"  |  turn {current_turn}/{total_turns}"
        lines.append(step_line)

        # Show exam progress if actively running (not stale)
        for name, g in state.get("groups", {}).items():
            if name not in valid_groups:
                continue
            live_exam = g.get("exam")
            if live_exam and live_exam.get("total", 0) > 0:
                progress = live_exam.get("progress", "")
                # Skip stale exams — only show if genuinely in progress
                is_active = False
                if progress:
                    try:
                        done_count, total_count = progress.split("/")
                        is_active = int(done_count) < int(total_count)
                    except (ValueError, TypeError):
                        is_active = True
                if not is_active:
                    continue
                # Only show live exam for the currently active group
                if name != current:
                    continue
                ec = live_exam.get("correct", 0)
                ew = live_exam.get("wrong", 0)
                et = live_exam.get("total", 0)
                ea = live_exam.get("accuracy", 0)
                by_cat = live_exam.get("by_category", {})
                adv_str = ""
                if by_cat:
                    adv = by_cat.get("adversarial", {})
                    adv_c = adv.get("correct", 0)
                    adv_t = adv.get("total", 0)
                    non_c = ec - adv_c
                    non_t = et - adv_t
                    adv_acc = adv_c / adv_t if adv_t else 0
                    non_acc = non_c / non_t if non_t else 0
                    adv_str = f"  non-adv:{non_acc:.1%} adv:{adv_acc:.1%}"
                if progress:
                    lines.append(f"  {YELLOW}EXAM [{name}]: {progress}  {ec}c/{ew}w  {ea:.1%}{adv_str}{RESET}")

    lines.append("")

    # Results table
    groups = state.get("groups", {})
    if groups:
        # Header
        lines.append(f"{BOLD}{UNDERLINE}{'Group':<22} {'C':>4} {'P':>4} {'W':>4} {'?':>4} {'Acc':>7} {'Turns':>8} {'ms':>7} {'Ctx':>5}{RESET}")

        for name, g in groups.items():
            if name not in valid_groups:
                continue
            c = g.get("correct", 0)
            p = g.get("partial", 0)
            w = g.get("wrong", 0)
            u = g.get("unknown", 0)
            turns = g.get("turns", 0)
            acc = g.get("accuracy", 0)
            ms = g.get("avg_retrieval_ms", 0)
            ctx = g.get("context_exchanges", 0)
            done = g.get("done", False)
            elapsed_g = g.get("elapsed_s", 0)

            # Color the accuracy
            if acc >= 0.08:
                acc_str = f"{GREEN}{acc:>6.1%}{RESET}"
            elif acc >= 0.05:
                acc_str = f"{YELLOW}{acc:>6.1%}{RESET}"
            else:
                acc_str = f"{RED}{acc:>6.1%}{RESET}"

            status_icon = f"{GREEN}+{RESET}" if done else f"{CYAN}>{RESET}" if name == current else " "
            turn_str = f"{turns:>8}" if done else f"{turns:>8}"

            lines.append(f"{status_icon} {name:<20} {c:>4} {p:>4} {w:>4} {u:>4} {acc_str} {turn_str} {ms:>6.0f} {ctx:>5}")

            # Exam history (completed exams with step names)
            exam_history = g.get("exam_history", [])
            if exam_history:
                for eh in exam_history:
                    ec = eh.get("correct", 0)
                    ep = eh.get("partial", 0)
                    ew = eh.get("wrong", 0)
                    et = eh.get("total", 0)
                    ea = eh.get("accuracy", 0)
                    step = eh.get("step", eh.get("at_turns", "?"))
                    learn = eh.get("learning", None)
                    learn_str = " ON" if learn else " OFF" if learn is not None else ""
                    ea_str = f"{GREEN}{ea:>6.1%}{RESET}" if ea >= 0.5 else f"{YELLOW}{ea:>6.1%}{RESET}" if ea >= 0.3 else f"{RED}{ea:>6.1%}{RESET}"
                    # Adversarial vs non-adversarial breakdown (only if adversarial questions exist)
                    by_cat = eh.get("by_category", {})
                    adv_str = ""
                    if by_cat:
                        adv = by_cat.get("adversarial", {})
                        adv_c = adv.get("correct", 0)
                        adv_t = adv.get("total", 0)
                        if adv_t > 0:
                            non_c = ec - adv_c
                            non_t = et - adv_t
                            adv_acc = adv_c / adv_t if adv_t else 0
                            non_acc = non_c / non_t if non_t else 0
                            adv_color = GREEN if adv_acc >= 0.5 else YELLOW if adv_acc >= 0.3 else RED
                            non_color = GREEN if non_acc >= 0.5 else YELLOW if non_acc >= 0.3 else RED
                            adv_str = f"  {non_color}non-adv:{non_acc:>5.1%}{RESET} {adv_color}adv:{adv_acc:>5.1%}{RESET}"
                    # Label poison vs clean
                    if step.startswith("poison_"):
                        label = f"{RED}POISON{RESET} {step[7:]}"
                    else:
                        label = step
                    lines.append(f"  {DIM}{label}{learn_str}:{RESET}  {ec:>4}c {ep:>4}p {ew:>4}w  {ea_str}  ({et} Qs){adv_str}")
            # Show live in-progress exam only if it's genuinely in progress
            # (not a stale snapshot from a completed exam)
            live_exam = g.get("exam")
            show_live = False
            if live_exam and live_exam.get("total", 0) > 0:
                progress = live_exam.get("progress", "")
                if progress:
                    try:
                        done_count, total_count = progress.split("/")
                        # Still in progress if not at the end
                        show_live = int(done_count) < int(total_count)
                    except (ValueError, TypeError):
                        show_live = True
                else:
                    show_live = True
                # Hide if this group isn't the currently active one
                if name != current:
                    show_live = False
            if show_live:
                ec = live_exam.get("correct", 0)
                ep = live_exam.get("partial", 0)
                ew = live_exam.get("wrong", 0)
                et = live_exam.get("total", 0)
                ea = live_exam.get("accuracy", 0)
                ea_str = f"{YELLOW}{ea:>6.1%}{RESET}"
                by_cat = live_exam.get("by_category", {})
                adv_str = ""
                if by_cat:
                    adv = by_cat.get("adversarial", {})
                    adv_c = adv.get("correct", 0)
                    adv_t = adv.get("total", 0)
                    if adv_t > 0:
                        non_c = ec - adv_c
                        non_t = et - adv_t
                        adv_acc = adv_c / adv_t if adv_t else 0
                        non_acc = non_c / non_t if non_t else 0
                        adv_color = GREEN if adv_acc >= 0.5 else YELLOW if adv_acc >= 0.3 else RED
                        non_color = GREEN if non_acc >= 0.5 else YELLOW if non_acc >= 0.3 else RED
                        adv_str = f"  {non_color}non-adv:{non_acc:>5.1%}{RESET} {adv_color}adv:{adv_acc:>5.1%}{RESET}"
                # Show total for the correct exam type
                exam_total = 1986 if et > 100 else 76
                # Label as POISON if this group is being actively tested in a poison benchmark
                bench = state.get("benchmark", "")
                is_poison_exam = "POISON" in bench.upper() and name == current
                live_label = f"{RED}POISON (live){RESET}" if is_poison_exam else "EXAM (live)"
                lines.append(f"  {DIM}{live_label}:{RESET}          {ec:>4} {ep:>4} {ew:>4}      {ea_str} {et}/{exam_total}{adv_str}")
            elif g.get("exam") and not exam_history:
                # Fallback: show single exam if no history yet
                fallback_exam = g.get("exam")
                ec = fallback_exam.get("correct", 0)
                ep = fallback_exam.get("partial", 0)
                ew = fallback_exam.get("wrong", 0)
                et = fallback_exam.get("total", 0)
                ea = fallback_exam.get("accuracy", 0)
                ea_str = f"{GREEN}{ea:>6.1%}{RESET}" if ea >= 0.5 else f"{YELLOW}{ea:>6.1%}{RESET}" if ea >= 0.3 else f"{RED}{ea:>6.1%}{RESET}"
                lines.append(f"  {DIM}EXAM:{RESET}                {ec:>4} {ep:>4} {ew:>4}      {ea_str} {et:>4} Qs")
                # Per-category breakdown for latest
                by_cat = fallback_exam.get("by_category", {})
                if by_cat:
                    for cat, stats in sorted(by_cat.items()):
                        cc = stats.get("correct", 0)
                        cw = stats.get("wrong", 0)
                        ct = stats.get("total", 0)
                        ca = cc / ct if ct else 0
                        ca_color = GREEN if ca >= 0.5 else YELLOW if ca >= 0.3 else RED
                        lines.append(f"    {DIM}{cat:<18}{RESET} {cc:>3}c/{cw:>3}w  {ca_color}{ca:>5.1%}{RESET}")

    lines.append("")

    # ETA
    if elapsed > 60 and completed < total_groups:
        avg_per_group = elapsed / max(completed + (current_turn / total_turns), 0.1)
        remaining = avg_per_group * (total_groups - completed - current_turn / total_turns)
        eta_str = f"{int(remaining//3600)}h {int((remaining%3600)//60)}m"
        lines.append(f"{DIM}ETA: ~{eta_str} remaining{RESET}")

    lines.append("")

    # Live message feed — show full conversation
    feed = state.get("feed", [])
    if feed:
        lines.append(f"{BOLD}--- Live Feed ---{RESET}")
        for msg in feed:
            turn = msg.get("turn", 0)
            grp = msg.get("group", "")
            query = msg.get("query", "")
            ans = msg.get("answer", "")
            j = msg.get("judgment", "")
            fu = msg.get("followup", "")
            mems = msg.get("memories", 0)
            ms = msg.get("retrieval_ms", 0)

            j_color = GREEN if j == "correct" else RED if j == "wrong" else YELLOW if j == "partial" else DIM

            lines.append(f"{DIM}[{grp} t{turn}] ({mems} mems, {ms:.0f}ms){RESET}")
            # Wrap long text to ~100 visible chars (ANSI codes don't count)
            import re
            def visible_len(s):
                return len(re.sub(r'\033\[[0-9;]*m', '', s))

            try:
                term_width = os.get_terminal_size().columns - 2
            except (ValueError, OSError):
                term_width = 118
            def wrap(text, prefix, width=0):
                width = width or term_width
                words = text.split()
                if not words:
                    return [prefix]
                lines_out = []
                current = prefix + words[0]
                indent = "       "
                for w in words[1:]:
                    if visible_len(current) + len(w) + 1 > width:
                        lines_out.append(current)
                        current = indent + w
                    else:
                        current += " " + w
                lines_out.append(current)
                return lines_out

            for wl in wrap(query, f"  {CYAN}User:{RESET} "):
                lines.append(wl)
            for wl in wrap(ans, f"  {BOLD}LLM:{RESET}  "):
                lines.append(wl)
            if fu:
                char = msg.get("character", "")
                label = f"{char}" if char else "Reaction"
                for wl in wrap(fu, f"  {j_color}{label} [{j}]:{RESET} "):
                    lines.append(wl)
            lines.append("")

    return "\n".join(lines)


def main():
    state_file = Path("results/live_state.json")

    print(f"Watching {state_file}...")
    print("(Start the benchmark in another terminal: python -m benchmark.runner)")
    print()

    last_mtime = 0
    while True:
        try:
            if state_file.exists():
                mtime = state_file.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    sys.stdout.write(CLEAR)
                    sys.stdout.write(render(state))
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                    # Don't exit on "complete" — run_full.py resets status between passes
                    # Dashboard just keeps watching
            time.sleep(2)
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(2)


if __name__ == "__main__":
    main()
