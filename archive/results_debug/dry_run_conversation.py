#!/usr/bin/env python
"""Dry run: test run_conversation with 2 turns on a single character."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import (
    GroupConfig, run_conversation, load_character_sheets, _write_live_state,
)
from strategies.semantic_reranker import SemanticRerankerStrategy
from context.window import WindowContext

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"


async def main():
    # Load just conv_0, just Caroline, first 12 facts (2 turns of 6)
    sheets = load_character_sheets()
    dry_sheets = {
        "conv_0": {"Caroline": sheets["conv_0"]["Caroline"][:12]}
    }
    print(f"Dry run: {sum(len(f) for c in dry_sheets.values() for f in c.values())} facts")

    data_dir = "runs/dry_run_conversation"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    config = GroupConfig(
        name="dry_run",
        strategy_factory=lambda d: SemanticRerankerStrategy(persist_dir=d),
        context_factory=lambda: WindowContext(window_size=4),
    )

    live_state = {
        "current_group": "dry_run",
        "groups": {"dry_run": {}},
        "feed": [],
        "updated_at": time.time(),
    }

    print("\nStarting dry run...\n")
    results, strategy = await run_conversation(
        config=config,
        character_sheets=dry_sheets,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        data_dir=data_dir,
        live_state=live_state,
    )

    print(f"\n{'='*60}")
    print(f"DRY RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Turns: {results.turns}")
    print(f"Memories stored: {results.memories_stored}")
    print(f"Avg retrieval: {results.avg_retrieval_ms():.0f}ms")
    print(f"\nTurn log:")
    for t in results.turn_log:
        print(f"  Turn {t['turn']}: {t['character']} | "
              f"{t['facts_batch']} facts | "
              f"{t['memories_surfaced']} mems | "
              f"{t['exchange_outcome']}")
        print(f"    B: {t['b_preview'].encode('ascii', 'replace').decode()}")
        print(f"    A: {t['a_preview'].encode('ascii', 'replace').decode()}")

    print(f"\nLive state feed:")
    for f in live_state.get("feed", []):
        print(f"  {f.get('character', '?')}: {f.get('query', '')[:80].encode('ascii', 'replace').decode()}")

    await strategy.cleanup() if hasattr(strategy, 'cleanup') else None


if __name__ == "__main__":
    asyncio.run(main())
