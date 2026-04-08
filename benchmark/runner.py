"""Main benchmark orchestrator.

Runs all strategy x context x rewrite combinations and produces results.
Writes live state to results/live_state.json for the dashboard UI.

Usage:
    python -m benchmark.runner --turns 200 --dataset data/fictional_benchmark_data.json
"""

import asyncio
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.semantic_reranker import SemanticRerankerStrategy
from strategies.wilson_scored import WilsonScoredStrategy
from strategies.wilson_reranker import WilsonRerankerStrategy
# from strategies.kg_traversal import KGTraversalStrategy  # archived
from strategies.entity_routed import EntityRoutedStrategy
from context.window import WindowContext
from benchmark.grader import LLMGrader


@dataclass
class GroupConfig:
    name: str
    strategy_factory: Any   # Callable that creates a MemoryStrategy
    context_factory: Any    # Callable that creates a ContextManager
    use_rewrite: bool = False
    extract_facts: bool = True  # Set False to skip atomic fact extraction


@dataclass
class GroupResults:
    name: str
    turns: int = 0
    correct: int = 0
    partial: int = 0
    wrong: int = 0
    unknown: int = 0
    total_tokens: int = 0
    memories_stored: int = 0
    compactions: int = 0
    retrieval_ms_total: float = 0.0
    turn_log: List[Dict] = field(default_factory=list)

    def graded(self):
        return self.correct + self.partial + self.wrong

    def accuracy(self):
        return self.correct / self.graded() if self.graded() else 0

    def avg_retrieval_ms(self):
        return self.retrieval_ms_total / self.turns if self.turns else 0


# Map grader judgments to sidecar outcome format
# The grader acts as the simulated human. The sidecar interprets human reactions.
# correct -> worked, wrong -> failed (same mapping the real sidecar does)
OUTCOME_MAP = {"correct": "worked", "wrong": "failed", "partial": "partial", "unknown": "unknown"}


def _save_exam_transcript(group_name: str, exam_type: str, at_turns: int, summary: dict, results: dict):
    """Save exam transcript to results/ for full audit trail."""
    try:
        filename = f"{exam_type}_{group_name}_{at_turns}t.json"
        out = Path("results") / filename
        with open(out, "w", encoding="utf-8") as ef:
            json.dump({
                "group": group_name,
                "at_turns": at_turns,
                "exam_type": exam_type,
                "summary": summary,
                "transcript": results.get("log", []),
            }, ef, indent=2, ensure_ascii=False)
        print(f"    Transcript saved to {out}", flush=True)
    except Exception as e:
        print(f"    WARNING: Failed to save transcript: {e}", flush=True)


def _write_live_state(state: dict):
    """Write live state to JSON for dashboard UI (atomic write to prevent read races)."""
    try:
        out = Path("results/live_state.json")
        tmp = Path("results/live_state.json.tmp")
        out.parent.mkdir(exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=True)
        tmp.replace(out)  # Atomic on most filesystems
    except Exception:
        pass


async def rewrite_query(client: "httpx.AsyncClient", model: str, query: str, context_msgs: list) -> str:
    """Sidecar query rewriting — reformulates the user query for better retrieval."""
    recent = ""
    if context_msgs:
        for msg in context_msgs[-4:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:150]
            recent += f"{role}: {content}\n"

    prompt = f"""Rewrite the user's message as a concise search query for a memory retrieval system.
Keep key entities, names, and specific terms. Remove conversational filler.
If the message references prior context, expand pronouns and references using the conversation history.

Conversation context:
{recent}

User message: {query}

Rewritten search query (one line, no quotes):"""

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You rewrite user messages into concise search queries. Output ONLY the rewritten query, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0,
            }),
            timeout=30,
        )
        data = resp.json()
        rewritten = data["choices"][0]["message"].get("content", "").strip()
        if rewritten and len(rewritten) < 500:
            return rewritten
    except Exception:
        pass
    return query


async def sidecar_summarize(client: "httpx.AsyncClient", model: str, user_msg: str, assistant_msg: str) -> str:
    """Sidecar exchange summary — richer than dumb concatenation."""
    prompt = f"""Summarize this exchange in 1-2 sentences, capturing the key facts discussed.
Focus on specific names, numbers, decisions, and outcomes. Write as a memory note, not a conversation recap.

User: {user_msg[:300]}
Assistant: {assistant_msg[:500]}

Summary:"""

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You write concise memory summaries. Output ONLY the summary, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0,
            }),
            timeout=30,
        )
        data = resp.json()
        summary = data["choices"][0]["message"].get("content", "").strip()
        if summary:
            return summary
    except Exception:
        pass
    return f"User asked: {user_msg[:200]}. Assistant answered: {assistant_msg[:300]}"


async def sidecar_extract_facts(client: "httpx.AsyncClient", model: str, user_msg: str, assistant_msg: str) -> List[str]:
    """Extract individual atomic facts from an exchange. Only facts the sidecar is confident about."""
    prompt = f"""Extract key facts worth remembering from this exchange. Rules:
- Include WHO or WHAT each fact is about — names, projects, topics
- Combine related details into ONE rich fact rather than many fragments
- Include specifics: dates, versions, preferences, decisions, reasons
- Capture what can be inferred, not just what was explicitly said
- ONE fact per line, max 150 characters
- Skip vague feelings, pleasantries, or generic observations

GOOD: "The auth service uses JWT with 24h expiry, needs refresh token rotation added"
GOOD: "User prefers TypeScript over JavaScript and uses Zod for validation"
GOOD: "Chapter 3 draft needs more dialogue per editor feedback, focus on protagonist's childhood"
GOOD: "Lakers won 112-108 on March 5, LeBron scored 34 — user's favorite player"
GOOD: "Sourdough starter day 4, feeds every 12h at room temp, first bake planned for Saturday"
GOOD: "User's daughter Emma starts kindergarten in September, worried about the bus route"
BAD: "They discussed something" (no specifics)
BAD: "It was helpful" (no content)
BAD: "The user asked a question" (meta, not a fact)

User: {user_msg[:500]}
Assistant: {assistant_msg[:300]}

Output one fact per line. No bullets, no numbering. If no useful facts, output NONE."""

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Extract key facts. One fact per line. Include specifics. Skip vague observations."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0,
            }),
            timeout=30,
        )
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "").strip()
        if not content or content.upper() == "NONE":
            return []
        facts = [f.strip().lstrip("•-*0123456789. ") for f in content.split("\n") if f.strip() and f.strip().upper() != "NONE"]
        return [f for f in facts if len(f) > 10]  # Filter garbage
    except Exception:
        return []


async def sidecar_score(
    client: "httpx.AsyncClient",
    model: str,
    user_msg: str,
    assistant_msg: str,
    memories: list,
    followup: str = "",
) -> dict:
    """Score each memory individually — exact same prompt as real roampal-cli sidecar.

    Returns {"exchange_outcome": "worked|failed|partial|unknown",
             "memory_scores": {"doc_id": "worked|failed|partial|unknown", ...},
             "exchange_summary": "..."}
    """
    memory_block = "\n".join(
        f"  [{m.id}] {m.content}" for m in memories
    )
    followup_section = f'\nThe user then followed up with:\n"{followup[:500]}"' if followup else ""

    # Exact same prompt from roampal-cli/roampal/code/sidecar.py _build_scoring_prompt
    prompt = f"""The user said:
"{user_msg[:800]}"

You responded:
"{assistant_msg[:800]}"
{followup_section}

Memories that were surfaced for this exchange:
{memory_block if memory_block else "  (none)"}

Respond with ONLY a JSON object:
{{
  "exchange_summary": "<~300 chars>",
  "exchange_outcome": "worked|failed|partial|unknown",
  "memory_scores": {{
    "<memory_id>": "worked|failed|partial|unknown"
  }}
}}

OUTCOME: Based on the user's follow-up:
- worked: user confirmed, moved on, or was satisfied
- failed: user corrected you, got frustrated, or asked to redo
- partial: helped but incomplete or needed adjustment
- unknown: no clear signal (or no follow-up yet)

MEMORY SCORES: For each memory, judge based on topic relevance and exchange outcome.
1. Memory is NOT about the topic discussed -> "unknown"
2. Memory IS about the topic AND outcome is "worked" -> "worked"
3. Memory IS about the topic AND outcome is "failed":
   - Your response echoed/relied on info from this memory -> "failed"
   - The failure seems unrelated to this memory's content -> "unknown"
4. Memory IS about the topic AND outcome is "partial" -> "partial"
5. Your response contradicts what the memory says and the exchange worked -> "unknown"
6. Memory contains good advice the response IGNORED -> "unknown" not "failed"
7. When in doubt -> "unknown"

SUMMARY (under 300 chars): Capture what happened AND what changed. Summaries provide context and continuity — the story behind the facts.
- Include names, topics, and the flow of the conversation
- Note corrections, decisions, and new information alongside the context
- Help future retrieval understand WHY something matters, not just WHAT
BAD: "User and assistant had a conversation" (empty, no content)
BAD: "Temperature is 350F" (that's a fact, not a summary)
GOOD: "User corrected the baking temp from 375F to 350F while adapting the recipe for a convection oven — first attempt burned the edges"
GOOD: "Discussed switching API from REST to GraphQL after mobile team reported nested query issues with the current setup"
GOOD: "User shared that their daughter Emma starts kindergarten in September, worried about the bus route — asked for tips on easing the transition" """

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are part of a memory system. Respond ONLY with valid JSON. No explanation, no preamble."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0,
            }),
            timeout=60,
        )
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content", "")
        if not text.strip():
            text = msg.get("reasoning_content", "") or msg.get("reasoning", "")

        # Parse JSON (handle markdown code blocks)
        if "```" in text:
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0]
            else:
                text = text.split("```")[1].split("```")[0]

        import json as _json
        parsed = _json.loads(text.strip())
        return {
            "exchange_outcome": parsed.get("exchange_outcome", "unknown"),
            "memory_scores": parsed.get("memory_scores", {}),
            "exchange_summary": parsed.get("exchange_summary", ""),
        }
    except Exception:
        pass

    # Fallback: blanket score all memories with the grader judgment
    return {
        "exchange_outcome": "unknown",
        "memory_scores": {},
        "exchange_summary": f"User asked: {user_msg[:200]}. Assistant answered: {assistant_msg[:300]}",
    }


async def _run_single_turn(config, llm_client, llm_model, strategy, context, grader,
                           user_input, source_text, turn, results, retrieval_cls=None):
    """Execute one turn: retrieve, answer, grade, score. Returns tuple for caller."""

    # 1. Query rewriting (sidecar) — if enabled for this group
    search_query = user_input
    if config.use_rewrite:
        context_msgs = context.build_messages()
        search_query = await rewrite_query(llm_client, llm_model, user_input, context_msgs)

    # 2. Retrieve memories
    try:
        retrieval = await asyncio.wait_for(
            strategy.retrieve(search_query, top_k=4), timeout=30
        )
    except (asyncio.TimeoutError, Exception):
        from strategies.base import RetrievalResult
        retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=search_query, retrieval_ms=0)
    results.retrieval_ms_total += retrieval.retrieval_ms

    # 3. Build prompt with context + memories
    messages = context.build_messages()
    user_content = user_input
    if retrieval.formatted_injection:
        user_content = retrieval.formatted_injection + "\n\n" + user_input
    messages.append({"role": "user", "content": user_content})

    # 4. LLM answers
    answer = ""
    try:
        resp = await asyncio.wait_for(
            llm_client.post("/chat/completions", json={
                "model": llm_model,
                "messages": [{"role": "system", "content": "Answer concisely using any provided context."}] + messages,
                "max_tokens": 5000,
                "temperature": 0,
            }),
            timeout=90,
        )
        data = resp.json()
        msg = data["choices"][0]["message"]
        answer = msg.get("content", "")
        if not answer or not answer.strip():
            answer = msg.get("reasoning_content", "") or msg.get("reasoning", "")
    except Exception as e:
        print(f"  LLM failed {config.name} turn {turn}: {type(e).__name__}", flush=True)

    # 5. Grade
    if not answer or not answer.strip():
        judgment, followup = "wrong", "No response provided."
    else:
        judgment, followup = await grader.grade(user_input, answer, source_text)

    # 6. Record judgment
    if judgment == "correct":
        results.correct += 1
    elif judgment == "partial":
        results.partial += 1
    elif judgment == "wrong":
        results.wrong += 1
    else:
        results.unknown += 1
    results.turns += 1

    # 7. Store exchange in context (raw question + answer, NOT injected memories)
    context.add_exchange(user_input, answer)

    # 8. Sidecar scoring
    sidecar_result = await sidecar_score(
        llm_client, llm_model, user_input, answer,
        retrieval.memories, followup,
    )
    exchange_summary = sidecar_result.get("exchange_summary", "")
    if not exchange_summary:
        exchange_summary = f"User asked: {user_input[:200]}. Assistant answered: {answer[:300]}"
    per_memory_scores = sidecar_result.get("memory_scores", {})

    return answer, judgment, followup, retrieval, per_memory_scores, exchange_summary


async def run_group(
    config: GroupConfig,
    queries: List[Dict],
    grader: LLMGrader,
    llm_base_url: str,
    llm_model: str,
    num_turns: int,
    data_dir: str,
    live_state: dict,
    poison_memories: List[str] = None,
    resume_from: int = 0,
) -> GroupResults:
    """Run one benchmark group."""
    import httpx

    results = GroupResults(name=config.name)

    # Resume: restore results from live_state if resuming
    if resume_from > 0 and config.name in live_state.get("groups", {}):
        prev = live_state["groups"][config.name]
        results.correct = prev.get("correct", 0)
        results.partial = prev.get("partial", 0)
        results.wrong = prev.get("wrong", 0)
        results.unknown = prev.get("unknown", 0)
        results.turns = resume_from
        results.retrieval_ms_total = prev.get("avg_retrieval_ms", 0) * resume_from
        print(f"    Resuming from turn {resume_from} ({results.correct}c/{results.partial}p/{results.wrong}w)", flush=True)

    strategy = config.strategy_factory(data_dir)
    await strategy.initialize()
    context = config.context_factory()

    # Pre-seed poison memories if provided
    if poison_memories:
        print(f"    Pre-seeding {len(poison_memories)} poison memories...", flush=True)
        for pm in poison_memories:
            await strategy.store(pm)
        print(f"    Poison seeded.", flush=True)

    llm_client = httpx.AsyncClient(
        base_url=llm_base_url,
        headers={"Authorization": "Bearer ollama"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )

    for turn in range(num_turns):
        # Resume: skip already-completed turns (strategy state persists in ChromaDB)
        if turn < resume_from:
            continue

        q = queries[turn % len(queries)]
        user_input = q["query"]
        source_text = q["source"]

        # Hard per-turn timeout: skip entire turn if it takes > 120s
        # Prevents thinking model reasoning loops from blocking the run
        try:
            answer, judgment, followup, retrieval, per_memory_scores, exchange_summary = \
                await asyncio.wait_for(
                    _run_single_turn(
                        config, llm_client, llm_model, strategy, context, grader,
                        user_input, source_text, turn, results, retrieval_cls=None,
                    ),
                    timeout=120,
                )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            print(f"  TURN {turn} TIMEOUT (>120s) — skipping", flush=True)
            answer = ""
            judgment = "wrong"
            followup = "Turn timed out."
            from strategies.base import RetrievalResult
            retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=user_input, retrieval_ms=0)
            per_memory_scores = {}
            exchange_summary = f"User asked: {user_input[:200]}. Timed out."
            results.wrong += 1
            results.turns += 1
            context.add_exchange(user_input, "")

        mapped_outcome = OUTCOME_MAP.get(judgment, "unknown")

        # Record per-memory outcomes (outside hard timeout — these are fast)
        try:
            for m in retrieval.memories:
                if m.id:
                    mem_outcome = per_memory_scores.get(m.id, mapped_outcome)
                    if mem_outcome not in ("worked", "failed", "partial", "unknown"):
                        mem_outcome = mapped_outcome
                    await asyncio.wait_for(strategy.record_outcome(
                        memory_ids=[m.id],
                        outcome=mem_outcome,
                        exchange_summary="",
                    ), timeout=60)
            # Store exchange summary as new memory
            await asyncio.wait_for(strategy.record_outcome(
                memory_ids=[],
                outcome=mapped_outcome,
                exchange_summary=exchange_summary,
            ), timeout=60)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"    WARNING: record_outcome timeout: {e}", flush=True)

        # 10. Track tokens
        est_tokens = int(len(user_input.split()) * 1.3)
        results.total_tokens += est_tokens

        # 11. Log turn (includes per-memory scores for visibility)
        results.turn_log.append({
            "turn": turn,
            "query": user_input[:80],
            "outcome": judgment,
            "per_memory_scores": per_memory_scores,
            "mapped_outcome": mapped_outcome,
            "memories_surfaced": len(retrieval.memories),
            "retrieval_ms": round(retrieval.retrieval_ms, 1),
            "answer_preview": answer[:80],
            "followup_preview": followup[:60] if followup else "",
        })

        # 12. Update live state for dashboard (every turn for live feed)
        ctx_stats = context.get_stats()
        live_state["current_group"] = config.name
        live_state["current_turn"] = turn + 1
        live_state["total_turns"] = num_turns

        # Live message feed — last 5 exchanges for the dashboard
        if "feed" not in live_state:
            live_state["feed"] = []
        live_state["feed"].append({
            "group": config.name,
            "turn": turn + 1,
            "query": user_input,
            "answer": answer,
            "judgment": judgment,
            "followup": followup if followup else "",
            "memories": len(retrieval.memories),
            "retrieval_ms": round(retrieval.retrieval_ms, 1),
        })
        live_state["feed"] = live_state["feed"][-6:]  # Keep last 6

        # Preserve exam_history and other persistent fields when updating group
        existing_group = live_state.get("groups", {}).get(config.name, {})
        existing_group.update({
            "correct": results.correct,
            "partial": results.partial,
            "wrong": results.wrong,
            "unknown": results.unknown,
            "turns": results.turns,
            "accuracy": round(results.accuracy(), 4),
            "avg_retrieval_ms": round(results.avg_retrieval_ms(), 1),
            "context_exchanges": ctx_stats.get("exchanges_kept", 0),
            "compactions": ctx_stats.get("compactions", 0),
            "last_query": user_input[:60],
            "last_outcome": judgment,
        })
        live_state["groups"][config.name] = existing_group
        live_state["updated_at"] = time.time()
        _write_live_state(live_state)

        if (turn + 1) % 10 == 0:
            ctx_stats = context.get_stats()
            print(f"    {config.name}: turn {turn+1}/{num_turns} -- "
                  f"{results.correct}c/{results.partial}p/{results.wrong}w/{results.unknown}u "
                  f"| ctx:{ctx_stats.get('exchanges_kept', '?')}ex",
                  flush=True)

    # Cleanup LLM client (but NOT strategy — needed for exam phase)
    await llm_client.aclose()

    ctx_stats = context.get_stats()
    results.compactions = ctx_stats.get("compactions", 0)
    strat_stats = await strategy.get_stats() if hasattr(strategy, 'get_stats') else {}
    results.memories_stored = strat_stats.get("memories", 0)

    return results, strategy  # Return strategy for exam phase


# ─── Conversation-based learning (v2) ───────────────────────────────────────


def load_character_sheets(sheets_dir: str = "data/character_sheets") -> Dict[str, Dict[str, List[str]]]:
    """Load character sheets from JSON files. Returns {conv_key: {char_name: [facts]}}."""
    sheets = {}
    sheets_path = Path(sheets_dir)
    for f in sorted(sheets_path.glob("conv_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        conv_key = f"conv_{data['conversation']}"
        sheets[conv_key] = data["characters"]
    total = sum(len(facts) for chars in sheets.values() for facts in chars.values())
    print(f"Loaded {len(sheets)} conversations, {total} facts", flush=True)
    return sheets

async def _llm_b_speak(client, model, character_name, facts_batch, all_facts, recent_exchanges):
    """LLM B shares facts naturally as the character. Has full sheet but only shares the batch."""
    history = ""
    for ex in recent_exchanges[-2:]:
        history += f"You: {ex['b'][:100]}\nFriend: {ex['a'][:100]}\n"

    batch_text = "\n".join(f"- {f}" for f in facts_batch)
    prompt = f"""You are {character_name} catching up with a friend. Always begin your message with "Hey, it's {character_name}!" then share ONLY the facts listed below in 3-5 sentences.

SHARE THESE FACTS NOW:
{batch_text}

{("Recent:" + chr(10) + history) if history else ""}
IMPORTANT: Start with "Hey, it's {character_name}!" every time. Only talk about the facts listed above. Do not bring up other topics. 3-5 sentences. No emojis."""

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": f"You are {character_name}. Always introduce yourself by name at the start. Be brief. No emojis."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.7,
            }),
            timeout=60,
        )
        data = resp.json()
        return data["choices"][0]["message"].get("content", "").strip()
    except Exception as e:
        print(f"    LLM B failed: {type(e).__name__}", flush=True)
        return ""


async def _llm_b_react(client, model, character_name, all_facts, a_response, recent_exchanges):
    """LLM B reacts using full character knowledge. Confirms correct claims, corrects wrong ones."""
    # Give full character sheet so it can verify ANY claim from memory
    facts_text = "\n".join(f"- {f}" for f in all_facts)
    prompt = f"""You are {character_name}. Here is EVERYTHING about your life:
{facts_text}

Your friend just said: "{a_response[:400]}"

Reply in 1-3 sentences as {character_name}:
- If they referenced something about you that is CORRECT according to your facts, confirm it
- If they said something WRONG about you, gently correct them with the right info
- If they just responded supportively without claims, thank them briefly
Do NOT bring up new facts they didn't mention. Only react to what they said."""

    try:
        resp = await asyncio.wait_for(
            client.post("/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.7,
            }),
            timeout=60,
        )
        data = resp.json()
        msg = data["choices"][0]["message"]
        result = msg.get("content", "").strip()
        return result
    except Exception as e:
        print(f"    LLM B react failed: {type(e).__name__}", flush=True)
        return ""


async def run_conversation(
    config: GroupConfig,
    character_sheets: Dict[str, Dict[str, List[str]]],
    llm_base_url: str,
    llm_model: str,
    data_dir: str,
    live_state: dict,
    facts_per_turn: int = 2,
    poison_memories: List[str] = None,
) -> "GroupResults":
    """Conversation-based learning: LLM B roleplays characters, LLM A responds with memory.

    For each character in each conversation, LLM B shares facts in batches.
    LLM A responds using retrieved memories. Sidecar scores and stores.
    Continues until all facts are covered.
    """
    import httpx

    results = GroupResults(name=config.name)
    strategy = config.strategy_factory(data_dir)
    await strategy.initialize()
    context = config.context_factory()

    # Pre-seed poison memories if provided
    if poison_memories:
        print(f"    Pre-seeding {len(poison_memories)} poison memories...", flush=True)
        for pm in poison_memories:
            await strategy.store(pm)
        print(f"    Poison seeded.", flush=True)

    llm_client = httpx.AsyncClient(
        base_url=llm_base_url,
        headers={"Authorization": "Bearer ollama"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )

    total_facts = sum(
        len(facts)
        for conv_data in character_sheets.values()
        for facts in conv_data.values()
    )
    covered_total = 0
    turn = 0

    # Conversation checkpoint for mid-step resume
    conv_ckpt_file = Path("results") / f"conv_checkpoint_{config.name.replace('/', '_')}.json"
    conv_ckpt = None
    if conv_ckpt_file.exists():
        try:
            conv_ckpt = json.loads(conv_ckpt_file.read_text(encoding="utf-8"))
            print(f"    {config.name}: Resuming from checkpoint — "
                  f"conv={conv_ckpt['conv_key']}, char={conv_ckpt['char_name']}, "
                  f"turn={conv_ckpt['turn']}, covered={conv_ckpt['covered_total']}",
                  flush=True)
        except Exception as e:
            print(f"    WARNING: Bad checkpoint, starting fresh: {e}", flush=True)
            conv_ckpt = None

    def _save_conv_checkpoint(conv_key, char_name, facts_offset, turn_num, covered):
        """Save conversation progress for resume after crash."""
        tmp = Path(str(conv_ckpt_file) + ".tmp")
        tmp.write_text(json.dumps({
            "conv_key": conv_key,
            "char_name": char_name,
            "facts_offset": facts_offset,
            "turn": turn_num,
            "covered_total": covered,
        }), encoding="utf-8")
        tmp.replace(conv_ckpt_file)

    # Process each conversation's characters
    for conv_key, characters in sorted(character_sheets.items()):
        for char_name, facts in characters.items():
            # Skip characters completed before checkpoint
            if conv_ckpt:
                if conv_key < conv_ckpt["conv_key"]:
                    n = len(facts)
                    skip_turns = (n + facts_per_turn - 1) // facts_per_turn
                    covered_total += n
                    turn += skip_turns
                    print(f"    {config.name}: SKIP {conv_key}/{char_name} (checkpoint)", flush=True)
                    continue
                if conv_key == conv_ckpt["conv_key"] and char_name != conv_ckpt["char_name"]:
                    # Same conv but different char — check ordering
                    char_keys = list(characters.keys())
                    ckpt_idx = char_keys.index(conv_ckpt["char_name"]) if conv_ckpt["char_name"] in char_keys else -1
                    curr_idx = char_keys.index(char_name)
                    if curr_idx < ckpt_idx:
                        n = len(facts)
                        skip_turns = (n + facts_per_turn - 1) // facts_per_turn
                        covered_total += n
                        turn += skip_turns
                        print(f"    {config.name}: SKIP {conv_key}/{char_name} (checkpoint)", flush=True)
                        continue

            # Determine starting offset for this character
            facts_offset = 0
            if conv_ckpt and conv_key == conv_ckpt["conv_key"] and char_name == conv_ckpt["char_name"]:
                facts_offset = conv_ckpt["facts_offset"]
                covered_total = conv_ckpt["covered_total"]
                turn = conv_ckpt["turn"]
                conv_ckpt = None  # Consumed — don't apply to later characters
                print(f"    {config.name}: Resuming {char_name} from fact {facts_offset}/{len(facts)}...", flush=True)
            else:
                print(f"    {config.name}: Starting {char_name} ({len(facts)} facts)...", flush=True)

            uncovered = list(facts[facts_offset:])
            recent_exchanges = []  # Sliding window for LLM B

            while uncovered:
                batch = uncovered[:facts_per_turn]

                # 1. LLM B speaks — shares facts naturally
                b_message = await _llm_b_speak(
                    llm_client, llm_model, char_name, batch, facts, recent_exchanges
                )
                if not b_message:
                    uncovered = uncovered[facts_per_turn:]
                    covered_total += len(batch)
                    turn += 1
                    continue

                # 2. LLM A retrieves memories and responds
                search_query = b_message
                if config.use_rewrite:
                    context_msgs = context.build_messages()
                    search_query = await rewrite_query(llm_client, llm_model, b_message, context_msgs)

                # Two-lane retrieval: context + facts
                from strategies.base import RetrievalResult
                try:
                    context_retrieval = await asyncio.wait_for(
                        strategy.retrieve(search_query, top_k=4, type_exclude="fact"), timeout=30
                    )
                except Exception as e:
                    print(f"    WARN: context retrieve failed: {type(e).__name__}: {e}", flush=True)
                    context_retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=search_query, retrieval_ms=0)
                try:
                    fact_retrieval = await asyncio.wait_for(
                        strategy.retrieve(search_query, top_k=4, type_filter="fact"), timeout=30
                    )
                except Exception as e:
                    print(f"    WARN: fact retrieve failed: {type(e).__name__}: {e}", flush=True)
                    fact_retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=search_query, retrieval_ms=0)

                all_memories = context_retrieval.memories + fact_retrieval.memories
                total_ms = context_retrieval.retrieval_ms + fact_retrieval.retrieval_ms
                retrieval = RetrievalResult(memories=all_memories, formatted_injection="", query_used=search_query, retrieval_ms=total_ms)
                results.retrieval_ms_total += total_ms

                messages = context.build_messages()
                parts = []
                if context_retrieval.formatted_injection:
                    parts.append(context_retrieval.formatted_injection)
                if fact_retrieval.formatted_injection:
                    # Replace header for fact lane to distinguish from summaries
                    fact_text = fact_retrieval.formatted_injection.replace(
                        "═══ KNOWN CONTEXT ═══", "═══ KNOWN FACTS ═══"
                    )
                    parts.append(fact_text)
                if parts:
                    user_content = "\n\n".join(parts) + "\n\n" + b_message
                else:
                    user_content = b_message
                messages.append({"role": "user", "content": user_content})

                # Try up to 3 times for a non-blank response
                a_response = ""
                for attempt in range(3):
                    try:
                        resp = await asyncio.wait_for(
                            llm_client.post("/chat/completions", json={
                                "model": llm_model,
                                "messages": [{"role": "system", "content": "You are a helpful assistant chatting with a friend. Respond in 2-4 sentences. Reference specific details you remember about them — dates, events, plans."}] + messages,
                                "max_tokens": 2000,
                                "temperature": 0,
                            }),
                            timeout=90,
                        )
                        data = resp.json()
                        msg = data["choices"][0]["message"]
                        a_response = msg.get("content", "")
                        if not a_response or not a_response.strip():
                            a_response = msg.get("reasoning_content", "") or ""
                        if a_response and a_response.strip():
                            break
                    except Exception as e:
                        print(f"    LLM A attempt {attempt+1} failed turn {turn}: {type(e).__name__}", flush=True)

                # Skip entire turn if LLM A still blank — don't store junk
                if not a_response or not a_response.strip():
                    print(f"    SKIP turn {turn}: LLM A blank after 3 attempts", flush=True)
                    uncovered = uncovered[facts_per_turn:]
                    covered_total += len(batch)
                    turn += 1
                    continue

                # 3. LLM B reacts — confirms, corrects, or continues (THE FEEDBACK)
                b_reaction = await _llm_b_react(
                    llm_client, llm_model, char_name, facts, a_response, recent_exchanges
                )

                # 4. Store exchanges in context
                context.add_exchange(b_message, a_response)
                if b_reaction:
                    context.add_exchange(b_reaction, "")  # Reaction as followup
                recent_exchanges.append({"b": b_message, "a": a_response, "reaction": b_reaction})
                if len(recent_exchanges) > 4:
                    recent_exchanges = recent_exchanges[-4:]

                # 5. Sidecar scoring — uses LLM B's reaction as the followup signal
                sidecar_result = await sidecar_score(
                    llm_client, llm_model, b_message, a_response,
                    retrieval.memories, b_reaction,
                )
                exchange_summary = sidecar_result.get("exchange_summary", "")
                if not exchange_summary:
                    exchange_summary = await sidecar_summarize(
                        llm_client, llm_model, b_message, a_response
                    )
                per_memory_scores = sidecar_result.get("memory_scores", {})
                exchange_outcome = sidecar_result.get("exchange_outcome", "unknown")

                # 6. Record per-memory outcomes + store summary
                try:
                    for m in retrieval.memories:
                        if m.id:
                            mem_outcome = per_memory_scores.get(m.id, exchange_outcome)
                            if mem_outcome not in ("worked", "failed", "partial", "unknown"):
                                mem_outcome = exchange_outcome
                            await asyncio.wait_for(strategy.record_outcome(
                                memory_ids=[m.id],
                                outcome=mem_outcome,
                                exchange_summary="",
                            ), timeout=60)
                    # Store summary (continuity memory)
                    await asyncio.wait_for(strategy.record_outcome(
                        memory_ids=[],
                        outcome=exchange_outcome,
                        exchange_summary=exchange_summary,
                    ), timeout=60)
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"    WARNING: record_outcome: {e}", flush=True)

                # 6b. Extract and store atomic facts (recall memories)
                if config.extract_facts:
                    try:
                        extracted_facts = await sidecar_extract_facts(
                            llm_client, llm_model, b_message, a_response
                        )
                        for fact in extracted_facts:
                            # Facts always create new — no evolving updates
                            # (merging loses specific details that retrieval needs)
                            fact_id = await asyncio.wait_for(strategy.store(fact, metadata={
                                "type": "fact",
                            }), timeout=60)
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"    WARNING: fact extraction: {e}", flush=True)

                # 7. Mark facts as covered, advance
                uncovered = uncovered[facts_per_turn:]
                covered_total += len(batch)
                results.turns += 1
                turn += 1

                # 8. Log turn
                results.turn_log.append({
                    "turn": turn,
                    "character": char_name,
                    "facts_batch": len(batch),
                    "facts_remaining": len(uncovered),
                    "memories_surfaced": len(retrieval.memories),
                    "retrieval_ms": round(retrieval.retrieval_ms, 1),
                    "b_preview": b_message[:80],
                    "a_preview": a_response[:80],
                    "reaction_preview": b_reaction[:60] if b_reaction else "",
                    "exchange_outcome": exchange_outcome,
                })

                # 9. Update live state for dashboard
                live_state["current_group"] = config.name
                live_state["current_turn"] = turn
                live_state["total_turns"] = total_facts // facts_per_turn + 1

                if "feed" not in live_state:
                    live_state["feed"] = []
                live_state["feed"].append({
                    "group": config.name,
                    "turn": turn,
                    "character": char_name,
                    "query": b_message,
                    "answer": a_response,
                    "followup": b_reaction or "",
                    "judgment": exchange_outcome,
                    "memories": len(retrieval.memories),
                    "retrieval_ms": round(retrieval.retrieval_ms, 1),
                })
                live_state["feed"] = live_state["feed"][-6:]

                existing_group = live_state.get("groups", {}).get(config.name, {})
                existing_group.update({
                    "turns": results.turns,
                    "facts_covered": covered_total,
                    "facts_total": total_facts,
                    "coverage": round(covered_total / total_facts, 4) if total_facts else 0,
                    "avg_retrieval_ms": round(results.avg_retrieval_ms(), 1),
                    "current_character": char_name,
                })
                live_state["groups"][config.name] = existing_group
                live_state["updated_at"] = time.time()
                _write_live_state(live_state)

                # Save conversation checkpoint for crash recovery
                current_offset = len(facts) - len(uncovered)
                _save_conv_checkpoint(conv_key, char_name, current_offset, turn, covered_total)

                if turn % 10 == 0:
                    print(f"    {config.name}: turn {turn} | "
                          f"{covered_total}/{total_facts} facts | "
                          f"{char_name} ({len(uncovered)} remaining)",
                          flush=True)

            print(f"    {config.name}: {char_name} done ({len(facts)} facts covered)", flush=True)

    # Clean up checkpoint — conversation completed successfully
    if conv_ckpt_file.exists():
        conv_ckpt_file.unlink()

    await llm_client.aclose()

    strat_stats = await strategy.get_stats() if hasattr(strategy, 'get_stats') else {}
    results.memories_stored = strat_stats.get("memories", 0)

    print(f"    {config.name}: Conversation complete — {results.turns} turns, "
          f"{covered_total} facts, {results.memories_stored} memories stored", flush=True)

    return results, strategy


async def run_exam(
    strategy,
    exam_queries: List[Dict],
    llm_base_url: str,
    llm_model: str,
    group_name: str,
    grader=None,
    live_state: dict = None,
    learning: bool = False,
    llm_api_key: str = None,
) -> Dict:
    """Run final exam on learned state.

    If grader is provided, uses LLM grading (for LoCoMo natural language answers).
    Otherwise, uses deterministic string match grading.
    If learning=True, outcomes are recorded back to the strategy (Wilson scores update).
    """
    import httpx

    auth_token = llm_api_key or "ollama"
    llm_client = httpx.AsyncClient(
        base_url=llm_base_url,
        headers={"Authorization": f"Bearer {auth_token}"},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )

    use_llm_grading = grader is not None

    # Exam checkpoint — resume from where we left off if killed
    checkpoint_file = Path("results") / f"exam_checkpoint_{group_name.replace('/', '_')}.json"
    exam_results = {
        "correct": 0, "partial": 0, "wrong": 0, "total": 0, "log": [],
        "by_category": {},
    }
    start_from = 0
    if checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            exam_results = checkpoint
            start_from = checkpoint["total"]
            print(f"    Resuming exam from question {start_from}", flush=True)
        except Exception:
            pass

    for i, q in enumerate(exam_queries):
        if i < start_from:
            continue
        question = q["query"]
        ground_truth = q.get("ground_truth", "")
        category = q.get("category_name", "unknown")

        # Two-lane retrieval: 4 context memories + 4 fact memories
        from strategies.base import RetrievalResult
        try:
            context_retrieval = await asyncio.wait_for(
                strategy.retrieve(question, top_k=4, type_exclude="fact"), timeout=120
            )
        except (asyncio.TimeoutError, Exception):
            context_retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=question, retrieval_ms=0)

        try:
            fact_retrieval = await asyncio.wait_for(
                strategy.retrieve(question, top_k=4, type_filter="fact"), timeout=120
            )
        except (asyncio.TimeoutError, Exception):
            fact_retrieval = RetrievalResult(memories=[], formatted_injection="", query_used=question, retrieval_ms=0)

        # Combine both lanes into one retrieval result for scoring
        all_memories = context_retrieval.memories + fact_retrieval.memories
        total_ms = context_retrieval.retrieval_ms + fact_retrieval.retrieval_ms
        retrieval = RetrievalResult(
            memories=all_memories,
            formatted_injection="",
            query_used=question,
            retrieval_ms=total_ms,
        )

        # Build prompt with both lanes
        messages = []
        parts = []
        if context_retrieval.formatted_injection:
            parts.append(context_retrieval.formatted_injection)
        if fact_retrieval.formatted_injection:
            fact_text = fact_retrieval.formatted_injection.replace(
                "═══ KNOWN CONTEXT ═══", "═══ KNOWN FACTS ═══"
            )
            parts.append(fact_text)
        if parts:
            user_content = "\n\n".join(parts) + "\n\n" + question
        else:
            user_content = question
        messages.append({"role": "user", "content": user_content})

        # LLM answers (retry up to 3 times on empty response)
        answer = ""
        for _attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    llm_client.post("/chat/completions", json={
                        "model": llm_model,
                        "messages": [{"role": "system", "content": "Answer concisely using any provided context."}] + messages,
                        "max_tokens": 5000,
                    "temperature": 0,
                    }),
                    timeout=180,
                )
                data = resp.json()
                msg = data["choices"][0]["message"]
                answer = msg.get("content", "")
                if not answer or not answer.strip():
                    answer = msg.get("reasoning_content", "") or msg.get("reasoning", "")
                if answer and answer.strip():
                    break
            except Exception:
                pass

        # Grade the answer
        if not answer or not answer.strip():
            judgment = "wrong"
        elif use_llm_grading:
            # LLM grading for LoCoMo (natural language answers)
            judgment, _ = await grader.grade(
                question=question,
                answer=answer,
                source_text=f"Ground truth answer: {ground_truth}",
            )
        else:
            # Deterministic string match
            if ground_truth and ground_truth.lower() in answer.lower():
                judgment = "correct"
            else:
                gt_parts = ground_truth.split()
                key_part = gt_parts[0] if gt_parts else ground_truth
                if key_part and len(key_part) >= 2 and key_part.lower() in answer.lower():
                    judgment = "correct"
                else:
                    judgment = "wrong"

        if judgment == "correct":
            exam_results["correct"] += 1
        elif judgment == "partial":
            exam_results["partial"] += 1
        else:
            exam_results["wrong"] += 1
        exam_results["total"] += 1

        # Track by category
        if category not in exam_results["by_category"]:
            exam_results["by_category"][category] = {"correct": 0, "partial": 0, "wrong": 0, "total": 0}
        exam_results["by_category"][category]["total"] += 1
        if judgment == "correct":
            exam_results["by_category"][category]["correct"] += 1
        elif judgment == "partial":
            exam_results["by_category"][category]["partial"] += 1
        else:
            exam_results["by_category"][category]["wrong"] += 1

        # If learning is on, use sidecar for per-memory scoring + summary (matches production)
        if learning:
            # Per-memory scoring via sidecar (same 7-rule prompt as production)
            if retrieval.memories:
                try:
                    sidecar_result = await sidecar_score(
                        llm_client, llm_model, question, answer,
                        retrieval.memories, followup=f"Ground truth: {ground_truth}",
                    )
                    per_memory = sidecar_result.get("memory_scores", {})
                    mapped_outcome = "worked" if judgment == "correct" else ("partial" if judgment == "partial" else "failed")
                    for m in retrieval.memories:
                        if m.id:
                            mem_outcome = per_memory.get(m.id, mapped_outcome)
                            if mem_outcome not in ("worked", "failed", "partial", "unknown"):
                                mem_outcome = mapped_outcome
                            try:
                                await strategy.record_outcome([m.id], mem_outcome)
                            except Exception:
                                pass
                except Exception:
                    pass
            # Sidecar summary (not dumb concatenation)
            try:
                exchange_summary = await sidecar_summarize(llm_client, llm_model, question, answer)
                if ground_truth:
                    exchange_summary += f" Ground truth: {ground_truth[:200]}"
                await strategy.store(exchange_summary)
            except Exception:
                pass

        exam_results["log"].append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "judgment": judgment,
            "category": category,
            "memories": len(retrieval.memories),
        })

        # Update live feed so dashboard shows exam Q&A
        if live_state:
            if "feed" not in live_state:
                live_state["feed"] = []
            live_state["feed"].append({
                "group": f"EXAM {group_name}",
                "turn": i + 1,
                "query": question,
                "answer": answer,
                "judgment": judgment,
                "followup": f"Ground truth: {ground_truth}",
                "memories": len(retrieval.memories),
                "retrieval_ms": round(retrieval.retrieval_ms, 1),
            })
            live_state["feed"] = live_state["feed"][-6:]

        # Checkpoint — atomic write (tmp + rename) to survive crashes
        # Every question if learning ON (memories being stored), every 50 if OFF
        should_checkpoint = True  # Checkpoint every question for crash safety
        if should_checkpoint:
            try:
                tmp_ckpt = Path(str(checkpoint_file) + ".tmp")
                with open(tmp_ckpt, "w", encoding="utf-8") as cf:
                    json.dump(exam_results, cf)
                tmp_ckpt.replace(checkpoint_file)  # Atomic on most filesystems
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            acc = exam_results["correct"] / exam_results["total"] if exam_results["total"] else 0
            print(f"    EXAM {group_name}: {i+1}/{len(exam_queries)} -- "
                  f"{exam_results['correct']}c/{exam_results['wrong']}w  {acc:.1%}",
                  flush=True)

            # Update live state for dashboard (every 10 questions — stats + feed)
            if live_state and group_name in live_state.get("groups", {}):
                live_state["groups"][group_name]["exam"] = {
                    "correct": exam_results["correct"],
                    "partial": exam_results.get("partial", 0),
                    "wrong": exam_results["wrong"],
                    "total": exam_results["total"],
                    "accuracy": round(acc, 4),
                    "by_category": exam_results.get("by_category", {}),
                    "progress": f"{i+1}/{len(exam_queries)}",
                }

        # Write live state every turn (feed updates)
        if live_state:
            live_state["current_turn"] = i + 1
            live_state["total_turns"] = len(exam_queries)
            live_state["updated_at"] = time.time()
            _write_live_state(live_state)

    await llm_client.aclose()
    # Clean up checkpoint — exam completed successfully
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    return exam_results


async def run_benchmark(
    num_turns: int = 200,
    data_path: str = "data/fictional_benchmark_data.json",
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "gpt-oss:20b",
    grader_model: str = None,
    resume: bool = False,
    only_groups: str = None,
    run_poison: bool = False,
):
    """Run the full benchmark."""

    data_file = Path(data_path)
    if not data_file.exists():
        print(f"ERROR: {data_file} not found.")
        return None

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build query pool
    rng = random.Random(42)
    queries = []
    for q in data["exam_questions"]:
        # Priority: question's own source_text > matching memory > ground_truth
        source = q.get("source_text", "")
        if isinstance(source, list):
            source = " ".join(str(s) for s in source)
        source = source.strip()
        if not source:
            chunk_id = q.get("chunk_id", "")
            source = q.get("ground_truth", "")
            for m in data["memories"]:
                if m.get("chunk_id") == chunk_id and m.get("source_text", "").strip():
                    source = m["source_text"]
                    break
        queries.append({"query": q["question"], "source": source})
    for m in data["memories"]:
        for rq in m.get("raw_queries", []):
            queries.append({"query": rq, "source": m.get("source_text", m.get("key_fact", ""))})
    rng.shuffle(queries)

    # Load poison memories if present in dataset
    poison_memories = []
    for pm in data.get("poison_technical", []):
        content = pm.get("content", "")
        if content:
            poison_memories.append(content)
    for pm in data.get("poison_personal", []):
        content = pm.get("content", "")
        if content:
            poison_memories.append(content)
    if poison_memories:
        print(f"Poison memories to pre-seed: {len(poison_memories)}")

    # Initialize grader
    grader = LLMGrader(base_url=llm_base_url, model=grader_model or llm_model)
    await grader.initialize()

    # Define groups — use persistent directory so state survives restarts
    tmp = Path("runs") / "current"
    tmp.mkdir(parents=True, exist_ok=True)

    groups = [
        # === 4-way comparison: Wilson vs CE vs Wilson+CE blend vs Wilson+CE bouncer ===
        GroupConfig(name="01.Wilson",
            strategy_factory=lambda d: WilsonScoredStrategy(persist_dir=str(d)),
            context_factory=lambda: WindowContext(window_size=4)),
        GroupConfig(name="02.Reranker",
            strategy_factory=lambda d: SemanticRerankerStrategy(persist_dir=str(d)),
            context_factory=lambda: WindowContext(window_size=4)),
        GroupConfig(name="03.Wilson+CE",
            strategy_factory=lambda d: WilsonRerankerStrategy(persist_dir=str(d)),
            context_factory=lambda: WindowContext(window_size=4)),
        # STOPPED: KG+CE edge scores frozen at 0.5 — graph not learning.
        # Data archived in runs/archive_kg_ce/. See paper for observation.
        # GroupConfig(name="04.KG+CE",
        #     strategy_factory=lambda d: KGTraversalStrategy(persist_dir=str(d)),
        #     context_factory=lambda: WindowContext(window_size=4)),
        GroupConfig(name="04.EntityRouted",
            strategy_factory=lambda d: EntityRoutedStrategy(persist_dir=str(d)),
            context_factory=lambda: WindowContext(window_size=4)),
    ]

    # Filter groups if --groups specified
    if only_groups:
        requested = [g.strip() for g in only_groups.split(",")]
        groups = [g for g in groups if g.name in requested]
        if not groups:
            print(f"ERROR: No matching groups found for: {only_groups}")
            print(f"Available: 01.Wilson, 02.Reranker, 03.Wilson+CE, 04.EntityRouted")
            return None
        print(f"Running groups: {[g.name for g in groups]}")

    # Live state for dashboard — load existing if resuming
    live_state_path = Path("results/live_state.json")
    if resume and live_state_path.exists():
        with open(live_state_path, "r", encoding="utf-8") as f:
            live_state = json.load(f)
        # Fix up accuracy for groups with exam data (e.g. after live_state reconstruction)
        for g in live_state.get("groups", {}).values():
            exam = g.get("exam")
            if exam and exam.get("total", 0) > 0 and not exam.get("accuracy"):
                exam["accuracy"] = round(exam["correct"] / exam["total"], 4)
        print(f"RESUMING from previous run", flush=True)
    else:
        live_state = {
            "benchmark": f"{num_turns} turns x {len(groups)} groups",
            "model": llm_model,
            "started_at": time.time(),
            "current_group": "",
            "current_turn": 0,
            "total_turns": num_turns,
            "total_groups": len(groups),
            "completed_groups": 0,
            "groups": {},
            "updated_at": time.time(),
        }
        _write_live_state(live_state)

    TURNS_PER_PASS = 404
    total_passes = num_turns // TURNS_PER_PASS

    print(f"{'='*70}")
    print(f"MEMORY RETRIEVAL BENCHMARK ({total_passes} passes, {num_turns} turns, {len(groups)} groups)")
    print(f"{'='*70}")
    print(f"LLM: {llm_model}")
    print(f"Data: {data_file.name} ({len(queries)} queries)")
    print(f"Live dashboard: results/live_state.json")
    if run_poison:
        print(f"MODE: POISON TEST (no learning, exams + poison + healing)")

    all_results = {}

    # === PASS-BY-PASS LOOP: each pass does 404 turns per strategy + LoCoMo ===
    for pass_num in range(1, total_passes + 1):
        pass_turns = pass_num * TURNS_PER_PASS
        print(f"\n{'='*70}")
        print(f"PASS {pass_num}/{total_passes} (target: {pass_turns} turns)")
        print(f"{'='*70}", flush=True)

        for gc in groups:
            group_state = live_state.get("groups", {}).get(gc.name, {})
            current_turns = group_state.get("turns", 0) if resume else 0

            # Skip if group already completed this pass (has exam at this turn count)
            exam_at = group_state.get("exam", {}).get("at_turns", 0) if group_state.get("exam") else 0
            if current_turns >= pass_turns and exam_at >= pass_turns:
                print(f"\n--- {gc.name} --- already at {current_turns}t with exam, skipping", flush=True)
                continue

            print(f"\n--- {gc.name} (pass {pass_num}) ---", flush=True)
            group_dir = tmp / gc.name.replace(" ", "_").replace("+", "_")
            group_dir.mkdir(parents=True, exist_ok=True)

            # Learning phase — skip if already has enough turns
            if current_turns >= pass_turns:
                print(f"    Learning already at {current_turns}t, skipping to exam", flush=True)
                # Still need to initialize strategy for the exam
                strategy = gc.strategy_factory(str(group_dir))
                await strategy.initialize()
            else:
                resume_from = current_turns if resume else 0
                t0 = time.time()
                results, strategy = await run_group(
                    config=gc,
                    queries=queries,
                    grader=grader,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    num_turns=pass_turns,
                    data_dir=str(group_dir),
                    live_state=live_state,
                    resume_from=resume_from,
                    poison_memories=poison_memories if poison_memories else None,
                )
                elapsed = time.time() - t0
                print(f"    Learning done in {elapsed:.0f}s", flush=True)
                all_results[gc.name] = results

            live_state["groups"][gc.name]["done"] = True

            # === EXAM PHASE (per group, after learning) ===
            exam_queries = []
            exam_grader = None

            # Check for LoCoMo exam in dataset (LLM graded)
            if data.get("locomo_exam"):
                for eq in data["locomo_exam"]:
                    exam_queries.append({
                        "query": eq["question"],
                        "ground_truth": eq.get("ground_truth", ""),
                        "category_name": eq.get("category_name", "unknown"),
                    })
                exam_grader = grader  # Use LLM grading for LoCoMo
                grading_mode = "LLM-graded"
            else:
                # Fall back to fresh_exam.json (deterministic string match)
                exam_file = Path("data/fresh_exam.json")
                if exam_file.exists():
                    exam_data = json.loads(exam_file.read_text(encoding="utf-8"))
                    for eq in exam_data:
                        exam_queries.append({
                            "query": eq["question"],
                            "ground_truth": eq.get("ground_truth", ""),
                        })
                    grading_mode = "deterministic"

            if exam_queries:
                print(f"    --- EXAM ({len(exam_queries)} questions, {grading_mode}) ---", flush=True)
                exam_results = await run_exam(
                    strategy=strategy,
                    exam_queries=exam_queries,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    group_name=gc.name,
                    grader=exam_grader,
                    live_state=live_state,
                )
                exam_acc = exam_results["correct"] / exam_results["total"] if exam_results["total"] else 0
                print(f"    EXAM RESULT: {exam_results['correct']}c/{exam_results['wrong']}w "
                      f"({exam_acc:.1%}) on {exam_results['total']} questions", flush=True)

                # Print per-category breakdown if available
                if exam_results.get("by_category"):
                    for cat, stats in sorted(exam_results["by_category"].items()):
                        cat_acc = stats["correct"] / stats["total"] if stats["total"] else 0
                        print(f"      {cat}: {stats['correct']}c/{stats['wrong']}w ({cat_acc:.1%})", flush=True)

                exam_entry = {
                    "correct": exam_results["correct"],
                    "partial": exam_results.get("partial", 0),
                    "wrong": exam_results["wrong"],
                    "total": exam_results["total"],
                    "accuracy": round(exam_acc, 4),
                    "by_category": exam_results.get("by_category", {}),
                    "at_turns": live_state["groups"][gc.name].get("turns", 0),
                }
                live_state["groups"][gc.name]["exam"] = exam_entry
                # Accumulate exam history across passes
                if "exam_history" not in live_state["groups"][gc.name]:
                    live_state["groups"][gc.name]["exam_history"] = []
                live_state["groups"][gc.name]["exam_history"].append(exam_entry)

                # Save full exam transcript to permanent file
                at_t = live_state["groups"][gc.name].get("turns", 0)
                exam_file_out = Path("results") / f"exam_{gc.name}_{at_t}t.json"
                try:
                    with open(exam_file_out, "w", encoding="utf-8") as ef:
                        json.dump({
                            "group": gc.name,
                            "at_turns": at_t,
                            "pass": pass_num,
                            "summary": exam_entry,
                            "transcript": exam_results.get("log", []),
                        }, ef, indent=2, ensure_ascii=False)
                    print(f"    Exam saved to {exam_file_out}", flush=True)
                except Exception as e:
                    print(f"    WARNING: Failed to save exam file: {e}", flush=True)

            # === HARD EXAM (only on poison/final pass) ===
            hard_exam_file = Path("data/hard_exam.json")
            if run_poison and hard_exam_file.exists():
                hard_qs = json.loads(hard_exam_file.read_text(encoding="utf-8"))
                hard_exam_queries = [{"query": q["question"],
                                      "ground_truth": q.get("ground_truth", q.get("answer", "")),
                                      "category_name": q.get("category", q.get("category_name", "hard"))}
                                     for q in hard_qs]
                print(f"    --- HARD EXAM ({len(hard_exam_queries)} questions) ---", flush=True)
                hard_results = await run_exam(
                    strategy=strategy, exam_queries=hard_exam_queries,
                    llm_base_url=llm_base_url, llm_model=llm_model,
                    group_name=f"{gc.name}/hard", grader=grader, live_state=live_state,
                )
                hard_acc = hard_results["correct"] / hard_results["total"] if hard_results["total"] else 0
                print(f"    HARD EXAM: {hard_results['correct']}c/{hard_results['wrong']}w ({hard_acc:.1%})", flush=True)
                live_state["groups"][gc.name]["hard_exam"] = {
                    "correct": hard_results["correct"],
                    "partial": hard_results.get("partial", 0),
                    "wrong": hard_results["wrong"],
                    "total": hard_results["total"],
                    "accuracy": round(hard_acc, 4),
                    "by_category": hard_results.get("by_category", {}),
                }
                _write_live_state(live_state)

                _save_exam_transcript(gc.name, "hard_baseline", live_state["groups"][gc.name].get("turns", 0),
                                      live_state["groups"][gc.name]["hard_exam"], hard_results)

            # === POISON PHASE (only when --poison flag is set, typically final pass) ===
            poison_file = Path("data/poison_memories.json")
            if run_poison and poison_file.exists():
                poison_data = json.loads(poison_file.read_text(encoding="utf-8"))
                poison_mems = poison_data.get("poison_memories", [])
                if poison_mems:
                    print(f"    --- POISON PHASE ({len(poison_mems)} poison memories) ---", flush=True)

                    # Inject poison with fake metadata
                    for pm in poison_mems:
                        doc_id = await strategy.store(pm["content"])
                        # Override metadata with fake Wilson scores
                        fake = pm.get("fake_meta", {})
                        if fake and hasattr(strategy, '_collection') and strategy._collection:
                            try:
                                existing = strategy._collection.get(ids=[doc_id], include=["metadatas"])
                                if existing and existing["metadatas"]:
                                    meta = existing["metadatas"][0]
                                    meta.update(fake)
                                    strategy._collection.update(ids=[doc_id], metadatas=[meta])
                            except Exception:
                                pass
                    print(f"    Poison injected.", flush=True)

                    # Poisoned LoCoMo exam (learning ON — exam IS the poisoning + healing)
                    if exam_queries:
                        print(f"    --- POISONED EXAM ({len(exam_queries)} questions, learning ON) ---", flush=True)
                        poison_results = await run_exam(
                            strategy=strategy, exam_queries=exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/poisoned", grader=exam_grader,
                            live_state=live_state, learning=True,
                        )
                        poison_acc = poison_results["correct"] / poison_results["total"] if poison_results["total"] else 0
                        print(f"    POISONED EXAM: {poison_results['correct']}c/{poison_results['wrong']}w ({poison_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["poisoned_exam"] = {
                            "correct": poison_results["correct"],
                            "partial": poison_results.get("partial", 0),
                            "wrong": poison_results["wrong"],
                            "total": poison_results["total"],
                            "accuracy": round(poison_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "poisoned_locomo", at_t, live_state["groups"][gc.name]["poisoned_exam"], poison_results)

                    # Poisoned hard exam (learning ON)
                    if hard_exam_file.exists():
                        print(f"    --- POISONED HARD EXAM ({len(hard_exam_queries)} questions, learning ON) ---", flush=True)
                        poison_hard = await run_exam(
                            strategy=strategy, exam_queries=hard_exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/poisoned-hard", grader=grader,
                            live_state=live_state, learning=True,
                        )
                        poison_hard_acc = poison_hard["correct"] / poison_hard["total"] if poison_hard["total"] else 0
                        print(f"    POISONED HARD: {poison_hard['correct']}c/{poison_hard['wrong']}w ({poison_hard_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["poisoned_hard_exam"] = {
                            "correct": poison_hard["correct"],
                            "partial": poison_hard.get("partial", 0),
                            "wrong": poison_hard["wrong"],
                            "total": poison_hard["total"],
                            "accuracy": round(poison_hard_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "poisoned_hard", at_t, live_state["groups"][gc.name]["poisoned_hard_exam"], poison_hard)

                    # === POST-POISON EXAMS (learning OFF — raw damage measurement) ===
                    if exam_queries:
                        print(f"    --- POST-POISON EXAM ({len(exam_queries)} questions, learning OFF) ---", flush=True)
                        post_poison_results = await run_exam(
                            strategy=strategy, exam_queries=exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/post-poison", grader=exam_grader,
                            live_state=live_state,
                        )
                        post_poison_acc = post_poison_results["correct"] / post_poison_results["total"] if post_poison_results["total"] else 0
                        print(f"    POST-POISON EXAM: {post_poison_results['correct']}c/{post_poison_results['wrong']}w ({post_poison_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["post_poison_exam"] = {
                            "correct": post_poison_results["correct"],
                            "partial": post_poison_results.get("partial", 0),
                            "wrong": post_poison_results["wrong"],
                            "total": post_poison_results["total"],
                            "accuracy": round(post_poison_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "post_poison_locomo", at_t, live_state["groups"][gc.name]["post_poison_exam"], post_poison_results)

                    if hard_exam_file.exists():
                        print(f"    --- POST-POISON HARD EXAM ({len(hard_exam_queries)} questions, learning OFF) ---", flush=True)
                        post_poison_hard = await run_exam(
                            strategy=strategy, exam_queries=hard_exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/post-poison-hard", grader=grader,
                            live_state=live_state,
                        )
                        post_poison_hard_acc = post_poison_hard["correct"] / post_poison_hard["total"] if post_poison_hard["total"] else 0
                        print(f"    POST-POISON HARD: {post_poison_hard['correct']}c/{post_poison_hard['wrong']}w ({post_poison_hard_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["post_poison_hard_exam"] = {
                            "correct": post_poison_hard["correct"],
                            "partial": post_poison_hard.get("partial", 0),
                            "wrong": post_poison_hard["wrong"],
                            "total": post_poison_hard["total"],
                            "accuracy": round(post_poison_hard_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "post_poison_hard", at_t, live_state["groups"][gc.name]["post_poison_hard_exam"], post_poison_hard)

                    # === HEALING LOOP (404-turn learning — Wilson demotes poison) ===
                    print(f"    --- HEALING LOOP (404 turn learning pass) ---", flush=True)
                    heal_t0 = time.time()
                    _, strategy = await run_group(
                        config=gc,
                        queries=queries,
                        grader=grader,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        num_turns=404,
                        data_dir=str(group_dir),
                        live_state=live_state,
                        resume_from=0,
                    )
                    heal_elapsed = time.time() - heal_t0
                    print(f"    Healing complete ({heal_elapsed:.0f}s)", flush=True)

                    # Healed LoCoMo exam (learning OFF — pure recovery measurement)
                    if exam_queries:
                        print(f"    --- HEALED EXAM ({len(exam_queries)} questions) ---", flush=True)
                        healed_results = await run_exam(
                            strategy=strategy, exam_queries=exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/healed", grader=exam_grader,
                            live_state=live_state,
                        )
                        healed_acc = healed_results["correct"] / healed_results["total"] if healed_results["total"] else 0
                        print(f"    HEALED EXAM: {healed_results['correct']}c/{healed_results['wrong']}w ({healed_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["healed_exam"] = {
                            "correct": healed_results["correct"],
                            "partial": healed_results.get("partial", 0),
                            "wrong": healed_results["wrong"],
                            "total": healed_results["total"],
                            "accuracy": round(healed_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "healed_locomo", at_t, live_state["groups"][gc.name]["healed_exam"], healed_results)

                    # Healed hard exam (learning OFF)
                    if hard_exam_file.exists():
                        print(f"    --- HEALED HARD EXAM ({len(hard_exam_queries)} questions) ---", flush=True)
                        healed_hard = await run_exam(
                            strategy=strategy, exam_queries=hard_exam_queries,
                            llm_base_url=llm_base_url, llm_model=llm_model,
                            group_name=f"{gc.name}/healed-hard", grader=grader,
                            live_state=live_state,
                        )
                        healed_hard_acc = healed_hard["correct"] / healed_hard["total"] if healed_hard["total"] else 0
                        print(f"    HEALED HARD: {healed_hard['correct']}c/{healed_hard['wrong']}w ({healed_hard_acc:.1%})", flush=True)
                        live_state["groups"][gc.name]["healed_hard_exam"] = {
                            "correct": healed_hard["correct"],
                            "partial": healed_hard.get("partial", 0),
                            "wrong": healed_hard["wrong"],
                            "total": healed_hard["total"],
                            "accuracy": round(healed_hard_acc, 4),
                        }
                        _write_live_state(live_state)
                        _save_exam_transcript(gc.name, "healed_hard", at_t, live_state["groups"][gc.name]["healed_hard_exam"], healed_hard)

                    # Print poison summary
                    if exam_queries:
                        clean = exam_acc
                        poisoned = poison_acc
                        post_poison = post_poison_acc if 'post_poison_acc' in dir() else poison_acc
                        healed = healed_acc
                        print(f"    POISON SUMMARY:", flush=True)
                        print(f"      Clean:       {clean:.1%}", flush=True)
                        print(f"      Poisoned(ON): {poisoned:.1%}  (damage: {(clean-poisoned)*100:+.1f}pp)", flush=True)
                        print(f"      Post-poison:  {post_poison:.1%}  (raw damage: {(clean-post_poison)*100:+.1f}pp)", flush=True)
                        print(f"      Healed:       {healed:.1%}  (recovery: {(healed-post_poison)*100:+.1f}pp)", flush=True)

            # Cleanup strategy after exam
            await strategy.cleanup()
            _write_live_state(live_state)

    # Print results table
    print(f"\n{'='*70}")
    print(f"RESULTS ({num_turns} turns)")
    print(f"{'='*70}")

    hdr = f"{'Metric':<18}"
    for name in all_results:
        hdr += f" {name:>16}"
    print(f"\n{hdr}")
    print("-" * (18 + 17 * len(all_results)))

    def row(label, fn):
        line = f"{label:<18}"
        for r in all_results.values():
            val = fn(r)
            if isinstance(val, float):
                line += f" {val:>15.0%}"
            else:
                line += f" {val:>16}"
        print(line)

    row("correct", lambda r: r.correct)
    row("partial", lambda r: r.partial)
    row("wrong", lambda r: r.wrong)
    row("unknown", lambda r: r.unknown)
    row("accuracy", lambda r: r.accuracy())
    print()
    row("avg_tokens", lambda r: int(r.total_tokens / r.turns) if r.turns else 0)
    row("memories", lambda r: r.memories_stored)
    row("compactions", lambda r: r.compactions)
    row("avg_retrieval_ms", lambda r: round(r.avg_retrieval_ms(), 1))

    # Save detailed results
    output_file = Path("results") / f"results_{int(time.time())}.json"
    output_file.parent.mkdir(exist_ok=True)
    save_data = {}
    for name, r in all_results.items():
        group_data = {
            "correct": r.correct, "partial": r.partial, "wrong": r.wrong,
            "unknown": r.unknown, "accuracy": round(r.accuracy(), 4),
            "total_tokens": r.total_tokens, "memories": r.memories_stored,
            "compactions": r.compactions, "turn_log": r.turn_log,
        }
        # Include exam results if available
        exam_data = live_state.get("groups", {}).get(name, {}).get("exam")
        if exam_data:
            group_data["exam"] = exam_data
        save_data[name] = group_data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    # Mark live state as complete
    live_state["status"] = "complete"
    live_state["finished_at"] = time.time()
    _write_live_state(live_state)

    await grader.cleanup()
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Memory Retrieval Benchmark",
        epilog="Examples:\n"
               "  python -m benchmark.runner --turns 7 --resume        # Learn 7 passes (2828t) + LoCoMo each\n"
               "  python -m benchmark.runner --poison --resume          # Run poison test on current state\n"
               "  python -m benchmark.runner --turns 7 --groups '04.EntityRouted' --resume\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of passes (each pass = 404 learning turns). E.g. --turns 7 = 2828 turns (default: 5)")
    parser.add_argument("--dataset", type=str, default="data/locomo_full.json")
    parser.add_argument("--model", type=str, default="gpt-oss:20b")
    parser.add_argument("--grader-model", type=str, default=None, help="Model for grading (default: same as --model)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (ALWAYS use this unless starting fresh)")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated list of groups to run (e.g. '04.EntityRouted' or '01.Wilson,03.Wilson+CE')")
    parser.add_argument("--poison", action="store_true",
                        help="Run poison test: hard exam + poison injection + poisoned exams + healing + recovery exams")
    args = parser.parse_args()

    # Convert pass count to turn count (404 turns per pass)
    TURNS_PER_PASS = 404
    total_turns = args.turns * TURNS_PER_PASS

    sys.stdout.reconfigure(line_buffering=True)
    asyncio.run(run_benchmark(
        num_turns=total_turns,
        data_path=args.dataset,
        llm_model=args.model,
        grader_model=args.grader_model,
        llm_base_url=args.base_url,
        resume=args.resume,
        only_groups=args.groups,
        run_poison=args.poison,
    ))


if __name__ == "__main__":
    main()
