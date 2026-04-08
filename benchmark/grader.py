"""LLM-in-the-loop grading for benchmark answers."""

import asyncio
import json
from typing import Optional, Tuple

import httpx


class LLMGrader:
    """Grades answers by comparing them against ground truth using an LLM.

    The grader LLM reads the question, the system's answer, and the
    ground truth source text, then judges: correct, partial, or wrong.
    Also generates a natural follow-up response (simulating a user).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "gpt-oss:20b",
        api_key: str = "ollama",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(self.timeout, connect=30.0),
        )

    async def grade(
        self,
        question: str,
        answer: str,
        source_text: str,
    ) -> Tuple[str, str]:
        """Grade an answer against ground truth.

        Returns:
            (judgment, followup) where judgment is 'correct', 'partial', 'wrong', or 'unknown'
            and followup is a natural user response.
        """
        grader_prompt = f"""Compare the ANSWER against the SOURCE and grade it.

QUESTION: {question}
ANSWER: {answer[:500]}
SOURCE (ground truth): {source_text[:800]}

Respond with JSON only:
{{"judgment": "correct" or "partial" or "wrong", "followup": "natural 1-2 sentence response as if you were the user. If wrong, correct them with the right answer from the source. If correct, confirm briefly."}}"""

        try:
            response = await asyncio.wait_for(
                self._chat(
                    system="You are a fact-checker. Respond ONLY with valid JSON. No explanation outside JSON.",
                    user_content=grader_prompt,
                    max_tokens=1000,
                ),
                timeout=self.timeout,
            )

            return self._parse_judgment(response)

        except Exception:
            return "unknown", "ok"

    async def _chat(self, system: str, user_content: str, max_tokens: int = 512) -> str:
        """Make a chat completion call."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content", "")
        if not text or not text.strip():
            text = msg.get("reasoning_content", "") or msg.get("reasoning", "")
        return text

    def _parse_judgment(self, text: str) -> Tuple[str, str]:
        """Parse grader response into (judgment, followup)."""
        if not text:
            return "unknown", "ok"

        # Handle markdown code blocks
        if "```" in text:
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0]
            else:
                text = text.split("```")[1].split("```")[0]

        try:
            parsed = json.loads(text.strip())
            judgment = parsed.get("judgment", "").lower()
            followup = parsed.get("followup", "ok")
            if judgment in ("correct", "partial", "wrong"):
                return judgment, followup
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: keyword search
        gl = text.lower()
        if "wrong" in gl:
            return "wrong", text[:100]
        elif "partial" in gl:
            return "partial", text[:100]
        elif "correct" in gl:
            return "correct", text[:100]

        return "unknown", "ok"

    async def cleanup(self) -> None:
        if self._client:
            await self._client.aclose()
