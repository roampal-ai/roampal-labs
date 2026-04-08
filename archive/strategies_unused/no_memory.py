"""Baseline: no memory retrieval at all."""

from typing import Any, Dict, List, Optional
from .base import MemoryStrategy, MemoryRecord, RetrievalResult


class NoMemoryStrategy:
    """Baseline — LLM answers purely from parametric knowledge."""

    name = "no_memory"

    async def initialize(self) -> None:
        pass

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return ""  # No-op

    async def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        return RetrievalResult(
            memories=[],
            formatted_injection="",
            query_used=query,
        )

    async def record_outcome(
        self, memory_ids: List[str], outcome: str, exchange_summary: str = ""
    ) -> None:
        pass

    async def get_stats(self) -> Dict[str, Any]:
        return {"memories": 0, "strategy": self.name}

    async def cleanup(self) -> None:
        pass
