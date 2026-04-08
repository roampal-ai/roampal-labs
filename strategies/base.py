"""Base protocol for memory retrieval strategies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class MemoryRecord:
    """A single stored memory."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.5          # Wilson score (or equivalent)
    access_count: int = 0
    collection: str = "working"  # working, history, patterns, memory_bank


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    memories: List[MemoryRecord]
    formatted_injection: str     # Ready-to-inject context string
    query_used: str              # The query that was actually searched (may be rewritten)
    retrieval_ms: float = 0.0   # Time spent retrieving


@runtime_checkable
class MemoryStrategy(Protocol):
    """Interface that all memory strategies must implement.

    This is the core abstraction — swap strategies to compare
    retrieval quality, token efficiency, and learning curves.
    """

    name: str

    async def initialize(self) -> None:
        """Set up storage (ChromaDB collections, models, etc.)."""
        ...

    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory. Returns the memory ID."""
        ...

    async def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Retrieve relevant memories for a query."""
        ...

    async def record_outcome(
        self, memory_ids: List[str], outcome: str, exchange_summary: str = ""
    ) -> None:
        """Record whether retrieved memories were helpful.

        Args:
            memory_ids: IDs of memories that were in context
            outcome: 'worked', 'partial', 'failed', 'unknown'
            exchange_summary: Summary of the exchange for potential new memory creation
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Return strategy-specific stats (memory count, avg score, etc.)."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...
