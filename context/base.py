"""Context management strategies for conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class Exchange:
    """One conversational exchange."""
    user_msg: str
    assistant_msg: str
    summary: str = ""
    token_estimate: int = 0


@runtime_checkable
class ContextManager(Protocol):
    """Interface for conversation context management."""

    name: str

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Add a completed exchange."""
        ...

    def build_messages(self) -> List[Dict[str, str]]:
        """Build the messages array for the LLM."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return context stats (exchanges kept, tokens, compactions)."""
        ...
