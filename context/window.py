"""Sliding window context — keep last N exchanges."""

from collections import deque
from typing import Any, Dict, List

from .base import Exchange


class WindowContext:
    """Keep only the last N exchanges. Simple and token-efficient."""

    name = "window"

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.exchanges: deque[Exchange] = deque(maxlen=window_size)

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        est_tokens = int((len(user_msg.split()) + len(assistant_msg.split())) * 1.3)
        self.exchanges.append(Exchange(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            token_estimate=est_tokens,
        ))

    def build_messages(self) -> List[Dict[str, str]]:
        messages = []
        for ex in self.exchanges:
            messages.append({"role": "user", "content": ex.user_msg})
            messages.append({"role": "assistant", "content": ex.assistant_msg})
        return messages

    def get_stats(self) -> Dict[str, Any]:
        total_tokens = sum(ex.token_estimate for ex in self.exchanges)
        return {
            "context": self.name,
            "window_size": self.window_size,
            "exchanges_kept": len(self.exchanges),
            "est_tokens": total_tokens,
            "compactions": 0,
        }
