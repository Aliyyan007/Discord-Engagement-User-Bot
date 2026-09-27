"""Per-channel conversation memory.

Stores a rolling window of recent conversation turns per channel so the agent
has continuity when someone replies or mentions the bot again. Memory is
in-memory only (not persisted) — it resets on restart, which is fine for a
chat agent.

Each channel keeps at most MAX_MESSAGES turns. When the bot is triggered, the
recent history is passed to the agent as prior conversation context.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List

from config.settings import settings

# Max turns kept per channel (each turn = one user + one assistant exchange
# counted as 2 entries). 10 entries = ~5 exchanges — keeps token usage low.
MAX_ENTRIES = 10


class ConversationMemory:
    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._data: Dict[int, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=max_entries)
        )

    def add_user(self, channel_id: int, name: str, content: str) -> None:
        self._data[channel_id].append({
            "role": "user",
            "content": f"[{name}]: {content}",
        })

    def add_assistant(self, channel_id: int, content: str) -> None:
        self._data[channel_id].append({
            "role": "assistant",
            "content": content,
        })

    def get_history(self, channel_id: int) -> List[dict]:
        return list(self._data.get(channel_id, []))

    def clear(self, channel_id: int) -> None:
        self._data.pop(channel_id, None)


# Module-level singleton.
memory = ConversationMemory()
