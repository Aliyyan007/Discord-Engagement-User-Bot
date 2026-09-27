"""Human-like typing simulation.

Before sending a reply the bot pauses for a duration that mimics a real
person typing the message out. The delay is derived from a configurable
WPM setting (default 85), then nudged with jitter, extra hesitation before
whitespace/punctuation, periodic "thinking" pauses, and a sentiment-based
speed modifier (excited = faster, sad = slower). When a Discord channel is
supplied the native typing indicator is shown for the computed duration so
other users see "Bot is typing…" just like a human. A small typo may
optionally be injected to further sell the illusion.
"""
from __future__ import annotations

import asyncio
import random
import string

from config.settings import settings
from utils.logger import logger

_SENTIMENT_MODIFIERS = {
    "excited": 0.6,
    "urgent": 0.5,
    "happy": 0.8,
    "sad": 1.4,
    "angry": 1.3,
    "neutral": 1.0,
}

_PUNCTUATION = set(string.punctuation)


async def human_typing_delay(
    text: str, *, channel=None, sentiment: str = "neutral"
) -> float:
    """Pause for a human-like duration before sending ``text``.

    If ``channel`` is given, the Discord typing indicator is shown for the
    computed duration. Returns the actual delay in seconds (0 if disabled).
    """
    if not settings.typing_sim_enabled:
        return 0.0

    wpm = max(settings.typing_wpm, 1)
    base_ms = 60000 / (wpm * 5)  # ms per character

    total_ms = 0.0
    words = 0
    for i, ch in enumerate(text):
        ms = base_ms
        # Extra hesitation before whitespace / punctuation (+35%).
        if ch.isspace() or ch in _PUNCTUATION:
            ms *= 1.35
        # Jitter ±20%.
        ms *= random.uniform(0.8, 1.2)
        total_ms += ms
        if ch.isspace():
            words += 1
            # Thinking pause every ~12 words (300-500ms).
            if words % 12 == 0:
                total_ms += random.uniform(300, 500)

    # Sentiment speed modifier.
    modifier = _SENTIMENT_MODIFIERS.get(sentiment, 1.0)
    total_ms *= modifier

    delay = max(1.5, min(8.0, total_ms / 1000.0))

    if channel is not None:
        try:
            async with channel.typing():
                await asyncio.sleep(delay)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"typing indicator failed: {e}")
            await asyncio.sleep(delay)
    else:
        await asyncio.sleep(delay)

    return delay


def maybe_add_typo(text: str, typo_chance: float = 0.05) -> str:
    """Occasionally introduce a single subtle typo into ``text``.

    Only acts on words of 3+ characters and at most one typo per message.
    Returns the (possibly modified) text.
    """
    if random.random() > typo_chance:
        return text

    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 3]
    if not candidates:
        return text

    idx = random.choice(candidates)
    word = list(words[idx])
    kind = random.choice(("transpose", "double"))

    if kind == "transpose":
        # Swap two adjacent characters (not at the very end).
        pos = random.randint(0, len(word) - 2)
        word[pos], word[pos + 1] = word[pos + 1], word[pos]
    else:
        # Double a character.
        pos = random.randint(0, len(word) - 1)
        word.insert(pos, word[pos])

    words[idx] = "".join(word)
    return " ".join(words)
