"""Mood engine — the bot's emotional state drifts with the room.

A single scalar ``score`` (-1..+1) updated by transcript sentiment and room
activity, plus a coarse label the prompt consumes. Deterministic and cheap —
no extra LLM calls.

    rude speech   → score drops   → annoyed/tired of this
    warm speech   → score rises   → happy/chatty
    being ignored → score decays  → bored, more likely to dip
    active talk   → engagement up → content

The label drives: prompt tone line, ask-to-leave timing, silence-breaking
chance, and (when mood < -0.6) actually storming off after being dissed.
"""
from __future__ import annotations

import re
import time

_POSITIVE = re.compile(
    r"\b(love|luv|ily|thanks|thank|thx|tysm|lol|lmao|haha|hilarious|nice|"
    r"cool|awesome|sick|lit|w\b|dope|fire|amazing|best|smart|funny|good|"
    r"great|sweet|cutee?|hehe|haha|omw|agree|real|based|poggers|pog)\b",
    re.I,
)
_NEGATIVE = re.compile(
    r"\b(shut up|stfu|hate|annoying|annoyed|stupid|dumb|bad bot|cringe|"
    r"trash|useless|lame|boring|leave|fuck off|off bot|go away|nobody asked|"
    r"who asked|shush|quiet|ugly|sucks|worst|L\b|loser)\b",
    re.I,
)
_QUESTION = re.compile(r"\?\s*$|^(what|who|why|how|when|where|wanna|do you|"
                       r"are you|can you|did you|is it|should we)\b", re.I)

STATES = ["annoyed", "tired", "bored", "content", "happy", "hyped"]


class MoodEngine:
    def __init__(self) -> None:
        self.score = 0.15          # slightly warm by default
        self._last_active = time.monotonic()
        self._rude_streak = 0

    # ------------------------------------------------------------------ #
    @property
    def label(self) -> str:
        s = self.score
        if s <= -0.6:
            return "annoyed"
        if s <= -0.25:
            return "tired"
        if s <= 0.05:
            return "bored"
        if s <= 0.45:
            return "content"
        if s <= 0.8:
            return "happy"
        return "hyped"

    # ------------------------------------------------------------------ #
    def hear(self, text: str, *, directed: bool = False) -> None:
        """Update mood from one transcript line."""
        t = text.lower()
        pos = len(_POSITIVE.findall(t))
        neg = len(_NEGATIVE.findall(t))
        self._last_active = time.monotonic()
        if neg:
            # insults aimed at the bot hurt more
            self.score -= 0.08 * neg * (1.6 if directed else 1.0)
            self._rude_streak += 1 if directed else 0
        elif pos:
            self.score += 0.05 * pos * (1.4 if directed else 0.8)
            self._rude_streak = 0
        else:
            # neutral chatter nudges gently toward content
            self.score += 0.01 if directed else 0.004
        self.score = max(-1.0, min(1.0, self.score))

    def tick_idle(self, seconds: float) -> None:
        """Called periodically — drifting boredom when ignored."""
        self.score -= min(0.02, seconds / 6000.0)

    # ------------------------------------------------------------------ #
    def prompt_line(self) -> str:
        return {
            "annoyed": "You're honestly kinda annoyed right now — short, dry, maybe a little passive-aggressive. Not dropping essays.",
            "tired": "You're low energy / tired — keep it very brief, mellow.",
            "bored": "You're bored — casual, a bit playful, might start a topic or ask what people are doing.",
            "content": "You're chill — natural, friendly, relaxed.",
            "happy": "You're in a good mood — warm, energetic, playful.",
            "hyped": "You're vibing hard — high energy, jokes, gas people up.",
        }[self.label]

    # ------------------------------------------------------------------ #
    @property
    def wants_to_leave(self) -> bool:
        """Angry + been dissed repeatedly -> ready to storm off."""
        return self.score <= -0.65 and self._rude_streak >= 3

    @property
    def proactive_chance(self) -> float:
        """How likely the bot spontaneously speaks (bored/happy talk more)."""
        return {
            "annoyed": 0.10, "tired": 0.15, "bored": 0.65,
            "content": 0.55, "happy": 0.75, "hyped": 0.8,
        }[self.label]
