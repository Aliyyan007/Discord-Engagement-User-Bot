"""Eudora's mood/emotion engine — PAD model with D1 persistence."""

import time
import random
import math
from typing import Optional, Tuple

from loguru import logger

from core.d1_client import get_d1_client

# ---------------------------------------------------------------------------
# Mood vocabulary
# ---------------------------------------------------------------------------

MOODS = [
    "hyped", "giddy", "happy", "playful", "proud", "curious", "silly", "chill",
    "bored", "tired", "flat", "nostalgic", "annoyed", "sad", "angry",
    "romantic", "horny",
]

# Prefer nearby moods over extreme jumps
MOOD_NEIGHBORS = {
    "hyped":     ["happy", "giddy", "playful", "silly"],
    "giddy":     ["hyped", "silly", "playful", "happy"],
    "happy":     ["hyped", "playful", "curious", "chill", "romantic"],
    "playful":   ["silly", "happy", "curious", "hyped"],
    "curious":   ["playful", "happy", "bored", "silly"],
    "silly":     ["playful", "giddy", "happy", "bored"],
    "proud":     ["happy", "chill", "flat"],
    "chill":     ["happy", "bored", "flat", "curious", "romantic"],
    "bored":     ["flat", "tired", "curious", "annoyed", "chill"],
    "tired":     ["bored", "flat", "chill"],
    "flat":      ["bored", "tired", "nostalgic", "annoyed", "chill"],
    "nostalgic": ["happy", "bored", "flat", "chill"],
    "annoyed":   ["flat", "bored", "tired", "angry"],
    "sad":       ["flat", "nostalgic", "bored", "tired"],
    "angry":     ["annoyed", "flat", "bored"],
    "romantic":  ["happy", "chill", "playful", "horny"],
    "horny":     ["romantic", "playful", "happy", "chill"],
}

# Keyword tuples that nudge mood toward target moods
CONTENT_NUDGES = {
    ("lmao", "lol", "lmaoo", "died", "💀", "😭", "bro what", "npc", "blud",
     "nah", "skull", "💀💀", "lmfao"):                        ["silly", "playful", "hyped"],
    ("how", "why", "explain", "wait", "actually", "what if",
     "interesting", "huh", "really?"):                        ["curious", "curious", "playful"],
    ("stop", "blocked", "hate", "worst", "ugh", "annoying",
     "stupid", "dumb", "trash", "mid"):                        ["flat", "annoyed"],
    ("hug", "love", "miss", "remember", "used to", "nostalgia",
     "good times", "back then", "miss you"):                   ["happy", "nostalgic"],
    ("omg", "no way", "finally", "lets go", "yooo", "w", "clutch",
     "gg", "insane", "crazy", "wild"):                         ["hyped", "giddy", "happy"],
    ("cute", "kiss", "date", "babe", "baby", "love you", "miss you",
     "heart", "❤️", "💕", "🥰", "blush", "pretty"):             ["romantic", "happy"],
    ("sad", "depressed", "anxious", "stress", "hate", "tired", "alone",
     "lonely", "cry", "hurt", "pain", "rough", "hard time"):   ["sad", "flat"],
}

# ---------------------------------------------------------------------------
# PAD -> discrete mood mapping
# ---------------------------------------------------------------------------

def pad_to_mood(valence: float, arousal: float) -> str:
    """Convert (valence, arousal) in [-1, 1] to a discrete mood label."""
    v, a = valence, arousal
    high_v, low_v = v > 0.35, v < -0.25
    high_a, low_a = a > 0.35, a < -0.25

    if high_v and high_a:
        return "hyped"
    if high_v and low_a:
        return "chill" if v < 0.55 else "happy"
    if low_v and high_a:
        return "angry" if v < -0.55 else "annoyed"
    if low_v and low_a:
        if v < -0.55 and a < -0.45:
            return "tired"
        return "sad" if v < -0.4 else "flat"
    # mid / mixed
    if v > 0.1 and a > 0.1:
        return "playful"
    if v > 0.1:
        return "curious"
    if a < -0.2:
        return "bored"
    return "chill"


# Short prompt instructions per mood
MOOD_INSTRUCTIONS = {
    "hyped":     "be energetic, loud, hyped up, lots of exclamation",
    "giddy":     "be giddy, giggly, easily amused, bubbly",
    "happy":     "be warm, cheerful, upbeat, friendly",
    "playful":   "be playful, teasing, lighthearted, joking",
    "proud":     "be confident, proud, a bit smug but warm",
    "curious":   "be curious, ask questions, engaged, attentive",
    "silly":     "be silly, goofy, chaotic, unserious",
    "chill":     "be chill, relaxed, easygoing, low-key",
    "bored":     "be a bit bored, low energy, half-interested",
    "tired":     "be tired, sleepy, slow, low effort",
    "flat":      "be flat, neutral, emotionally muted, dry",
    "nostalgic": "be nostalgic, reflective, wistful, warm about the past",
    "annoyed":   "be annoyed, short, a little snippy, impatient",
    "sad":       "be sad, quiet, soft, down",
    "angry":     "be angry, sharp, irritated, curt",
    "romantic":  "be romantic, sweet, affectionate, tender",
    "horny":     "be flirty, teasing, a little thirsty, suggestive",
}


# ---------------------------------------------------------------------------
# Mood engine
# ---------------------------------------------------------------------------

class MoodEngine:
    """Maintains a persistent emotional state with PAD values + D1 storage."""

    DRIFT_MIN_SECS = 20 * 60   # 20 min before drift may kick in
    DRIFT_MAX_SECS = 45 * 60   # 45 min — near-certain drift by here

    def __init__(self) -> None:
        self.mood: str = "chill"
        self.valence: float = 0.3
        self.arousal: float = 0.3
        self.dominance: float = 0.0
        self.intensity: float = 0.5
        self.mood_set_at: float = time.time()
        self.guild_id: str = ""

    # ---- persistence -------------------------------------------------------

    async def load(self) -> None:
        """Load the latest mood row from D1, or keep defaults if none."""
        try:
            client = get_d1_client()
            rows = client.execute(
                "SELECT mood, valence, arousal, dominance, intensity, updated_at "
                "FROM bot_mood ORDER BY id DESC LIMIT 1",
                [],
            )
            if rows:
                r = rows[0]
                self.mood = r.get("mood", self.mood)
                self.valence = float(r.get("valence", self.valence))
                self.arousal = float(r.get("arousal", self.arousal))
                self.dominance = float(r.get("dominance", self.dominance))
                self.intensity = float(r.get("intensity", self.intensity))
                self.mood_set_at = float(r.get("updated_at", time.time()))
                logger.info(
                    f"[mood] loaded from D1: {self.mood} "
                    f"(V={self.valence:.2f} A={self.arousal:.2f} D={self.dominance:.2f})"
                )
            else:
                logger.info("[mood] no D1 row found, using defaults")
        except Exception as e:
            logger.warning(f"[mood] load failed, using defaults: {e}")

    async def save(self) -> None:
        """Persist current mood to D1 (upsert the single row for this guild)."""
        try:
            client = get_d1_client()
            now = time.time()
            self.mood_set_at = now
            # Try update first
            updated = client.execute_write(
                "UPDATE bot_mood SET mood=?, valence=?, arousal=?, dominance=?, "
                "intensity=?, updated_at=? WHERE guild_id=?",
                [self.mood, self.valence, self.arousal, self.dominance,
                 self.intensity, now, self.guild_id],
            )
            if not updated:
                client.execute_write(
                    "INSERT INTO bot_mood (guild_id, mood, valence, arousal, "
                    "dominance, intensity, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [self.guild_id, self.mood, self.valence, self.arousal,
                     self.dominance, self.intensity, now],
                )
            logger.debug(
                f"[mood] saved to D1: {self.mood} "
                f"(V={self.valence:.2f} A={self.arousal:.2f})"
            )
        except Exception as e:
            logger.warning(f"[mood] save failed: {e}")

    # ---- drift / nudges ----------------------------------------------------

    def _content_nudge(self, text: str) -> Optional[str]:
        """Return a target mood if any nudge keywords match the text."""
        lowered = text.lower()
        for keywords, targets in CONTENT_NUDGES.items():
            if any(k in lowered for k in keywords):
                return random.choice(targets)
        return None

    def maybe_drift(self, trigger_text: str = "") -> str:
        """Possibly shift mood based on content nudges + time-based drift."""
        # (a) Content nudge — 15% chance when keywords match
        if trigger_text:
            target = self._content_nudge(trigger_text)
            if target and target in MOODS and random.random() < 0.15:
                self._shift_to(target)
                logger.debug(f"[mood] nudge -> {self.mood}")
                return self.mood

        # (b) Time-based drift toward a neighbor
        elapsed = time.time() - self.mood_set_at
        if elapsed < self.DRIFT_MIN_SECS:
            return self.mood

        # probability ramps from ~0.2 at DRIFT_MIN_SECS to ~0.9 at DRIFT_MAX_SECS
        frac = min((elapsed - self.DRIFT_MIN_SECS) /
                   (self.DRIFT_MAX_SECS - self.DRIFT_MIN_SECS), 1.0)
        prob = 0.2 + 0.7 * frac
        if random.random() < prob:
            neighbors = MOOD_NEIGHBORS.get(self.mood, ["chill"])
            self._shift_to(random.choice(neighbors))
            logger.debug(f"[mood] drift -> {self.mood} (elapsed {elapsed:.0f}s)")
        return self.mood

    def _shift_to(self, new_mood: str) -> None:
        """Shift mood label and nudge PAD values toward the new mood's region."""
        if new_mood not in MOODS:
            return
        self.mood = new_mood
        self.mood_set_at = time.time()
        # gently pull PAD toward the new mood's typical region
        targets = {
            "hyped": (0.8, 0.8), "giddy": (0.7, 0.6), "happy": (0.6, 0.4),
            "playful": (0.5, 0.5), "proud": (0.5, 0.2), "curious": (0.3, 0.3),
            "silly": (0.5, 0.6), "chill": (0.3, 0.1), "bored": (-0.1, -0.3),
            "tired": (-0.2, -0.6), "flat": (-0.1, -0.2), "nostalgic": (0.2, -0.1),
            "annoyed": (-0.4, 0.4), "sad": (-0.5, -0.3), "angry": (-0.7, 0.7),
            "romantic": (0.6, 0.2), "horny": (0.7, 0.5),
        }
        tv, ta = targets.get(new_mood, (0.3, 0.3))
        self.valence = 0.7 * self.valence + 0.3 * tv
        self.arousal = 0.7 * self.arousal + 0.3 * ta
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

    # ---- empathy / user emotion -------------------------------------------

    def update_from_user_emotion(self, user_valence: float, user_arousal: float) -> None:
        """Shift bot PAD toward user's emotion with inertia + empathy + noise."""
        inertia, empathy = 0.6, 0.4
        noise = lambda: random.uniform(-0.05, 0.05)
        self.valence = max(-1.0, min(1.0,
            inertia * self.valence + empathy * user_valence + noise()))
        self.arousal = max(-1.0, min(1.0,
            inertia * self.arousal + empathy * user_arousal + noise()))
        # intensity tracks how far from neutral the PAD is
        self.intensity = max(-1.0, min(1.0,
            math.sqrt(self.valence ** 2 + self.arousal ** 2) / math.sqrt(2)))
        # refresh mood label from PAD
        self.mood = pad_to_mood(self.valence, self.arousal)
        self.mood_set_at = time.time()

    # ---- owner override ----------------------------------------------------

    def set_mood(self, mood: str) -> bool:
        """Force-set mood (owner command). Returns True if valid."""
        if mood not in MOODS:
            return False
        self._shift_to(mood)
        logger.info(f"[mood] force-set to {mood}")
        return True

    # ---- accessors ---------------------------------------------------------

    def get_mood(self) -> str:
        return self.mood

    def get_pad(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    def get_intensity(self) -> float:
        return self.intensity

    def get_mood_context(self) -> str:
        """Return a context string for the AI prompt."""
        instr = MOOD_INSTRUCTIONS.get(self.mood, "be yourself")
        return (f"[MOOD: {self.mood} (V={self.valence:.2f} A={self.arousal:.2f}) "
                f"— {instr}]")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[MoodEngine] = None


def get_mood_engine() -> MoodEngine:
    global _engine
    if _engine is None:
        _engine = MoodEngine()
    return _engine


async def init_mood_engine() -> MoodEngine:
    global _engine
    _engine = MoodEngine()
    await _engine.load()
    return _engine
