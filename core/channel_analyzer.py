"""Channel nature classification and caching with D1 persistence.

Classifies each Discord channel into a *category* (chat, bump, bot-command,
intro, etc.) based on its name/topic/slowmode/nsfw flag, and a *nature*
(gaming, art, music, venting, ...) based on recent message content.

Results are cached in the D1 ``channel_cache`` table so the bot doesn't
re-analyse every channel on each restart.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Dict, List, Optional

import discord
from loguru import logger

from core.d1_client import get_d1_client

# ------------------------------------------------------------------ #
# Keyword dictionaries
# ------------------------------------------------------------------ #
CATEGORY_KEYWORDS: Dict[str, set] = {
    "chat": {"chat", "general", "talk", "lounge", "hangout", "social", "random",
             "chill", "off-topic", "off topic", "vibing", "culture", "tech"},
    "bump": {"bump", "bumps", "disboard", "bumpin", "promote", "promotion",
             "server-boost"},
    "bot_command": {"bot", "bot-commands", "commands", "cmd", "cmds", "bot-cmd",
                    "ask-bot", "bot-spam"},
    "introduction": {"intro", "introductions", "intros", "welcome", "new",
                     "join", "say-hi", "meet"},
    "announcement": {"announce", "announcements", "news", "updates",
                     "patch-notes", "blog", "notices"},
    "nsfw": {"nsfw", "lewd", "adult", "18+"},
    "slowmode": set(),  # driven by slowmode_delay, not keywords
    "voice": {"voice", "vc", "music", "stage"},
}

NATURE_KEYWORDS: Dict[str, List[str]] = {
    "gaming": ["game", "play", "valorant", "minecraft", "fortnite", "lol",
               "csgo", "gta", "rpg", "level", "boss", "quest", "gaming",
               "server", "lag", "ping", "fps"],
    "art": ["draw", "art", "paint", "sketch", "design", "creative", "wip",
            "artwork", "doodle", "canvas", "illustration"],
    "music": ["song", "music", "album", "artist", "playlist", "spotify",
              "lofi", "beat", "track", "listen", "band"],
    "venting": ["sad", "depressed", "anxious", "stress", "hate", "tired",
                "alone", "lonely", "cry", "hurt", "pain", "struggle", "mental"],
    "study": ["study", "homework", "exam", "test", "school", "college",
              "university", "assignment", "grade", "professor", "class"],
    "tech": ["code", "programming", "python", "javascript", "bug", "error",
             "server", "api", "database", "linux", "git"],
    "general": ["hello", "hi", "hey", "how", "what", "anyone", "sup", "yo",
                "wbu", "hru"],
    "memes": ["meme", "lol", "lmao", "fr", "based", "cringe", "mid", "💀",
              "😭", "😂"],
    "anime": ["anime", "manga", "otaku", "weeb", "naruto", "one piece", "aot",
              "episode", "season"],
    "food": ["food", "eat", "cook", "recipe", "dinner", "lunch", "breakfast",
             "pizza", "coffee", "tea"],
    "utility": ["rank", "level", "bump", "count", "counting", "boost",
                "verify", "verification", "ticket", "command", "bot command",
                "leaderboard", "stats", "server stats", "member count",
                "level up", "xp", "points", "score", "poll", "vote",
                "giveaway", "suggestion", "starboard"],
}

# Channels with slowmode >= this (1 hour) are considered not speakable.
SLOWMODE_THRESHOLD: int = 3600

# Categories where the bot should NOT chat.
_NON_SPEAKABLE_CATEGORIES = {
    "bump", "bot_command", "introduction", "announcement", "nsfw", "voice",
}

# Cache TTL: 24 hours.
_CACHE_TTL: float = 24 * 60 * 60


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _normalise(text: str) -> str:
    """Lower-case and collapse non-alphanumeric runs for keyword matching."""
    return re.sub(r"[^a-z0-9+#]+", " ", (text or "").lower()).strip()


def _match_keywords(haystack: str, keywords: set) -> int:
    """Count how many keywords appear in the normalised haystack."""
    if not keywords:
        return 0
    hay = f" {haystack} "
    return sum(1 for kw in keywords if f" {kw} " in hay)


# ------------------------------------------------------------------ #
# Classification
# ------------------------------------------------------------------ #
def classify_category(channel: discord.TextChannel) -> str:
    """Classify a channel by name/topic/nsfw/slowmode into a category."""
    name = _normalise(getattr(channel, "name", "") or "")
    topic = _normalise(getattr(channel, "topic", "") or "")
    combined = f"{name} {topic}"

    # NSFW flag takes priority if the channel is marked nsfw.
    if getattr(channel, "nsfw", False):
        return "nsfw"

    # Voice channels are classified by type.
    if isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        return "voice"

    # Keyword-based categories (order matters: bump/bot_command before chat).
    priority = ("bump", "bot_command", "introduction", "announcement",
                "voice", "nsfw", "chat")
    for cat in priority:
        if cat == "nsfw":
            continue  # handled by flag above
        kws = CATEGORY_KEYWORDS.get(cat, set())
        if _match_keywords(combined, kws) > 0:
            return cat

    # Slowmode-driven category.
    slowmode = getattr(channel, "slowmode_delay", 0) or 0
    if slowmode >= SLOWMODE_THRESHOLD:
        return "slowmode"

    return "general"


def classify_nature(messages: List[str]) -> str:
    """Classify channel nature from message content via keyword frequency."""
    if not messages:
        return "general"

    combined = _normalise(" ".join(messages))
    if not combined:
        return "general"

    scores: Counter = Counter()
    for nature, kws in NATURE_KEYWORDS.items():
        hay = f" {combined} "
        count = sum(1 for kw in kws if kw in hay)
        if count:
            scores[nature] = count

    if not scores:
        return "general"

    return scores.most_common(1)[0][0]


def is_speakable(category: str, slowmode_seconds: int) -> bool:
    """Return True if the bot should chat in this channel."""
    if category in _NON_SPEAKABLE_CATEGORIES:
        return False
    if category == "slowmode" and slowmode_seconds >= SLOWMODE_THRESHOLD:
        return False
    return True


# ------------------------------------------------------------------ #
# Full channel analysis
# ------------------------------------------------------------------ #
async def analyze_channel(channel: discord.TextChannel) -> Dict:
    """Full analysis of a single channel (name, topic, history, nature)."""
    category = classify_category(channel)

    # Fetch last 20 messages for nature classification.
    contents: List[str] = []
    try:
        async for msg in channel.history(limit=20):
            if msg.author.bot:
                continue
            text = (msg.content or "").strip()
            if text:
                contents.append(text)
    except discord.Forbidden:
        logger.debug(f"No read access to #{channel.name} — skipping history")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read history for #{channel.name}: {e}")

    nature = classify_nature(contents)

    slowmode = getattr(channel, "slowmode_delay", 0) or 0
    speakable = is_speakable(category, slowmode)
    now = time.time()

    return {
        "channel_id": str(channel.id),
        "guild_id": str(channel.guild.id),
        "name": channel.name or "",
        "topic": channel.topic or "",
        "nsfw": 1 if getattr(channel, "nsfw", False) else 0,
        "slowmode_seconds": int(slowmode),
        "category": category,
        "nature": nature,
        "speakable": 1 if speakable else 0,
        "analyzed_at": now,
        "expires_at": now + _CACHE_TTL,
    }


# ------------------------------------------------------------------ #
# D1 persistence
# ------------------------------------------------------------------ #
_SELECT_SQL = (
    "SELECT channel_id, guild_id, name, topic, nsfw, slowmode_seconds, "
    "category, nature, speakable, analyzed_at, expires_at "
    "FROM channel_cache WHERE channel_id = ? AND expires_at > ?"
)

_UPSERT_SQL = (
    "INSERT OR REPLACE INTO channel_cache "
    "(channel_id, guild_id, name, topic, nsfw, slowmode_seconds, "
    "category, nature, speakable, analyzed_at, expires_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_DELETE_SQL = "DELETE FROM channel_cache WHERE channel_id = ?"


async def get_or_load_channel(channel: discord.TextChannel) -> Dict:
    """Return cached analysis if fresh, otherwise analyse and persist."""
    client = get_d1_client()
    now = time.time()

    try:
        rows = await client.execute(_SELECT_SQL, [str(channel.id), now])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"D1 select failed for channel {channel.id}: {e}")
        rows = []

    if rows:
        row = rows[0]
        # Normalise integer-ish columns back to native types.
        row["nsfw"] = int(row.get("nsfw", 0))
        row["slowmode_seconds"] = int(row.get("slowmode_seconds", 0))
        row["speakable"] = int(row.get("speakable", 1))
        row["analyzed_at"] = float(row.get("analyzed_at", 0))
        row["expires_at"] = float(row.get("expires_at", 0))
        logger.debug(f"Channel #{channel.name} loaded from D1 cache")
        return row

    # Cache miss / expired — analyse and persist.
    analysis = await analyze_channel(channel)
    try:
        await client.execute_write(_UPSERT_SQL, [
            analysis["channel_id"],
            analysis["guild_id"],
            analysis["name"],
            analysis["topic"],
            analysis["nsfw"],
            analysis["slowmode_seconds"],
            analysis["category"],
            analysis["nature"],
            analysis["speakable"],
            analysis["analyzed_at"],
            analysis["expires_at"],
        ])
        logger.info(
            f"Cached channel #{analysis['name']} "
            f"(category={analysis['category']}, nature={analysis['nature']})"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"D1 upsert failed for channel {channel.id}: {e}")

    return analysis


async def invalidate_channel(channel_id: int) -> None:
    """Delete a channel's cached analysis (e.g. on create/update/delete)."""
    client = get_d1_client()
    try:
        await client.execute_write(_DELETE_SQL, [str(channel_id)])
        logger.debug(f"Invalidated D1 cache for channel {channel_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"D1 delete failed for channel {channel_id}: {e}")


# ------------------------------------------------------------------ #
# Guild-wide analysis
# ------------------------------------------------------------------ #
async def analyze_all_channels(guild: discord.Guild) -> Dict[str, Dict]:
    """Analyse every text channel in a guild, skipping fresh cache entries.

    Returns a mapping of ``channel_id -> analysis dict``.
    """
    results: Dict[str, Dict] = {}
    for channel in guild.text_channels:
        try:
            analysis = await get_or_load_channel(channel)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to analyse #{channel.name}: {e}")
            continue
        results[str(channel.id)] = analysis
    logger.info(
        f"Analysed {len(results)} channels in '{guild.name}' "
        f"({sum(1 for a in results.values() if a['speakable'])} speakable)"
    )
    return results


# ------------------------------------------------------------------ #
# Human-readable summary (for AI prompts)
# ------------------------------------------------------------------ #
def get_channel_summary(analysis: Dict) -> str:
    """Return a concise, prompt-friendly summary of a channel analysis."""
    speakable = "yes" if int(analysis.get("speakable", 0)) else "no"
    return (
        f"[CHANNEL: #{analysis.get('name', '?')} — "
        f"category={analysis.get('category', 'general')}, "
        f"nature={analysis.get('nature', 'general')}, "
        f"speakable={speakable}]"
    )
