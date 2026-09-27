"""Global voice-session registry.

One VoiceSession per guild (a user account can only be in one VC per guild
anyway). Tools and events call ``start_session``/``stop_session`` — the
session object owns the whole listen→think→speak pipeline.
"""
from __future__ import annotations

from typing import Optional

from utils.logger import logger

_sessions: dict[int, "object"] = {}  # guild_id -> VoiceSession


def current_session(guild_id: int):
    return _sessions.get(guild_id)


def all_sessions() -> list:
    return list(_sessions.values())


async def start_session(ctx, channel, vc) -> "object":
    """Create + start a VoiceSession for this voice connection."""
    from .session import VoiceSession

    guild_id = channel.guild.id if channel.guild else 0
    old = _sessions.pop(guild_id, None)
    if old is not None:
        try:
            await old.stop("replaced")
        except Exception:  # noqa: BLE001
            pass
    session = VoiceSession(ctx, channel, vc)
    _sessions[guild_id] = session
    await session.start()
    logger.success(f"Voice session started in #{channel.name}.")
    return session


async def stop_session(guild_id: int) -> None:
    session = _sessions.pop(guild_id, None)
    if session is not None:
        try:
            await session.stop("manual")
        except Exception:  # noqa: BLE001
            pass


def drop_session(guild_id: int) -> None:
    _sessions.pop(guild_id, None)


async def stop_all() -> None:
    for gid in list(_sessions):
        await stop_session(gid)


async def say_in_voice(guild_id: int, text: str) -> bool:
    """Speak a line in the guild's active voice session. False if none."""
    session = _sessions.get(guild_id)
    if session is None:
        return False
    await session.say(text)
    return True
