"""Owner command tools: instructions, mood control, engagement toggle.

These tools are only callable by the owner (enforced upstream by the agent
loop via ``settings.owner_user_id``). They persist owner directives to D1,
force-set the bot's emotional state, and toggle the engagement engine at
runtime.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from config.settings import settings
from core.d1_client import get_d1_client
from core.emotions import get_mood_engine, MOODS
from tools.context import ToolContext

# Module-level flag toggled by set_engagement / read by is_engagement_enabled.
_engagement_enabled: bool = True


def is_engagement_enabled() -> bool:
    """Return whether the engagement engine is currently enabled."""
    return _engagement_enabled


# ------------------------------------------------------------------ #
#  Owner instructions
# ------------------------------------------------------------------ #
async def store_instruction(
    ctx: ToolContext, instruction: str, *, confirm: bool = False
) -> dict:
    """Store an owner instruction in D1.

    Two-phase confirmation flow:
      1. ``confirm=False`` (default) -> stored as ``pending``.
      2. ``confirm=True``             -> stored as ``active``.
    """
    try:
        db = get_d1_client()
        now = time.time()
        status = "active" if confirm else "pending"
        guild_id = str(ctx.guild.id) if ctx.guild else ""
        await db.execute_write(
            """INSERT INTO owner_instructions
                 (owner_id, guild_id, instruction, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [str(ctx.author_id or next(iter(settings.owner_ids), 0)),
             guild_id, instruction, status, now, now],
        )
        if confirm:
            return {"stored": True, "activated": True, "message": "Instruction activated."}
        return {
            "stored": True,
            "confirmation_needed": True,
            "message": "Instruction stored. Reply with confirm=true to activate it.",
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"store_instruction failed: {e}")
        return {"error": str(e)}


async def list_instructions(ctx: ToolContext, *, active_only: bool = True) -> dict:
    """List owner instructions from D1, optionally filtered to active."""
    try:
        db = get_d1_client()
        if active_only:
            rows = await db.execute(
                """SELECT id, instruction, status, created_at
                   FROM owner_instructions
                   WHERE status = 'active'
                   ORDER BY created_at DESC""",
            )
        else:
            rows = await db.execute(
                """SELECT id, instruction, status, created_at
                   FROM owner_instructions
                   ORDER BY created_at DESC""",
            )
        instructions = [
            {
                "id": r.get("id"),
                "instruction": r.get("instruction"),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
            }
            for r in rows
        ]
        return {"instructions": instructions}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"list_instructions failed: {e}")
        return {"error": str(e), "instructions": []}


async def remove_instruction(ctx: ToolContext, instruction_id: int) -> dict:
    """Deactivate (mark removed) an owner instruction by ID."""
    try:
        db = get_d1_client()
        rows = await db.execute(
            "SELECT id FROM owner_instructions WHERE id = ?",
            [instruction_id],
        )
        if not rows:
            return {"error": "not found"}
        await db.execute_write(
            """UPDATE owner_instructions
                 SET status = 'removed', updated_at = ?
               WHERE id = ?""",
            [time.time(), instruction_id],
        )
        return {"removed": True}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"remove_instruction failed: {e}")
        return {"error": str(e)}


# ------------------------------------------------------------------ #
#  Mood control
# ------------------------------------------------------------------ #
async def set_mood(ctx: ToolContext, mood: str) -> dict:
    """Force-set the bot's mood. ``mood`` must be in ``MOODS``."""
    if mood not in MOODS:
        return {"error": f"Invalid mood '{mood}'. Valid moods: {MOODS}"}
    try:
        engine = get_mood_engine()
        engine.set_mood(mood)
        await engine.save()
        return {"mood_set": mood}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"set_mood failed: {e}")
        return {"error": str(e)}


async def get_mood(ctx: ToolContext) -> dict:
    """Return the current mood state (mood + VAD + intensity)."""
    try:
        engine = get_mood_engine()
        return {
            "mood": engine.get_mood(),
            "valence": getattr(engine, "valence", 0.0),
            "arousal": getattr(engine, "arousal", 0.0),
            "dominance": getattr(engine, "dominance", 0.0),
            "intensity": getattr(engine, "intensity", 0.0),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"get_mood failed: {e}")
        return {"error": str(e)}


# ------------------------------------------------------------------ #
#  Engagement engine toggle
# ------------------------------------------------------------------ #
async def set_engagement(ctx: ToolContext, enabled: bool) -> dict:
    """Enable or disable the engagement engine at runtime."""
    global _engagement_enabled
    _engagement_enabled = enabled
    logger.info(f"Engagement engine {'enabled' if enabled else 'disabled'} via tool.")
    return {"engagement": "enabled" if enabled else "disabled"}
