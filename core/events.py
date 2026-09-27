"""Discord event handlers.

Each handler forwards a concise textual description of the event to the
central agent (event_mode=True). The agent decides autonomously whether to
act (e.g. greet a new member) or reply NOACTION.

We also maintain an in-memory leave log on the bot instance
(`bot.engager_leaves`) so the `get_recent_leaves` tool can report it.
"""
from __future__ import annotations

import asyncio
import random

import discord

from ai.agent import Agent
from config.prompts import EVENT_MODE
from config.settings import settings
from tools.context import ToolContext
from utils.logger import logger


async def _maybe_autojoin(bot: discord.Client, ctx: ToolContext,
                          member: discord.Member, channel) -> None:
    """Mood-like autonomy: sometimes join a VC a human just entered.

    Higher chance for the owner / someone the bot was just talking to, small
    for strangers — and only when the bot isn't already in a voice channel.
    """
    if not settings.voice_auto_join or member.bot:
        return
    try:
        guild = channel.guild
        if guild.voice_client and guild.voice_client.is_connected():
            return
        humans = [m for m in channel.members if not m.bot]
        if len(humans) != 1:
            return  # only auto-join a person alone — feels intentional
        chance = 0.6 if member.id == settings.owner_user_id else 0.15
        if random.random() >= chance:
            return
        await asyncio.sleep(random.uniform(4, 14))
        # re-check: still a lone human in that VC and we're still free
        if not channel.members or all(m.bot for m in channel.members):
            return
        if guild.voice_client and guild.voice_client.is_connected():
            return
        from tools.voice import join_voice
        await join_voice(ctx, channel_query=channel.name)
        logger.info(f"Auto-joined VC #{channel.name} (mood: social).")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"auto-join skipped: {e}")


def _register(bot: discord.Client, ctx: ToolContext, agent: Agent) -> None:

    @bot.event
    async def on_member_join(member: discord.Member):
        logger.info(f"Member joined {member.guild.name}: {member.display_name}")
        from tools.prefs import get_pref
        welcome_ch = get_pref("welcome_channel")
        pref_hint = (
            f" IMPORTANT: the owner's preferred welcome channel is "
            f"'{welcome_ch}' — greet them THERE, not the default channel."
            if welcome_ch else ""
        )
        desc = (
            f"A new member just joined the server: {member.display_name} "
            f"(username {member}, id {member.id}). "
            f"Total members now: {member.guild.member_count}.{pref_hint}"
        )
        try:
            await agent.run(desc, mode=EVENT_MODE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"on_member_join agent error: {e}")

    @bot.event
    async def on_member_remove(member: discord.Member):
        logger.info(f"Member left {member.guild.name}: {member.display_name}")
        # record in the in-memory leave log
        leaves = getattr(bot, "engager_leaves", [])
        leaves.append({
            "id": member.id,
            "name": member.display_name,
            "username": str(member),
            "left_at": discord.utils.utcnow().isoformat(),
        })
        # cap the log
        if len(leaves) > 100:
            del leaves[: len(leaves) - 100]
        desc = (
            f"A member just left the server: {member.display_name} "
            f"(username {member}, id {member.id}). "
            f"Total members now: {member.guild.member_count}."
        )
        try:
            await agent.run(desc, mode=EVENT_MODE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"on_member_remove agent error: {e}")

    @bot.event
    async def on_voice_state_update(member, before, after):
        if before.channel == after.channel:
            return
        # Private/DM calls surface PrivateCall objects — no .name/.guild/.members.
        for ch in (before.channel, after.channel):
            if ch is not None and not isinstance(
                ch, (discord.VoiceChannel, discord.StageChannel)
            ):
                return
        # The bot itself was disconnected/moved — clean up the voice session.
        if member.id == bot.user.id:
            if after.channel is None and before.channel is not None:
                try:
                    from voice import manager as voice_manager
                    await voice_manager.stop_session(
                        before.channel.guild.id if before.channel.guild else 0
                    )
                    logger.info("Bot left voice — voice session cleaned up.")
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"voice session cleanup error: {e}")
            return
        if before.channel is None and after.channel is not None:
            desc = (
                f"{member.display_name} joined voice channel "
                f"#{after.channel.name}."
            )
        elif before.channel is not None and after.channel is None:
            desc = (
                f"{member.display_name} left voice channel "
                f"#{before.channel.name}."
            )
        else:
            desc = (
                f"{member.display_name} moved from #{before.channel.name} "
                f"to #{after.channel.name}."
            )
        logger.info(desc)
        # the bot's own choice: a lone human joining a VC may trigger us
        if after.channel is not None:
            asyncio.create_task(_maybe_autojoin(bot, ctx, member, after.channel))
        try:
            await agent.run(desc, mode=EVENT_MODE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"on_voice_state_update agent error: {e}")

    @bot.event
    async def on_guild_channel_create(channel):
        desc = f"A new channel was created: #{channel.name} (type {channel.type})."
        logger.info(desc)
        try:
            await agent.run(desc, mode=EVENT_MODE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"on_guild_channel_create agent error: {e}")

    @bot.event
    async def on_guild_channel_delete(channel):
        desc = f"A channel was deleted: #{channel.name}."
        logger.info(desc)
        try:
            await agent.run(desc, mode=EVENT_MODE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"on_guild_channel_delete agent error: {e}")

    @bot.event
    async def on_disconnect():
        logger.warning("Discord gateway disconnected.")

    @bot.event
    async def on_resumed():
        logger.info("Discord gateway resumed.")

    @bot.event
    async def on_guild_join(guild):
        if ctx.guild is None:
            ctx.guild = guild
            logger.info(f"Primary guild set on join: {guild.name}")

    logger.info("Event handlers registered.")
