"""The Engager self-bot client.

Wires together:
  - the discord.py-self client,
  - the ToolContext (bot + primary guild),
  - the central Agent (with chat / command / event modes),
  - the event handlers,
  - the full conversational message handler:

Trigger scenarios (all covered):
  1. OWNER COMMAND — the owner posts in the command channel or DMs the bot.
     → COMMAND_MODE: the agent uses tools to execute the instruction.
  2. DIRECT MENTION — anyone @mentions the bot (<@bot_id>) in any channel.
     → CHAT_MODE: the agent responds naturally as a human.
  3. REPLY TO BOT — anyone replies to one of the bot's messages.
     → CHAT_MODE: the agent continues the conversation.
  4. NAME MENTION — someone says the bot's display name or username in chat
     (without an @mention). → CHAT_MODE (fuzzy match on the name).
  5. DM FROM ANYONE — anyone DMs the bot.
     → CHAT_MODE: the agent chats with them.

In chat mode, the bot also reads the last few messages in the channel for
context, and maintains per-channel conversation memory so it has continuity.
"""
from __future__ import annotations

import asyncio
import random
import re
from typing import Optional

import discord

from ai.agent import Agent
from ai.memory import memory
from config.prompts import CHAT_MODE, COMMAND_MODE
from config.settings import settings
from core.events import _register
from tools.context import ToolContext
from utils.fuzzy import fuzzy_search
from utils.logger import logger

# How many recent channel messages to read for context in chat mode.
# Kept small to stay under Groq free-tier TPM limits.
CONTEXT_MESSAGE_COUNT = 4
# Name-mention: minimum fuzzy score to consider it a mention.
NAME_MENTION_THRESHOLD = 80


class EngagerBot(discord.Client):
    """Self-bot client with full conversational triggers."""

    def __init__(self) -> None:
        # discord.py-self runs on a user account — no Intents() needed.
        super().__init__()
        from collections import deque
        self.engager_leaves: deque = deque(maxlen=1000)
        self.ctx: ToolContext = ToolContext(bot=self)
        self.agent: Agent = Agent(self.ctx)
        self._msg_lock = asyncio.Lock()
        _register(self, self.ctx, self.agent)
        # Cache the bot's names for name-mention detection.
        self._bot_names: list[str] = []
        logger.info("EngagerBot initialised — agent & events wired.")

    async def close(self) -> None:
        """Graceful shutdown: cancel auto-bump + voice tasks before closing."""
        try:
            from tools.bump_manager import cancel_all_auto_bumps
            cancel_all_auto_bumps()
        except Exception:
            pass
        try:
            from voice import manager as voice_manager
            await voice_manager.stop_all()
        except Exception:
            pass
        await super().close()

    async def on_ready(self):
        logger.success(
            f"Logged in as {self.user} ({self.user.id}) — "
            f"{len(self.guilds)} guild(s)."
        )
        if self.ctx and self.ctx.guild is None and self.guilds:
            self.ctx.guild = self.guilds[0]
            logger.info(f"Primary guild resolved: {self.ctx.guild.name} ({self.ctx.guild.id})")

        # Build the list of names the bot responds to.
        display = self.user.display_name or ""
        username = str(self.user)  # e.g. "eudora_00"
        # Also try the username without the discriminator.
        username_clean = username.split("#")[0] if "#" in username else username
        self._bot_names = [
            n.lower().strip() for n in [display, username, username_clean]
            if n and n.strip()
        ]
        # Deduplicate.
        self._bot_names = list(set(self._bot_names))
        logger.info(f"Responding to names: {self._bot_names}")

        # Auto-join the configured server invite if provided & not already in.
        if settings.server_invite and not self.guilds:
            try:
                await self.accept_invite(settings.server_invite)
                logger.info(f"Accepted invite: {settings.server_invite}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to accept invite: {e}")

    # ------------------------------------------------------------------ #
    #  Trigger detection
    # ------------------------------------------------------------------ #
    def _is_direct_mention(self, message: discord.Message) -> bool:
        """Check if the message contains <@bot_id> or <@!bot_id>."""
        content = message.content or ""
        patterns = [f"<@{self.user.id}>", f"<@!{self.user.id}>"]
        return any(p in content for p in patterns)

    def _is_reply_to_bot(self, message: discord.Message) -> bool:
        """Check if this message is a reply to one of the bot's messages."""
        ref = message.reference
        if ref is None:
            return False
        # discord.py-self may resolve the referenced message into ref.resolved
        # — but it can also be a DeletedReferencedMessage (no .author).
        resolved = getattr(ref, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved.author.id == self.user.id
        # If not resolved, we can't tell — don't trigger on uncertainty.
        return False

    def _is_name_mention(self, message: discord.Message) -> bool:
        """Check if the bot's name appears in the message text (fuzzy)."""
        content = (message.content or "").lower()
        if not content or len(content) > 500:
            return False
        # Quick substring check first (fast path).
        for name in self._bot_names:
            if name in content:
                return True
        # Fuzzy check: does any word/phrase in the content match a bot name?
        words = re.findall(r"\b[\w]{3,}\b", content)
        if not words:
            return False
        for name in self._bot_names:
            if not name:
                continue
            results = fuzzy_search(name, words, score_cutoff=NAME_MENTION_THRESHOLD, limit=1)
            if results:
                return True
        return False

    def _strip_mention(self, content: str) -> str:
        """Remove the <@bot_id> mention from the content, leaving clean text."""
        content = content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        return content.strip()

    # ------------------------------------------------------------------ #
    #  Context gathering
    # ------------------------------------------------------------------ #
    async def _gather_context(self, channel: discord.abc.Messageable) -> str:
        """Read the last N messages in the channel for context (chat mode).

        Returns a formatted string of recent messages (excluding the bot's own).
        Uses lightweight placeholders for images (no vision calls) to save tokens.
        """
        try:
            history = [m async for m in channel.history(limit=CONTEXT_MESSAGE_COUNT)]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Context gather failed: {e}")
            return ""
        # Reverse to chronological, exclude the triggering message (last) and
        # the bot's own messages.
        history.reverse()
        lines: list[str] = []
        for m in history[:-1]:  # skip the last (the triggering message)
            if m.author.id == self.user.id:
                continue
            author = m.author.display_name
            text = (m.content or "").strip()
            # Lightweight attachment placeholders (no vision — saves tokens/time).
            if m.attachments:
                from tools.messages import _attachment_kind
                parts = []
                for att in m.attachments:
                    kind = _attachment_kind(att)
                    parts.append(f"[{kind}: {att.filename}]" if kind != "image" else f"[image: {att.filename}]")
                if parts:
                    text = f"{text} {' '.join(parts)}".strip() if text else " ".join(parts)
            # Compact embed summaries.
            if m.embeds:
                from utils.embeds import embed_to_json
                for e in m.embeds:
                    ej = embed_to_json(e)
                    title = ej.get("title", "")
                    desc = ej.get("description", "")
                    if title or desc:
                        text += f" [embed: {title} — {desc}]".strip()
            if not text:
                continue
            lines.append(f"[{author}]: {text}")
        return "\n".join(lines[-4:])  # keep last 4 for compactness

    # ------------------------------------------------------------------ #
    #  Main message handler
    # ------------------------------------------------------------------ #
    async def on_message(self, message: discord.Message) -> None:
        # Never react to our own messages.
        if message.author.id == self.user.id:
            return

        # Track bump bot messages BEFORE ignoring bots (for cooldown detection).
        from tools.bump_manager import handle_bump_message
        try:
            bump_handled = await handle_bump_message(self.ctx, message)
            if bump_handled:
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Bump handler error: {e}")

        # Ignore other bots (optional — comment out to respond to bots too).
        if message.author.bot:
            return
        # Must have content, attachments, or embeds to process.
        if not (message.content and message.content.strip()) and not message.attachments and not message.embeds:
            return

        # Serialize agent processing to avoid race conditions on ctx.current_channel_id.
        async with self._msg_lock:
            await self._process_message(message)

    async def _process_message(self, message: discord.Message) -> None:
        """Process a single message through the agent (called under _msg_lock)."""
        is_owner = message.author.id == settings.owner_user_id
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_command_channel = (
            settings.command_channel_id
            and message.channel.id == settings.command_channel_id
        )
        channel_id = message.channel.id
        speaker = message.author.display_name

        # ---- Determine trigger type and mode ---------------------------
        mode: Optional[str] = None
        clean_text = message.content.strip()

        if is_direct_mention := self._is_direct_mention(message):
            mode = CHAT_MODE
            clean_text = self._strip_mention(clean_text)
            logger.info(f"CHAT trigger (mention) from {speaker} in ch#{channel_id}: {clean_text[:80]}")
        elif self._is_reply_to_bot(message):
            mode = CHAT_MODE
            logger.info(f"CHAT trigger (reply) from {speaker} in ch#{channel_id}: {clean_text[:80]}")
        elif is_dm:
            mode = CHAT_MODE
            logger.info(f"CHAT trigger (DM) from {speaker}: {clean_text[:80]}")
        elif is_owner and is_command_channel:
            mode = COMMAND_MODE
            logger.info(f"COMMAND trigger from owner in ch#{channel_id}: {clean_text[:80]}")
        elif self._is_name_mention(message):
            mode = CHAT_MODE
            logger.info(f"CHAT trigger (name) from {speaker} in ch#{channel_id}: {clean_text[:80]}")
        else:
            return  # Not triggered.

        # If the mention was stripped and nothing meaningful remains, don't trigger.
        if not clean_text and not message.attachments and not message.embeds:
            return
        # Enrich the triggering message with image descriptions and embed content.
        if message.attachments:
            from tools.messages import _describe_attachment, _attachment_kind
            parts = []
            for att in message.attachments:
                kind = _attachment_kind(att)
                if kind in ("image", "gif"):
                    try:
                        desc = await _describe_attachment(att)
                        parts.append(f"[image: {desc}]")
                    except Exception:
                        parts.append(f"[image: {att.filename}]")
                else:
                    parts.append(f"[{kind}: {att.filename}]")
            if parts:
                clean_text = f"{clean_text} {' '.join(parts)}".strip() if clean_text else " ".join(parts)
        if message.embeds:
            from utils.embeds import embed_to_json
            for e in message.embeds:
                ej = embed_to_json(e)
                parts = []
                if ej.get("title"):
                    parts.append(ej["title"])
                if ej.get("description"):
                    parts.append(ej["description"])
                for f in (ej.get("fields") or []):
                    parts.append(f"{f.get('name', '')}: {f.get('value', '')}")
                if ej.get("footer", {}).get("text"):
                    parts.append(f"footer: {ej['footer']['text']}")
                if ej.get("author", {}).get("name"):
                    parts.append(f"author: {ej['author']['name']}")
                if ej.get("image"):
                    parts.append(f"image: {ej['image']}")
                if ej.get("thumbnail"):
                    parts.append(f"thumbnail: {ej['thumbnail']}")
                if ej.get("provider", {}).get("name"):
                    parts.append(f"provider: {ej['provider']['name']}")
                embed_summary = f"[embed: {' | '.join(parts)}]" if parts else "[embed: (empty)]"
                clean_text = f"{clean_text} {embed_summary}".strip() if clean_text else embed_summary

        # If this is a reply, also include the referenced message's content/embeds.
        if message.reference and message.reference.message_id:
            try:
                ref_msg = message.reference.resolved
                if ref_msg is None:
                    ref_ch = message.channel
                    ref_msg = await ref_ch.fetch_message(message.reference.message_id)
                if isinstance(ref_msg, discord.Message):
                    ref_parts = []
                    if ref_msg.content:
                        ref_parts.append(f"content: {ref_msg.content[:300]}")
                    if ref_msg.embeds:
                        from utils.embeds import embed_to_json
                        for e in ref_msg.embeds:
                            ej = embed_to_json(e)
                            ep = []
                            if ej.get("title"):
                                ep.append(ej["title"])
                            if ej.get("description"):
                                ep.append(ej["description"])
                            for f in (ej.get("fields") or []):
                                ep.append(f"{f.get('name', '')}: {f.get('value', '')}")
                            if ej.get("footer", {}).get("text"):
                                ep.append(f"footer: {ej['footer']['text']}")
                            if ej.get("author", {}).get("name"):
                                ep.append(f"author: {ej['author']['name']}")
                            if ep:
                                ref_parts.append(f"embed: {' | '.join(ep)}")
                    if ref_msg.attachments:
                        for att in ref_msg.attachments:
                            ref_parts.append(f"attachment: {att.filename}")
                    if ref_parts:
                        ref_summary = f"[replied message from {ref_msg.author.display_name}: {'; '.join(ref_parts)}]"
                        clean_text = f"{clean_text} {ref_summary}".strip() if clean_text else ref_summary
            except Exception:
                pass  # Couldn't fetch referenced message.

        # ---- Gather context (chat mode only) --------------------------
        context_str = None
        if mode == CHAT_MODE and not is_dm:
            context_str = await self._gather_context(message.channel)

        # ---- Get conversation history from memory ----------------------
        history = memory.get_history(channel_id)

        # ---- Set the current channel on the context (for "here"/"this") --
        self.ctx.current_channel_id = channel_id

        # ---- Intent routing: chat vs action ------------------------------
        # Conversation messages get a tool-free single-shot call (cheap,
        # fast, no hallucinated tool calls). Action requests get a filtered
        # tool subset matched to the task instead of all 56 schemas.
        route_kwargs: dict = {}
        try:
            from ai.router import classify, classify_llm, ROUTE_ACTION
            if mode == CHAT_MODE:
                route = await classify_llm(clean_text)
                if route.kind == ROUTE_ACTION:
                    route_kwargs = {"categories": route.categories}
                else:
                    route_kwargs = {"no_tools": True}
            elif mode == COMMAND_MODE:
                r = classify(clean_text)  # heuristic only — no arbiter latency
                if r is not None and r.kind == ROUTE_ACTION and r.categories:
                    route_kwargs = {"categories": r.categories}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"routing skipped: {e}")

        # ---- Run the agent ---------------------------------------------
        try:
            async with message.channel.typing():
                reply = await self.agent.run(
                    clean_text,
                    mode=mode,
                    history=history if history else None,
                    context=context_str,
                    speaker_name=speaker if mode == CHAT_MODE else None,
                    **route_kwargs,
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent run failed")
            reply = None

        # ---- Store in memory & send reply ------------------------------
        if reply and reply.strip().upper() not in ("NOACTION", "NULL"):
            memory.add_user(channel_id, speaker, clean_text)
            memory.add_assistant(channel_id, reply)
            # Burst-split casual chat: a long reply lands as 2-3 rapid
            # messages like a human typing, not one wall of text.
            bursts = self._burst_split(reply) if mode == CHAT_MODE else [reply]
            try:
                for burst in bursts:
                    for i in range(0, len(burst), 2000):
                        await message.channel.send(burst[i:i + 2000])
                    if len(bursts) > 1:
                        await asyncio.sleep(random.uniform(0.4, 1.2))
            except discord.HTTPException as e:
                logger.warning(f"Failed to send reply: {e}")
        elif mode == COMMAND_MODE and not reply:
            # In command mode, always confirm even if reply is empty.
            try:
                await message.channel.send("done")
            except discord.HTTPException as e:
                logger.warning(f"Failed to send confirmation: {e}")

    @staticmethod
    def _burst_split(text: str) -> list[str]:
        """Split a chat reply into human-like burst messages: sentence
        groups sent back-to-back. Only fires on long-ish casual replies —
        short lines and 'done' confirmations stay single."""
        text = text.strip()
        if len(text) < 140:
            return [text]
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) < 2:
            return [text]
        bursts: list[str] = []
        cur = ""
        for p in parts:
            if cur and len(cur) + len(p) > 90 and len(bursts) < 2:
                bursts.append(cur.strip())
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        if cur:
            bursts.append(cur)
        return [b for b in bursts if b] or [text]

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        """Handle message edits — mainly for deferred bot embed updates.

        Some bots (OneBump, DISBOARD) send an empty message first, then edit
        it to add embeds. This handler catches those edits and processes them
        through the bump manager.
        """
        if after.author.id == self.user.id:
            return
        # Only process if embeds were added.
        if not after.embeds:
            return
        # Check if this is a bump bot response.
        from tools.bump_manager import handle_bump_message
        try:
            await handle_bump_message(self.ctx, after)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Bump handler error on edit: {e}")
