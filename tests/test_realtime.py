"""Real-time integration test harness.

This does NOT use mocks — it logs into Discord with the real token, resolves
the primary guild, and exercises the tool layer + the central agent against
live data. Run it to verify the whole stack end-to-end:

    python -m tests.test_realtime
    python -m tests.test_realtime --agent "ping the newest member in #general saying welcome"
    python -m tests.test_realtime --tool search_channels --args '{"query":"general"}'

Safety: side-effecting tools (send_message, send_gif, react_*, join_voice,
send_sticker, send_dm) are SKIPPED by default. Pass --allow-side-effects to
run them. The agent command is always allowed (it may itself call side
effects, so only use --agent with --allow-side-effects on a test server).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import discord

from config.settings import settings
from core.bot import EngagerBot
from tools.context import ToolContext
from tools.registry import build_tools, dispatch, get_tool_names
from utils.logger import logger

SIDE_EFFECTING = {
    "send_message", "send_dm", "react_to_message", "react_to_user_latest",
    "send_gif", "send_sticker", "join_voice", "leave_voice", "speak_in_stage",
}


async def _wait_for_guild(bot: EngagerBot, timeout: float = 45.0) -> bool:
    """Wait until the bot is ready AND the primary guild cache is populated."""
    loop = asyncio.get_event_loop()
    start = loop.time()
    # First wait for the ready event.
    while not bot.is_ready():
        if loop.time() - start > timeout:
            return False
        await asyncio.sleep(0.3)
    # Then wait until the primary guild has channels cached.
    while True:
        if loop.time() - start > timeout:
            return False
        if bot.guilds and bot.guilds[0].channels:
            break
        await asyncio.sleep(0.3)
    if bot.ctx and bot.ctx.guild is None:
        bot.ctx.guild = bot.guilds[0]
    # Ensure members are loaded (self-bots may need an explicit chunk).
    guild = bot.ctx.guild
    try:
        if guild and not guild.chunked:
            await guild.chunk()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"guild.chunk() failed: {e}")
    return True


async def run_tool_tests(ctx: ToolContext, allow_side_effects: bool) -> None:
    """Run every read-only tool and report pass/fail."""
    tools = get_tool_names()
    passed, failed = 0, 0
    # Tools that take a single `query` arg.
    QUERY_TOOLS = {
        "search_channels", "get_channel_details", "get_channel_mention",
        "get_channel_link", "search_members", "get_member_details",
        "get_member_mention",
    }
    # Tools that need a channel_query.
    CHAN_QUERY_TOOLS = {"get_recent_messages", "get_vc_text_chat"}
    # Tools that take no args.
    NO_ARG_TOOLS = {
        "list_channels", "get_member_count", "get_online_members",
        "get_recent_joins", "get_recent_leaves", "list_stickers",
        "list_voice_channels", "get_voice_state", "leave_voice",
        "speak_in_stage",
    }
    # Tools requiring resources we don't have at test time -> skip.
    SKIP_TOOLS = {
        "react_to_message", "react_to_user_latest", "get_message_by_link",
        "send_message", "send_dm", "send_sticker", "join_voice",
    }
    for name in tools:
        if name in SIDE_EFFECTING and not allow_side_effects:
            logger.info(f"SKIP (side-effecting): {name}")
            continue
        if name in SKIP_TOOLS:
            logger.info(f"SKIP (needs real resource): {name}")
            continue
        # Build args based on the tool's signature.
        args: dict = {}
        if name in QUERY_TOOLS:
            args = {"query": "general"}
        elif name in CHAN_QUERY_TOOLS:
            args = {"channel_query": "general", "limit": 5} if name == "get_recent_messages" else {"channel_query": "general"}
        elif name == "search_gifs":
            args = {"query": "cat", "limit": 3}
        elif name == "send_gif":
            args = {"channel_query": "general", "query": "cat"}
        elif name in NO_ARG_TOOLS:
            args = {}

        try:
            result = await dispatch(ctx, name, args)
            logger.info(f"PASS {name}: {result[:160]}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"FAIL {name}: {e}")
            failed += 1
    logger.info(f"Tool tests done — {passed} passed, {failed} failed.")


async def run_agent(ctx, agent_cmd: str) -> None:
    from ai.agent import Agent
    agent = Agent(ctx)
    logger.info(f"Running agent command: {agent_cmd}")
    reply = await agent.run(agent_cmd)
    logger.info(f"Agent reply:\n{reply}")


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", help="Run a single tool by name")
    parser.add_argument("--args", help='JSON args for --tool', default="{}")
    parser.add_argument("--agent", help="Run a natural-language command through the agent")
    parser.add_argument("--allow-side-effects", action="store_true",
                        help="Allow side-effecting tools to actually run")
    parser.add_argument("--list-tools", action="store_true")
    args = parser.parse_args(argv)

    if args.list_tools:
        for t in get_tool_names():
            print(t)
        return 0

    if not settings.discord_token:
        logger.error("DISCORD_TOKEN missing.")
        return 1

    bot = EngagerBot()
    # Start the bot in the background.
    task = asyncio.create_task(bot.start(settings.discord_token))
    try:
        # Wait for ready.
        ready = await _wait_for_guild(bot, timeout=40.0)
        if not ready:
            logger.error("Bot did not connect / find a guild in time.")
            return 2
        ctx = bot.ctx
        assert ctx is not None

        if args.tool:
            tool_args = json.loads(args.args)
            if args.tool in SIDE_EFFECTING and not args.allow_side_effects:
                logger.warning(f"{args.tool} is side-effecting — pass --allow-side-effects to run.")
            else:
                result = await dispatch(ctx, args.tool, tool_args)
                print(result)
        elif args.agent:
            if not args.allow_side_effects:
                logger.warning("Agent may perform side effects — pass --allow-side-effects.")
            await run_agent(ctx, args.agent)
        else:
            await run_tool_tests(ctx, args.allow_side_effects)
    finally:
        await bot.close()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
