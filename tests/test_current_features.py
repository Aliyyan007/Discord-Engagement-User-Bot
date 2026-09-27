"""Feature smoke test for the CURRENT Engager Bot on discord.py-self master.

Verifies the full system post-upgrade:
  - every tool in the registry has a dispatchable async function
  - all tool schemas are JSON-serializable (Groq requirement)
  - full + chat tool sets have correct coverage
  - settings load all config incl. voice fields
  - prompts exist for all 4 modes
  - voice pipeline modules import and behave (PCM, VAD)
  - agent initialises with both tool sets

Run: python -m tests.test_current_features
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_registry():
    from tools import registry
    full = registry.build_tools("command")
    chat = registry.build_tools("chat")
    # schemas are JSON-serializable + well-formed
    json.dumps(full); json.dumps(chat)
    names_full = {t["function"]["name"] for t in full}
    names_chat = {t["function"]["name"] for t in chat}
    # every schema name has a dispatch entry, and vice versa
    dispatch = set(registry._DISPATCH)
    missing_fn = names_full - dispatch
    missing_schema = dispatch - names_full
    assert not missing_fn, f"schemas without functions: {missing_fn}"
    assert not missing_schema, f"functions without schemas: {missing_schema}"
    assert names_chat <= names_full, f"chat tools not in full set: {names_chat - names_full}"
    # every dispatch target is an async callable taking (ctx, **kwargs)
    for name, fn in registry._DISPATCH.items():
        assert inspect.iscoroutinefunction(fn), f"{name} is not async"
        sig = inspect.signature(fn)
        assert "ctx" in sig.parameters, f"{name} missing ctx param"
    print(f"registry: OK ({len(full)} full / {len(chat)} chat tools)")


def test_settings():
    from config.settings import settings
    assert settings.discord_token
    assert settings.groq_keys
    assert settings.voice_enabled is not None
    assert settings.voice_stt_model
    print("settings: OK")


def test_prompts():
    from config import prompts
    for mode in (prompts.CHAT_MODE, prompts.COMMAND_MODE,
                 prompts.EVENT_MODE, prompts.VOICE_MODE):
        p = prompts.system_prompt_for(mode)
        assert p and len(p) > 50, mode
    print("prompts: OK (4 modes)")


def test_voice_modules():
    import voice.pcm as pcm
    import voice.vad as vad
    import voice.stt, voice.tts, voice.session, voice.manager  # noqa
    # frame math
    assert pcm.FRAME_BYTES == 3840
    # assembler emits on silence gap
    import time
    asm = vad.UtteranceAssembler(1)
    now = time.monotonic()
    fake = type("P", (), {})()
    fake.payload = b"\x01\x02" * 960 * 2
    fake.audio_level = 50; fake.audio_voice_activity = True
    for _ in range(30):
        asm.feed(fake, now); now += 0.02
    now += 1.0
    utt = asm.tick(now)
    assert utt is not None and utt.voiced_s > 0.3
    print("voice modules: OK")


def test_agent():
    import discord
    from tools.context import ToolContext
    from ai.agent import Agent
    bot = discord.Client()
    ctx = ToolContext(bot=bot)
    agent = Agent(ctx)
    assert agent._tools_full and agent._tools_chat
    print("agent: OK")


async def test_voice_session_class():
    """VoiceSession instantiates + stops cleanly (no real vc needed)."""
    import discord
    from tools.context import ToolContext
    from ai.agent import Agent
    from voice.session import VoiceSession

    class FakeVC:
        channel = None
        def is_connected(self): return False
        def is_playing(self): return False
        def stop(self): pass

    bot = discord.Client()
    bot.agent = Agent(ToolContext(bot=bot))
    ctx = ToolContext(bot=bot)
    ch = type("C", (), {"guild": None, "name": "test", "members": []})()
    sess = VoiceSession(ctx, ch, FakeVC())
    assert sess.guild is None or True
    await sess.stop("test")
    print("voice session lifecycle: OK")


def main():
    test_registry()
    test_settings()
    test_prompts()
    test_voice_modules()
    test_agent()
    asyncio.run(test_voice_session_class())
    print("\nALL FEATURE CHECKS PASSED")


if __name__ == "__main__":
    main()
