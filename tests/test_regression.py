"""Regression tests for the Engager Bot.

Verifies that the new engagement-system additions did NOT break the existing
AI action-performing capabilities:

  1. Tool registry integrity (all original tools still dispatchable)
  2. Agent initialization (Groq tool-calling loop wiring)
  3. Import integrity (every public module still imports)
  4. Settings integrity (existing + new config present)
  5. Prompt integrity (persona + owner-command docs + dynamic prompt)
  6. Bot.py structure (message handling + trigger detection methods)

Run with:
    python -m pytest tests/test_regression.py -v
    python -m unittest tests.test_regression -v
"""
from __future__ import annotations

import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so `import tools.registry` works
# regardless of how pytest/unittest is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
#  1. Tool registry integrity
# ============================================================
class TestToolRegistry(unittest.TestCase):
    """All 55 original tools must still be present and well-formed."""

    # The original tool names that MUST still be dispatchable.
    REQUIRED_TOOLS = [
        "send_message", "send_dm", "delete_message", "edit_message",
        "react_to_message", "send_gif", "send_sticker",
        "search_channels", "search_members",
        "bump_with_bot", "bump_all", "find_bump_commands", "get_bump_status",
        "join_voice", "leave_voice", "use_slash_command",
        "change_bio", "change_status",
    ]

    def test_required_tools_in_dispatch(self):
        from tools.registry import _DISPATCH
        missing = [t for t in self.REQUIRED_TOOLS if t not in _DISPATCH]
        self.assertEqual(missing, [], f"Missing tools from _DISPATCH: {missing}")

    def test_build_tools_command_at_least_55(self):
        from tools.registry import build_tools
        tools = build_tools("command")
        self.assertIsInstance(tools, list)
        self.assertGreaterEqual(len(tools), 55,
            f"Expected >= 55 command-mode tools, got {len(tools)}")

    def test_build_tools_chat_at_least_39(self):
        from tools.registry import build_tools
        tools = build_tools("chat")
        self.assertIsInstance(tools, list)
        self.assertGreaterEqual(len(tools), 39,
            f"Expected >= 39 chat-mode tools, got {len(tools)}")

    def test_get_tool_names_matches_dispatch(self):
        from tools.registry import get_tool_names, _DISPATCH
        names = get_tool_names()
        self.assertEqual(set(names), set(_DISPATCH.keys()))

    def test_each_tool_schema_has_required_keys(self):
        from tools.registry import build_tools
        for mode in ("command", "chat"):
            tools = build_tools(mode)
            for t in tools:
                self.assertIn("type", t, f"tool in {mode} missing 'type': {t}")
                self.assertEqual(t["type"], "function",
                    f"tool in {mode} has wrong type: {t.get('type')}")
                self.assertIn("function", t, f"tool in {mode} missing 'function': {t}")
                fn = t["function"]
                self.assertIn("name", fn, f"tool in {mode} function missing 'name': {fn}")
                self.assertIn("description", fn,
                    f"tool {fn.get('name')} missing 'description'")
                self.assertIn("parameters", fn,
                    f"tool {fn.get('name')} missing 'parameters'")

    def test_dispatch_callable(self):
        """Every entry in _DISPATCH must be an awaitable callable."""
        from tools.registry import _DISPATCH
        for name, fn in _DISPATCH.items():
            self.assertTrue(callable(fn), f"Dispatch target for '{name}' is not callable")
            self.assertTrue(inspect.iscoroutinefunction(fn),
                f"Dispatch target for '{name}' is not async")


# ============================================================
#  2. Agent initialization
# ============================================================
class TestAgentInit(unittest.TestCase):
    """Agent(ctx) must construct and wire up tools + max_rounds."""

    def test_agent_constructs(self):
        from ai.agent import Agent
        mock_ctx = MagicMock()
        with patch("ai.agent.get_pool") as mock_get_pool:
            mock_get_pool.return_value = MagicMock()
            agent = Agent(mock_ctx)
        self.assertIsNotNone(agent)

    def test_agent_tools_non_empty(self):
        from ai.agent import Agent
        mock_ctx = MagicMock()
        with patch("ai.agent.get_pool") as mock_get_pool:
            mock_get_pool.return_value = MagicMock()
            agent = Agent(mock_ctx)
        self.assertIsInstance(agent._tools_full, list)
        self.assertGreater(len(agent._tools_full), 0,
            "_tools_full must be non-empty")
        self.assertIsInstance(agent._tools_chat, list)
        self.assertGreater(len(agent._tools_chat), 0,
            "_tools_chat must be non-empty")

    def test_agent_max_rounds_matches_settings(self):
        from ai.agent import Agent
        from config.settings import settings
        mock_ctx = MagicMock()
        with patch("ai.agent.get_pool") as mock_get_pool:
            mock_get_pool.return_value = MagicMock()
            agent = Agent(mock_ctx)
        self.assertEqual(agent.max_rounds, settings.max_tool_rounds)

    def test_agent_has_run_method(self):
        from ai.agent import Agent
        mock_ctx = MagicMock()
        with patch("ai.agent.get_pool") as mock_get_pool:
            mock_get_pool.return_value = MagicMock()
            agent = Agent(mock_ctx)
        self.assertTrue(hasattr(agent, "run"))
        self.assertTrue(callable(getattr(agent, "run")))


# ============================================================
#  3. Import integrity
# ============================================================
class TestImports(unittest.TestCase):
    """Every public module must import without error."""

    def test_import_bot(self):
        from core.bot import EngagerBot  # noqa: F401

    def test_import_agent(self):
        from ai.agent import Agent  # noqa: F401

    def test_import_registry_build_dispatch(self):
        from tools.registry import build_tools, dispatch  # noqa: F401

    def test_import_bump_manager(self):
        from tools.bump_manager import (  # noqa: F401
            handle_bump_message, bump_all, find_bump_commands,
        )

    def test_import_messaging(self):
        from tools.messaging import send_message, send_dm, delete_message  # noqa: F401

    def test_import_reactions(self):
        from tools.reactions import react_to_message  # noqa: F401

    def test_import_stickers_gifs(self):
        from tools.stickers_gifs import send_gif, send_sticker  # noqa: F401

    def test_import_voice(self):
        from tools.voice import join_voice, leave_voice  # noqa: F401

    def test_import_prompts(self):
        from config.prompts import (  # noqa: F401
            CHAT_SYSTEM_PROMPT, COMMAND_SYSTEM_PROMPT,
            system_prompt_for, build_dynamic_prompt,
        )

    def test_import_settings(self):
        from config.settings import settings  # noqa: F401


# ============================================================
#  4. Settings integrity
# ============================================================
class TestSettings(unittest.TestCase):
    """Existing + new config fields must be populated."""

    def test_discord_token_nonempty(self):
        from config.settings import settings
        self.assertTrue(settings.discord_token,
            "settings.discord_token is empty")

    def test_owner_user_id_positive(self):
        from config.settings import settings
        self.assertGreater(settings.owner_user_id, 0,
            f"owner_user_id should be > 0, got {settings.owner_user_id}")

    def test_groq_keys_nonempty(self):
        from config.settings import settings
        self.assertTrue(settings.groq_keys,
            "settings.groq_keys is empty — no GROQ_API_KEY* found")

    def test_cf_d1_account_id_nonempty(self):
        from config.settings import settings
        self.assertTrue(settings.cf_d1_account_id,
            "settings.cf_d1_account_id is empty (new D1 config)")

    def test_chat_revive_ping_role_positive(self):
        from config.settings import settings
        self.assertGreater(settings.chat_revive_ping_role, 0,
            f"chat_revive_ping_role should be > 0, got {settings.chat_revive_ping_role}")

    def test_engagement_enabled_true(self):
        from config.settings import settings
        self.assertTrue(settings.engagement_enabled,
            "settings.engagement_enabled should be True")


# ============================================================
#  5. Prompt integrity
# ============================================================
class TestPrompts(unittest.TestCase):
    """Persona + owner-command docs + dynamic prompt builder."""

    def test_chat_prompt_contains_eudora(self):
        from config.prompts import CHAT_SYSTEM_PROMPT
        self.assertIn("Eudora", CHAT_SYSTEM_PROMPT,
            "CHAT_SYSTEM_PROMPT must embed the Eudora persona")

    def test_command_prompt_contains_store_instruction(self):
        from config.prompts import COMMAND_SYSTEM_PROMPT
        self.assertIn("store_instruction", COMMAND_SYSTEM_PROMPT,
            "COMMAND_SYSTEM_PROMPT must document owner commands (store_instruction)")

    def test_system_prompt_for_chat(self):
        from config.prompts import CHAT_SYSTEM_PROMPT, system_prompt_for
        self.assertEqual(system_prompt_for("chat"), CHAT_SYSTEM_PROMPT)

    def test_system_prompt_for_command(self):
        from config.prompts import COMMAND_SYSTEM_PROMPT, system_prompt_for
        self.assertEqual(system_prompt_for("command"), COMMAND_SYSTEM_PROMPT)

    def test_build_dynamic_prompt_contains_eudora_and_mood(self):
        from config.prompts import build_dynamic_prompt
        prompt = build_dynamic_prompt("chat", mood_context="[MOOD: happy]")
        self.assertIn("Eudora", prompt,
            "Dynamic chat prompt must still contain the Eudora persona")
        self.assertIn("[MOOD: happy]", prompt,
            "Dynamic chat prompt must contain the injected mood context")


# ============================================================
#  6. Bot.py structure (read the file, don't instantiate)
# ============================================================
class TestBotStructure(unittest.TestCase):
    """EngagerBot must retain its message-handling + trigger-detection shape.

    We read the source file rather than instantiate the discord.Client
    subclass (which would require a live token / gateway).
    """

    @classmethod
    def setUpClass(cls):
        cls.bot_path = os.path.join(_PROJECT_ROOT, "core", "bot.py")
        with open(cls.bot_path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_init_creates_ctx_agent_msg_lock(self):
        # __init__ must assign self.ctx, self.agent, self._msg_lock
        self.assertIn("self.ctx", self.src,
            "EngagerBot.__init__ must create self.ctx")
        self.assertIn("self.agent", self.src,
            "EngagerBot.__init__ must create self.agent")
        self.assertIn("self._msg_lock", self.src,
            "EngagerBot.__init__ must create self._msg_lock")

    def test_has_trigger_detection_methods(self):
        for method in ("_is_direct_mention", "_is_reply_to_bot", "_is_name_mention"):
            self.assertIn(f"def {method}", self.src,
                f"EngagerBot must define {method}")

    def test_on_message_calls_handle_bump_message_first(self):
        # on_message must call handle_bump_message before other processing.
        self.assertIn("def on_message", self.src,
            "EngagerBot must define on_message")
        self.assertIn("handle_bump_message", self.src,
            "on_message must call handle_bump_message")
        # Verify handle_bump_message appears before the _process_message call
        # in the on_message body (i.e. bump detection happens first).
        idx_bump = self.src.find("handle_bump_message")
        idx_process = self.src.find("await self._process_message")
        self.assertNotEqual(idx_bump, -1, "handle_bump_message not found in bot.py")
        self.assertNotEqual(idx_process, -1, "_process_message call not found in bot.py")
        self.assertLess(idx_bump, idx_process,
            "handle_bump_message must be called BEFORE _process_message in on_message")

    def test_process_message_handles_chat_and_command_modes(self):
        self.assertIn("def _process_message", self.src,
            "EngagerBot must define _process_message")
        self.assertIn("CHAT_MODE", self.src,
            "_process_message must handle CHAT_MODE")
        self.assertIn("COMMAND_MODE", self.src,
            "_process_message must handle COMMAND_MODE")

    def test_event_handlers_registered(self):
        """on_message, on_message_edit must exist; events wired via _register."""
        self.assertIn("def on_message", self.src)
        self.assertIn("def on_message_edit", self.src)
        self.assertIn("_register", self.src,
            "EngagerBot.__init__ must call _register to wire event handlers")


if __name__ == "__main__":
    unittest.main(verbosity=2)
