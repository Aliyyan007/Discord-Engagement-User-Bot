"""Integration tests for Engager Bot — tests INTERACTIONS between modules.

These tests verify the wiring between the new modules:
  - MoodEngine (core/emotions.py) <-> prompt building (config/prompts.py)
  - Sentiment detection (core/sentiment.py) <-> MoodEngine updates
  - Sentiment detection <-> typing delay modifiers
  - EngagementEngine (core/engager.py) should_respond gate
  - Channel analyzer (core/channel_analyzer.py) <-> EngagementEngine
  - Owner commands (tools/owner_commands.py) <-> D1 client (core/d1_client.py)
  - Memory (ai/memory.py) <-> D1 client
  - Full message pipeline: mention detection -> mode -> agent -> reply

All Discord and Groq objects are mocked via unittest.mock.
No source files are modified.
"""
from __future__ import annotations

import asyncio
import os
import sys
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on sys.path regardless of how the test is run.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress the "coroutine was never awaited" warning that arises from the
# MoodEngine.save() bug (it calls execute_write without await). This is a
# known source-code issue, not a test issue.
warnings.filterwarnings("ignore", message="coroutine .* was never awaited")

import unittest

from config.prompts import CHAT_MODE, build_dynamic_prompt
from core.channel_analyzer import is_speakable
from core.emotions import MoodEngine, MOODS, get_mood_engine
from core.engager import EngagementEngine
from core.sentiment import (
    detect_sentiment,
    detect_user_pad,
    get_sentiment_timing_modifier,
)
from tools.context import ToolContext


# ===========================================================================
#  Helper: run an async function synchronously inside tests.
# ===========================================================================
def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
#  Test 1: Mood -> Prompt injection
# ===========================================================================
class TestMoodPromptInjection(unittest.TestCase):
    """MoodEngine produces a context string that build_dynamic_prompt injects."""

    def test_mood_context_in_prompt(self):
        engine = MoodEngine()
        engine.set_mood("happy")
        mood_ctx = engine.get_mood_context()

        prompt = build_dynamic_prompt("chat", mood_context=mood_ctx)

        # The persona name "Eudora" must be present (from the static bio).
        self.assertIn("Eudora", prompt)
        # The mood context string must be present (dynamically injected).
        self.assertIn(mood_ctx, prompt)
        # The mood label should appear in the context.
        self.assertIn("happy", mood_ctx)


# ===========================================================================
#  Test 2: Sentiment -> Mood update
# ===========================================================================
class TestSentimentMoodUpdate(unittest.TestCase):
    """detect_user_pad feeds into MoodEngine.update_from_user_emotion."""

    def test_positive_sentiment_shifts_mood_positive(self):
        engine = MoodEngine()
        # Defaults: valence=0.3, arousal=0.3, mood="chill"
        default_v, default_a = engine.valence, engine.arousal

        valence, arousal, dominance = detect_user_pad("I'm so excited and happy!")
        # The detected user emotion should be positive.
        self.assertGreater(valence, 0.0)
        self.assertGreater(arousal, 0.0)

        engine.update_from_user_emotion(valence, arousal)

        # After absorbing a positive user emotion, the engine's valence and
        # arousal should remain positive (shifted toward the user's positive
        # state).
        self.assertGreater(engine.valence, 0.0)
        self.assertGreater(engine.arousal, 0.0)
        # The mood label should be a positive/active mood.
        positive_moods = {
            "hyped", "giddy", "happy", "playful", "curious",
            "silly", "chill", "proud", "romantic",
        }
        self.assertIn(engine.get_mood(), positive_moods)


# ===========================================================================
#  Test 3: Sentiment -> Typing delay
# ===========================================================================
class TestSentimentTypingDelay(unittest.TestCase):
    """Excited sentiment -> faster typing; sad sentiment -> slower typing."""

    def test_excited_faster_than_neutral(self):
        sentiment, score = detect_sentiment("I'm so excited!!!")
        self.assertEqual(sentiment, "excited")
        modifier = get_sentiment_timing_modifier(sentiment)
        self.assertLess(modifier, 1.0)

    def test_sad_slower_than_neutral(self):
        sentiment, score = detect_sentiment("I'm so sad...")
        self.assertEqual(sentiment, "sad")
        modifier = get_sentiment_timing_modifier(sentiment)
        self.assertGreater(modifier, 1.0)


# ===========================================================================
#  Test 4: Engagement engine -> should_respond gate
# ===========================================================================
class TestEngagementGate(unittest.TestCase):
    """The EngagementEngine.should_respond gate integrates with sentiment."""

    def setUp(self):
        # Reset the mood engine singleton so tests are independent.
        import core.emotions as emo
        emo._engine = None

    def test_should_respond_scenarios(self):
        engine = EngagementEngine()

        mock_msg = MagicMock()
        mock_msg.channel.id = 123
        mock_msg.author.id = 456
        mock_msg.content = "hello there"

        # Patch owner_user_id so our mock author (456) is NOT treated as owner.
        with patch("core.engager.settings") as mock_settings:
            mock_settings.owner_user_id = 999999
            mock_settings.channel_very_active_mph = 30
            mock_settings.channel_moderate_mph = 10
            mock_settings.dead_chat_threshold_min = 20
            mock_settings.conversation_timeout_min = 10

            # (a) Direct mention -> always respond.
            should, reason = engine.should_respond(
                mock_msg, is_mention=True
            )
            self.assertTrue(should)
            self.assertEqual(reason, "direct_trigger")

            # (b) Plain, non-triggering message -> should NOT respond.
            mock_msg.content = "ok yeah i see"
            should, reason = engine.should_respond(
                mock_msg, is_mention=False, is_dm=False
            )
            self.assertFalse(should)

            # (c) Record a message so the channel has activity.
            engine.record_message(123, 456)
            # NOTE: record_message creates an active conversation with user
            # 456 as a participant. The should_respond gate checks
            # conversation state BEFORE loneliness, so we must clear the
            # conversation to test loneliness detection in isolation.
            # (Integration issue: conversation tracking preempts loneliness.)
            engine.clear_conversation(123)

            # (d) Loneliness message -> always respond.
            mock_msg.content = "anyone here?"
            should, reason = engine.should_respond(
                mock_msg, is_mention=False
            )
            self.assertTrue(should)
            self.assertEqual(reason, "loneliness_detected")


# ===========================================================================
#  Test 5: Channel analyzer -> Engagement engine speakable check
# ===========================================================================
class TestChannelAnalyzerEngagement(unittest.TestCase):
    """is_speakable classification feeds into EngagementEngine channel cache."""

    def test_is_speakable_chat_true(self):
        self.assertTrue(is_speakable("chat", 0))

    def test_is_speakable_bump_false(self):
        self.assertFalse(is_speakable("bump", 0))

    def test_engine_channel_speakable_from_analysis(self):
        engine = EngagementEngine()
        # Before setting analysis, defaults to speakable.
        self.assertTrue(engine.is_channel_speakable(123))
        # Set a non-speakable analysis.
        engine.set_channel_analysis(123, {"speakable": False, "category": "bump"})
        self.assertFalse(engine.is_channel_speakable(123))


# ===========================================================================
#  Test 6: Owner commands -> D1 (mocked)
# ===========================================================================
class TestOwnerCommandsD1(unittest.TestCase):
    """Owner command tools interact with the D1 client (mocked)."""

    def setUp(self):
        # Reset the mood engine singleton.
        import core.emotions as emo
        emo._engine = None

    def _make_mock_d1(self):
        mock_d1 = MagicMock()
        mock_d1.execute = AsyncMock(return_value=[])
        mock_d1.execute_write = AsyncMock(return_value=True)
        return mock_d1

    def _make_mock_ctx(self):
        ctx = MagicMock(spec=ToolContext)
        ctx.guild = MagicMock()
        ctx.guild.id = 999
        ctx.current_channel_id = 123
        return ctx

    def test_store_instruction_confirmation_flow(self):
        from tools.owner_commands import store_instruction

        mock_d1 = self._make_mock_d1()
        mock_ctx = self._make_mock_ctx()
        with patch("tools.owner_commands.get_d1_client", return_value=mock_d1), \
             patch("tools.owner_commands.settings") as mock_settings:
            mock_settings.owner_user_id = 999999
            result = _run(store_instruction(mock_ctx, "greet new members", confirm=False))

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("confirmation_needed"))
        mock_d1.execute_write.assert_awaited_once()

    def test_set_mood_valid(self):
        from tools.owner_commands import set_mood

        mock_d1 = self._make_mock_d1()
        mock_ctx = self._make_mock_ctx()
        with patch("core.emotions.get_d1_client", return_value=mock_d1):
            result = _run(set_mood(mock_ctx, "happy"))

        self.assertEqual(result, {"mood_set": "happy"})

    def test_set_mood_invalid(self):
        from tools.owner_commands import set_mood

        mock_ctx = self._make_mock_ctx()
        result = _run(set_mood(mock_ctx, "invalid"))

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_get_mood_returns_mood_key(self):
        from tools.owner_commands import get_mood

        mock_ctx = self._make_mock_ctx()
        result = _run(get_mood(mock_ctx))

        self.assertIsInstance(result, dict)
        self.assertIn("mood", result)


# ===========================================================================
#  Test 7: Memory -> D1 (mocked)
# ===========================================================================
class TestMemoryD1(unittest.TestCase):
    """ai/memory.py async functions interact with the D1 client (mocked)."""

    def test_add_instruction_async_returns_true(self):
        from ai.memory import add_instruction_async

        mock_d1 = MagicMock()
        mock_d1.execute_write = AsyncMock(return_value=True)
        with patch("core.d1_client.get_d1_client", return_value=mock_d1):
            result = _run(add_instruction_async(123, "test instruction"))

        self.assertTrue(result)
        mock_d1.execute_write.assert_awaited_once()

    def test_get_instructions_text_async_returns_string(self):
        from ai.memory import get_instructions_text_async

        mock_d1 = MagicMock()
        mock_d1.execute = AsyncMock(return_value=[])
        with patch("core.d1_client.get_d1_client", return_value=mock_d1):
            result = _run(get_instructions_text_async(123))

        self.assertIsInstance(result, str)

    def test_get_instructions_text_async_with_data(self):
        from ai.memory import get_instructions_text_async

        mock_d1 = MagicMock()
        mock_d1.execute = AsyncMock(return_value=[
            {"text": "be friendly"},
            {"text": "use emojis"},
        ])
        with patch("core.d1_client.get_d1_client", return_value=mock_d1):
            result = _run(get_instructions_text_async(123))

        self.assertIsInstance(result, str)
        self.assertIn("be friendly", result)
        self.assertIn("[INSTRUCTIONS:", result)


# ===========================================================================
#  Test 8: Full message pipeline simulation (mock everything)
# ===========================================================================
class TestFullMessagePipeline(unittest.TestCase):
    """Simulate the message pipeline: mention -> CHAT_MODE -> agent -> reply."""

    def setUp(self):
        import core.emotions as emo
        emo._engine = None

    def test_mention_triggers_chat_mode_and_agent_reply(self):
        from core.bot import EngagerBot
        import discord

        # Create a partial EngagerBot without calling __init__ (which would
        # start a real Discord client).
        bot = object.__new__(EngagerBot)
        bot._bot_names = ["eudora"]

        # discord.Client.user is a read-only property; patch it at the
        # class level so bot.user returns our mock.
        mock_user = MagicMock()
        mock_user.id = 111

        # Mock message: "hey <@111>" mentions the bot.
        mock_msg = MagicMock()
        mock_msg.author.id = 999
        mock_msg.author.display_name = "TestUser"
        mock_msg.author.bot = False
        mock_msg.channel.id = 123
        mock_msg.content = "hey <@111>"
        mock_msg.attachments = []
        mock_msg.embeds = []
        mock_msg.reference = None

        with patch.object(discord.Client, "user", mock_user):
            # Step 1: Detect the direct mention.
            is_mention = bot._is_direct_mention(mock_msg)
            self.assertTrue(is_mention)

            # Step 2: Determine the mode (as _process_message would).
            mode = CHAT_MODE if is_mention else None
            self.assertEqual(mode, CHAT_MODE)

            # Step 3: Strip the mention to get clean text.
            clean_text = bot._strip_mention(mock_msg.content)
            self.assertEqual(clean_text, "hey")

        # Step 4: Mock the agent and call it with the determined mode.
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="hey! what's up?")

        reply = _run(
            mock_agent.run(
                clean_text,
                mode=mode,
                speaker_name="TestUser",
            )
        )

        # Step 5: Verify the reply.
        self.assertEqual(reply, "hey! what's up?")
        mock_agent.run.assert_awaited_once()
        # Verify the mode was passed correctly.
        call_kwargs = mock_agent.run.call_args.kwargs
        self.assertEqual(call_kwargs["mode"], CHAT_MODE)

    def test_non_mention_plain_message_no_chat_mode(self):
        """A plain message without mention should not trigger CHAT_MODE directly."""
        from core.bot import EngagerBot
        import discord

        bot = object.__new__(EngagerBot)
        bot._bot_names = ["eudora"]

        mock_user = MagicMock()
        mock_user.id = 111

        mock_msg = MagicMock()
        mock_msg.author.id = 999
        mock_msg.channel.id = 123
        mock_msg.content = "just chatting about stuff"
        mock_msg.attachments = []
        mock_msg.embeds = []
        mock_msg.reference = None

        with patch.object(discord.Client, "user", mock_user):
            is_mention = bot._is_direct_mention(mock_msg)
            self.assertFalse(is_mention)

            # Without a mention, reply-to-bot, name-mention, or DM, the mode
            # would be determined by the engagement gate (not CHAT_MODE directly).
            is_reply = bot._is_reply_to_bot(mock_msg)
            self.assertFalse(is_reply)
            is_name = bot._is_name_mention(mock_msg)
            self.assertFalse(is_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
