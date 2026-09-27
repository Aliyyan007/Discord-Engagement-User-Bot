"""Offline simulation tests for the voice pipeline — no Discord, no network.

Simulates the whole chain:
    fake MediaPackets -> UtteranceAssembler -> _utterance_q -> _handle_batch
    -> (mocked) transcribe -> (mocked) classify_llm -> (fake) Agent.run
    -> (mocked) tts.synthesize -> _play_pcm -> FakeVC.play

Network boundaries replaced:
    voice.session.transcribe     -> canned transcripts (in call order)
    voice.session.classify_llm   -> canned router Route (no LLM arbiter)
    voice.tts.synthesize         -> records the text it was asked to speak,
                                    returns 100 ms of silent PCM
    session.agent                -> FakeAgent returning canned replies

The real MoodEngine, _sayable/_NO_SPEAK/_SPEAKER_PREFIX/_NOISE_LINE filters,
backchannel/echo/directed gates, barge-in ticker and VAD all run for real.

Run:  python -m tests.test_voice_simulation
  or: python tests/test_voice_simulation.py

Plain asserts, pytest-free. Tests listed in KNOWN_BUGS document genuine
source bugs — they assert the EXPECTED behaviour and report BUG (not FAIL)
until the source is fixed.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

# allow running as a plain script (python tests/test_voice_simulation.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.router import ROUTE_ACTION, ROUTE_CHAT, Route, classify  # noqa: E402
from voice.pcm import FRAME_BYTES, pcm_duration  # noqa: E402
from voice.vad import Utterance, UtteranceAssembler  # noqa: E402
from voice.session import (  # noqa: E402
    BARGE_IN_VOICED_MS,
    BARGE_IN_WINDOW_S,
    VoiceSession,
)

try:  # real sink when native_voice is installed (it is in this env)
    from discord.ext.native_voice import AsyncQueueSink
except Exception:  # noqa: BLE001
    AsyncQueueSink = None


BOT_ID = 999
ALICE_ID = 101
BOB_ID = 102

# --------------------------------------------------------------------- #
#  synthetic audio
# --------------------------------------------------------------------- #


def _sine_pcm(seconds: float, freq: float = 220.0, amp: float = 0.3) -> bytes:
    """Tone at 48 kHz stereo s16 — stands in for voiced speech energy."""
    n = int(48000 * seconds)
    t = np.arange(n, dtype=np.float32) / 48000
    tone = (np.sin(2 * math.pi * freq * t) * amp * 32767).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


def _silence_pcm(seconds: float) -> bytes:
    return b"\x00" * int(48000 * seconds * 2 * 2)


def _noise_pcm(seconds: float, amp: float = 0.001, seed: int = 0) -> bytes:
    """Very low-energy noise (~-60 dBFS) — below ENERGY_VOICE_DBFS."""
    n = int(48000 * seconds)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(n) * amp * 32767).astype(np.int16)
    return np.repeat(noise, 2).tobytes()


def _frames(pcm: bytes) -> list[bytes]:
    return [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


def _pkt(payload: bytes, uid: int = ALICE_ID, ssrc: int | None = None,
         audio_level=None, audio_voice_activity=None):
    """Fake decoded MediaPacket (mirrors native_voice.MediaPacket fields)."""
    return SimpleNamespace(
        media_type="audio", codec="pcm", payload=payload,
        user_id=uid, ssrc=ssrc if ssrc is not None else 1000 + (uid or 0),
        audio_level=audio_level, audio_voice_activity=audio_voice_activity,
        marker=False, sequence=0, timestamp=0, received_at=None,
    )


def _utt(uid: int, ended_ago: float = 5.0, voiced_s: float = 0.6) -> Utterance:
    """A finished Utterance. ended_ago>1 skips the human-like reply delay."""
    return Utterance(
        user_id=uid, pcm=_sine_pcm(voiced_s), voiced_s=voiced_s,
        ended_at=time.monotonic() - ended_ago,
    )


# --------------------------------------------------------------------- #
#  Discord stand-ins
# --------------------------------------------------------------------- #


class FakeMember:
    def __init__(self, uid: int, name: str, bot: bool = False) -> None:
        self.id = uid
        self.name = name
        self.display_name = name
        self.bot = bot


class FakeGuild:
    def __init__(self, gid: int, members: list[FakeMember]) -> None:
        self.id = gid
        self.members = list(members)
        self.me = next((m for m in members if m.id == BOT_ID), None)

    def get_member(self, uid: int):
        return next((m for m in self.members if m.id == uid), None)


class FakeChannel:
    """Stands in for discord.VoiceChannel."""

    def __init__(self, name: str, members: list[FakeMember], guild: FakeGuild) -> None:
        self.name = name
        self.members = list(members)
        self.guild = guild


class FakeVC:
    """Stands in for the voice client. play() finishes instantly."""

    def __init__(self, channel: FakeChannel) -> None:
        self.channel = channel
        self._connected = True
        self._playing = False
        self._paused = False
        self._after = None
        self.play_calls = 0
        self.played_sources: list = []
        self.stop_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.listen_calls: list = []
        self.disconnected = False

    def is_connected(self) -> bool:
        return self._connected

    def is_playing(self) -> bool:
        return self._playing

    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self.pause_calls += 1
        self._paused = True

    def resume(self) -> None:
        self.resume_calls += 1
        self._paused = False

    def play(self, source, after=None) -> None:
        self.play_calls += 1
        self.played_sources.append(source)
        self._playing = True
        self._after = after
        if after is not None:
            after(None)  # pretend the clip finished instantly
        self._playing = False

    def stop(self) -> None:
        self.stop_calls += 1
        self._playing = False

    def listen(self, sink) -> None:
        self.listen_calls.append(sink)

    def is_listening(self) -> bool:
        return False

    def stop_listening(self) -> None:
        pass

    async def disconnect(self) -> None:
        self._connected = False
        self.disconnected = True


class FakeAgent:
    """Stands in for ai.agent.Agent — records calls, returns canned replies."""

    def __init__(self, replies: list | None = None) -> None:
        self.replies = list(replies) if replies else ["ok cool"]
        self.calls: list[SimpleNamespace] = []

    async def run(self, user_request, *, mode=None, history=None, context=None,
                  speaker_name=None, categories=None, no_tools=False,
                  **_extra) -> str:
        self.calls.append(SimpleNamespace(
            request=user_request, mode=mode, history=history, context=context,
            speaker_name=speaker_name, categories=categories,
            no_tools=no_tools, extra=_extra,
        ))
        r = self.replies.pop(0) if self.replies else "ok cool"
        return r(user_request) if callable(r) else r

    async def run_stream(self, user_request, *, context=None):
        """Streaming counterpart — yields the canned reply as two deltas so
        _speak_streamed's sentence aggregation gets exercised."""
        self.calls.append(SimpleNamespace(
            request=user_request, mode="voice", context=context,
            stream=True, no_tools=True,
        ))
        r = self.replies.pop(0) if self.replies else "ok cool"
        text = r(user_request) if callable(r) else r
        mid = max(1, len(text) // 2)
        yield text[:mid]
        yield text[mid:]


class FakeBot:
    def __init__(self, agent: FakeAgent) -> None:
        self.user = SimpleNamespace(id=BOT_ID, display_name="eudora", bot=True)
        self.agent = agent
        self._bot_names = ["eudora"]


class _QueueSinkStub:
    """Fallback for AsyncQueueSink — only .get() is used by _packet_pump."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def get(self):
        return await self.queue.get()


def build_env(*, replies: list | None = None,
              channel_name: str = "the-lab") -> SimpleNamespace:
    """Wire a VoiceSession to fakes. `spoken` collects every string that
    reaches tts.synthesize — i.e. exactly what the bot would say out loud.
    `routes` collects every transcript the router was asked to classify."""
    alice = FakeMember(ALICE_ID, "alice")
    bob = FakeMember(BOB_ID, "bob")
    me = FakeMember(BOT_ID, "eudora", bot=True)
    guild = FakeGuild(1, [alice, bob, me])
    channel = FakeChannel(channel_name, [alice, bob, me], guild)
    vc = FakeVC(channel)
    agent = FakeAgent(replies)
    bot = FakeBot(agent)
    ctx = SimpleNamespace(bot=bot)
    session = VoiceSession(ctx, channel, vc)
    return SimpleNamespace(
        session=session, vc=vc, agent=agent, bot=bot,
        guild=guild, channel=channel, spoken=[], routes=[],
        members={"alice": alice, "bob": bob, "me": me},
    )


@contextlib.contextmanager
def patched_io(env: SimpleNamespace, transcripts: list[str] = (),
               route: Route | None = None):
    """Replace the network boundary: STT + router + TTS.

    transcripts: returned by transcribe() in call order ('' after exhaustion).
    route:       the Route classify_llm returns (default ROUTE_CHAT).
    """
    tx = list(transcripts)
    idx = [0]
    canned = route if route is not None else Route(ROUTE_CHAT, via="test")

    async def fake_transcribe(pcm, *, speaker_hint="", context_hint="",
                              speaker_id=None):
        i = idx[0]
        idx[0] += 1
        return tx[i] if i < len(tx) else ""

    async def fake_classify(text):
        env.routes.append(text)
        return canned(text) if callable(canned) else canned

    async def fake_synth(text, *a, **k):
        env.spoken.append(text)
        return b"\x00" * (FRAME_BYTES * 5)  # 100 ms of silence

    with mock.patch("voice.session.transcribe", new=fake_transcribe), \
            mock.patch("voice.session.classify_llm", new=fake_classify), \
            mock.patch("voice.tts.synthesize", new=fake_synth):
        yield


async def _run_ticker(session, seconds: float):
    """Run the real _endpoint_ticker for a while, then cancel it."""
    task = asyncio.create_task(session._endpoint_ticker())
    try:
        await asyncio.sleep(seconds)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _cancel_tasks(tasks):
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t


# --------------------------------------------------------------------- #
#  tests
# --------------------------------------------------------------------- #

async def test_noise_rejection():
    """Silence / low-energy noise never emits an utterance; a <300 ms blip
    is dropped by MIN_SPEECH_S; whisper noise-fragments are filtered —
    nothing reaches the agent."""
    # A. pure assembler: silence + quiet noise -> never enters speech
    asm = UtteranceAssembler(555)
    t = 1000.0
    for f in _frames(_silence_pcm(0.5)):
        asm.feed(_pkt(f, uid=555), t)
        t += 0.02
    for f in _frames(_noise_pcm(0.4)):
        asm.feed(_pkt(f, uid=555), t)
        t += 0.02
    assert asm.tick(t) is None
    assert asm.speaking is False

    # B. 200 ms voiced blip -> starts speech but is discarded on close
    for f in _frames(_sine_pcm(0.20)):
        asm.feed(_pkt(f, uid=555), t)
        t += 0.02
    assert asm.speaking is True            # onset happened
    for f in _frames(_silence_pcm(1.0)):
        asm.feed(_pkt(f, uid=555), t)
        t += 0.02
    assert asm.tick(t) is None             # dropped: voiced < MIN_SPEECH_S
    assert asm.speaking is False

    # C. same input through the session's real endpoint ticker -> queue empty
    env = build_env(replies=["hey"])
    with patched_io(env, transcripts=["eudora hi"]):
        session = env.session
        session._running = True
        a2 = session._assemblers.setdefault(555, UtteranceAssembler(555))
        ticker = asyncio.create_task(session._endpoint_ticker())
        try:
            for f in _frames(_silence_pcm(0.4)):
                a2.feed(_pkt(f, uid=555))
            for f in _frames(_noise_pcm(0.3, seed=1)):
                a2.feed(_pkt(f, uid=555))
            for f in _frames(_sine_pcm(0.20)):     # the blip
                a2.feed(_pkt(f, uid=555))
            await asyncio.sleep(0.9)               # > GAP_S: blip would close here
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
        assert session._utterance_q.empty()
        assert env.agent.calls == []
        assert env.spoken == []

    # D. whisper noise-fragments ("uh", "hmm") die at the _NOISE_LINE filter
    env2 = build_env(replies=["yo"])
    with patched_io(env2, transcripts=["uh", "hmm"]):
        await env2.session._handle_batch([_utt(ALICE_ID), _utt(BOB_ID)])
        assert env2.agent.calls == []
        assert env2.spoken == []
    print("    noise+blip+fragments produced zero utterances, agent untouched")


async def test_backchannel_skip():
    """'yeah' with no recent bot speech -> _should_skip -> agent not called."""
    env = build_env(replies=["sure thing"])
    s = env.session
    line = "[alice]: yeah"
    assert s._is_directed(line) is False
    assert s._should_skip(line) is True
    with patched_io(env, transcripts=["yeah"]):
        await s._handle_batch([_utt(ALICE_ID)])
    assert env.agent.calls == []
    assert env.spoken == [] and env.vc.play_calls == 0
    # ambient transcript still recorded it (mood heard it too)
    assert any("yeah" in l for _, l in s._ambient)
    print("    backchannel filtered before agent")


async def test_directed_response():
    """Transcript naming the bot -> agent runs -> reply is spoken."""
    env = build_env(replies=["honestly idk, probably tomorrow"])
    s = env.session
    assert s._is_directed("[alice]: eudora what do you think") is True
    with patched_io(env, transcripts=["eudora what do you think"]):
        await s._handle_batch([_utt(ALICE_ID)])
    assert len(env.agent.calls) == 1
    call = env.agent.calls[0]
    assert call.mode == "voice"
    assert "[alice]: eudora what do you think" in call.request
    assert "the-lab" in (call.context or "")     # room context was built
    assert call.no_tools is True                 # chat route -> no tool schemas
    assert env.spoken == ["honestly idk, probably tomorrow"]
    assert env.vc.play_calls >= 1
    print(f"    agent replied -> vc.play x{env.vc.play_calls}")


async def test_no_reply_suppression():
    """Agent status strings / empty / NOACTION must NEVER be spoken."""
    for reply in (
        "",
        "NOACTION",
        "noaction",
        "(no reply)",
        "no reply",
        "...",
        "(sorry, i had trouble processing that)",
        "(reached max tool rounds without a final answer)",
    ):
        env = build_env(replies=[reply])
        with patched_io(env, transcripts=["eudora you awake"]):
            await env.session._handle_batch([_utt(ALICE_ID)])
        assert env.agent.calls, f"agent never ran for reply {reply!r}"
        assert env.spoken == [] and env.vc.play_calls == 0, (
            f"agent returned {reply!r} and the bot SPOKE it aloud: "
            f"{env.spoken!r}"
        )
    print("    all status-string replies stayed silent")


async def test_speaker_prefix_stripping():
    """'[Eudora]: hey what's up' must be spoken as 'hey what's up'."""
    env = build_env(replies=["[Eudora]: hey what's up"])
    with patched_io(env, transcripts=["eudora say hi"]):
        await env.session._handle_batch([_utt(ALICE_ID)])
    assert env.spoken, "nothing was spoken at all"
    assert env.spoken[-1] == "hey what's up", (
        f"speaker prefix leaked into TTS: {env.spoken[-1]!r}"
    )
    # the public say() path strips too
    env2 = build_env()
    with patched_io(env2):
        await env2.session.say("[Eudora]: testing one two")
    assert env2.spoken[-1] == "testing one two", env2.spoken
    print("    speaker prefix stripped before TTS")


async def test_echo_rejection():
    """A transcript matching our own recent speech is dropped."""
    env = build_env(replies=["yeah"])
    s = env.session
    s._recent_spoken.append((time.monotonic(), "oh yeah"))
    assert s._is_echo("oh yeah") is True
    assert s._is_echo("a totally different sentence about cars") is False
    with patched_io(env, transcripts=["oh yeah"]):
        await s._handle_batch([_utt(ALICE_ID)])
    assert env.agent.calls == []
    assert len(s._ambient) == 0                  # echo never hit the transcript
    assert env.spoken == []
    print("    own voice filtered out")


async def test_barge_in():
    """Sustained human speech while the bot talks -> playback stops."""
    env = build_env()
    s = env.session
    s._running = True

    # direct: stops playback + records who interrupted
    s._speaking = True
    s._barge_in(ALICE_ID)
    assert env.vc.stop_calls == 1
    assert s._interrupted_by == ALICE_ID

    # no-op when the bot isn't speaking
    s._speaking = False
    s._interrupted_by = None
    env.vc.stop_calls = 0
    s._barge_in(ALICE_ID)
    assert env.vc.stop_calls == 0
    assert s._interrupted_by is None

    # driven by the real endpoint ticker: sustained overlap PAUSES first
    # (LiveKit false-interruption), a real turn (>=0.9s voiced) interrupts
    s._speaking = True
    s._interrupted_by = None
    asm = s._assemblers.setdefault(ALICE_ID, UtteranceAssembler(ALICE_ID))
    for f in _frames(_sine_pcm(1.2)):
        asm.feed(_pkt(f, uid=ALICE_ID))
    assert asm.voiced_ms_in_window(BARGE_IN_WINDOW_S) >= BARGE_IN_VOICED_MS
    await _run_ticker(s, 0.25)
    assert env.vc.pause_calls == 1 and s._paused_speech
    # utterance emits after hangover — >=0.9s voiced = real interruption
    await _run_ticker(s, 1.2)
    assert env.vc.stop_calls >= 1
    assert s._interrupted_by == ALICE_ID

    # short overlap (<0.9s voiced) = backchannel/click -> pause then RESUME
    env2 = build_env()
    s2 = env2.session
    s2._running = True
    s2._speaking = True
    asm2 = s2._assemblers.setdefault(BOB_ID, UtteranceAssembler(BOB_ID))
    for f in _frames(_sine_pcm(0.6)):
        asm2.feed(_pkt(f, uid=BOB_ID))
    await _run_ticker(s2, 0.25)
    assert env2.vc.pause_calls == 1 and s2._paused_speech
    await _run_ticker(s2, 1.2)
    assert env2.vc.resume_calls == 1      # backchannel didn't kill playback
    assert env2.vc.stop_calls == 0
    print(f"    pause->interrupt ok; short overlap resumed x{env2.vc.resume_calls}")


async def test_multi_user_merge():
    """Utterances from two users in one batch -> both lines reach the agent."""
    env = build_env(replies=["yeah i'm here"])
    s = env.session
    with patched_io(env, transcripts=["what's up everyone",
                                      "eudora you there?"]):
        await s._handle_batch([_utt(ALICE_ID), _utt(BOB_ID)])
    assert len(env.agent.calls) == 1
    req = env.agent.calls[0].request
    assert "[alice]:" in req and "[bob]:" in req
    assert "what's up everyone" in req and "you there" in req
    assert env.spoken == ["yeah i'm here"]
    print(f"    merged transcript: {req!r}")


async def test_pref_capture():
    """'i really hate loud music' lands in session._prefs under the speaker."""
    env = build_env(replies=["fair enough"])
    s = env.session
    with patched_io(env, transcripts=["i really hate loud music"]):
        await s._handle_batch([_utt(ALICE_ID)])
    assert "alice" in s._prefs
    assert "hate" in s._prefs["alice"].lower()
    # prefs show up in the room context handed to the agent next turn
    ctx = s._room_context()
    assert "alice" in ctx and "hate" in ctx
    print(f"    pref stored: {s._prefs!r}")


async def test_action_routing():
    """An action utterance routes with tool categories and no_tools=False."""
    env = build_env(replies=["on it"])
    route = Route(ROUTE_ACTION, {"messaging"}, via="test")
    with patched_io(env, transcripts=["eudora send a message to ali"],
                    route=route):
        await env.session._handle_batch([_utt(ALICE_ID)])
    assert len(env.agent.calls) == 1
    call = env.agent.calls[0]
    assert call.categories == {"messaging"}
    assert call.no_tools is False
    assert env.routes == ["[alice]: eudora send a message to ali"]
    assert env.spoken == ["on it"]
    print("    action route -> agent(categories={'messaging'})")


def test_router_heuristic():
    """ai.router.classify: verbs+categories -> action; plain talk -> chat;
    keyword-but-no-verb -> None (would go to the LLM arbiter)."""
    r = classify("send a gif to the channel")
    assert r is not None and r.kind == ROUTE_ACTION
    assert "media" in r.categories or "messaging" in r.categories
    r = classify("eudora bump the server")
    assert r is not None and r.kind == ROUTE_ACTION and "bump" in r.categories
    r = classify("what's everyone up to tonight")
    assert r is not None and r.kind == ROUTE_CHAT
    assert classify("") is not None
    assert classify("that channel over there") is None   # ambiguous -> arbiter
    print("    router heuristic ok")


def test_vad_end_to_end():
    """Real UtteranceAssembler: tone -> silence-gap -> Utterance with
    correct voiced_s and metadata."""
    asm = UtteranceAssembler(7)
    t = 5000.0
    for f in _frames(_silence_pcm(0.4)):
        asm.feed(_pkt(f, uid=7), t)
        t += 0.02
    assert asm.speaking is False and asm.tick(t) is None

    for f in _frames(_sine_pcm(0.6)):
        asm.feed(_pkt(f, uid=7), t)
        t += 0.02
    assert asm.speaking is True
    assert asm.tick(t) is None                   # speech still in progress

    for f in _frames(_silence_pcm(0.8)):         # hangover silence -> close
        asm.feed(_pkt(f, uid=7), t)
        t += 0.02
    utt = asm.tick(t)
    assert utt is not None
    assert utt.user_id == 7
    assert 0.55 <= utt.voiced_s <= 0.65, utt.voiced_s
    assert utt.ended_at == t
    assert utt.interrupted_bot is False
    assert pcm_duration(utt.pcm) >= utt.voiced_s
    # assembler reset cleanly — trailing silence emits nothing further
    t += 1.0
    assert asm.tick(t) is None and asm.speaking is False
    print(f"    utterance: voiced={utt.voiced_s:.2f}s pcm={pcm_duration(utt.pcm):.2f}s")


def test_clean_for_speech():
    """Markdown / emoji / discord tags / direction tags stripped correctly."""
    from voice.tts import clean_for_speech
    assert clean_for_speech("**bold** _it_ `x` ~s~") == "bold it x s"
    assert clean_for_speech("hello 🔥😂💀 world") == "hello world"
    assert clean_for_speech("<@123456> yo <#789>") == "yo"
    assert clean_for_speech("[link](https://example.com) here") == "link here"
    assert clean_for_speech("[laughs] that's funny") == "that's funny"
    assert clean_for_speech("(giggles) hi") == "hi"
    # keep_tags=True preserves vocal-direction tags for Orpheus-style TTS
    assert clean_for_speech("[laughs] that's funny", keep_tags=True) == \
        "[laughs] that's funny"
    assert clean_for_speech("(giggles) hi", keep_tags=True) == "(giggles) hi"
    assert clean_for_speech("a    lot\t of\nspace") == "a lot of space"
    print("    clean_for_speech cases ok")


async def test_full_pipeline_e2e():
    """End-to-end through the REAL tasks: fake packets -> _packet_pump ->
    _endpoint_ticker -> _conversation_loop -> agent -> vc.play."""
    env = build_env(replies=["yo i hear you"])
    s = env.session
    if AsyncQueueSink is not None:
        s._queue_sink = AsyncQueueSink(
            asyncio.Queue(maxsize=3000),
            media_types=["audio"], codecs=["pcm"], drop_oldest=True)
    else:
        s._queue_sink = _QueueSinkStub()
    s._running = True
    tasks = [
        asyncio.create_task(s._packet_pump()),
        asyncio.create_task(s._endpoint_ticker()),
        asyncio.create_task(s._conversation_loop()),
    ]
    try:
        with patched_io(env, transcripts=["hey eudora can you hear me"]):
            # stream ~0.6 s of "speech" as 20 ms packets
            for f in _frames(_sine_pcm(0.6)):
                s._queue_sink.queue.put_nowait(_pkt(f, uid=ALICE_ID))
                await asyncio.sleep(0.004)
            # wait for endpoint gap -> merge -> delay -> agent -> play
            # (smart-turn model cold-load can take ~30s on first call)
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline and not env.vc.play_calls:
                await asyncio.sleep(0.05)
            assert env.agent.calls, "pipeline produced no agent call"
            req = env.agent.calls[0].request
            assert "[alice]:" in req and "eudora" in req
            assert env.vc.play_calls >= 1
            assert env.spoken[-1] == "yo i hear you"
            print(f"    packets->speech ok, agent said: {env.spoken[-1]!r}")
    finally:
        await _cancel_tasks(tasks)


# --------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------- #

# Tests that document genuine bugs in voice/session.py — they assert the
# EXPECTED behaviour so they fail today and pass once the source is fixed.
KNOWN_BUGS: dict = {}


async def _main() -> int:
    tests = [
        test_noise_rejection,
        test_backchannel_skip,
        test_directed_response,
        test_no_reply_suppression,
        test_speaker_prefix_stripping,
        test_echo_rejection,
        test_barge_in,
        test_multi_user_merge,
        test_pref_capture,
        test_action_routing,
        test_router_heuristic,
        test_vad_end_to_end,
        test_clean_for_speech,
        test_full_pipeline_e2e,
    ]
    results: dict[str, str] = {}
    for fn in tests:
        name = fn.__name__
        print(f"--- {name}")
        try:
            out = fn()
            if asyncio.iscoroutine(out):
                await out
            if name in KNOWN_BUGS:
                print(f"[PASS] {name}  (documented bug appears FIXED)")
            else:
                print(f"[PASS] {name}")
            results[name] = "PASS"
        except AssertionError as e:
            if name in KNOWN_BUGS:
                print(f"[BUG ] {name}: expected-behaviour check failed ->")
                print(f"        {e}")
                print(f"        source bug: {KNOWN_BUGS[name]}")
                results[name] = "BUG"
            else:
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc(limit=3)
                results[name] = "FAIL"
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {name}: {e!r}")
            traceback.print_exc(limit=5)
            results[name] = "ERROR"

    print("\n================ summary ================")
    for n, st in results.items():
        tag = {"PASS": "PASS", "BUG": "BUG (documented source bug)"}.get(st, st)
        print(f"  {tag:<32} {n}")
    fails = [n for n, st in results.items() if st in ("FAIL", "ERROR")]
    bugs = [n for n, st in results.items() if st == "BUG"]
    print(f"\n{sum(1 for s in results.values() if s == 'PASS')} passed, "
          f"{len(bugs)} documented source bug(s), {len(fails)} unexpected failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
