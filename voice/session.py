"""Live voice-chat session — the persona engine.

Pipeline (all asyncio):
    UDP RTP ──▶ PCMDecodeSink ──▶ AsyncQueueSink ──▶ _packet_pump
        ──▶ UtteranceAssembler[user]  (energy + packet-gap VAD)
        ──▶ utterance_q ──▶ _conversation_loop
        ──▶ STT (Groq whisper) ──▶ Agent (VOICE_MODE) ──▶ TTS ──▶ vc.play

Human behaviour layer:
    - natural response delays (humans answer ~200-700 ms after turn end)
    - barge-in: stops speaking when a human talks over it for >300 ms
    - echo rejection: drops transcripts matching its own recent speech
    - doesn't reply to everything — backchannels/room talk get NOACTION
    - asks to leave when ignored for a while; leaves when alone
    - short greeting on join (like a person joining a call)
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import Counter, deque
from typing import Optional

import discord

from ai.agent import Agent
from ai.router import classify_llm, ROUTE_ACTION
from ai import user_memory
from config.prompts import VOICE_MODE
from config.settings import settings
from utils.logger import logger

from . import smart_turn
from . import tts
from .mood import MoodEngine
from .pcm import FRAME_BYTES, apply_gain, chunk_frames, pcm_duration
from .stt import _GARBLE_RX, transcribe
from .vad import UtteranceAssembler

def _install_native_voice() -> bool:
    """Runtime self-heal: Render/PIP builds sometimes skip the package
    (blueprint changes to buildCommand don't retro-apply). Install it
    once at import so voice receive works on any host."""
    try:
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
             "discord-native-voice"],
            capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


try:
    import discord.ext.native_voice  # noqa: F401 — probe first
except ModuleNotFoundError:
    _install_native_voice()          # self-heal, then let the import retry

try:
    from discord.ext.native_voice import (
        AsyncQueueSink,
        AudioFrameSource,
        PCMDecodeSink,
    )
    from discord.ext.native_voice.media import MediaSink

    class _ResilientSink(MediaSink):
        """Drops per-packet decode errors instead of letting one malformed
        frame (e.g. a DAVE packet before the MLS handshake completes) kill the
        whole listener — native_voice stops listening on ANY sink exception."""

        def __init__(self, destination) -> None:
            super().__init__(destination)
            self.decode_errors = 0

        def wants_media(self, media_type: str, codec: str) -> bool:
            return self._child.wants_media(media_type, codec)

        def write(self, packet):
            try:
                return self._child.write(packet)
            except Exception as e:  # noqa: BLE001
                self.decode_errors += 1
                if self.decode_errors <= 5 or self.decode_errors % 200 == 0:
                    logger.warning(
                        f"VC packet dropped (decode #{self.decode_errors}): {e}"
                    )

        def cleanup(self):
            try:
                self._child.cleanup()
            except Exception:  # noqa: BLE001
                pass
            self._closed = True

    _NATIVE_VOICE = True
    _NATIVE_ERR = None
except Exception as _e:  # noqa: BLE001
    _NATIVE_VOICE = False
    _NATIVE_ERR = _e
    AsyncQueueSink = PCMDecodeSink = _ResilientSink = None

    class AudioFrameSource(discord.AudioSource):
        """Fallback PCM source for the stock voice client — same interface
        native_voice's AudioFrameSource exposes. Keeps SPEAKING working on
        hosts where native_voice can't import (e.g. missing libopus)."""

        def __init__(self, frames: list[bytes]) -> None:
            self._frames = frames
            self._i = 0

        def read(self) -> bytes:
            if self._i >= len(self._frames):
                return b""
            frame = self._frames[self._i]
            self._i += 1
            return frame

        def is_opus(self) -> bool:
            return False

        def cleanup(self) -> None:
            self._frames = []

# --- behaviour tuning --------------------------------------------------- #
MERGE_WINDOW_S = 0.28        # collect utterances finishing within this window
INCOMPLETE_HOLD_S = 1.6      # wait for same-speaker continuation on mid-thought
RESPONSE_DELAY_MIN = 0.05    # aggressive turn-taking (humans ~200ms)
RESPONSE_DELAY_MAX = 0.35
RESPONSE_DELAY_REACTION = 0.12
FILLER_AFTER_S = 0.5         # play a "hmm" filler if reply takes longer
BARGE_IN_VOICED_MS = 500     # sustained human speech during bot speech = interrupt
BARGE_IN_WINDOW_S = 0.65
# LiveKit-style false-interruption: pause on overlap, interrupt only when
# the speech becomes a real turn; clicks/backchannels auto-resume.
FALSE_INTERRUPT_S = 2.0      # resume playback after this long with no turn
FALSE_INTERRUPT_MIN_S = 0.9  # utterances shorter than this = backchannel
ECHO_WINDOW_S = 30.0         # how long we remember what we said
ECHO_SIMILARITY = 78         # fuzzy-match cutoff for rejecting our own voice
ASK_LEAVE_REPLY_S = 30.0     # wait for a reply after asking to leave
MAX_TRANSCRIPT_AGE_S = 90    # ambient transcript window
_BACKCHANNEL = re.compile(
    r"^(yeah|yea|yup|ok|okay|lol|lmao|haha|lmfao|true|fr|frfr|real|damn|"
    r"bet|nah|nahh|mhm+|mm+|hmm+|uh+huh|ight|aight|fax|word|same|crazy|"
    r"wild|for real|no way|w|l|bruh|bro|dude|what|huh|omg|nice|facts)[\s.!]*$",
    re.I,
)
_ACTION_FILLER_LINES = [
    "gimme a sec", "one sec", "on it, sec",
]
_FILLER_LINES = [
    "hmm", "uhh", "oh", "mhm", "let's see", "hmm yeah",
    "wait lemme think", "uhhh", "ohh ok",
]
_JOIN_GREETS = [
    "yo {name}", "hey {name}", "{name} what's up", "ayy {name}",
    "yo {name} pull up",
]
_LEAVE_REACTS = [
    "later {name}", "cya {name}", "peace {name}",
]
_SILENCE_BREAKERS = [
    "y'all quiet over here",
    "sooo what are we doing",
    "this call is dead silent huh",
    "anyone alive?",
    "y'all fell asleep or what",
]
# strings the agent produces that must NEVER be spoken out loud
_NO_SPEAK_EXACT = {
    "noaction", "no reply", "no-reply", "noreply", "(no reply)",
    "none", "nothing to say", "no response",
}
_NO_SPEAK_MARKERS = (
    "reached max tool rounds", "had trouble processing",
    "trouble processing that", "(sorry", "error:", "exception",
    "<|", "|>", "\ufffd",   # leaked special tokens / corrupted generations
)
_SPEAKER_PREFIX = re.compile(r"^\[[^\]]{1,40}\]:\s*")
# whisper noise-floor rejects — fragments that aren't real speech
_NOISE_LINE = re.compile(r"^[\W_]{1,3}$|^(uh|um|hmm|mm|ah|oh|eh|mhm|huh)\.?$", re.I)
# lightweight user-preference capture ("i hate/loud", "don't call me x")
_PREF_RE = re.compile(
    r"\b(i (?:really )?(?:like|love|hate|don'?t like|prefer)|"
    r"don'?t call me|call me \w+|i'?m (?:not )?into)\b",
    re.I,
)
# "i'm from X" / "i live in X" / "call me X" / "i'm N" — explicit memory writes
_CALL_ME_RE = re.compile(r"\bcall me (\w+)", re.I)
_IM_FROM_RE = re.compile(r"\bi'?m from ([a-zA-Z ]{2,25})", re.I)
_I_LIVE_RE = re.compile(r"\bi live in ([a-zA-Z ]{2,25})", re.I)
_IM_AGE_RE = re.compile(r"\bi'?m (\d{1,2})(?:\s*years?\s*old)?\b", re.I)
_REMEMBER_RE = re.compile(r"\bremember (?:that |this[:,]? )(.{4,120})", re.I)

# real-question detection (STT rarely emits "?" reliably — match phrasing too)
_QUESTION_RX = re.compile(
    r"\?\s*$|^(?:what|who|why|how|when|where|which|whose|wanna|do you|"
    r"are you|can you|could you|will you|would you|did you|is it|is that|"
    r"should we|have you|does it)\b",
    re.I,
)
_NON_QUESTION_RX = re.compile(
    r"^(what the (fuck|hell)|who cares|who asked|idgaf|guess what)\b", re.I)
# "anyone there" / "y'all quiet" — always answered, like text-mode loneliness
_LONELINESS_RX = re.compile(
    r"\b(anyone (there|here|awake|alive)|y'?all (quiet|asleep|dead)|"
    r"someone (talk|answer)|dead call|dead vc|hellooo+|i'?m bored|"
    r"is anyone (there|here))\b",
    re.I,
)
# cheap sentiment -> response-delay multiplier (excited=fast, sad/angry=slow)
_SENTI_SPEED = [
    (re.compile(r"\b(hurry|asap|quick|now|come on|urgent)\b|[!?]{2,}", re.I), 0.55),
    (re.compile(r"\b(sad|cry|upset|depressed|died|failed|broke up)\b", re.I), 1.35),
    (re.compile(r"\b(angry|pissed|mad|fucking|wtf|hate)\b", re.I), 1.25),
]
# pre-synth'd spoken acknowledgements — the voice analog of emoji reactions:
# zero-LLM presence markers so the bot doesn't sound dead when not replying
_ACK_POOLS = {
    "agree": ["fr", "true", "facts", "real", "fair"],
    "funny": ["lmao", "dead", "nah that's crazy"],
    "wow":   ["damn", "yo", "wait really?"],
    "mellow": ["hmm", "mhm", "fair enough"],
}
_ACK_COOLDOWN_S = 35.0        # seconds between spoken acks
_ACK_LAST = 5                 # dedup window
_FUNNY_RX = re.compile(r"\b(lol|lmao|haha|funny|dying|dead|joke)\b|[\U0001f600-\U0001f64f]", re.I)
# stopwords excluded from the overused-word counter
_STOP = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at",
    "it", "its", "is", "was", "are", "were", "be", "been", "i", "im", "i'm",
    "you", "your", "yours", "u", "ur", "me", "my", "we", "they", "he", "she",
    "this", "that", "these", "those", "what", "who", "how", "when", "where",
    "just", "like", "really", "very", "not", "no", "yes", "yeah", "ok",
    "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "should", "if", "then", "than", "for", "with", "about",
}


class VoiceSession:
    """One live voice-channel session (one per guild at a time)."""

    def __init__(self, ctx, channel: discord.VoiceChannel, vc) -> None:
        self.ctx = ctx
        self.channel = channel
        self.vc = vc
        self.bot = ctx.bot
        self.agent: Agent = ctx.bot.agent
        self.guild = channel.guild

        # sinks / queues
        self._queue_sink = None
        self._decode_sink = None
        self._root_sink = None
        self._last_relisten_at = 0.0
        self._assemblers: dict = {}          # user_id|ssrc -> UtteranceAssembler
        self._utterance_q: asyncio.Queue = asyncio.Queue()

        # state
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._speaking = False               # bot is currently playing audio
        self._busy = False                   # agent/tts pipeline in flight
        self._interrupted_by: Optional[int] = None
        # false-interruption (LiveKit pattern): overlap pauses playback
        # first; a real turn interrupts, a click/backchannel auto-resumes
        self._paused_speech = False
        self._pause_user = None
        self._pause_deadline = 0.0
        self._play_done: Optional[asyncio.Event] = None
        # serializes vc.play — second-beat say() vs streamed chunks were
        # racing each other ("Already playing media" collisions)
        self._play_lock = asyncio.Lock()
        self._epoch = 0

        # memory
        self._ambient: deque = deque(maxlen=40)      # [(ts, line)]
        self._recent_spoken: deque = deque(maxlen=20)  # [(ts, text)]
        self._fillers: dict = {}                     # phrase -> pcm
        self._name_cache: dict = {}

        # behaviour timers
        self._joined_at = time.monotonic()
        self._last_directed_at = time.monotonic()
        self._last_spoke_at = 0.0
        self._last_room_speech_at = time.monotonic()
        self._asked_leave = False
        self._asked_leave_at = 0.0
        self._alone_since: Optional[float] = None
        self._silence_broken_at = 0.0
        self._known_members: set = set()

        # persona
        self.mood = MoodEngine()
        self._prefs: dict[str, str] = {}          # speaker -> remembered prefs
        self._unanswered: deque = deque(maxlen=6)  # [(ts, speaker, line)]
        self._ack_clips: dict = {}                 # phrase -> pcm
        self._ack_history: deque = deque(maxlen=_ACK_LAST)
        self._ack_last_at = 0.0
        # per-speaker engagement: key -> {"n": int, "lat": deque, "len": deque}
        self._engagement: dict = {}
        self._silence_stage = 0                    # dead-air escalation rung
        self._last_speaker_key: Optional[object] = None
        self._greeted_users: dict = {}             # user_id -> last greet ts
        self._last_vc_greet_at = 0.0
        self._idle_muted = False                   # self-mute theater state
        self._reply_ts: deque = deque(maxlen=12)   # recent reply timestamps
        self._awaited_name = ""                    # who we asked a question
        self._awaited_until = 0.0
        # per-speaker rolling audio tail (last ~8.5s of emitted utterances)
        # — feeds the smart-turn end-of-turn model
        self._turn_pcm: dict = {}                  # user_key -> deque[(ts, pcm)]

    # ================================================================== #
    #  lifecycle
    # ================================================================== #
    async def start(self) -> None:
        """Attach the listener + spawn worker tasks."""
        try:
            if _NATIVE_VOICE and hasattr(self.vc, "listen"):
                self._attach_listener()
                asyncio.create_task(self._dave_diag())
            else:
                logger.warning(
                    f"native_voice unavailable ({_NATIVE_ERR!r}) — session "
                    "runs DEAF (speak-only).")
        except Exception as e:  # noqa: BLE001
            logger.error(f"VC listen failed — speak-only mode: {e}")

        await self._spawn()

    def _attach_listener(self) -> None:
        """(Re)build the sink chain and attach it — sinks can't be re-used
        after teardown, so every (re)listen gets fresh objects."""
        # AsyncQueueSink requires an explicit asyncio.Queue (default is a
        # MISSING sentinel which the type check rejects). drop_oldest keeps a
        # stalled consumer from unbounded memory growth. Pass the running loop
        # explicitly — lazy resolution can bind the wrong loop and silently
        # evaporate every packet.
        self._queue_sink = AsyncQueueSink(
            asyncio.Queue(maxsize=3000),
            loop=asyncio.get_running_loop(),
            media_types=["audio"], codecs=["pcm"], drop_oldest=True,
        )
        decode = PCMDecodeSink(self._queue_sink, fec=True)
        self._root_sink = _ResilientSink(decode)

        def _listen_after(exc):  # a dead sink teardowns silently
            if exc is not None:
                logger.error(f"VC listener died: {exc!r}")

        self.vc.listen(self._root_sink, after=_listen_after)
        logger.info("Voice session: listening (native_voice PCM sink).")

    async def _spawn(self) -> None:
        """Seed member state, spawn worker tasks, stage/greeting setup."""
        self._known_members = {
            m.id for m in self.channel.members
            if m.id != self.bot.user.id and not m.bot
        }
        self._running = True
        self._tasks = [
            asyncio.create_task(self._packet_pump(), name="vc-pump"),
            asyncio.create_task(self._endpoint_ticker(), name="vc-endpoint"),
            asyncio.create_task(self._conversation_loop(), name="vc-convo"),
            asyncio.create_task(self._idle_watcher(), name="vc-idle"),
            asyncio.create_task(self._pregen_fillers(), name="vc-fillers"),
        ]
        # warm the smart-turn model (first load ~20s) + fish websocket
        # off the critical path — real turns should never pay cold costs
        if smart_turn.available():
            asyncio.create_task(smart_turn.turn_complete_prob(
                bytes(FRAME_BYTES * 2)))
        asyncio.create_task(tts.warm_fish(), name="vc-fish-warm")

        # Stage channels need an unsuppress request before audio can play.
        if isinstance(self.channel, discord.StageChannel):
            try:
                await self.channel.guild.me.edit(suppress=False)
                logger.info("Requested to speak on stage.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Stage unsuppress failed: {e}")

        # Human-like join: brief pause, then a short greeting most of the time.
        if settings.voice_greet_on_join:
            asyncio.create_task(self._join_greeting())

    async def stop(self, reason: str = "") -> None:
        """Tear down the session (does NOT disconnect the vc)."""
        if not self._running:
            return
        self._running = False
        logger.info(f"Voice session stopping ({reason}).")
        for t in self._tasks:
            t.cancel()
        try:
            if hasattr(self.vc, "is_listening") and self.vc.is_listening():
                self.vc.stop_listening()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.vc.is_playing():
                self.vc.stop()
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== #
    #  receive path
    # ================================================================== #
    async def _packet_pump(self) -> None:
        """Drain the decoded-audio queue into per-user assemblers."""
        rx = 0
        while self._running:
            try:
                if self._queue_sink is None:
                    await asyncio.sleep(1.0)
                    continue
                packet = await self._queue_sink.get()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"sink get error: {e}")
                await asyncio.sleep(0.05)
                continue

            if packet.media_type != "audio" or not packet.payload:
                continue
            # skip our own stream if it ever shows up
            if packet.user_id == self.bot.user.id:
                continue
            rx += 1
            if rx == 1 or rx % 250 == 0:
                logger.info(
                    f"VC rx: {rx} pcm packets "
                    f"({len(self._assemblers)} speaker stream(s))"
                )
            key = packet.user_id or packet.ssrc
            asm = self._assemblers.get(key)
            if asm is None:
                asm = self._assemblers[key] = UtteranceAssembler(key)
            asm.feed(packet)

    async def _endpoint_ticker(self) -> None:
        """50 ms scan: close finished utterances + watch for barge-in."""
        while self._running:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            stale = []
            for key, asm in self._assemblers.items():
                # barge-in: sustained human speech while bot is speaking —
                # pause first (LiveKit false-interruption), interrupt only
                # when it becomes a real turn
                if self._speaking and not self._paused_speech and \
                        asm.voiced_ms_in_window(BARGE_IN_WINDOW_S, now) >= BARGE_IN_VOICED_MS:
                    self._pause_playback(key)
                if self._paused_speech and key == self._pause_user and \
                        asm.voiced_ms_in_window(BARGE_IN_WINDOW_S, now):
                    self._pause_deadline = now + FALSE_INTERRUPT_S
                utt = asm.tick(now)
                if utt is not None:
                    logger.info(
                        f"VC utterance: {self._display_name(utt.user_id)} "
                        f"{utt.voiced_s:.2f}s voiced"
                    )
                    # rolling audio tail for smart-turn EOT scoring
                    tail = self._turn_pcm.setdefault(utt.user_id, deque())
                    tail.append((utt.ended_at, utt.pcm))
                    cutoff = utt.ended_at - 8.5
                    while tail and tail[0][0] < cutoff:
                        tail.popleft()
                    self._utterance_q.put_nowait(utt)
                    # real turn while paused → interrupt; short burst →
                    # backchannel, resume playback
                    if self._paused_speech and utt.user_id == self._pause_user:
                        if utt.voiced_s >= FALSE_INTERRUPT_MIN_S:
                            self._barge_in(key)
                        else:
                            self._resume_playback()
                if not asm.speaking and asm.last_packet_at and now - asm.last_packet_at > 30:
                    stale.append(key)
            if self._paused_speech and now > self._pause_deadline:
                self._resume_playback()
            for k in stale:
                del self._assemblers[k]

    def _pause_playback(self, user_key) -> None:
        """Overlap detected — pause (don't cancel) playback. LiveKit's
        resume_false_interruption: a PTT click or "mhm" shouldn't kill the
        reply; it resumes automatically if no real turn follows."""
        self._paused_speech = True
        self._pause_user = user_key
        self._pause_deadline = time.monotonic() + FALSE_INTERRUPT_S
        try:
            self.vc.pause()          # no-op when nothing is playing
            logger.debug(
                f"Overlap from {self._display_name(user_key)} — paused "
                f"(resume in {FALSE_INTERRUPT_S}s if not a real turn)")
        except Exception:  # noqa: BLE001
            pass

    def _resume_playback(self) -> None:
        """No real turn came — the overlap was a click/backchannel. Resume."""
        who = self._display_name(self._pause_user) if self._pause_user else "?"
        self._paused_speech = False
        self._pause_user = None
        try:
            if self.vc.is_paused():
                self.vc.resume()
                logger.debug(f"False interruption from {who} — resumed")
        except Exception:  # noqa: BLE001
            pass

    def _barge_in(self, user_key) -> None:
        """A human talked over the bot — stop playback immediately."""
        if not self._speaking:
            return
        logger.info(f"Barge-in by {self._display_name(user_key)} — stopping playback.")
        self._paused_speech = False
        self._pause_user = None
        self._interrupted_by = user_key
        try:
            self.vc.stop()
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== #
    #  conversation path
    # ================================================================== #
    async def _conversation_loop(self) -> None:
        while self._running:
            try:
                first = await self._utterance_q.get()
            except asyncio.CancelledError:
                raise
            batch = [first]
            # merge utterances that finished close together (multi-speaker
            # turns) — but bail early if nobody else is actually speaking;
            # waiting the full window for silent speakers wastes ~280ms.
            while len(batch) < 8:
                if not any(a.speaking for a in self._assemblers.values()):
                    if self._utterance_q.empty():
                        break
                try:
                    batch.append(await asyncio.wait_for(
                        self._utterance_q.get(), MERGE_WINDOW_S))
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    raise
            try:
                await self._handle_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Voice turn failed: {e}")

    async def _handle_batch(self, batch: list) -> None:
        """Transcribe a batch of utterances, gate, and maybe respond."""
        # --- transcribe all utterances in parallel
        # prompt-echo fix: whisper was fed EVERY guild member (~40 names) —
        # on low-confidence decodes it injected them into transcripts
        # ("Aliyyan-Denis, Tete, Nayeem..."). VC occupants only, ~12 max.
        try:
            names = [m.display_name for m in self.vc.channel.members
                     if not m.bot and m.id != self.bot.user.id][:12]
        except Exception:  # noqa: BLE001
            names = []
        # the bot's own name too — Whisper kept decoding "Eudora" as "Aparo"
        names.insert(0, "Eudora")
        hint = ", ".join(names)
        # context continuity — last few transcript lines bias vocab decode
        ctx_hint = " / ".join(
            line.split("]:", 1)[-1].strip()[:80]
            for ts, line in list(self._ambient)[-4:]
        )
        # smart-turn runs CONCURRENTLY with STT on the last speaker's audio
        # tail — the EOT verdict lands by the time transcripts return, so it
        # costs ~0 ms of critical-path latency.
        last_utt = batch[-1]
        tail0 = b"".join(p for _, p in self._turn_pcm.get(last_utt.user_id, ()))
        prob_task = (
            asyncio.create_task(smart_turn.turn_complete_prob(tail0))
            if smart_turn.available() and tail0 else None
        )
        results = await asyncio.gather(
            *(transcribe(u.pcm, speaker_hint=hint, context_hint=ctx_hint,
                         speaker_id=u.user_id)
              for u in batch),
            return_exceptions=True,
        )

        lines: list[tuple[float, str, int]] = []   # (ended_at, line, user_key)
        for utt, res in zip(batch, results):
            name = self._display_name(utt.user_id)
            if isinstance(res, Exception) or not res:
                logger.info(f"STT empty/failed for {name} ({utt.voiced_s:.2f}s)")
                continue
            text = res.strip()
            # noise floor — drop fragments whisper hallucinated off sounds
            if _NOISE_LINE.match(text):
                logger.debug(f"Noise rejected: {text!r}")
                continue
            if self._is_echo(text):
                logger.debug(f"Echo rejected: {text!r}")
                continue
            # garble flag: weird mid-word caps ("RAcmy") mean Whisper heard
            # noise and guessed — flag it so the model asks for a repeat
            # instead of parroting gibberish back
            garbled = bool(_GARBLE_RX.search(text))
            line = (f"[{name}]: (unclear?) {text}" if garbled
                    else f"[{name}]: {text}")
            self._ambient.append((utt.ended_at, line))
            lines.append((utt.ended_at, line, utt.user_id))
            self._last_room_speech_at = utt.ended_at
            self._last_speaker_key = utt.user_id
            self._remember_speech(utt.user_id, name, text, utt)
            if self._awaited_name and name.lower() == self._awaited_name:
                self._awaited_name = ""     # they answered — ownership done

        if not lines:
            return

        # --- end-of-turn hold -------------------------------------------- #
        # smart-turn (trained Whisper-encoder classifier) judges the last
        # <=8s of this speaker's audio; regex heuristic is the fallback.
        # INCOMPLETE -> wait briefly for the same speaker's continuation
        # instead of replying into a mid-thought pause.
        for _ in range(2):                       # max 2 continuation merges
            last_key = lines[-1][2]
            bare = lines[-1][1].split("]:", 1)[-1].strip()
            if prob_task is not None and last_key == last_utt.user_id:
                try:
                    prob = await prob_task
                except Exception:  # noqa: BLE001
                    prob = 1.0
                logger.debug(
                    f"smart-turn p(complete)={prob:.2f} "
                    f"({self._display_name(last_key)})")
                done = prob >= 0.5
            elif smart_turn.available():
                tail = b"".join(
                    p for _, p in self._turn_pcm.get(last_key, ()))
                if tail:
                    prob = await smart_turn.turn_complete_prob(tail)
                    logger.debug(
                        f"smart-turn p(complete)={prob:.2f} "
                        f"({self._display_name(last_key)})")
                    done = prob >= 0.5
                else:
                    done = not self._looks_incomplete(bare)
            else:
                done = not self._looks_incomplete(bare)
            if prob_task is not None:
                prob_task.cancel()
                prob_task = None                 # verdict consumed
            if done:
                break
            try:
                more = await asyncio.wait_for(
                    self._utterance_q.get(), INCOMPLETE_HOLD_S)
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                raise
            batch.append(more)
            res = await transcribe(more.pcm, speaker_hint=hint,
                                   context_hint=ctx_hint,
                                   speaker_id=more.user_id)
            if isinstance(res, Exception) or not res:
                logger.info(
                    f"STT empty/failed for {self._display_name(more.user_id)} "
                    f"({more.voiced_s:.2f}s)")
                continue
            text = res.strip()
            if _NOISE_LINE.match(text):
                logger.debug(f"Noise rejected (continuation): {text!r}")
                continue
            if self._is_echo(text):
                logger.debug(f"Echo rejected (continuation): {text!r}")
                continue
            name = self._display_name(more.user_id)
            garbled = bool(_GARBLE_RX.search(text))
            logger.info(f"VC transcript (continuation): [{name}]: {text}")
            line = (f"[{name}]: (unclear?) {text}" if garbled
                    else f"[{name}]: {text}")
            self._ambient.append((more.ended_at, line))
            lines.append((more.ended_at, line, more.user_id))
            self._last_room_speech_at = more.ended_at
            self._last_speaker_key = more.user_id
            self._remember_speech(more.user_id, name, text, more)
            if more.user_id != lines[-1][2]:
                break  # different speaker took over — their turn, not ours

        transcript = "\n".join(line for _, line, _ in lines)
        last_text = lines[-1][1]
        last_key = lines[-1][2]
        logger.info(f"VC transcript: {transcript}")

        bare_last = last_text.split("]:", 1)[-1].strip()
        # "anyone there?" type utterances always get an answer — that's the
        # job of an engaged friend, same rule as the text bot's loneliness sig.
        directed = self._is_directed(last_text) or bool(
            _LONELINESS_RX.search(bare_last))
        if directed:
            self._last_directed_at = time.monotonic()
            self._asked_leave = False   # conversation resumed
            self._unanswered.clear()
            # occasional silence even when addressed — humans don't pounce
            # on every word; ~10% skip when we've been chatty lately
            _now_t = time.monotonic()
            n_recent = sum(1 for t in self._reply_ts if _now_t - t < 60)
            if n_recent >= 3 and random.random() < 0.10:
                logger.debug("Human variability: letting this one pass.")
                return
        elif self._should_skip(last_text):
            logger.debug("Skipped backchannel/noise (not directed).")
            # mood still hears it — ambient chatter warms the room
            self.mood.hear(bare_last, directed=False)
            # ...but humans acknowledge aloud sometimes — cheap spoken ack
            if self._should_ack(bare_last):
                await self._play_ack()
            return
        else:
            # track unanswered room questions for proactive pickup
            if self._is_question(bare_last):
                self._unanswered.append((lines[-1][0], last_key, last_text))
            elif self._should_ack(bare_last):
                # cheap spoken ack replaces a full reply this turn
                await self._play_ack()
                self.mood.hear(bare_last, directed=False)
                return

        # mood hears everything that made it to transcript
        for _, line, _ in lines:
            self.mood.hear(line.split("]:", 1)[-1], directed=directed)
        if self.mood.wants_to_leave:
            await self.say(random.choice([
                "aight i'm good, i'm out", "yeah nah i'm off, cya",
                "okay rude — later",
            ]))
            await self._leave("mood: annoyed")
            return

        # --- human-like response delay ---------------------------------- #
        # scaled by the speaker's vibe: excited/urgent -> snap, sad -> pause
        speed = 1.0
        for rx, mult in _SENTI_SPEED:
            if rx.search(bare_last):
                speed = mult
                break
        end_ts = lines[-1][0]
        delay_target = (
            random.uniform(0.05, RESPONSE_DELAY_REACTION)
            if _BACKCHANNEL.match(bare_last)
            else random.uniform(RESPONSE_DELAY_MIN, RESPONSE_DELAY_MAX)
        ) * speed
        elapsed = time.monotonic() - end_ts
        if delay_target > elapsed:
            await asyncio.sleep(delay_target - elapsed)

        # --- route: conversation vs action ------------------------------- #
        # Conversation calls carry NO tool schemas (~4x fewer tokens, no
        # hallucinated tool calls). Action utterances get a filtered subset.
        route = await classify_llm(transcript)
        is_action = route.kind == ROUTE_ACTION and directed

        # --- think ------------------------------------------------------- #
        self._busy = True
        # action turns get a "gimme a sec"-style filler; conversations get
        # thinking sounds — masks tool-call latency differently
        filler_pool = _ACTION_FILLER_LINES if is_action else _FILLER_LINES
        filler_task = (
            asyncio.create_task(
                self._play_filler_after(FILLER_AFTER_S, filler_pool))
            if directed else None
        )
        interrupted_note = ""
        if self._interrupted_by is not None:
            who = self._display_name(self._interrupted_by)
            partial = self._recent_spoken[-1][1] if self._recent_spoken else ""
            interrupted_note = (
                f'\n[{who} cut you off mid-sentence while you were saying '
                f'"{partial[:80]}"]'
            )
            self._interrupted_by = None
        try:
            if not is_action:
                # STREAMED path: LLM tokens -> sentence chunks -> synth each
                # -> play as ready. First-audio lands ~1.5-2s sooner than the
                # serial run()+say() because generation/synthesis/playback
                # overlap (Pipecat/LiveKit pattern).
                reply = await self._speak_streamed(
                    transcript + interrupted_note, filler_task)
            else:
                reply = await self.agent.run(
                    transcript + interrupted_note,
                    mode=VOICE_MODE,
                    context=self._room_context(),
                    speaker_name=None,
                    categories=route.categories,
                    no_tools=False,
                )
                reply = _SPEAKER_PREFIX.sub("", (reply or "").strip()).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Voice agent error: {e}")
            reply = ""
        finally:
            self._busy = False
            if filler_task:
                filler_task.cancel()

        if is_action:
            if not self._sayable(reply):
                return
            if self._is_repeat(reply):
                reply = ""
            if not reply:
                return
            self._reply_ts.append(time.monotonic())
            await self.say(reply)

        # --- second beat: humans sometimes add a follow-up line ---------- #
        # higher odds when the room is quiet — that's when follow-ups keep
        # a conversation alive instead of letting it die
        quiet = time.monotonic() - self._last_room_speech_at > 8
        if (
            self.mood.label in ("happy", "hyped", "bored", "content")
            and random.random() < (0.45 if quiet else 0.30)
        ):
            asyncio.create_task(self._second_beat(transcript))

        # --- sampled memory extraction (cheap, background) --------------- #
        if random.random() < 0.06:
            asyncio.create_task(self._extract_memories(lines))

    # ------------------------------------------------------------------ #
    def _sayable(self, reply: str) -> bool:
        """Gate: is this string OK to speak out loud?"""
        if not reply:
            return False
        low = reply.lower().strip("() .")
        if low in _NO_SPEAK_EXACT or reply.upper() == "NOACTION":
            return False
        if any(m in low for m in _NO_SPEAK_MARKERS):
            return False
        if not re.search(r"[a-zA-Z0-9]", reply):   # "...", "?!", pure emoji
            return False
        return True

    # ------------------------------------------------------------------ #
    def _remember_speech(self, user_key, name: str, text: str, utt) -> None:
        """Per-speaker bookkeeping: prefs -> persistent memory + engagement."""
        # stated preferences -> in-session cache AND persistent store
        if _PREF_RE.search(text):
            prev = self._prefs.get(name, "")
            self._prefs[name] = f"{prev}; {text[:90]}" if prev else text[:90]
            if not isinstance(user_key, str):     # ssrc keys aren't user ids
                user_memory.add_facts(user_key, name, [text[:120]])
        # explicit memory writes
        if not isinstance(user_key, str):
            if m := _CALL_ME_RE.search(text):
                user_memory.set_real_name(user_key, name, m.group(1))
                self._name_cache[user_key] = m.group(1)
            facts = []
            if m := (_IM_FROM_RE.search(text) or _I_LIVE_RE.search(text)):
                facts.append(f"from {m.group(1).strip()}")
            if m := _IM_AGE_RE.search(text):
                facts.append(f"{m.group(1)} years old")
            if m := _REMEMBER_RE.search(text):
                facts.append(m.group(1).strip().rstrip(".,!"))
            if facts:
                user_memory.add_facts(user_key, name, facts)
        # engagement signal for the follow-up nudge
        eng = self._engagement.setdefault(
            user_key, {"n": 0, "lat": deque(maxlen=8), "len": deque(maxlen=8)})
        eng["n"] += 1
        eng["len"].append(utt.voiced_s)
        if self._last_spoke_at:
            eng["lat"].append(utt.ended_at - self._last_spoke_at)

    @staticmethod
    def _is_question(text: str) -> bool:
        t = text.strip()
        return bool(_QUESTION_RX.search(t)) and not _NON_QUESTION_RX.match(t)

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        """Cheap semantic endpointing: did this utterance end mid-thought?
        Terminal punctuation ends a turn; trailing conjunctions / no
        punctuation on 4+ words means they're probably still going."""
        t = text.strip().rstrip(",")
        if re.search(r"[.!?…]+$", t):
            return False
        words = t.split()
        if len(words) < 4:
            return False                      # short turn ("yeah", "nah fr")
        tail = words[-1].lower().strip(".,!?")
        if tail in {
            "and", "but", "so", "because", "bc", "if", "then", "or", "with",
            "like", "um", "uh", "the", "a", "to", "of", "in", "on", "my",
            "your", "their", "that", "was", "is", "i", "we", "they", "mean",
        }:
            return True
        return True                            # 4+ words, no punctuation

    def _track_awaited(self, text: str) -> None:
        """If our spoken line asks a member a question, their next utterance
        owns the reply window (~15s) regardless of name-matching."""
        if "?" not in text:
            return
        try:
            members = self.vc.channel.members
        except Exception:  # noqa: BLE001
            return
        low = text.lower()
        for m in members:
            if m.id == self.bot.user.id or m.bot:
                continue
            if m.display_name.lower() in low:
                self._awaited_name = m.display_name.lower()
                self._awaited_until = time.monotonic() + 15
                return

    def _engagement_level(self, user_key) -> str:
        """high/medium/low/minimal from response latency + utterance length."""
        eng = self._engagement.get(user_key)
        if not eng or not eng["n"]:
            return "minimal"
        lat = [x for x in eng["lat"] if 0 < x < 120]
        avg_lat = sum(lat) / len(lat) if lat else 30
        avg_len = sum(eng["len"]) / len(eng["len"])
        score = (
            (3 if avg_lat < 5 else 2 if avg_lat < 12 else 1 if avg_lat < 30 else 0)
            + (3 if avg_len > 3.5 else 2 if avg_len > 1.2 else 1)
            + (2 if eng["n"] > 8 else 1)
        )
        return "high" if score >= 6 else "medium" if score >= 3 else "low" if score >= 1 else "minimal"

    def _is_repeat(self, reply: str) -> bool:
        """Jaccard + shared-opener duplicate check vs our last spoken lines."""
        if not self._recent_spoken:
            return False
        words = {w for w in re.findall(r"[a-z']+", reply.lower())
                 if w not in _STOP and len(w) > 1}
        for _, prev in list(self._recent_spoken)[-4:]:
            pl = prev.lower()
            if reply.lower() == pl:
                return True
            if len(reply) > 4 and reply.lower()[:8] == pl[:8]:
                return True
            pw = {w for w in re.findall(r"[a-z']+", pl)
                  if w not in _STOP and len(w) > 1}
            if words and pw and len(words & pw) / len(words | pw) > 0.65:
                return True
        return False

    def _should_ack(self, text: str) -> bool:
        """Probability ladder for a cheap spoken acknowledgement."""
        if not self._ack_clips:
            return False
        if self._speaking or self._busy:
            return False
        if time.monotonic() - self._ack_last_at < _ACK_COOLDOWN_S:
            return False
        chance = 0.15
        if self._last_spoke_at and time.monotonic() - self._last_spoke_at < 10:
            chance += 0.10
        if _FUNNY_RX.search(text):
            chance += 0.08
        if self.mood.label in ("happy", "hyped"):
            chance += 0.05
        elif self.mood.label in ("annoyed", "tired"):
            chance -= 0.08
        return random.random() < min(chance, 0.40)

    def _pick_ack(self, text: str) -> Optional[str]:
        """Category-matched ack phrase, dedup'd against the last few."""
        pools = []
        if _FUNNY_RX.search(text):
            pools.append("funny")
        if self._is_question(text):
            pools.append("wow")
        pools += ["agree", "mellow"]
        for pool in pools:
            cands = [p for p in _ACK_POOLS[pool] if p not in self._ack_history]
            cands = [p for p in cands if p in self._ack_clips]
            if cands:
                return random.choice(cands)
        return None

    async def _play_ack(self) -> None:
        """Play a pre-synthesized acknowledgement clip (zero-latency)."""
        phrase = self._pick_ack(self._ambient[-1][1] if self._ambient else "")
        if not phrase:
            return
        pcm = self._ack_clips.get(phrase)
        if pcm:
            self._ack_last_at = time.monotonic()
            self._ack_history.append(phrase)
            await self._play_pcm(pcm)

    async def _second_beat(self, transcript: str) -> None:
        """After the main reply, sometimes add a natural follow-up beat —
        the voice version of the reference bot's burst_reply field."""
        try:
            await asyncio.sleep(random.uniform(0.6, 1.6))
            if not self._running or self._speaking or self._busy:
                return
            just_said = (self._recent_spoken[-1][1]
                         if self._recent_spoken else "")
            reply = await self.agent.run(
                transcript +
                f'\n[you just said "{just_said[:90]}" — now add a DIFFERENT '
                f'follow-up: a question back at them or a related take. '
                f'DO NOT restate or rephrase what you already said — a new '
                f'beat only, or NOACTION if the moment\'s done]',
                mode=VOICE_MODE,
                context=self._room_context(),
                no_tools=True,
            )
            reply = _SPEAKER_PREFIX.sub("", (reply or "").strip()).strip()
            if self._sayable(reply) and not self._is_repeat(reply):
                self._reply_ts.append(time.monotonic())
                await self.say(reply)
        except Exception:  # noqa: BLE001
            pass

    async def _extract_memories(self, lines) -> None:
        """Sampled LLM memory pass — extract per-speaker facts from the
        transcript, store persistently (reference bot's 2% distillation,
        raised to 6% since voice turns are rarer than text messages)."""
        try:
            transcript = "\n".join(line for _, line, _ in lines)
            out = await self.agent.run(
                f"{transcript}\n\n[Extract NEW facts about each speaker as a "
                f"JSON object {{\"speaker_name\": [\"fact\", ...]}} — "
                f"preferences, names, places, facts they stated. [] or {{}} "
                f"if nothing new. No commentary.]",
                mode=VOICE_MODE,
                no_tools=True,
            )
            m = re.search(r"\{.*\}", out or "", re.S)
            if not m:
                return
            data = json.loads(m.group(0))
            # map speaker display names back to user keys
            speakers = {self._display_name(k): k for _, _, k in lines}
            for name, facts in data.items():
                key = speakers.get(name)
                if key and isinstance(facts, list):
                    user_memory.add_facts(key, name, [str(f) for f in facts])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"memory extraction failed: {e}")

    # ------------------------------------------------------------------ #
    async def _speak_streamed(self, prompt: str, _filler_task=None) -> str:
        """Streamed reply: LLM token stream -> sentence-boundary chunks ->
        each chunk synthesizes CONCURRENTLY -> playback in order.

        Turns the serial (generate-all, then synth-all, then play-all) path
        into a pipeline — the single biggest first-audio latency win:
        first sentence starts playing while later ones are still generating.
        """
        t_start = time.monotonic()
        buf = ""
        full = ""
        seg_tasks: list[tuple[str, asyncio.Task]] = []

        async def _synth(seg: str):
            try:
                return await tts.synthesize(seg, mood=self.mood.label)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"chunk synth failed: {e}")
                return None

        def _emit(text: str) -> None:
            clean = _SPEAKER_PREFIX.sub("", text).strip()
            if self._sayable(clean) and not self._is_repeat(clean):
                seg_tasks.append((clean, asyncio.create_task(_synth(clean))))

        def _flush(force: bool = False) -> None:
            nonlocal buf
            # cut at sentence end / newline / long-buffer comma fallback
            m = re.search(r"[.!?…]+\s|\n", buf)
            if m:
                seg, buf = buf[:m.end()].strip(), buf[m.end():]
                if seg:
                    _emit(seg)
            elif force and buf.strip():
                seg, buf = buf.strip(), ""
                _emit(seg)
            elif len(buf) > 140:
                cut = buf.rfind(",", 0, 140)
                if cut < 40:
                    cut = buf.rfind(" ", 0, 140)
                if cut > 0:
                    seg, buf = buf[:cut].strip(), buf[cut:]
                    _emit(seg)

        async def _read_stream(extra: str = "") -> None:
            nonlocal buf, full
            try:
                async for delta in self.agent.run_stream(
                    prompt + extra, context=self._room_context()
                ):
                    buf += delta
                    full += delta
                    if "<|" in buf:      # leaked special token -> truncate
                        buf = buf.split("<|", 1)[0]
                        _flush(force=True)
                        break
                    _flush()
                else:
                    _flush(force=True)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"stream read failed: {e}")
                _flush(force=True)

        await _read_stream()
        if not seg_tasks and self._sayable(full.strip()):
            # the model produced speakable text but every segment died in
            # the per-segment gates (repeat-echo) — dead air on a real turn
            # is worse than a mild repeat; retry once with a nudge.
            # (empty/NOACTION replies are deliberate silence — not retried)
            buf, full = "", ""
            await _read_stream(
                "  [your last line didn't land — say something different, "
                "don't just repeat yourself]")

        # ---- ordered playback; cancel rest on barge-in ------------------- #
        spoken: list[str] = []
        first_logged = False
        for seg, task in seg_tasks:
            if self._interrupted_by is not None:
                for _, t in seg_tasks:
                    t.cancel()
                break
            pcm = await task
            if not pcm:
                continue
            if not first_logged:
                first_logged = True
                logger.info(
                    f"VC stream first-audio {time.monotonic() - t_start:.2f}s: "
                    f"{seg[:80]}"
                )
            else:
                logger.info(f"VC stream chunk: {seg[:80]}")
            await self._play_pcm(pcm)
            # commit to context only AFTER playout — LiveKit's
            # spoken-text rule: interrupted chunks never count as said
            self._recent_spoken.append((time.monotonic(), seg))
            spoken.append(seg)
            self._last_spoke_at = time.monotonic()
            self._track_awaited(seg)

        reply = " ".join(spoken) if spoken else full.strip()
        logger.info(f"Agent final reply (voice): {reply[:200]}")
        return reply

    # ------------------------------------------------------------------ #
    async def say(self, text: str) -> None:
        """Public API: synthesize + play a line in the VC (used by tools too)."""
        if not self.vc or not self.vc.is_connected():
            return
        text = _SPEAKER_PREFIX.sub("", (text or "").strip()).strip()
        if not self._sayable(text):
            return
        # self-mute theater: unmute like a human leaning back in
        if self._idle_muted:
            self._idle_muted = False
            try:
                await self.guild.change_voice_state(
                    channel=self.vc.channel, self_mute=False)
                await asyncio.sleep(random.uniform(0.15, 0.35))
            except Exception:  # noqa: BLE001
                pass
        t0 = time.monotonic()
        pcm = await tts.synthesize(text, mood=self.mood.label)
        if not pcm:
            return
        logger.info(
            f"VC say ({time.monotonic() - t0:.2f}s tts): {text[:120]}"
        )
        self._recent_spoken.append((time.monotonic(), text))
        await self._play_pcm(pcm)
        self._last_spoke_at = time.monotonic()
        self._track_awaited(text)

    async def _play_pcm(self, pcm: bytes) -> None:
        """Play raw 48 kHz stereo PCM through the voice client, interruptibly."""
        # natural loudness jitter so the voice print isn't robotic-identical
        pcm = apply_gain(pcm, random.uniform(0.92, 1.06))
        frames = chunk_frames(pcm)
        if not frames:
            return
        done = asyncio.Event()
        self._play_done = done
        self._epoch += 1
        self._speaking = True
        loop = asyncio.get_running_loop()

        def _after(err):
            if err:
                logger.debug(f"play finished with error: {err}")
            loop.call_soon_threadsafe(done.set)

        async with self._play_lock:          # one playback at a time
            try:
                self.vc.play(AudioFrameSource(frames), after=_after)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"vc.play failed: {e}")
                self._speaking = False
                return
            try:
                await asyncio.wait_for(done.wait(), pcm_duration(pcm) + 15)
            except asyncio.TimeoutError:
                try:
                    self.vc.stop()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._speaking = False
                self._play_done = None
                if self._paused_speech:
                    # playback ended while paused — unpause so the next
                    # play doesn't start frozen
                    self._resume_playback()
                elif self.vc.is_paused():
                    try:
                        self.vc.resume()
                    except Exception:  # noqa: BLE001
                        pass

    # ------------------------------------------------------------------ #
    async def _play_filler_after(self, delay: float,
                                 pool: list | None = None) -> None:
        """If a reply is still cooking after `delay`, emit a thinking sound."""
        pool = pool or _FILLER_LINES
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._speaking or not self._running:
            return
        pcm = self._fillers.get(random.choice(pool))
        if pcm:
            await self._play_pcm(pcm)

    async def _pregen_fillers(self) -> None:
        """Pre-synthesize the filler + ack clips in the background."""
        phrases = (_FILLER_LINES + _ACTION_FILLER_LINES
                   + [p for pool in _ACK_POOLS.values() for p in pool])
        for phrase in phrases:
            if not self._running:
                return
            target = (self._ack_clips if any(
                          phrase in pool for pool in _ACK_POOLS.values())
                      else self._fillers)
            if phrase in target:
                continue
            try:
                pcm = await tts.synthesize(phrase + "...",
                                           mood=self.mood.label)
                if pcm:
                    target[phrase] = pcm
            except Exception as e:  # noqa: BLE001
                logger.debug(f"clip pregen failed for {phrase!r}: {e}")
            await asyncio.sleep(0.3)

    async def _dave_diag(self) -> None:
        """Dump receive-path diagnostics 2s after listen() — DAVE handshake
        state decides whether encrypted audio reaches us at all."""
        try:
            await asyncio.sleep(2)
            conn = getattr(self.vc, "_connection", None)
            me_voice = self.guild.me.voice if self.guild and self.guild.me else None
            dave = getattr(conn, "dave_session", None)
            logger.info(
                f"VC diag: listening={getattr(self.vc, 'is_listening', lambda: '?')()} "
                f"self_deaf={getattr(me_voice, 'self_deaf', '?')} "
                f"dave_proto={getattr(conn, 'dave_protocol_version', '?')} "
                f"dave_session={type(dave).__name__ if dave is not None else None} "
                f"dave_ready={getattr(dave, 'ready', '?') if dave is not None else None} "
                f"ssrc_map={len(getattr(conn, 'ssrc_user_ids', {}) or {})} "
                f"dropped={getattr(self._queue_sink, 'dropped', '?')} "
                f"decode_errors={getattr(self._root_sink, 'decode_errors', '?')}"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"dave diag failed: {e}")

    async def _join_greeting(self) -> None:
        """Wait a beat like a human, then greet the call."""
        try:
            await asyncio.sleep(random.uniform(0.7, 1.4))
            if not self._running or self._speaking:
                return
            if random.random() < 0.85:
                greet = random.choice([
                    "hey", "yo", "hey what's up", "yo what's good", "heyy",
                ])
                await self.say(greet)
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== #
    #  social behaviour — idle / alone / leave
    # ================================================================== #
    async def _idle_watcher(self) -> None:
        """Every 5 s: greet joiners, ack leavers, break dead silence,
        leave when alone, ask to leave when ignored."""
        while self._running:
            await asyncio.sleep(5)
            try:
                if not self.vc or not self.vc.is_connected():
                    break
                now = time.monotonic()

                # --- listener watchdog: a single decode error can silently
                # detach the sink (native_voice stops listening on exception)
                if (
                    self._root_sink is not None
                    and hasattr(self.vc, "is_listening")
                    and not self.vc.is_listening()
                    and now - self._last_relisten_at > 10
                ):
                    self._last_relisten_at = now
                    logger.warning("VC listener dropped — re-listening (fresh sink).")
                    try:
                        self._attach_listener()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"re-listen failed: {e}")
                humans = [
                    m for m in self.vc.channel.members
                    if m.id != self.bot.user.id and not m.bot
                ]
                human_ids = {m.id for m in humans}

                # --- member diff: greet joiners / acknowledge leavers ------ #
                joined = human_ids - self._known_members
                left = self._known_members - human_ids
                self._known_members = human_ids
                if joined and not self._speaking and not self._busy:
                    uid = next(iter(joined))
                    name = self._display_name(uid)
                    # per-user 1h cooldown + per-call 30s cooldown
                    if (now - self._greeted_users.get(uid, 0) > 3600
                            and now - self._last_vc_greet_at > 30
                            and random.random() < 0.9):
                        self._greeted_users[uid] = now
                        self._last_vc_greet_at = now
                        greet = random.choice(_JOIN_GREETS).format(name=name)
                        prof = user_memory.get_profile(uid)
                        sessions = user_memory.note_session_join(uid, name)
                        if prof.get("real_name") and random.random() < 0.5:
                            greet = random.choice([
                                f"yo {prof['real_name']}",
                                f"ay {prof['real_name']} what's good",
                            ])
                        elif sessions > 2 and random.random() < 0.5:
                            greet = random.choice([
                                f"wb {name}", f"ayy {name}'s back",
                            ])
                        asyncio.create_task(self._delayed_say(
                            greet, delay=random.uniform(0.4, 1.1)))
                elif left and not self._speaking and random.random() < 0.45:
                    name = self._display_name(next(iter(left)))
                    asyncio.create_task(self._delayed_say(
                        random.choice(_LEAVE_REACTS).format(name=name),
                        delay=random.uniform(0.4, 0.9),
                    ))

                # --- alone in channel ------------------------------------ #
                if not humans:
                    if self._alone_since is None:
                        self._alone_since = now
                    elif now - self._alone_since >= settings.voice_alone_seconds:
                        await self._leave("alone")
                        return
                else:
                    self._alone_since = None

                # mood drifts with the room
                self.mood.tick_idle(5)

                # --- pick up unanswered room questions -------------------- #
                if self._unanswered and humans and not self._busy and not self._speaking:
                    q = self._unanswered[0]
                    if now - q[0] > 5:
                        self._unanswered.popleft()
                        if random.random() < self.mood.proactive_chance:
                            asyncio.create_task(self._proactive_reply(
                                f"[{q[2].split(']:')[0].strip('[')} asked the "
                                f"room a question and nobody answered — "
                                f"answer it casually yourself.] {q[2]}"
                            ))

                # --- dead-air escalation ladder ---------------------------- #
                # rung 0->1: silence >90s -> address someone by name
                # rung 1->2: still silent +4min -> general icebreaker
                # rung 2->3: still silent -> asks-to-leave (existing path)
                silent_for = now - self._last_room_speech_at
                if humans and not self._busy and not self._speaking:
                    if (self._silence_stage == 0 and silent_for > 90
                            and now - self._silence_broken_at > 140
                            and random.random() < self.mood.proactive_chance):
                        self._silence_broken_at = now
                        self._silence_stage = 1
                        name = (self._display_name(self._last_speaker_key)
                                if self._last_speaker_key else
                                random.choice(humans).display_name)
                        asyncio.create_task(self._proactive_reply(
                            f"[Dead air. Direct-address {name} by name — "
                            f"check they're still there or pull them into a "
                            f"topic. One short line.]"
                        ))
                    elif (self._silence_stage == 1 and silent_for > 150
                            and now - self._silence_broken_at > 200
                            and random.random() < self.mood.proactive_chance):
                        self._silence_broken_at = now
                        self._silence_stage = 2
                        asyncio.create_task(self._proactive_reply(
                            "[The call is dead silent. Break the ice — ask "
                            "what people are up to, crack a joke, or start a "
                            "topic. One short line.]"
                        ))
                if silent_for < 30:
                    self._silence_stage = 0  # conversation alive — reset ladder

                # --- self-mute theater: idle humans mute themselves -------- #
                if humans and not self._speaking and not self._busy:
                    if silent_for > 150 and not self._idle_muted:
                        self._idle_muted = True
                        try:
                            await self.guild.change_voice_state(
                                channel=self.vc.channel, self_mute=True)
                        except Exception:  # noqa: BLE001
                            self._idle_muted = False

                # --- ignored for a long time ------------------------------- #
                idle_s = now - self._last_directed_at
                # bored/annoyed bots want out sooner
                idle_limit = settings.voice_idle_ask_minutes * 60
                if self.mood.label in ("bored", "tired", "annoyed"):
                    idle_limit *= 0.6
                if not self._asked_leave and idle_s > idle_limit and not self._busy:
                    self._asked_leave = True
                    self._asked_leave_at = now
                    await self.say(random.choice([
                        "hey, i might dip soon",
                        "y'all still need me or can i dip?",
                        "i'm kinda quiet over here, want me to head out?",
                    ]))
                elif self._asked_leave and now - self._asked_leave_at > ASK_LEAVE_REPLY_S:
                    # nobody objected
                    await self._leave("no reply after asking to leave")
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug(f"idle watcher error: {e}")
        # loop ended => vc gone
        await self._cleanup_gone()

    async def _delayed_say(self, text: str, delay: float) -> None:
        """Say something after a natural pause, unless we got busy."""
        try:
            await asyncio.sleep(delay)
            if self._running and not self._speaking and not self._busy:
                await self.say(text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    async def _proactive_reply(self, nudge: str) -> None:
        """Generate a spontaneous line through the conversation path
        (no tools) and say it — the bot's independent voice."""
        self._busy = True
        try:
            reply = await self.agent.run(
                nudge,
                mode=VOICE_MODE,
                context=self._room_context(),
                no_tools=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"proactive reply failed: {e}")
            return
        finally:
            self._busy = False
        reply = _SPEAKER_PREFIX.sub("", (reply or "").strip()).strip()
        if self._sayable(reply):
            await self.say(reply)

    async def _leave(self, why: str) -> None:
        """Say a quick bye if we recently spoke, then disconnect."""
        logger.info(f"Voice session leaving ({why}).")
        try:
            if self._last_spoke_at and time.monotonic() - self._last_spoke_at < 300:
                await self.say(random.choice(["aight i'm out", "ight see y'all", "cya"]))
        except Exception:  # noqa: BLE001
            pass
        try:
            await self.vc.disconnect()
        except Exception:  # noqa: BLE001
            pass
        await self._cleanup_gone()

    async def _cleanup_gone(self) -> None:
        await self.stop("vc gone")
        from .manager import drop_session
        drop_session(self.guild.id if self.guild else 0)

    # ================================================================== #
    #  helpers
    # ================================================================== #
    def _display_name(self, user_key) -> str:
        if isinstance(user_key, str) and user_key.startswith("ssrc:"):
            return "someone"
        cached = self._name_cache.get(user_key)
        if cached:
            return cached
        name = "someone"
        try:
            m = self.guild.get_member(int(user_key)) if self.guild else None
            if m:
                name = m.display_name
        except Exception:  # noqa: BLE001
            pass
        self._name_cache[user_key] = name
        return name

    def _is_directed(self, line: str) -> bool:
        """Heuristic: was this utterance aimed at the bot?"""
        # answer-ownership: we asked X a question → X's next line is for us
        speaker = line.split("]:", 1)[0].strip("[] ").lower()
        if (self._awaited_name and speaker == self._awaited_name
                and time.monotonic() < self._awaited_until):
            return True
        text = line.split("]:", 1)[-1].lower()
        raw_names = getattr(self.bot, "_bot_names", None) or ["eudora"]
        # match full names AND their word tokens ("eudora_02" -> "eudora")
        names = set(raw_names)
        for n in raw_names:
            names.update(t for t in re.split(r"[^a-z0-9]+", n) if len(t) >= 4)
        if any(n and n in text for n in names):
            return True
        # replying to something the bot said recently
        if self._last_spoke_at and time.monotonic() - self._last_spoke_at < 25:
            return True
        # questions to the room shortly after bot joined
        if "?" in text and time.monotonic() - self._last_spoke_at < 60:
            return True
        return False

    def _should_skip(self, line: str) -> bool:
        """Cheap pre-filter: skip pure backchannels when we haven't spoken lately."""
        text = line.split("]:", 1)[-1].strip()
        if _BACKCHANNEL.match(text):
            # let backchannels through if we spoke recently (they're directed at us)
            return not (self._last_spoke_at and time.monotonic() - self._last_spoke_at < 45)
        return False

    def _is_echo(self, text: str) -> bool:
        """Reject transcripts that match something WE recently said
        (mic bleed from a user's speakers picking up the bot's TTS)."""
        if not self._recent_spoken:
            return False
        from rapidfuzz import fuzz
        now = time.monotonic()
        for ts, spoken in self._recent_spoken:
            if now - ts > ECHO_WINDOW_S:
                continue
            if fuzz.partial_ratio(text.lower(), spoken.lower()) >= ECHO_SIMILARITY:
                return True
        return False

    def _room_context(self) -> str:
        """Room-state header + mood + prefs + ambient transcript."""
        humans = [
            m.display_name for m in self.vc.channel.members
            if m.id != self.bot.user.id and not m.bot
        ]
        parts = [
            f"In voice call '{self.vc.channel.name}' with: "
            f"{', '.join(humans) or 'nobody'}."
        ]
        # --- social state ------------------------------------------------ #
        now = time.monotonic()
        recent_speakers = {
            line.split("]:")[0] for ts, line in self._ambient
            if now - ts < 25
        }
        if len(recent_speakers) >= 2:
            parts.append(
                "(Multiple people are talking to each other right now — "
                "don't butt in unless addressed or it adds real value.)"
            )
        parts.append(f"Your mood: {self.mood.label}. {self.mood.prompt_line()}")

        # --- anti-repetition (the "oh yeah?" killer): show the model its  #
        # own recent lines + ban the openers/words it's been leaning on --- #
        recent_said = [t for ts, t in self._recent_spoken
                       if now - ts < 240][-6:]
        if recent_said:
            openers = {r.split()[0].lower() for r in recent_said if r.split()}
            counts = Counter(
                w for r in recent_said
                for w in re.findall(r"[a-z']+", r.lower())
                if w not in _STOP and len(w) > 2
            )
            overused = [w for w, c in counts.items() if c >= 3]
            block = "[your recent lines: " + " | ".join(
                f'"{r[:40]}"' for r in recent_said) + "]"
            if openers:
                block += (f"\n[openers already used: {', '.join(sorted(openers))}"
                          f" — do NOT start with these]")
            if overused:
                block += (f"\n[FORBIDDEN words (overused by you): "
                          f"{', '.join(overused)}]")
            parts.append(block)

        # --- what we know about the people talking ----------------------- #
        if self._prefs:
            prefs = "; ".join(f"{k}: {v}" for k, v in self._prefs.items())
            parts.append(f"What you remember about people: {prefs}")
        mem_lines = []
        for key in list(self._engagement.keys()):
            if isinstance(key, str):
                continue
            txt = user_memory.memory_line(key)
            if txt:
                mem_lines.append(txt)
        if mem_lines:
            parts.append("[you know these people — " +
                         " / ".join(mem_lines[:4]) + "]")

        # --- engagement nudge for the last speaker ------------------------ #
        if self._last_speaker_key is not None:
            lvl = self._engagement_level(self._last_speaker_key)
            if lvl == "high":
                parts.append(
                    "[they're engaged — warm tone, a follow-up question "
                    "fits naturally here]"
                )
            elif lvl == "minimal":
                parts.append(
                    "[they're barely engaging — don't over-invest, keep it "
                    "light]"
                )

        # --- unanswered room questions ------------------------------------ #
        pending = [q for ts, k, q in self._unanswered if now - ts < 90]
        if pending:
            parts.append(
                "[nobody answered these yet — you could:] " +
                " / ".join(q[-70:] for q in pending[-2:])
            )

        if self._ambient:
            recent = [line for ts, line in self._ambient
                      if now - ts < MAX_TRANSCRIPT_AGE_S]
            if recent:
                parts.append("Recent conversation:\n" + "\n".join(recent[-12:]))
        return "\n".join(parts)
