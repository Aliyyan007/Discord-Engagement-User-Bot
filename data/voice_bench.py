"""Fake-VC latency + recognition bench.

Drives the REAL pipeline — packet pump -> VAD assembler -> smart-turn ->
Groq Whisper STT -> agent stream -> fish TTS -> fake vc.play. Only the
Discord socket is faked; every ms measured is real.

Speech source: synthesized once via fish (real human voice audio), cached
to data/bench/*.wav, then degraded (faded gain / noise / overlap) to test
recognition under bad input.

Run:  python -m data.voice_bench            # full matrix
      python -m data.voice_bench quick      # clean speech only
"""
from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.session import VoiceSession, FRAME_BYTES
from voice.pcm import chunk_frames, apply_gain
from voice import stt as stt_mod
from voice import tts as tts_mod

BENCH_DIR = Path(__file__).resolve().parent / "bench"
BENCH_DIR.mkdir(exist_ok=True)

FRAME = FRAME_BYTES          # 20ms 48k stereo s16


# --------------------------------------------------------------------- #
#  fakes
# --------------------------------------------------------------------- #
class BenchVC:
    def __init__(self):
        self.plays = []          # (t_monotonic, source)
        self._paused = False

    def is_playing(self):  return True
    def is_paused(self):   return self._paused
    def is_listening(self): return True
    def pause(self):       self._paused = True
    def resume(self):      self._paused = False
    def stop(self):        pass
    def listen(self, sink, after=None): pass
    def play(self, source, after=None):
        self.plays.append(time.monotonic())
        if after:
            # pretend instant playback
            after(None)


def _mk_session() -> tuple[VoiceSession, BenchVC]:
    from ai.agent import Agent
    from tools.context import ToolContext
    from discord.ext.native_voice import AsyncQueueSink

    vc = BenchVC()
    m = SimpleNamespace(display_name="tester", id=42, bot=False,
                        mention="<@42>", name="tester")
    channel = MagicMock()
    channel.id = 123
    channel.members = [m]
    channel.name = "bench-vc"
    guild = MagicMock()
    guild.members = [m]
    guild.get_member = lambda _id: m
    guild.me = SimpleNamespace(voice=SimpleNamespace(
        self_mute=False, self_deaf=False))
    channel.guild = guild
    vc.channel = channel

    bot = SimpleNamespace(user=SimpleNamespace(id=999, display_name="eudora"))
    bot.agent = Agent(ToolContext(bot=bot))
    ctx = SimpleNamespace(bot=bot)
    s = VoiceSession(ctx, channel, vc)
    s._queue_sink = AsyncQueueSink(
        asyncio.Queue(maxsize=3000),
        loop=asyncio.get_running_loop(),
        media_types=["audio"], codecs=["pcm"], drop_oldest=True)
    s._running = True
    return s, vc


def _pkt(pcm20ms: bytes, uid: int = 42):
    """Fake decoded MediaPacket (mirrors native_voice.MediaPacket)."""
    return SimpleNamespace(
        media_type="audio", codec="pcm", payload=pcm20ms,
        user_id=uid, ssrc=1000 + uid,
        audio_level=None, audio_voice_activity=None,
        marker=False, sequence=0, timestamp=0, received_at=None,
    )


def _real_speech_pcm(text: str, cache: str) -> bytes:
    """Synthesize real speech via fish (cached) → 48k stereo PCM bytes."""
    path = BENCH_DIR / cache
    if path.exists():
        return path.read_bytes()
    pcm = asyncio.run(tts_mod._tts_fish(text))
    path.write_bytes(pcm)
    tts_mod._http = None     # client bound to the now-closed asyncio.run loop
    return pcm


def _degrade(pcm: bytes, kind: str) -> bytes:
    """Audio degradation scenarios."""
    a = np.frombuffer(pcm, dtype=np.int16).copy()
    if kind == "faded":                     # quiet speaker (~-18 dB)
        a = (a.astype(np.float32) * 0.12).astype(np.int16)
    elif kind == "noisy":                   # +white noise, ~12 dB SNR
        noise = np.random.normal(0, np.std(a) * 0.35, len(a))
        a = np.clip(a.astype(np.float32) + noise, -32768, 32767).astype(np.int16)
    elif kind == "mingled":                 # two voices overlapped
        other = np.roll(a, len(a) // 3)
        a = np.clip(a.astype(np.float32) * 0.7 + other * 0.45,
                    -32768, 32767).astype(np.int16)
    elif kind == "rough":                   # clipped/distorted (bad mic)
        a = np.clip(a.astype(np.float32) * 2.2, -14000, 14000).astype(np.int16)
    return a.tobytes()


@dataclass
class Result:
    scenario: str
    transcript: str = ""
    reply: str = ""
    emit_s: float = 0.0        # last packet -> utterance emit
    stt_s: float = 0.0
    llm_s: float = 0.0         # stream TTFT (first chunk)
    tts_s: float = 0.0         # first segment synth
    total_s: float = 0.0       # utterance END -> first audio out
    failed: str = ""


async def _run_scenario(text: str, degrade: str) -> Result:
    """One speaker, one utterance, full real pipeline."""
    r = Result(scenario=f"{degrade or 'clean'}")
    s, vc = _mk_session()
    base = await asyncio.to_thread(_real_speech_pcm, text,
                                   f"{degrade or 'clean'}_{abs(hash(text))%999}.pcm")
    pcm = _degrade(base, degrade) if degrade else base

    # instrument the real stages
    t_last_pkt = time.monotonic()
    stamps = {"emit": 0.0, "stt0": 0.0, "stt_done": 0.0,
              "tts0": 0.0, "first_audio": 0.0}
    orig_tx = stt_mod.transcribe
    async def timed_tx(*a, **k):
        if not stamps["stt0"]:
            stamps["stt0"] = time.monotonic()
        out = await orig_tx(*a, **k)
        stamps["stt_done"] = time.monotonic()
        r.transcript = (out or "")[:60]
        return out
    stt_mod.transcribe = timed_tx
    # voice.session imported the symbol directly — patch there too
    import voice.session as sess_mod
    sess_mod.transcribe = timed_tx

    orig_syn = tts_mod.synthesize
    async def timed_syn(t, **k):
        if not stamps["tts0"]:
            stamps["tts0"] = time.monotonic()
        return await orig_syn(t, **k)
    sess_mod.tts.synthesize = timed_syn

    orig_put = s._utterance_q.put_nowait
    def timed_put(u):
        if not stamps["emit"]:
            stamps["emit"] = time.monotonic()
        return orig_put(u)
    s._utterance_q.put_nowait = timed_put

    try:
        pump = asyncio.create_task(s._packet_pump())
        tick = asyncio.create_task(s._endpoint_ticker())
        convo = asyncio.create_task(s._conversation_loop())
        t_first = {"t": None}
        orig_play = vc.play
        def timed_play(src, after=None):
            if t_first["t"] is None:
                t_first["t"] = time.monotonic()
            return orig_play(src, after=after)
        vc.play = timed_play
        try:
            for f in chunk_frames(pcm):
                s._queue_sink.queue.put_nowait(_pkt(f))
                t_last_pkt = time.monotonic()
                await asyncio.sleep(0.005)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline and t_first["t"] is None:
                await asyncio.sleep(0.05)
        finally:
            for t in (pump, tick, convo):
                t.cancel()
    finally:
        sess_mod.transcribe = orig_tx
        stt_mod.transcribe = orig_tx
        sess_mod.tts.synthesize = orig_syn

    if t_first["t"]:
        r.total_s = t_first["t"] - t_last_pkt
        stamps["first_audio"] = t_first["t"]
    if stamps["emit"]:
        r.emit_s = stamps["emit"] - t_last_pkt
    if stamps["stt_done"] and stamps["stt0"]:
        r.stt_s = stamps["stt_done"] - stamps["stt0"]
    if stamps["tts0"] and stamps["stt_done"]:
        r.llm_s = stamps["tts0"] - stamps["stt_done"]   # llm->first synth req
    if stamps["first_audio"] and stamps["tts0"]:
        r.tts_s = stamps["first_audio"] - stamps["tts0"]
    r.reply = s._recent_spoken[-1][1] if s._recent_spoken else "(nothing)"
    if not t_first["t"]:
        r.failed = "no audio emitted"
    return r


async def main():
    quick = "quick" in sys.argv
    texts = [
        "hey eudora, what's the latest episode you're watching right now?",
        "can you tell me something interesting about yourself?",
    ]
    scenarios = [(t, k) for t in texts
                 for k in (["clean"] if quick else
                           ["clean", "faded", "noisy", "mingled", "rough"])]

    print(f"{'scenario':<10} {'emit':>6} {'stt':>6} {'llm*':>6} {'tts':>6} "
          f"{'TOTAL':>7}  transcript / reply")
    print("-" * 100)
    for text, kind in scenarios:
        r = await _run_scenario(text, kind if kind != "clean" else "")
        if r.failed:
            print(f"{r.scenario:<10} {'—':>6} {'—':>6} {'—':>6} {'—':>6} "
                  f"{'FAIL':>7}  {r.failed} | heard {r.transcript!r}")
        else:
            print(f"{r.scenario:<10} {r.emit_s:5.2f}s {r.stt_s:5.2f}s "
                  f"{r.llm_s:5.2f}s {r.tts_s:5.2f}s {r.total_s:6.2f}s  "
                  f"{r.transcript[:32]!r} -> {r.reply[:40]!r}")
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
