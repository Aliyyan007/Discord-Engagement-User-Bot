"""Conversational-naturalness probe — REAL LLM, fake STT/TTS boundaries.

Each scenario drives ONE ``VoiceSession._handle_batch`` call with a fabricated
``Utterance`` whose transcript is injected by patching
``voice.session.transcribe`` (STT bypassed — the transcript text IS the input,
same trick as tests/test_voice_simulation.py's ``patched_io``). This tests the
conversation path (transcript -> gates -> router -> agent -> spoken text)
without burning Whisper or fish.audio quota:

- ``voice.tts.synthesize`` is faked: records the text it was asked to speak
  (== what would be said aloud) and returns 1 s of silent PCM. FakeVC.play
  fires ``after()`` instantly so playback "finishes" immediately.
- ``voice.session.classify_llm`` stays REAL (wrapped to record the route) —
  ambiguous lines hit the actual llama-3.1 arbiter like production.
- ``session.agent`` is a REAL ``ai.agent.Agent`` wired exactly like
  ``data/voice_bench.py::_mk_session`` (``Agent(ToolContext(bot=bot))``), so
  replies are genuine Groq output through the real voice system prompt.
- ``ai.user_memory`` writes are stubbed so the probe never touches
  ``data/user_memory.json``.

All 7 scenarios run SEQUENTIALLY on ONE session so ambient context builds like
a real conversation — that's what lets "no im saying im single" be judged on
contextual pickup, and what makes "im doing too" a true in-context garble
(the failure mode under test: nonsense like "how the doing to going" = FAIL;
a repair / graceful guess = PASS).

Run:  python -m data._conv_probe     (or: python data/_conv_probe.py)
"""
from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows console is cp1252 — replies may contain Unicode (’ — ‑ etc.)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# --- logging: console only — don't touch the rotating file sink ----------- #
from utils.logger import logger  # noqa: E402

logger.remove()
logger.add(
    sys.stderr, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
           "<level>{message}</level>",
)

from ai.agent import Agent  # noqa: E402
from ai import user_memory  # noqa: E402
from tools.context import ToolContext  # noqa: E402
import voice.session as sess_mod  # noqa: E402
import voice.tts as tts_mod  # noqa: E402
from tests.test_voice_simulation import (  # noqa: E402
    ALICE_ID,
    build_env,
    _utt,
)

SILENCE_PCM = b"\x00" * (3840 * 50)   # 1 s of 48 kHz stereo s16 per synth call

# (transcript, what a human would naturally do)
SCENARIOS = [
    ("can i ask you a question",
     "natural go-ahead"),
    ("are you single",
     "natural playful reply"),
    ("im doing too",          # garbled "i'm single too" — THE failure mode
     "repair ('wait what?') or graceful guess; nonsense/parroting = FAIL"),
    ("no im saying im single",
     "contextual pickup (she asked about being single)"),
    ("what do you think about pineapple pizza",
     "casual opinion"),
    ("eudora",
     "name-call -> 'yeah?'/'what's up' style ack"),
    ("tell me about your family",
     "persona-grounded answer (French mum, British dad, London)"),
]


def _real_agent(env, inflight: dict) -> list:
    """Swap the FakeAgent for the real Groq-backed Agent (voice_bench wiring)
    and record every prompt + reply. `inflight["n"]` tracks calls still in
    flight so _settle can wait out a slow second-beat before we snapshot."""
    real = Agent(ToolContext(bot=env.bot))
    calls: list[dict] = []
    orig_run = real.run
    orig_stream = real.run_stream

    async def rec_run(request, **kw):
        c = {"kind": "run", "request": request, "kw": kw, "reply": ""}
        calls.append(c)
        inflight["n"] += 1
        try:
            c["reply"] = await orig_run(request, **kw)
            return c["reply"]
        finally:
            inflight["n"] -= 1

    def rec_stream(request, **kw):
        c = {"kind": "stream", "request": request, "kw": kw, "reply": ""}
        calls.append(c)
        return _track_stream(c, orig_stream(request, **kw))

    async def _track_stream(c, agen):
        inflight["n"] += 1
        try:
            async for d in agen:
                c["reply"] += d
                yield d
        finally:
            inflight["n"] -= 1

    real.run = rec_run
    real.run_stream = rec_stream
    env.bot.agent = real
    env.session.agent = real
    return calls


@contextlib.contextmanager
def real_io(env, transcripts: list[str]):
    """Patch ONLY the network boundaries; keep the router + agent real.

    Yields a list that accumulates the real Route objects classify_llm
    produced (heuristic or arbiter — whichever the real code chose).
    """
    tx = list(transcripts)
    idx = [0]
    routes: list = []

    async def fake_transcribe(pcm, *, speaker_hint="", context_hint="",
                              speaker_id=None):
        i = idx[0]
        idx[0] += 1
        return tx[i] if i < len(tx) else ""

    real_classify = sess_mod.classify_llm

    async def rec_classify(text):
        r = await real_classify(text)
        routes.append(r)
        return r

    async def fake_synth(text, *a, **k):
        env.spoken.append(text)
        return SILENCE_PCM

    with mock.patch.object(sess_mod, "transcribe", new=fake_transcribe), \
            mock.patch.object(sess_mod, "classify_llm", new=rec_classify), \
            mock.patch.object(tts_mod, "synthesize", new=fake_synth), \
            mock.patch.object(user_memory, "add_facts",
                              new=lambda *a, **k: []), \
            mock.patch.object(user_memory, "set_real_name",
                              new=lambda *a, **k: None):
        yield routes


async def _settle(spoken: list, inflight: dict,
                  quiet_s: float = 3.0, cap_s: float = 30.0):
    """Wait for stragglers — a second-beat's agent.run can still be in flight
    (sleeping 0.6-1.6 s + LLM time) when the main reply has already played.
    Returns when no agent call is in flight AND `spoken` has been unchanged
    for quiet_s, or at cap_s."""
    t0 = time.monotonic()
    last_n = -1
    last_change = t0
    while time.monotonic() - t0 < cap_s:
        if len(spoken) != last_n or inflight["n"]:
            last_n = len(spoken)
            last_change = time.monotonic()
        elif time.monotonic() - last_change >= quiet_s:
            return
        await asyncio.sleep(0.25)


async def main() -> int:
    random.seed(11)                     # deterministic human-variability dice
    env = build_env()                   # FakeVC/FakeGuild/FakeChannel harness
    s = env.session
    s._running = True                   # lets fillers/second-beat behave real
    inflight = {"n": 0}
    agent_calls = _real_agent(env, inflight)

    transcripts = [t for t, _ in SCENARIOS]
    results = []

    print("=" * 78)
    print("CONV PROBE — real Groq agent, canned transcripts, silent TTS")
    print("=" * 78)

    with real_io(env, transcripts) as routes:
        for i, (text, expect) in enumerate(SCENARIOS, 1):
            sp0, ca0, rt0 = len(env.spoken), len(agent_calls), len(routes)
            amb0 = len(s._ambient)
            t0 = time.monotonic()
            err = ""
            try:
                await asyncio.wait_for(
                    s._handle_batch([_utt(ALICE_ID)]), timeout=90)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            await _settle(env.spoken, inflight)
            dt = time.monotonic() - t0

            new_spoken = env.spoken[sp0:]
            new_calls = agent_calls[ca0:]
            new_routes = routes[rt0:]
            ambient_line = (s._ambient[-1][1] if len(s._ambient) > amb0
                            else "(dropped before transcript)")

            results.append({
                "n": i, "heard": text, "expect": expect,
                "ambient": ambient_line,
                "route": (f"{new_routes[-1].kind}/{new_routes[-1].via}"
                          if new_routes else "—"),
                "calls": new_calls, "spoken": new_spoken,
                "dt": dt, "err": err,
            })

            print(f"\n--- scenario {i}: {text!r}  ({dt:.1f}s)")
            print(f"    expect : {expect}")
            print(f"    line   : {ambient_line}")
            print(f"    route  : {results[-1]['route']}")
            for c in new_calls:
                kind = "STREAM" if c["kind"] == "stream" else "RUN"
                # second-beat / memory-extraction side calls get tagged so
                # their replies aren't misattributed to the main turn
                tag = ""
                if "[you just said" in str(c["request"]):
                    tag = " (2nd-beat)"
                elif "Extract NEW facts" in str(c["request"]):
                    tag = " (mem-extract)"
                req = " ".join(str(c["request"]).split())
                print(f"    agent<- [{kind}{tag}] {req[:110]!r}")
                rep = " ".join(str(c.get("reply") or "").split())
                if rep:
                    print(f"    agent->  {rep[:140]!r}")
            if err:
                print(f"    ERROR  : {err}")
            if not new_spoken:
                print("    spoken : (silent)")
            for j, line in enumerate(new_spoken):
                tag = "spoken " if j == 0 else "      + "
                print(f"    {tag}: {line!r}")
            await asyncio.sleep(0.4)

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print(f"{'#':<3} {'transcript':<42} {'route':<14} reply (verbatim)")
    print("-" * 78)
    for r in results:
        reply = " || ".join(r["spoken"]) if r["spoken"] else "(silent)"
        print(f"{r['n']:<3} {r['heard'][:41]:<42} {r['route']:<14} {reply}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
