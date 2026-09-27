"""Scratch probe — measure fish.audio TTS latency paths.

Run from project root:  python data/_fish_probe2.py

Measures:
  1. WS cold  — fresh socket, start+text+flush, first-audio + total time
     (short + long sentence)
  2. WS warm  — repeat long sentence on the SAME socket
  3. HTTP     — POST /v1/tts, time to full body
  4. WS @44100 — does a 44100 start request produce audio or hang?
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx          # noqa: E402
import ormsgpack      # noqa: E402
import websockets     # noqa: E402

from config.settings import settings  # noqa: E402

WS_URL = "wss://api.fish.audio/v1/tts/live"
HTTP_URL = "https://api.fish.audio/v1/tts"

SHORT = "hey what's up"
LONG = ("So yesterday I was thinking about moving to a new flat, "
        "the rent is insane but the view is worth it")

SILENCE_GAP = 0.45      # quiet gap between audio chunks = utterance done
FIRST_TIMEOUT = 15.0    # max wait for first audio chunk
CALL_CAP = 45.0         # hard cap per utterance


def _headers() -> dict:
    h = {"Authorization": f"Bearer {settings.fish_api_key}"}
    if settings.fish_model:
        h["model"] = settings.fish_model
    return h


def _start_req(sample_rate: int) -> bytes:
    return ormsgpack.packb({"event": "start", "request": {
        "text": "",
        "sample_rate": sample_rate,
        "latency": "balanced",
        "format": "pcm",
        "normalize": True,
        "prosody": {"speed": 1.0, "volume": 0},
        "reference_id": settings.fish_voice_id or None,
        "temperature": 0.7,
        "top_p": 0.75,
    }})


async def ws_connect(sample_rate: int = 24000):
    """Open a socket + send the session start frame."""
    t0 = time.monotonic()
    ws = await websockets.connect(
        WS_URL, additional_headers=_headers(),
        ping_interval=20, ping_timeout=20)
    connect_ms = (time.monotonic() - t0) * 1000
    await ws.send(_start_req(sample_rate))
    return ws, connect_ms


async def ws_call(ws, text: str, label: str, rate: int = 24000):
    """Send text+flush on an open socket; collect until silence/finish."""
    t_send = time.monotonic()
    await ws.send(ormsgpack.packb({"event": "text", "text": text}))
    await ws.send(ormsgpack.packb({"event": "flush"}))
    first_ms: float | None = None
    nbytes = 0
    nchunks = 0
    events: list[str] = []
    end_reason = "?"
    while True:
        timeout = SILENCE_GAP if nbytes else FIRST_TIMEOUT
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout)
        except asyncio.TimeoutError:
            end_reason = "silence-gap" if nbytes else "NO-AUDIO-15s"
            break
        except websockets.ConnectionClosed as e:
            end_reason = f"socket-closed({e.code})"
            break
        try:
            msg = ormsgpack.unpackb(raw)
        except Exception:
            msg = None
        if isinstance(msg, dict):
            ev = msg.get("event")
            if ev == "audio":
                chunk = msg.get("audio") or b""
                if chunk:
                    if first_ms is None:
                        first_ms = (time.monotonic() - t_send) * 1000
                    nbytes += len(chunk)
                    nchunks += 1
            elif ev == "finish":
                events.append(f"finish(reason={msg.get('reason')})")
                end_reason = "finish-event"
                break
            else:
                events.append(str(msg)[:200])
                if ev in ("close", "restart", "error"):
                    end_reason = f"{ev}-event"
                    break
        else:
            # non-msgpack frame — count as audio bytes, note it
            events.append(f"non-msgpack frame {len(raw)}B")
            if first_ms is None:
                first_ms = (time.monotonic() - t_send) * 1000
            nbytes += len(raw)
            nchunks += 1
        if time.monotonic() - t_send > CALL_CAP:
            end_reason = "cap-45s"
            break
    total_ms = (time.monotonic() - t_send) * 1000
    audio_s = nbytes / (rate * 2) if nbytes else 0.0
    fa = f"{first_ms:.0f}ms" if first_ms is not None else "NONE"
    print(f"  [{label:<14}] first_audio={fa:>8}  total={total_ms:7.0f}ms  "
          f"bytes={nbytes:>7} ({audio_s:.2f}s @{rate}Hz)  chunks={nchunks:>3}  "
          f"end={end_reason}")
    for e in events:
        print(f"      event: {e}")
    return first_ms, total_ms, nbytes, end_reason


async def main() -> None:
    if not settings.fish_api_key:
        print("FATAL: no fish_api_key in settings/.env")
        return
    print(f"model={settings.fish_model!r}  "
          f"voice_id={settings.fish_voice_id!r}")
    print(f"short={SHORT!r}")
    print(f"long ={LONG!r}\n")

    # ------------------------------------------------------------------ #
    print("== 1. WS COLD (fresh socket per call) ==")
    try:
        ws, c_ms = await ws_connect(24000)
        t_conn = time.monotonic()
        r = await ws_call(ws, SHORT, "cold-short")
        print(f"  connect+start={c_ms:.0f}ms  "
              f"first-from-connect={c_ms + (r[0] or 0):.0f}ms")
        await ws.close()
    except Exception as e:
        print(f"  cold-short FAILED: {e!r}")

    try:
        ws, c_ms = await ws_connect(24000)
        r = await ws_call(ws, LONG, "cold-long")
        print(f"  connect+start={c_ms:.0f}ms  "
              f"first-from-connect={c_ms + (r[0] or 0):.0f}ms")
    except Exception as e:
        print(f"  cold-long FAILED: {e!r}")
        ws = None

    # ------------------------------------------------------------------ #
    print("\n== 2. WS WARM (reuse socket from cold-long) ==")
    if ws is not None:
        try:
            await ws_call(ws, LONG, "warm-1")
            await ws_call(ws, LONG, "warm-2")
        except Exception as e:
            print(f"  warm FAILED: {e!r}")
        try:
            await ws.close()
        except Exception:
            pass
    else:
        print("  skipped — no live socket")

    # ------------------------------------------------------------------ #
    print("\n== 3. HTTP POST /v1/tts (time to full body) ==")
    body = {
        "format": "wav",
        "sample_rate": 44100,
        "latency": "balanced",
        "normalize": True,
        "prosody": {"speed": 1.0, "volume": 0},
        "temperature": 0.7,
        "top_p": 0.75,
        "reference_id": settings.fish_voice_id or None,
    }
    async with httpx.AsyncClient(timeout=40) as client:
        for label, text in (("http-short", SHORT), ("http-long", LONG)):
            try:
                t0 = time.monotonic()
                resp = await client.post(
                    HTTP_URL, headers=_headers(), json={**body, "text": text})
                ms = (time.monotonic() - t0) * 1000
                print(f"  [{label:<14}] status={resp.status_code}  "
                      f"total={ms:7.0f}ms  bytes={len(resp.content)}")
                if resp.status_code != 200:
                    print(f"      body: {resp.text[:200]}")
            except Exception as e:
                print(f"  [{label:<14}] FAILED: {e!r}")

    # ------------------------------------------------------------------ #
    print("\n== 4. WS with sample_rate=44100 ==")
    try:
        ws, c_ms = await ws_connect(44100)
        r = await ws_call(ws, SHORT, "ws-44100", rate=44100)
        if r[3] == "NO-AUDIO-15s" or not r[2]:
            print("  -> 44100 hangs (no audio within 15s)")
        else:
            print("  -> 44100 PRODUCES AUDIO")
        try:
            await ws.close()
        except Exception:
            pass
    except Exception as e:
        print(f"  ws-44100 FAILED: {e!r}")

    print("\ndone.")


if __name__ == "__main__":
    asyncio.run(main())
