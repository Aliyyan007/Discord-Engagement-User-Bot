"""Text-to-speech — fish.audio only (most human voice available).

Returns raw 48 kHz stereo s16 PCM for Discord playback. The prosody
(speed/volume) is driven by the bot's mood so the voice actually sounds
different when excited vs bored vs annoyed — a flat voice is the fastest
"that's a bot" tell.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

from config.settings import settings
from utils.logger import logger

from .pcm import apply_gain, mono_to_stereo, resample, wav_to_pcm48

# --------------------------------------------------------------------- #
#  Text cleanup — strip things TTS pronounces badly
# --------------------------------------------------------------------- #
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_MARK = re.compile(r"[*_`~#>|]")
_DISCORD_TAG = re.compile(r"<@!?\d+>|<#\d+>|<@&\d+>|<a?:\w+:\d+>")
_BRACKET_TAG = re.compile(r"\[[a-zA-Z ]+\]")
_EMOTE_PAREN = re.compile(
    r"\((?:laughs?|sighs?|breath|breathes?|pause|whispers?|crying|"
    r"giggles?|chuckles?|scoffs?|gasps?|yawns?|coughs?|sniffs?|hums?|"
    r"clears throat|laughing|sighing|whispering|smiling|excited|sad|"
    r"angry|curious|sarcastic|dramatic|hesitant|cheerful)\)", re.I)


def clean_for_speech(text: str, *, keep_tags: bool = False) -> str:
    """Remove markdown/emoji/tags so TTS only reads speakable words.
    keep_tags preserves [direction]/(marker) tags — only for models that
    interpret them as vocal direction (fish free tier doesn't reliably)."""
    t = _MD_LINK.sub(r"\1", text)
    t = _DISCORD_TAG.sub("", t)
    t = _MD_MARK.sub("", t)
    if not keep_tags:
        t = _BRACKET_TAG.sub("", t)
        t = _EMOTE_PAREN.sub("", t)
    # strip emoji + non-latin scripts fish can't voice
    t = re.sub(r"[一-鿿぀-ヿ가-힯]", "", t)
    t = re.sub(r"[🀀-🫿☀-➿️]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------------------------- #
#  Mood -> fish prosody (speed / volume). The default voice skews quiet,
#  so baseline volume sits above zero; mood pushes it further either way.
# --------------------------------------------------------------------- #
_MOOD_PROSODY = {
    #            speed  vol_dB  temperature
    "annoyed": (1.08,   4.0,    0.85),   # clipped, harder edge
    "tired":   (0.90,  -1.0,    0.55),   # slow, flat, low energy
    "bored":   (0.96,   1.0,    0.60),
    "content": (1.02,   2.5,    0.70),
    "happy":   (1.08,   3.0,    0.80),
    "hyped":   (1.16,   4.5,    0.90),   # fast, loud, excited
}
_DEFAULT_PROSODY = _MOOD_PROSODY["content"]

# extra output loudness — the "feeble" complaint. Post-PCM gain so every
# line lands clearly above room noise.
_OUTPUT_GAIN = 1.28

# shared keep-alive client — kills per-call TLS+TCP handshake (~100-300ms)
_http = None


def _get_http():
    global _http
    if _http is None:
        import httpx
        _http = httpx.AsyncClient(
            timeout=25,
            limits=httpx.Limits(
                max_keepalive_connections=8, keepalive_expiry=60),
        )
    return _http


def _fish_params(mood: str) -> tuple[float, float, float]:
    return _MOOD_PROSODY.get(mood, _DEFAULT_PROSODY)


# --------------------------------------------------------------------- #
#  Fish WebSocket — persistent connection, per-utterance text+flush.
#  Kills the per-call TLS handshake AND gets audio chunks streaming the
#  moment synthesis starts (protocol lifted from pipecat's fish service).
# --------------------------------------------------------------------- #
class _FishStream:
    WS_URL = "wss://api.fish.audio/v1/tts/live"

    def __init__(self) -> None:
        self.ws = None
        self.mood: str | None = None
        self.lock = asyncio.Lock()

    async def _connect(self, mood: str = "content"):
        import ormsgpack
        import websockets
        from websockets.protocol import State

        if self.ws and self.ws.state is State.OPEN and self.mood == mood:
            return
        if self.ws and self.ws.state is State.OPEN:
            # prosody is session-scoped — mood change needs a reconnect
            try:
                await self.ws.send(ormsgpack.packb({"event": "stop"}))
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None
        key = getattr(settings, "fish_api_key", "") or ""
        if not key:
            raise RuntimeError("no fish.audio key")
        headers = {"Authorization": f"Bearer {key}"}
        model = getattr(settings, "fish_model", "") or "s2.1-pro-free"
        if model:
            headers["model"] = model
        self.ws = await websockets.connect(
            self.WS_URL, additional_headers=headers,
            ping_interval=20, ping_timeout=20)
        # session start — config lives on the socket, not per-message
        speed, vol, temp = _fish_params(mood)
        start = {"event": "start", "request": {
            "text": "",
            "sample_rate": 24000,     # ws path — 44100 made the server hang
            "latency": "balanced",
            "format": "pcm",
            "normalize": True,
            "prosody": {"speed": speed, "volume": vol},
            "reference_id": getattr(settings, "fish_voice_id", "") or None,
            "temperature": temp,
            "top_p": 0.75,
        }}
        await self.ws.send(ormsgpack.packb(start))
        self.mood = mood

    async def synth(self, text: str, mood: str) -> bytes:
        """Send text+flush, collect audio chunks until 'finish'.
        Returns 44100 Hz mono s16 PCM bytes."""
        import ormsgpack
        from websockets.protocol import State

        async with self.lock:
            if not self.ws or self.ws.state is not State.OPEN \
                    or self.mood != mood:
                await self._connect(mood)
            ws = self.ws
            try:
                await ws.send(ormsgpack.packb({"event": "text", "text": text}))
                await ws.send(ormsgpack.packb({"event": "flush"}))
                buf = bytearray()
                # fish keeps the socket open for the next text — no
                # per-utterance "finish" frame, so a quiet gap in the audio
                # stream is the utterance boundary
                while True:
                    try:
                        # 0.9s audio-gap = utterance end. First-chunk cap
                        # is 8s not 15 — a dead socket wastes the whole
                        # 18s wall before the HTTP fallback can try.
                        timeout = 0.9 if buf else 8.0
                        raw = await asyncio.wait_for(ws.recv(), timeout)
                    except asyncio.TimeoutError:
                        break      # audio gap -> utterance done
                    msg = ormsgpack.unpackb(raw)
                    if not isinstance(msg, dict):
                        continue
                    ev = msg.get("event")
                    if ev == "audio":
                        chunk = msg.get("audio")
                        if chunk:
                            buf.extend(chunk)
                    elif ev == "finish":
                        if msg.get("reason") == "error":
                            raise RuntimeError("fish ws synthesis error")
                        break
                    elif ev in ("close", "restart"):
                        break
                return bytes(buf)
            except Exception:
                # drop the socket — next call reconnects cleanly
                try:
                    await ws.close()
                except Exception:  # noqa: BLE001
                    pass
                self.ws = None
                raise


_fish_ws: _FishStream | None = None


async def warm_fish() -> None:
    """Pre-connect the fish websocket — first utterance pays no handshake."""
    try:
        await _get_fish_ws()._connect()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"fish ws warm-up failed (will retry on demand): {e}")


def _get_fish_ws() -> _FishStream:
    global _fish_ws
    if _fish_ws is None:
        _fish_ws = _FishStream()
    return _fish_ws


async def _tts_fish_ws(text: str, mood: str) -> bytes:
    """WebSocket path — returns 24k mono → resample → stereo 48k."""
    import numpy as np
    pcm24 = await _get_fish_ws().synth(text, mood)
    if not pcm24:
        raise RuntimeError("fish ws returned no audio")
    mono = np.frombuffer(pcm24, dtype=np.int16)
    return mono_to_stereo(resample(mono, 24000, 48000))


async def _tts_fish(text: str, mood: str = "content") -> bytes:
    key = getattr(settings, "fish_api_key", "") or ""
    if not key:
        raise RuntimeError("no fish.audio key")

    # fast path: persistent websocket — first audio lands the moment fish
    # starts generating instead of after a full HTTP round-trip
    try:
        pcm = await _tts_fish_ws(text, mood)
        return apply_gain(pcm, _OUTPUT_GAIN)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"fish ws path failed, falling back to http: {e}")

    # HTTP fallback — whole-file synth
    speed, vol, temp = _fish_params(mood)
    body: dict = {
        "text": text,
        "format": "wav",
        "sample_rate": 44100,
        "latency": "balanced",
        "normalize": True,
        "temperature": temp,
        "top_p": 0.75,
        "prosody": {"speed": speed, "volume": vol},
    }
    voice_id = getattr(settings, "fish_voice_id", "") or ""
    if voice_id:
        body["reference_id"] = voice_id

    last_err: Exception | None = None
    for _ in range(2):
        try:
            resp = await _get_http().post(
                "https://api.fish.audio/v1/tts",
                headers={
                    "Authorization": f"Bearer {key}",
                    "model": getattr(settings, "fish_model", "")
                    or "s2.1-pro-free",
                },
                json=body,
            )
            resp.raise_for_status()
            pcm = wav_to_pcm48(resp.content)
            return apply_gain(pcm, _OUTPUT_GAIN)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.debug(f"fish http attempt failed: {e}")
    raise last_err


async def synthesize(text: str, *, mood: str = "content") -> Optional[bytes]:
    """Synthesize text -> 48 kHz stereo PCM via fish.audio.
    Returns None on failure (callers treat as silence, never speak errors).
    """
    cleaned = clean_for_speech(text)
    if not cleaned:
        return None
    try:
        # hard wall — a stalled synth must not block the turn loop
        pcm = await asyncio.wait_for(_tts_fish(cleaned, mood), timeout=18.0)
        if pcm:
            return pcm
    except asyncio.TimeoutError:
        logger.warning("fish TTS timed out (18s)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fish TTS failed: {e}")
    logger.error("TTS failed — nothing spoken.")
    return None
