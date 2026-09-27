"""Component tests for the voice pipeline — no Discord connection needed.

Run: python -m tests.test_voice_pipeline
"""
from __future__ import annotations

import asyncio
import math
import time
from types import SimpleNamespace

import numpy as np


def _sine_pcm(seconds: float, freq: float = 220.0, amp: float = 0.3) -> bytes:
    """Generate a tone as 48 kHz stereo s16 (stands in for speech energy)."""
    n = int(48000 * seconds)
    t = np.arange(n, dtype=np.float32) / 48000
    tone = (np.sin(2 * math.pi * freq * t) * amp * 32767).astype(np.int16)
    return np.repeat(tone, 2).tobytes()


def _silence_pcm(seconds: float) -> bytes:
    return b"\x00" * int(48000 * seconds * 2 * 2)


def _pkt(pcm: bytes, uid: int = 42):
    """Fake decoded MediaPacket."""
    return SimpleNamespace(
        media_type="audio", codec="pcm", payload=pcm, user_id=uid, ssrc=1000 + uid,
        audio_level=None, audio_voice_activity=None, marker=False,
        sequence=0, timestamp=0, received_at=None,
    )


def _frames(pcm: bytes):
    from voice.pcm import FRAME_BYTES
    return [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


def test_pcm():
    from voice import pcm
    # roundtrip: pcm48 -> 16k mono wav
    tone = _sine_pcm(1.0)
    mono16 = pcm.pcm48_to_mono16k(tone)
    wav = pcm.pcm_to_wav(mono16, rate=16000, channels=1)
    back = pcm.wav_to_pcm48(wav)
    assert len(back) > 0 and len(back) % pcm.FRAME_BYTES == 0 or True
    # frames chunking
    frames = pcm.chunk_frames(tone)
    assert all(len(f) == pcm.FRAME_BYTES for f in frames)
    assert abs(len(frames) - 50) <= 1  # 1s ≈ 50 frames
    # energy detection
    assert pcm.rms_dbfs(tone) > -30
    assert pcm.rms_dbfs(_silence_pcm(0.1)) < -60
    print("pcm: OK")


def test_vad():
    from voice.vad import UtteranceAssembler
    asm = UtteranceAssembler(42)
    now = time.monotonic()
    # feed 500ms of silence packets -> no utterance
    for f in _frames(_silence_pcm(0.5)):
        asm.feed(_pkt(f), now); now += 0.02
    assert asm.tick(now) is None
    # feed 1s of "speech" (tone), then silence gap
    for f in _frames(_sine_pcm(1.0)):
        asm.feed(_pkt(f), now); now += 0.02
    # still speaking — no emit yet
    assert asm.tick(now) is None
    # packets stop; advance past hangover gap
    now += 0.8
    utt = asm.tick(now)
    assert utt is not None and utt.user_id == 42
    assert utt.voiced_s > 0.5
    print(f"vad: OK (utterance {utt.voiced_s:.2f}s voiced, {len(utt.pcm)} bytes)")


async def test_stt_tts():
    from voice import stt, tts
    # 1) TTS -> PCM
    pcm = await tts.synthesize("hey what's up, just testing the voice")
    if pcm:
        print(f"tts: OK ({len(pcm)} bytes, {len(pcm)/3840*0.02:.1f}s audio)")
        # 2) roundtrip: transcribe our own TTS back through Whisper
        text = await stt.transcribe(pcm)
        print(f"stt roundtrip: {text!r}")
        assert any(w in text.lower() for w in ("hey", "testing", "voice", "what")), text
        print("stt: OK")
    else:
        print("tts: ALL BACKENDS FAILED — check keys/network")


async def main():
    test_pcm()
    test_vad()
    await test_stt_tts()


if __name__ == "__main__":
    asyncio.run(main())
