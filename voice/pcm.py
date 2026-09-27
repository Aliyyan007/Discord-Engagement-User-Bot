"""PCM audio helpers for the Discord voice pipeline.

Discord's voice wire format is 48 kHz, 16-bit signed little-endian, stereo,
delivered in 20 ms frames (960 samples per channel = 3840 bytes per frame).
Everything here operates on raw ``bytes`` unless noted.
"""
from __future__ import annotations

import io
import math
import wave

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2          # bytes per sample (s16le)
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * CHANNELS * SAMPLE_WIDTH  # 3840


# --------------------------------------------------------------------- #
#  Basic conversions
# --------------------------------------------------------------------- #
def pcm_duration(pcm: bytes) -> float:
    """Duration in seconds of a 48 kHz stereo s16 PCM buffer."""
    return len(pcm) / FRAME_BYTES * (FRAME_MS / 1000.0)


def chunk_frames(pcm: bytes, *, pad: bool = True) -> list[bytes]:
    """Split a PCM buffer into 3840-byte frames (pads the tail with silence)."""
    frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if frames and pad and len(frames[-1]) < FRAME_BYTES:
        frames[-1] = frames[-1].ljust(FRAME_BYTES, b"\x00")
    return [f for f in frames if f]


def stereo_to_mono(pcm: bytes, rate: int = SAMPLE_RATE) -> np.ndarray:
    """48 kHz stereo s16 -> mono int16 numpy array (average of channels)."""
    stereo = np.frombuffer(pcm, dtype=np.int16)
    if len(stereo) % 2:
        stereo = stereo[:-1]
    stereo = stereo.reshape(-1, 2)
    return stereo.astype(np.int32).mean(axis=1).astype(np.int16)


def mono_to_stereo(mono: np.ndarray) -> bytes:
    """mono int16 array -> interleaved stereo s16le bytes."""
    return np.repeat(mono.astype(np.int16), 2).tobytes()


def resample(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a mono int16 array. Uses polyphase when scipy is present,
    otherwise linear interpolation — both fine for voice."""
    if src_rate == dst_rate:
        return mono.astype(np.int16)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_rate, dst_rate)
        out = resample_poly(mono.astype(np.float32), dst_rate // g, src_rate // g)
        return np.clip(out, -32768, 32767).astype(np.int16)
    except Exception:
        n_out = int(len(mono) * dst_rate / src_rate)
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        idx = np.linspace(0, len(mono) - 1, n_out)
        return np.interp(idx, np.arange(len(mono)), mono.astype(np.float32)).astype(np.int16)


def pcm48_to_mono16k(pcm: bytes) -> bytes:
    """48 kHz stereo PCM -> 16 kHz mono s16 bytes (what Whisper wants)."""
    mono = stereo_to_mono(pcm)
    return resample(mono, SAMPLE_RATE, 16000).tobytes()


# --------------------------------------------------------------------- #
#  Container encode/decode
# --------------------------------------------------------------------- #
def pcm_to_wav(pcm: bytes, *, rate: int, channels: int) -> bytes:
    """Wrap raw s16 PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm48(wav_bytes: bytes) -> bytes:
    """Decode a WAV container to 48 kHz stereo s16 PCM bytes."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != SAMPLE_WIDTH:
        raise ValueError(f"unsupported wav sample width: {width}")
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        samples = stereo_to_mono(raw)
    elif channels != 1:
        raise ValueError(f"unsupported wav channels: {channels}")
    samples = resample(samples, rate, SAMPLE_RATE)
    return mono_to_stereo(samples)


def compressed_to_pcm48(data: bytes) -> bytes:
    """Decode compressed audio (mp3/ogg/flac...) to 48 kHz stereo s16 PCM.

    Uses PyAV in-process (no ffmpeg binary needed).
    """
    import av  # PyAV — installed in this env
    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="s16", layout="stereo", rate=SAMPLE_RATE)
    out = io.BytesIO()
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            out.write(rf.to_ndarray().tobytes())
    for rf in resampler.resample(None):
        out.write(rf.to_ndarray().tobytes())
    return out.getvalue()


# --------------------------------------------------------------------- #
#  Energy / VAD helpers
# --------------------------------------------------------------------- #
def apply_gain(pcm: bytes, gain: float) -> bytes:
    """Scale PCM loudness by `gain` (1.0 = unchanged). Used for natural
    per-utterance loudness jitter so the voice print isn't identical."""
    if gain == 1.0 or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767)
    return samples.astype(np.int16).tobytes()


def rms_dbfs(pcm: bytes) -> float:
    """dBFS energy of a PCM buffer (<= 0). Silence sits around -inf/-60."""
    if not pcm:
        return -120.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(samples * samples)) + 1e-9)
    return 20.0 * math.log10(max(rms, 1.0) / 32767.0)
