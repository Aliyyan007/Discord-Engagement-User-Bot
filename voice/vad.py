"""Per-user utterance segmentation (voice activity detection).

Discord only transmits RTP audio while the sender's client detects speech,
so most segmentation is simple *packet-gap* endpointing. But some clients
transmit continuously, so we also run an energy check on each decoded frame
and use the RTP ``audio_level``/``audio_voice_activity`` extensions when
present.

State machine per user:
    SILENCE --(>=ONSET voiced frames)--> SPEAKING --(hangover silence)--> emit
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .pcm import FRAME_MS, rms_dbfs

# --- tuning constants ------------------------------------------------- #
PRE_ROLL_FRAMES = 15          # 300 ms kept before speech onset
ONSET_FRAMES = 3              # ~60 ms consecutive voiced frames to start speech
HANGOVER_S = 0.42             # trailing silence that ends an utterance
GAP_S = 0.45                  # no packets at all for this long = utterance end
MIN_SPEECH_S = 0.40           # discard utterances with less voiced audio (anti-blip)
MAX_UTTERANCE_S = 15.0        # force-cut very long monologues
AUDIO_LEVEL_VOICE = 115       # RTP audio-level: 0 loud .. 127 silence
ENERGY_VOICE_DBFS = -45.0     # PCM RMS threshold when extensions are absent


@dataclass
class Utterance:
    """A completed piece of speech from one user."""

    user_id: int
    pcm: bytes                # 48 kHz stereo s16
    voiced_s: float           # seconds of voiced audio inside
    ended_at: float           # monotonic time the utterance ended
    interrupted_bot: bool = False  # speaker talked over the bot


class UtteranceAssembler:
    """Collects decoded PCM packets for one speaker and emits Utterances."""

    __slots__ = (
        "user_id", "_preroll", "_buf", "_voiced_run", "_silent_run",
        "_voiced_total", "_in_speech", "_last_packet_at", "_recent",
    )

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self._preroll: deque[bytes] = deque(maxlen=PRE_ROLL_FRAMES)
        self._buf: list[bytes] = []
        self._voiced_run = 0
        self._silent_run = 0.0
        self._voiced_total = 0.0
        self._in_speech = False
        self._last_packet_at: float = 0.0
        # ring buffer of (monotonic_ts, voiced) for barge-in measurement
        self._recent: deque[tuple[float, bool]] = deque(maxlen=100)

    # ------------------------------------------------------------------ #
    def _is_voiced(self, packet) -> bool:
        """Decide if a packet's audio is speech using best available signal."""
        if getattr(packet, "audio_voice_activity", None) is True:
            return True
        level = getattr(packet, "audio_level", None)
        if level is not None and level <= AUDIO_LEVEL_VOICE:
            return True
        if getattr(packet, "audio_voice_activity", None) is False and level is not None:
            return False
        return rms_dbfs(packet.payload) > ENERGY_VOICE_DBFS

    # ------------------------------------------------------------------ #
    def feed(self, packet, now: Optional[float] = None) -> None:
        """Feed one decoded PCM MediaPacket (called from the pump task)."""
        now = time.monotonic() if now is None else now
        voiced = self._is_voiced(packet)
        self._recent.append((now, voiced))
        self._last_packet_at = now

        if not self._in_speech:
            self._preroll.append(packet.payload)
            if voiced:
                self._voiced_run += 1
                if self._voiced_run >= ONSET_FRAMES:
                    self._in_speech = True
                    self._buf = list(self._preroll)
                    self._preroll.clear()
                    self._voiced_total = self._voiced_run * (FRAME_MS / 1000.0)
                    self._silent_run = 0.0
            else:
                self._voiced_run = 0
            return

        # In speech — append frame, track trailing silence.
        self._buf.append(packet.payload)
        if voiced:
            self._voiced_total += FRAME_MS / 1000.0
            self._silent_run = 0.0
        else:
            self._silent_run += FRAME_MS / 1000.0

    # ------------------------------------------------------------------ #
    def tick(self, now: Optional[float] = None) -> Optional[Utterance]:
        """Periodic check (call ~every 50 ms) — returns a finished Utterance."""
        now = time.monotonic() if now is None else now
        if not self._in_speech:
            return None

        duration = len(self._buf) * FRAME_MS / 1000.0
        packet_gap = now - self._last_packet_at if self._last_packet_at else 0.0

        done = (
            self._silent_run >= HANGOVER_S
            or packet_gap >= GAP_S
            or duration >= MAX_UTTERANCE_S
        )
        if not done:
            return None

        pcm = b"".join(self._buf)
        voiced = self._voiced_total
        self._reset()
        if voiced < MIN_SPEECH_S:
            return None
        return Utterance(
            user_id=self.user_id, pcm=pcm, voiced_s=voiced, ended_at=now,
        )

    # ------------------------------------------------------------------ #
    def voiced_ms_in_window(self, window_s: float, now: Optional[float] = None) -> float:
        """How much voiced audio arrived in the last `window_s` seconds
        (used for barge-in detection while the bot is speaking)."""
        now = time.monotonic() if now is None else now
        cutoff = now - window_s
        ms = 0.0
        for ts, voiced in self._recent:
            if voiced and ts >= cutoff:
                ms += FRAME_MS
        return ms

    @property
    def speaking(self) -> bool:
        return self._in_speech

    @property
    def last_packet_at(self) -> float:
        return self._last_packet_at

    def _reset(self) -> None:
        self._buf = []
        self._preroll.clear()
        self._voiced_run = 0
        self._silent_run = 0.0
        self._voiced_total = 0.0
        self._in_speech = False
