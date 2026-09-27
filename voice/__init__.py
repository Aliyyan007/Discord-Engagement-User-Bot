"""Real-time voice chat pipeline for the Engager self-bot.

Modules:
    pcm      — raw PCM helpers (48 kHz stereo s16le, 20 ms frames)
    vad      — per-user utterance segmentation (energy + packet-gap VAD)
    stt      — Groq Whisper transcription
    tts      — human-like speech synthesis (edge-tts / Groq Orpheus chain)
    session  — VoiceSession: the live VC pipeline + human-behaviour engine
    manager  — global session registry used by tools/events
"""
from __future__ import annotations
