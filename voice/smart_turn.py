"""Smart Turn v3.2 — trained end-of-turn classifier (pipecat-ai, BSD-2).

Replaces regex pause-guessing with a real model: Whisper-Tiny encoder +
linear head, 8.7 MB int8 ONNX, ~50-200 ms per call on CPU. Given the last
<=8 s of a speaker's audio, predicts P(turn complete). We run it when our
VAD emits an utterance that *might* be a mid-thought pause — INCOMPLETE
means hold for the continuation instead of replying into the gap.

Model: https://huggingface.co/pipecat-ai/smart-turn-v3
Deps: onnxruntime + transformers (WhisperFeatureExtractor, no torch needed).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .pcm import pcm48_to_mono16k

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "smart-turn-v3.2-cpu.onnx"
MODEL_URL = ("https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/"
             "main/smart-turn-v3.2-cpu.onnx")

_MAX = 8 * 16000          # model consumes last <=8 s @ 16 kHz


def _ensure_model() -> bool:
    """Download the ONNX file if missing (ephemeral disks like Render's
    wipe it on every deploy)."""
    if MODEL_PATH.exists():
        return True
    try:
        import urllib.request
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return True
    except Exception:  # noqa: BLE001
        return False

_fe = None
_session = None
_pool: ThreadPoolExecutor | None = None
_available = None


def _load() -> bool:
    """Heavy init — runs INSIDE the executor thread (transformers import +
    ONNX session load takes ~20s cold; never run it on the event loop)."""
    global _fe, _session, _available
    if _available is not None:
        return _available
    try:
        if not _ensure_model():
            raise RuntimeError("model download failed")
        from transformers import WhisperFeatureExtractor
        import onnxruntime as ort
        _fe = WhisperFeatureExtractor(chunk_length=8)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=so,
            providers=["CPUExecutionProvider"])
        _available = True
    except Exception as e:  # noqa: BLE001
        _available = False
        from utils.logger import logger
        logger.warning(f"smart-turn unavailable ({type(e).__name__}: {e}) "
                       f"— falling back to heuristic endpointing")
    return _available


def _prob_sync(pcm48: bytes) -> float:
    _load()
    audio = np.frombuffer(pcm48_to_mono16k(pcm48), dtype=np.int16) \
        .astype(np.float32) / 32768.0
    if len(audio) > _MAX:
        audio = audio[-_MAX:]
    elif len(audio) < _MAX:
        audio = np.pad(audio, (_MAX - len(audio), 0))
    feats = _fe(audio, sampling_rate=16000, return_tensors="np",
                padding="max_length", max_length=_MAX, truncation=True,
                do_normalize=True).input_features
    feats = feats.squeeze(0).astype(np.float32)[None]    # (1, 80, 800)
    return float(_session.run(None, {"input_features": feats})[0].item())


async def turn_complete_prob(pcm48: bytes) -> float:
    """P(the speaker's turn is complete) — 0..1. Returns 1.0 (complete)
    when the model isn't available so callers fail open."""
    from config.settings import settings
    if not getattr(settings, "voice_smart_turn", True):
        return 1.0
    if not MODEL_PATH.exists() and _available is not None:
        return 1.0   # download already attempted and failed
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_pool, _prob_sync, pcm48)
    except Exception:  # noqa: BLE001
        return 1.0


def available() -> bool:
    """Cheap check — the heavy model load happens on the executor thread."""
    from config.settings import settings
    if not getattr(settings, "voice_smart_turn", True):
        return False
    return _available is not False
