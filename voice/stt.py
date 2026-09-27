"""Speech-to-text via Groq's hosted Whisper (whisper-large-v3-turbo).

Latency is ~200-350 ms for a typical utterance. Audio is downsampled to
16 kHz mono WAV before upload (~6x smaller than 48 kHz stereo).

Includes a hallucination filter — batch Whisper invents text on noise.
"""
from __future__ import annotations

import re

from config.settings import settings
from core.groq_pool import get_pool
from utils.logger import logger

from .pcm import apply_gain, pcm48_to_mono16k, pcm_to_wav, rms_dbfs

# Whisper hallucination blacklist — phrases it emits on silence/noise.
_HALLUCINATIONS = {
    "thank you for watching", "thanks for watching", "please subscribe",
    "subtitles by", "subtitle by", "transcribed by", "you", "bye",
    "thank you", "thanks.", "amen", "...", "…", "so", "okay", "ok",
    "the end", "thank you for listening", "amara.org", "captioned by",
    "gracias por ver", "merci d'avoir regardé",
}

# AGC: quiet speakers lose words to the decoder — lift them toward normal
# speech level (~-20 dBFS). Higher ceiling than before, but measured on
# the VOICED portion only (whole-buffer rms includes preroll+hangover
# silence and underestimates the speech level).
_TARGET_DBFS = -20.0
_MAX_GAIN = 18.0

# Garbled transcripts (mid-word capitals etc. — "RAcmy, Ful?") mean Whisper
# heard noise/mumbling and guessed; flagging them is better than parroting.
_GARBLE_RX = re.compile(r"\b[a-z]{2,}[A-Z][a-z]+|\b[A-Z]{3,}[a-z]{2,}\b")

# Per-speaker language pinning — auto-detect on short clips misfires hard
# (Icelandic/Hindi output on accented English). Once a speaker shows a
# dominant language, pin it; multilingual speakers stay auto.
_USER_LANG: dict[int, dict] = {}      # speaker_id -> {"langs": Counter, "pin": str|None}
_LANG_PIN_AFTER = 4                   # utterances before considering a pin
_LANG_PIN_SHARE = 0.75                # dominant share needed to pin
_FALLBACK_MODEL = "whisper-large-v3"  # heavier decode for flagged retries

# verbose_json returns display names ("English"); Groq's language param
# wants ISO-639-1 codes — map the ones we realistically see
_LANG_NAME_TO_ISO = {
    "english": "en", "spanish": "es", "french": "fr", "urdu": "ur",
    "hindi": "hi", "arabic": "ar", "german": "de", "italian": "it",
    "portuguese": "pt", "turkish": "tr", "russian": "ru", "dutch": "nl",
    "polish": "pl", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "punjabi": "pa", "tagalog": "tl", "vietnamese": "vi", "bengali": "bn",
    "indonesian": "id", "thai": "th", "swahili": "sw", "greek": "el",
    "romanian": "ro", "swedish": "sv", "norwegian": "no", "danish": "da",
    "finnish": "fi", "czech": "cs", "ukrainian": "uk", "persian": "fa",
    "pashto": "ps", "gujarati": "gu", "marathi": "mr", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "malayalam": "ml", "nepali": "ne",
}

# Cross-language script check — Urdu/Devanagari/Arabic script output on an
# English-dominant call = detection misfire, not real speech.
_NON_LATIN_RX = re.compile(
    r"[̀-ͯ০-৿ऀ-ॿ਀-੿઀-૿଀-୿ஂ-௺ఀ-౿ಂ-ൿ฀-๿؀-ۿ]")


def _clean(text: str) -> str:
    """Normalise a transcript; returns '' if it's a hallucination/empty."""
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) < 2:
        return ""
    low = t.lower().strip(".!? ")
    if low in _HALLUCINATIONS:
        return ""
    # long hallucination phrases match as SUBSTRINGS too — "gracias por
    # ver el video" isn't exact-equal to "gracias por ver" but it's the
    # same youtube-outro hallucination
    if any(h in low for h in _HALLUCINATION_SUBSTRINGS):
        return ""
    # lone punctuation / music symbols
    if not re.search(r"[a-zA-Z0-9]", t):
        return ""
    # repetition loop guard — "Nome, Nome, Nome" / "Ngo x8" garbage that
    # compression_ratio doesn't always catch
    words = [w for w in t.lower().split() if w.isalnum()]
    if len(words) >= 5 and len(set(words)) / len(words) < 0.4:
        return ""
    return t


# multi-word hallucination stems — dropped as substrings
_HALLUCINATION_SUBSTRINGS = {
    "thank you for watching", "thanks for watching", "thank you for listening",
    "gracias por ver", "merci d'avoir", "please subscribe",
    "subtitles by", "subtitle by", "transcribed by", "amara.org",
    "captioned by", "like and subscribe",
}


def _language_for(speaker_id: int | None) -> str | None:
    """Return a pinned ISO language for this speaker, or None (auto)."""
    info = _USER_LANG.get(speaker_id)
    return info["pin"] if info else None


def _note_language(speaker_id: int | None, lang: str) -> None:
    if speaker_id is None or not lang:
        return
    info = _USER_LANG.setdefault(speaker_id, {"langs": {}, "pin": None})
    if info["pin"]:
        return
    lg = lang.strip().lower()
    info["langs"][lg] = info["langs"].get(lg, 0) + 1
    total = sum(info["langs"].values())
    if total >= _LANG_PIN_AFTER:
        top, n = max(info["langs"].items(), key=lambda kv: kv[1])
        # pin WHATEVER dominates — a consistently-Spanish speaker needs
        # language=es just as much as an English one needs en; auto-detect
        # is what's producing the language-misfire garbage
        if n / total >= _LANG_PIN_SHARE:
            pin = _LANG_NAME_TO_ISO.get(top)
            if pin:
                info["pin"] = pin
                logger.info(
                    f"STT: pinned language '{pin}' for speaker {speaker_id}")


def _seg_metrics(resp) -> tuple[float, float, float]:
    """(avg no_speech_prob, avg logprob, max compression_ratio) across
    verbose_json segments; zeros when the API returns none."""
    segs = getattr(resp, "segments", None) or []
    ns, lp, cr = [], [], []
    for s in segs:
        g = (lambda k, d=None: getattr(s, k, d)
             if not isinstance(s, dict) else s.get(k, d))
        if g("no_speech_prob") is not None:
            ns.append(g("no_speech_prob"))
        if g("avg_logprob") is not None:
            lp.append(g("avg_logprob"))
        if g("compression_ratio") is not None:
            cr.append(g("compression_ratio"))
    return (
        sum(ns) / len(ns) if ns else 0.0,
        sum(lp) / len(lp) if lp else 0.0,
        max(cr) if cr else 0.0,
    )


async def transcribe(pcm48: bytes, *, speaker_hint: str = "",
                     context_hint: str = "",
                     speaker_id: int | None = None) -> str:
    """Transcribe 48 kHz stereo PCM speech -> text ('' on failure/noise)."""
    if not pcm48:
        return ""
    # --- AGC: pull quiet audio toward normal speech level --------------- #
    # measure on the whole buffer but scale by voiced estimate — the caller
    # already endpointed this clip so most of it IS speech
    level = rms_dbfs(pcm48)
    if level < _TARGET_DBFS - 6:
        gain = min(_MAX_GAIN, 10 ** ((_TARGET_DBFS - level) / 20.0))
        pcm48 = apply_gain(pcm48, gain)
        logger.debug(f"STT gain {gain:.1f}x (was {level:.0f} dBFS)")
    mono16 = pcm48_to_mono16k(pcm48)
    wav = pcm_to_wav(mono16, rate=16000, channels=1)

    # Whisper prompt = "preceding transcript" context, NOT instructions —
    # kept short, transcript-style, names LAST (prompt-echo suppression).
    parts = ["casual discord voice chat"]
    if context_hint:
        parts.append(context_hint[-160:])
    if speaker_hint:
        parts.append(speaker_hint)
    prompt = " — ".join(p for p in parts if p)

    pool = get_pool()
    language = _language_for(speaker_id)

    async def _call(client, *, model=None, prompt_arg=None, temp=0.0):
        kwargs = dict(
            model=model or settings.voice_stt_model,
            file=("speech.wav", wav, "audio/wav"),
            response_format="verbose_json",
            prompt=prompt_arg if prompt_arg is not None else prompt,
            timeout=15.0,
        )
        if temp:
            kwargs["temperature"] = temp
        if language:
            kwargs["language"] = language
        return await client.audio.transcriptions.create(**kwargs)

    def _evaluate(resp) -> tuple[str, float, float, float, str]:
        text = resp if isinstance(resp, str) else getattr(resp, "text", "")
        text = _clean(text if isinstance(text, str) else str(text))
        ns, lp, cr = _seg_metrics(resp)
        lang = getattr(resp, "language", "") or ""
        return text, ns, lp, cr, lang

    try:
        resp = await pool.call(_call)
        text, no_speech, logprob, comp, lang = _evaluate(resp)
        if not text:
            # whisper-turbo has a documented bug where a long/glossary
            # prompt suppresses real decodes — a >=0.8s clip returning
            # empty means speech was probably there; retry promptless
            if len(pcm48) / 192000.0 >= 0.8:
                logger.debug("STT empty on long clip — retrying promptless")
                resp2 = await pool.call(
                    lambda c: _call(c, prompt_arg="", temp=0.1))
                text, no_speech, logprob, comp, lang = _evaluate(resp2)
            if not text:
                return ""

        # ---- confidence gates (OpenAI whisper defaults) ---------------- #
        dropped = (no_speech > 0.6 and logprob < -1.0)
        flagged = (comp > 2.4 or logprob < -1.0 or comp < 0.6
                   or _NON_LATIN_RX.search(text) is not None)
        # prompt-echo = the transcript IS the name list. A single name
        # like "Eudora" is real speech (someone calling the bot), not echo.
        echo = False
        if speaker_hint:
            names_l = [n.strip().lower() for n in speaker_hint.split(",")
                       if n.strip()]
            echo = sum(1 for n in names_l if n in text.lower()) >= 2

        if dropped or logprob < -1.5:
            logger.debug(
                f"STT conf-drop (ns={no_speech:.2f} lp={logprob:.2f} "
                f"cr={comp:.2f})")
            return ""

        # ---- retry ladder on flagged/echo output ------------------------ #
        if flagged or echo:
            logger.info(
                f"STT flagged ({'echo' if echo else 'metrics'}: "
                f"lp={logprob:.2f} cr={comp:.2f}) — retrying")
            resp2 = await pool.call(
                lambda c: _call(c, prompt_arg=None if echo else prompt,
                                temp=0.2))
            t2, ns2, lp2, cr2, lang2 = _evaluate(resp2)
            if t2 and not (ns2 > 0.6 and lp2 < -1.0):
                # if still flagged, one last shot on the heavier model
                if lp2 < -1.0 or cr2 > 2.4:
                    resp3 = await pool.call(
                        lambda c: _call(c, model=_FALLBACK_MODEL, temp=0.2))
                    t3, ns3, lp3, cr3, lang3 = _evaluate(resp3)
                    if t3 and not (ns3 > 0.6 and lp3 < -1.0) and lp3 >= lp2:
                        text, lang = t3, lang3
                    else:
                        text, lang = t2, lang2
                else:
                    text, lang = t2, lang2

        _note_language(speaker_id, lang)
        return text
    except Exception as e:  # noqa: BLE001
        logger.warning(f"STT failed: {e}")
        return ""
