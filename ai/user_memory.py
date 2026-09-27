"""Persistent per-user memory — what the bot knows about each person.

Ported from the reference bot's ai/memory.py design (JSON backend, atomic
writes, algorithmic contradiction-dedup) minus the D1/sync-wrapper plumbing.
Facts are stored keyed by user_id (not mutable display names), capped, and
injected into the voice room-context so the bot "knows" people across
sessions, not just within one call.

    data/user_memory.json
    {
      "users": {
        "<uid>": {"name": "...", "real_name": "", "facts": [], "sessions": N}
      }
    }
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PATH = _DATA_DIR / "user_memory.json"
_LOCK = threading.Lock()
_cache: dict | None = None

MAX_FACTS = 20

# facts naming the same category replace each other ("call me ali"
# overrides "my name is aliyyan") instead of stacking forever.
_FACT_CATEGORIES = [
    (re.compile(r"(?:real\s+name\s+is|name\s+is|called|known\s+as|call me|"
                r"goes by)\s+(\w+)", re.I), "name"),
    (re.compile(r"(?:years?\s+old|age\s+is)\s*(\d+)", re.I), "age"),
    (re.compile(r"(?:from|lives?\s+in|resides?\s+in)\s+(\w+)", re.I), "location"),
    (re.compile(r"(?:likes?|enjoys?|loves?)\s+(\w+)", re.I), "like"),
    (re.compile(r"(?:hates?|dislikes?)\s+(\w+)", re.I), "dislike"),
]


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        _cache = {"users": {}}
    _cache.setdefault("users", {})
    return _cache


def _save() -> None:
    """Atomic write: tmp -> .bak -> replace (Windows-safe, retried)."""
    _DATA_DIR.mkdir(exist_ok=True)
    tmp = _PATH.with_suffix(".tmp")
    bak = _PATH.with_suffix(".bak")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    for _ in range(3):
        try:
            if _PATH.exists():
                os.replace(_PATH, bak)
            os.replace(tmp, _PATH)
            if bak.exists():
                bak.unlink(missing_ok=True)
            return
        except OSError:
            time.sleep(0.1)


def _category(fact: str) -> str | None:
    for rx, cat in _FACT_CATEGORIES:
        if rx.search(fact):
            return cat
    return None


def add_facts(user_id: int | str, username: str, facts: list[str]) -> list[str]:
    """Append facts for a user; same-category facts get replaced.
    Returns the facts actually stored."""
    if not facts:
        return []
    with _LOCK:
        mem = _load()
        u = mem["users"].setdefault(
            str(user_id), {"name": username, "facts": [], "sessions": 0})
        u["name"] = username  # display names drift; keep fresh
        stored = []
        for f in facts:
            f = f.strip().rstrip(".")
            if not f or len(f) > 140:
                continue
            cat = _category(f)
            existing = u["facts"]
            if cat:
                # replace same-category fact with different text
                existing[:] = [
                    x for x in existing
                    if _category(x) != cat or x.lower() == f.lower()
                ]
            if f.lower() in (x.lower() for x in existing):
                continue
            existing.append(f)
            stored.append(f)
        u["facts"] = u["facts"][-MAX_FACTS:]
        _save()
        return stored


def set_real_name(user_id: int | str, username: str, real_name: str) -> None:
    with _LOCK:
        mem = _load()
        u = mem["users"].setdefault(
            str(user_id), {"name": username, "facts": [], "sessions": 0})
        u["name"] = username
        old = u.get("real_name", "")
        if old and old.lower() != real_name.lower():
            # scrub stale "called X" facts
            u["facts"] = [f for f in u["facts"] if old.lower() not in f.lower()]
        u["real_name"] = real_name
        _save()


def get_profile(user_id: int | str) -> dict:
    return dict(_load()["users"].get(str(user_id), {}))


def memory_line(user_id: int | str, username: str = "") -> str:
    """'about ali: goes by Aliyyan. things you know: likes valorant; from Lahore'"""
    u = _load()["users"].get(str(user_id))
    if not u:
        return ""
    name = u.get("real_name") or u.get("name") or username or "them"
    bits = []
    if u.get("real_name"):
        bits.append(f"their real name is {u['real_name']}")
    facts = u.get("facts") or []
    if facts:
        bits.append("things you know: " + "; ".join(facts[-8:]))
    if not bits:
        return ""
    return f"about {u.get('name', name)}: " + ". ".join(bits)


def note_session_join(user_id: int | str, username: str) -> int:
    """Bump the session counter — returns how many VCs they've shared."""
    with _LOCK:
        mem = _load()
        u = mem["users"].setdefault(
            str(user_id), {"name": username, "facts": [], "sessions": 0})
        u["name"] = username
        u["sessions"] = int(u.get("sessions", 0)) + 1
        u["last_seen"] = time.time()
        _save()
        return u["sessions"]
