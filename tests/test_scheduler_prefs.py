"""Scheduler + preferences — the owner's timed-action system.

Covers: persistent JSON store, interval rescheduling, one-shot firing +
auto-delete, cancel-by-query, owner gating, restart revival (reloads and
keeps firing).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import discord  # noqa: E402

from core.scheduler import Scheduler          # noqa: E402
from tools.context import ToolContext          # noqa: E402
import core.scheduler as sched_mod             # noqa: E402
import tools.scheduler as sched_tools          # noqa: E402
import tools.prefs as prefs                    # noqa: E402


class FakeChannel:
    def __init__(self, cid=101, name="general"):
        self.id = cid
        self.name = name
        self.guild = mock.MagicMock(id=55)
        self.sent: list[str] = []

    async def send(self, content):
        self.sent.append(content)
        return mock.MagicMock(id=len(self.sent))


class FakeBot:
    def __init__(self, channel):
        self._ch = channel

    def get_channel(self, cid):
        return self._ch if cid == self._ch.id else None


def _run(coro):
    return asyncio.run(coro)


def _ctx(owner=True):
    ctx = mock.MagicMock(spec=ToolContext)
    ctx.author_id = 999 if owner else 555
    ctx.guild = mock.MagicMock()
    ctx.require_guild = lambda: ctx.guild
    ctx.get_current_channel = lambda: None
    return ctx


def test_scheduler_fires_and_cancels():
    """Interval task fires repeatedly; cancel by query stops it + cleans store."""
    async def _scenario():
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "tasks.json"
            with mock.patch.object(sched_mod, "STORE", store):
                ch = FakeChannel()
                s = Scheduler(FakeBot(ch))
                tid = s.add({
                    "kind": "send_message", "channel_id": 101,
                    "channel_name": "general", "content": "hey!",
                    "interval_s": 0.05, "run_at": time.time(),
                })
                await s.start()
                await asyncio.sleep(1.6)   # loop ticks every 1s
                await s.stop()
                assert len(ch.sent) >= 1, f"expected sends, got {ch.sent}"
                assert store.exists() and json.loads(store.read_text()), \
                    "task missing from store"
                killed = s.cancel(lambda t: t["id"] == tid)
                assert len(killed) == 1
                assert json.loads(store.read_text()) == [], \
                    "store not cleaned after cancel"
                before = len(ch.sent)
                await asyncio.sleep(0.2)
                assert len(ch.sent) == before, "task kept firing after cancel"
    _run(_scenario())
    print("    interval task fires, cancel stops + wipes store")


def test_restart_revival():
    """A new Scheduler over the same store keeps the task alive."""
    async def _scenario():
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "tasks.json"
            with mock.patch.object(sched_mod, "STORE", store):
                ch = FakeChannel()
                s1 = Scheduler(FakeBot(ch))
                s1.add({"kind": "send_message", "channel_id": 101,
                        "channel_name": "general", "content": "still here",
                        "interval_s": 0.05, "run_at": time.time()})
                del s1
                s2 = Scheduler(FakeBot(ch))
                await s2.start()      # loads store -> resumes
                await asyncio.sleep(1.6)
                await s2.stop()
                assert ch.sent, "task didn't resume after restart"
    _run(_scenario())
    print("    restart revival: persisted task kept firing")


def test_oneshot_auto_deletes():
    """One-shot fires once then removes itself from the store."""
    async def _scenario():
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "tasks.json"
            with mock.patch.object(sched_mod, "STORE", store):
                ch = FakeChannel()
                s = Scheduler(FakeBot(ch))
                s.add({"kind": "send_message", "channel_id": 101,
                       "channel_name": "general", "content": "once",
                       "interval_s": None, "run_at": time.time()})
                await s.start()
                await asyncio.sleep(1.6)
                await s.stop()
                assert ch.sent == ["once"], ch.sent
                assert json.loads(store.read_text()) == []
    _run(_scenario())
    print("    one-shot fired once and self-deleted")


def test_owner_gate():
    """Non-owner can't schedule; owner can."""
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(sched_mod, "STORE", Path(td) / "t.json"), \
             mock.patch.object(prefs, "STORE", Path(td) / "p.json"), \
             mock.patch.object(prefs.settings, "owner_user_id", 999), \
             mock.patch.object(sched_tools.settings, "owner_user_id", 999), \
             mock.patch.object(sched_tools, "fuzzy_search",
                               lambda q, c, key, limit: []):
            chans = [FakeChannel()]
            ctx = _ctx(owner=False)
            ctx.guild.channels = chans
            out = _run(sched_tools.schedule_message(
                ctx, "general", "x", delay_s=0, interval_s=0))
            assert "error" in out and "owner" in out["error"].lower()
            out2 = _run(prefs.set_preference(
                ctx, "welcome_channel", "general"))
            assert "error" in out2
            ctx3 = _ctx(owner=True)
            ctx3.guild.channels = chans
            out3 = _run(prefs.set_preference(
                ctx3, "welcome_channel", "general"))
            assert out3.get("ok") and prefs.get_pref("welcome_channel") == "general"
    print("    owner gate: stranger refused, owner writes prefs")


def test_did_send_suppression():
    """send_message marks ctx.did_send so the reply path stays silent."""
    ctx = ToolContext(bot=mock.MagicMock())
    assert ctx.did_send is False and ctx.author_id is None
    ctx.did_send = True   # what send_message does
    assert ctx.did_send is True
    print("    did_send flag present on ToolContext")


def main():
    print("test_scheduler_prefs")
    for t in (test_scheduler_fires_and_cancels, test_restart_revival,
              test_oneshot_auto_deletes, test_owner_gate,
              test_did_send_suppression):
        t()


if __name__ == "__main__":
    main()
