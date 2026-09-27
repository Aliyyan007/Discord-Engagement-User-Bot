"""Offline tests for the intent router + agent action path — no Discord,
no network.

Covers the routing chain end to end (with the network boundary mocked):
    ai.router.classify / classify_llm   -> Route(kind, categories)
    tools.registry.build_tools(mode, categories)  -> filtered schemas
    ai.agent.Agent.run(mode=..., categories=..., no_tools=...) -> reply

Network boundaries replaced:
    Agent.pool / ai.router.get_pool    -> FakePool returning canned Groq
                                         responses (types.SimpleNamespace)
    ai.agent.dispatch                  -> recording stub (no real tools)

Run:  python -m tests.test_action_routing
  or: python tests/test_action_routing.py

Plain asserts, pytest-free. Tests listed in KNOWN_BUGS document genuine
source bugs — they assert the EXPECTED behaviour and report BUG (not FAIL)
until the source is fixed.
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# allow running as a plain script (python tests/test_action_routing.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.agent import Agent  # noqa: E402
from ai.router import ROUTE_ACTION, ROUTE_CHAT, classify, classify_llm  # noqa: E402
from config.prompts import CHAT_MODE, COMMAND_MODE, VOICE_MODE  # noqa: E402
from tools.context import ToolContext  # noqa: E402
from tools.registry import _CATEGORIES, _RESOLVER_TOOLS, build_tools  # noqa: E402


# --------------------------------------------------------------------- #
#  Groq stand-ins
# --------------------------------------------------------------------- #


def _resp(content: str | None = None, tool_calls: list | None = None):
    """Canned Groq chat-completion response (SimpleNamespace-shaped)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
        )],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call(name: str, args: dict, call_id: str = "call_1"):
    """Canned tool_call object (mirrors Groq's tc.function.name/.arguments)."""
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakePool:
    """Stands in for core.groq_pool.GroqPool — records chat() kwargs and
    returns canned responses in call order."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakePool ran out of canned responses")
        r = self.responses.pop(0)
        return r(kwargs) if callable(r) else r


def _make_agent(pool: FakePool) -> Agent:
    """Construct a real Agent with the pool swapped for a FakePool.

    get_pool is patched only for __init__ (the real GroqPool would read
    .env keys and build AsyncGroq clients — unnecessary here). The real
    build_tools() still runs, so tool selection is genuinely exercised.
    """
    ctx = ToolContext(bot=SimpleNamespace(id=999, display_name="eudora", bot=True))
    with mock.patch("ai.agent.get_pool", return_value=pool):
        return Agent(ctx)


# --------------------------------------------------------------------- #
#  tests
# --------------------------------------------------------------------- #


def test_router_heuristic_routes():
    """classify(): action verbs + keyword cues -> ACTION with the right
    categories; plain talk -> CHAT. Category sets are asserted as the
    REQUIRED subset — the router legitimately adds adjacent categories
    (e.g. 'channel' also maps 'channels', 'ping' also maps 'roles')."""
    action_cases = [
        # utterance, categories that MUST be present
        ("ping Aliyyan in the chill zone channel and say hi",
         {"messaging", "members"}),
        ("join my vc", {"voice"}),
        ("send 3 gifs in memes", {"media", "messaging"}),
        ("bump the server", {"bump"}),
        ("delete my last message", {"messaging"}),
    ]
    for text, need in action_cases:
        r = classify(text)
        assert r is not None, f"{text!r}: heuristic returned None (ambiguous)"
        assert r.kind == ROUTE_ACTION, f"{text!r}: routed {r.kind}, want action"
        assert need <= r.categories, (
            f"{text!r}: categories {sorted(r.categories)} missing {need}"
        )
        assert r.via == "heuristic"
        print(f"    action: {text!r} -> {sorted(r.categories)}")

    for text in ("yo whats good", "what do you think about it"):
        r = classify(text)
        assert r is not None and r.kind == ROUTE_CHAT, (
            f"{text!r}: routed {r and r.kind}, want chat"
        )
        print(f"    chat:   {text!r}")

    # cue word but no action verb -> None (goes to the LLM arbiter).
    # NOTE: "that thing we talked about" has NO cue words at all, so it
    # resolves CHAT heuristically and never reaches the arbiter.
    assert classify("that thing we talked about").kind == ROUTE_CHAT
    assert classify("that channel over there") is None
    assert classify("the voice channel") is None
    print("    ambiguous utterances -> None (arbiter path)")


def test_filtered_tool_schemas():
    """build_tools('command', categories) ships only the matching category
    tools plus the universal resolvers — the action-path token saving."""
    tools = build_tools("command", {"messaging", "members"})
    names = {t["function"]["name"] for t in tools}

    must_have = {
        "send_message", "send_dm", "search_members", "get_member_mention",
    } | _RESOLVER_TOOLS
    assert must_have <= names, f"missing: {sorted(must_have - names)}"

    for excluded in ("bump_all", "send_gif", "list_slash_commands",
                     "bump_with_bot", "find_bump_commands", "join_voice",
                     "use_slash_command", "change_nickname"):
        assert excluded not in names, f"{excluded} leaked into filtered set"

    assert len(tools) < 30, f"filtered set too big: {len(tools)}"
    print(f"    filtered set: {len(tools)} tools "
          f"(vs {len(build_tools('command'))} full)")


async def test_agent_no_tools_path():
    """CHAT_MODE + no_tools -> pool.chat called ONCE with tools=None,
    canned content returned verbatim."""
    pool = FakePool([_resp(content="not much, you?")])
    agent = _make_agent(pool)

    out = await agent.run("yo", mode=CHAT_MODE, no_tools=True)

    assert out == "not much, you?", out
    assert len(pool.calls) == 1, f"expected 1 llm call, got {len(pool.calls)}"
    call = pool.calls[0]
    assert call["tools"] is None
    assert call["tool_choice"] == "none"
    print(f"    no_tools: tools={call['tools']}, reply={out!r}")


async def test_agent_categories_path():
    """COMMAND_MODE + categories={'messaging'} -> pool.chat receives a
    filtered tools list (messaging + resolvers only)."""
    pool = FakePool([_resp(content="done")])
    agent = _make_agent(pool)

    out = await agent.run("say hi in general", mode=COMMAND_MODE,
                          categories={"messaging"})

    assert out == "done"
    tools = pool.calls[0]["tools"]
    assert tools, "categories path sent no tools"
    names = {t["function"]["name"] for t in tools}
    allowed = {n for n, c in _CATEGORIES.items() if c == "messaging"} \
        | _RESOLVER_TOOLS
    assert names <= allowed, f"unexpected tools leaked: {sorted(names - allowed)}"
    assert "send_message" in names
    assert "bump_all" not in names
    print(f"    categories={{'messaging'}} -> {len(names)} tools")


async def test_empty_content_safety():
    """Empty model content: voice mode -> '' (silence — never spoken);
    chat mode -> '(no reply)' status string."""
    pool = FakePool([_resp(content="")])
    agent = _make_agent(pool)
    out = await agent.run("hi", mode=VOICE_MODE, no_tools=True)
    assert out == "", f"voice mode returned {out!r}, want '' (silence)"
    print("    voice empty content -> '' (stays silent)")

    pool2 = FakePool([_resp(content="")])
    agent2 = _make_agent(pool2)
    out2 = await agent2.run("hi", mode=CHAT_MODE, no_tools=True)
    assert out2 == "(no reply)", f"chat mode returned {out2!r}"
    print("    chat empty content -> '(no reply)'")


async def test_tool_call_loop():
    """Round 1 returns a send_message tool_call -> dispatch runs once with
    parsed args -> round 2 plain content is the final reply."""
    args = {"channel_query": "general", "content": "hi"}
    pool = FakePool([
        _resp(tool_calls=[_tool_call("send_message", args)]),
        _resp(content="done, sent it"),
    ])
    agent = _make_agent(pool)

    dispatched: list[tuple[str, dict]] = []

    async def fake_dispatch(ctx, name, args_):
        dispatched.append((name, args_))
        return json.dumps({"ok": True})

    with mock.patch("ai.agent.dispatch", new=fake_dispatch):
        out = await agent.run("say hi in general", mode=COMMAND_MODE)

    assert dispatched == [("send_message", args)], dispatched
    assert out == "done, sent it", out
    assert len(pool.calls) == 2
    # the tool result was fed back as a 'tool' role message in round 2
    round2_msgs = pool.calls[1]["messages"]
    tool_msgs = [m for m in round2_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "send_message"
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    print(f"    dispatch({dispatched[0][0]}, {dispatched[0][1]}) -> {out!r}")


async def test_arbiter_routing():
    """classify_llm: ambiguous utterance (cue word, no action verb) hits
    the fake arbiter; 'ACTION' verdict -> Route(action) with cue-derived
    categories, 'CHAT' verdict -> Route(chat). Arbiter errors -> chat
    fallback."""
    # 'that channel over there' matches the channels cue but has no
    # action verb -> heuristic returns None -> arbiter fires.
    pool = FakePool([_resp(content="ACTION")])
    with mock.patch("ai.router.get_pool", return_value=pool):
        r = await classify_llm("that channel over there")
    assert len(pool.calls) == 1, "arbiter was never called"
    arb_call = pool.calls[0]
    assert arb_call["tools"] is None and arb_call["tool_choice"] == "none"
    assert arb_call["max_tokens"] == 8
    assert r.kind == ROUTE_ACTION and r.via == "arbiter"
    assert "channels" in r.categories
    print(f"    arbiter ACTION -> Route({r.kind}, {sorted(r.categories)})")

    # CHAT verdict -> conversation route.
    pool2 = FakePool([_resp(content="CHAT")])
    with mock.patch("ai.router.get_pool", return_value=pool2):
        r2 = await classify_llm("the voice channel")
    assert len(pool2.calls) == 1
    assert r2.kind == ROUTE_CHAT and r2.via == "arbiter"
    print(f"    arbiter CHAT -> Route({r2.kind})")

    # Unambiguous text never reaches the arbiter (0 llm calls).
    pool3 = FakePool([_resp(content="ACTION")])
    with mock.patch("ai.router.get_pool", return_value=pool3):
        r3 = await classify_llm("bump the server")
    assert pool3.calls == [], "heuristic hit leaked into the arbiter"
    assert r3.kind == ROUTE_ACTION and r3.via == "heuristic"
    print("    unambiguous action skipped the arbiter")

    # Arbiter crash -> safe chat fallback (via='fallback').
    pool4 = FakePool([RuntimeError("arbiter boom")])

    async def boom(**kwargs):
        raise RuntimeError("arbiter boom")
    pool4.chat = boom
    pool4.calls.clear()
    with mock.patch("ai.router.get_pool", return_value=pool4):
        r4 = await classify_llm("that channel over there")
    assert r4.kind == ROUTE_CHAT and r4.via == "fallback"
    print("    arbiter exception -> Route(chat, via='fallback')")


# --------------------------------------------------------------------- #
#  runner
# --------------------------------------------------------------------- #

# Tests that document genuine bugs — they assert the EXPECTED behaviour so
# they fail today and pass once the source is fixed.
KNOWN_BUGS: dict = {}


async def _main() -> int:
    tests = [
        test_router_heuristic_routes,
        test_filtered_tool_schemas,
        test_agent_no_tools_path,
        test_agent_categories_path,
        test_empty_content_safety,
        test_tool_call_loop,
        test_arbiter_routing,
    ]
    results: dict[str, str] = {}
    for fn in tests:
        name = fn.__name__
        print(f"--- {name}")
        try:
            out = fn()
            if asyncio.iscoroutine(out):
                await out
            if name in KNOWN_BUGS:
                print(f"[PASS] {name}  (documented bug appears FIXED)")
            else:
                print(f"[PASS] {name}")
            results[name] = "PASS"
        except AssertionError as e:
            if name in KNOWN_BUGS:
                print(f"[BUG ] {name}: expected-behaviour check failed ->")
                print(f"        {e}")
                print(f"        source bug: {KNOWN_BUGS[name]}")
                results[name] = "BUG"
            else:
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc(limit=3)
                results[name] = "FAIL"
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {name}: {e!r}")
            traceback.print_exc(limit=5)
            results[name] = "ERROR"

    print("\n================ summary ================")
    for n, st in results.items():
        tag = {"PASS": "PASS", "BUG": "BUG (documented source bug)"}.get(st, st)
        print(f"  {tag:<32} {n}")
    fails = [n for n, st in results.items() if st in ("FAIL", "ERROR")]
    bugs = [n for n, st in results.items() if st == "BUG"]
    print(f"\n{sum(1 for s in results.values() if s == 'PASS')} passed, "
          f"{len(bugs)} documented source bug(s), {len(fails)} unexpected failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
