#!/usr/bin/env python3
"""Stage-level latency breakdown for the Engager voice pipeline.

Parses loguru logs in logs/, pairs utterance -> transcript -> tokens/reply
-> say chains, and reports per-stage latency, TTS backend distribution,
noise rate, fragmentation, and bottlenecks.

Usage:  python data/_latency_analysis.py            (all files)
        python data/_latency_analysis.py --newest   (newest file only)
"""
from __future__ import annotations

import re
import sys
import statistics
from pathlib import Path
from collections import deque, Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
ENV_FILE = ROOT / ".env"
RESPONSIVE_WINDOW_S = 30.0   # transcript->reply pairing window
TTS_WINDOW_S = 180.0         # reply->say pairing window

LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| \w+\s* \| "
    r"[\w.]+:\w+:\d+ - (.*)$"
)
RE_UTT = re.compile(r"VC utterance: (.+?) ([\d.]+)s voiced")
RE_TRAN = re.compile(r"VC transcript(?: \(continuation\))?: (.*)")
RE_NAME = re.compile(r"\[([^\]]{1,40})\]:")
RE_EMPTY = re.compile(r"STT empty/failed for (.+?) \(([\d.]+)s\)")
RE_NOISE = re.compile(r"Noise rejected: (.*)")
RE_ECHO = re.compile(r"Echo rejected: (.*)")
RE_TOK = re.compile(r"Tokens \(voice r(\d+)\): prompt=(\d+) "
                    r"completion=(\d+) total=(\d+)")
RE_REPLY = re.compile(r"Agent (?:final|fallback) reply \(voice\): ?(.*)")
RE_SAY = re.compile(r"VC say \(([\d.]+)s tts\): ?(.*)")
RE_STREAM = re.compile(r"VC stream first-audio ([\d.]+)s: ?(.*)")
RE_BARGE = re.compile(r"Barge-in by (.+?)(?:\s*—|$)")
RE_TTSF = re.compile(r"TTS backend '(\w+)' failed")
RE_TTSD = re.compile(r"TTS backend '(\w+)' disabled")
RE_ALLF = re.compile(r"All TTS backends failed")
RE_LISTEN = re.compile(r"Voice session: listening")


def ts(s: str) -> float:
    from datetime import datetime
    d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    return d.timestamp()


def tts_chain() -> list[str]:
    if ENV_FILE.exists():
        for ln in ENV_FILE.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
            if ln.startswith("VOICE_TTS_BACKENDS"):
                return [p.strip().lower() for p in
                        ln.split("=", 1)[1].strip('[]"').split(",")
                        if p.strip()]
    return ["orpheus", "edge"]


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def row(name, vals):
    if not vals:
        print(f"  {name:<38} n=0")
        return
    print(f"  {name:<38} n={len(vals):<4} p50={statistics.median(vals):6.2f}s"
          f"  mean={statistics.mean(vals):6.2f}s  p90={pct(vals, .9):6.2f}s"
          f"  p99={pct(vals, .99):6.2f}s  max={max(vals):6.2f}s")


def main() -> None:
    newest_only = "--newest" in sys.argv
    files = sorted(LOG_DIR.glob("engager_*.log"),
                   key=lambda p: p.stat().st_mtime)
    if newest_only:
        files = files[-1:]
    print(f"Parsing {len(files)} log file(s):")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    chain = tts_chain()
    print(f"TTS chain (from .env): {chain}\n")

    events = []            # (ts, kind, payload, file)
    for fp in files:
        for raw in fp.read_text(encoding="utf-8",
                                errors="replace").splitlines():
            m = LINE_RE.match(raw)
            if not m:
                continue
            t, msg = ts(m.group(1)), m.group(2)
            for kind, rx in (("utt", RE_UTT), ("tran", RE_TRAN),
                             ("empty", RE_EMPTY), ("noise", RE_NOISE),
                             ("echo", RE_ECHO), ("tok", RE_TOK),
                             ("reply", RE_REPLY), ("say", RE_SAY),
                             ("stream", RE_STREAM), ("barge", RE_BARGE),
                             ("ttsf", RE_TTSF), ("ttsd", RE_TTSD),
                             ("allf", RE_ALLF), ("listen", RE_LISTEN)):
                mm = rx.search(msg)
                if mm:
                    events.append((t, kind, mm, fp.name))
                    break
    events.sort(key=lambda e: e[0])

    # ---------------- pairing state ----------------
    pending = deque()              # utterances awaiting a disposition
    open_trans = deque()           # transcripts awaiting a reply
    last_reply = None              # reply awaiting a say
    disabled = set()               # benched TTS backends
    fails_since = []               # (ts, backend) failure feed
    turns, orphan_says = [], []
    outcomes = Counter()           # transcript/empty/noise/echo/lost
    say_records, barge_count = [], 0
    stream_firsts = []
    session_starts = 0

    def resolve_utt(t, name, outcome, dur=None):
        """Match an utterance to its disposition (same name, exact dur)."""
        best = None
        for u in pending:
            if u["done"]:
                continue
            if name and u["name"] != name:
                continue
            if dur is not None and abs(u["dur"] - dur) > 0.06:
                continue
            best = u
            break
        if best is None:           # fallback: oldest unresolved
            best = next((u for u in pending if not u["done"]), None)
        if best is None:
            return None
        best["done"], best["outcome"], best["out_ts"] = True, outcome, t
        outcomes[outcome] += 1
        return best

    for t, kind, m, f in events:
        if kind == "listen":
            session_starts += 1
            disabled.clear()       # new session/process -> fresh bench set
        elif kind == "utt":
            pending.append({"ts": t, "name": m.group(1),
                            "dur": float(m.group(2)), "done": False,
                            "file": f})
        elif kind == "empty":
            resolve_utt(t, m.group(1), "empty", float(m.group(2)))
        elif kind == "noise":
            resolve_utt(t, None, "noise")
        elif kind == "echo":
            resolve_utt(t, None, "echo")
        elif kind == "tran":
            names = RE_NAME.findall(m.group(1))
            utts = [resolve_utt(t, n, "transcript") for n in names]
            if not names:
                utts = [resolve_utt(t, None, "transcript")]
            utts = [u for u in utts if u]
            open_trans.append({"ts": t, "utts": utts,
                               "text": m.group(1), "tok": [], "file": f})
        elif kind == "tok":
            if open_trans:
                open_trans[-1]["tok"].append(t)
        elif kind == "reply":
            ev = {"ts": t, "text": m.group(1), "tran": None, "file": f}
            if (open_trans and t - open_trans[0]["ts"]
                    <= RESPONSIVE_WINDOW_S):
                ev["tran"] = open_trans.popleft()
            last_reply = ev
        elif kind == "say":
            synth = float(m.group(1))
            win_lo = (last_reply["ts"] if last_reply else
                      (say_records[-1]["ts"] if say_records else t - 60))
            fails = {b for ft, b in fails_since if win_lo <= ft <= t}
            backend = next((b for b in chain
                            if b not in fails and b not in disabled),
                           chain[-1])
            rec = {"ts": t, "synth": synth, "text": m.group(2),
                   "reply": last_reply, "backend": backend,
                   "fails": sorted(fails), "file": f}
            if last_reply and t - last_reply["ts"] <= TTS_WINDOW_S:
                turns.append(rec)
            else:
                orphan_says.append(rec)
            say_records.append(rec)
            last_reply = None
        elif kind == "stream":
            stream_firsts.append(float(m.group(1)))
            say_records.append({"ts": t, "synth": float(m.group(1)),
                                "text": m.group(2), "reply": last_reply,
                                "backend": "stream", "file": f})
            last_reply = None
        elif kind == "barge":
            barge_count += 1
            if say_records:
                say_records[-1]["barge"] = True
        elif kind == "ttsf":
            fails_since.append((t, m.group(1)))
        elif kind == "ttsd":
            disabled.add(m.group(1))
        elif kind == "allf":
            say_records.append({"ts": t, "synth": None, "all_failed": True,
                                "reply": last_reply, "file": f})

    lost = [u for u in pending if not u["done"]]
    outcomes["lost_no_disposition"] = len(lost)

    # ---------------- report ----------------
    stt_g, llm_g, tts_g, tot_g = [], [], [], []
    q_stall = 0.0
    tran_len_vs_stt, tran_len_vs_llm = [], []
    reply_chars_vs_tts = []
    for rec in turns:
        tr = (rec["reply"] or {}).get("tran")
        tts_g.append(rec["ts"] - rec["reply"]["ts"])
        reply_chars_vs_tts.append((len(rec["text"]), rec["synth"]))
        if tr:
            llm_g.append(rec["reply"]["ts"] - tr["ts"])
            tran_len_vs_llm.append((len(tr["text"]),
                                    rec["reply"]["ts"] - tr["ts"]))
            for u in tr["utts"]:
                stt_g.append(tr["ts"] - u["ts"])
                tran_len_vs_stt.append((u["dur"], tr["ts"] - u["ts"]))
                if tr["ts"] - u["ts"] > 10:
                    q_stall += tr["ts"] - u["ts"]
            if tr["utts"]:
                tot_g.append(rec["ts"] - tr["utts"][-1]["ts"])
    for u in pending:
        if u["done"] and u["outcome"] in ("empty", "noise", "echo"):
            if u["out_ts"] - u["ts"] < 120:
                stt_g.append(u["out_ts"] - u["ts"])

    total_utt = outcomes["transcript"] + outcomes["empty"] + \
        outcomes["noise"] + outcomes["echo"] + outcomes["lost_no_disposition"]

    print("=" * 78)
    print("1) STAGE GAPS  (utterance end -> ... -> playback start)")
    print("=" * 78)
    row("utterance->transcript (STT leg)", stt_g)
    row("transcript->reply (delay+router+LLM)", llm_g)
    row("reply->say-start (TTS synth wait)", tts_g)
    row("TOTAL utterance->say-start (perceived)", tot_g)
    if stream_firsts:
        row("streamed first-audio (new path)", stream_firsts)
    print(f"\n  Responsive chains: {len(turns)} | orphan/proactive says:"
          f" {len(orphan_says)} | sessions: {session_starts} |"
          f" barge-ins: {barge_count}")
    stall = [g for g in stt_g if g > 10]
    print(f"  Queue-stalled utterances (STT leg >10s): {len(stall)}"
          f"  ({q_stall:.0f}s of dead queue time)")

    print("\n" + "=" * 78)
    print("2) TTS SYNTH TIME PER BACKEND  (inferred from failure windows)")
    print("=" * 78)
    by_be = defaultdict(list)
    for r in say_records:
        if r.get("synth") is not None and r["synth"] > 0.01:
            by_be[r.get("backend", "?")].append(r["synth"])
    for b, v in sorted(by_be.items()):
        row(f"{b:<38}", v)
    fails = Counter(b for _, b in fails_since)
    print(f"  Backend failures: {dict(fails)} | 'all failed' events:"
          f" {sum(1 for r in say_records if r.get('all_failed'))}")
    print(f"  Says interrupted by barge-in:"
          f" {sum(1 for r in say_records if r.get('barge'))}")
    zero = sum(1 for r in say_records
               if r.get("synth") is not None and r["synth"] <= 0.01)
    print(f"  Synthetic '0.00s tts' lines (test data, excluded): {zero}")

    print("\n" + "=" * 78)
    print("3) TRANSCRIPT/UTTERANCE LENGTH vs STAGE TIME")
    print("=" * 78)
    for label, pairs in (("utterance voiced_s -> STT gap", tran_len_vs_stt),
                         ("transcript chars -> LLM gap", tran_len_vs_llm),
                         ("reply chars -> TTS synth", reply_chars_vs_tts)):
        if len(pairs) < 8:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        sx = (sum((x - mx) ** 2 for x in xs) or 1) ** .5
        sy = (sum((y - my) ** 2 for y in ys) or 1) ** .5
        r = cov / (sx * sy)
        med = statistics.median(xs)
        lo = [y for x, y in pairs if x <= med]
        hi = [y for x, y in pairs if x > med]
        print(f"  {label:<34} pearson r={r:+.2f} | "
              f"short<=med: {statistics.median(lo):.2f}s | "
              f"long>med:   {statistics.median(hi):.2f}s")

    print("\n" + "=" * 78)
    print("4) NOISE / DROP RATE")
    print("=" * 78)
    print(f"  utterances seen: {total_utt}")
    for k in ("transcript", "empty", "noise", "echo",
              "lost_no_disposition"):
        n = outcomes[k]
        print(f"    {k:<22} {n:>4}  ({n / max(total_utt, 1) * 100:5.1f}%)")

    print("\n" + "=" * 78)
    print("5) UTTERANCE FRAGMENTATION")
    print("=" * 78)
    durs = [u["dur"] for u in pending]
    if durs:
        row("utterance voiced_s", durs)
        print(f"  <1s voiced: {sum(1 for d in durs if d < 1)}"
              f" ({sum(1 for d in durs if d < 1) / len(durs) * 100:.0f}%) |"
              f" >=10s (monologue/15s force-cut):"
              f" {sum(1 for d in durs if d >= 10)}")

    print("\n" + "=" * 78)
    print("BOTTLENECKS (share of median perceived total)")
    print("=" * 78)
    med_tot = statistics.median(tot_g) if tot_g else 0
    for name, v in (("STT leg", stt_g), ("LLM leg", llm_g),
                    ("TTS leg", tts_g)):
        if v and med_tot:
            print(f"  {name:<8} median {statistics.median(v):.2f}s "
                  f"= {statistics.median(v) / med_tot * 100:.0f}% of total")
    print(f"  queue-stall waste: {q_stall:.0f}s across "
          f"{len(stall)} utterances")


if __name__ == "__main__":
    main()
