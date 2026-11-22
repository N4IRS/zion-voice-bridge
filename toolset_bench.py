#!/usr/bin/env python3
"""toolset_bench.py — should the tool gates stay? Measured, not argued.

The gates (`wants_ha_tools`, `looks_imperative`, `wants_music_tools`) exist to keep the
prompt small, on the theory that ~60-80 extra schema tokens cost ~2 s of prompt eval on
this CPU. Prefix caching undercuts that theory: Ollama keeps the prompt prefix warm, so
a schema costs once and is free afterwards — while a *changing* tool set invalidates the
prefix and pays again.

So the gates are only worth keeping if they buy accuracy. This measures both:

    GATED  = what the shipped code offers for this utterance
    FIXED  = every tool, every turn

For each utterance it records which tool the model chose, whether that was right, and
the prompt-eval cost — including "offered nothing" cases, because a model that reaches
for a tool it was not given is the failure that produced "handoff needs data".

    python3 toolset_bench.py                 # 2 runs per utterance per mode
    python3 toolset_bench.py --runs 3 --mode fixed
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

from bridge import (CLOCK_TOOL, DO_TOOL, HANDOFF_TOOL, LOOKUP_TOOL,
                    SYSTEM_PROMPT)
from ha import looks_imperative, wants_ha_tools
from ma import wants_music_tools, MusicAssistant

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b"

#: the music schemas, taken from the module rather than copied
MUSIC_TOOLS = MusicAssistant.tools()

#: (utterance, the tool a correct answer needs, tools it must NEVER get, note)
#:
#: `must` is what a good turn looks like. `must_not` is what makes a turn
#: dangerous: a write chosen for a question or a statement. Those are scored
#: separately, because "picked the wrong read" and "opened the garage door when
#: he remarked that it was open" are not the same kind of error.
CASES = [
    # no tool: language, explanation, or something no tool can answer
    ("What is a balun?", None, (), "pure knowledge"),
    ("Explain why a folded dipole has a 4 to 1 impedance step up.", None, (), "explanation"),
    # reads
    ("How warm is the office?", "ha_lookup", ("ha_do",), "sensor read"),
    ("Is the patio light on?", "ha_lookup", ("ha_do",), "question, must not write"),
    # writes, and only on an instruction
    ("Turn off the office fan.", "ha_do", (), "imperative write"),
    ("Turn the patio light on.", "ha_do", (), "imperative write"),
    ("Switch off the garage light.", "ha_do", (), "imperative write"),
    ("This is the first office fan on.", None, ("ha_do",), "statement, must not write"),
    # music
    ("Play pink floyd.", "music_play", (), "music play"),
    ("Play pink floyd on the esp32.", "music_play", (), "music play, target"),
    ("Pause the music.", "music_transport", (), "transport"),
    ("Stop the music.", "music_transport", (), "transport"),
    # the owner's records: only Muse has these, so handoff is the only correct tool
    ("What's on my list for tomorrow?", "handoff", (), "owner's records"),
    ("Where did we leave the duplexer order?", "handoff", (), "owner's records"),
]

#: never reaches a model in production (the calc fast path answers it), so it is
#: printed but not scored. Kept because the tool-syntax-in-prose failure it shows
#: is the one that produced "handoff needs data" on the Dot.
NOT_SCORED = [
    ("What is the SWR of a 3 to 1 mismatch doing to my power?", "fast path owns it"),
]

#: tools a gated turn may offer. `clock_tool` is off in production, so host_clock is not
#: in either mode; the fast path answers clock questions before the model is reached.
ALL_TOOLS = HANDOFF_TOOL + [LOOKUP_TOOL, DO_TOOL] + MUSIC_TOOLS


def gated_tools(text: str) -> list[dict]:
    """Exactly what the shipped `_answer` would offer this utterance."""
    tools: list[dict] = list(HANDOFF_TOOL)          # handoff policy is 'auto' by default
    if wants_ha_tools(text):
        tools.append(LOOKUP_TOOL)
        if looks_imperative(text):
            tools.append(DO_TOOL)
    if wants_music_tools(text):
        tools.extend(MUSIC_TOOLS)
    return tools


def ask(text: str, tools: list[dict]) -> tuple[list[str], float, int, int]:
    payload = {
        "model": MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        "options": {"temperature": 0.0, "num_predict": 120},
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    elapsed = time.monotonic() - started
    message = data.get("message") or {}
    chosen = [((c.get("function") or {}).get("name") or "")
              for c in (message.get("tool_calls") or [])]
    prose = (message.get("content") or "")
    if not chosen and ("needs=" in prose or "needs =" in prose):
        chosen = ["<tool-syntax-in-prose>"]
    return chosen, elapsed, data.get("prompt_eval_count") or 0, \
        data.get("prompt_eval_duration") or 0


def run(mode: str, runs: int) -> None:
    correct = total = danger = prose = 0
    latencies: list[float] = []
    prompt_evals: list[float] = []
    print(f"\n=== {mode.upper()} ({runs} run(s) per utterance) ===")
    print(f"{'utterance':44s} {'expect':16s} {'got':20s} verdict")
    for text, expect, must_not, note in CASES:
        tools = ALL_TOOLS if mode == "fixed" else gated_tools(text)
        offered = {(t.get("function") or {}).get("name") for t in tools}
        for _ in range(runs):
            chosen, elapsed, prompt_tokens, prompt_ns = ask(text, tools)
            got = chosen[0] if chosen else "none"
            total += 1
            latencies.append(elapsed)
            prompt_evals.append(prompt_ns / 1e9)
            if got == "<tool-syntax-in-prose>":
                verdict = "PROSE"
                prose += 1
            elif got in must_not:
                verdict = "DANGER: write on a non-instruction"
                danger += 1
            elif got == (expect or "none"):
                verdict = "ok"
                correct += 1
            elif got == "none" and expect:
                verdict = "MISS: no tool, answered unverified"
            elif expect is None:
                verdict = f"extra read ({got}) - harmless but unneeded"
            else:
                verdict = "MISS: wrong tool"
            withheld = ("  [NOT OFFERED]"
                        if got not in offered and got != "none" else "")
            print(f"{text[:42]:44s} {str(expect or 'none'):16s} {got[:18]:20s} "
                  f"{verdict}{withheld}  {elapsed:4.1f}s "
                  f"(prompt {prompt_tokens}t/{prompt_ns/1e9:.2f}s)")
    print(f"  accuracy {correct}/{total} = {correct/total:.0%}   "
          f"dangerous writes {danger}/{total}   tool-syntax prose {prose}/{total}")
    print(f"  median {statistics.median(latencies):.2f}s   "
          f"median prompt eval {statistics.median(prompt_evals):.2f}s")
    for text, note in NOT_SCORED:
        tools = ALL_TOOLS if mode == "fixed" else gated_tools(text)
        chosen, elapsed, prompt_tokens, prompt_ns = ask(text, tools)
        print(f"{text[:42]:44s} {'(unscored)':16s} "
              f"{(chosen[0] if chosen else 'none')[:18]:20s} {note}")


def flip(runs: int) -> None:
    """Does a changing tool set cost a prompt rebuild? The C hypothesis said yes.

    Ollama keeps recently used prompt prefixes in KV cache, so a prefix that
    repeats is cheap and a prefix that was just evicted is not. The first attempt
    at this measured the wrong thing: it primed with one set and then measured the
    same set twice, so its "alternating" group was really a warm repeat.

    Here every measured call is preceded by a prime call with a *known* prefix:

      repeat  = prime with the same set   -> the prefix should be warm
      flip    = prime with the other set  -> the prefix was just evicted

    and both are done for the small (gated) and the large (all-tools) set, so a
    size effect cannot masquerade as a flip effect.
    """
    text = "Turn off the office fan."
    small = gated_tools(text)          # handoff + ha_lookup + ha_do
    large = ALL_TOOLS                  # every tool
    groups: dict[str, list[tuple[float, float, int]]] = {
        "small, repeat": [], "small, after flip": [],
        "large, repeat": [], "large, after flip": [],
    }
    for _ in range(runs):
        for label, tools, primer in (
                ("small, repeat", small, small),
                ("small, after flip", small, large),
                ("large, repeat", large, large),
                ("large, after flip", large, small)):
            ask(text, primer)
            _, elapsed, tokens, ns = ask(text, tools)
            groups[label].append((elapsed, ns / 1e9, tokens))

    print(f"\n=== FLIP: does a changing tool set cost a prompt rebuild? ({runs} each) ===")
    print(f"{'measured call':22s} {'prompt tokens':14s} {'prompt eval':12s} {'total':8s}")
    for label, rows in groups.items():
        tokens = statistics.median(r[2] for r in rows)
        peval = statistics.median(r[1] for r in rows)
        total = statistics.median(r[0] for r in rows)
        print(f"{label:22s} {tokens:14.0f} {peval:11.3f}s {total:7.2f}s")
    print("  flip penalty (small): "
          f"{statistics.median(r[1] for r in groups['small, after flip']) - statistics.median(r[1] for r in groups['small, repeat']):+.3f}s")
    print("  size penalty (warm) : "
          f"{statistics.median(r[1] for r in groups['large, repeat']) - statistics.median(r[1] for r in groups['small, repeat']):+.3f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--mode", choices=("gated", "fixed", "both", "flip", "all"),
                   default="both")
    args = p.parse_args()
    if args.mode in ("gated", "both", "all"):
        run("gated", args.runs)
    if args.mode in ("fixed", "both", "all"):
        run("fixed", args.runs)
    if args.mode in ("flip", "all"):
        flip(args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
