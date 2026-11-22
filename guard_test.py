#!/usr/bin/env python3
"""guard_test.py — a write must not fire on a remark.

The measured failure this pins down: STT heard "This is the first office fan on."
(a remark about the fan), the model called the write tool, and the fan was
switched on. Two layers now stand between a sentence and a device:

  1. the gate — `ha_do` is only *offered* when `looks_imperative(utterance)` is
     true, so a model cannot choose a tool it was never given;
  2. the dispatcher — `_run_ha_tool` refuses `ha_do` when the utterance is not an
     instruction, even if a model calls it anyway (swapped model, changed prompt,
     a future path that offers the full tool set).

Layer 2 exists because layer 1 is an assumption about model behaviour, and
`toolset_bench.py` measured that assumption failing: with every tool offered on
every turn, qwen2.5:3b chose `ha_do` on a question and on a remark in 4 of 28
turns. It never gets the chance here — but the check is cheap and it is the
difference between a wrong answer and a device changing state.

    python3 guard_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bridge import Bridge, SYSTEM_PROMPT
from ha import looks_imperative, wants_ha_tools

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label:62s}" + (f" [{detail}]" if detail else ""))
    failures += 0 if ok else 1


def args_namespace(**kw):
    base = dict(
        ollama="http://127.0.0.1:11434", model="qwen2.5:3b", system_prompt=SYSTEM_PROMPT,
        max_tokens=120, lane_timeout=25, cloud_model="voice-escalation",
        cloud_url="http://127.0.0.1:20129/v1", cloud_key=None, cloud_max_tokens=80,
        cloud_system_prompt=None, cloud_name="Muse", local_name="Zion",
        muse=False, bus_url="", bus_token="", muse_agent="muse", muse_name="Muse",
        bus_type="voice-handoff", device_name="Satellite-1", answer_lane="local",
        stt=False, llm=True, tts=False, whisper_model="base.en", initial_prompt="",
        threads=8, piper_voice="", outdir="captures", media_port=8701, media_host=None,
        host="127.0.0.1", port=16001, seconds=0, once=True, silence_ms=1200,
        min_speech_ms=160, verbose=False, handoff="off", ack_text=None,
        attribution="off", handoff_poll_s=5, handoff_max_attempts=12,
        announce_timeout=30, handoff_ttl_s=900, ha=False, ha_url="", ha_token="",
        memory="off", memory_turns=3, memory_ttl_s=300.0, verify="off",
        verify_max_tokens=400, music=False, ma_url="", ma_token="",
        music_target="Satellite-1", stream_tts=False, fastpath="on", clock_tool=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class FakeHa:
    """Counts calls. Anything that reaches it would have changed a device."""

    def __init__(self):
        self.calls = []

    def lookup(self, query, limit=3):
        return []

    def do(self, query, service=""):
        self.calls.append((query, service))
        return {"ok": True, "service": service, "entity_id": query, "was": "off"}


async def main() -> int:
    bridge = Bridge(args_namespace())
    bridge.ha = FakeHa()

    # 1. the gate: nothing but handoff is offered for a remark or a question
    for utterance in ("This is the first office fan on.", "Is the patio light on?",
                      "The fan is on.", "Do you turn off the light?"):
        # ask the gate directly, in the order _answer asks it
        gated = []
        if wants_ha_tools(utterance):
            gated.append("ha_lookup")
            if looks_imperative(utterance):
                gated.append("ha_do")
        check(f"gate withholds ha_do: {utterance[:40]!r}", "ha_do" not in gated,
              f"offered {gated}")

    # 2. the dispatcher: even handed the tool by name, it refuses
    for utterance in ("This is the first office fan on.", "Is the patio light on?",
                      "the fan is on", ""):
        result = bridge._run_ha_tool(1, "ha_do", {"query": "office fan", "service": "turn_on"},
                                     utterance)
        check(f"dispatcher refuses: {utterance[:40]!r}",
              result.get("ok") is False and not bridge.ha.calls,
              f"{result.get('error', '')[:40]}")

    # 3. and a real instruction still goes through (the guard must not block work)
    result = bridge._run_ha_tool(1, "ha_do", {"query": "office fan", "service": "turn_off"},
                                 "Turn off the office fan.")
    check("real instruction still writes",
          result.get("ok") is True and len(bridge.ha.calls) == 1,
          f"{bridge.ha.calls}")

    # 4. reads are never gated: a question may read
    result = bridge._run_ha_tool(1, "ha_lookup", {"query": "patio light"}, "Is the patio light on?")
    check("reads are never gated", result.get("ok") is True)

    await bridge.brain.lanes.aclose()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


sys.exit(asyncio.run(main()))
