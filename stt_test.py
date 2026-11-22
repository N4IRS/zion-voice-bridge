#!/usr/bin/env python3
"""stt_test.py — the confidence gate that re-hears a bad transcript.

Measured failure this covers (2026-11-22): the Dot heard "Ask Muse what's on my
schedule." and base.en transcribed "ASM Muse was not much schedule." The local
model then invented "ASM Muse had no scheduled activities today." — a confident
answer to a sentence nobody said. small.en gets the sentence exactly right.

So transcription runs twice only when the first pass is unsure. The gate is the
model's own worst-segment logprob:

    good transcripts on this bench   -0.40 .. -0.63
    the misheard one                 -0.885

That is two turns of evidence, not a calibration, so the threshold is a flag
(--stt-confidence) and this test pins the *logic* plus, when the capture still
exists, the real re-hearing. `captures/` is gitignored because it holds voice
audio, so the live half skips rather than failing on a clean checkout.

    python3 stt_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from bridge import Bridge, SYSTEM_PROMPT

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label:58s}" + (f" [{detail}]" if detail else ""))
    failures += 0 if ok else 1


def ns(**kw):
    base = dict(
        ollama="http://127.0.0.1:11434", model="qwen2.5:3b", system_prompt=SYSTEM_PROMPT,
        max_tokens=120, lane_timeout=25, cloud_model=None, cloud_url="", cloud_key=None,
        cloud_max_tokens=80, cloud_system_prompt=None, cloud_name="Muse", local_name="Zion",
        muse=False, bus_url="", bus_token="", muse_agent="muse", muse_name="Muse",
        bus_type="voice-handoff", device_name="Satellite-1", answer_lane="local",
        stt=False, llm=True, tts=False, whisper_model="base.en", initial_prompt="",
        threads=8, piper_voice="", outdir="captures", media_port=8700,
        media_host="192.168.190.4", host="127.0.0.1", port=16001, seconds=0, once=True,
        silence_ms=1200, min_speech_ms=160, verbose=False, handoff="off", ack_text=None,
        attribution="off", handoff_poll_s=5, handoff_max_attempts=12, announce_timeout=30,
        handoff_ttl_s=900, ha=False, ha_url="", ha_token="", memory="off", memory_turns=3,
        memory_ttl_s=300.0, verify="off", verify_max_tokens=400, music=False, ma_url="",
        ma_token="", music_target="Satellite-1", stream_tts=False, fastpath="on",
        clock_tool=False, whisper_fallback="small.en", stt_confidence=-0.75,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def gate_cases():
    """The decision, with no model involved."""
    bridge = Bridge(ns(stt=False, tts=False))
    cases = [
        ("measured good: office fan", "Is the office fan on?", -0.502, False),
        ("measured good: clock", "What time is it?", -0.401, False),
        ("measured bad: the mishearing", "S mu is what I must schedule.", -0.885, True),
        ("empty transcript", "", 0.0, True),
        ("exactly at the threshold stays fast", "some words", -0.75, False),
        ("just under it re-hears", "some words", -0.751, True),
    ]
    for label, text, logprob, expect in cases:
        got = bridge.brain._stt_unsure(text, logprob)
        check(f"gate: {label}", got == expect, f"unsure={got}")

    off = Bridge(ns(stt=False, tts=False, whisper_fallback="off"))
    check("gate: --whisper-fallback off disables it",
          off.brain._stt_unsure("S mu is what I must schedule.", -0.99) is False)

    tuned = Bridge(ns(stt=False, tts=False, stt_confidence=-0.95))
    check("gate: --stt-confidence moves the threshold",
          tuned.brain._stt_unsure("some words", -0.885) is False)


def live():
    """The real models, against a reference clip that does not move.

    The clip and its reference text live in .fixtures/, which is gitignored: it is
    the operator's voice and this repo is public. `captures/` is overwritten every
    turn, so pinning a regression test to it means the test silently changes what
    it asserts - which is exactly what happened once already.
    """
    ref = ".fixtures/stt-ref.wav"
    expect = "schedule"
    if not os.path.exists(ref):
        print(f"skip live: {ref} not present (operator voice, deliberately not in git)")
        return
    import wave

    from faster_whisper import WhisperModel

    with wave.open(ref) as w:
        pcm = w.readframes(w.getnframes())
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

    bridge = Bridge(ns())
    bridge.brain.whisper = WhisperModel("base.en", device="cpu", compute_type="int8",
                                        cpu_threads=8, local_files_only=True)
    base_runs = {bridge.brain._transcribe_with(bridge.brain.whisper, audio)[0]
                 for _ in range(3)}
    print(f"     base.en on the reference clip: {sorted(base_runs)}")

    small = Bridge(ns(whisper_model="small.en"))
    small.brain.whisper = WhisperModel("small.en", device="cpu", compute_type="int8",
                                       cpu_threads=8, local_files_only=True)
    t = time.monotonic()
    runs = {small.brain._transcribe_with(small.brain.whisper, audio)[0] for _ in range(3)}
    elapsed = time.monotonic() - t
    check("small.en gives the same answer every run", len(runs) == 1,
          f"{sorted(runs)}")
    check("and it is the right answer", any(expect in r.lower() for r in runs),
          f"in {elapsed / 3:.2f} s per run")


def main() -> int:
    gate_cases()
    live()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


sys.exit(main())
