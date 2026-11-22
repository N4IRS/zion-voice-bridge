#!/usr/bin/env python3
"""stream_test.py — streaming TTS cancellation.

The gap this covers, found in review: a barge-in cancelled the turn's task but not the
sentence pump or its announcement queue, so sentence 2 of a streamed reply could be
spoken over the user. A new turn has the same problem for the same reason.
"""
import asyncio
import sys
import types

from bridge import Bridge, SYSTEM_PROMPT

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    failures += 0 if ok else 1


class FakeLane:
    """Yields one sentence every 40 ms, so a test can interrupt mid-reply."""

    sync_capable = True
    name = "local"

    async def stream_chat(self, messages, tools=None):
        for i in range(1, 6):
            await asyncio.sleep(0.04)
            yield f"Sentence number {i} of the reply."


def fake_bridge(announced, announce_delay=0.05):
    b = Bridge.__new__(Bridge)
    b.args = types.SimpleNamespace(
        llm=True, stt=True, tts=False, answer_lane="local",
        system_prompt=SYSTEM_PROMPT, attribution="off",
        device_name="Satellite-1", memory="off", memory_turns=3, memory_ttl_s=300.0,
        stream_tts=True, handoff="off", verify="off", verify_max_tokens=400,
        fastpath="off", clock_tool=False,
        max_tokens=100, media_port=8701, media_host="127.0.0.1", announce_timeout=30,
        cloud_name="Muse", local_name="Zion", muse=False, ha=False, music=False,
        silence_ms=1200, min_speech_ms=160, initial_prompt="", whisper_model="base.en",
        threads=8, piper_voice="", outdir="captures", ollama="http://127.0.0.1:11434",
        model="qwen2.5:3b", lane_timeout=25, cloud_model=None, cloud_url="", cloud_key=None,
        cloud_max_tokens=80, cloud_system_prompt=None, host="127.0.0.1", port=16001,
        seconds=0, once=False, verbose=False, ack_text=None, handoff_poll_s=5,
        handoff_max_attempts=12, handoff_ttl_s=900, ha_url="", ha_token="", ma_url="",
        ma_token="", music_target="Satellite-1", bus_url="", bus_token="", muse_agent="muse",
        muse_name="Muse", bus_type="voice-handoff")
    b.brain = types.SimpleNamespace(lanes=types.SimpleNamespace(
        answer_lane=lambda n: FakeLane(),
        async_=[], __contains__=lambda self, k: False))
    b.client = None
    b.ha = None
    b.ma = None
    b.pending = {}
    b.pending_path = "/tmp/stream-test-pending.json"
    b.in_turn = False
    b.announce_lock = asyncio.Lock()
    b._bg_tasks = set()
    b.media_dir = "/tmp"
    b.history = {}
    b.conversation = "c"
    b._handoff_lane = lambda: None
    b._voice_trim = lambda t: t

    async def fake_announce(text):
        announced.append(text)
        await asyncio.sleep(announce_delay)
        return True
    b._announce = fake_announce
    return b


async def main():
    # 1. a normal streamed reply hands back sentence 1 and announces the rest
    announced = []
    b = fake_bridge(announced)
    first, _, note = await b._answer("tell me things", 1)
    await asyncio.sleep(0.6)
    check("first sentence returned", bool(first and "Sentence number 1" in first), repr(first))
    check("remaining sentences announced", len(announced) >= 2, str(announced))

    # 2. cancelling mid-stream stops the announcements
    announced.clear()
    b = fake_bridge(announced, announce_delay=0.2)
    first, _, _ = await b._answer("tell me things", 2)
    await asyncio.sleep(0.25)          # sentence 1 announced, 2 queued/being spoken
    b._cancel_stream_tasks()
    seen = len(announced)
    await asyncio.sleep(0.8)
    check("cancel stops further announcements", len(announced) <= seen + 1,
          f"{seen} then {len(announced)}")
    check("no tasks left behind", len(b._bg_tasks) == 0, str(b._bg_tasks))

    print("\nFAILURES:", failures)
    return 1 if failures else 0


sys.exit(asyncio.run(main()))
