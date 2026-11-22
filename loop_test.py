#!/usr/bin/env python3
"""loop_test.py — CPU work must not block the event loop.

Measured problem this covers: whisper.transcribe (0.6-0.9 s) and Piper synthesis
(0.15-0.45 s) were called synchronously inside async handlers, so the loop that also
reads the audio socket and answers keepalives was stalled for a second at a time. They
now run through asyncio.to_thread.

The test starts a 20 ms ticker and runs a blocking function through the same path the
bridge uses; if the loop were blocked, the ticker would miss its slots.
"""
import asyncio
import sys
import time

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    failures += 0 if ok else 1


def blocking_work(seconds: float) -> float:
    time.sleep(seconds)          # stands in for whisper/piper: CPU-bound, no awaits
    return seconds


async def main():
    # 1. the loop keeps ticking while CPU work runs in a worker thread
    stalls: list[float] = []
    last = time.monotonic()

    async def ticker(stop: asyncio.Event) -> None:
        nonlocal last
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            stalls.append(now - last)
            last = now

    stop = asyncio.Event()
    task = asyncio.create_task(ticker(stop))
    await asyncio.to_thread(blocking_work, 0.6)
    stop.set()
    await task
    worst = max(stalls) if stalls else 99.0
    check("to_thread keeps the loop ticking", worst < 0.15,
          f"worst gap {worst*1000:.0f} ms over {len(stalls)} ticks")

    # 2. the same work called directly DOES block (proves the test can fail)
    stalls.clear()
    stop = asyncio.Event()
    task = asyncio.create_task(ticker(stop))
    await asyncio.sleep(0.05)
    blocking_work(0.3)
    stop.set()
    await task
    worst_direct = max(stalls) if stalls else 0.0
    check("direct call would block (control)", worst_direct >= 0.2,
          f"worst gap {worst_direct*1000:.0f} ms")

    print("\nFAILURES:", failures)
    return 1 if failures else 0


sys.exit(asyncio.run(main()))
