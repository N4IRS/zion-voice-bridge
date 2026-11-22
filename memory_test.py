#!/usr/bin/env python3
"""memory_test.py — conversation memory: what is carried, and in which shape.

Two shapes exist and they are easy to confuse: the local lane wants chat turns
({"role", "content"}), the hand-off envelope wants {"role", "text"}. Confusing them
raised KeyError('text') on the first live follow-up that also needed a hand-off,
which no earlier test reached.
"""
import sys
import types

from bridge import Bridge

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    failures += 0 if ok else 1


def fake_bridge(memory="auto"):
    b = Bridge.__new__(Bridge)
    b.args = types.SimpleNamespace(memory=memory, device_name="Satellite-1",
                                   memory_turns=2, memory_ttl_s=300.0)
    b.history = {}
    b.conversation = "conv-1"
    return b


b = fake_bridge()
b.remember("Satellite-1", "user", "which needs more room, 40m or 20m?")
b.remember("Satellite-1", "assistant", "the 40m one is longer")

chat = b.history_for("Satellite-1", "and the other one?")
check("follow-up carries history", len(chat) == 2, str(len(chat)))
check("chat shape is {role, content}", set(chat[0]) == {"role", "content"}, str(chat[0].keys()))

env = b._handoff_context("and the other one?")
check("envelope context is carried", len(env) == 2, str(len(env)))
check("envelope shape is {role, text}", set(env[0]) == {"role", "text"}, str(env[0].keys()))
check("envelope text is not empty", all(e["text"] for e in env))

self_contained = b._handoff_context("what is a balun?")
check("self-contained question carries none", self_contained == [], str(self_contained))

off = fake_bridge(memory="off")
off.remember("Satellite-1", "user", "something")
check("memory=off carries nothing", off.history_for("Satellite-1", "and that?") == [])
check("memory=off hand-off context empty", off._handoff_context("and that?") == [])

b.remember("Satellite-1", "user", "one"); b.remember("Satellite-1", "assistant", "two")
b.remember("Satellite-1", "user", "three"); b.remember("Satellite-1", "assistant", "four")
check("history capped at memory_turns*2", len(b.history["Satellite-1"]) == 4,
      str(len(b.history["Satellite-1"])))

print("\nFAILURES:", failures)
sys.exit(1 if failures else 0)
