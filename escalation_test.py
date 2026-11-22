#!/usr/bin/env python3
"""escalation_test.py — where a question should go, as cases.

Two pure decisions, both of which were wrong at least once on live turns:

    _effective_needs(model_label, question)  -> "data" | "reasoning"
    _hedge_kind(answer)                      -> "agent" | "knowledge" | None

Every case below is a failure or a near-miss from a real turn, not a hypothetical.
"""
import sys

from bridge import Bridge

failures = 0


def check(label, got, expect):
    global failures
    ok = got == expect
    failures += 0 if ok else 1
    print(f"{'ok  ' if ok else 'FAIL'} {label:58s} {got!r} (expect {expect!r})" if not ok
          else f"ok   {label:58s} {got!r}")


# (question, model's label, effective destination)
NEEDS = [
    # arithmetic mislabelled as data by the 3B — the override exists for these
    ("What ERP do I get from 100 watts into an antenna with 6 dB of gain?", "data", "reasoning"),
    ("What is the free space path loss at 440 MHz over 10 kilometres?", "data", "reasoning"),
    ("How long is a half wave dipole for 14.2 megahertz, in feet?", "data", "reasoning"),
    ("Why does a folded dipole have a 4 to 1 impedance step up?", "data", "reasoning"),
    # records only Muse has — no unit, no number, so the label stands
    ("What's on my list for tomorrow?", "data", "data"),
    ("Where did we leave the duplexer order?", "data", "data"),
    ("How many tasks do I have?", "data", "data"),
    # the model's own reasoning label is respected
    ("What time is sunset in Denver today?", "reasoning", "reasoning"),
    ("What is a balun?", "data", "data"),
]

for question, label, expect in NEEDS:
    check(f"needs({question[:44]!r})", Bridge._effective_needs(label, question), expect)

# a nonsense label must not become a third lane
check("needs(invalid label)", Bridge._effective_needs("banana", "what is a balun?"), "data")

HEDGES = [
    # data deferrals: the agent holds the records, a cloud model never will
    ("I need Muse for this.", "agent"),
    ("I need your own records to answer that.", "agent"),
    ("I can't see your calendar from here.", "agent"),
    ("You would need to ask Muse about that.", "agent"),
    # knowledge gaps: the cloud lane answers these in about a second
    ("I don't have real-time information.", "knowledge"),
    ("I'm not sure of the exact figure.", "knowledge"),
    ("I don't have access to weather data.", "knowledge"),
    # not a hedge at all
    ("You lose 1.25 dB with 25 percent of power reflected.", None),
    ("The office is 81 degrees.", None),
]

for text, expect in HEDGES:
    check(f"hedge_kind({text[:44]!r})", Bridge._hedge_kind(text), expect)

print("\nFAILURES:", failures)
sys.exit(1 if failures else 0)
