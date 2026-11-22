#!/usr/bin/env python3
"""fastpath_test.py — which questions are answered in code rather than by the model.

The fast path exists because two classes of question have authoritative answers that
are not language tasks: the clock (the OS knows) and a handful of workshop formulas
(arithmetic, where the model was measured wrong 5 times in 6). This pins the routing:
what must be computed, and what must be left to the model.
"""
import sys

import calc
import clock

failures = 0


def check(label, got, expect_substring):
    global failures
    ok = bool(got) and expect_substring in got
    failures += 0 if ok else 1
    print(f"{'ok  ' if ok else 'FAIL'} {label:50s} {str(got)[:64]!r}")


def silent(label, got):
    global failures
    ok = got is None
    failures += 0 if ok else 1
    print(f"{'ok  ' if ok else 'FAIL'} {label:50s} {str(got)[:64]!r}")


CLOCK_QUESTIONS = [
    "what time is it?", "What's the time?", "what is the date",
    "what day is it?", "what day of the week is it?", "which day is it?",
    "what's today's date?", "time and date",
]
for question in CLOCK_QUESTIONS:
    got = clock.answer(question)
    failures += 0 if got else 1
    print(f"{'ok  ' if got else 'FAIL'} {'clock: ' + question:50s} {str(got)[:64]!r}")

CALC_QUESTIONS = [
    ("With an SWR of 3 to 1, what percentage of my power is reflected and how many dB do I lose?",
     "25 percent"),
    ("What is the free space path loss at 440 MHz over 10 kilometres?", "105.32 dB"),
    ("What ERP do I get from 100 watts into an antenna with 6 dBd of gain?", "398.11 watts"),
    ("How long is a half wave dipole for 14.2 megahertz?", "32.96 feet"),
    ("how many days until Christmas?", "days from today"),
]
for question, expect in CALC_QUESTIONS:
    check(f"calc: {question[:44]}", calc.answer(question), expect)

# must reach the model: language, opinion, or data only he has
for question in ("explain why a folded dipole has a 4 to 1 impedance step up",
                 "what is a balun",
                 "what's on my list for tomorrow",
                 "how warm is the office",
                 "what is my SWR reading right now",
                 "how many days of leave do I have"):
    silent(f"model: {question[:44]}", calc.answer(question) or clock.answer(question))

print("\nFAILURES:", failures)
sys.exit(1 if failures else 0)
