#!/usr/bin/env python3
"""verify_test.py — the numeric-verification policy, as cases.

Pure decision table for Bridge.should_verify: when to spend a cloud call checking a
local answer's numbers. Every "skip" below is a measured failure or a measured cost.
"""
import sys

from bridge import Bridge

# (question, answer, used_tools, mode, expect, why)
CASES = [
    # the case this exists for: local model confidently wrong, no hedge
    ("With an SWR of 3 to 1, what % is reflected and how many dB do I lose?",
     "Eight percent reflected, about half a dB loss.", False, "numbers", True),
    ("What is the free space path loss at 440 MHz over 10 km?",
     "Approximately 135.8 dB.", False, "numbers", True),
    # local context is the authority — a cold cloud answer invented a national park
    ("Tell me about Zion", "Zion is this machine.", False, "numbers", False),
    ("What are you?", "I am Zion, the local assistant.", False, "numbers", False),
    # numbers we supplied
    ("What time is it?", "It is 9:04 PM.", False, "numbers", False),
    # numbers a tool supplied — including the rounding a model does
    ("How warm is the office?", "It is 81.4 degrees.", True, "numbers", False),
    # nothing numeric to check
    ("What is a balun?", "A balun converts balanced to unbalanced.", False, "numbers", False),
    # switches
    ("What is a balun?", "1 example is a choke.", False, "off", False),
    ("What is a balun?", "No numbers here.", False, "always", True),
    # spelled-out numbers must trigger: the original failure had no digits
    ("With an SWR of 3 to 1, what % is reflected?",
     "Eight percent reflected, about half a dB loss.", False, "numbers", True),
]

#: the numbers a Home Assistant lookup returns for "how warm is the office?"
TOOL_NUMBERS = {"81.446", "unavailable"}

failures = 0
for question, answer, tools, mode, expect in CASES:
    got, why = Bridge.should_verify(question, answer, tools, mode=mode, cloud=True,
                                    tool_numbers=TOOL_NUMBERS)
    ok = got == expect
    failures += 0 if ok else 1
    print(f"{'ok  ' if ok else 'FAIL'} verify={got!s:5s} expect={expect!s:5s} "
          f"[{why}] {question[:54]!r}")

# a tool result is not a licence for arithmetic: the date came from the clock, the
# 84 days did not (correct: 34)
got, why = Bridge.should_verify("How many days until Christmas?",
                                "There are 84 days until Christmas.", True,
                                mode="numbers", cloud=True,
                                tool_numbers={"2026", "11", "21", "22", "57"})
print(f"{'ok  ' if got is True else 'FAIL'} computed on top of a tool -> verify  [{why}]")
failures += 0 if got is True else 1

# a missing cloud lane can never verify
got, why = Bridge.should_verify("how many dB?", "3 dB", False, mode="numbers", cloud=False)
print(f"{'ok  ' if got is False else 'FAIL'} no cloud lane -> no verification [{why}]")
failures += 0 if got is False else 1

print("\nFAILURES:", failures)
sys.exit(1 if failures else 0)
