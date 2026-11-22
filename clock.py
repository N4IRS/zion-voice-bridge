#!/usr/bin/env python3
"""clock.py — the OS answers time and date questions, not a model.

Why this exists: "what time is it?" was costing a full model turn (2-5 s, ~290 prompt
tokens) to paraphrase something the host already knows exactly. Worse, the model could
get it wrong — asked without context it once answered *"It is 3:47 PM, the workshop
clock reads this time because its mechanical hands rotate…"* at 18:48 — and when the
time WAS injected it kept appending *"The current time is…"* to unrelated answers.

So the canonical clock questions are answered here, deterministically, in microseconds
and with no possibility of error. Anything subtler ("how long until sunset", "what day
is the market open") still goes to the model, with the date supplied as context.

Spoken form is spelled out, because a synthesiser reads "6:54 PM" as two letters and
"21st" inconsistently: "it's six fifty-four in the evening", "today is Saturday the
twenty-first of November, twenty twenty-six".

    python3 clock.py --selftest
"""

from __future__ import annotations

import datetime
import re
import sys

UNITS = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = ("", "", "twenty", "thirty", "forty", "fifty")
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
            "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _words(n: int) -> str:
    if n < 20:
        return UNITS[n]
    tens, unit = divmod(n, 10)
    return TENS[tens] + (f"-{UNITS[unit]}" if unit else "")


def _ordinal(n: int) -> str:
    word = _words(n)
    if "-" in word:                       # twenty-first
        head, tail = word.split("-")
        return f"{head}-{ORDINALS.get(tail, tail + 'th')}"
    return ORDINALS.get(word, word + "th")


def _year(n: int) -> str:
    high, low = divmod(n, 100)
    if low == 0:
        return f"{_words(high)} hundred"
    if low < 10:
        return f"{_words(high)} oh {_words(low)}"
    return f"{_words(high)} {_words(low)}"


def spoken_time(now: datetime.datetime) -> str:
    hour, minute = now.hour, now.minute
    if hour < 6:
        part = "at night"
    elif hour < 12:
        part = "in the morning"
    elif hour < 17:
        part = "in the afternoon"
    elif hour < 22:
        part = "in the evening"
    else:
        part = "at night"
    spoken_hour = hour % 12 or 12
    if minute == 0:
        clock = f"{_words(spoken_hour)} o'clock"
    elif minute == 30:
        clock = f"half past {_words(spoken_hour)}"
    elif minute == 15:
        clock = f"quarter past {_words(spoken_hour)}"
    elif minute == 45:
        clock = f"quarter to {_words(spoken_hour % 12 + 1)}"
    else:
        clock = f"{_words(spoken_hour)} {_words(minute)}"
    return f"It's {clock} {part}."


def spoken_date(now: datetime.datetime) -> str:
    return (f"Today is {now:%A} the {_ordinal(now.day)} of {now:%B}, "
            f"{_year(now.year)}.")


#: questions this module answers itself. Deliberately narrow: exact intents, not
#: prefixes of longer questions ("what time does the shop close" is NOT ours).
TIME_Q = re.compile(r"^\s*(what(?:'s| is)?(?: the)? time(?: is it)?|"
                    r"what time is it|got the time|time is it)\??\s*$", re.IGNORECASE)
DATE_Q = re.compile(
    r"^\s*(?:"
    r"what(?:'s| is)?(?: the)? date(?: today)?|"
    r"(?:what|which) day(?: of the week)? is (?:it|today)|"
    r"(?:what|which) day(?: of the week)?(?: is it)? today|"
    r"what is today(?:'s date)?|what'?s today(?:'s date)?"
    r")\??\s*$", re.IGNORECASE)
BOTH_Q = re.compile(r"^\s*(?:(?:what(?:'s| is)?(?: the)? )?"
                    r"(?:time and date|date and time))\??\s*$", re.IGNORECASE)

#: a question that needs the current date as context even though we cannot answer it
#: from the clock alone ("how many days until Christmas", "what's the forecast tomorrow")
TIME_CONTEXT = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (?:morning|afternoon|evening|week|month)|"
    r"next (?:week|month)|days? (?:until|till|to)|how long until|sunset|sunrise|"
    r"current (?:time|date)|right now|so far|day of the week|what day)\b", re.IGNORECASE)


def answer(question: str, now: datetime.datetime | None = None) -> str | None:
    """A spoken answer if this is a clock/date question, else None."""
    text = (question or "").strip()
    if not text:
        return None
    now = now or datetime.datetime.now()
    if BOTH_Q.match(text):
        return f"{spoken_time(now)} {spoken_date(now)}"
    if TIME_Q.match(text):
        return spoken_time(now)
    if DATE_Q.match(text):
        return spoken_date(now)
    return None


def wants_time_context(question: str) -> bool:
    """Whether the prompt should carry the date — only when the turn is temporal."""
    return bool(TIME_CONTEXT.search(question or ""))


# ── selftest ─────────────────────────────────────────────────────────────────

def _selftest() -> int:
    failures = 0

    def check(label: str, got, expect) -> None:
        nonlocal failures
        ok = got == expect
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label:52s} {got!r}"
              + ("" if ok else f"  (expect {expect!r})"))

    at = datetime.datetime(2026, 11, 21, 18, 54)
    check("evening time", spoken_time(at), "It's six fifty-four in the evening.")
    check("midnight", spoken_time(at.replace(hour=0, minute=5)), "It's twelve five at night.")
    check("noon", spoken_time(at.replace(hour=12, minute=0)), "It's twelve o'clock in the afternoon.")
    check("half past", spoken_time(at.replace(hour=9, minute=30)), "It's half past nine in the morning.")
    check("quarter to", spoken_time(at.replace(hour=10, minute=45)), "It's quarter to eleven in the morning.")
    check("date", spoken_date(at), "Today is Saturday the twenty-first of November, twenty twenty-six.")
    check("date ordinal", spoken_date(at.replace(day=1)), "Today is Sunday the first of November, twenty twenty-six.")
    check("date year 2005", spoken_date(at.replace(year=2005)), "Today is Monday the twenty-first of November, twenty oh five.")
    check("date year 2020", spoken_date(at.replace(year=2020)), "Today is Saturday the twenty-first of November, twenty twenty.")
    check("date year 2000", spoken_date(at.replace(year=2000)), "Today is Tuesday the twenty-first of November, twenty hundred.")

    for q, expect_kind in (("what time is it?", "time"), ("What's the time?", "time"),
                           ("what is the date", "date"), ("what day is it?", "date"),
                           ("what's today's date?", "date"), ("time and date", "both"),
                           ("what day of the week is it?", "date"),
                           ("which day is it?", "date")):
        got = answer(q, at) or ""
        if expect_kind == "time":
            ok = "six fifty-four" in got
        elif expect_kind == "date":
            ok = "twenty-first" in got
        else:
            ok = "six fifty-four" in got and "twenty-first" in got
        check(f"answers {q!r}", ok, True)

    # must NOT be stolen from the model: these need reasoning, not a clock
    for q in ("what time does the shop close", "how long until sunset",
              "what day is the market open", "why is time slow"):
        check(f"leaves {q[:26]!r} to the model", answer(q, at), None)

    check("date context for temporal questions", wants_time_context("what's the forecast tomorrow?"), True)
    check("no date context for plain questions", wants_time_context("what is a balun?"), False)

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_selftest())
