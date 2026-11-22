#!/usr/bin/env python3
"""calc.py — deterministic answers for the formula classes we actually get asked.

Verification (re-asking a cloud model) took the local model from 1-of-6 to 5-of-6
correct on this bench. That is an improvement, not a guarantee: the one it still got
wrong was the dipole length, where the cloud computed a quarter wave for a half wave
(16.48 ft instead of 33 ft). Arithmetic is not a language task, so for the handful of
formulas this workshop actually asks, the answer is computed here and never predicted.

Covered: SWR (reflected power, mismatch loss), free-space path loss, ERP/EIRP from a
power and a gain, dipole element lengths, and days until a fixed date.

Deliberately narrow. Anything not matched here falls through to the model, and the
patterns are written to miss rather than misfire — a miss costs a slow answer, a
misfire costs a wrong one.

    python3 calc.py --selftest
"""

from __future__ import annotations

import datetime
import math
import re
import sys

DBD_TO_DBI = 2.15


# ── helpers ──────────────────────────────────────────────────────────────────

def _num(text: str) -> float:
    return float(text)


def _spoken(value: float, unit: str = "", places: int = 2) -> str:
    if value == int(value):
        return f"{int(value)}{unit}"
    return f"{round(value, places)}{unit}"


SWR_Q = re.compile(r"\bswr\b[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(?:[:/]|to)\s*(\d+(?:\.\d+)?)",
                   re.IGNORECASE)
SWR_Q2 = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[:/]|to|-)\s*(\d+(?:\.\d+)?)\s*(?:swr|:1)",
                    re.IGNORECASE)

FSPL_Q = re.compile(r"(?:free[- ]space\s+)?path\s+loss", re.IGNORECASE)
FREQ = re.compile(r"(\d+(?:\.\d+)?)\s*(ghz|gigahertz|mhz|megahertz|khz|kilohertz)\b",
                  re.IGNORECASE)
DIST = re.compile(r"(\d+(?:\.\d+)?)\s*(kilomet(?:re|er)s?|km|met(?:re|er)s?|m|miles?|mi|feet|foot|ft)\b",
                  re.IGNORECASE)

ERP_Q = re.compile(r"\b(erp|eirp|effective radiated power)\b", re.IGNORECASE)
WATTS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:watts?|w)\b", re.IGNORECASE)
GAIN = re.compile(r"(\d+(?:\.\d+)?)\s*(dbd|dbi)\b", re.IGNORECASE)

DIPOLE_Q = re.compile(r"\b(half|quarter)[- ]wave\b[^?]{0,40}\bdipole\b|"
                      r"\bdipole\b[^?]{0,40}\b(half|quarter)[- ]wave\b", re.IGNORECASE)
FREQ_ONLY = FREQ

DAYS_Q = re.compile(r"how many days (?:until|till|to)\b", re.IGNORECASE)


# ── SWR ──────────────────────────────────────────────────────────────────────

def swr_answer(text: str) -> str | None:
    """Reflected power fraction and mismatch loss from a standing-wave ratio."""
    match = SWR_Q.search(text) or SWR_Q2.search(text)
    if not match:
        return None
    forward, reflected = _num(match.group(1)), _num(match.group(2))
    if forward <= 0 or reflected <= 0:
        return None
    ratio = max(forward, reflected) / min(forward, reflected)
    if ratio < 1:
        return None
    gamma = (ratio - 1) / (ratio + 1)
    reflected_fraction = gamma ** 2
    mismatch_db = -10 * math.log10(max(1e-12, 1 - reflected_fraction))
    wants_db = bool(re.search(r"\bdb\b|loss|attenuat", text, re.IGNORECASE))
    wants_pct = bool(re.search(r"percent|%|reflect", text, re.IGNORECASE))
    percent = _spoken(reflected_fraction * 100, " percent")
    if wants_db and not wants_pct:
        return f"The mismatch loss is {_spoken(mismatch_db, ' dB')} for a {_spoken(ratio)} to 1 SWR."
    if wants_pct and not wants_db:
        return f"About {percent} of the power is reflected."
    return (f"About {percent} of the power is reflected, and the mismatch loss is "
            f"{_spoken(mismatch_db, ' dB')}.")


# ── free-space path loss ─────────────────────────────────────────────────────

def _to_mhz(token: str, unit: str) -> float:
    value = float(token)
    unit = unit.lower()
    if unit.startswith("gh"):
        return value * 1000.0
    if unit.startswith("kh"):
        return value / 1000.0
    return value


def _to_km(token: str, unit: str) -> float:
    value = float(token)
    unit = unit.lower()
    if unit.startswith("kilomet"):
        return value
    if unit in ("km",):
        return value
    if unit.startswith("met") or unit == "m":
        return value / 1000.0
    if unit.startswith("mile") or unit == "mi":
        return value * 1.609344
    if unit in ("feet", "foot", "ft"):
        return value * 0.0003048
    return value


def fspl_answer(text: str) -> str | None:
    if not FSPL_Q.search(text):
        return None
    freq = FREQ.search(text)
    dist = DIST.search(text)
    if not freq or not dist:
        return None
    mhz = _to_mhz(freq.group(1), freq.group(2))
    km = _to_km(dist.group(1), dist.group(2))
    if mhz <= 0 or km <= 0:
        return None
    loss = 32.45 + 20 * math.log10(mhz) + 20 * math.log10(km)
    return (f"The free-space path loss at {_spoken(mhz, ' MHz')} over "
            f"{_spoken(km, ' km')} is {_spoken(loss, ' dB')}.")


# ── ERP / EIRP ───────────────────────────────────────────────────────────────

def erp_answer(text: str) -> str | None:
    if not ERP_Q.search(text):
        return None
    watts = WATTS.search(text)
    gain = GAIN.search(text)
    if not watts or not gain:
        return None
    power = float(watts.group(1))
    db = float(gain.group(1))
    reference = gain.group(2).lower()
    out = power * (10 ** (db / 10))
    if reference == "dbd":
        return f"That is {_spoken(out, ' watts')} of ERP."
    return f"That is {_spoken(out, ' watts')} of EIRP (about {_spoken(out / (10 ** (DBD_TO_DBI / 10)), ' watts')} ERP)."


# ── dipole length ────────────────────────────────────────────────────────────

def dipole_answer(text: str) -> str | None:
    match = DIPOLE_Q.search(text)
    if not match:
        return None
    kind = (match.group(1) or match.group(2) or "").lower()
    freq = FREQ_ONLY.search(text)
    if not freq:
        return None
    mhz = _to_mhz(freq.group(1), freq.group(2))
    if mhz <= 0:
        return None
    # 468/f is the classic half-wave length in feet; 234/f is a quarter wave.
    half = kind == "half"
    feet = (468 if half else 234) / mhz
    metres = (143 if half else 71.5) / mhz
    label = "half-wave" if half else "quarter-wave"
    return (f"A {label} dipole for {_spoken(mhz, ' MHz')} is about "
            f"{_spoken(feet, ' feet')} ({_spoken(metres, ' metres')}) long.")


# ── days until a date ────────────────────────────────────────────────────────

HOLIDAYS = ("christmas", "new year", "new year's", "halloween", "independence day",
            "fourth of july", "thanksgiving")


def _next_holiday(name: str, today: datetime.date) -> datetime.date | None:
    name = name.lower()
    if "christmas" in name:
        target = datetime.date(today.year, 12, 25)
    elif "new year" in name:
        target = datetime.date(today.year + 1, 1, 1)
    elif "halloween" in name:
        target = datetime.date(today.year, 10, 31)
    elif "independence" in name or "fourth of july" in name:
        target = datetime.date(today.year, 7, 4)
    elif "thanksgiving" in name:
        # fourth Thursday of November
        nov = datetime.date(today.year, 11, 1)
        thursdays = [nov + datetime.timedelta(days=d) for d in range(30)
                     if (nov + datetime.timedelta(days=d)).weekday() == 3]
        target = thursdays[3]
    else:
        return None
    if target < today:
        if "christmas" in name:
            target = datetime.date(today.year + 1, 12, 25)
        elif "new year" in name:
            target = datetime.date(today.year + 2, 1, 1)
        elif "halloween" in name:
            target = datetime.date(today.year + 1, 10, 31)
        elif "independence" in name or "fourth of july" in name:
            target = datetime.date(today.year + 1, 7, 4)
        else:
            nov = datetime.date(today.year + 1, 11, 1)
            thursdays = [nov + datetime.timedelta(days=d) for d in range(30)
                         if (nov + datetime.timedelta(days=d)).weekday() == 3]
            target = thursdays[3]
    return target


DATE_TOKEN = re.compile(
    r"\b(?:on\s+)?(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.IGNORECASE)


def days_answer(text: str, today: datetime.date | None = None) -> str | None:
    if not DAYS_Q.search(text):
        return None
    today = today or datetime.date.today()
    lowered = text.lower()
    target: datetime.date | None = None
    for holiday in HOLIDAYS:
        if holiday in lowered:
            target = _next_holiday(holiday, today)
            break
    if target is None:
        token = DATE_TOKEN.search(text)
        if token:
            month = datetime.datetime.strptime(token.group(1)[:3], "%b").month
            day = int(token.group(2))
            try:
                target = datetime.date(today.year, month, day)
            except ValueError:
                return None
            if target < today:
                target = datetime.date(today.year + 1, month, day)
    if target is None:
        return None
    days = (target - today).days
    return f"That is {days} day{'s' if days != 1 else ''} from today."


# ── entry point ──────────────────────────────────────────────────────────────

def answer(text: str, today: datetime.date | None = None) -> str | None:
    """A computed answer, or None to let the model handle it."""
    question = (text or "").strip()
    if not question:
        return None
    for handler in (swr_answer, fspl_answer, erp_answer, dipole_answer):
        try:
            spoken = handler(question)
        except (ValueError, ZeroDivisionError, OverflowError):
            spoken = None
        if spoken:
            return spoken
    return days_answer(question, today)


# ── selftest ─────────────────────────────────────────────────────────────────

def _selftest() -> int:
    failures = 0

    def check(label: str, got, expect: str) -> None:
        nonlocal failures
        ok = bool(got) and expect in got
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label:46s} {got!r}"
              + ("" if ok else f"  (expected to contain {expect!r})"))

    def silent(label: str, got) -> None:
        nonlocal failures
        ok = got is None
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label:46s} {got!r}"
              + ("" if ok else "  (expected no answer)"))

    # SWR 3:1 -> 25% reflected, 1.25 dB mismatch loss (the case that broke twice)
    check("SWR 3:1, both", swr_answer("With an SWR of 3 to 1, what percentage of my "
                                      "power is reflected and how many dB do I lose?"),
          "25 percent")
    check("SWR 3:1, dB", swr_answer("what is the mismatch loss at SWR 3:1"), "1.25 dB")
    check("SWR 2:1, reflected", swr_answer("3-1 swr what is reflected"),
          "25 percent")
    check("SWR 2:1 proper", swr_answer("swr 2:1 reflected percentage"), "11.11 percent")

    # FSPL 440 MHz / 10 km = 105.3 dB (the cloud said 85.32 once, for 1 km)
    check("FSPL 440 MHz 10 km", fspl_answer("What is the free space path loss at 440 MHz "
                                            "over 10 kilometres?"), "105.32 dB")
    check("FSPL 146 MHz 1 km", fspl_answer("path loss at 146 MHz over 1 km"), "75.74 dB")

    # ERP: 100 W into 6 dBd = 398 W (the cloud and the local model both disagreed)
    check("ERP 100 W 6 dBd", erp_answer("What ERP do I get from 100 watts into an "
                                        "antenna with 6 dBd of gain?"), "398.11 watts of ERP")
    check("EIRP 100 W 6 dBi", erp_answer("what is my EIRP with 100 watts and 6 dBi"),
          "398.11 watts of EIRP")

    # dipole length: 468/14.2 = 32.96 ft (the cloud called this 16.48 ft)
    check("half-wave 14.2 MHz", dipole_answer("How long is a half wave dipole for "
                                              "14.2 megahertz?"), "32.96 feet")
    check("quarter-wave 14.2 MHz", dipole_answer("how long is a quarter wave dipole "
                                                 "on 14.2 MHz"), "16.48 feet")

    # days until
    today = datetime.date(2026, 11, 22)
    check("days until Christmas", days_answer("how many days until Christmas?", today),
          "33 days")
    check("days until 25 December", days_answer("how many days until December 25", today),
          "33 days")

    # must NOT fire: a miss costs a slow answer, a misfire costs a wrong one
    for question in ("how many days of leave do I have",
                     "what is my SWR reading on the meter",
                     "what time is sunset in Denver",
                     "explain why a folded dipole steps up impedance",
                     "how long is a piece of string"):
        silent(f"falls through: {question[:30]!r}", answer(question, today))

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_selftest())
