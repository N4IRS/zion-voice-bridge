#!/usr/bin/env python3
"""envelope.py — the message contract between the voice bridge and an answering agent.

The bus moves bytes; it does not know what a request is. Everything that makes a
hand-off answerable lives here, in one versioned object:

    { "v": 1,
      "type": "voice.handoff",
      "id": "<uuid>",                 # this message's identity
      "created_at": "...Z",
      "ttl_s": 900,                   # after which the asker no longer wants it
      "from": "bridge", "to": "muse",
      "device": {"name": "Satellite-1", "kind": "echo-dot"},
      "reply_to": {"agent": "bridge", "bus_id": 1535},
      "request": {"question": "...", "voice": true, "max_words": 30, "hint": "..."} }

and the answer:

    { "v": 1,
      "type": "voice.answer",
      "id": "<uuid>",
      "created_at": "...Z",
      "from": "muse", "to": "bridge",
      "correlation": "<the handoff envelope's id — a UUID, not the bus id>",
      "in_reply_to": 1535,                      # the bus message it answers
      "status": "ok" | "error",
      "answer": {"text": "...", "words": 24},   # required when status=ok
      "error": {"code": "no_data", "message": "..."} }   # required when status=error

Three rules the schema exists to enforce, each from a failure this project hit:

1. **An empty body is never a valid answer.** `status=ok` requires non-empty
   `answer.text`; an agent that cannot answer sends `status=error`. The bus's
   `(empty reply)` placeholder was spoken aloud once and must be impossible.
2. **The correlation is echoed.** The asker matches on `correlation` (or, for
   agents that cannot set it, on `in_reply_to`), so a correct answer is never left
   unmatched for minutes.
3. **TTL is carried, not assumed.** A worker skips anything past its TTL, so a late
   consumer does not answer a stale question and the asker does not speak it.

Backward compatibility: `parse()` returns None for a body that is not an envelope,
and the bridge falls back to plain text. Adopting the envelope is therefore safe
before every agent does.

    python3 envelope.py --selftest
"""

from __future__ import annotations

import datetime
import json
import sys
import uuid

VERSION = 1
HANDOFF = "voice.handoff"
ANSWER = "voice.answer"

#: recommended error codes. Validation requires a non-empty code but does not
#: constrain it to this set — a new code from an agent must not be rejected by an
#: older bridge; the bridge logs unknown codes rather than refusing them.
ERROR_CODES = ("no_data", "unavailable", "bad_question", "timeout", "internal")


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _stamp(when: datetime.datetime | None = None) -> str:
    return (when or _now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_stamp(text: str) -> datetime.datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def make_handoff(question: str, *, device: str = "satellite", from_agent: str = "bridge",
                 to: str = "muse", bus_id: int | None = None, ttl_s: int = 900,
                 max_words: int = 30, hint: str = "", conversation: str = "",
                 context: list[dict] | None = None) -> dict:
    """Build a hand-off.

    `context` is the recent exchange as the asker saw it — each entry
    {"role": "user"|"assistant", "text": ...}, oldest first, already trimmed by the
    caller. It exists so a responder never has to remember state between calls: the
    envelope is self-contained, and a responder may still cache deeper state of its
    own as an optimisation.
    """
    return {
        "v": VERSION,
        "type": HANDOFF,
        "id": str(uuid.uuid4()),
        "created_at": _stamp(),
        "ttl_s": int(ttl_s),
        "from": from_agent,
        "to": to,
        "device": {"name": device, "kind": "echo-dot"},
        "reply_to": {"agent": from_agent, "bus_id": bus_id},
        "conversation": conversation or "",
        "request": {"question": question.strip(), "voice": True,
                    "max_words": int(max_words), "hint": hint,
                    # optional speaking-style hint. Added 2026-11-21 as a drift
                    # probe: an adopting agent whose prompt predates it should
                    # notice the unknown field and consult /schema.json rather
                    # than guess. Currently always "plain" and safe to ignore.
                    "tone": "plain",
                    "context": [c for c in (context or [])][-6:]},
    }


def make_answer(correlation: str, text: str = "", *, from_agent: str = "muse",
                to: str = "bridge", in_reply_to: int | None = None,
                error_code: str | None = None, error_message: str = "") -> dict:
    if error_code:
        status, answer, error = "error", None, {"code": error_code, "message": error_message}
    else:
        status = "ok"
        words = len(text.split())
        answer = {"text": text.strip(), "words": words} if text.strip() else None
        error = None
        if answer is None:
            raise ValueError("a status=ok answer needs non-empty text; use error_code instead")
    return {
        "v": VERSION,
        "type": ANSWER,
        "id": str(uuid.uuid4()),
        "created_at": _stamp(),
        "from": from_agent,
        "to": to,
        "correlation": correlation,
        "in_reply_to": in_reply_to,
        "status": status,
        "answer": answer,
        "error": error,
    }


def parse(body: str) -> dict | None:
    """An envelope, or None if the body is plain text (backward compatible)."""
    if not isinstance(body, str):
        return None
    stripped = body.strip()
    if not stripped.startswith("{"):
        return None
    try:
        env = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return env if isinstance(env, dict) and env.get("type", "").startswith("voice.") else None


def validate(env: dict) -> list[str]:
    """Everything wrong with this envelope, in order. Empty list means valid."""
    problems: list[str] = []
    if not isinstance(env, dict):
        return ["not an object"]
    if env.get("v") != VERSION:
        problems.append(f"unsupported version {env.get('v')!r} (expected {VERSION})")
    kind = env.get("type")
    if kind not in (HANDOFF, ANSWER):
        problems.append(f"unknown type {kind!r}")
    for field in ("id", "from", "to", "created_at"):
        if not env.get(field):
            problems.append(f"missing {field}")
    if _parse_stamp(env.get("created_at", "")) is None:
        problems.append("created_at is not an ISO-8601 UTC stamp")
    if kind == HANDOFF:
        if int(env.get("ttl_s") or 0) <= 0:
            problems.append("ttl_s must be positive")
        question = ((env.get("request") or {}).get("question") or "").strip()
        if not question:
            problems.append("request.question is empty")
        if not (env.get("device") or {}).get("name"):
            problems.append("device.name is missing")
        if not (env.get("reply_to") or {}).get("agent"):
            problems.append("reply_to.agent is missing")
    elif kind == ANSWER:
        if not env.get("correlation"):
            problems.append("correlation is missing (the answer cannot be matched)")
        status = env.get("status")
        if status not in ("ok", "error"):
            problems.append(f"status {status!r} must be ok or error")
        if status == "ok":
            if not ((env.get("answer") or {}).get("text") or "").strip():
                problems.append("status=ok requires non-empty answer.text")
        elif not ((env.get("error") or {}).get("code") or "").strip():
            problems.append("status=error requires error.code")
    return problems


def is_expired(env: dict, now: datetime.datetime | None = None) -> bool:
    created = _parse_stamp(env.get("created_at", ""))
    ttl = int(env.get("ttl_s") or 0)
    if created is None or ttl <= 0:
        return True
    return (now or _now()) > created + datetime.timedelta(seconds=ttl)


def expires_at(env: dict) -> datetime.datetime | None:
    created = _parse_stamp(env.get("created_at", ""))
    ttl = int(env.get("ttl_s") or 0)
    return created + datetime.timedelta(seconds=ttl) if created else None


def answer_text(env: dict) -> str | None:
    """The speakable text of a valid, ok answer; None otherwise."""
    if env.get("type") != ANSWER or env.get("status") != "ok":
        return None
    text = ((env.get("answer") or {}).get("text") or "").strip()
    return text or None


def answer_error(env: dict) -> tuple[str, str] | None:
    if env.get("type") != ANSWER or env.get("status") != "error":
        return None
    err = env.get("error") or {}
    return (err.get("code") or "unknown", err.get("message") or "")


def schema_summary() -> dict:
    """Machine-readable summary, derived from this module rather than written twice.

    Served by the bridge at /schema.json so an agent can discover the contract it
    must speak without fetching source or trusting a cached copy.
    """
    return {
        "v": VERSION,
        "types": {"handoff": HANDOFF, "answer": ANSWER},
        "error_codes": list(ERROR_CODES),
        "handoff": {
            "required": ["v", "type", "id", "created_at", "ttl_s", "from", "to",
                         "device.name", "reply_to.agent", "request.question"],
            "optional": ["reply_to.bus_id", "conversation", "request.context",
                         "request.max_words", "request.hint", "request.tone"],
            "field_notes": {
                "request.tone": "optional speaking-style hint, currently always "
                                "'plain'; safe to ignore, added 2026-11-21",
            },
            "request_context": "list of {role, text}, oldest first, max 6 entries",
        },
        "answer": {
            "required": ["v", "type", "id", "created_at", "from", "to",
                         "correlation", "status"],
            "required_when_ok": ["answer.text (non-empty)"],
            "required_when_error": ["error.code"],
            "optional": ["in_reply_to", "answer.words", "error.message"],
            "correlation": "the request envelope's id (UUID), not the bus row id",
        },
        "rules": [
            "status=ok requires non-empty answer.text; an empty body is never valid",
            "the answer echoes correlation (and/or in_reply_to), or it cannot be matched",
            "ttl_s belongs to the request; a worker skips work past its TTL",
        ],
    }


# ── selftest ─────────────────────────────────────────────────────────────────

def _selftest() -> int:
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures += 1

    hand = make_handoff("What time is sunset?", device="Satellite-1", bus_id=1535)
    check("handoff validates", validate(hand) == [], str(validate(hand)))
    with_ctx = make_handoff("and the other one?", device="Satellite-1", conversation="c1",
                            context=[{"role": "user", "text": "what time is sunset?"},
                                     {"role": "assistant", "text": "4:40 PM."}])
    check("context is carried", len(with_ctx["request"]["context"]) == 2)
    check("context validates", validate(with_ctx) == [])
    check("handoff round-trips through JSON", parse(json.dumps(hand)) == hand)
    check("handoff is not expired", not is_expired(hand))
    check("plain text is not an envelope", parse("Tomorrow looks fine.") is None)
    check("json that is not ours is ignored", parse('{"hello": "world"}') is None)

    ans = make_answer(hand["id"], "Sunset is at 4:40 PM.", in_reply_to=1535)
    check("answer validates", validate(ans) == [], str(validate(ans)))
    check("answer carries the correlation", ans["correlation"] == hand["id"])
    check("answer text is speakable", answer_text(ans) == "Sunset is at 4:40 PM.")

    # the rule that exists because a placeholder was spoken once
    try:
        make_answer(hand["id"], "   ")
        check("empty ok-answer is refused", False, "constructed")
    except ValueError as exc:
        check("empty ok-answer is refused", True, str(exc)[:48])

    bad = dict(ans, answer={"text": "  "})
    check("empty answer.text fails validation", "status=ok requires non-empty answer.text"
          in " | ".join(validate(bad)))

    err = make_answer(hand["id"], error_code="no_data", error_message="no forecast source")
    check("typed error validates", validate(err) == [])
    check("typed error is not speakable text", answer_text(err) is None)
    check("typed error carries a code", answer_error(err) == ("no_data", "no forecast source"))

    check("missing correlation is rejected",
          any("correlation" in p for p in validate(dict(ans, correlation=""))))
    check("wrong version is rejected",
          any("version" in p for p in validate(dict(ans, v=99))))
    check("error codes are documented", len(ERROR_CODES) >= 3, ", ".join(ERROR_CODES))
    summary = schema_summary()
    check("schema summary is derivable", summary["v"] == VERSION
          and summary["types"]["handoff"] == HANDOFF
          and len(summary["error_codes"]) == len(ERROR_CODES))
    check("unknown type is rejected",
          any("unknown type" in p for p in validate(dict(ans, type="voice.other"))))

    old = dict(hand, created_at=(_now() - datetime.timedelta(seconds=hand["ttl_s"] + 5)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    check("a stale handoff is expired", is_expired(old))
    check("expiry is computable", expires_at(hand) is not None)

    print("\nFAILURES:", failures)
    print("\nexample handoff:\n" + json.dumps(hand, indent=1))
    print("\nexample answer:\n" + json.dumps(ans, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_selftest())
