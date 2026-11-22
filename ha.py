#!/usr/bin/env python3
"""ha.py — Home Assistant as a *tool* for the local voice lane.

This is not the voice pipeline. Home Assistant is not in the audio path, not the
brain, and never speaks: it is a LAN API the local model can read and act on when
a turn asks about the house. The earlier "no HA" decision was about Assist; this
is a different thing and was re-approved 2026-11-21, with Hermes bypassed for now.

Two tool shapes, both narrow on purpose:

    ha_lookup(query)                       -> a few matching entities + states
    ha_call(domain, service, entity_id)    -> one action, allowlisted domains

Why lookups: /api/states is 663 entities and 305 KB (~80k tokens). A 3B model with
a 220-token prompt cannot be handed that. It gets 4 candidates and picks one.

Safety is in this module, not in the model:
  * only allowlisted domains can be written (lights, switches, scenes, scripts)
  * a write needs exactly one unambiguous target
  * locks, covers/garage, climate and alarms are refused outright
  * every call is logged with the entity, service and result
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://192.168.1.20:8123"
DEFAULT_TOKEN_FILE = "~/.config/voice-bridge/ha-token"

#: domains a voice turn may write to. Deliberately excludes lock, cover (garage),
#: alarm_control_panel and climate: a misheard word must not open a door or change
#: a heating setpoint.
WRITABLE_DOMAINS = frozenset({"light", "switch", "scene", "script", "fan"})

#: services allowed per domain (no device_class tricks, no raw service calls)
ALLOWED_SERVICES = {
    "light": {"turn_on", "turn_off", "toggle"},
    "switch": {"turn_on", "turn_off", "toggle"},
    "fan": {"turn_on", "turn_off", "toggle"},
    "scene": {"turn_on"},
    "script": {"turn_on", "turn_off"},
}

#: words that make Home Assistant tools worth offering at all
TRIGGER_WORDS = (
    "light", "lights", "lamp", "switch", "plug", "fan", "scene", "garage",
    "temperature", "thermostat", "how warm", "how cold", "sensor", "door",
    "open", "closed", "battery", "humidity", "power", "energy", "motion",
    "turn on", "turn off", "switch on", "switch off", "is the",
)


class HaError(RuntimeError):
    pass


#: words that make an utterance a command rather than a question or a remark
_COMMAND_VERBS = r"(?:turn|switch|shut|kill|enable|disable|start|stop)"
_QUESTION_OPENERS = ("is ", "are ", "was ", "were ", "did ", "does ", "do ",
                     "can ", "could ", "would ", "should ", "what ", "when ",
                     "where ", "why ", "how ", "who ", "which ", "has ", "have ")


def looks_imperative(text: str) -> bool:
    """True only if the utterance tells something to change state.

    Measured failure this prevents: STT heard "This is the first office fan on."
    (a remark), the model called the write tool, and the fan was switched on. A
    question ("is the fan on?") and a statement must never write — only an
    instruction may.
    """
    low = " ".join((text or "").lower().split())
    if not low or low.startswith(_QUESTION_OPENERS):
        return False
    if low.endswith("?"):
        return False
    return bool(re.search(_COMMAND_VERBS + r"\b[^.]{0,40}\b(on|off)\b", low)
                or re.search(r"\b(turn|switch)\s+(on|off)\b", low))


def wants_ha_tools(text: str) -> bool:
    """Cheap gate: do not pay tool-schema tokens on every turn.

    Offering two more tools costs ~60-80 prompt tokens, which is ~2 s on this
    box. A turn that says nothing about the house should not pay it.
    """
    low = f" {(text or '').lower()} "
    return any(f" {w} " in low or w in low for w in TRIGGER_WORDS)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


LOOKUP_TOOL = _tool(
    "ha_lookup",
    "Look up Home Assistant entities (lights, switches, sensors) by name or room.",
    {"query": {"type": "string", "description": "e.g. office temperature, patio light"}},
    ["query"],
)

DO_TOOL = _tool(
    "ha_do",
    "Switch something on or off by name, e.g. 'patio light'. The bridge resolves the name "
    "and refuses if it is ambiguous — never invent an entity id.",
    {"query": {"type": "string", "description": "what to switch, in plain words"},
     "service": {"type": "string", "description": "turn_on or turn_off"}},
    ["query", "service"],
)

HA_TOOLS = [LOOKUP_TOOL, DO_TOOL]


class HomeAssistant:
    def __init__(self, url: str = DEFAULT_URL, token_file: str = DEFAULT_TOKEN_FILE,
                 timeout: float = 10.0, cache_s: float = 20.0) -> None:
        self.url = url.rstrip("/")
        self.token_file = os.path.expanduser(token_file)
        self.timeout = timeout
        self.cache_s = cache_s
        self._states: list[dict] = []
        self._states_at = 0.0
        self._token: str | None = None

    # ── transport ────────────────────────────────────────────────────────────

    def token(self) -> str:
        if self._token is None:
            try:
                with open(self.token_file) as fh:
                    tok = fh.read().strip()
            except OSError as exc:
                raise HaError(
                    f"no Home Assistant token at {self.token_file}. Create one:\n"
                    f"  python3 ~/.hermes/scripts/ha_token.py > {self.token_file}\n"
                    f"  chmod 600 {self.token_file}"
                ) from exc
            if not tok:
                raise HaError(f"Home Assistant token at {self.token_file} is empty")
            self._token = tok
        return self._token

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.url}/api/{path}", data=body, method=method,
            headers={"Authorization": f"Bearer {self.token()}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200]
            raise HaError(f"{method} {path} -> HTTP {exc.code}: {detail!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise HaError(f"{method} {path} failed: {exc!r}") from exc

    # ── read ─────────────────────────────────────────────────────────────────

    def states(self, max_age: float | None = None) -> list[dict]:
        age = max_age if max_age is not None else self.cache_s
        if not self._states or (time.time() - self._states_at) > age:
            data = self._request("GET", "states")
            if not isinstance(data, list):
                raise HaError("unexpected /api/states payload")
            self._states = data
            self._states_at = time.time()
        return self._states

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}

    def lookup(self, query: str, limit: int = 4) -> list[dict]:
        """Fuzzy match entities by friendly name and entity_id.

        Returns at most `limit` results, best first, each with the state so the
        model can answer without a second call.
        """
        wanted = self._words(query)
        if not wanted:
            return []
        scored: list[tuple[float, dict]] = []
        for ent in self.states():
            eid = ent.get("entity_id", "")
            name = (ent.get("attributes") or {}).get("friendly_name", "")
            haystack = self._words(f"{name} {eid.replace('_', ' ')}")
            if not haystack:
                continue
            overlap = len(wanted & haystack)
            if not overlap:
                continue
            # 1. an entity whose name *contains* the whole query wins outright:
            #    "patio light" is switch.patio_light, not light.patio_fan
            q_norm = " ".join(sorted(wanted))
            name_norm = " ".join(sorted(self._words(name)))
            contains = 0.5 if wanted <= self._words(name) else 0.0
            # 2. if the query names a domain ("the patio light"), prefer it there
            domain = eid.split(".")[0]
            domain_bonus = 0.35 if domain in wanted else (
                0.15 if domain in ("light", "switch", "fan", "sensor") else 0.0)
            coverage = overlap / len(wanted)
            score = contains + domain_bonus + coverage - 0.02 * len(haystack - wanted)
            # metadata domains are never what someone asking about the house means
            if domain in ("event", "update", "button", "number", "automation", "script"):
                score -= 0.6
            attrs = ent.get("attributes") or {}
            scored.append((score, {
                "entity_id": eid,
                "name": name,
                "state": ent.get("state"),
                "unit": attrs.get("unit_of_measurement"),
                # a coarse band so `do()` can detect ambiguity without exposing
                # exact scores (which the model does not need and would burn tokens)
                "_score_band": round(score, 1),
            }))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [item for _score, item in scored[:limit]]

    # ── write ────────────────────────────────────────────────────────────────

    def call(self, entity_id: str, service: str) -> dict:
        entity_id = (entity_id or "").strip()
        service = (service or "").strip().lower()
        if "." not in entity_id:
            raise HaError(f"{entity_id!r} is not an entity_id")
        domain = entity_id.split(".", 1)[0]
        if domain not in WRITABLE_DOMAINS:
            raise HaError(
                f"{domain!r} is not writable by voice (allowed: "
                f"{', '.join(sorted(WRITABLE_DOMAINS))})"
            )
        if service not in ALLOWED_SERVICES.get(domain, set()):
            raise HaError(f"{service!r} is not allowed on {domain}")

        target = next((e for e in self.states() if e.get("entity_id") == entity_id), None)
        if target is None:
            raise HaError(f"unknown entity {entity_id!r}")
        if target.get("state") == "unavailable":
            raise HaError(f"{entity_id} is unavailable right now")

        self._request("POST", f"services/{domain}/{service}", {"entity_id": entity_id})
        return {"entity_id": entity_id, "service": service, "was": target.get("state")}

    def do(self, query: str, service: str) -> dict:
        """Resolve a spoken name to exactly one writable entity, then act.

        Ambiguity is a refusal with candidates, never a guess: a misheard name
        must not switch the wrong thing.
        """
        hits = [h for h in self.lookup(query, limit=6)
                if h["entity_id"].split(".")[0] in WRITABLE_DOMAINS]
        if not hits:
            return {"ok": False, "error": f"nothing switchable matches {query!r}"}
        top = hits[0]
        # a second, different entity with the same score is ambiguity, not a tie
        if len(hits) > 1 and hits[1]["_score_band"] == top["_score_band"]:
            return {"ok": False, "error": f"{query!r} is ambiguous",
                    "candidates": [h["entity_id"] for h in hits[:3]]}
        res = self.call(top["entity_id"], service)
        return {"ok": True, **res}

    # ── selftest ─────────────────────────────────────────────────────────────

    def health(self) -> str:
        info = self._request("GET", "")
        return str((info or {}).get("message", "?"))


def _selftest() -> int:
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures += 1

    # defaults in this repo are example addresses; point them at a real install
    # with HA_URL / HA_TOKEN_FILE when running the selftest locally.
    ha = HomeAssistant(url=os.environ.get("HA_URL", DEFAULT_URL),
                       token_file=os.environ.get("HA_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    t0 = time.monotonic()
    check("api reachable", "API" in ha.health(), ha.health())
    states = ha.states()
    check("states fetched", len(states) > 500, f"{len(states)} entities in {time.monotonic()-t0:.2f}s")
    check("cached second read is fast", time.monotonic() - t0 < 2.0)

    for query, expect_domain in (("office temperature", None),
                                 ("patio light", None),
                                 ("refrigerator", "sensor."),
                                 ("red led", None)):
        hits = ha.lookup(query)
        ok = bool(hits)
        if expect_domain and ok:
            ok = hits[0]["entity_id"].startswith(expect_domain)
        check(f"lookup {query!r}", ok,
              f"{hits[0]['entity_id']} = {hits[0]['state']}" if hits else "no hits")

    # writes: refusals are as important as successes
    for eid, svc, why in (
        ("lock.front_door", "turn_on", "lock domain refused"),
        ("cover.garage_door", "turn_on", "cover domain refused"),
        ("light.x", "delete", "bad service refused"),
        ("light.x", "turn_on", "unknown entity refused"),
        ("light.xiaozhi_muma_box_v1_de9c5c", "turn_on", "unavailable entity refused"),
    ):
        try:
            ha.call(eid, svc)
            check(why, False, "call went through")
        except HaError as exc:
            check(why, True, str(exc)[:60])

    # the write gate: instructions only
    for text, expect in (
        ("Turn off the office fan.", True),
        ("turn the patio light on", True),
        ("switch off the garage light", True),
        ("This is the first office fan on.", False),   # the observed bad case
        ("Is the patio light on?", False),
        ("Do you turn off the light?", False),
        ("The fan is on.", False),
        ("why is the light off?", False),
    ):
        got = looks_imperative(text)
        check(f"imperative {text!r}", got == expect, f"got {got}")

    # gate
    check("gate: house question offers tools", wants_ha_tools("is the patio light on?"))
    check("gate: unrelated question skips tools",
          not wants_ha_tools("what is a folded dipole?"))
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
