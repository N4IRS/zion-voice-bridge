#!/usr/bin/env python3
"""lanes.py — answer lanes for the voice bridge.

A *lane* is a place a turn can be answered from. The point of this module is the
`sync_capable` flag: some lanes can answer inside the turn (a local model, a
cloud model over HTTP), some cannot (Muse over the agent bus, where delivery is
measured in minutes). The bridge asks the flag rather than hard-coding which is
which, so a lane can move between the two classes without touching the turn code.

    lane                     sync_capable  shape of an answer
    -----------------------  ------------  ------------------------------------
    HttpLane (Ollama/OmniRoute)   True      ask() -> Reply, seconds
    BusLane  (Muse/any agent)     False     submit() -> correlation_id, then
                                           the reply arrives later for announcing

Both HTTP shapes are OpenAI-compatible, so the same class serves a local Ollama
(`http://127.0.0.1:11434/v1`) and a cloud gateway (`http://127.0.0.1:20129/v1`);
only base_url, model and key differ.

BusLane is deliberately NOT wired into the turn flow yet (no delivery path to
`muse` exists on this host — bus-dispatch.py handles only opencode/hermes). It is
implemented and testable so that turning it on is a config change, not a rewrite.

    python3 lanes.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

LANE_LOCAL_SYSTEM_PROMPT = (
    "You are Zion, the local voice assistant on the owner's amateur radio workshop "
    "machine. "
    "Answer out loud in one or two short sentences, under 25 words. "
    "No lists, no markdown, no URLs, no code, no spelling things out. "
    "If you do not know, say so in a few words."
)


# ── results and errors ───────────────────────────────────────────────────────

@dataclass
class Reply:
    text: str
    lane: str
    latency_s: float
    stats: dict[str, Any] = field(default_factory=dict)
    #: OpenAI-shaped tool calls the model asked for instead of answering
    tool_calls: list[dict] = field(default_factory=list)


class LaneError(RuntimeError):
    """A lane could not produce an answer."""


class NotSyncCapable(LaneError):
    """This lane cannot answer inside a turn — submit() and wait instead."""


class NotAsyncCapable(LaneError):
    """This lane answers immediately; there is nothing to submit."""


# ── base ─────────────────────────────────────────────────────────────────────

class Lane:
    """One place an answer can come from."""

    sync_capable: bool = True
    #: human-readable note about why a lane is (not) synchronous
    latency_note: str = ""
    #: how this lane is named out loud when attributing an answer
    spoken_name: str = "the local model"

    def __init__(self, name: str, spoken_name: str | None = None) -> None:
        self.name = name
        if spoken_name:
            self.spoken_name = spoken_name

    async def aclose(self) -> None:
        return None

    # synchronous lanes
    async def ask(self, text: str, history: list[dict] | None = None) -> Reply:
        raise NotSyncCapable(
            f"lane {self.name!r} is not sync-capable; use submit() and announce the reply"
        )

    # asynchronous lanes
    async def submit(self, text: str, *, device: str | None = None,
                     history: list[dict] | None = None,
                     envelope: dict | None = None) -> str:
        raise NotAsyncCapable(f"lane {self.name!r} answers inside the turn; use ask()")

    def describe(self) -> str:
        kind = "sync " if self.sync_capable else "async"
        note = f"  ({self.latency_note})" if self.latency_note else ""
        return f"{kind}  {self.name}{note}"


# ── HTTP lane: any OpenAI-compatible endpoint ─────────────────────────────────

class HttpLane(Lane):
    """`POST {base_url}/chat/completions`.

    Serves a local Ollama (`/v1`) and a cloud gateway (`OmniRoute /v1`) with the
    same code; only the endpoint and model differ.
    """

    sync_capable = True

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        spoken_name: str | None = None,
        system_prompt: str = LANE_LOCAL_SYSTEM_PROMPT,
        max_tokens: int = 80,
        # 0.0, not 0.4: on the SWR question this model answers "1.25 dB"
        # deterministically at 0.0 and slipped to "6 dB" on one of four runs at
        # 0.4. Factual voice answers should not sample.
        temperature: float = 0.0,
        timeout: float = 25.0,
    ) -> None:
        super().__init__(name, spoken_name)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._session = None

    async def _get_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _messages(self, text: str, history: list[dict] | None) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for turn in history or []:
            msgs.append(turn)
        msgs.append({"role": "user", "content": text})
        return msgs

    async def ask(self, text: str, history: list[dict] | None = None,
                  tools: list[dict] | None = None) -> Reply:
        return await self.chat(self._messages(text, history), tools=tools)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        """One model step over an explicit message list (for tool loops)."""
        session = await self._get_session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        t0 = time.monotonic()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise LaneError(f"{self.name}: HTTP {resp.status}: {body[:200]}")
                data = json.loads(body)
        except asyncio.TimeoutError as exc:
            raise LaneError(f"{self.name}: timed out after {self.timeout:.0f}s") from exc
        elapsed = time.monotonic() - t0

        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        reply = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        usage = data.get("usage") or {}
        stats = {
            "model": data.get("model", self.model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "reply_tokens": usage.get("completion_tokens"),
        }
        if not reply and not tool_calls:
            raise LaneError(f"{self.name}: empty reply")
        return Reply(text=reply, lane=self.name, latency_s=elapsed, stats=stats,
                     tool_calls=tool_calls)


# ── bus lane: an agent, whose answer arrives later ────────────────────────────

class BusLane(Lane):
    """Submit a question to another agent over the agent bus.

    sync_capable is False *by measurement, not by preference*: bus-dispatch.py
    handles only `opencode` and `hermes`, and `muse` has not polled since
    2026-11-13, so a reply takes minutes at best. Such an answer cannot be spoken
    in-turn; it is announced when it lands (a separate, already-proven path).
    """

    sync_capable = False
    latency_note = "agent bus: minutes, not seconds"

    def __init__(
        self,
        name: str = "muse",
        *,
        bus_url: str = "http://192.168.1.10:8765",
        token_path: str = "~/.agent-bus/tokens/bridge",
        to: str = "muse",
        spoken_name: str | None = None,
        #: a distinct type is what lets Muse's hook ignore everything else in
        #: her inbox (old results, stale pings) and act only on voice turns
        msg_type: str = "voice-handoff",
        timeout: float = 15.0,
        consume: bool = False,
    ) -> None:
        super().__init__(name, spoken_name or to.capitalize() if to else None)
        self.bus_url = bus_url.rstrip("/")
        self.token_path = os.path.expanduser(token_path)
        self.to = to
        self.msg_type = msg_type
        self.timeout = timeout
        # peek-only by default: reading an inbox non-destructively leaves the
        # bus's own delivery accounting intact, which matters when other agents
        # may also be reading.
        self.consume = consume
        self.last_message_id: int | None = None
        self._session = None

    def _token(self) -> str:
        try:
            with open(self.token_path) as fh:
                token = fh.read().strip()
        except OSError as exc:
            raise LaneError(
                f"no bus token at {self.token_path}. Mint one and restart the bus "
                f"server (it reads tokens/ at startup):\n"
                f"  openssl rand -hex 32 > {self.token_path} && chmod 600 {self.token_path}"
            ) from exc
        if not token:
            raise LaneError(f"bus token at {self.token_path} is empty")
        return token

    async def _get_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def submit(self, text: str, *, device: str | None = None,
                     history: list[dict] | None = None,
                     envelope: dict | None = None) -> str:
        """Queue the question. Returns the bus correlation_id to match later."""
        session = await self._get_session()
        device = device or "satellite"
        body = text
        if history:
            recent = " / ".join(
                f"{m.get('role')}: {str(m.get('content'))[:120]}" for m in history[-4:]
            )
            body = f"{text}\n\n[voice turn from {device or 'satellite'}; context: {recent}]"
        else:
            body = f"{text}\n\n[voice turn from {device or 'satellite'}]"
        # The reply is spoken aloud, un-interruptible, on a small speaker. Say so
        # in the message: a general agent otherwise answers with a bulleted list.
        body += ("\n[Speakable answer requested: one or two short sentences, plain "
                 "text, no lists, no markdown, no URLs. Include the key result.]")
        if envelope is not None:
            # the envelope IS the contract; the bus only carries it
            import json as _json
            body = _json.dumps(envelope)
        payload = {
            "to": self.to,
            "type": self.msg_type,
            "subject": f"voice: {device}",
            "body": body,
        }
        if envelope is not None:
            payload["correlation_id"] = envelope.get("id")
        headers = {"X-Agent-Token": self._token(), "Content-Type": "application/json"}
        try:
            async with session.post(f"{self.bus_url}/send", json=payload,
                                    headers=headers) as resp:
                raw = await resp.text()
                if resp.status != 200:
                    raise LaneError(f"{self.name}: bus send HTTP {resp.status}: {raw[:200]}")
                data = json.loads(raw)
        except asyncio.TimeoutError as exc:
            raise LaneError(f"{self.name}: bus send timed out") from exc
        corr = data.get("correlation_id")
        if not corr:
            raise LaneError(f"{self.name}: bus returned no correlation_id: {raw[:200]}")
        # Remember both ids: our own correlation_id and the bus's numeric id,
        # because a responder's reply may cite either one.
        ids = data.get("ids") if isinstance(data.get("ids"), list) else []
        self.last_message_id = data.get("id") or (ids[0] if ids else None)
        return corr

    async def poll(self, correlation_ids: list[str] | None = None) -> list[dict]:
        """Return replies addressed back to us, newest last.

        Matches on `in_reply_to` when correlation ids are given. Peek-only
        unless `consume=True`.
        """
        session = await self._get_session()
        params = "" if self.consume else "?peek=1"
        headers = {"X-Agent-Token": self._token()}
        try:
            async with session.get(f"{self.bus_url}/inbox/{self._me()}{params}",
                                   headers=headers) as resp:
                raw = await resp.text()
                if resp.status != 200:
                    raise LaneError(f"{self.name}: inbox HTTP {resp.status}: {raw[:200]}")
                data = json.loads(raw)
        except asyncio.TimeoutError as exc:
            raise LaneError(f"{self.name}: inbox poll timed out") from exc
        if isinstance(data, dict):
            messages = data.get("messages") or []
        else:
            messages = data or []
        messages = [m for m in messages if isinstance(m, dict)]
        if correlation_ids:
            wanted = {str(c) for c in correlation_ids}
            def _matches(m: dict) -> bool:
                # the reply may carry our correlation_id, or only the bus id it
                # answers -- dispatch uses the numeric id, so accept both.
                if str(m.get("correlation_id")) in wanted:
                    return True
                return str(m.get("in_reply_to")) in wanted
            messages = [m for m in messages if _matches(m)]
        return messages

    def _me(self) -> str:
        # the bus derives identity from the token; the inbox path needs the name,
        # which for this project is the token file's basename.
        return os.path.basename(self.token_path)


# ── the set of lanes a bridge has ────────────────────────────────────────────

class Lanes:
    """The configured lanes, and the only place that asks `sync_capable`."""

    def __init__(self, lanes: list[Lane]) -> None:
        self.lanes = {lane.name: lane for lane in lanes}

    def __getitem__(self, name: str) -> Lane:
        return self.lanes[name]

    def __iter__(self):
        return iter(self.lanes.values())

    @property
    def sync(self) -> list[Lane]:
        return [lane for lane in self.lanes.values() if lane.sync_capable]

    @property
    def async_(self) -> list[Lane]:
        return [lane for lane in self.lanes.values() if not lane.sync_capable]

    def answer_lane(self, name: str) -> Lane:
        """The lane a turn should ask directly — refuses an async lane loudly."""
        lane = self.lanes[name]
        if not lane.sync_capable:
            raise NotSyncCapable(
                f"{name!r} cannot answer inside a turn ({lane.latency_note}); "
                f"submit() it instead and announce the reply when it lands"
            )
        return lane

    def describe(self) -> str:
        return "\n".join(lane.describe() for lane in self.lanes.values())

    def routing_table(self) -> dict[str, str]:
        """For logging/telemetry: lane -> 'sync' | 'async'."""
        return {
            lane.name: ("sync" if lane.sync_capable else "async")
            for lane in self.lanes.values()
        }

    async def aclose(self) -> None:
        for lane in self.lanes.values():
            await lane.aclose()


# ── selftest ─────────────────────────────────────────────────────────────────

async def _selftest() -> int:
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures += 1

    http = HttpLane("local", "http://127.0.0.1:11434/v1", "qwen2.5:3b")
    bus = BusLane("muse", token_path="/nonexistent/token")
    lanes = Lanes([http, bus])

    # 1. the flag contract
    check("http lane is sync_capable", http.sync_capable is True)
    check("bus lane is NOT sync_capable", bus.sync_capable is False)
    check("one sync lane", [l.name for l in lanes.sync] == ["local"])
    check("one async lane", [l.name for l in lanes.async_] == ["muse"])
    check("routing table", lanes.routing_table() == {"local": "sync", "muse": "async"},
          str(lanes.routing_table()))

    # 2. an async lane must refuse to answer in-turn, loudly
    try:
        await bus.ask("what is a dipole?")
        check("bus.ask() raises NotSyncCapable", False, "no exception")
    except NotSyncCapable as exc:
        check("bus.ask() raises NotSyncCapable", True, str(exc)[:60])

    # 3. the router refuses it too, rather than silently answering elsewhere
    try:
        lanes.answer_lane("muse")
        check("Lanes.answer_lane('muse') refuses", False, "no exception")
    except NotSyncCapable:
        check("Lanes.answer_lane('muse') refuses", True)

    # 4. a sync lane answers -> the local path still works end to end
    try:
        reply = await lanes.answer_lane("local").ask("What is a dipole antenna?")
        check("local lane answers", bool(reply.text), f"{reply.latency_s:.2f}s {reply.text[:60]!r}")
        check("reply carries usage", reply.stats.get("reply_tokens") is not None,
              str(reply.stats))
    except LaneError as exc:
        check("local lane answers", False, f"{exc}")

    # 5. submit builds the right shape without needing a live bus
    payloads: list[tuple[str, dict]] = []
    captured: dict[str, str] = {}

    class _FakeResp:
        status = 200

        def __init__(self, body: str) -> None:
            self._body = body

        async def text(self) -> str:
            return self._body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def post(self, url, json=None, headers=None):
            payloads.append((url, json))
            captured["token"] = (headers or {}).get("X-Agent-Token", "")
            return _FakeResp('{"id": 1, "correlation_id": "corr-123"}')

        def get(self, url, headers=None):
            return _FakeResp('{"messages": [{"in_reply_to": "corr-123", "body": "hello"}]}')

    bus2 = BusLane("muse", token_path="/tmp/opencode/_selftest_token")
    with open("/tmp/opencode/_selftest_token", "w") as fh:
        fh.write("deadbeef\n")
    bus2._session = _FakeSession()
    corr = await bus2.submit("price a duplexer", device="Satellite-1")
    check("submit returns correlation_id", corr == "corr-123", corr)
    url, json_body = payloads[0]
    check("send path", url.endswith("/send"), url)
    check("send type is voice-handoff", json_body.get("type") == "voice-handoff",
          str(json_body.get("type")))
    check("send to", json_body.get("to") == "muse")
    check("subject names the device", "Satellite-1" in (json_body.get("subject") or ""))
    check("body flags a voice turn", "[voice turn" in (json_body.get("body") or ""))
    check("token sent from file", captured["token"] == "deadbeef")
    got = await bus2.poll(["corr-123"])
    check("poll matches in_reply_to", len(got) == 1 and got[0]["body"] == "hello", str(got))

    # 6. a missing token is a clear error, not a crash
    try:
        await bus.submit("x")
        check("missing token raises LaneError", False, "no exception")
    except LaneError as exc:
        check("missing token raises LaneError", "no bus token" in str(exc), str(exc)[:50])

    await lanes.aclose()
    print()
    print(lanes.describe())
    print()
    print("FAILURES:", failures)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="verify the lane contract")
    args = p.parse_args(argv)
    if args.selftest:
        return asyncio.run(_selftest())
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
