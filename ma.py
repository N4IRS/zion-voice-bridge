#!/usr/bin/env python3
"""ma.py — Music Assistant as a tool for the local voice lane, direct over its WS API.

No Home Assistant in the path: Music Assistant (MA) exposes a WebSocket API on
:8095, and that is what this speaks. HA is used only where MA has nothing to offer.

Command names here were **measured** against MA 2.10.12, not recalled:

    auth                                {"token": ...}          -> {authenticated: true}
    players/all                                                 -> players[]  (has volume_level)
    player_queues/all                                           -> queues[]   (state, current_item)
    music/search                        {"search_query", "limit"}
                                        -> {tracks[], albums[], playlists[], radio[], ...}
    player_queues/play_media            {"queue_id", "media": <item|list>, "option": "replace"}
    player_queues/play_pause            {"queue_id"}
    player_queues/stop                  {"queue_id"}

Two shapes that cost a round trip each to learn:

* `play_media` wants a **media item object** (as returned by `music/search`), not a
  `provider://type/id` string — a URI gives "Value of type NoneType is invalid for media".
* playback is addressed by **queue_id**, not player_id. For these players the two ids
  are identical, which is exactly the kind of coincidence that breaks later.

Safety: targets are resolved from words against the live queue list and refused if
ambiguous; nothing is guessed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

DEFAULT_URL = "ws://192.168.1.20:8095/ws"
DEFAULT_TOKEN_FILE = "~/.config/voice-bridge/ma-token"
DEFAULT_TARGET = "Satellite-1"

#: words a person might use for a player, mapped to substrings of its display name.
#: matched against display_name, lowercased; first match wins, longer aliases first.
TARGET_ALIASES = {
    "dot": "satellite", "satellite": "satellite", "echo": "satellite", "kitchen dot": "satellite",
    "esp32": "esp32", "a1s": "esp32", "esp": "esp32",
    "office speaker": "office speaker", "office": "office speaker",
    "muma": "muma", "xiaozhi": "muma",
    "tv": None,                      # ambiguous on purpose: two TVs
    "roku": None,
}

TRACK, ALBUM, PLAYLIST, RADIO = "track", "album", "playlist", "radio"

#: words that make the music tools worth offering. Two more tool schemas cost ~2 s
#: of prompt on this box, so a question about antennas should not pay for them.
MUSIC_WORDS = ("play ", "music", " song", "album", "artist", "playlist", "radio",
               "pause", "resume", "volume", "skip", "next track", "stop the music",
               "louder", "quieter", "turn it up", "turn it down")


def wants_music_tools(text: str) -> bool:
    low = f" {(text or '').lower()} "
    return any(word in low for word in MUSIC_WORDS)


class MaError(RuntimeError):
    pass


class MusicAssistant:
    def __init__(self, url: str = DEFAULT_URL, token_file: str = DEFAULT_TOKEN_FILE,
                 default_target: str = DEFAULT_TARGET, timeout: float = 15.0) -> None:
        self.url = url
        self.token_file = os.path.expanduser(token_file)
        self.default_target = default_target
        self.timeout = timeout
        self._session = None
        self._ws = None
        self._mid = 0
        self._lock = asyncio.Lock()
        self._token: str | None = None

    # ── transport ────────────────────────────────────────────────────────────

    def token(self) -> str:
        if self._token is None:
            try:
                with open(self.token_file) as fh:
                    tok = fh.read().strip()
            except OSError as exc:
                raise MaError(
                    f"no Music Assistant token at {self.token_file}. Create one in the "
                    f"MA UI (Settings -> API keys) and save it there (chmod 600)."
                ) from exc
            if not tok:
                raise MaError(f"Music Assistant token at {self.token_file} is empty")
            self._token = tok
        return self._token

    async def connect(self) -> None:
        import aiohttp

        if self._ws is not None and not self._ws.closed:
            return
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientWSTimeout(ws_close=self.timeout)
        except AttributeError:            # older aiohttp
            timeout = self.timeout
        self._ws = await self._session.ws_connect(self.url, timeout=timeout, heartbeat=30)
        await self._ws.receive()                       # server info on connect
        reply = await self.command("auth", {"token": self.token()})
        if not (isinstance(reply, dict) and reply.get("authenticated")):
            raise MaError(f"Music Assistant rejected the token: {reply!r}")

    async def aclose(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def command(self, command: str, args: dict | None = None, *, _retry: bool = True):
        """Send one command and return its result (or raise MaError)."""
        await self._ensure_socket()
        async with self._lock:
            self._mid += 1
            mid = str(self._mid)
            payload = {"message_id": mid, "command": command}
            if args is not None:
                payload["args"] = args
            try:
                await self._ws.send_json(payload)
                while True:
                    msg = await asyncio.wait_for(self._ws.receive(), timeout=self.timeout)
                    if msg.type != 1:              # TEXT
                        raise MaError(f"connection closed while waiting for {command}")
                    data = json.loads(msg.data)
                    if str(data.get("message_id")) != mid:
                        continue                    # unsolicited event; ignore
                    if "error_code" in data:
                        raise MaError(f"{command}: {data.get('details')} ({data['error_code']})")
                    return data.get("result")
            except (asyncio.TimeoutError, ConnectionError) as exc:
                await self._drop_socket()
                if _retry:
                    await self.connect()
                    return await self.command(command, args, _retry=False)
                raise MaError(f"{command} failed: {exc!r}") from exc

    async def _ensure_socket(self) -> None:
        if self._ws is None or self._ws.closed:
            await self.connect()

    async def _drop_socket(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - closing a broken socket
                pass
            self._ws = None

    # ── discovery ────────────────────────────────────────────────────────────

    async def queues(self) -> list[dict]:
        return await self.command("player_queues/all") or []

    async def players(self) -> list[dict]:
        return await self.command("players/all") or []

    @staticmethod
    def resolve_in(queues: list[dict], hint: str | None, default: str) -> tuple[str, str]:
        want = (hint or default or "").strip().lower()
        if not want:
            raise MaError("no music target given and no default configured")
        # exact display-name match first
        for q in queues:
            if (q.get("display_name") or "").lower() == want:
                return q["queue_id"], q["display_name"]
        # alias -> substring
        needle = want
        for alias, target in TARGET_ALIASES.items():
            if alias in want:
                if target is None:
                    raise MaError(f"{hint!r} matches more than one player; say which room")
                needle = target
                break
        matches = [q for q in queues if needle in (q.get("display_name") or "").lower()]
        if len(matches) == 1:
            return matches[0]["queue_id"], matches[0]["display_name"]
        if not matches:
            known = ", ".join(sorted({q.get("display_name", "") for q in queues}))
            raise MaError(f"no player matches {hint!r}. Known players: {known}")
        names = ", ".join(m["display_name"] for m in matches)
        raise MaError(f"{hint!r} is ambiguous: {names}")

    # ── actions ──────────────────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 3) -> dict:
        return await self.command("music/search", {"search_query": query, "limit": limit}) or {}

    @staticmethod
    def candidates(results: dict) -> list[dict]:
        """Playable candidates in preference order: track, album, playlist, radio."""
        out: list[dict] = []
        for kind in (TRACK + "s", ALBUM + "s", PLAYLIST + "s", RADIO):
            out.extend(results.get(kind) or [])
        return out

    @staticmethod
    def best_item(results: dict) -> dict | None:
        items = MusicAssistant.candidates(results)
        return items[0] if items else None

    async def play(self, query: str, target: str | None = None) -> dict:
        if not (query or "").strip():
            raise MaError("nothing to play")
        queues = await self.queues()
        queue_id, name = self.resolve_in(queues, target, self.default_target)
        results = await self.search(query)
        candidates = self.candidates(results)
        if not candidates:
            raise MaError(f"Music Assistant found nothing for {query!r}")
        # A search hit is not a promise: a library entry from an unconfigured or
        # unavailable provider is returned by search and then refused by play_media
        # ("the requested media item could not be found"). Walk the chain.
        last_error: Exception | None = None
        for item in candidates[:5]:
            try:
                await self.command("player_queues/play_media",
                                   {"queue_id": queue_id, "media": item,
                                    "option": "replace"})
                return {"target": name, "playing": item.get("name") or item.get("item_id"),
                        "media_type": item.get("media_type")}
            except MaError as exc:
                last_error = exc
                continue
        raise MaError(f"nothing playable for {query!r} ({last_error})")

    async def transport(self, action: str, target: str | None = None) -> dict:
        action = (action or "").strip().lower().replace(" ", "_")
        commands = {"play": "player_queues/play", "pause": "player_queues/pause",
                    "play_pause": "player_queues/play_pause", "stop": "player_queues/stop",
                    "next": "player_queues/next", "previous": "player_queues/previous"}
        if action not in commands:
            raise MaError(f"unknown transport action {action!r}; "
                          f"use one of {', '.join(sorted(commands))}")
        queues = await self.queues()
        queue_id, name = self.resolve_in(queues, target, self.default_target)
        await self.command(commands[action], {"queue_id": queue_id})
        return {"target": name, "action": action}

    async def volume(self, level: int, target: str | None = None) -> dict:
        queues = await self.queues()
        queue_id, name = self.resolve_in(queues, target, self.default_target)
        level = max(0, min(100, int(level)))
        await self.command("players/cmd/volume_set", {"player_id": queue_id,
                                                     "volume_level": level})
        return {"target": name, "volume": level}

    # ── tools the model may call ─────────────────────────────────────────────

    @staticmethod
    def tools() -> list[dict]:
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required}}}
        return [
            fn("music_play", "Play music, an artist, an album or a radio station by name.",
               {"query": {"type": "string", "description": "what to play, in plain words"},
                "target": {"type": "string", "description": "optional speaker, e.g. 'dot' or 'esp32'"}},
               ["query"]),
            fn("music_transport", "Pause, resume, stop, or skip the music.",
               {"action": {"type": "string", "description": "play, pause, play_pause, stop, next, previous"},
                "target": {"type": "string", "description": "optional speaker"}},
               ["action"]),
            fn("music_volume", "Set the music volume on a speaker.",
               {"level": {"type": "integer", "description": "0 to 100"},
                "target": {"type": "string", "description": "optional speaker"}},
               ["level"]),
        ]


# ── selftest ─────────────────────────────────────────────────────────────────

async def _selftest() -> int:
    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures += 1

    ma = MusicAssistant(url=os.environ.get("MA_URL", DEFAULT_URL),
                        token_file=os.environ.get("MA_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    await ma.connect()
    check("authenticated", True)

    players = await ma.players()
    check("players listed", len(players) > 0, f"{len(players)} players")
    avail = [p for p in players if p.get("available")]
    check("some players available", len(avail) > 0,
          ", ".join(p["display_name"] for p in avail)[:60])

    queues = await ma.queues()
    check("queues listed", len(queues) > 0, f"{len(queues)} queues")

    # target resolution: words in, queue out, ambiguity refused
    default_id, default_name = MusicAssistant.resolve_in(queues, None, DEFAULT_TARGET)
    check("default target resolves", bool(default_id), default_name)
    for hint, expect_sub in (("dot", "satellite"), ("on the esp32", "esp32"),
                             ("a1s", "esp32")):
        try:
            _qid, name = MusicAssistant.resolve_in(queues, hint, DEFAULT_TARGET)
            check(f"target {hint!r}", expect_sub in name.lower(), name)
        except MaError as exc:
            check(f"target {hint!r}", False, str(exc)[:50])
    for hint in ("tv", "roku", "the moon"):
        try:
            MusicAssistant.resolve_in(queues, hint, DEFAULT_TARGET)
            check(f"target {hint!r} refused", False, "resolved instead of refusing")
        except MaError as exc:
            check(f"target {hint!r} refused", True, str(exc)[:44])

    # search is read-only
    res = await ma.search("pink floyd")
    item = MusicAssistant.best_item(res)
    check("search returns something playable", item is not None,
          f"{item.get('name','')[:40]}" if item else "nothing")

    await ma.aclose()
    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(_selftest()))
