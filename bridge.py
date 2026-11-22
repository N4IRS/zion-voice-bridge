#!/usr/bin/env python3
"""voice-bridge — a local voice backend for the repurposed Echo Dots.

It takes Home Assistant's seat on the EchoMuse controller's ESPHome voice
satellite port and answers with a fully local stack:

    Dot mic -> controller -> [this] -> faster-whisper -> Ollama -> piper -> Dot

Nothing leaves the LAN.

Stages can be enabled independently, so each layer is provable on its own:

    --stt            whisper transcribes; the transcript is logged
    --tts            piper speaks; with --stt but no --llm the Dot repeats you
    --llm            Ollama answers
    (none of them)   a two-tone chime answers - transport only

Two protocol facts that are not in the ESPHome spec, learned on this bench:

* The Dot wakes in *private listening*, so the device never sends a VAD-end
  sentinel - the backend is the endpointing authority (HA uses Silero, so this
  does too).  Without it the controller waits out its 20 s hard cap.
* The reply is delivered as a URL in TTS_END (data["url"]); the controller
  fetches, decodes, shapes and plays it.  Announcements are a separate,
  out-of-band path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import functools
import http.server
import json
import logging
import math
import os
import re
import socket
import struct
import sys
import threading
import time
import uuid
import wave

import numpy as np

from aioesphomeapi import APIClient
from aioesphomeapi import VoiceAssistantEventType as ET

import calc
import clock
import envelope as env_schema
from ha import (DO_TOOL, HA_TOOLS, LOOKUP_TOOL, HaError, HomeAssistant,
                looks_imperative, wants_ha_tools)
from ma import MaError, MusicAssistant, wants_music_tools
from lanes import BusLane, HttpLane, LaneError, Lanes, NotSyncCapable

LOG = logging.getLogger("bridge")

MIC_RATE = 16000          # the controller streams mono S16_LE at 16 kHz
MIC_WIDTH = 2

# Fed to whisper as `initial_prompt`. It costs nothing and it fixes domain words:
# without it, base.en heard "4 to 1 in petals step up" on this bench; with it,
# "4 to 1 impedance step up". small.en was no better and three times slower.
ASR_PROMPT = (
    "Questions about amateur radio antennas and RF: SWR, impedance, folded dipole, "
    "balun, duplexer, resonance, reflected power, dBm, MHz, coax, feedline, VFO. "
    # the assistant and the cloud agent are named in nearly every hand-off, and
    # "Muse" was decoded as "news" on a live turn
    "Names: Zion, Muse, Clippy, Satellite."
)

# Wording matters more than model size here. Measured on this bench with the
# same 70B and the same two questions:
#   "under 25 words"              -> "8% reflected, half a dB"   (wrong, twice,
#                                    and a different wrong answer each time)
#   "silent maths + be precise"   -> "25% ... 1.25 dB"           (right)
# A hard word limit makes a model skip the arithmetic and emit a prior. Keep the
# sentences short, but do not ration the *working*.
# Naming: Zion is this machine (whisper + Ollama + Piper, all local); Muse is
# whatever answers from outside it. The assistant speaks as Zion when it answers
# from here, and hands work to Muse when it cannot.
SYSTEM_PROMPT = (
    "You are Zion, a voice assistant in an amateur radio workshop. "
    "Answer in at most two spoken sentences. Do arithmetic silently and state the "
    "result with the number. Be precise; do not guess. No lists or markdown. "
    "For the current time or date, call host_clock; never guess one. "
    "If a question needs live data or knowledge you lack, call handoff — do not "
    "reply that you need Muse. handoff needs='data' means the user's own records "
    "(only Muse has them); needs='reasoning' means calculation or world facts. Do not mention the current time unless asked."
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}  {msg}", flush=True)


# ── media serving ────────────────────────────────────────────────────────────

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves reply media, plus the contract this service actually speaks.

    The schema routes exist because a cached copy of the source went stale the
    night the memory fields landed: an agent fetched the old page and got a schema
    the bridge no longer spoke. Whatever is served here comes from the running
    process, so it cannot disagree with the wire.
    """

    server_version = "voice-bridge"

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        LOG.debug("http: " + fmt, *args)

    def _send(self, body: bytes, ctype: str, *, last_modified: float | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # no-store is the point: a cached schema is the failure this route prevents
        self.send_header("Cache-Control", "no-store, max-age=0")
        if last_modified:
            self.send_header("Last-Modified", self.date_time_string(last_modified))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/envelope.py":
            source = env_schema.__file__
            with open(source, "rb") as fh:
                self._send(fh.read(), "text/x-python; charset=utf-8",
                           last_modified=os.path.getmtime(source))
            return
        if path == "/schema.json":
            body = json.dumps(env_schema.schema_summary(), indent=1).encode()
            self._send(body, "application/json; charset=utf-8",
                       last_modified=os.path.getmtime(env_schema.__file__))
            return
        if path == "/healthz":
            self._send(json.dumps({"ok": True, "schema_version": env_schema.VERSION,
                                   "types": [env_schema.HANDOFF, env_schema.ANSWER]}).encode(),
                       "application/json; charset=utf-8")
            return
        return super().do_GET()


def serve_dir(directory: str, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_QuietHandler, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.20", 8123))  # HA box; no packets are sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def write_wav(path: str, audio: np.ndarray, rate: int, channels: int = 1) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    if channels == 2 and pcm.ndim == 1:
        pcm = np.repeat(pcm, 2)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def write_tone(path: str, freq: float = 660.0, seconds: float = 1.0, rate: int = 48000) -> None:
    n = int(rate * seconds)
    fade = int(rate * 0.02)
    t = np.arange(n) / rate
    env = np.minimum(1.0, np.minimum(np.arange(n) / fade, (n - np.arange(n)) / fade))
    wave_f = np.where(t < seconds / 2, freq, freq * 1.5)
    audio = 0.25 * env * np.sin(2 * np.pi * wave_f * t)
    write_wav(path, audio, rate, channels=2)


# ── Silero VAD (the model HA uses, bundled with faster-whisper) ───────────────

class SileroVad:
    WINDOW = 512          # 32 ms at 16 kHz
    CONTEXT = 512         # one extra window, so the model has real context

    def __init__(self, threshold=0.5, neg_threshold=0.35, min_silence_ms=600, min_speech_ms=160):
        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.pending = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(self.CONTEXT, dtype=np.float32)
        self.speaking = False
        self._speech_ms = 0.0
        self._quiet_ms = 0.0
        self.speech_ms = 0.0
        self.max_p = 0.0
        self.frames = 0

    def feed(self, pcm: bytes) -> str | None:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, x])
        edge = None
        step_ms = self.WINDOW / MIC_RATE * 1000.0
        while len(self.pending) >= self.WINDOW:
            win = self.pending[: self.WINDOW]
            self.pending = self.pending[self.WINDOW:]
            p = float(self.model(np.concatenate([self.tail, win]), self.WINDOW, 64)[-1])
            self.tail = win
            self.frames += 1
            self.max_p = max(self.max_p, p)
            edge = self._state(p, step_ms) or edge
        return edge

    def _state(self, p: float, step_ms: float) -> str | None:
        if not self.speaking:
            if p >= self.threshold:
                self._speech_ms += step_ms
                if self._speech_ms >= self.min_speech_ms:
                    self.speaking = True
                    self._quiet_ms = 0.0
                    return "start"
            else:
                self._speech_ms = max(0.0, self._speech_ms - step_ms)
            return None
        self.speech_ms += step_ms
        if p < self.neg_threshold:
            self._quiet_ms += step_ms
            if self._quiet_ms >= self.min_silence_ms:
                # Close the utterance and re-arm; without this reset every
                # following window would report "end" again.
                self.speaking = False
                self._speech_ms = 0.0
                self._quiet_ms = 0.0
                return "end"
        else:
            self._quiet_ms = 0.0
        return None


# ── the local brain ──────────────────────────────────────────────────────────

# Offered to the local lane when `--handoff` is on. The model calls this when a
# turn needs live facts or knowledge it does not have; the restated question is
# what goes over the bus (it improves the answer and limits what leaves).
#: The host owns the clock, so the model asks for it instead of guessing. The tool is
#: only offered when the turn is temporal; a schema on every turn costs ~35 prompt
#: tokens (~1 s here) for a fact most questions do not need.
CLOCK_TOOL = [{
    "type": "function",
    "function": {
        "name": "host_clock",
        "description": "Current local time and/or date from the host. Use this for "
                       "any question about now, today, or a date you need to compute with.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["time", "date", "both"]},
            },
            "required": ["kind"],
        },
    },
}]

# Deliberately terse. Every word here is paid on EVERY turn that offers the tool
# (~40 tokens/s of prompt evaluation on this box), and the model only needs enough
# to choose: the meanings live in the system prompt, which is cached.
HANDOFF_TOOL = [{
    "type": "function",
    "function": {
        "name": "handoff",
        "description": "Ask Muse for the user's own records; ask the cloud model "
                       "for calculations or knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "needs": {"type": "string", "enum": ["data", "reasoning"]},
            },
            "required": ["question", "needs"],
        },
    },
}]

# A local answer that says one of these is not an answer. Measured on this bench,
# qwen2.5:3b produced "I don't have access to real-time information" for "what time
# is it?" and "I don't have real-time weather forecasting capabilities" for the
# weather — both are hand-off triggers under `--handoff auto`.
HEDGE_MARKERS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "not sure of", "i don't have", "i do not have", "don't have access",
    "no access to", "don't have real-time", "do not have real-time",
    "i cannot", "i can't", "i'm unable", "i am unable", "unable to",
    "not able to", "without more context", "need more context",
    "not aware of", "cannot provide", "can't provide", "cannot verify",
    "check a reliable", "consult a", "as an ai",
    # meta-hedges: the model narrating a hand-off instead of calling the tool.
    # Measured: given the tool, qwen2.5:3b still answered "I need Muse for this."
    # to "what time is sunset?" with tools=0. The sentence is a deferral; honour it.
    "need muse", "needs muse", "ask muse", "get muse", "requires muse",
    "let muse", "muse would", "hand this to muse", "hand it to muse",
    "i need help", "needs help from",
    # deferrals to data the model cannot see. Measured: with a terser tool schema
    # the model stopped calling handoff and said "I need your own records to answer
    # that" — a dead end unless it is treated as the deferral it is.
    "need your", "needs your", "your own records", "don't have your",
    "do not have your", "can't see your", "cannot see your", "your records",
    "your notes", "your list", "your calendar", "your account",
)

# Spoken phrases that mean "don't try, hand it over" without asking the model
EXPLICIT_HANDOFF = (
    "ask muse", "hand it to muse", "hand off", "hand it off",
    "look it up", "check with muse", "let muse handle",
)

# Muse polls her inbox every 10-15 s and answers in her own time; measured
# expectation is 30-60 s end to end. The ack has to cover that wait, not promise
# instant service.
ACK_TEMPLATE = "I've asked {name}. It may take about a minute."

#: utterances that only make sense with the previous turn in view. Referential
#: words only — an earlier version also treated any short question as a follow-up,
#: which put history in front of "what is a balun?" for no benefit.
FOLLOWUP_PREFIXES = ("and ", "what about", "how about", "the other", "same ",
                     "do that", "do it", "that one", "one more")
FOLLOWUP_MARKERS = (" it", " its", " that", " those", " them", " they",
                    " the other", " instead", " again", " both", " either")

#: verification prompt. The model must have room to work: a word cap made it skip
#: the arithmetic entirely (it answered "11.1%" for SWR 3:1, which is the answer for
#: 2:1), and a truncated budget cut the final line off. Two lines of working plus an
#: extractable ANSWER line is the version that measured right.
VERIFY_SYSTEM = (
    "You are a radio engineering expert. Keep any working to at most two short lines, "
    "then end with one single line exactly of the form:\n"
    "ANSWER: <one short spoken sentence containing every number asked for>"
)

def _numbers(text: str) -> set[str]:
    """Numeric tokens, normalised enough to compare two answers' claims."""
    return {m.group(0).rstrip(".") for m in re.finditer(r"\d+(?:\.\d+)?", text or "")}


def _numeric(text: str) -> list[float]:
    out = []
    for token in _numbers(text):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _covered_by(answer: str, tool_numbers: set[str]) -> bool:
    """True when every number in the answer came from a tool (rounding allowed).

    A model rounds "81.446" to "81.4" and reformats "22:57" from "22:57"; neither is
    the model inventing a number. 84 days from a date the tool supplied is.
    """
    tools = _numeric(" ".join(tool_numbers))
    if not tools:
        return False
    for value in _numeric(answer):
        if not any(abs(value - t) <= max(0.5, abs(t) * 0.01) for t in tools):
            return False
    return True


#: A model that writes tool syntax as prose instead of calling the tool. Measured:
#: "What day of the week is it?" produced the literal text "handoff needs='data'",
#: which was then spoken to the user.
TOOL_SYNTAX = re.compile(
    r"^\s*(?:handoff|host_clock|music_play|music_transport|music_volume|"
    r"ha_lookup|ha_do)\b|needs\s*=\s*['\"]|\bneeds\s*=\s*(?:data|reasoning)\b",
    re.IGNORECASE)

#: sentence boundary for streaming TTS: terminal punctuation followed by space or end
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

#: questions whose numbers come from the injected clock, not from the model
CLOCK_QUESTION = re.compile(
    r"^(what|which)\s+(time|date|day)\b|what day is it|today'?s date|"
    r"what is the date", re.IGNORECASE)

#: A number the model produced is worth a second opinion; a number it was told is
#: not. Spelled-out numbers count: the exact failure this exists for was "eight
#: percent reflected, about half a dB loss", which contains no digits at all.
HAS_NUMBER = re.compile(
    r"\d|\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|half|"
    r"quarter|percent)\b", re.IGNORECASE)

#: questions where the LOCAL context is the authority, not the wider world. A cold
#: cloud re-answer has no idea that "Zion" is this machine — it answered "Zion
#: National Park is 229 square miles" and that replaced a correct local answer.
SELF_REFERENTIAL = re.compile(
    r"\b(you|your|yours|yourself|zion|muse|workshop|satellite|this machine|"
    r"this box|my station|my setup|here)\b", re.IGNORECASE)

#: questions that ask for computation, whatever the model calls them. A 3B asked
#: "what ERP do I get from 100 W into 6 dBd?" labelled it needs="data" and sent it to
#: the agent lane (35 s) when the cloud lane answers it in ~1 s.
COMPUTE_UNITS = re.compile(
    r"\b(db|dbm|dbd|dbi|swr|erp|eirp|watt|watts|impedance|wavelength|"
    r"percent|percentage|gain|loss|length|distance|frequency|"
    r"hz|khz|mhz|ghz|feet|foot|mile|miles|metre|metres|meter|meters|"
    r"kilometre|kilometres|kilometer|kilometers)\b", re.IGNORECASE)
COMPUTE_VERBS = re.compile(
    r"\b(how many|how much|convert|calculate|compute|how long|how far)\b",
    re.IGNORECASE)

#: an answer repeating the injected clock is using context the cloud cannot see
CLOCK_IN_ANSWER = re.compile(r"\b\d{1,2}:\d{2}\s*(am|pm)\b|\b(mon|tues|wednes|thurs|fri|satur|sun)day\b",
                             re.IGNORECASE)


class Brain:
    """whisper -> ollama -> piper, each stage optional."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.whisper = None
        self.piper = None
        self.piper_rate = 22050
        self._tts_lock = threading.Lock()      # PiperVoice is not documented thread-safe
        self._fallback = None                 # second whisper model, loaded on demand
        self._fallback_tried = False
        self._fallback_lock = threading.Lock()
        self.lanes = self._build_lanes()
        self.local = self.lanes["local"]

    def _build_lanes(self) -> Lanes:
        """Assemble the lanes. `sync_capable` lives on each lane, not here.

        A lane is anything that can answer a turn. Only the flag decides whether
        it can answer *inside* one: HttpLane (Ollama, or a cloud gateway) can,
        BusLane (Muse over the agent bus) cannot, because its reply takes
        minutes. Wiring a lane into a turn is therefore a config change.
        """
        lanes = [
            HttpLane(
                "local",
                spoken_name=self.args.local_name,
                base_url=self.args.ollama.rstrip("/") + "/v1",
                model=self.args.model,
                system_prompt=self.args.system_prompt,
                max_tokens=self.args.max_tokens,
                timeout=self.args.lane_timeout,
            )
        ]
        if self.args.cloud_model:
            lanes.append(
                HttpLane(
                    "cloud",
                    spoken_name=self.args.cloud_name,
                    base_url=self.args.cloud_url,
                    model=self.args.cloud_model,
                    api_key=self.args.cloud_key,
                    system_prompt=self.args.cloud_system_prompt or self.args.system_prompt,
                    max_tokens=self.args.cloud_max_tokens,
                    timeout=self.args.lane_timeout,
                )
            )
        if self.args.muse:
            lanes.append(
                BusLane(
                    "muse",
                    bus_url=self.args.bus_url,
                    token_path=self.args.bus_token,
                    to=self.args.muse_agent,
                    spoken_name=self.args.muse_name,
                    msg_type=self.args.bus_type,
                )
            )
        return Lanes(lanes)

    def load(self) -> None:
        # Fail at startup, not on the first turn: an async lane can never be the
        # answer lane, and finding that out mid-conversation is worse.
        self.lanes.answer_lane(self.args.answer_lane)
        if self.args.stt or self.args.llm:
            from faster_whisper import WhisperModel

            t = time.monotonic()
            self.whisper = WhisperModel(
                self.args.whisper_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.args.threads,
                # Measured: with the model already in the HuggingFace cache, this
                # constructor still opens an HTTPS session to the Hub on every
                # start (108.138.246.x:443, ~2 kB up / 7.7 kB down). That single
                # call made "nothing leaves the LAN" false. HF_HUB_OFFLINE=1 in
                # the unit covers the library's other paths; this covers the one
                # call the bridge actually makes, so a shell without the env var
                # cannot reintroduce it.
                local_files_only=True,
            )
            log(f"whisper {self.args.whisper_model} loaded in {time.monotonic() - t:.1f} s")
        if self.args.tts:
            from piper import PiperVoice

            t = time.monotonic()
            self.piper = PiperVoice.load(self.args.piper_voice)
            self.piper_rate = self.piper.config.sample_rate
            log(f"piper {os.path.basename(self.args.piper_voice)} loaded in {time.monotonic() - t:.1f} s ({self.piper_rate} Hz)")

    def _transcribe_with(self, model, audio) -> tuple[str, float]:
        """Transcribe with one model; return the text and its worst logprob."""
        segments, _info = model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.args.initial_prompt or None,
        )
        seen: list[str] = []
        worst = 0.0
        for segment in segments:          # the work happens while iterating
            seen.append(segment.text.strip())
            worst = min(worst, segment.avg_logprob)
        return " ".join(part for part in seen if part).strip(), worst

    def _stt_unsure(self, text: str, logprob: float) -> bool:
        """Should a bigger model re-hear this?

        base.en's worst-segment logprob on this bench's *good* transcripts sat at
        -0.40 .. -0.63. The transcript that reached the Dot as "ASM Muse was not
        much schedule" measured -0.885, so the threshold is -0.75. That is two
        turns of evidence, not a calibration - hence --stt-confidence, so it can
        be moved without editing code.
        """
        model = getattr(self.args, "whisper_fallback", None)
        if not model or model == "off":
            return False
        if model == getattr(self.args, "whisper_model", None):
            return False          # re-running the same model proves nothing
        if not (text or "").strip():
            return True                   # nothing recognised: give the bigger one a turn
        return logprob < getattr(self.args, "stt_confidence", -0.75)

    def _fallback_model(self):
        """Load the second model once, on first need. Never mid-turn on a lock."""
        with self._fallback_lock:
            if not self._fallback_tried:
                self._fallback_tried = True
                from faster_whisper import WhisperModel   # imported here by load()
                try:
                    t = time.monotonic()
                    self._fallback = WhisperModel(
                        self.args.whisper_fallback, device="cpu", compute_type="int8",
                        cpu_threads=self.args.threads, local_files_only=True,
                    )
                    log(f"whisper fallback {self.args.whisper_fallback} loaded in "
                        f"{time.monotonic() - t:.1f} s")
                except Exception as exc:  # noqa: BLE001
                    log(f"whisper fallback {self.args.whisper_fallback} unavailable "
                        f"({exc!r}); pre-fetch it or pass --whisper-fallback off")
                    self._fallback = None
        return self._fallback

    def transcribe(self, pcm: bytes) -> tuple[str, float]:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        t = time.monotonic()
        text, logprob = self._transcribe_with(self.whisper, audio)
        if self._stt_unsure(text, logprob):
            alt = getattr(self.args, "whisper_fallback", "small.en")
            log(f"stt: {self.args.whisper_model} unsure (logprob {logprob:.3f} < "
                f"{getattr(self.args, 'stt_confidence', -0.75):.2f}) on {text[:50]!r}; "
                f"re-hearing with {alt}")
            model = self._fallback_model()
            if model is not None:
                t2 = time.monotonic()
                retry, retry_logprob = self._transcribe_with(model, audio)
                log(f"stt: {alt} {time.monotonic() - t2:.2f} s "
                    f"logprob {retry_logprob:.3f}: {retry[:60]!r}")
                if retry:
                    text = retry
        return text, time.monotonic() - t

    async def ask(self, text: str, tools: list[dict] | None = None):
        """Answer from the chosen sync lane. Refuses loudly if it is not one.

        The current time rides along with the question. Both lanes have no clock,
        and without it a cloud model invents one: measured on this bench, asked
        "what time is it" at 18:48 it answered "It is 3:47 PM, the workshop clock
        reads this time because its mechanical hands rotate...". Attached to the
        user turn rather than the system prompt so the cached system prefix on
        the local lane still matches, but the clock is always current.
        """
        context = time.strftime("%A %d %B %Y, %I:%M %p")
        return await self.lanes.answer_lane(self.args.answer_lane).ask(
            f"{text}\n\n[current local time: {context}]", tools=tools
        )

    def speak(self, text: str, path: str) -> tuple[float, float]:
        """Write a WAV; returns (seconds of audio, synthesis seconds)."""
        t = time.monotonic()
        with self._tts_lock:
            chunks = list(self.piper.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(0, np.float32)
        write_wav(path, audio, self.piper_rate)
        return len(audio) / self.piper_rate, time.monotonic() - t


# ── the bridge ───────────────────────────────────────────────────────────────

class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.brain = Brain(args)
        self.client: APIClient | None = None
        self.done = asyncio.Event()
        self.turn = 0
        self.task: asyncio.Task | None = None
        self.playback_done = asyncio.Event()
        self.audio_s = 0.0
        self.in_turn = False
        self.announce_lock = asyncio.Lock()
        self._bg_tasks: set[asyncio.Task] = set()
        self.poller: asyncio.Task | None = None
        self.pending: dict[str, dict] = {}
        self.pending_path = os.path.join(args.outdir, "pending-handoffs.json")
        self.ha = (HomeAssistant(url=args.ha_url, token_file=args.ha_token)
                   if args.ha else None)
        self.ma = (MusicAssistant(url=args.ma_url, token_file=args.ma_token,
                                  default_target=args.music_target)
                   if args.music else None)
        self.history: dict[str, list[dict]] = {}
        self.conversation = ""
        self._reset()

    def _reset(self) -> None:
        self.mic = bytearray()
        self.vad = SileroVad(
            min_silence_ms=self.args.silence_ms,
            min_speech_ms=self.args.min_speech_ms,
        )
        self.t_start = 0.0
        self.t_speech_end = 0.0
        self.answered = False

    # ── inbound ──────────────────────────────────────────────────────────────

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        self.turn += 1
        self._cancel_stream_tasks()
        self._reset()
        self.conversation = conversation_id or ""
        self.t_start = time.monotonic()
        log(f"[turn {self.turn}] wake: conversation={conversation_id[:8]} phrase={wake_word_phrase!r} flags={flags}")
        self.send_event(ET.VOICE_ASSISTANT_RUN_START)
        return 0

    async def handle_audio(self, data: bytes, data2: bytes | None) -> None:
        if not data:
            return
        self.mic += data
        edge = self.vad.feed(data)
        if edge == "start":
            log(f"[turn {self.turn}] speech start +{(time.monotonic() - self.t_start) * 1000:.0f} ms (p={self.vad.max_p:.2f})")
            self.send_event(ET.VOICE_ASSISTANT_STT_VAD_START)
        elif edge == "end":
            self.t_speech_end = time.monotonic()
            log(
                f"[turn {self.turn}] speech end +{(self.t_speech_end - self.t_start) * 1000:.0f} ms "
                f"({self.vad.speech_ms:.0f} ms speech, {len(self.mic) / 32000:.2f} s captured)"
            )
            self.send_event(ET.VOICE_ASSISTANT_STT_VAD_END)
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self.respond())

    async def handle_stop(self, aborted: bool) -> None:
        if aborted:
            log(f"[turn {self.turn}] aborted by controller")
            self._cancel_stream_tasks()
            if self.task and not self.task.done():
                self.task.cancel()
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
            self.finish("aborted")
            return
        if not self.answered and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self.respond())

    async def handle_announcement_finished(self, finished) -> None:
        """The controller's "the satellite is back at idle" signal.

        It arrives after the reply has actually been fetched and played, which
        makes it the honest end-of-turn marker - RUN_END only says the pipeline
        finished, while the controller may still be fetching our URL.  Exiting
        on RUN_END kills the media server mid-fetch.
        """
        log(f"[turn {self.turn}] playback finished (success={getattr(finished, 'success', None)})")
        self.playback_done.set()
        if self.args.once:
            self.done.set()

    # ── the turn ─────────────────────────────────────────────────────────────

    async def respond(self) -> None:
        if self.answered:
            return
        self.answered = True
        self.in_turn = True
        t0 = time.monotonic()
        turn = self.turn
        try:
            # Measured on 2026-11-22: the wake word fired, the person said nothing,
            # and whisper transcribed the silence as "Good time." - which the model
            # then answered with "The good time is 14:23 UTC." If the endpointing
            # never saw speech, there is nothing to transcribe.
            if self.vad.speech_ms < self.args.min_speech_ms:
                log(f"[turn {turn}] no speech detected "
                    f"({self.vad.speech_ms:.0f} ms < {self.args.min_speech_ms} ms); "
                    f"not asking the models")
                self.send_event(ET.VOICE_ASSISTANT_ERROR, {"code": "stt-no-speech"})
                self.send_event(ET.VOICE_ASSISTANT_RUN_END)
                self.finish("no speech")
                return
            pcm = bytes(self.mic)
            await self._dump(pcm, turn)

            # 1. speech to text
            if self.brain.whisper:
                text, stt_s = await asyncio.to_thread(self.brain.transcribe, pcm)
                log(f"[turn {turn}] STT {stt_s * 1000:.0f} ms: {text!r}")
            else:
                text, stt_s = "(no stt)", 0.0
            if text:
                self.remember(self.args.device_name, "user", text, self.conversation)
            if not text:
                self.send_event(ET.VOICE_ASSISTANT_ERROR, {"code": "stt-no-text"})
                self.send_event(ET.VOICE_ASSISTANT_RUN_END)
                self.finish("no speech recognised")
                return
            self.send_event(ET.VOICE_ASSISTANT_STT_END, {"text": text})

            # 2. the reply — local model, unless the turn is handed off
            handoff_note = None
            if self.args.handoff == "off" and self._explicit_handoff(text):
                log(f"[turn {turn}] explicit handoff phrase, but --handoff off")

            reply, handoff_args, note = await self._answer(text, turn)
            if handoff_args is not None:
                reply, handoff_note = await self._handoff(
                    [{"function": {"name": "handoff",
                                   "arguments": json.dumps(handoff_args)}}],
                    text, turn,
                )
            elif (self.args.handoff == "auto" and self.brain.lanes.async_
                  and self._looks_like_hedge(reply)):
                hedge = self._hedge_phrase(reply)
                kind = self._hedge_kind(reply)
                log(f"[turn {turn}] local answer hedged ({hedge!r}) -> "
                    f"escalating as {kind}")
                answered = None
                if kind == "knowledge" and "cloud" in self.brain.lanes:
                    answered = await self._cloud_answer(text)
                if answered:
                    reply, _stats = answered
                    handoff_note = "escalated (hedge→reasoning)"
                else:
                    reply, handoff_note = await self._handoff(
                        [], text, turn,
                        why=f"local answer hedged ({hedge}): {reply[:120]}",
                    )
            if note:
                log(f"[turn {turn}] answer note: {note}")
            self.send_event(ET.VOICE_ASSISTANT_INTENT_START)
            self.send_event(ET.VOICE_ASSISTANT_INTENT_END, {"continue_conversation": "0"})

            # 3. speak it, and hand the controller a URL to fetch
            name = f"reply-{turn:03d}-{uuid.uuid4().hex[:6]}.wav"
            path = os.path.join(self.media_dir, name)
            if self.brain.piper:
                audio_s, tts_s = await asyncio.to_thread(self.brain.speak, reply, path)
                self.audio_s = audio_s
                log(f"[turn {turn}] TTS {tts_s * 1000:.0f} ms for {audio_s:.2f} s of speech")
            else:
                write_tone(path)
                log(f"[turn {turn}] TTS: chime (no --tts)")
            if self.args.attribution == "always" and not handoff_note:
                reply = f"{self.brain.lanes.answer_lane(self.args.answer_lane).spoken_name} says: {reply}"
                await asyncio.to_thread(self.brain.speak, reply, path)
            url = f"http://{self.args.media_host}:{self.args.media_port}/{name}"
            self.send_event(ET.VOICE_ASSISTANT_TTS_START)
            self.send_event(ET.VOICE_ASSISTANT_TTS_END, {"url": url})
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
            if not handoff_note:
                self.remember(self.args.device_name, "assistant", reply, self.conversation)
            log(f"[turn {turn}] reply handed over (+{(time.monotonic() - t0) * 1000:.0f} ms): {url}"
                + (f"  [{handoff_note}]" if handoff_note else ""))
            self.finish("ok")
        except asyncio.CancelledError:
            log(f"[turn {turn}] cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"[turn {turn}] FAILED: {exc!r}")
            self.send_event(ET.VOICE_ASSISTANT_ERROR, {"code": "bridge-error"})
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
            self.finish("error")
        finally:
            self.in_turn = False

    # ── handoff ──────────────────────────────────────────────────────────────

    #: deferrals that mean "only Muse has this" rather than "I cannot work it out"
    AGENT_HEDGE = ("need muse", "needs muse", "ask muse", "requires muse",
                   "muse would", "hand this to muse", "hand it to muse",
                   # deferrals to records only the agent has — no point asking a
                   # cloud model first, it cannot see them either
                   "your own records", "your records", "your notes", "your list",
                   "your calendar", "your account", "need your", "needs your",
                   "don't have your", "do not have your", "can't see your",
                   "cannot see your")

    @staticmethod
    def _effective_needs(model_needs: str, question: str) -> str:
        """The model's label, overridden when the question is plainly a calculation.

        The 3B mislabels arithmetic as "data" often enough to matter; a numeric
        question sent to the agent lane costs 30-60 s for an answer the cloud lane
        gives in about one. A mislabel in the other direction is caught by the
        refusal cascade: if the cloud lane says it has no access, the turn falls
        back to the agent.
        """
        needs = (model_needs or "data").strip().lower()
        if needs not in ("data", "reasoning"):
            needs = "data"
        text = question or ""
        # a quantity unit in the question is the strongest signal: "what ERP from
        # 100 W into 6 dBd" is arithmetic no matter what the model called it. A
        # how-much/how-many with a number is the weaker one ("how long is a
        # half-wave for 14.2 MHz" vs "how much did I spend on the dipole").
        if COMPUTE_UNITS.search(text):
            return "reasoning"
        if COMPUTE_VERBS.search(text) and re.search(r"\d", text):
            return "reasoning"
        return needs

    @staticmethod
    def _hedge_kind(text: str) -> str | None:
        """'agent' for a data deferral, 'knowledge' for a reasoning gap, else None."""
        if not Bridge._looks_like_hedge(text):
            return None
        low = (text or "").lower()
        return "agent" if any(m in low for m in Bridge.AGENT_HEDGE) else "knowledge"

    @staticmethod
    def _looks_like_hedge(text: str) -> bool:
        return Bridge._hedge_phrase(text) is not None

    @staticmethod
    def _hedge_phrase(text: str) -> str | None:
        low = (text or "").lower()
        for marker in HEDGE_MARKERS:
            if marker in low:
                return marker
        return None

    def _explicit_handoff(self, text: str) -> bool:
        low = text.lower()
        return any(phrase in low for phrase in EXPLICIT_HANDOFF)

    def _should_offer_handoff(self, text: str) -> bool:
        """Whether this turn can be handed off at all."""
        if self.args.handoff == "off":
            return False
        if not self.brain.lanes.async_:
            return False
        # 'ask' = the model decides (tool) or the speaker asks for it;
        # 'auto' = also when the local answer looks like a hedge.
        return True

    def _handoff_context(self, question: str) -> list[dict]:
        """Recent turns in envelope shape: chat history is {"role","content"},
        the envelope wants {"role","text"}. Getting this wrong raised
        KeyError('text') on the first follow-up that also needed a hand-off."""
        if self.args.memory == "off":
            return []
        return [{"role": t["role"], "text": t.get("content") or t.get("text") or ""}
                for t in self.history_for(self.args.device_name, question)]

    def _handoff_lane(self):
        lanes = self.brain.lanes.async_
        return lanes[0] if lanes else None

    async def _handoff(self, calls: list[dict], original: str, turn: int,
                       why: str = "") -> tuple[str, str]:
        """Submit the model's restated question, and say so out loud."""
        lane = self.brain.lanes.async_[0]
        args = {}
        for call in calls:
            fn = call.get("function") or {}
            if fn.get("name") != "handoff":
                continue
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"question": original, "why": "unparsed tool arguments"}
            break
        question = (args.get("question") or original).strip()
        why = (args.get("why") or why or "").strip()
        env = env_schema.make_handoff(
            question,
            device=self.args.device_name,
            from_agent="bridge",
            to=lane.name,
            ttl_s=self.args.handoff_ttl_s,
            max_words=30,
            hint="answer in one or two spoken sentences",
            conversation=self.conversation,
            context=self._handoff_context(question),
            needs=str(args.get("needs") or "data"),
        )
        try:
            corr = await lane.submit(question, device=self.args.device_name, envelope=env)
        except LaneError as exc:
            log(f"[turn {turn}] handoff failed: {exc}")
            return "I couldn't reach Muse just now.", f"handoff error: {exc}"
        self.pending[corr] = {
            "device": self.args.device_name,
            "subject": f"voice: {self.args.device_name}",
            "envelope_id": env["id"],
            "bus_id": getattr(lane, "last_message_id", None),
            "question": question,
            "why": why,
            "submitted": time.time(),
            "attempts": 0,
        }
        self._save_pending()
        log(f"[turn {turn}] handed off to {lane.name} (corr={corr}): {question!r} why={why!r}")
        ack = (self.args.ack_text or ACK_TEMPLATE.format(name=lane.spoken_name))
        return ack, f"handoff {corr} via {lane.spoken_name}"

    #: what bus-dispatch puts in a reply when the agent produced nothing
    EMPTY_REPLIES = {"", "none", "null", "error", "(empty reply)", "(no reply)",
                     "no reply", "(error)", "empty reply"}

    @classmethod
    def _is_real_answer(cls, text: str) -> bool:
        cleaned = (text or "").strip().strip("()").strip().lower()
        return len(cleaned) >= 2 and cleaned not in {
            t.strip("()").strip().lower() for t in cls.EMPTY_REPLIES
        }

    @staticmethod
    def _voice_trim(text: str, max_words: int = 30) -> str:
        """Keep an announcement speakable; agents do not always obey brevity."""
        text = re.sub(r"[*_`#>]", "", text)                  # emphasis / code fences
        text = re.sub(r"(?m)^\s*[-•]\s*", "", text)          # bullet markers
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if len(words) <= max_words:
            return text
        out: list[str] = []
        used = 0
        for sentence in text.split(". "):
            n = len(sentence.split())
            if used + n > max_words:
                if not out:
                    # a single oversized sentence: hard-cap it, or the cap means
                    # nothing (a price list arrives as one "sentence")
                    out.append(" ".join(sentence.split()[:max_words]))
                    used = max_words
                break
            out.append(sentence)
            used += n
        trimmed = ". ".join(out).rstrip(".")
        return f"{trimmed}. (Full answer in the log.)"

    async def _announce(self, text: str) -> bool:
        """Speak out of band, outside any turn. Returns True if it played.

        Serialised: two overlapping announcements stack the controller's duck
        layers and (measured) end with the device dropping its data plane.
        """
        if self.client is None:
            log("announce skipped: no controller connection")
            return False
        async with self.announce_lock:
            return await self._announce_once(text)

    async def _announce_once(self, text: str) -> bool:
        text = self._voice_trim(text)
        lane = self._handoff_lane()
        if self.args.attribution != "off" and lane is not None:
            text = f"{lane.spoken_name} says: {text}"
        name = f"announce-{uuid.uuid4().hex[:8]}.wav"
        path = os.path.join(self.media_dir, name)
        audio_s = 0.0
        if self.brain.piper:
            audio_s, _tts_s = await asyncio.to_thread(self.brain.speak, text, path)
        else:
            write_tone(path)
        url = f"http://{self.args.media_host}:{self.args.media_port}/{name}"
        # The await returns only when playback ENDS, so a flat timeout must be
        # longer than the audio or a good announcement reads as a failure and is
        # spoken twice (30 s timeout vs 35 s of speech did exactly that).
        timeout = max(self.args.announce_timeout, audio_s + 10.0)
        log(f"announcing {audio_s:.1f} s of speech (timeout {timeout:.0f} s)")
        started = time.monotonic()
        try:
            finished = await self.client.send_voice_assistant_announcement_await_response(
                media_id=url, text=text, timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001 - busy device, timeout, ...
            elapsed = time.monotonic() - started
            if elapsed >= audio_s - 2.0:
                # The await gives up when AnnounceFinished does not arrive, but
                # by then most of the audio has played on a healthy device. Do
                # not say it twice because of a reporting gap.
                log(f"announce finished without confirmation after {elapsed:.0f} s "
                    f"of {audio_s:.0f} s audio - treating as played ({exc!r})")
                return True
            log(f"announce failed after {elapsed:.0f} s ({exc!r}); will retry")
            return False
        log(f"announced: {text!r} (success={getattr(finished, 'success', None)})")
        self.remember(self.args.device_name, "assistant", text, self.conversation)
        return True

    async def _poll_handoffs(self) -> None:
        """Turn a late bus reply into a spoken announcement."""
        lane = self.brain.lanes.async_[0]
        interval = self.args.handoff_poll_s
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.pending or self.in_turn:
                    continue
                messages = await lane.poll()
                for msg in messages:
                    corr = self._match_handoff(msg)
                    entry = self.pending.get(corr) if corr else None
                    if not entry:
                        continue
                    raw = (msg.get("body") or "").strip()
                    env = env_schema.parse(raw)
                    if env is not None:
                        problems = env_schema.validate(env)
                        if problems:
                            log(f"handoff {corr}: malformed envelope, ignored: {problems}")
                            continue
                        if env.get("type") != env_schema.ANSWER:
                            continue
                        # TTL belongs to the REQUEST. An answer carries no ttl_s,
                        # and asking it whether it has expired always says yes
                        # (fail-safe), which silently dropped every reply until the
                        # live envelope test caught it. Age the pending entry instead.
                        age = time.time() - float(entry.get("submitted", time.time()))
                        if age > self.args.handoff_ttl_s:
                            log(f"handoff {corr}: answer arrived after {age:.0f}s, "
                                f"past the {self.args.handoff_ttl_s}s TTL, dropped")
                            self.pending.pop(corr, None)
                            self._save_pending()
                            continue
                        typed_error = env_schema.answer_error(env)
                        if typed_error:
                            code, message = typed_error
                            log(f"handoff {corr}: agent reported {code}: {message}")
                            self.pending.pop(corr, None)
                            self._save_pending()
                            await self._announce(
                                f"{lane.spoken_name} couldn't answer that one.")
                            continue
                        text = env_schema.answer_text(env) or ""
                    else:
                        text = raw
                    if not self._is_real_answer(text):
                        entry["attempts"] += 1
                        log(f"handoff {corr}: empty/placeholder reply {text!r} "
                            f"(attempt {entry['attempts']}) — not speaking it")
                        if entry["attempts"] >= self.args.handoff_max_attempts:
                            self.pending.pop(corr, None)
                            self._save_pending()
                            await self._announce(
                                f"{lane.spoken_name} didn't come back with an answer on that one."
                            )
                        continue
                    age = time.time() - entry["submitted"]
                    log(f"handoff reply {corr} after {age:.0f} s: {text[:120]!r}")
                    if await self._announce(text):
                        self.pending.pop(corr, None)
                        self._save_pending()
                    else:
                        entry["attempts"] += 1
                        backoff = min(120.0, 15.0 * entry["attempts"])
                        log(f"handoff {corr}: retry {entry['attempts']} in {backoff:.0f} s")
                        await asyncio.sleep(backoff)
                        if entry["attempts"] >= self.args.handoff_max_attempts:
                            log(f"handoff {corr}: giving up after {entry['attempts']} attempts")
                            self.pending.pop(corr, None)
                            self._save_pending()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the poller must not die
                log(f"handoff poll error: {exc!r}")

    @staticmethod
    def _msg_epoch(msg: dict) -> float | None:
        """Bus timestamps are UTC 'YYYY-MM-DD HH:MM:SS'."""
        stamp = msg.get("timestamp")
        if not stamp:
            return None
        try:
            return datetime.datetime.strptime(str(stamp), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None

    def _match_handoff(self, msg: dict) -> str | None:
        """Find which pending hand-off a reply answers.

        Muse's hook replies with a *new* correlation_id and no in_reply_to, so the
        ids cannot be relied on alone. Fall back to the subject plus time: a reply
        can only answer a question submitted before it arrived, and among those it
        answers the most recent. Exact ids always win when a responder does echo
        them (opencode does).
        """
        env = env_schema.parse(msg.get("body") or "")
        if env:
            correlation = str(env.get("correlation") or "")
            if correlation in self.pending:
                return correlation              # the envelope id: primary path
            for key, entry in self.pending.items():
                # tolerant: a responder that put the bus id in `correlation`
                if entry.get("bus_id") is not None and str(entry["bus_id"]) == correlation:
                    return key
        corr = str(msg.get("correlation_id") or "")
        if corr in self.pending:
            return corr
        ireply = str(msg.get("in_reply_to"))
        for key, entry in self.pending.items():
            if entry.get("bus_id") is not None and str(entry["bus_id"]) == ireply:
                return key
        # subject + time fallback
        ts = self._msg_epoch(msg)
        if ts is None or not str(msg.get("subject", "")).startswith("Re:"):
            return None
        eligible = [(float(e.get("submitted", 0)), k) for k, e in self.pending.items()
                    if float(e.get("submitted", 0)) <= ts + 5]
        if not eligible:
            return None
        return max(eligible)[1]

    def _save_pending(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.pending_path) or ".", exist_ok=True)
            with open(self.pending_path, "w") as fh:
                json.dump(self.pending, fh, indent=1)
        except OSError as exc:
            log(f"could not persist pending handoffs: {exc!r}")

    def _load_pending(self) -> None:
        try:
            with open(self.pending_path) as fh:
                self.pending = json.load(fh) or {}
            log(f"resumed {len(self.pending)} pending handoff(s)")
        except (OSError, json.JSONDecodeError):
            self.pending = {}

    # ── conversation memory ─────────────────────────────────────────────────
    #
    # Ollama is stateless and the bus has no session concept, so the bridge owns
    # the recent turns for a device and hands them out: to the local lane as
    # chat history, and to a person's hand-off inside the request envelope.
    # A responder may cache deeper state, but correctness never depends on it.

    def _prune_history(self) -> None:
        now = time.time()
        ttl = self.args.memory_ttl_s
        for key in list(self.history):
            self.history[key] = [t for t in self.history[key] if now - t["ts"] <= ttl]
            if not self.history[key]:
                del self.history[key]

    def remember(self, key: str, role: str, text: str, conversation: str = "") -> None:
        if self.args.memory == "off" or not text.strip():
            return
        turns = self.history.setdefault(key, [])
        turns.append({"role": role, "text": text.strip()[:400], "ts": time.time(),
                      "conversation": conversation})
        limit = self.args.memory_turns * 2
        self.history[key] = turns[-limit:]

    def history_for(self, key: str, utterance: str) -> list[dict]:
        """Chat history worth paying prompt tokens for, or []."""
        if self.args.memory == "off":
            return []
        self._prune_history()
        turns = self.history.get(key) or []
        if not turns:
            return []
        if self.args.memory == "auto" and not self._looks_like_followup(utterance):
            return []
        return [{"role": t["role"], "content": t["text"]} for t in turns]

    @staticmethod
    def _looks_like_followup(utterance: str) -> bool:
        low = " ".join((utterance or "").lower().split())
        if not low:
            return False
        padded = f" {low} "
        if any(low.startswith(prefix) for prefix in FOLLOWUP_PREFIXES):
            return True
        return any(marker in padded for marker in FOLLOWUP_MARKERS)

    @staticmethod
    def should_verify(question: str, answer: str, used_tools: bool,
                      mode: str = "numbers", cloud: bool = True,
                      tool_numbers: set[str] | None = None) -> tuple[bool, str]:
        """Whether to spend a cloud call checking this answer, and why not.

        Pure, so the policy is testable without a bridge, a model or a network.
        Measured failure it exists for: the local model answered a 3:1 SWR question
        with "eight percent reflected, half a dB" — confidently wrong, never a
        hedge, so no other trigger catches it.
        """
        if mode == "off":
            return False, "verification is off"
        if not cloud:
            return False, "no cloud lane"
        # A tool use is not a licence for arithmetic. "How many days until Christmas?"
        # called the clock and then computed 84 days (correct: 34) — the date came from
        # the tool, the number did not. So skip only when the answer's numbers are all
        # numbers a tool returned, rounding included.
        if used_tools:
            if not _numbers(answer):
                return False, "the answer contains no numbers"
            if _covered_by(answer, tool_numbers or set()):
                return False, "the answer only repeats numbers a tool returned"
            return True, "the answer computed numbers beyond the tool result"
        question = (question or "").strip()
        if CLOCK_QUESTION.search(question):
            return False, "the numbers come from the injected clock"
        if SELF_REFERENTIAL.search(question):
            return False, "the question is about local context"
        if CLOCK_IN_ANSWER.search(answer or ""):
            return False, "the answer uses the injected clock"
        if mode == "always":
            return True, "always"
        if not HAS_NUMBER.search(answer or ""):
            return False, "the answer contains no numbers"
        return True, "the answer contains numbers"

    def _should_verify(self, question: str, answer: str, used_tools: bool,
                       tool_numbers: set[str] | None = None) -> bool:
        """Whether to spend a cloud call checking this answer's numbers.

        Measured failure this exists for: the local model answered a 3:1 SWR
        question with "eight percent reflected, half a dB" — confidently wrong,
        and never a hedge, so nothing else catches it. A question the model was
        *told* (the clock) or read from a tool (a sensor) is not at risk.
        """
        decision, why = self.should_verify(
            question, answer, used_tools,
            mode=self.args.verify, cloud="cloud" in self.brain.lanes,
            tool_numbers=tool_numbers or set(),
        )
        if not decision and why not in ("verification is off", "no cloud lane"):
            log(f"verify skipped: {why}")
        return decision

    async def _cloud_answer(self, question: str) -> tuple[str, dict] | None:
        """Ask the cloud lane one question cold, with room to work. Returns the
        extractable ANSWER line (or None if the reply had none)."""
        if "cloud" not in self.brain.lanes:
            return None
        lane = self.brain.lanes["cloud"]
        prompt = question
        if clock.wants_time_context(question):
            # the cloud lane cannot call host_clock, so a temporal question carries the
            # date with it ("how many days until Christmas" needs today)
            prompt = (f"{question}\n\n[host clock: "
                      f"{datetime.datetime.now():%A %d %B %Y, %I:%M %p}]")
        try:
            reply = await lane.chat(
                [{"role": "system", "content": VERIFY_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=self.args.verify_max_tokens,
            )
        except LaneError as exc:
            log(f"cloud answer failed ({exc})")
            return None
        match = re.search(r"ANSWER:\s*(.+)", reply.text, re.IGNORECASE | re.DOTALL)
        spoken = match.group(1).strip().splitlines()[0].strip() if match else ""
        if not spoken:
            log("cloud answer had no ANSWER: line; unusable")
            return None
        return spoken, reply.stats

    async def _verify(self, question: str, candidate: str) -> tuple[str, dict] | None:
        """Ask the cloud lane the same question, WITHOUT showing it the local answer.

        The candidate is deliberately withheld. Shown one, the model anchored on it:
        asked to check "25% reflected, about 6 dB" it "corrected" it to
        "11.1% reflected, 6 dB" — worse than the answer it was checking. Re-answering
        cold on the same question is the only version that has measured right.
        """
        lane = self.brain.lanes["cloud"]
        prompt = question
        if clock.wants_time_context(question):
            # the cloud lane cannot call host_clock, so a temporal question carries the
            # date with it ("how many days until Christmas" needs today)
            prompt = (f"{question}\n\n[host clock: "
                      f"{datetime.datetime.now():%A %d %B %Y, %I:%M %p}]")
        try:
            reply = await lane.chat(
                [{"role": "system", "content": VERIFY_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=self.args.verify_max_tokens,
            )
        except LaneError as exc:
            log(f"verify: cloud lane failed ({exc}); keeping the local answer")
            return None
        match = re.search(r"ANSWER:\s*(.+)", reply.text, re.IGNORECASE | re.DOTALL)
        spoken = match.group(1).strip().splitlines()[0].strip() if match else ""
        if not spoken:
            # never speak a wall of working; an unusable reply means keep what we have
            log("verify: no ANSWER: line in the reply; keeping the local answer")
            return None
        return spoken, reply.stats

    async def _answer(self, text: str, turn: int):
        """Answer a turn, running tools if the question wants them.

        Returns (spoken_text, handoff_args_or_None, note). Tool calls are executed
        here, in the bridge, so the model never touches Home Assistant directly.

        The clock and the workshop's formulas are answered from code before any
        model is invoked: both are authoritative facts, not language tasks, and the
        model was measured wrong on the arithmetic 5 times in 6.
        """
        if self.args.fastpath:
            for label, handler in self.FAST_PATHS:
                try:
                    spoken = handler(text)
                except Exception as exc:  # noqa: BLE001 - a parser must not break a turn
                    log(f"[turn {turn}] fast path {label} failed: {exc!r}")
                    spoken = None
                if spoken:
                    log(f"[turn {turn}] answered from the fast path ({label}), no model "
                        f"call: {spoken[:60]!r}")
                    return spoken, None, f"fastpath ({label})"

        if not self.args.llm:
            return text, None, "loopback"

        lane = self.brain.lanes.answer_lane(self.args.answer_lane)
        tools: list[dict] = []
        if self._should_offer_handoff(text):
            tools.extend(HANDOFF_TOOL)   # a list of schemas, not one schema
        if self.ha is not None and wants_ha_tools(text):
            # Reads are always offered for a house question. Writes only when the
            # utterance is an instruction: a remark ("the fan is on") and a
            # question must never be able to change a device.
            tools.append(LOOKUP_TOOL)
            if looks_imperative(text):
                tools.append(DO_TOOL)
        if self.ma is not None and wants_music_tools(text):
            tools.extend(self.ma.tools())
        if self.args.clock_tool and (clock.answer(text) or clock.wants_time_context(text)):
            tools.extend(CLOCK_TOOL)   # retired: the fast path answers these in µs
        # No clock injection. It cost ~14 tokens on every turn and made the model
        # append "the current time is..." to unrelated answers (observed three times).
        # The date now arrives only if the model asks for it via host_clock.
        context = ""
        history = self.history_for(self.args.device_name, text)
        if history:
            log(f"[turn {turn}] carrying {len(history)} earlier turn(s) into the prompt")
        tool_numbers: set[str] = set()
        seen_calls: set[str] = set()
        messages = [{"role": "system", "content": self.args.system_prompt}]
        messages.extend(history)
        if context:
            messages.append(
                {"role": "user", "content": f"{text}\n\n[current local time: {context}]"})
        else:
            messages.append({"role": "user", "content": text})
        if self.args.stream_tts and not tools:
            return await self._answer_streaming(messages, turn, text)

        reply = None
        for step in range(3):
            try:
                reply = await lane.chat(messages, tools=tools or None)
            except LaneError as exc:
                # Measured: given a tool schema, qwen2.5:3b sometimes emits a tool
                # call that Ollama discards, returning HTTP 200 with empty content
                # and `eval_count` > 0. Retrying without tools gets a plain answer
                # (often a hedge, which then escalates properly) instead of failing
                # the turn.
                log(f"[turn {turn}] model returned nothing usable ({exc}); "
                    f"retrying without tools")
                try:
                    reply = await lane.chat(messages, tools=None)
                except LaneError as exc2:
                    log(f"[turn {turn}] still nothing ({exc2}); answering briefly")
                    return ("Sorry, I couldn't work that one out just now.",
                            None, f"model error: {exc2}")
            log(f"[turn {turn}] LLM step{step + 1} {reply.latency_s:.2f} s {reply.stats} "
                f"tools={[c.get('function', {}).get('name') for c in reply.tool_calls]} "
                f"text={reply.text[:60]!r}")
            if not reply.tool_calls:
                final = reply.text
                if TOOL_SYNTAX.search(final or ""):
                    log(f"[turn {turn}] model wrote tool syntax as text "
                        f"({final[:50]!r}); retrying without tools")
                    try:
                        plain = await lane.chat(messages, tools=None)
                        final = plain.text
                    except LaneError as exc:
                        log(f"[turn {turn}] retry failed: {exc}")
                        return ("Sorry, I got confused by that one.",
                                None, "tool-syntax")
                    # The retry can return the SAME syntax, and qwen2.5:3b does it
                    # deterministically: "ask Muse what's on my schedule" with no
                    # handoff tool emitted `handoff needs='data'` on both passes,
                    # and this line spoke it out loud on the Dot. Tool syntax is
                    # never an answer - if the model wants a tool it does not have,
                    # say so in words.
                    if TOOL_SYNTAX.search(final or ""):
                        log(f"[turn {turn}] retry still emitted tool syntax "
                            f"({final[:40]!r}); refusing to speak it")
                        return self._no_source_line(), None, "tool-syntax (twice)"
                used_tools = step > 0
                note = ""
                if self._should_verify(text, final, used_tools, tool_numbers):
                    checked = await self._verify(text, final)
                    if checked:
                        verified, stats = checked
                        # compare the NUMBERS, not the wording: "the FSPL increases by
                        # 6 dB" and "the free space path loss increases by 6 dB" are the
                        # same answer and were being logged as a correction
                        if _numbers(verified) != _numbers(final):
                            log(f"[turn {turn}] verify corrected the answer: "
                                f"{final!r} -> {verified!r} {stats}")
                            note = "verified (corrected)"
                        else:
                            note = "verified (agreed)"
                        final = verified
                        self.verify_stats = getattr(self, "verify_stats", 0) + 1
                return final, None, note
            messages.append({"role": "assistant", "content": reply.text or None,
                             "tool_calls": reply.tool_calls})
            for call in reply.tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                raw = fn.get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    args = {}
                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if call_key in seen_calls:
                    # The model is looping on a fact it already has. Waiting for the
                    # step limit costs ~10 s of speech delay, so break out now and let
                    # the reasoning lane do the arithmetic.
                    log(f"[turn {turn}] {name} repeated the same call; escalating")
                    answered = await self._cloud_answer(text)
                    if answered:
                        log(f"[turn {turn}] loop -> reasoning lane: {answered[0][:70]!r}")
                        return answered[0], None, "loop→reasoning"
                    return ("Sorry, I got stuck working that one out.",
                            None, "loop")
                seen_calls.add(call_key)
                if name == "handoff":
                    needs = self._effective_needs(str(args.get("needs") or "data"),
                                                  str(args.get("question") or text))
                    question = str(args.get("question") or text)
                    if needs == "reasoning":
                        # Send the transcript, not the model's paraphrase. Measured:
                        # asked "FSPL at 440 MHz over 10 kilometres" the 3B restated it
                        # without the distance and the cloud correctly answered for
                        # 1 km (85.32 dB instead of 105.3). A paraphrase may drop a
                        # parameter; the user's words cannot.
                        if question.strip() != text.strip():
                            log(f"[turn {turn}] reasoning uses the transcript, not the "
                                f"restatement (was {question[:70]!r})")
                        answered = await self._cloud_answer(text)
                        if answered:
                            spoken, stats = answered
                            if self._looks_like_hedge(spoken):
                                # the cloud cannot see the owner's records either;
                                # that is the agent's job, and worth the wait
                                log(f"[turn {turn}] cloud answer was a refusal "
                                    f"({spoken[:60]!r}) -> agent lane")
                            else:
                                log(f"[turn {turn}] escalated to the cloud lane "
                                    f"(reasoning): {spoken[:80]!r} {stats}")
                                return spoken, None, "escalated (reasoning)"
                        else:
                            log(f"[turn {turn}] reasoning escalation unavailable; "
                                f"falling back to the agent lane")
                    return "", {"question": question, "needs": needs}, f"handoff ({needs})"
                if name == "host_clock":
                    kind = str(args.get("kind") or "both").lower()
                    now = datetime.datetime.now()
                    spoken = {"time": clock.spoken_time,
                              "date": clock.spoken_date}.get(
                        kind, lambda n: f"{clock.spoken_time(n)} {clock.spoken_date(n)}")(now)
                    log(f"[turn {turn}] host_clock({kind}) -> {spoken!r}")
                    # The model gets the spoken form and nothing else: handing it
                    # {"spoken", "digits", "iso"} at once produced
                    # "twenty-six hundred twenty-one year November twenty-first…"
                    result = {"spoken": spoken}
                    tool_digits = (now.strftime("%Y-%m-%d %H:%M") + " " + spoken)
                elif name.startswith("music_"):
                    result = await self._run_music_tool(turn, name, args)
                else:
                    # `text` is this function's parameter. `question` is only
                    # assigned inside the handoff branch below, so passing it here
                    # was an UnboundLocalError on every HA tool call - caught live
                    # on the Dot, not by guard_test, which called the tool
                    # function directly and never executed this line.
                    result = self._run_ha_tool(turn, name, args, text)
                tool_numbers |= _numbers(str(result) + " " + str(locals().get("tool_digits", "")))
                tool_digits = ""
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": json.dumps(result)[:700]})
        log(f"[turn {turn}] tool loop hit its step limit")
        spoken = (reply.text or "").strip() if reply else ""
        if not spoken:
            # Measured: given the clock tool, qwen2.5:3b loops asking for the date
            # instead of doing date arithmetic. That is a reasoning failure, so hand
            # it to the reasoning lane instead of apologising.
            answered = await self._cloud_answer(text)
            if answered:
                log(f"[turn {turn}] tool loop -> reasoning lane: {answered[0][:70]!r}")
                return answered[0], None, "tool-limit→reasoning"
            return ("Sorry, I got stuck working that one out.", None, "tool-limit")
        return spoken, None, "tool-limit"

    async def _run_music_tool(self, turn: int, name: str, args: dict) -> dict:
        if self.ma is None:
            return {"ok": False, "error": "music is not configured"}
        try:
            if name == "music_play":
                res = await self.ma.play(str(args.get("query", "")),
                                         args.get("target") or None)
                log(f"[turn {turn}] music play {res['playing']!r} on {res['target']}")
            elif name == "music_transport":
                res = await self.ma.transport(str(args.get("action", "")),
                                              args.get("target") or None)
                log(f"[turn {turn}] music {res['action']} on {res['target']}")
            elif name == "music_volume":
                res = await self.ma.volume(int(args.get("level", 50)),
                                           args.get("target") or None)
                log(f"[turn {turn}] music volume {res['volume']} on {res['target']}")
            else:
                return {"ok": False, "error": f"unknown tool {name!r}"}
            return {"ok": True, **res}
        except (MaError, ValueError) as exc:
            log(f"[turn {turn}] music tool {name} refused: {exc}")
            return {"ok": False, "error": str(exc)}

    #: answered from code, never predicted.
    FAST_PATHS = (("calc", calc.answer), ("clock", clock.answer))

    def _no_source_line(self) -> str:
        """What to say when the model reached for a tool it does not have.

        Honest by construction: it names where the answer would have come from
        instead of pretending to know. With hand-off off, saying "I can't check
        that here" is the whole point of running local-only - the alternative,
        measured on 2026-11-22, was inventing "Muse had no scheduled activities
        today" from a misheard transcript.
        """
        name = self.args.muse_name or "Muse"
        if self.args.handoff != "off" and self.brain.lanes.async_:
            return f"I couldn't reach {name} for that one."
        return f"That needs {name}, and it isn't available right now."

    async def _answer_streaming(self, messages: list[dict], turn: int, question: str):
        """Hand back sentence 1 as soon as it exists; announce the rest as they arrive.

        Measured: the controller BUFFERS its fetch (a growing HTTP body played only
        after EOF), so "speak while fetching" is impossible here. Sentence-per-
        announcement is the only shape, and it costs 0.23 s per announcement.

        The point of returning the FIRST sentence rather than the whole reply: on a
        3-sentence local answer the first sentence existed 5.5 s before the last, and
        returning the full text would have thrown all of that away.
        """
        lane = self.brain.lanes.answer_lane(self.args.answer_lane)
        queue: asyncio.Queue = asyncio.Queue()
        speaker = asyncio.create_task(self._speak_queue(queue, turn))
        self._bg_tasks.add(speaker)
        speaker.add_done_callback(self._bg_tasks.discard)
        holder: dict = {"first": None, "first_at": 0.0}
        ready = asyncio.Event()

        async def pump() -> None:
            buf, started = "", time.monotonic()
            try:
                async for piece in lane.stream_chat(messages):
                    buf += piece
                    while True:
                        match = SENTENCE_END.search(buf)
                        if not match:
                            break
                        sentence, buf = buf[:match.end()].strip(), buf[match.end():]
                        if len(sentence.split()) < 4:
                            buf = sentence + " " + buf      # too short: keep buffering
                            break
                        if holder["first"] is None:
                            holder["first"] = sentence
                            holder["first_at"] = time.monotonic() - started
                            log(f"[turn {turn}] sentence 1 at {holder['first_at']:.2f}s "
                                f"handed over: {sentence[:50]!r}")
                            ready.set()
                        else:
                            await queue.put(sentence)
            except LaneError as exc:
                log(f"[turn {turn}] stream failed after "
                    f"{time.monotonic() - started:.2f}s: {exc}")
            tail = buf.strip()
            if tail:
                if holder["first"] is None:
                    holder["first"] = tail
                    holder["first_at"] = time.monotonic() - started
                else:
                    await queue.put(tail)
            log(f"[turn {turn}] stream finished at {time.monotonic() - started:.2f}s")
            ready.set()
            await queue.put(None)

        task = asyncio.create_task(pump())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        await ready.wait()
        if holder["first"] is None:
            return ("Sorry, I didn't get an answer to that.", None, "stream-empty")
        return holder["first"], None, f"streamed (sentence 1 at {holder['first_at']:.2f}s)"

    def _cancel_stream_tasks(self) -> None:
        """Stop anything still generating or announcing from a previous turn.

        Measured gap found in review: a barge-in (`VoiceAssistantRequest(start=False)`)
        cancelled the turn's own task but not the sentence pump or the announcement
        queue, so sentence 2 of a streamed reply could be spoken over the user. Any
        new turn supersedes leftover speech for the same reason.
        """
        if self._bg_tasks:
            log(f"cancelling {len(self._bg_tasks)} stream task(s) from the previous turn")
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()

    async def _speak_queue(self, queue: asyncio.Queue, turn: int) -> None:
        """Announce queued sentences in order, waiting out a busy device."""
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            for attempt in range(6):
                if await self._announce(sentence):
                    break
                await asyncio.sleep(0.4 * (attempt + 1))
            else:
                log(f"[turn {turn}] gave up announcing {sentence[:40]!r}")

    async def _answer_no_stream(self, messages: list[dict], turn: int, question: str):
        """Non-streamed fallback with the same tool handling as the normal path."""
        lane = self.brain.lanes.answer_lane(self.args.answer_lane)
        reply = await lane.chat(messages)
        return reply.text, None, "stream-fallback"

    def _run_ha_tool(self, turn: int, name: str, args: dict, question: str = "") -> dict:
        if self.ha is None:
            return {"ok": False, "error": "Home Assistant is not configured"}
        if name == "ha_do" and not looks_imperative(question):
            # Defence in depth. The write tool is only *offered* when the utterance
            # is an instruction, so this should be unreachable - but the gate is a
            # model-behaviour assumption, and ha.do() has no check of its own.
            # Measured with every tool offered on every turn (toolset_bench.py),
            # qwen2.5:3b chose ha_do for a question and for a remark in 4 of 28
            # turns; those would have actuated.
            log(f"[turn {turn}] refused ha_do: {question[:60]!r} is not an instruction")
            return {"ok": False, "error": "refused: the utterance is not an instruction"}
        try:
            if name == "ha_lookup":
                hits = self.ha.lookup(str(args.get("query", "")), limit=3)
                log(f"[turn {turn}] ha_lookup({args.get('query')!r}) -> "
                    + ", ".join(f"{h['entity_id']}={h['state']}" for h in hits[:3]))
                return {"ok": True, "results": [
                    {k: h[k] for k in ("entity_id", "name", "state", "unit") if k in h}
                    for h in hits]}
            if name == "ha_do":
                res = self.ha.do(str(args.get("query", "")), str(args.get("service", "")))
                if res.get("ok"):
                    log(f"[turn {turn}] HA {res['service']} {res['entity_id']} (was {res['was']})")
                else:
                    log(f"[turn {turn}] HA refusal: {res.get('error')} {res.get('candidates', '')}")
                return res
            return {"ok": False, "error": f"unknown tool {name!r}"}
        except HaError as exc:
            log(f"[turn {turn}] HA tool {name} refused: {exc}")
            return {"ok": False, "error": str(exc)}

    async def _dump(self, pcm: bytes, turn: int) -> None:
        os.makedirs(self.args.outdir, exist_ok=True)
        path = os.path.join(self.args.outdir, f"turn{turn:03d}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(MIC_WIDTH)
            w.setframerate(MIC_RATE)
            w.writeframes(pcm)
        log(f"[turn {turn}] captured {len(pcm) / 32000:.2f} s -> {path}")

    def send_event(self, event: ET, data: dict[str, str] | None = None) -> None:
        if self.client is None:
            return
        try:
            self.client.send_voice_assistant_event(event, data)
            log(f"[turn {self.turn}] -> {event.name} {data or ''}".rstrip())
        except Exception as exc:  # noqa: BLE001
            log(f"[turn {self.turn}] send {event.name} failed: {exc!r}")

    def finish(self, why: str) -> None:
        log(f"[turn {self.turn}] done: {why} (+{(time.monotonic() - self.t_start) * 1000:.0f} ms)")
        if self.args.once:
            # Keep serving until the controller says playback finished; the
            # 5 s fallback covers a turn that produced no audio at all.
            asyncio.create_task(self._exit_when_played())

    async def _exit_when_played(self) -> None:
        # The reply may be many seconds of speech, and AnnounceFinished only
        # arrives when playback ENDS.  A flat grace killed the media server
        # mid-turn on an 11 s answer.
        grace = max(5.0, self.audio_s + 10.0)
        try:
            await asyncio.wait_for(self.playback_done.wait(), grace)
        except asyncio.TimeoutError:
            log(f"no playback confirmation within {grace:.0f} s - exiting anyway")
        await asyncio.sleep(1.0)  # let the last bytes drain
        self.done.set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_stop(self, expected: bool) -> None:
        log(f"connection closed by peer (expected={expected})")
        self.done.set()

    async def run(self) -> int:
        # Validate configuration before binding a port: an async lane can never
        # answer a turn, and that should fail here rather than mid-conversation.
        self.brain.lanes.answer_lane(self.args.answer_lane)

        self.media_dir = os.path.abspath(os.path.join(self.args.outdir, "media"))
        os.makedirs(self.media_dir, exist_ok=True)
        self.args.media_host = self.args.media_host or lan_ip()
        serve_dir(self.media_dir, self.args.media_port)
        log(f"media at http://{self.args.media_host}:{self.args.media_port}/")

        self.brain.load()
        log("lanes: " + " | ".join(
            f"{lane.name}={'sync' if lane.sync_capable else 'async'}"
            for lane in self.brain.lanes
        ))
        if self.brain.lanes.async_ and self.args.handoff == "off":
            log(
                "hand-off is off, so these async lanes will not be used: "
                + ", ".join(lane.name for lane in self.brain.lanes.async_)
            )

        log(f"connecting to ws://{self.args.host}:{self.args.port}")
        self.client = APIClient(
            self.args.host, self.args.port, client_info="voice-bridge", keepalive=20
        )
        try:
            await self.client.connect(on_stop=self.on_stop)
        except Exception as exc:  # noqa: BLE001
            log(f"connect failed: {exc!r}")
            return 2
        info = await self.client.device_info()
        log(f"connected to {getattr(info, 'name', None)!r} ({getattr(info, 'model', None)})")

        self.client.subscribe_voice_assistant(
            handle_start=self.handle_start,
            handle_stop=self.handle_stop,
            handle_audio=self.handle_audio,
            handle_announcement_finished=self.handle_announcement_finished,
        )
        log("listening for the wake word")

        self._load_pending()
        if self.args.handoff != "off" and self.brain.lanes.async_:
            self.poller = asyncio.create_task(self._poll_handoffs())
            log(
                f"handoff: policy={self.args.handoff} lane={self.brain.lanes.async_[0].name} "
                f"poll every {self.args.handoff_poll_s:.0f}s"
            )

        try:
            await asyncio.wait_for(self.done.wait(), self.args.seconds or None)
        except asyncio.TimeoutError:
            log("timeout")
        finally:
            if self.poller is not None:
                self.poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.poller
            if self.ma is not None:
                with contextlib.suppress(Exception):
                    await self.ma.aclose()
            with contextlib.suppress(Exception):
                await self.brain.lanes.aclose()
            with contextlib.suppress(Exception):
                await self.client.disconnect()
            log("disconnected")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.1.10")
    p.add_argument("--port", type=int, default=16001)
    p.add_argument("--seconds", type=float, default=0, help="0 = run until killed")
    p.add_argument("--once", action="store_true", help="exit after the first turn")
    p.add_argument("--outdir", default="captures")
    p.add_argument("--media-port", type=int, default=8700)
    p.add_argument("--media-host", default=None)

    p.add_argument("--stt", action="store_true", help="transcribe with faster-whisper")
    p.add_argument("--llm", action="store_true", help="answer with Ollama (implies --stt)")
    p.add_argument("--tts", action="store_true", help="speak with piper (else a chime)")

    # small.en, not base.en: measured 2026-11-22 on real Dot audio, base.en's
    # transcript for a single sentence varied run to run ("S mu is what I must
    # schedule", "Ask news what's on my schedule", "Ask me what's on my
    # schedule") while small.en was stable and correct every time. ~1.9 s per
    # utterance instead of ~0.6 s; correctness is worth the 1.3 s here.
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--initial-prompt", default=ASR_PROMPT,
                   help="whisper vocabulary hint; '' to disable")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="local OpenAI-compatible endpoint")
    p.add_argument("--answer-lane", default="local", help="which sync lane answers a turn")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    p.add_argument("--piper-voice", default="voices/en_US-lessac-medium.onnx")

    # ── lanes other than the local model ────────────────────────────────────
    p.add_argument("--cloud-url", default="http://127.0.0.1:20129/v1",
                   help="OpenAI-compatible cloud gateway (OmniRoute)")
    p.add_argument("--cloud-model", default=None, help="enable the 'cloud' sync lane")
    p.add_argument("--local-name", default="Zion", help="how the local lane is named aloud")
    p.add_argument("--cloud-name", default="Muse", help="how the cloud lane is named aloud")
    p.add_argument("--cloud-key", default=None, help="bearer token for the cloud lane")
    p.add_argument("--cloud-max-tokens", type=int, default=80)
    p.add_argument("--cloud-system-prompt", default=None)
    p.add_argument("--lane-timeout", type=float, default=25.0)
    p.add_argument("--handoff", choices=("off", "ask", "auto"), default="ask",
                   help="offer the handoff tool to the local model ('off' disables it)")
    p.add_argument("--attribution", choices=("off", "handoff", "always"), default="handoff",
                   help="say which lane answered: only hand-offs (default), every turn, or never")
    p.add_argument("--ack-text", default=None,
                   help="spoken acknowledgement (default names the lane, e.g. 'I've asked Muse...')")
    p.add_argument("--handoff-poll-s", type=float, default=10.0)
    p.add_argument("--bus-type", default="voice-handoff",
                   help="message type Muse's hook filters on")
    p.add_argument("--device-name", default="Satellite-1",
                   help="device named in the handoff subject/body")
    p.add_argument("--handoff-ttl-s", type=int, default=900,
                   help="how long a hand-off stays answerable")
    p.add_argument("--handoff-max-attempts", type=int, default=12,
                   help="days-of-patience for a busy device before giving up")
    p.add_argument("--announce-timeout", type=float, default=30.0)
    p.add_argument("--muse", action="store_true",
                   help="enable the 'muse' async lane (agent bus; NOT wired into turns yet)")
    p.add_argument("--bus-url", default="http://192.168.1.10:8765")
    p.add_argument("--bus-token", default="~/.agent-bus/tokens/bridge")
    p.add_argument("--muse-agent", default="muse", help="bus agent to hand off to")
    p.add_argument("--muse-name", default="Muse", help="how that agent is named aloud")

    p.add_argument("--ha", dest="ha", action="store_true", default=True,
                   help="let the local lane use Home Assistant tools (default on)")
    p.add_argument("--no-ha", dest="ha", action="store_false")
    p.add_argument("--ha-url", default="http://192.168.1.20:8123")
    p.add_argument("--ha-token", default="~/.config/voice-bridge/ha-token")
    p.add_argument("--music", dest="music", action="store_true", default=True,
                   help="let the local lane control Music Assistant directly (default on)")
    p.add_argument("--no-music", dest="music", action="store_false")
    p.add_argument("--ma-url", default="ws://192.168.1.20:8095/ws")
    p.add_argument("--ma-token", default="~/.config/voice-bridge/ma-token")
    p.add_argument("--music-target", default="Satellite-1",
                   help="default speaker when the request does not name one")

    p.add_argument("--whisper-fallback", default="small.en",
                   help="second model re-hears a low-confidence transcript ('off' to disable)")
    p.add_argument("--stt-confidence", type=float, default=-0.75,
                   help="logprob below which the transcript is re-heard by the fallback model")
    p.add_argument("--fastpath", choices=("on", "off"), default="on",
                   help="answer clock/date and the workshop's formulas in code, not the model")
    p.add_argument("--clock-tool", dest="clock_tool", action="store_true", default=False,
                   help="offer host_clock() to the model (retired: the fast path is faster)")
    p.add_argument("--stream-tts", dest="stream_tts", action="store_true", default=True,
                   help="speak the first sentence while the rest is still generating")
    p.add_argument("--no-stream-tts", dest="stream_tts", action="store_false")
    p.add_argument("--verify-max-tokens", type=int, default=400,
                   help="room for the verifier to show its working before the ANSWER line")
    p.add_argument("--verify", choices=("off", "numbers", "always"), default="numbers",
                   help="have the cloud lane re-answer a local answer that contains "
                        "numbers (the local model is confidently wrong on arithmetic)")
    p.add_argument("--memory", choices=("off", "auto", "always"), default="auto",
                   help="carry recent turns: only for follow-ups (auto), always, or never")
    p.add_argument("--memory-turns", type=int, default=3, help="exchanges to keep")
    p.add_argument("--memory-ttl-s", type=float, default=300.0)

    p.add_argument("--silence-ms", type=int, default=1200,
                   help="quiet time that closes the utterance; too short cuts a "
                        "question at its first pause (600 ms did exactly that)")
    p.add_argument("--min-speech-ms", type=int, default=160)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    if args.llm:
        args.stt = True

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        return asyncio.run(Bridge(args).run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
