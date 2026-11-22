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

from ha import (DO_TOOL, HA_TOOLS, LOOKUP_TOOL, HaError, HomeAssistant,
                looks_imperative, wants_ha_tools)
from lanes import BusLane, HttpLane, LaneError, Lanes, NotSyncCapable

LOG = logging.getLogger("bridge")

MIC_RATE = 16000          # the controller streams mono S16_LE at 16 kHz
MIC_WIDTH = 2

# Fed to whisper as `initial_prompt`. It costs nothing and it fixes domain words:
# without it, base.en heard "4 to 1 in petals step up" on this bench; with it,
# "4 to 1 impedance step up". small.en was no better and three times slower.
ASR_PROMPT = (
    "Questions about amateur radio antennas and RF: SWR, impedance, folded dipole, "
    "balun, duplexer, resonance, reflected power, dBm, MHz, coax, feedline, VFO."
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
    "If a question needs live data or knowledge you lack, call handoff — do not "
    "reply that you need Muse. Do not mention the current time unless asked."
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}  {msg}", flush=True)


# ── media serving ────────────────────────────────────────────────────────────

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        LOG.debug("http: " + fmt, *args)


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
HANDOFF_TOOL = [{
    "type": "function",
    "function": {
        "name": "handoff",
        "description": "Ask Muse, a cloud agent, when the question needs live data "
                       "or knowledge you lack.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "the question, restated"},
            },
            "required": ["question"],
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


class Brain:
    """whisper -> ollama -> piper, each stage optional."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.whisper = None
        self.piper = None
        self.piper_rate = 22050
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
            )
            log(f"whisper {self.args.whisper_model} loaded in {time.monotonic() - t:.1f} s")
        if self.args.tts:
            from piper import PiperVoice

            t = time.monotonic()
            self.piper = PiperVoice.load(self.args.piper_voice)
            self.piper_rate = self.piper.config.sample_rate
            log(f"piper {os.path.basename(self.args.piper_voice)} loaded in {time.monotonic() - t:.1f} s ({self.piper_rate} Hz)")

    def transcribe(self, pcm: bytes) -> tuple[str, float]:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        t = time.monotonic()
        segments, _info = self.whisper.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.args.initial_prompt or None,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
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
        self.poller: asyncio.Task | None = None
        self.pending: dict[str, dict] = {}
        self.pending_path = os.path.join(args.outdir, "pending-handoffs.json")
        self.ha = (HomeAssistant(url=args.ha_url, token_file=args.ha_token)
                   if args.ha else None)
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
        self._reset()
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
            pcm = bytes(self.mic)
            await self._dump(pcm, turn)

            # 1. speech to text
            if self.brain.whisper:
                text, stt_s = self.brain.transcribe(pcm)
                log(f"[turn {turn}] STT {stt_s * 1000:.0f} ms: {text!r}")
            else:
                text, stt_s = "(no stt)", 0.0
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
                log(f"[turn {turn}] local answer hedged ({hedge!r}) -> escalating")
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
                audio_s, tts_s = self.brain.speak(reply, path)
                self.audio_s = audio_s
                log(f"[turn {turn}] TTS {tts_s * 1000:.0f} ms for {audio_s:.2f} s of speech")
            else:
                write_tone(path)
                log(f"[turn {turn}] TTS: chime (no --tts)")
            if self.args.attribution == "always" and not handoff_note:
                reply = f"{self.brain.lanes.answer_lane(self.args.answer_lane).spoken_name} says: {reply}"
                self.brain.speak(reply, path)
            url = f"http://{self.args.media_host}:{self.args.media_port}/{name}"
            self.send_event(ET.VOICE_ASSISTANT_TTS_START)
            self.send_event(ET.VOICE_ASSISTANT_TTS_END, {"url": url})
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
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
        try:
            corr = await lane.submit(question, device=self.args.device_name)
        except LaneError as exc:
            log(f"[turn {turn}] handoff failed: {exc}")
            return "I couldn't reach Muse just now.", f"handoff error: {exc}"
        self.pending[corr] = {
            "device": self.args.device_name,
            "subject": f"voice: {self.args.device_name}",
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
            audio_s, _tts_s = self.brain.speak(text, path)
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
                    text = (msg.get("body") or "").strip()
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

    async def _answer(self, text: str, turn: int):
        """Answer a turn, running tools if the question wants them.

        Returns (spoken_text, handoff_args_or_None, note). Tool calls are executed
        here, in the bridge, so the model never touches Home Assistant directly.
        """
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
        context = time.strftime("%A %d %B %Y, %I:%M %p")
        messages = [
            {"role": "system", "content": self.args.system_prompt},
            {"role": "user", "content": f"{text}\n\n[current local time: {context}]"},
        ]
        reply = None
        for step in range(3):
            reply = await lane.chat(messages, tools=tools or None)
            log(f"[turn {turn}] LLM step{step + 1} {reply.latency_s:.2f} s {reply.stats} "
                f"tools={[c.get('function', {}).get('name') for c in reply.tool_calls]} "
                f"text={reply.text[:60]!r}")
            if not reply.tool_calls:
                return reply.text, None, ""
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
                if name == "handoff":
                    return "", args or {"question": text}, "handoff requested"
                result = self._run_ha_tool(turn, name, args)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": json.dumps(result)[:700]})
        log(f"[turn {turn}] tool loop hit its step limit")
        return (reply.text if reply else ""), None, "tool-limit"

    def _run_ha_tool(self, turn: int, name: str, args: dict) -> dict:
        if self.ha is None:
            return {"ok": False, "error": "Home Assistant is not configured"}
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
        if self.brain.lanes.async_:
            log(
                "async lanes are configured but not wired into turns yet: "
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

    p.add_argument("--whisper-model", default="base.en")
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
