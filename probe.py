#!/usr/bin/env python3
"""Seam probe: take Home Assistant's seat on an EchoMuse satellite port.

Purpose
-------
Drive a complete voice turn against the EchoMuse controller with no AI in the
loop, so the transport is proven before whisper/ollama/piper exist.

It does the four things HA does, in the order the controller expects:

    RUN_START
    (mic audio arrives; we endpoint it ourselves - see below)
    STT_VAD_START / STT_VAD_END
    STT_END {text}
    INTENT_START / INTENT_END
    TTS_START / TTS_END {url}      <- the reply, fetched by the controller
    RUN_END

Endpointing is OURS, not the device's.  On this Echo the wake is a *private
listening* session (`oww_local_only`): the device streams 0x07 session audio and
never emits the 0x04 "turn ended" sentinel that a bounded turn would.  In that
mode the backend is the endpointing authority - HA does it with Silero - and if
we say nothing the controller waits out its 20 s hard cap.  So this probe
carries a plain energy VAD to close the utterance; the real bridge should use
Silero for the same job.

The reply is served over a local HTTP port and named by URL in TTS_END.  The
controller fetches it, decodes it, shapes it and plays it on the Dot - the same
path HA's TTS takes.  `--announce` additionally exercises the out-of-band
announcement path (VoiceAssistantAnnounceRequest), which is what timers and
notifications use.

    .venv/bin/python probe.py --port 16001 --once
    .venv/bin/python probe.py --port 16001 --announce --seconds 300
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import logging
import math
import os
import socket
import struct
import sys
import threading
import time
import wave

from aioesphomeapi import APIClient, VoiceAssistantEventType as ET

LOG = logging.getLogger("probe")

# The controller streams mic PCM as mono S16_LE at 16 kHz
# (docs/device-controller-interface.md).  Its reply media may be any rate; the
# controller decodes and resamples, so 48 kHz stereo is the safe default.
MIC_RATE = 16000
MIC_WIDTH = 2
MIC_CHANNELS = 1


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}  {msg}", flush=True)


# ── reply media ──────────────────────────────────────────────────────────────

def write_tone(path: str, freq: float = 660.0, seconds: float = 1.2, rate: int = 48000) -> None:
    """A short two-tone chime, so a pass is unmistakable by ear."""
    frames = bytearray()
    n = int(rate * seconds)
    fade = int(rate * 0.02)
    for i in range(n):
        env = min(1.0, i / fade, (n - i) / fade)
        t = i / rate
        f = freq if t < seconds / 2 else freq * 1.5
        v = int(8000 * env * math.sin(2 * math.pi * f * t))
        frames += struct.pack("<hh", v, v)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


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


# ── energy VAD ───────────────────────────────────────────────────────────────

def rms_s16(data: bytes) -> float:
    n = len(data) // 2
    if not n:
        return 0.0
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


class EnergyVad:
    """Close the utterance after a run of quiet frames.

    Deliberately plain: a noise floor that adapts on quiet frames, speech that
    must exceed 3x floor, and a hangover.  Good enough to prove the transport;
    the bridge should use Silero (what HA uses) for real rooms.
    """

    def __init__(self, silence_ms: int = 700, min_speech_ms: int = 250, ratio: float = 3.0):
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.ratio = ratio
        self.floor = 40.0
        self.speaking = False
        self.speech_ms = 0.0
        self.quiet_ms = 0.0
        self.ended = False

    def feed(self, data: bytes) -> str | None:
        """Returns 'start', 'end' or None."""
        rms = rms_s16(data)
        ms = len(data) / (MIC_RATE * MIC_WIDTH) * 1000.0
        if not self.speaking and rms < self.floor * self.ratio:
            self.floor = 0.9 * self.floor + 0.1 * max(rms, 20.0)
            return None
        if not self.speaking:
            if rms >= self.floor * self.ratio:
                self.speaking = True
                self.speech_ms = ms
                self.quiet_ms = 0.0
                return "start"
            return None
        # speaking
        if rms >= self.floor * self.ratio:
            self.speech_ms += ms
            self.quiet_ms = 0.0
        else:
            self.quiet_ms += ms
            if self.quiet_ms >= self.silence_ms and self.speech_ms >= self.min_speech_ms:
                self.ended = True
                return "end"
        return None


# ── the probe ────────────────────────────────────────────────────────────────

class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client: APIClient | None = None
        self.turn = 0
        self.done = asyncio.Event()
        self._reset_turn()

    def _reset_turn(self) -> None:
        self.mic = bytearray()
        self.vad = EnergyVad(self.args.silence_ms, self.args.min_speech_ms)
        self.t_start = 0.0
        self.t_first_audio = 0.0
        self.t_speech_end = 0.0
        self.answered = False
        self.answer_lock = asyncio.Lock()

    # ── inbound ──────────────────────────────────────────────────────────────

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        self.turn += 1
        self._reset_turn()
        self.t_start = time.monotonic()
        log(
            f"[turn {self.turn}] VoiceAssistantRequest(start=True) "
            f"conversation_id={conversation_id!r} flags={flags} "
            f"wake_word_phrase={wake_word_phrase!r} audio={audio_settings}"
        )
        self.send_event(ET.VOICE_ASSISTANT_RUN_START)
        return 0

    async def handle_audio(self, data: bytes, data2: bytes | None) -> None:
        # MUST be a coroutine: aioesphomeapi wraps the result in a task.
        if not data:
            return
        if not self.t_first_audio:
            self.t_first_audio = time.monotonic()
            log(f"[turn {self.turn}] first audio frame (+{(self.t_first_audio - self.t_start) * 1000:.0f} ms)")
        self.mic += data
        edge = self.vad.feed(data)
        if edge == "start":
            log(f"[turn {self.turn}] speech start (+{(time.monotonic() - self.t_start) * 1000:.0f} ms) rms_floor={self.vad.floor:.0f}")
            self.send_event(ET.VOICE_ASSISTANT_STT_VAD_START)
        elif edge == "end":
            self.t_speech_end = time.monotonic()
            log(
                f"[turn {self.turn}] speech end (+{(self.t_speech_end - self.t_start) * 1000:.0f} ms) "
                f"after {self.vad.speech_ms:.0f} ms speech, {self.vad.quiet_ms:.0f} ms quiet"
            )
            # Tell the controller to stop feeding the mic; it is waiting for us.
            self.send_event(ET.VOICE_ASSISTANT_STT_VAD_END)
            # Do not wait for end=True - the controller may or may not send it
            # promptly once we have claimed the endpoint.
            asyncio.create_task(self.respond())

    async def handle_stop(self, aborted: bool) -> None:
        log(f"[turn {self.turn}] audio ended by controller (aborted={aborted})")
        if aborted:
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
            self.finish_turn("aborted")
            return
        if not self.answered:
            await self.respond()

    # ── outbound ─────────────────────────────────────────────────────────────

    async def respond(self) -> None:
        async with self.answer_lock:
            if self.answered or self.turn == 0:
                return
            self.answered = True
            await self._write_utterance()
            self.send_event(ET.VOICE_ASSISTANT_STT_END, {"text": "(probe)"})
            self.send_event(ET.VOICE_ASSISTANT_INTENT_START)
            self.send_event(ET.VOICE_ASSISTANT_INTENT_END, {"continue_conversation": "0"})

            if self.args.announce:
                # Out-of-band path (timers / notifications), proven separately.
                await self.announce_tone()
                self.send_event(ET.VOICE_ASSISTANT_RUN_END)
                self.finish_turn("announced")
                return

            # Canonical turn reply: the controller fetches this URL and plays
            # it as the answer (em_earlytts / TTS_END data["url"]).
            url = self.media_url()
            self.send_event(ET.VOICE_ASSISTANT_TTS_START)
            self.send_event(ET.VOICE_ASSISTANT_TTS_END, {"url": url})
            self.send_event(ET.VOICE_ASSISTANT_RUN_END)
            self.finish_turn(f"tts url sent: {url}")

    def media_url(self) -> str:
        return f"http://{self.args.media_host}:{self.args.media_port}/{self.args.media_file}"

    async def announce_tone(self) -> None:
        url = self.media_url()
        log(f"[turn {self.turn}] announcing {url}")
        t0 = time.monotonic()
        try:
            finished = await self.client.send_voice_assistant_announcement_await_response(  # type: ignore[union-attr]
                media_id=url, text="probe tone", timeout=self.args.announce_timeout
            )
            log(f"[turn {self.turn}] AnnounceFinished success={getattr(finished, 'success', None)} after {time.monotonic() - t0:.2f} s")
        except Exception as exc:  # noqa: BLE001
            log(f"[turn {self.turn}] announce failed after {time.monotonic() - t0:.2f} s: {exc!r}")

    async def _write_utterance(self) -> None:
        os.makedirs(self.args.outdir, exist_ok=True)
        path = os.path.join(self.args.outdir, f"turn{self.turn:02d}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(MIC_CHANNELS)
            w.setsampwidth(MIC_WIDTH)
            w.setframerate(MIC_RATE)
            w.writeframes(bytes(self.mic))
        log(f"[turn {self.turn}] utterance {len(self.mic)} B = {len(self.mic) / 32000:.2f} s -> {path}")

    def send_event(self, event: ET, data: dict[str, str] | None = None) -> None:
        if self.client is None:
            return
        try:
            self.client.send_voice_assistant_event(event, data)
            log(f"[turn {self.turn}] -> {event.name} {data or ''}".rstrip())
        except Exception as exc:  # noqa: BLE001
            log(f"[turn {self.turn}] send {event.name} failed: {exc!r}")

    def finish_turn(self, why: str) -> None:
        log(f"[turn {self.turn}] turn done: {why} (+{(time.monotonic() - self.t_start) * 1000:.0f} ms)")
        if self.args.once:
            self.done.set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_stop(self, expected: bool) -> None:
        log(f"connection closed by peer (expected={expected})")
        self.done.set()

    async def run(self) -> int:
        log(f"connecting to ws://{self.args.host}:{self.args.port}")
        self.client = APIClient(
            self.args.host, self.args.port, client_info="voice-bridge-probe", keepalive=20
        )
        try:
            await self.client.connect(on_stop=self.on_stop)
        except Exception as exc:  # noqa: BLE001
            log(f"CONNECT FAILED: {exc!r}")
            return 2
        log("connected")

        info = await self.client.device_info()
        log(f"device_info: name={getattr(info, 'name', None)!r} model={getattr(info, 'model', None)!r}")

        entities, _services = await self.client.list_entities_services()
        log(f"entities: {[getattr(e, 'object_id', None) or e.name for e in entities]}")

        try:
            cfg = await self.client.get_voice_assistant_configuration(timeout=5.0)
            log(f"wake words: active={getattr(cfg, 'active_wake_words', None)!r}")
        except Exception as exc:  # noqa: BLE001
            log(f"wake word config failed: {exc!r}")

        self.client.subscribe_voice_assistant(
            handle_start=self.handle_start,
            handle_stop=self.handle_stop,
            handle_audio=self.handle_audio,
        )
        log("subscribed - say the wake word")

        try:
            await asyncio.wait_for(self.done.wait(), self.args.seconds or None)
        except asyncio.TimeoutError:
            log("timeout reached")
        finally:
            with contextlib.suppress(Exception):
                await self.client.disconnect()
            log("disconnected")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.1.10")
    p.add_argument("--port", type=int, default=16001)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--once", action="store_true")
    p.add_argument("--outdir", default="captures")
    p.add_argument("--announce", action="store_true", help="use the announcement path instead of a TTS reply")
    p.add_argument("--announce-timeout", type=float, default=30.0)
    p.add_argument("--media-port", type=int, default=8700)
    p.add_argument("--media-host", default=None)
    p.add_argument("--silence-ms", type=int, default=700, help="quiet time that closes the utterance")
    p.add_argument("--min-speech-ms", type=int, default=250)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    media_dir = os.path.abspath(os.path.join(args.outdir, "media"))
    os.makedirs(media_dir, exist_ok=True)
    args.media_file = "probe_tone.wav"
    write_tone(os.path.join(media_dir, args.media_file))
    args.media_host = args.media_host or lan_ip()
    serve_dir(media_dir, args.media_port)
    log(f"serving {media_dir} at http://{args.media_host}:{args.media_port}/")

    try:
        return asyncio.run(Probe(args).run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
