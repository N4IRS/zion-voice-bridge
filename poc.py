#!/usr/bin/env python3
"""poc.py — the whole voice loop, in one file, local only.

    Dot mic -> [this] -> whisper -> Ollama -> Piper -> Dot

No tools, no hand-off, no memory, no verification, no cloud, no vocabulary hints.
If a question needs something the model does not know, the model will make
something up — that is what a proof of concept is, and it is why the full bridge
exists.

Two protocol facts this file depends on, neither of which is in the ESPHome spec:

1. The Dot wakes in *private listening*, so it never sends a VAD-end sentinel.
   Whoever holds the connection must decide when the person stopped talking, or
   the controller waits out its 20 s hard cap. Here that is a 30 ms block RMS
   gate; the full bridge uses Silero because it is better at it.
2. The reply is not audio on the wire. TTS_END carries a URL, and the Dot fetches,
   decodes and plays it — so the HTTP server must stay up until playback finishes.

    python3 poc.py                 # run until Ctrl-C
    python3 poc.py --once          # one turn, exit after playback

Defaults: the Dot's ESPHome port 16001, Ollama on 127.0.0.1:11434, and the Piper
voice already on this bench.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import socket
import tempfile
import threading
import time
import urllib.request
import uuid
import wave

import numpy as np
from aioesphomeapi import APIClient
from aioesphomeapi import VoiceAssistantEventType as ET
from faster_whisper import WhisperModel
from piper import PiperVoice

HZ = 16000                      # the controller sends 16 kHz mono s16le
SYSTEM = ("You are a voice assistant in a ham radio workshop. Answer in one or two "
          "short spoken sentences. No lists, no markdown, no numbers read digit by digit.")


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def write_wav(path: str, samples, rate: int) -> None:
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.astype("<i2").tobytes())


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):        # one line per fetch is noise here
        pass


def serve(directory: str, port: int) -> http.server.ThreadingHTTPServer:
    """Serves the reply WAVs — the Dot fetches over HTTP, it never gets audio inline.

    Bound here, in the calling thread, so a port already in use is a startup
    failure rather than a thread that dies quietly while every reply goes unplayed.
    """
    handler = functools.partial(Quiet, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.port = port
    return httpd


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.190.104", 8123))     # routed, but no packet is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Vad:
    """Where does the sentence end? A 30 ms RMS gate with a tracked noise floor."""

    def __init__(self, silence_ms: int = 800, min_speech_ms: int = 200) -> None:
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.floor = 200.0          # tracked quiet level
        self.buf = bytearray()
        self.speech_ms = 0.0
        self.quiet_ms = 0.0
        self.talking = False

    def feed(self, data: bytes) -> str | None:
        self.buf.extend(data)
        block = int(HZ * 0.03) * 2                 # 30 ms of s16le
        edge = None
        while len(self.buf) >= block:
            chunk = bytes(self.buf[:block])
            del self.buf[:block]
            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
            level = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
            if level > max(450.0, self.floor * 2.5):
                self.speech_ms += 30
                self.quiet_ms = 0.0
                if not self.talking and self.speech_ms >= self.min_speech_ms:
                    self.talking, edge = True, "start"
            else:
                self.quiet_ms += 30
                self.floor = min(max(self.floor * 0.98 + level * 0.02, 80.0), 3000.0)
                if self.talking and self.quiet_ms >= self.silence_ms:
                    return "end"
        return edge


class Poc:
    def __init__(self, args) -> None:
        self.args = args
        self.media_dir = tempfile.mkdtemp(prefix="poc-media-")
        self.media_host = args.media_host or lan_ip()
        serve(self.media_dir, args.media_port)

        log("loading whisper")
        self.stt = WhisperModel(args.whisper, device="cpu", compute_type="int8",
                                cpu_threads=8, local_files_only=True)
        log("loading piper")
        self.tts = PiperVoice.load(args.piper_voice)
        self.rate = self.tts.config.sample_rate

        self.client: APIClient | None = None
        self.mic = bytearray()
        self.vad = Vad(args.silence_ms, args.min_speech_ms)
        self.task: asyncio.Task | None = None
        self.playback_done = asyncio.Event()

    # ── the four stages ──────────────────────────────────────────────────────
    def transcribe(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self.stt.transcribe(audio, beam_size=1, language="en",
                                          vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def ask(self, text: str) -> str:
        body = json.dumps({
            "model": self.args.model, "stream": False,
            "messages": [{"role": "system", "content": self.args.prompt},
                         {"role": "user", "content": text}],
            "options": {"temperature": 0.3, "num_predict": 90},
        }).encode()
        req = urllib.request.Request(f"{self.args.ollama}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            message = (json.load(response).get("message") or {})
        return (message.get("content") or "").strip() or "Sorry, I got confused by that one."

    def speak(self, text: str, path: str) -> float:
        chunks = list(self.tts.synthesize(text))
        audio = (np.concatenate([c.audio_float_array for c in chunks])
                 if chunks else np.zeros(0, np.float32))
        write_wav(path, (audio * 32767).clip(-32768, 32767), self.rate)
        return len(audio) / self.rate

    # ── the turn ─────────────────────────────────────────────────────────────
    def event(self, kind: ET, data: dict | None = None) -> None:
        if self.client is not None:
            self.client.send_voice_assistant_event(kind, data)

    async def respond(self) -> None:
        started = time.monotonic()
        if not self.vad.talking:
            # Measured on 2026-11-22: the wake word fired, nothing crossed the level
            # gate, and whispering the silence produced "Good time." - which the model
            # then answered with "The good time is 14:23 UTC." Never transcribe what
            # was never speech.
            log("no speech crossed the gate; not asking the models")
            self.event(ET.VOICE_ASSISTANT_ERROR, {"code": "stt-no-speech"})
            self.event(ET.VOICE_ASSISTANT_RUN_END)
            return
        pcm = bytes(self.mic)
        text = await asyncio.to_thread(self.transcribe, pcm)
        log(f"heard {text!r} ({(time.monotonic() - started):.1f}s in)")
        if not text:
            self.event(ET.VOICE_ASSISTANT_ERROR, {"code": "stt-no-text"})
            self.event(ET.VOICE_ASSISTANT_RUN_END)
            return
        self.event(ET.VOICE_ASSISTANT_STT_END, {"text": text})

        reply = await asyncio.to_thread(self.ask, text)
        log(f"reply {reply[:70]!r}")

        name = f"reply-{uuid.uuid4().hex[:6]}.wav"
        seconds = await asyncio.to_thread(self.speak, reply,
                                          os.path.join(self.media_dir, name))
        log(f"spoke {seconds:.1f}s of audio")
        self.event(ET.VOICE_ASSISTANT_TTS_START)
        self.event(ET.VOICE_ASSISTANT_TTS_END,
                   {"url": f"http://{self.media_host}:{self.args.media_port}/{name}"})
        self.event(ET.VOICE_ASSISTANT_RUN_END)
        log(f"turn took {time.monotonic() - started:.1f}s after speech ended")

    # ── the controller's callbacks ───────────────────────────────────────────
    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        self.mic = bytearray()
        self.vad = Vad(self.args.silence_ms, self.args.min_speech_ms)
        self.task = None
        log(f"wake {wake_word_phrase!r}")
        self.event(ET.VOICE_ASSISTANT_RUN_START)
        # Return 0, not None. aioesphomeapi treats this return value as the port of
        # a voice-assistant server it starts for the device; None makes it log
        # "Server could not be started" and send error=True on every wake. 0 means
        # "no server - send the audio over this connection", which is the path used
        # here. The full bridge has always returned 0, which is why its journal
        # never shows that error.
        return 0

    async def handle_audio(self, data: bytes, data2: bytes | None) -> None:
        if not data:
            return
        self.mic.extend(data)
        edge = self.vad.feed(data)
        if edge == "start":
            self.event(ET.VOICE_ASSISTANT_STT_VAD_START)
        elif edge == "end":
            self.event(ET.VOICE_ASSISTANT_STT_VAD_END)
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self.respond())

    async def handle_stop(self, aborted: bool) -> None:
        if aborted:
            log("aborted by the controller")
            if self.task and not self.task.done():
                self.task.cancel()
            self.event(ET.VOICE_ASSISTANT_RUN_END)
        elif self.task is None or self.task.done():
            self.task = asyncio.create_task(self.respond())

    async def handle_playback(self, finished) -> None:
        log(f"playback finished (success={getattr(finished, 'success', None)})")
        self.playback_done.set()

    # ── run ──────────────────────────────────────────────────────────────────
    async def run(self) -> int:
        self.client = APIClient(self.args.host, self.args.port,
                                client_info="poc", keepalive=20)
        await self.client.connect()
        log(f"connected to {self.args.host}:{self.args.port}")
        self.client.subscribe_voice_assistant(
            handle_start=self.handle_start,
            handle_stop=self.handle_stop,
            handle_audio=self.handle_audio,
            handle_announcement_finished=self.handle_playback,
        )
        log(f"listening — say the wake word (media on http://{self.media_host}:{self.args.media_port}/)")
        if self.args.once:
            await self.playback_done.wait()
            await asyncio.sleep(1.0)      # let the fetch finish before we exit
            return 0
        while True:
            await asyncio.sleep(3600)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.190.4")
    p.add_argument("--port", type=int, default=16001)
    p.add_argument("--media-port", type=int, default=8700)
    p.add_argument("--media-host", default="")
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--whisper", default="small.en")
    p.add_argument("--piper-voice",
                   default="/srv/WorkSpace/voice-bridge/voices/en_US-hfc_female-medium.onnx")
    p.add_argument("--prompt", default=SYSTEM)
    p.add_argument("--silence-ms", type=int, default=800)
    p.add_argument("--min-speech-ms", type=int, default=200)
    p.add_argument("--once", action="store_true", help="one turn, then exit")
    args = p.parse_args()
    try:
        return asyncio.run(Poc(args).run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
