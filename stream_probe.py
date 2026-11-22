#!/usr/bin/env python3
"""stream_probe.py — does the controller play a growing WAV as it arrives?

Streaming TTS only works if the controller's fetch-to-playback path is incremental. If
it buffers the whole response before playing, there is nothing to gain from synthesising
early, and the design has to be different (multiple announcements, or no streaming).

The probe serves a WAV whose body is written slowly:

    t=0.0   HTTP 200, WAV header, first 1.5 s of audio
    t=1.5   (nothing — the client must already be playing or waiting)
    t=5.5   the next 1.5 s of audio, then EOF

So: 3 s of audio delivered across 5.5 s of wall time.

    played early  -> audio starts at ~0 s, finishes at ~5.5 s (it runs dry in the gap)
    buffered      -> nothing is heard until 5.5 s, then 3 s of audio

The server records when the GET arrives and when the body finishes; the controller's
own log records AnnounceRequest and "Playback complete". Compare the two clocks.

    systemctl --user stop voice-bridge      # single claimant
    .venv/bin/python stream_probe.py
    systemctl --user start voice-bridge
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import math
import socketserver
import struct
import threading
import time
import wave

from aioesphomeapi import APIClient

RATE, CHANNELS = 48000, 1
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"{time.monotonic() - T0:6.2f}s  {msg}", flush=True)


def pcm(seconds: float, freq: float, amp: int = 6000) -> bytes:
    n = int(RATE * seconds)
    frames = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * freq * i / RATE))
        frames += struct.pack("<h", v)
    return bytes(frames)


class Handler(http.server.BaseHTTPRequestHandler):
    #: seconds of audio per piece, and the gap between pieces
    pieces = (1.5, 1.5)
    gap = 4.0
    freqs = (990.0, 990.0)

    @classmethod
    def as_whole(cls) -> None:
        """One complete 1.5 s clip, sent at once — the baseline for overhead."""
        cls.pieces = (1.5,)
        cls.gap = 0.0

    def log_message(self, *args):  # keep the timing lines clean
        pass

    def do_GET(self):  # noqa: N802 - stdlib naming
        log(f"GET {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        # no Content-Length on purpose: the body is open-ended
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def chunk(data: bytes) -> None:
            self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        # WAV header with a placeholder data size (0xFFFFFFFF); ffmpeg reads to EOF
        header = bytearray(b"RIFF")
        header += struct.pack("<I", 0xFFFFFFFF)
        header += b"WAVEfmt "
        header += struct.pack("<IHHIIHH", 16, 1, CHANNELS, RATE,
                              RATE * CHANNELS * 2, CHANNELS * 2, 16)
        header += b"data" + struct.pack("<I", 0xFFFFFFFF)
        chunk(bytes(header))
        log("header sent")

        for index, seconds in enumerate(self.pieces):
            if index:
                log(f"silence gap {self.gap}s (nothing sent)")
                time.sleep(self.gap)
            body = pcm(seconds, self.freqs[index])
            chunk(body)
            log(f"sent {seconds}s at {self.freqs[index]:.0f} Hz")

        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        log("body finished (EOF)")


def serve(port: int) -> socketserver.TCPServer:
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def announce(host: str, port_: int, url: str, timeout: float, repeat: int = 1) -> None:
    client = APIClient(host, port_, client_info="stream-probe", keepalive=20)
    await client.connect()
    info = await client.device_info()
    log(f"connected to {getattr(info, 'name', None)!r}")
    for i in range(repeat):
        log(f"announce {i + 1}/{repeat}: {url}")
        try:
            finished = await client.send_voice_assistant_announcement_await_response(
                media_id=url, text="stream probe", timeout=timeout
            )
            log(f"  finished success={getattr(finished, 'success', None)}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not crash
            log(f"  announce failed: {exc!r}")
    await client.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.190.4")
    p.add_argument("--port", type=int, default=16001)
    p.add_argument("--media-port", type=int, default=8702)
    p.add_argument("--media-host", default="192.168.190.4")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--repeat", type=int, default=1,
                   help="announce the same clip N times, to measure per-announcement overhead")
    p.add_argument("--whole", action="store_true",
                   help="serve one complete 3 s clip instead of a buffered-then-gap stream")
    args = p.parse_args()

    if args.whole:
        Handler.as_whole()
    serve(args.media_port)
    log(f"serving on {args.media_host}:{args.media_port}")
    url = f"http://{args.media_host}:{args.media_port}/grow.wav"
    asyncio.run(announce(args.host, args.port, url, args.timeout, args.repeat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
