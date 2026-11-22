# Zion Voice Bridge — proof of concept

A repurposed Amazon Echo Dot (running [EchoMuse](https://github.com/wilbowes/EchoMuse)'s
`emos` firmware) as a voice assistant that answers **locally** — whisper → Ollama →
Piper — hands questions it cannot answer to a **cloud agent over a message bus**, and
can read and switch **Home Assistant** devices by voice.

No Home Assistant Assist pipeline, no Alexa, no cloud speech.

> **Status: proof of concept.** Every path here was driven by real voice turns on real
> hardware. The parts that are thin are documented rather than hidden — see
> `VOICE-BRIDGE.md` §13–§14, including a blunt list of what the message bus still
> needs before anything should depend on it.

## What it does

```
Echo Dot (emos firmware: wake word, VAD, AEC, LED ring, playback)
   │  wss  /control + /data
   ▼
echomuse-controller            device supervisor, satellite protocol, OTA, timers
   │  ESPHome voice-assistant API (we take Home Assistant's seat)
   ▼
voice-bridge                   this project — one Python process
   ├─ Silero VAD               endpointing (the backend's job, not the device's)
   ├─ faster-whisper base.en   speech → text
   ├─ Ollama qwen2.5:3b        local answers, tool calling
   ├─ Piper                    text → speech
   ├─ agent bus  (async)       hands questions to a cloud agent, announces the reply later
   └─ Home Assistant REST      reads sensors, switches lights, with a write gate
```

A local turn takes roughly 2–5 s of "thinking" after you stop talking. A bus
hand-off takes 30–60 s and is spoken **later, out of band**, with no wake word.

## Files

| file | purpose |
|---|---|
| `bridge.py` | the service: seat claim, VAD, turns, tools, hand-off, announcements |
| `lanes.py` | answer lanes — `HttpLane` (sync) and `BusLane` (async), with the `sync_capable` flag |
| `envelope.py` | the message contract (v1) for hand-offs and answers: TTL, correlation, typed errors |
| `ha.py` | Home Assistant as a *tool*: fuzzy lookup, safe writes, imperative gate |
| `probe.py` | frozen minimal transport probe (no AI) used to prove the seam |
| `tools_test.py` | tool-calling reliability check (15 trials, 3 tools, 1 negative control) |
| `VOICE-BRIDGE.md` | the full design and build record, with measurements and traps |

## Requirements

* Python 3.13, `ffmpeg`, `espeak-ng`
* An EchoMuse controller with a paired Echo Dot
* Ollama with a small tool-capable model (`qwen2.5:3b` measured at 13–15 tok/s on a
  Xeon E5-2637 v3, CPU only)
* Optional: Home Assistant on the LAN; an agent-bus agent that answers messages

```bash
python3 -m venv .venv
.venv/bin/pip install aioesphomeapi aiohttp faster-whisper piper-tts
```

Piper voice (63 MB), not included:

```bash
mkdir -p voices
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
  curl -sLO --output-file voices/$f \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$f"
done
```

## Run

```bash
.venv/bin/python bridge.py \
  --host 192.168.1.10 --port 16001 \
  --llm --tts --model qwen2.5:3b \
  --muse --muse-agent muse --handoff auto
```

| flag | meaning |
|---|---|
| `--stt` / `--llm` / `--tts` | enable stages independently; no `--tts` answers with a chime |
| `--answer-lane` | which sync lane answers (`local`, or `cloud` if configured) |
| `--cloud-model` | enable a cloud lane through any OpenAI-compatible gateway |
| `--muse` / `--bus-url` / `--bus-token` | enable the async hand-off lane |
| `--handoff {off,ask,auto}` | `auto` also escalates when the local answer hedges |
| `--verify {off,numbers,always}` | cloud-check a local answer that contains numbers (the 3B is confidently wrong on arithmetic) |
| `--attribution {off,handoff,always}` | say who answered ("Muse says: …") |
| `--ha` / `--no-ha` | Home Assistant tools on/off |
| `--silence-ms` | quiet time that ends the utterance (600 ms cuts a question at its first pause) |

**Prerequisite:** the EchoMuse controller allows one voice backend per device. Disable
that device's ESPHome entry in Home Assistant before starting, or the connection is
refused by design.

## Run it as a service

`deploy/voice-bridge.service` is a systemd **user** unit (the EchoMuse controller and
the agent bus both run that way on this host). Adjust the paths and addresses, then:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/voice-bridge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-bridge.service

loginctl enable-linger "$USER"      # so it starts at boot without a login
journalctl --user -u voice-bridge -f
```

`Restart=on-failure` with `RestartSec=10` and `StartLimitBurst=6` means a missing
controller retries rather than hot-looping, and `PYTHONUNBUFFERED=1` is set because a
block-buffered log looks frozen and has already been mistaken for "it decoded nothing".

## The contract is served by the bridge

The hand-off envelope is versioned, and the **running service is the authority** —
not the copy in this repo, which is a mirror and a discovery path:

```bash
curl http://<bridge>:8700/schema.json    # derived schema: types, fields, error codes, rules
curl http://<bridge>:8700/envelope.py    # the exact source the service is running
curl http://<bridge>:8700/healthz
```

All three send `Cache-Control: no-store`. That is deliberate: a cached schema is how
an adopting agent ends up speaking a version the bridge no longer accepts, which
happened once here when the conversation-memory fields landed.

## Configuration and secrets

Nothing sensitive is in this repo. Both tokens are read from files outside it:

| what | where | how |
|---|---|---|
| Home Assistant long-lived token | `~/.config/voice-bridge/ha-token` (mode 600) | `--ha-token` to point elsewhere |
| agent-bus token | `~/.agent-bus/tokens/<agent>` (mode 600) | `--bus-token` to point elsewhere |

The bus derives an agent's identity from its **token filename**, so the file must be
named after the agent.

## What is deliberately narrow

* **Writes are gated**, not trusted to the model: only `light`, `switch`, `fan`,
  `scene`, `script`; locks, garage covers, climate and alarms are refused; a name
  must resolve to exactly one entity; a write needs an imperative utterance
  ("turn off the light") — a question or a remark can only read.
* **No entity dumps.** 663 entities is ~80k tokens; the model gets ≤3 candidates.
* **Answers are capped** to ~30 words for announcements (~15 s of speech): longer
  audio was measured to break device playback reporting.

## Known limits

See `VOICE-BRIDGE.md` §13. The short list: no conversation memory across turns, no
systemd unit, no streaming TTS, one satellite (cross-device arbitration untested),
and the bus needs real work (§14.1).

## Licence / credits

EchoMuse (MIT) supplies the device firmware and the controller; this project is the
voice backend that replaces Home Assistant's Assist pipeline. Speech recognition is
OpenAI Whisper (via faster-whisper, MIT), speech synthesis is Piper (MIT), the local
model is Qwen2.5 (Apache-2.0).
