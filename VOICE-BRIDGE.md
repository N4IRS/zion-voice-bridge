# Zion Voice Bridge — design, build record and operating guide

**Status: proof of concept — working end to end.** A repurposed Amazon Echo Dot
answers spoken questions with a fully local stack (whisper → Ollama → Piper → Dot
speaker), hands questions it cannot answer to a cloud agent over a message bus, and
can read and switch Home Assistant devices. No Home Assistant Assist pipeline, no
Amazon.

It is a POC in the strict sense: every path below was driven by a real voice turn
on real hardware, and the parts that are thin are named in §13 and §14 rather than
hidden. The bus in particular is a home-grown store-and-forward queue that needs
substantial work before anything depends on it (§14).

**Written:** 2026-11-21, on the machine it runs on.
**Code:** `/srv/WorkSpace/voice-bridge/`
**Host:** a Xeon E5-2637 v3 workstation — the ThinkStation, called *Zion*. Every LAN
address in this document is illustrative (`192.168.1.x`); the deployment's are
private and are supplied by flags. Sanitising them is why `192.168.1.10` appears in
the traces where the real box used its own address.

**Naming (owner, 2026-11-21):** **Zion** is the local system — whisper,
Ollama and Piper on this machine. **Muse** is whatever answers from outside it —
the agent on the bus, or a cloud model behind OmniRoute. The assistant speaks as
Zion when it answers from here, and hands work to Muse when it cannot.

---

## Contents

1. Summary and status
2. What we built, and the three architectures considered
3. Parts inventory (hardware, software, models, files)
4. The EchoMuse controller: what it does, what we replaced
5. The two protocols in detail
6. Turn lifecycle, with a real measured trace
7. The local AI stack, with measured performance
8. Using it locally, and with a cloud lane
9. Observability and failure behaviour
10. Operations manual
11. Reproduce from scratch
12. Traps encountered (symptom → cause → fix)
13. Known limits and what is not built
14. The agent bus: hygiene, not transport — and what a production version needs
15. Roadmap
16. References

---

## 1. Summary and status

The Echo Dot is repurposed hardware: it runs **emos**, a replacement firmware from
the EchoMuse project, instead of FireOS. That firmware does the hard local work —
wake word, voice activity detection, echo cancellation, beamforming, LED ring,
buttons, volume, and the speaker output chain. What it lacks is a *voice backend*:
today that is Home Assistant's Assist pipeline, which does speech-to-text, intent
and text-to-speech in the cloud.

This project replaces Home Assistant's seat with a small Python process that does
all three stages locally with open models, and answers through the same protocol
Home Assistant uses.

```
                 LAN only ──────────────────────────────────────────────
  Echo Dot (2nd gen, board 'biscuit'; emos firmware, wake word on device)
        │
        │  ws://  /control (JSON)  /data (PCM)     device protocol
        ▼
  echomuse-controller container        :8767 device plane, :8768 dashboard
        │                              one ESPHome "satellite port" per device (16001+)
        │  ESPHome voice assistant API (ws://, plaintext)
        ▼
  voice-bridge (this project)          Python + asyncio, one process
        ├── Silero VAD                   endpointing (the backend's job)
        ├── faster-whisper base.en       speech → text
        ├── Ollama qwen2.5:3b            text → reply   (optional cloud lane)
        └── Piper hfc_female-medium     reply → speech
        └── HTTP :8700                   serves the reply WAV the controller fetches
```

**What works today (verified by live turns):**

| Capability | Evidence |
|---|---|
| Take HA's seat on the controller's satellite port | connected, listed entities, read wake-word config, subscribed |
| Drive a complete voice turn | 7 turns, controller reports `outcome=ok` |
| Correct transcription of real speech | *"What time is it?"*, *"Tell me about Zion"*, *"What model are you using?"* |
| Local LLM answers | `qwen2.5:3b`, 10.9–14.9 tok/s |
| Local speech synthesis | Piper, ~0.25–0.44 s for 4–11 s of speech |
| Dot speaks the reply | controller fetched 1 MB in 105 ms, device played 163 frames |
| Clean failure paths | `no_speech`, `no_tts`, `tts_error` all observed and handled |

**Built since this summary was first written:** the hand-off trigger and the Muse
lane (§8.3), conversation memory (§8.11), numeric verification (§8.12), the clock as
a host tool (§8.8), sentence-level streaming TTS (§8.9), Music Assistant (§8.5) and
Home Assistant tools (§8.6), all under systemd (§10.3).

---

## 2. What we built, and the three architectures we considered

The Dot cannot be driven directly. Its firmware dials *out* to a controller, in a
protocol of its own (documented, see §15). Only after that controller accepts it
does the controller open an ESPHome voice-satellite port that a *voice backend*
(Home Assistant) dials into. So a voice backend can be put in at two different
seams, and we evaluated three options:

### Option A — keep the controller, take Home Assistant's seat  ← **chosen**

The bridge is an ESPHome API client on the controller's per-device port. The
controller keeps arbitration, LED ring, barge-in, ducking, timers, music,
Sendspin, OTA updates, the dashboard and device supervision. The bridge only
supplies: audio in → text → reply → audio out.

*Pros:* smallest amount of new code; keeps firmware OTA alive (emos is actively
developed); keeps two-Dot arbitration; nothing about the device changes.
*Cons:* depends on the EchoMuse controller container; must be a faithful HA
client (the event stream is part of the contract).

### Option B — write a device-protocol server, drop the controller

Speak `/control` + `/data` to the Dot directly. ~300 lines, no controller.

*Pros:* fewest moving parts; the Dot talks to one process.
*Cons:* you reimplement turn management, and you lose OTA (the firmware would be
frozen at v2.18.1), the dashboard, timers, music/Sendspin configuration, and
cross-device arbitration. mDNS identity (`_emcontroller._tcp`) and the device's
pinned CA both have to be handled.

### Option C — keep Home Assistant, point its pipeline at Ollama

Enable HA's Ollama conversation agent plus Wyoming whisper/Piper and leave the
Dots on Assist.

*Pros:* almost no code (configuration only); HA's entity layer comes free.
*Cons:* HA stays in the voice path, which was explicitly not wanted; the pipeline
adds stages; and the answers are bounded by HA's prompt size, which is a real cost
on this CPU (~34 tok/s prompt eval).

**Decision:** Option A. It preserves everything the controller already does
(including the only firmware-update path for the hardware) while replacing exactly
the piece we wanted to replace.

---

## 3. Parts inventory

### 3.1 Hardware

| Part | Detail |
|---|---|
| Satellite | **Amazon Echo Dot (2nd gen)**, board id `biscuit`, device id `Satellite-1`, IP 192.168.1.31 |
| Firmware | `emos-v0.10` (`v2.18.1`), armv7l, 3.18.19 kernel — replaces FireOS |
| Second satellite | `Satellite-2` (Satellite-2, .195, port 16002) — **offline** since 2026-11-06 |
| Host | ThinkStation, Xeon E5-2637 v3, 16 threads, 94 GB RAM |
| GPU | AMD FirePro W2100 — **unused and unusable** for any of this; everything is CPU |
| Home Assistant | separate box, 192.168.1.20:8123, HAOS, **2026.11.0** (formerly the backend) |

### 3.2 Software already running on the host (not built by us)

| Component | Version / detail |
|---|---|
| `echomuse-controller` | container, `ghcr.io/wilbowes/echomuse-controller:latest`, `EM_CONTROLLER_VERSION=v2.26.1`, `network_mode: host` |
| controller data | `/srv/compose/echomuse/data/` — `echomuse.db` (device registry, turns, logs), `tls/` (its CA + server cert) |
| controller ports | 8767 device plane, 8768 dashboard/API, 16001+ one satellite port per device |
| controller settings | `EM_ESPHOME_PORT_BASE=16001`, `EM_TTS_WAIT_TIMEOUT=30`, `OWW_MODEL=hey_jarvis`, `REQUIRE_DEVICE_TLS=0`, `DEVICE_APPROVAL=strict` |
| Ollama | container `ollama-p700`, image `ollama/ollama:0.24.0`, publishes `:11434`, models on `/srv/data/ollama`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=2` |
| OmniRoute | container, OpenAI-compatible gateway on `:20128`/`:20129` (also `:20132`); **1954 models** catalogued; provider keys held there; `REQUIRE_API_KEY=false` on this install, loopback only, so no key is needed by a local caller |
| espeak-ng | `/usr/bin/espeak-ng` (Piper's phonemiser dependency) |

### 3.3 Models

| Model | Size | Quant | Where | Measured speed |
|---|---|---|---|---|
| `qwen2.5:3b` | 2.23 GB | Q4_K_M | Ollama | **13.5–14.9 tok/s** generation, ~34–36 tok/s prompt eval |
| `qwen2.5:7b` | 4.68 GB | Q4_K_M | Ollama | **4.4 tok/s** generation |
| `nomic-embed-text` | 0.27 GB | F16 | Ollama | (not used here) |
| whisper `base.en` | ~74 MB | int8 (ctranslate2) | `~/.cache` via faster-whisper | 0.6–0.9 s per 2.5–3.4 s utterance |
| Silero VAD | ~2 MB | ONNX | bundled with faster-whisper | 0.19 s CPU per 20 s of audio |
| Piper `en_US-hfc_female-medium` (**deployed**) | 63 MB | ONNX | `voices/` | RTF 0.03 (~5 s of speech in ~0.15 s) |
| Piper `en_US-lessac-medium`, `-high`, `alba`, Kokoro-82M | — | — | `voices/`, `/tmp` | measured for §7.4.1; not deployed |

The escalation side is not a model but an OmniRoute **combo**: `voice-escalation` →
`cf/@cf/meta/llama-3.3-70b-instruct-fp8-fast` (0.45 s) → `gemini/gemini-2.5-flash`
(1.34 s), both verified answering (§8.2).

### 3.4 Python environment

```
python 3.13.5, venv at /srv/WorkSpace/voice-bridge/.venv  (567 MB)

aioesphomeapi   46.30.1   ESPHome native API client — the HA seat
aiohttp         3.14.5    LLM HTTP client
faster-whisper  1.2.1     STT (ctranslate2 4.9.0, onnxruntime 1.32.0)
piper-tts       1.8.0     TTS
numpy           2.5.5
protobuf        7.36.2
av              19.0.2    (only used by faster-whisper's file decoding, which we bypass)
```

### 3.5 Files in the project

| File | Role |
|---|---|
| `bridge.py` | **the working backend.** Seat claim, VAD, STT, LLM, TTS, media server |
| `probe.py` | frozen minimal transport test used to prove the seam (no AI) |
| `stream_probe.py` | the test that showed the controller buffers a growing WAV instead of playing it |
| `clock.py` | the host clock as a tool: spoken time and date, 22 selftest checks |
| `ma.py` | Music Assistant over its WebSocket API: play/transport/volume, `--selftest` |
| `ha.py` | Home Assistant as a *tool* (§8.6): fuzzy lookup, allowlisted writes, imperative gate, `--selftest` |
| `escalation_test.py` | routing policy: data→Muse vs reasoning→cloud, and hedge kinds |
| `stream_test.py` | streaming TTS: sentence 1 handed over early, cancellation on barge-in |
| `voices/en_US-hfc_female-medium.onnx` | deployed Piper voice (lessac and others also present) |
| `lanes.py` | lane abstraction: `HttpLane` (sync) and `BusLane` (async), the `sync_capable` flag, and `--selftest` |
| `envelope.py` | the message contract (v1): request/answer, TTL, correlation, typed errors, `schema_summary()` |
| `verify_test.py` | numeric-verification policy as a 10-case decision table |
| `memory_test.py` | conversation memory: both turn shapes, the cap, `memory=off` |
| `tools_test.py` | tool-calling reliability check (15 trials, 3 tools, 1 negative control) |
| `captures/turnNNN.wav` | per-turn mic audio, 16 kHz mono |
| `captures/media/` | reply WAVs served to the controller |
| `deploy/voice-bridge.service` | systemd **user** unit: unbuffered logs, restart backoff, linger |
| `bridge.log` | legacy per-turn log; since 2026-11-21 the service logs to journald |

---

## 4. The EchoMuse controller: what it does, what we replaced

The controller is not an AI component. It is the device supervisor and the
satellite-protocol server. Concretely, for each connected Dot it:

- accepts three WebSocket planes (`/control` JSON, `/data` binary PCM,
  `/shell/{id}` on demand), authenticates by `X-EM-Token` and a pinned CA;
- pushes device configuration (wake word model, VAD settings, EQ/limiter, LED
  scenes, volumes);
- runs the wake pipeline: the Dot scores its own wake word, the controller
  arbitrates between Dots (700 ms window), plays the wake cue, lights the ring;
- opens **one ESPHome satellite port per device** (16001+) and waits for a voice
  backend to dial in — single claimant, a second connection is refused;
- during a turn: gates the mic audio through Silero, streams it to the backend,
  receives the reply URL, fetches it, decodes with ffmpeg, applies the output
  chain (or defers to the device), and plays it, with ducking and barge-in;
- records a turn trace (`outcome`, stage timings, transcript) in its dashboard
  and in `turns`/`device_logs` in SQLite.

**We replaced:** only the voice backend. Everything in that list still runs
unchanged, which is why OTA, LEDs, arbitration, music and timers keep working.

**The consequence:** the controller's contract *is* our interface, and its
behaviour on this hardware is now documented in §5 and §12.

---

## 5. The two protocols in detail

### 5.1 Dot ↔ controller (device protocol)

Documented by the project in `~/EchoMuse/docs/device-controller-interface.md`.
Only the parts that matter to us:

**Audio formats**

| Direction | Frame | Format |
|---|---|---|
| mic → controller | `0x01` | mono **S16_LE, 16 kHz** |
| controller → speaker | `0x02` (audio), `0x03` (EOS) | **S16_LE, 48 kHz** |
| private-listening session audio | `0x07` | `[0x07][session u32][seq u16][PCM]`, 16 kHz |
| bounded-turn end markers | `0x04` speech-then-silence, `0x05` no speech | — |

**Why this matters:** the backend receives 16 kHz mono and must return 48 kHz.
Our reply WAVs are written by Piper at 22050 Hz mono and the controller resamples
them itself — it decodes whatever the URL returns with ffmpeg, so the bridge does
not need to match the speaker rate.

**The one architectural surprise:** when the Dot is in *private listening*
(`oww_local_only`, which it is here), the device **never sends the `0x04`
"speech ended" sentinel**. The backend is the endpointing authority. Home
Assistant does this with Silero; the bridge does the same (§7.1). Without it, the
controller waits out its **20 s hard cap** before giving up on the mic.

### 5.2 Controller ↔ backend (ESPHome voice assistant API)

Plaintext WebSocket (no noise PSK: both devices have `esphome_noise_psk = NULL`).
Server reports API 1.10; Home Assistant 2026.11.0 announced 1.19 as a client.

**Handshake**

```
client → HelloRequest(client_info, api_major, api_minor)
server → HelloResponse
client → DeviceInfoRequest            → DeviceInfoResponse
client → ListEntitiesRequest          → entities + done
client → VoiceAssistantConfigurationRequest → wake words (['hey_jarvis_v0.1'])
client → SubscribeVoiceAssistantRequest(subscribe=True, flags)
```

Entities exposed by the controller for one Dot: `media_player`,
`action_button` (event), `mic_muted` (binary sensor). There is **no**
`voice_assistant` entity — the assistant is driven by the subscription, not an
entity, so a client must not look for one.

**Turn, server → client**

| Message | Notes |
|---|---|
| `VoiceAssistantRequest(start=True, conversation_id, flags, wake_word_phrase)` | `flags=0` = device already detected the wake word; `wake_word_phrase` is the **human string** `'hey jarvis'`, empty for a button turn |
| `VoiceAssistantAudio` chunks | 16 kHz mono PCM, arrives only after the controller's speech gate opens |
| `VoiceAssistantAudio(end=True)` | controller has finished feeding the mic |
| `VoiceAssistantRequest(start=False)` | **abort** (barge-in) — the client must stop and discard the in-flight turn |
| `VoiceAssistantAnnounceRequest(media_id, text, start_conversation)` | out-of-band announcement (timers, notifications, `ask_question`) — a separate path from the turn reply |

**Turn, client → server (the event stream is mandatory)**

| Event | Data | Purpose |
|---|---|---|
| `RUN_START` | optionally `url` (early TTS) | first event of the run |
| `STT_VAD_START` / `STT_VAD_END` | — | endpointing signals; `VAD_END` tells the controller to stop the mic |
| `STT_END` | `text` | the transcript (this is what fills the controller's own turn log) |
| `INTENT_START` / `INTENT_END` | `continue_conversation` | intent stage markers |
| `TTS_START` | — | reply is being produced |
| **`TTS_END`** | **`url`** | **the reply audio URL the controller fetches and plays** |
| `RUN_END` | — | pipeline finished |
| `ERROR` | `code` | failure; the controller rings the failure cue |

**Three facts that cost real time to discover:**

1. **The reply URL belongs in `TTS_END` (`data["url"]`)**, not in an announcement.
   Using `VoiceAssistantAnnounceRequest` plays audio correctly but the controller
   logs the turn `outcome=no_tts` — announcements are the out-of-band path.
2. **`VoiceAssistantAnnounceFinished` flows the other way** — the *controller*
   sends it when playback genuinely finishes, and it is the honest end-of-turn
   marker. `RUN_END` only means the pipeline finished; the controller may still be
   fetching. A bridge that exits on `RUN_END` kills its own media server mid-fetch
   (this happened, twice).
3. **Single claimant is enforced in code.** A second connection gets
   `DisconnectResponse` and a close. HA must let go of the device before the bridge
   can take the seat — which is what disabling the ESPHome entry in HA achieves.

---

## 6. Turn lifecycle, with a real measured trace

Real turn (turn 1 of the continuous run), bridge log on the left of the arrows and
the controller's own trace on the right:

```
17:03:30.235  wake: conversation=bb930067 phrase='hey jarvis' flags=0
17:03:30.235  → RUN_START
17:03:33.210  speech end +2976 ms (1888 ms speech, 3.01 s captured)
17:03:33.210  → STT_VAD_END                       (controller stops the mic here)
17:03:34.118  STT 904 ms: 'What model are you using?'
17:03:34.118  → STT_END {'text': ...}
17:03:38.469  LLM 4.27 s (78 prompt tok, 25 reply tok, 10.9 tok/s)
17:03:38.736  → INTENT_START / INTENT_END / TTS_START
17:03:38.736  TTS 266 ms for 6.36 s of speech
17:03:38.736  → TTS_END {'url': http://192.168.1.10:8700/reply-001-39cb47.wav}
17:03:38.736  → RUN_END
17:03:38.736  reply handed over (+5524 ms from wake)

controller:  [TURN] outcome=ok total=+15057ms
             first_frame=+66  vad_start=+1054  vad_end=+2978
             audio=38frames/3040ms  stt=+3884  tts_url=+5419
             playback=+5419  | ha_think=+879ms tts_gen=+535ms
```

Reproduced across 7 turns:

| turn | transcript | STT | LLM | tok/s | reply tokens | TTS | speech produced |
|---|---|---|---|---|---|---|---|
| 1 | "What model are you using?" | 904 ms | 4.27 s | 10.9 | 25 | 266 ms | 6.36 s |
| 2 | "your muse." | 690 ms | 1.57 s | 14.9 | 16 | 152 ms | 3.90 s |
| 3 | "Tell me about Zion" | 593 ms | 3.21 s | 12.5 | 33 | 428 ms | 10.55 s |
| 4 | "What's the weather tomorrow?" | 627 ms | 2.36 s | 12.6 | 22 | 252 ms | 6.87 s |
| 5 | "What time is it?" | 659 ms | 4.18 s | 14.7 | 30 | 284 ms | 7.53 s |
| 6 | "Tell me about Zion." | 632 ms | 2.69 s | 14.8 | 31 | 300 ms | 8.60 s |
| 7 | (silence) | 6 ms | — | — | — | — | `no_speech` |

**Budget: "you stop talking" → "the Dot starts talking" is ~2.5–5.5 s**, of which
the language model is 1.6–4.3 s, whisper 0.6–0.9 s, Piper 0.15–0.44 s, and the
controller's fetch+decode ~0.1 s. Everything except the LLM is noise in the
budget.

---

## 7. The local AI stack, with measured performance

### 7.1 Voice activity detection — Silero

The bridge maintains an energy-independent endpointer using Silero (the same model
HA uses), pulled from faster-whisper's bundle: 512-sample windows (32 ms) with one
window of context, speech at p ≥ 0.5 sustained 160 ms, end at p < 0.35 for 600 ms.

*Why not an energy threshold:* measured on a captured turn in this room, the noise
floor is ~270 counts (S16) and speech peaks at ~1770 — a 3× floor threshold
started on the first noisy frame and never closed, so the turn ran to the
controller's 20 s cap. Silero is stationary-noise robust in exactly the way a
threshold is not.

Cost: 0.19 s of CPU for 20 s of audio (two threads' worth of one core, essentially
free).

### 7.2 Speech to text — faster-whisper `base.en`, int8

Fed raw 16 kHz PCM as a numpy array (never as a file — see the PyAV trap in §12).
Beam size 1, `language="en"`, VAD filter on. 0.6–0.9 s for a 2.5–3.4 s utterance;
0.62 s for a 20 s capture. `small.en` is available and slower; `base.en` has been
correct on every clean utterance so far, and its errors have been *reference*
errors ("your muse" for "you're Muse"), not comprehension failures.

**Proper nouns are the fragile part.** The vocabulary prompt fixed *"impedance"*, and
the same mechanism was needed for names: on a live turn **"Muse" was decoded as
"news"**, so the hand-off asked her about a news icon. `Muse`, `Zion`, `Clippy` and
`Satellite` now sit in `ASR_PROMPT` beside the RF terms — the assistant's own name (and
its counterpart's) appear in nearly every hand-off, which makes them the highest-value
words in the vocabulary.

### 7.3 Language model — Ollama, `qwen2.5:3b`

Two models are resident (`KEEP_ALIVE=-1`), so there is no cold-load on a turn.
3B is the practical default: **13.5–14.9 tok/s** against the 7B's **4.4 tok/s**.
Prompt evaluation runs at ~34–36 tok/s, which is the reason the system prompt is
deliberately short (~76 tokens) — every 100 tokens of prompt costs ~3 s here.

**Tool calling works.** Not from one anecdote — 15 of 15 correct across five
scenarios with three tools offered at once (`get_clock`, `get_weather`,
`get_radio_status`), three repeats each:

| prompt | expected | result |
|---|---|---|
| "What is the weather in Denver right now?" | `get_weather` | 3/3, `{"location":"Denver"}` |
| "What's the forecast for Denver tomorrow at noon?" | `get_weather` | 3/3, `{"location":"Denver","when":"tomorrow noon"}` |
| "What time is it?" | `get_clock` | 3/3, `{}` |
| "Is anyone connected to the P25 reflector right now?" | `get_radio_status` | 3/3, `{"what":"P25 reflector"}` |
| "What is a dipole antenna?" | **no tool** | 3/3, no call |

Three things that matter there: it picks the right tool out of three; it fills
argument values properly, including an optional temporal argument that was not
required; and it does **not** invent a tool call when none is needed — which is
what keeps the ordinary path free of extra round trips.

Reproduce with `.venv/bin/python tools_test.py`. Scope of the claim, stated
honestly: one model, short single-turn prompts, one
tool per turn, string arguments only. Not tested: a chain that needs two tools in
sequence, tools whose arguments are objects or enums, or selection under a long
conversation history. The reliability of *multi-step* tool use is the open
question, not whether tool use happens at all.

That capability is what makes the handoff design in §8.2 possible: escalation can
be a tool, so it costs nothing on turns that do not need it.

### 7.3.1 The prompt, not the model — measured

The same cloud model (70B) and the same two questions, changing only the system
prompt, produced right and wrong answers:

| system prompt | SWR 3:1 answer | verdict |
|---|---|---|
| "…under **25 words**" (first version) | "Eight percent reflected, about half a dB" | wrong, and a *different* wrong answer on a repeat run |
| "do arithmetic silently, state only the result" | "You lose 75 percent of your power and 6 dB" | wrong |
| "…state the result **with the number**… **be precise; do not guess**" | "You lose 25 percent of your power and 1.25 dB" | correct, 12 words |

A hard word budget makes a model skip the arithmetic and emit a confident prior;
the phrase that fixes it is "be precise; do not guess", not "work silently". The
current prompt keeps the answer short without rationing the working.

Two related results from the same session:

- **Never compress a big model's answer with the small one.** A two-phase design
  (70B answers, 3B condenses to 25 words) turned "25% … 1.25 dB" into "-6dB for
  3:1 SWR" and took 6–7 s. The small model destroys facts it does not understand.
- **Sampling matters for facts.** At `temperature=0.4` the SWR answer slipped from
  1.25 dB to 6 dB on one run of four; at `0.0` it was correct 4/4. The lanes
  default to 0.0 as a result.

### 7.3.2 Neither lane has a clock (superseded by §8.8)

> **Superseded.** This section records the first fix — injecting the time into the
> user turn — and why it was not good enough. The deployed mechanism is the
> `host_clock` tool described in **§8.8**: the model asks the host, and nothing is
> injected. Read this one for the reasoning, not for the implementation.

Asked "what time is it?" the cloud model answered *"It is 3:47 PM, the workshop
clock reads this time because its mechanical hands rotate in a precise circular
motion"* — at 18:48. It invents both the time and a justification, and "be precise"
does not stop it. The bridge therefore appends the real local time to the *user
turn* (not the system prompt, so a local lane's cached prefix still matches):

```
Question: what time is it?
[current local time: Saturday 21 November 2026, 06:50 PM]
```

Verified after the change: *"It is 6:50 PM."* This is the general shape for any
fact the model cannot have — the clock is the cheapest example, and the same slot
serves location, units, and session context.

### 7.4 Speech synthesis — Piper

Deployed voice: **`en_US-hfc_female-medium`** (§7.4.1 for why, and what else was
measured), 22050 Hz mono, ONNX on CPU. **RTF 0.03**: ~5 s of speech in ~0.15 s. Writes a WAV into `captures/media/`, which the bridge
serves over HTTP on `:8700` and hands to the controller as the `TTS_END` URL.

The controller decodes whatever the URL returns with ffmpeg and applies its output
chain, so the bridge need not match 48 kHz stereo.

### 7.4.1 Voice choice — `hfc_female`, measured

Seven voices were rendered on the same sentence and compared; the deployed voice is
`en_US-hfc_female-medium`.

| voice | RTF | synthesis for a 5 s clip | note |
|---|---|---|---|
| `lessac-medium` (was deployed) | 0.03 | 0.2 s | clear but flat — good intelligibility, little expression |
| `hfc_female-medium` (**now**) | 0.03 | 0.2 s | same cost, more expressive; also ~18% slower paced |
| `lessac-high` | 0.16 | 1.0 s | 5× the cost for a marginal gain — not worth it |
| `alba-medium` (GB) | 0.03 | 0.2 s | flattest pitch variation of the four Piper voices |
| Kokoro `af_heart` / `am_michael` / `bf_emma` | **0.36** | 2.2 s | most natural of the set, 12× the synthesis cost |

Kokoro-82M is the naturalness leader in the CPU class (it is what EchoMuse's own notes
reference for streaming TTS), but at RTF 0.36 on this box it adds ~2 s **before** the
Dot starts speaking, because the reply is synthesised whole rather than streamed. Fine
for announcements; noticeable on quick local turns. It stays a documented option, not
the default.

An F0-variation proxy (median pitch and interquartile spread) was computed as a
tiebreaker and then deliberately **not** used for the decision: it measures pitch
movement, not naturalness, and it ranked hfc_female *and* Kokoro differently from how
either sounds. Perceptual questions get perceptual answers — the samples were served
and judged by ear on the real output chain (controller decode → EQ → limiter → Dot
speaker).

### 7.5 Where the time actually goes

| Stage | Local | Notes |
|---|---|---|
| silence tail before endpointing | 0.6 s | `--silence-ms`; lower = snappier, more cutting-you-off |
| whisper | 0.6–0.9 s | 8 CPU threads, int8 |
| Ollama prompt eval | 1.1–1.3 s | ~76-token prompt; grows with system prompt |
| Ollama generation | 1.0–3.0 s | ~15 tok/s at 3B → ~20 tokens ≈ 1.4 s |
| Piper | 0.15–0.44 s | RTF 0.05 |
| controller fetch + decode | ~0.1 s | 1 MB in 105 ms measured |

---

## 8. Using it locally, and with a cloud lane

The bridge is one process with pluggable stages. The local lane is what runs
today. The general "hand this one off" mechanism is split into two *classes of
lane* by how fast an answer can come back, and `lanes.py` now implements both:
`HttpLane` (`sync_capable = True`) and `BusLane` (`sync_capable = False`). Only
the flag decides where a lane may be used, so wiring one into a turn is a config
change rather than a rewrite.

### 8.1 Mode 1 — fully local (current)

```
--llm --tts --model qwen2.5:3b      whisper base.en + Piper lessac
```

Nothing leaves the LAN at runtime. The only internet contact in the whole project
was installation: HuggingFace for the whisper and Piper models, the Ollama
registry for `qwen2.5`.

**Suitable for:** facts, short explanations, household/bench chatter, anything
where a 3B model's knowledge is enough, and anywhere the audio or the transcript
must not leave the house.

**Its characteristic failure** is not comprehension, it is *knowledge*: asked
"What time is it?" it says it has no real-time access; asked "Tell me about Zion"
it invents a religious answer. Those two hedges from the live turns are the
motivating examples for the handoff below.

### 8.2 Mode 2 — local, with a cloud handoff

Everything in mode 1, plus a second lane:

| | Lane L (local) | Lane C (cloud) |
|---|---|---|
| endpoint | `127.0.0.1:11434/v1` (Ollama, OpenAI-compatible) | `127.0.0.1:20129/v1` (OmniRoute) |
| model | `qwen2.5:3b` | **`voice-escalation`** — an OmniRoute priority combo (see below) |
| temperature | 0.0 | 0.0 — factual answers should not sample (§7.3.1) |
| latency | 1.6–4.3 s | ~1–3 s to first token, then usually fast |
| cost / privacy | zero, on-box | per token, transcript leaves |
| good at | short facts, tool calls, phrasing | multi-step reasoning, code, current knowledge, long context |
| bad at | anything needing facts it was not trained on | voice brevity without instruction (writes paragraphs) |

Both speak `/v1/chat/completions`, so a single HTTP client with a different
`base_url` and `model` serves both. Provider keys live in OmniRoute, not in the
bridge.

**The cloud lane is a combo, not a model id.** `voice-escalation` is an OmniRoute
priority chain, so provider failover is OmniRoute's job instead of ours:

| priority | model | measured |
|---|---|---|
| 1 | `cf/@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 0.45 s |
| 2 | `gemini/gemini-2.5-flash` | 1.34 s |

That order came from probing the catalogue: `zro/…` and `mistral/…` returned *404 no
active credentials*, `openai/…gpt-5.4-nano` *502 empty response*, `groq/…` *403* — the
catalogue is full of entries that do not answer, which is exactly what a combo exists
to survive. **Never route through an `auto/*` alias**: `auto/claude-sonnet` answered an
antenna question with a *coder* model, because those aliases route by availability, not
by intent.

**Creating a combo is an API call, not a UI visit.** The management API lives on the
*dashboard* port (`:20128`, not the OpenAI shim on `:20129`) and authenticates with the
vault key `omniroute_api_key` as `Authorization: Bearer`:

```bash
KEY=$(vault get omniroute_api_key)
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:20128/api/combos            # list
curl -s -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     --data @combo.json http://127.0.0.1:20128/api/combos                            # 201 Created
```

`POST /v1/combos` on the API port returns **405** — the OpenAI shim lists combos but
does not manage them. The body schema is `{name, description, strategy, models:[{id,
kind:"model", model, providerId, weight}], config:{targetTimeoutMs}}`; `providerId` is
the catalogue prefix (`cf`, `gemini`), discoverable from `/api/models` and
`/api/providers`.

**Measured through OmniRoute, 2026-11-21** (40-token reply, one question):

| model id | wall | served by |
|---|---|---|
| `cf/@cf/meta/llama-3.3-70b-instruct-fp8-fast` | **0.75 s** (TTFT **0.23 s** streamed) | itself |
| `auto/claude-sonnet` | 3.20 s | `@cf/qwen/qwen2.5-coder-32b-instruct` (!) |
| `ddgw/gpt-5.4-nano` | 32 s, HTTP 418 | nothing — an anti-abuse-challenged proxy |

Three conclusions worth keeping:

- **The cloud lane is not automatically slower than local.** A hosted 70B answered
  in 0.75 s where the on-box 3B takes ~2.5 s, so the spoken cue can be brief and
  the "5–10 s with a cue" budget is comfortable, not tight.
- **Never use an `auto/*` alias for the answer lane.** They route by availability,
  not intent — `auto/claude-sonnet` returned a *coder* model. Ask for a concrete id.
- **The catalogue contains dead entries.** Some providers fail with an anti-abuse
  challenge after half a minute. A lane must have a timeout and a fallback, and the
  chosen id must be one that has actually been exercised.

**The handoff itself is a tool call**, which is why it is cheap:

```
handoff(question, needs)    → needs="reasoning" (cloud lane) or "data" (Muse)
```

The local model calls it when it judges the job beyond it, and the tool argument
*restates* the question — which improves the cloud answer and doubles as the
privacy control (what leaves is the model's refined question, not the whole
house context). No classifier pass runs on ordinary turns, so the cost is zero
until it is used.

Two deterministic backstops, because a 3B will sometimes hedge instead of asking:

- **explicit ask** — "ask the big model", "look it up", "get a second opinion"
  (transcript match, no model needed);
- **hedge detection** — if the local reply contains "I don't know / not sure /
  can't access / no information", escalate anyway. Both live examples in §6 match.

**Policy switch:** `--handoff {off, ask, auto}`.
`off` = today's behaviour; `ask` = only when asked or when the model calls the
tool; `auto` = also on a hedge.

**Four details that decide whether it feels good:**

1. **Speak a cue before the wait** ("let me think about that"). A cloud hop adds
   1–3 s to first token; silence reads as a broken device. ~60 ms of Piper.
2. **Shape the answer for speech at the far end** — ask the cloud model for two
   sentences and cap tokens. A big model left alone writes three paragraphs, which
   is 30 seconds of talking.
3. **Lane stickiness.** If the cloud answers, keep the next few turns there (short
   TTL), or a follow-up lands back on the 3B with no context.
4. **Failure must speak.** Timeout / no key / provider down → fall back to the
   local answer with a one-line apology, never silence. Log every handoff with the
   exact text sent, model, tokens and latency, so "what left the box" is
   auditable.

### 8.3 The Muse lane — asynchronous by measurement

Muse is the agent on the bus: the owner's PDA and orchestrator, which tasks OpenCode
for code, coordinates state changes on Zion through Hermes, and brings results
back. "Hand the hard ones to Muse" is a real requirement, and it is a *different
kind of lane* from a bigger model — not slower by taste, slower by mechanism.

**Why it cannot answer inside a turn (verified 2026-11-21 on the bus host):**

| fact | evidence |
|---|---|
| the dispatcher routes only two agents | `bus-dispatch.py`: `HANDLERS = {"opencode": …, "hermes": …}`; everything else is "left alone (muse/admin poll)" |
| nothing polls Muse | Muse's last *delivered* message is `2026-11-13 23:36`; three results sit undelivered in its inbox, one from tonight (`Re: Ping test from Muse`) |
| even the fast direction is slow | bus → OpenCode round trip measured at **~2 minutes** |
| a voice turn tolerates far less | ~5–10 s with a spoken cue, 20–30 s absolute maximum |

So a synchronous Muse handoff would time out and fall back **every single time**.
That is not a tuning problem; it is the wrong shape.

**The shape that fits: submit, acknowledge, announce.**

```
you   "Hey Jarvis, ask Muse to price a replacement duplexer"
Dot   "I've asked Muse. I'll tell you when she answers."      ← local TTS, ~0.1 s
      [turn closes normally, outcome=ok]
 ...  40 s, five minutes, or tomorrow ...
Dot   "Muse says the Diamond MX-72 is in stock at HRO for $189."   ← announcement
```

The acknowledgement is local Piper, so it is instant. The answer arrives later as
a `VoiceAssistantAnnounceRequest` — **a path already proven on this bench**: on the
first probe turn the controller fetched our WAV, the Dot played it, and
`AnnounceFinished success=True` came back. No truncation to what fits in 30
seconds, and no timeout theatre.

The costs, stated plainly: the Dot must be idle to announce, so pending answers
queue behind a live turn; the answer is spoken whenever it lands, even if you have
walked away; and a pending map (`correlation_id → device, question, submitted_at`)
must be persisted so a reboot does not lose it.

**The flag.** `lanes.py` implements the contract and keeps `sync_capable` on the
lane where it belongs:

```python
class Lane:          sync_capable: bool
class HttpLane:      sync_capable = True     ask(text)      -> Reply (seconds)
class BusLane:       sync_capable = False    submit(text)   -> correlation_id
                                             poll(ids)      -> replies (later)
```

`Lanes.answer_lane(name)` is the only place the flag is consulted: it refuses an
async lane as the turn's answer lane, and the bridge validates this at **startup**
(not on the first turn). Verified:

```
lanes: local=sync | cloud=sync | muse=async
async lanes are configured but not wired into turns yet: muse

$ bridge.py --answer-lane muse --muse
NotSyncCapable: 'muse' cannot answer inside a turn (agent bus: minutes, not
seconds); submit() it instead and announce the reply when it lands
```

`python3 lanes.py --selftest` runs 18 checks over that contract — the flag, the
refusal, a real answer from the local lane, and the bus payload shape
(`to`, `type`, `subject`, `[voice turn …]` marker, token from file, `in_reply_to`
matching) against a stub session, so no live bus is needed.

**Auth, which is cheaper than it looks.** `bus_server.py` derives identity from
the *filename* in `~/.agent-bus/tokens/` and reads that directory at startup, so a
`bridge` agent is one file plus a restart — no admin call, no registration:

```bash
openssl rand -hex 32 > ~/.agent-bus/tokens/bridge
chmod 600 ~/.agent-bus/tokens/bridge
sudo systemctl restart <bus service>     # the server caches token hashes at boot
```

Do not reuse the `muse` or admin token: the bus stamps the sender from the token,
so a shared token would make handoffs look like they came from Muse herself.

**De-risking without waiting for Muse.** The requirement doc says not to build the
bridge side until Muse delivery exists — correct for a *synchronous* design, and it
no longer blocks the *asynchronous* one, because `opencode` is dispatched and
answers. So: prove the async machinery end to end with `--muse-agent opencode`
(the measured ~2 min is exactly the latency class being designed for), then flip
one flag to `muse` the day Muse polls. The part that waits on Muse is a string.

**`auto` escalation — built.** With `--handoff auto` the bridge also escalates when
the local answer *looks like a hedge*, which is the failure mode the local model
actually has: it does not lie confidently about the weather, it says it cannot
know. Markers are matched on the spoken answer:

```
i don't know / i'm not sure / i don't have / don't have access / can't /
unable to / without more context / cannot verify / check a reliable / as an ai  …
```

Verified on real strings: the three truth hedges from this bridge's own log
("I don't have access to real-time information", "…weather forecasting
capabilities", "not sure of its exact location without more context") all trigger,
while numeric and factual answers — "You lose 25 percent…", "A folded dipole has
a 4 to 1 step up…", "Zion is the ThinkStation at…" — do not.

**The meta-hedge.** Given the handoff tool, `qwen2.5:3b` still sometimes
*announces* the hand-off instead of performing it: asked "what time is sunset?" it
answered **"I need Muse for this."** with `tools=0`. That reads to the user as a
hand-off and queues nothing, so the detector also matches deferrals in words —
"need Muse", "ask Muse", "requires Muse", "Muse would", "hand it to Muse" — and
the system prompt now says *"when a question needs her, call the handoff tool —
never reply by saying you need her"*. A model that narrates still gets escalated,
because the sentence is the signal. When it fires, the
**user's own transcript** goes to Muse (there is no tool call to restate it) with
the hedge recorded as the reason, and the turn still gets the normal spoken ack.

**Working end to end with the real Muse — 2026-11-21 19:51.** Muse's hook came up
and the full loop ran on live questions: asked about tomorrow's weather, the bridge
queued it at 00:46:06, Muse replied at 00:46:37, and the Dot spoke
*"Muse says: Tomorrow in the coast will be partly cloudy, with a high of 81 and a
low of 61…"* — a 31-second round trip, announced with no wake word and no turn.

That took one more bug, and it is the most instructive of the project:

**Muse's hook replies with its own fresh `correlation_id` and no `in_reply_to`.**
So every reply arrived perfectly and matched nothing, and the bridge announced
nothing for four minutes of live testing. Ids cannot be assumed; the matcher now
does exact ids first (opencode echoes them) and falls back to **subject + time**: a
reply can only answer a question submitted before it arrived, and among those it
answers the most recent. Verified on five cases, including a stale reply that must
be dropped and a non-reply that must be ignored.

The open request on Muse's side: **echo `in_reply_to`** (one line in her `POST
/send`). The fallback works, but it resolves by time, so two questions in flight at
once could swap answers.

**Built and working (2026-11-21).** The async lane is no longer a design:

```
turn  → 3B calls handoff(question, needs)  → BusLane.submit() → spoken ack → turn closes
                                          ↓ correlation_id persisted
       poller every 5 s → inbox match → Piper → VoiceAssistantAnnounceRequest → spoken
```

Verified three times: a real turn (`corr=230818a6`, OpenCode answered in **12 s**),
and two seeded correlations with synthetic replies (matched, trimmed, announced,
`success=True`, pending cleared). The token, the send, the poll, the match by
`correlation_id` *or* the bus message id, the announcement and the persistence
across a restart all work.

**Attribution — who answered.** Each lane carries a `spoken_name` (`Zion` for the
local lane, `Muse` for anything outside), and `--attribution` decides how much of
it you hear:

| mode | a local answer | a hand-off answer |
|---|---|---|
| `handoff` (default) | unlabelled | ack names the lane, then **"Muse says: …"** |
| `always` | **"Zion says: …"** | **"Muse says: …"** |
| `off` | unlabelled | unlabelled |

`--local-name` / `--cloud-name` / `--muse-name` set the spoken names (defaults
`Zion`, `Muse`, `Muse`).

The ack is templated from the lane too (`"I've asked Muse. I'll tell you when the
answer comes back."`), which is a **behaviour change worth noting**: the local
system prompt used to open with *"You are Muse"*, so the 3B introduced itself as
Muse and the bridge then said *"I've asked Muse"* about a different agent. The
prompt now opens *"You are Zion, the local voice assistant…"* and names **Muse as
the cloud agent it can hand work to** — which also gives the model the words it
needs to decide a hand-off is appropriate.

Four limits measured while getting there — all of them the *device's*, and all of
them constraints on what an async lane may say:

| limit | measurement | consequence |
|---|---|---|
| long announcements are unreliable | a 40 s clip (3.7 MB) — the device never reported `playback_stats`, the controller timed out at 90 s, and the data plane dropped with `keepalive ping timeout` | trim to ~30 words (~15 s); **cap the announcement, not just the prompt** |
| completion is reported slowly | a 3.6 s announcement confirmed after **19.5 s** | a flat await timeout turns a good announcement into a "failure" — the timeout must exceed the audio, and a timeout that arrives *after* the audio should be treated as played |
| retries stack ducking | each retry added a `duck held (depth N)` | serialise announcements; back off between retries instead of retrying at poll cadence |
| agents ignore brevity | OpenCode answered a spoken question with a bulleted price list | put the voice instruction **in the bus message** ("one or two short sentences, no lists, no markdown"), do not hope |
| an empty reply is spoken as if it were an answer | `bus-dispatch` puts `(empty reply)` in the body when the agent returns nothing; the bridge announced it verbatim | filter placeholders (`_is_real_answer`), retry with backoff, and after the attempt limit say *"Muse didn't come back with an answer on that one"* |
| offering the tool triples the turn | system prompt + tool schema + time context = 349 prompt tokens at ~30 tok/s cold ≈ 8 s | trim to 221 tokens; steady state **3.8 s** with tools offered (measured over three identical calls) |

### 8.3.1 The envelope (v1) — the message contract

`envelope.py` defines what a hand-off and an answer *are*, so a bus message is a
contract rather than a paragraph. It exists to make three observed failures
impossible, not to be tidy.

```json
// request
{ "v": 1, "type": "voice.handoff", "id": "<uuid>", "created_at": "...Z",
  "ttl_s": 900, "from": "bridge", "to": "muse",
  "device": {"name": "Satellite-1", "kind": "echo-dot"},
  "reply_to": {"agent": "bridge", "bus_id": 1535},
  "request": {"question": "...", "voice": true, "max_words": 30, "hint": "..."} }

// answer
{ "v": 1, "type": "voice.answer", "id": "<uuid>", "created_at": "...Z",
  "from": "muse", "to": "bridge",
  "correlation": "<the request's id>",     // the link; echo it
  "in_reply_to": 1535,                      // belt and braces
  "status": "ok", "answer": {"text": "...", "words": 24}, "error": null }
```

**The three rules, each from a failure:**

1. `status=ok` requires non-empty `answer.text`; an agent that cannot answer sends
   `status=error` with a code. The bus's `(empty reply)` placeholder was spoken
   aloud once and must be unrepresentable.
2. The answer echoes `correlation` (and/or `in_reply_to`), so a correct answer is
   never left unmatched — the bug that cost four minutes of live testing.
3. `ttl_s` travels with the request. A worker skips anything past its TTL; the
   asker drops an answer that arrives late instead of speaking a stale fact.

**Backward compatible:** `parse()` returns `None` for anything that is not an
envelope, and the bridge falls back to plain text, so an agent that has not adopted
it still works. `python3 envelope.py --selftest` runs 18 checks, including "empty
ok-answer refused" and "stale hand-off expired".

**Bus API facts an adopting agent needs** (verified, not read):

- `POST /send` accepts `correlation_id` and `in_reply_to`; both are stored.
- `correlation_id` is **unique per (correlation_id, recipient)** — reusing it for
  several replies to the same agent is silently deduped. `in_reply_to` (the
  request's bus id) is the collision-free link.

**TTL belongs to the request, never the answer.** An answer envelope has no `ttl_s`,
so asking *it* whether it has expired always answers yes (fail-safe) — which silently
dropped every reply until the live envelope test caught it. The asker ages its own
pending entry instead.

**Where to fetch the contract (authority, not mirror).** The bridge serves it from
the running process, so it cannot disagree with the wire:

| route | returns |
|---|---|
| `http://<bridge>:8700/schema.json` | derived summary: version, types, required/optional fields, error codes, the three rules |
| `http://<bridge>:8700/envelope.py` | the exact source the service is running (byte-identical) |
| `http://<bridge>:8700/healthz` | `{"ok": true, "schema_version": 1, "types": [...]}` |

All three send `Cache-Control: no-store` — a cached schema is the failure this
exists to prevent. It happened on 2026-11-21: an agent fetched the repo copy the
night the memory fields landed and got a schema the bridge no longer spoke, and only
a live pull revealed it. The public repo stays the discovery path and mirror; **the
bridge copy is the authority**, because it comes from the process that answers.
`/schema.json` is generated by `envelope.schema_summary()`, not written twice, so it
cannot drift from the validators.

**Additive changes are safe — proven 2026-11-21.** A new optional field
(`request.tone`) was added to the hand-off, documented only in the served schema, and
one hand-off sent. The adopting agent, whose prompt predated the field, noticed it
before fetching anything (*"it carries a tone field my prompt does not describe, so I
will fetch the current schema"*), pulled `/schema.json` with no-store and a
`Last-Modified`, found `tone` under `optional` with its field note, ignored it as
documented, and reported the unknown field in its own summary. The answer went out
normally. So: **additive fields are discoverable and self-documenting, and the
fallback is wired to field presence rather than luck.**

One caveat worth keeping: that verdict was recovered from the worker's *run record*,
not from a delivered summary — the summary never reached the conversation. The
worker now also writes its summary to a file in its state directory, so the next
such test produces a file to read rather than a dig to perform. **Evidence that has
to be excavated is weaker evidence**, and it is worth the one line to make it
durable.

**Error codes.** Recommended: `no_data`, `unavailable`, `bad_question`, `timeout`,
`internal`. Validation requires a *non-empty* code but does not restrict it to this
set — an agent must not be rejected by an older bridge for inventing a code, so the
the bridge logs unknown codes rather than refusing them.

**Which id goes where — the one thing easy to get wrong:**

| field | value | notes |
|---|---|---|
| request `id` | a UUID | the hand-off's identity; this is what the answer must echo |
| request bus `correlation_id` | **the same UUID** | set by the bridge on send |
| request bus row `id` | an integer (e.g. 1545) | the bus's own numbering |
| answer `correlation` | the request's **UUID** | primary match |
| answer `in_reply_to` | the request's **bus id** (1545) | secondary match, collision-free |

The bridge also accepts a reply that puts the *bus id* in `correlation` — liberal in
what it accepts, strict in what it sends — but the UUID is the correct value.

**Adoption checklist (Muse's four steps):** match the hook on the envelope `type`,
not the bus type; read `question`/`device`/`correlation`/`ttl_s` out of the
envelope and skip expired ones; echo `correlation` in the reply; never post an
empty body — post `status=error` instead.

**Muse's side, answered 2026-11-21 (from Muse, who runs in Meta's cloud).** The
open question above is now closed, and the answer is architectural, not a tuning
detail:

- **Muse runs in a hosted VM in Meta's cloud.** Not on the ThinkStation, not an
  OpenCode server, not a CLI. No Muse process exists on this host and none can be
  installed — her code does not run on this hardware.
- **The direction of the tunnel is one-way.** Port 3130 is *her outbound* path to
  the tailnet — how she reaches the ThinkStation. It does not let this host reach
  her. So `handle_muse()` in `bus-dispatch.py` is impossible: there is nothing to
  call. The dispatcher can only store mail; **delivery has to be her side waking
  up.**
- **The mechanism:** a hook on Muse's side polling her inbox every **10–15 s**
  (minimum 5). A `voice-handoff` wakes a worker, the worker answers and posts the
  reply to the bus, and the bridge picks it up from its own inbox and speaks it.
- **The wait budget is therefore ~60 s, not 20–30 s** end to end, with the spoken
  cue covering it.

What the ThinkStation side provides as a result (all implemented):

| requirement | implementation |
|---|---|
| a distinct type her hook can filter on | `type: voice-handoff` (`--bus-type`) |
| the device, not an IP | subject `voice: Satellite-1`, body `[voice turn from Satellite-1]` (`--device-name`) |
| the restated question | the model's `handoff` question, or the transcript when a hedge triggered it |
| an ack that matches a minute-long lane | *"I've asked Muse. It may take about a minute."* |
| a reply path already built | the bridge polls its own inbox every 5 s and announces what arrives |
| backlog policy | her hook acts only on `voice-handoff` newer than its start, so the Nov-13 results and the stale ping are never actioned; the live ping stays as a canary |

**Still open (Muse's side, not the bridge's):** the hook should echo `in_reply_to`
with each answer. It currently replies with a fresh correlation id and no
`in_reply_to`, which the bridge survives by matching on subject plus time (§8.3) —
but that resolves by arrival order, so two questions in flight at once could swap
their answers.

### 8.4 Mode 3 — coexistence with Home Assistant

The controller allows **one** backend per device, so the split is per device, not
per turn:

- **Satellite-1** (.143, port 16001) → this bridge
- **Satellite-2** (.195, port 16002) → Home Assistant, as before

To move a device between backends: disable/re-enable its ESPHome entry in HA.
Whichever backend owns the device also owns its announcements and timers.

### 8.5 Music Assistant as a tool — direct, not through Home Assistant

**Decision (owner, 2026-11-21): talk to Music Assistant directly.** HA is used for
what only HA can do; MA has its own WebSocket API and going through HA adds a hop and
an abstraction for no gain.

`ma.py` speaks `ws://<ma>:8095/ws` with a JWT. Every command name below was
**measured against MA 2.10.12**, not recalled:

| command | args | returns |
|---|---|---|
| `auth` | `{"token": …}` | `{authenticated: true}` |
| `players/all` | — | players, including `volume_level` |
| `player_queues/all` | — | queues: `state`, `current_item` |
| `music/search` | `{"search_query", "limit"}` | `{tracks, albums, playlists, radio, …}` |
| `player_queues/play_media` | `{"queue_id", "media": <item or list>, "option": "replace"}` | — |
| `player_queues/play_pause` / `pause` / `stop` / `next` / `previous` | `{"queue_id"}` | — |
| `players/cmd/volume_set` | `{"player_id", "volume_level"}` | — |

**Two shape traps, each costing a round trip:**

1. **`play_media` wants the media *item object*** (as returned by `music/search`), not
   a `provider://type/id` string — a URI returns *"Value of type NoneType is invalid
   for media"*. `{"command":"auth","token":…}` likewise returns *"token required in
   args"*: the envelope matters.
2. **Playback is addressed by `queue_id`, not `player_id`.** For these players the two
   ids happen to be identical, which is the kind of coincidence that breaks on the
   next device.

**A search hit is not a promise.** A library entry from an unavailable provider is
returned by `music/search` and then refused by `play_media` (*"the requested media item
could not be found"*), which is how the first live attempt failed after the same query
had worked a minute earlier. `play()` now walks the candidate chain — track, album,
playlist, radio — instead of betting on one.

**Targets come from words and are refused when ambiguous:**

| said | resolves to |
|---|---|
| *(nothing)* | default `Satellite-1` (`--music-target`) |
| "on the dot" / "satellite" | `Satellite-1` |
| "on the esp32" / "a1s" | `ESP32-A1S` |
| "on the tv" / "roku" | **refused — two TVs; say which** |
| "on the moon" | refused, with the list of known players |

Verified live: *"play pink floyd"* → playing on `Satellite-1`;
*"play pink floyd on the esp32"* → playing on `ESP32-A1S`; *"stop the music"* → idle.
The tools are offered only when the utterance mentions music (`wants_music_tools`), so
a question about antennas pays nothing for them.

**Token.** A long-lived JWT (1 year) with `role: admin`, stored at
`~/.config/voice-bridge/ma-token` (mode 600, gitignored). Admin is more than a voice
bridge needs — MA's UI does not obviously offer less — so treat it as a standing
privilege and rotate it if the transcript it was pasted into is ever shared.

**The Dot is both the voice satellite and an MA player.** Music on `Satellite-1`
therefore collides with turn audio; EchoMuse ducks music under a voice turn
(`duckDb`, device `audio_mix`), which is why defaulting to the Dot is safe — but it is
the first thing to watch when a turn sounds odd over music.

### 8.6 Home Assistant as a tool (not as a pipeline)

**Decision (owner, 2026-11-21): bypass Hermes for now.** Hermes is the designated
write-master in the agent-bus design, but it has been offline since 2026-11-13 and
is being reworked, so the local lane talks to Home Assistant directly. That is a
deliberate, temporary bypass — it should be revisited when Hermes is rebuilt.

This does *not* put HA in the voice path. HA is a LAN API the local model may read
and act on when a turn is about the house; it is not the brain, not the STT/TTS,
and never speaks. The original "the bridge does not talk to HA" decision was about
Assist as a pipeline, and it still holds.

Two tools, both narrow (`ha.py`):

| tool | purpose |
|---|---|
| `ha_lookup(query)` | fuzzy-match entities by name/room, return ≤3 with state + unit |
| `ha_do(query, service)` | resolve a spoken name to **exactly one** writable entity, then act |

Measured: `/api/` 3.4 ms, all **663 entities in 8 ms** (305 KB — hence lookups, not
dumps: 305 KB would be ~80k tokens against a 220-token prompt budget).

**Safety lives in `ha.py`, not in the model:**

- writable domains are only `light`, `switch`, `fan`, `scene`, `script` — **locks,
  covers/garage doors, climate and alarms are refused outright**
- a write resolves from *words*; the model never supplies an `entity_id` (asked to
  "turn on the red LED" it otherwise invented `light.red_led` and had to be told off)
- ambiguity is a refusal with candidates, never a guess: a misheard name must not
  switch the wrong device
- `unavailable` entities are refused with a reason
- every call is logged with service, entity and prior state

**Cost.** A house question takes two model steps (look up → answer), measured
8.0–14.3 s on this box, against ~4 s for a plain question. Verified end to end:

```
"Turn on the red LED"      -> ha_do  -> light.esp32_a1s_red_led turn_on (was off)
"Is the patio light on?"   -> lookup -> switch.patio_light=off  -> "currently off"
"Turn the patio light on"  -> ha_do  -> switch.patio_light turn_on (was off)
"How warm is the office?"  -> lookup -> sensor.office_office_environment_temperature 81.446 °F
```

**Statements must never write.** Live failure: STT heard *"This is the first office
fan on"* — a remark — and the model called the write tool, turning the fan **on**.
The gate is now in `ha.py` (`looks_imperative`): writes are only offered when the
utterance is an instruction ("turn/switch/shut … on/off") and is neither a question
(opener word) nor a question mark. Verified on eight utterances, including that
exact failure as a must-not-write case. Reads are always available.

**The gate.** Offering two more tool schemas costs ~60–80 prompt tokens (~2 s) on
*every* turn, so HA tools are only offered when the transcript mentions the house
(`wants_ha_tools`: light, switch, garage, temperature, open/closed, …). A question
about antennas pays nothing for them.

**Privacy:** entity data stays on the LAN and is only exposed to a model — if the
answer lane is set to the cloud lane, that data goes with the question. Local lane
by default keeps it home.

### 8.7 Two escalations, not one — data vs reasoning

The escalation ladder was wrong in a way that only showed once Muse was answering.
There are two different escalations, and they go to different places:

| the question needs | destination | evidence |
|---|---|---|
| **the owner's own state** — notes, lists, tasks, project history, accounts | **Muse**, over the bus, announced later (30-60 s is *appropriate*) | *"what's on my list for tomorrow?"*, *"where did we leave the duplexer order?"* |
| **calculation or world knowledge** | **cloud lane**, in-turn (~1 s) | *"what ERP from 100 W into 6 dBd?"* → 400 W in 1.3 s; *"FSPL at 440 MHz over 10 km?"* → 105.3 dB |

Muse is not a bigger model. She is the only holder of the PDA's state, which is why
no provider can be substituted for her — and equally why she should never be asked to
do arithmetic. Before this split, *"what ERP do I get from 100 W into 6 dBd"* waited
**35 seconds** for an agent that had nothing to add, when the cloud lane answers it in
one.

**Trimming the tool schema has a floor.** The handoff description was cut to *"Ask
Muse (data) or the cloud model (reasoning)"* to save ~55 prompt tokens (6–7 s → 4.9 s
of decision time). It cost the behaviour: the 3B stopped calling the tool at all and
answered *"I need your own records to answer that"* instead — a dead end. The
description is now ~12 words longer, and the deferral phrases it produces
("*need your…*", "*your records*", "*can't see your calendar*") are matched as
**agent** hedges, so a model that explains the deferral instead of calling the tool
still reaches Muse. Current state: 291–304 prompt tokens, 3.4–5.9 s of decision,
both routes correct.

**An empty reply must not fail a turn.** Measured: given a tool schema, `qwen2.5:3b`
sometimes emits a tool call that Ollama discards — HTTP 200, empty content, and
`eval_count` of 33 proving it generated something. The bridge now retries once
**without tools** (which yields a plain answer, often a hedge that then escalates
correctly) and, if that is also empty, speaks *"Sorry, I couldn't work that one out
just now"* instead of failing the turn.

**How the decision is made.** The `handoff` tool gained a `needs` field
(`data` | `reasoning`), and the bridge overrides a `data` label when the question is
plainly computation — a quantity unit (dB, SWR, ERP, watts, impedance, wavelength,
MHz, feet…) or a how-many/how-much carrying a number. The 3B labelled the ERP question
`data` on the first try, which is exactly why the override exists rather than trusting
the model.

**Two failure modes, both measured, both now fixed:**

- **A mislabel in the data direction costs 35 s; in the reasoning direction it costs
  nothing.** If the cloud lane's reply is itself a refusal (*"I don't have access to
  your records"*) the turn falls back to the agent instead of speaking it. So the
  cascade is: fast lane first, agent only when the fast lane says it cannot know.
- **Never send a paraphrase to the reasoning lane.** Asked *"FSPL at 440 MHz over 10
  kilometres"*, the 3B restated it *without the distance* and the cloud correctly
  answered **85.32 dB for 1 km** instead of 105.3. Reasoning escalations now send the
  transcript verbatim — a paraphrase can drop a parameter, the user's words cannot —
  and the two are logged when they differ.

Hedge routing follows the same split: *"I need Muse"* is a data deferral and goes to
her; *"I don't have real-time information"* is a knowledge gap and goes to the cloud
lane first.

### 8.8 The clock is a host tool, not model knowledge

Design decision (owner): the model should **ask the host** for the time rather than be
handed it or bypassed. So `host_clock(kind)` is offered whenever the turn is temporal,
the bridge answers it from the OS in microseconds, and the model speaks the result.

```python
host_clock(kind: "time" | "date" | "both")
  -> {"spoken": "It's ten fifty-nine at night."}      # one field, nothing else
```

The spoken form is spelled out ("six fifty-four in the evening", "the twenty-first of
November, twenty twenty-six") because a synthesiser reads "6:54 PM" as two letters and
"21st" inconsistently.

**What this replaced, and why.** Before: the clock was injected into every prompt as
`[current local time: …]`, which cost ~14 tokens on every turn and made the model
append *"the current time is…"* to unrelated answers (observed three times). It also
did not stop hallucination — asked without context it once answered *"It is 3:47 PM,
the workshop clock reads this time because its mechanical hands rotate…"* at 18:48.
There is no clock injection any more; the tool is the only source.

**Two failure modes it exposed, both fixed:**

| symptom | cause | fix |
|---|---|---|
| asked *"what day of the week is it?"* the Dot said **"handoff needs data"** | the date pattern did not match that phrasing, so no clock tool was offered — and the model wrote tool syntax as **prose** instead of calling anything | the pattern now covers "day of the week" / "which day"; and any reply matching tool syntax (`handoff needs=…`, `host_clock`, `music_*`, `ha_*`) is never spoken — the turn is retried without tools, then apologises |
| the model answered *"It's twenty-six hundred twenty-one year November twenty-first, two-hundred-twenty…"* | the tool result carried `{spoken, digits, iso}` and the model regurgitated all three | return **one** field; the verifier gets the digits bridge-side, not in the prompt |
| *"how many days until Christmas"* looped: three steps, five clock calls, 20 s, no answer | the 3B would rather re-ask for the date than do date arithmetic | a repeated tool call now **escalates to the reasoning lane** on the first repeat (34 days, and the cloud is given the host date because it cannot call tools) |

**Cost, measured:** a time question is two model steps, 5.4 s; a date question 6.4 s; a
non-temporal question is unaffected (139 prompt tokens, 4.0 s). That is *slower* than
answering from the OS directly — this is the price of letting the model own the
decision, which was the point.

**And it corrected the verification gate.** Verification used to be skipped for any
turn that used a tool, on the assumption the numbers came from the tool.
*"How many days until Christmas"* disproved that: the date came from the tool, the
**84** did not (correct: 34). The rule is now precise — verify unless every number in
the answer appears in the tool output, rounding allowed (`81.4` from `81.446` is fine,
`84` from a date is not).

### 8.9 Streaming TTS — what was possible, and what was built

**First, the measurement that decided the design.** The obvious approach — serve a
growing HTTP response and let the controller play as bytes arrive — does not work
here. `stream_probe.py` served a WAV whose body was written slowly (1.5 s of audio,
a 4 s gap, 1.5 s more):

```
0.01  GET arrives, first 1.5 s sent
4.08  second 1.5 s sent, EOF
10:34:14.868  "Streaming 288000 bytes (70 periods)"   ← 4.2 s after the GET
```

The controller **buffers the whole fetch, decodes it, then streams to the device.**
Nothing is heard early, so a growing response buys nothing at all.

**The only shape that works is one announcement per sentence**, and its cost is
small: a 1.5 s clip announced twice takes 3.46 s, i.e. **0.23 s of overhead per
announcement**. So the bridge streams tokens from the model (SSE), cuts at sentence
boundaries, hands **sentence 1 back as the turn's reply immediately** (so the normal
TTS_END path plays it), and announces the remaining sentences as they land, retrying
while the controller is busy with the first.

Measured, local lane:

| reply | sentence 1 handed over | full text finished | saved |
|---|---|---|---|
| 2 sentences (folded dipole) | **6.11 s** | 11.26 s | **5.15 s** |
| 1 sentence (what is a balun) | 3.13 s | 3.29 s | 0 — identical to before |

That second row is why this is safe to leave on by default: a one-sentence reply
produces exactly one announcement-worth of work, so nothing is added.

**A design mistake caught in the first test, worth recording.** The first version
collected the whole stream and *then* returned the first sentence — 11.81 s instead of
6.27 s, i.e. all the gain thrown away at the last step. Returning early is the entire
feature.

**Constraints, deliberately:**

- **Only when no tools are offered.** A tool call can arrive mid-stream, and abandoning
  a half-spoken answer to run a lookup is worse than waiting for the whole reply.
- **Remaining sentences are announcements**, so the controller records the turn as a
  reply of sentence 1 only. Its per-turn metrics under-report a long answer; ours do
  not.
- The sentence splitter requires ≥4 words before cutting, so "e.g." and abbreviations
  do not become announcements of one word.
- **A barge-in now cancels the queued sentences.** Found in review: `start=False`
  cancelled the turn's own task but not the sentence pump or its announcement queue, so
  sentence 2 of a streamed reply could be spoken over the user. The same happens when a
  new turn starts. Both now cancel any leftover stream tasks first; `stream_test.py`
  asserts that cancelling mid-reply stops the announcements.

### 8.10 What the cloud lane is *not* for

A cloud model has no clock and no weather either, so "what time is it" and
"what's the weather" are **data** problems, not model problems — the right fix is
a local tool (`clock()`, `weather()`) inside the local lane, not an escalation.
Handoff is for *reasoning the local model cannot do*, not for facts it does not
have. Keeping those two mechanisms separate is what stops the design turning into
a pile of per-question special cases.

---

### 8.11 Conversation memory — the bridge owns it

Every turn was stateless: Ollama keeps nothing, and the bus has no session concept,
so *"and the other one?"* had no referent. The bridge now keeps the last few
exchanges **per device** and hands them out in both directions.

**Why the bridge owns it, rather than the answering agent:**

1. The local lane needs it regardless — a store here is not optional work for the
   hand-off case, it is the same store.
2. A **self-contained envelope needs no server-side session.** Store-and-forward has
   no sessions, a responder restarts and redeploys, and this project already spent an
   evening on what "no consumer" looks like. A responder that must remember is a
   responder that can be wrong after a restart.
3. The device's owner should own its conversation. A responder may cache deeper
   state keyed by conversation id as an *optimisation*; correctness never depends
   on it.

**How it behaves:** history is kept for `--memory-ttl-s` (300 s) and at most
`--memory-turns` exchanges (3). It is injected into the prompt only when the
utterance needs a referent — `--memory auto` matches referential words ("it", "that",
"them", "the other", "again") and follow-up openers ("and", "what about"), not mere
brevity: an earlier version treated any short question as a follow-up and put history
in front of "what is a balun?" for nothing. `--memory always` / `off` override it.

Measured: a follow-up costs ~1.5 s more than a self-contained question (prompt grows
from 133 to 165 tokens for one exchange), and it resolves — asked *"which needs more
room, a 40 m dipole or a 20 m?"* then *"and the other one?"*, the answer was about
the 20 m one.

**In the envelope:** the hand-off carries `conversation` (the controller's own id for
this exchange) and `request.context` — up to three exchanges, each trimmed to 400
characters. Announced answers are written back into the history too, so a follow-up
after *"Muse says: ..."* still has a referent.

**Verified end to end 2026-11-21 20:45**, with a question that is meaningless in
isolation — *"and the other one?"* — sent as an envelope carrying two context
entries. Muse's worker answered *"The 20-meter dipole is the shorter one, so it needs
less room than the 40-meter one"*, matched on the envelope correlation, and the Dot
spoke it. A responder that reads the envelope needs no memory at all; a responder
that keeps its own is an optimisation, and now demonstrably not a requirement.

### 8.12 Numeric verification — the answer the assistant could get wrong

The local 3B is not usually wrong, but when it is wrong it is **confident**: asked
what a 3:1 SWR reflects it said *"eight percent reflected, about half a dB loss"*,
and asked the same thing later, *"94% transmitted, 6% reflected"*. Neither is a hedge,
so the hedge trigger never fires. This is the one remaining way the assistant can
teach you something false about your own antenna.

**The rule:** if the local answer contains a number and the turn was factual, the
cloud lane re-answers the question cold and its answer is spoken. Measured cost
~0.7–2.1 s, and only on turns that produce numbers.

**It re-asks; it does not ask "is this right?"** Showing the model the candidate made
it anchor: asked to check *"25% reflected, about 6 dB"* it "corrected" it to
**11.1%** — which is the answer for SWR **2:1**. Re-asking cold, with room to show its
working and an extractable final line, is the only version that measured right:

```
system   You are a radio engineering expert. Keep any working to at most two short
         lines, then end with one single line exactly of the form:
         ANSWER: <one short spoken sentence containing every number asked for>
user     With an SWR of 3 to 1, what percentage of my power is reflected and how many dB do I lose?
reply    … Γ = (3-1)/(3+1) = 0.5 …
         ANSWER: You lose 1.25 dB with 25 percent of power reflected.     ← spoken
```

Two traps found while building it, both already known in other forms:

- **A word cap destroys the arithmetic.** *"At most 25 spoken words"* made the
  verifier answer 11.1% (2:1) — the same failure as the earlier "under 25 words"
  prompt. It needs room to compute and a short line to speak, not a short answer.
- **Too small a token budget cut the `ANSWER:` line off** and the reply became an
  unparseable wall of working. If no `ANSWER:` line comes back, the local answer is
  kept rather than speaking the working.

**When it must NOT fire** (`Bridge.should_verify`, pure and tested in
`verify_test.py`, 10 cases):

| skip | because |
|---|---|
| numbers came from a tool | they are HA's reading, not the model's arithmetic |
| the question asks for the time/date | those numbers are the injected clock |
| the question is about local context ("you", "Zion", "Muse", "this machine") | a cold cloud answer has no context: asked *"tell me about Zion"* it replaced a correct answer with **"Zion National Park is 229 square miles"** |
| the answer repeats the injected clock | it is using context the cloud cannot see |
| the answer has no numbers | nothing to check |

Spelled-out numbers count: the original failure — *"eight percent… half a dB"* — has
no digits in it at all.

**A hand-off can pre-empt verification, and that is fine.** Asked live *"what ERP do I
get from 100 watts into an antenna with 6 dB of gain?"*, the 3B did not attempt the
arithmetic at all — it called `handoff`, Muse answered *"six dB is about four times the
power, so 100 watts gives you roughly 400 watts ERP"* (35 s, spoken as an
announcement), and the verifier never ran. A model that knows it cannot compute is a
better outcome than one that tries and needs checking; the two mechanisms do not need
to be ordered, because either path ends with a checked answer.

**What it does not fix — measured 2026-11-21.** Six questions through the live path:

| question | local (Zion) | verified (cloud) | truth |
|---|---|---|---|
| ERP for 100 W into 6 dBd | 70.7 W ❌ | **400 W** ✅ | 398 W |
| 100 W through 1.5 dB of line | 87.5 W ❌ | **70.8 W** ✅ | 70.8 W |
| SWR 3:1 reflected / loss | 25%, "9 dB" ❌ | **1.25 dB** ✅ | 25%, 1.25 dB |
| FSPL at 440 MHz / 10 km | 135.8 dB ❌ | **105.3 dB** ✅ | 105.3 dB |
| double the distance | 6 dB ✅ | 6 dB ✅ | 6.02 dB |
| half-wave dipole for 14.2 MHz | 38.5 ft ❌ | **16.48 ft** ❌ | 33 ft |

Four corrected, one a no-op, **and one where the cloud was also wrong** — it computed a
quarter wave (234/14.2 = 16.5 ft) instead of a half wave. So verification reduces
numeric error; it does not eliminate it, and it can replace one wrong number with
another. What it does guarantee is that the answer came from a model with room to show
its working, and every correction is logged with both answers so a wrong result is
auditable after the fact.

The `verified (agreed)` / `verified (corrected)` label compares **numbers, not
wording**, so a rephrase is not recorded as a correction.

## 9. Observability and failure behaviour

**Bridge log, per turn** (`journalctl --user -u voice-bridge`): wake conversation id, speech start/end
with Silero statistics, captured audio length, transcript, LLM latency with prompt
and reply token counts plus measured tok/s, TTS latency and speech duration, the
serve URL, and the total.

**Controller log, per turn** (`docker logs echomuse-controller`): `[TURN]` line
with `outcome`, `total`, `first_frame`, `vad_start`, `vad_end`, audio frames,
`stt`, `tts_url`, `tts_fetch`, `tts_bytes`, `playback`, and stage segments
(`ha_think`, `tts_gen`, `fetch`). This is the authoritative external view — it is
written by the controller, so it cannot be flattered by our own logging.

**Outcomes observed and what they mean**

| Control outcome | Meaning | Who caused it |
|---|---|---|
| `ok` | reply fetched and played | normal |
| `no_speech` | turn ended with no usable audio | our VAD never opened, or the device's gate closed first |
| `no_tts` | the turn ended without a reply URL | we sent `ERROR` (e.g. missing Python package) |
| `tts_error` | controller could not fetch our media URL | media server dead or not listening (happened twice) |
| `aborted` | superseded/barge-in | `VoiceAssistantRequest(start=False)`; the bridge cancels its turn |

**Rules the bridge follows so failures are visible rather than silent:** always
send `RUN_END` (even after `ERROR`); never exit while a reply may still be being
fetched (wait for `AnnounceFinished`, with a grace scaled to the audio length);
and speak a short fallback rather than nothing.

---

## 10. Operations manual

### 10.1 Prerequisites (one-time)

1. A Dot is connected to the controller. Check:
   `ss -ltn | grep 1600` should show its port, and
   `docker logs --since 5m echomuse-controller | grep "port 16001 up"`.
2. **Home Assistant must not hold the seat.** Disable the device's ESPHome entry
   in HA (Settings → Devices & Services → ESPHome → entry → Disable). Verify:
   `ss -tn | grep :16001` shows nothing, and the controller logs `HA disconnected`.
3. **Credentials live in files, and the flags take paths.** Nothing secret appears on
   the command line, so `ps aux` shows filenames, not tokens:

   | credential | file (mode 600) | flag (a path) | if absent |
   |---|---|---|---|
   | Home Assistant long-lived token | `~/.config/voice-bridge/ha-token` | `--ha-token` | `ha_lookup`/`ha_do` refuse with a clear error |
   | Music Assistant JWT | `~/.config/voice-bridge/ma-token` | `--ma-token` | `music_*` tools refuse |
   | agent-bus token | `~/.agent-bus/tokens/<agent>` | `--bus-token` | the hand-off lane refuses to submit |

   The only flag that would put a literal secret in the process list is `--cloud-key`,
   and it is unused because OmniRoute runs with `REQUIRE_API_KEY=false`. If that ever
   changes, put it in a systemd `EnvironmentFile=` rather than `ExecStart`.

### 10.2 Run

```bash
cd /srv/WorkSpace/voice-bridge
.venv/bin/python bridge.py --llm --tts --model qwen2.5:3b
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--port` | satellite port (16001 for Satellite-1) |
| `--stt` / `--llm` / `--tts` | enable stages independently; no `--tts` = a chime answers |
| `--model` | Ollama model (`qwen2.5:3b`, `qwen2.5:7b`) |
| `--whisper-model` | `base.en` (default), `small.en`, … |
| `--piper-voice` | path to a Piper `.onnx` |
| `--silence-ms` / `--min-speech-ms` | endpointing behaviour |
| `--media-port` / `--media-host` | where reply WAVs are served (must be reachable by the controller) |
| `--answer-lane` | which sync lane answers a turn (default `local`); an async lane is refused |
| `--cloud-model` / `--cloud-url` | enable the `cloud` sync lane (OmniRoute; `auto/claude-sonnet`, `gemini/…`, `openai/…`) |
| `--muse` / `--bus-url` / `--bus-token` / `--muse-agent` | the async hand-off lane (Muse over the agent bus) |
| `--handoff {off,ask,auto}` | the escalation policy: never / when asked or the model asks / also on a hedge |
| `--handoff-ttl-s` | how long a hand-off stays answerable (default 900) |
| `--verify {off,numbers,always}` | cloud-check a local answer that contains numbers |
| `--stream-tts` / `--no-stream-tts` | speak sentence 1 while the rest is generating (default on) |
| `--memory {off,auto,always}` · `--memory-turns` · `--memory-ttl-s` | conversation history: only for follow-ups (default), always, or never |
| `--attribution {off,handoff,always}` · `--local-name` · `--muse-name` | say who answered ("Muse says: …") and the spoken names |
| `--ha` / `--no-ha` · `--ha-url` · `--ha-token` | Home Assistant tools and credentials |
| `--music` / `--no-music` · `--ma-url` · `--ma-token` · `--music-target` | Music Assistant tools and the default speaker |
| `--bus-type` · `--device-name` | the bus message type Muse's hook filters on, and the device named in it |
| `--lane-timeout` | per-lane HTTP timeout (default 25 s) |
| `--once` · `--seconds` | exit after one turn (waits for playback), or a bounded run (0 = forever) |
| `--help` | the full list — there are ~50 flags, most with sane defaults |

**Staged proving:** `--stt --tts` (no `--llm`) makes the Dot repeat what it heard —
that isolates mic + whisper + Piper from the model. No `--tts` answers with a
two-tone chime, which isolates the transport alone.

### 10.3 Supervision (systemd)

The bridge is a systemd **user** unit (`~/.config/systemd/user/voice-bridge.service`,
template in the repo at `deploy/voice-bridge.service`). User-level because the agent
bus and the controller plumbing already run that way here, and `Linger=yes` is set so
it starts at boot without a login session.

```bash
systemctl --user enable --now voice-bridge.service
systemctl --user restart voice-bridge.service
journalctl --user -u voice-bridge -f          # per-turn log, live
```

`PYTHONUNBUFFERED=1` is not cosmetic: stdout to a file is block-buffered and a
perfectly healthy run reads as a frozen log — that trap cost a session on this bench.
`Restart=on-failure`, `RestartSec=10`, `StartLimitBurst=6` means a missing controller
retries instead of hot-looping, and gives up until the next boot after six tries.

### 10.4 Restart / stop by hand

The bridge holds the seat; the controller's listener stays up regardless. Kill the
process (`kill <pid>`) and restart — the controller accepts a new client
immediately. There is no state to preserve.

### 10.5 Health checks

```bash
ss -tn | grep :16001                       # we are connected to the controller
curl -s localhost:8700/ | head             # media server is up
curl -s localhost:11434/api/ps             # which models are resident
docker logs --since 5m echomuse-controller | grep -E "TURN\]|disconnected"
```

---

## 11. Reproduce from scratch

```bash
# 1. project + environment
mkdir -p /srv/WorkSpace/voice-bridge && cd /srv/WorkSpace/voice-bridge
python3 -m venv .venv
.venv/bin/pip install aioesphomeapi aiohttp faster-whisper piper-tts

# 2. models
docker exec ollama-p700 ollama pull qwen2.5:3b        # or via the API
# whisper base.en downloads itself on first use (~74 MB)

# 3. a Piper voice (63 MB)
mkdir -p voices
for f in en_US-hfc_female-medium.onnx en_US-hfc_female-medium.onnx.json; do
  curl -sLO --output-dir voices \
   "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/$f"
done

# 4. free the seat (HA) and run
.venv/bin/python bridge.py --llm --tts --model qwen2.5:3b
```

Prerequisites that are *not* created by this project: the controller container
(§3.2), a connected Dot, and HA being told to stop dialling that device.

---

## 12. Traps encountered (symptom → cause → fix)

Each of these cost a cycle, and each looks like something else.

| Symptom | Cause | Fix |
|---|---|---|
| Audio silently dropped, library logging "coroutine was expected, got None" per chunk | `handle_audio` was a plain function; aioesphomeapi 46 wraps the result in a task | make every `handle_*` an `async def` |
| Turn ran 20 s, then `outcome=no_tts` | Dot is in *private listening*: no `0x04` end sentinel, so the backend must endpoint | run a VAD (Silero); send `STT_VAD_END` when speech ends |
| VAD started on the first noisy frame and never ended | energy threshold with a floor that only adapted on non-speech frames — and the room's floor is ~270 counts | use Silero, not a threshold |
| VAD emitted `end` on every subsequent window | state machine never left the speaking state | reset `speaking`/counters when closing |
| `outcome=no_tts` although the tone played | reply delivered as `VoiceAssistantAnnounceRequest` instead of `TTS_END{url}` | put the URL in `TTS_END`; announcements are out-of-band only |
| `outcome=tts_error`, "Cannot connect to host …:8700" | bridge exited on `RUN_END` and killed its media server before the controller fetched | wait for `AnnounceFinished`, grace scaled to the audio duration |
| Second connection refused with `DisconnectResponse` | controller enforces a single claimant | disable the device in HA first |
| `ModuleNotFoundError: aiohttp` mid-turn | not a transitive dependency of aioesphomeapi | `pip install aiohttp` |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | faster-whisper 1.2.1 vs PyAV 19 file decoding | pass raw PCM/numpy, never a path |
| First reply was 11.1 s of speech | no length instruction | prompt: "under 25 words", cap `num_predict` |
| 7 s answers from a 7B model | 4.4 tok/s at 7B on this CPU | use 3B, or a GPU |
| Model answers "I don't know" to simple factual questions | a 3B has no clock, weather or local knowledge | tools for data; handoff for reasoning (§8) |
| Correct model gives confidently wrong numbers | a hard word limit ("under 25 words") made it skip the arithmetic | allow the working, cap the answer; add "be precise; do not guess" (§7.3.1) |
| A condensed answer loses the numbers | a small model was asked to compress a big model's reply | never compress with the small model; prompt the big one correctly instead (§7.3.1) |
| Voice assistant states a wrong time, with an explanation | no lane has a clock; the model invents one | append the real time to the user turn (§7.3.2) |
| A numeric answer changes between identical runs | `temperature=0.4` sampling | lanes default to 0.0 (§7.3.1) |
| Assistant says "I'm Muse" in a turn, then "I've asked Muse" about a hand-off | the local prompt named the model Muse while Muse is the remote agent | Zion = local, Muse = outside; `spoken_name` per lane (§8.3, naming note at the top) |
| An answered hand-off is never spoken | inbox parse (`{"messages": []}` fell through `or data`), 30 s await on 35 s audio, retries stacking ducking | fixed in three places; see the limits table above (§8.3) |
| Replies arrive but nothing is announced | the responder did not echo `correlation_id`/`in_reply_to`, so no id matched | match on subject + time as a fallback; ask responders to echo ids (§8.3) |
| A remark switches a device on | a statement was executed as a command | gate writes on imperative phrasing (`looks_imperative`, §8.5) |
| The hand-off asked the wrong question: "ask **news** what its icon is" | whisper decoded the agent's name as a common word | put proper nouns in the ASR prompt (`ASR_PROMPT`), same fix as "in petals" → "impedance" (§7.2) |
| The Dot said "handoff needs data" | the model wrote tool-call syntax as text, and the phrasing "day of the week" matched no date pattern so no tool was offered | pattern extended; tool syntax in an answer is detected and never spoken (§8.8) |
| First follow-up that also needed a hand-off failed with `KeyError('text')` | chat history is `{"role","content"}`, the envelope wants `{"role","text"}` — two shapes, one confusion | `_handoff_context()` maps between them; `memory_test.py` asserts both shapes (memory_test.py was written after the bug, not before) |
| Every envelope answer is dropped as "past its TTL" | `is_expired()` was applied to the *answer*, which carries no `ttl_s` and so fails safe as expired | age the pending entry against the request's TTL instead (§8.3.1) |
| New instance dies with `OSError: [Errno 98] Address already in use` | the old bridge still holds media port 8700 — and a `kill $(ps ... awk ...)` can match *its own shell's command line* (which contains `bridge.py`) and kill the shell instead of the bridge | kill by explicit PID; verify with `ss -ltnp \| grep 8700` before restarting |

---

## 13. Known limits and what is not built

Built and proven since the earlier draft of this list: conversation memory (§8.11),
the hand-off trigger and the Muse lane (§8.3), the systemd unit (§10.3), numeric
verification (§8.12), the clock tool (§8.8), Music Assistant (§8.5) and Home
Assistant tools (§8.6).

What genuinely remains:

- **Streaming TTS is sentence-level, not true streaming** (§8.9). The controller
  buffers its fetch, so the only shape is one announcement per sentence: sentence 1
  goes out as the turn reply immediately and the rest follow as announcements. That
  saved 5.15 s on a two-sentence answer and costs nothing on a one-sentence one, but
  a long answer still has a 0.23 s seam between sentences, and the controller's turn
  metrics only count sentence 1. Genuine mid-sentence streaming needs the controller
  to play a growing body, which it demonstrably does not.
- **One Dot.** Satellite-2 has been offline since 2026-11-06, so "loudest wins"
  arbitration and the two-device split have never been exercised.
- **The device's data plane is fragile under long announcements.** One 40 s clip
  (3.7 MB) made the Dot stop reporting `playback_stats`, then drop its plane with
  `keepalive ping timeout`. Trimming to ~30 words hides it; the cause is unproven.
- **Verification cannot fix a wrong cloud answer.** Four of six numeric questions were
  corrected; the dipole-length one got *worse* (cloud computed a quarter wave), and a
  date-difference answer came back 33 days instead of 34. Escalation improves the
  odds, it does not guarantee truth.
- **The clock is now two model steps** (5.4 s for the time, 6.4 s for the date) —
  slower than answering from the OS, by design: the model owns the decision.
- **`envelope.py` has no v2 path.** Additive fields are proven safe and discoverable
  (§8.3.1); nothing defines what a responder does with a breaking change.
- **Muse's lane depends on her side polling.** The bridge queues, persists and
  announces; if her hook is down, nothing is delivered and the turn ends with an ack.
- **Wake word is `hey_jarvis`** while the assistant is Zion and the agent is Muse. A
  "Hey Muse" model is trainable with `oww_forge`; cosmetic but increasingly odd.
- **Transcripts live in journald and in Muse's bus history.** No retention policy is
  defined beyond "admin-only".
- **The device's own outbound traffic was not audited.** The host-side path is proven
  local; proving the Dot never talks to Amazon would mean inspecting conntrack or the
  router.

---

## 14. The agent bus: hygiene, not transport

**Conclusion (2026-11-21, after Muse reviewed the options): do not swap the
transport.** The reasoning is structural and it survives scrutiny:

- **MCP is one-way per connection** (client calls, server answers) and Muse accepts
  no inbound connections — she runs in a hosted cloud VM. So MCP can never carry
  traffic *toward* her. It is the fast lane only for the other direction: she calls a
  tool on the ThinkStation and gets a typed answer in the same call. Worth adding
  later; parked for now.
- **The bus is not the slow part.** The dispatcher is event driven and routes on
  arrival. The measured ~2-minute PONG was OpenCode's own poll cycle and processing,
  and the slow part on Muse's side was having no consumer at all — which the hook
  fixed. The 31-second voice hand-off measured here was her hook interval plus her
  answering time.
- **A relay, WebSocket or MQTT subscription would only shave the poll** — a few
  seconds — while her answer time (tens of seconds) stays. It would cost a daemon on
  her side for almost nothing.
- **No protocol makes both directions fast**, because the limit is that one direction
  requires *her* to wake up and answer.

So the bus stays the durable backbone: messages survive restarts, long jobs belong
there, and it is the only thing that can carry traffic toward her. The table below is
therefore a list of **hygiene**, not performance work — every row is cheap, and every
row is a bug this project actually hit.

### 14.1 What a production version needs

The bus is the weakest component in the chain, and this project leaned on it hard
enough to expose why. Observed failures during one evening, each with a symptom a
user would notice:

| gap | what happened | what it needs |
|---|---|---|
| **delivery is per-agent and one-way** | `bus-dispatch.py` routes only `opencode` and `hermes`; `muse` had to run its own hook because the host cannot reach her cloud runtime | a delivery contract per agent (pull or push), declared by the agent itself |
| **no correlation discipline** | Muse's replies carried a new `correlation_id` and no `in_reply_to`; a correct answer was silently unmatched for minutes | the server should carry a `re`-field through, or require it on reply types |
| **placeholders look like answers** | `bus-dispatch` substitutes `(empty reply)` when an agent returns nothing; the bridge spoke it aloud | a distinct error status, never a body that can be mistaken for content |
| **no typed payloads** | `voice-handoff` is a naming convention; nothing validates that a voice turn has a device, a question and a reply-shape | schema per type, validated on send |
| **unbounded backlog** | 6 undelivered messages accumulated, including a Nov-13 result and a canary ping; a late consumer inherits history | TTL per message, and a "newer than subscribe" contract |
| **peek is the only safe read** | the bridge must peek (never consume) to avoid stealing other readers' mail, so messages stay `delivered=0` forever and the accounting is meaningless | explicit acknowledgement separate from reading |
| **no latency or priority class** | a voice hand-off (needs ~60 s) and a background task share one queue | priority/queue classes, or at least a deadline field |
| **no read audit** | nothing records who read what and when, so "did Muse see it?" was answered by querying the DB | delivery receipts |

None of that is a criticism of store-and-forward — it is the right shape for this
direction of traffic, and it is the only shape the topology allows. It is a list of
envelope-level guarantees that a durable channel needs before anything depends on it
being *correct*, as opposed to fast.

The cheapest three, and the ones that bit this project directly:

| fix | the bug it prevents |
|---|---|
| responders echo `in_reply_to` (or the request's correlation id) | a correct answer silently unmatched for minutes (tonight) |
| a TTL / expiry per message | a six-deep backlog of stale results inherited by a late consumer |
| a distinct error status instead of a body like `(empty reply)` | a placeholder spoken aloud as if it were the answer (tonight) |

## 15. Roadmap

**Everything the earlier draft listed as "next" is built and proven**: systemd
supervision (§10.3), conversation memory (§8.11), the hand-off trigger (§8.7), the
Muse lane with the envelope (§8.3), sentence-level streaming TTS (§8.9), the clock
tool (§8.8), Home Assistant and Music Assistant tools (§8.6, §8.5), numeric
verification (§8.12) and the OmniRoute combo (§8.2).

What is genuinely left, roughly in order of value:

1. **Two satellites.** Satellite-2 has been offline since 2026-11-06, so "loudest
   wins" arbitration and the one-device-per-backend split are untested code paths,
   not working features.
2. **The device's data plane under long audio.** A 40 s clip (3.7 MB) stopped
   `playback_stats` and then dropped the plane (`keepalive ping timeout`). Trimming
   to ~30 words hides it; the cause is unknown and a longer answer would hit it.
3. **A breaking-change path for the envelope.** Additive fields are proven safe and
   self-describing; `v: 2` is undefined. A responder that meets one has no rule.
4. **Radio-state tools.** MQTT (`:1883`, `p25-gateway/*`) or the DVSwitch dashboard
   API: "who is linked", "is the parrot up", "what is the reflector doing". This is
   the local data that would make it a *station* assistant rather than a general one.
5. **"Hey Muse" wake word** via `oww_forge`, since the assistant answers to
   `hey_jarvis` while being Zion and handing off to Muse.
6. **Transcript retention.** Journald and Muse's bus history both keep everything;
   no policy exists.
7. **True mid-sentence streaming** — blocked on the controller playing a growing
   body, which it demonstrably does not (§8.9). Not worth pursuing from this side.
8. **Hardware, if ever:** any CUDA-class GPU removes the model/quality trade-off
   (7B at 30–40 tok/s instead of 4.4) and would make Kokoro's naturalness free.

---

## 15.1 Filed, not decided: run Music Assistant outside Home Assistant

**Idea (owner, 2026-11-22):** Music Assistant is currently an HA add-on; the bridge
already talks to its WebSocket API rather than to HA, so MA could run on this
machine's Docker stack instead, with HA left to be HA.

**Feasible: yes, trivially.** `ghcr.io/music-assistant/server:stable` is multi-arch
(amd64/arm64) — the same image the add-on wraps — and the bridge's coupling is one
flag, `--ma-url ws://<host>:8095/ws`. This host has 16 cores, 47 GB RAM free, 7.4 TB
disk, and nothing listening on 8095.

**The fact that actually decides it:** the voice assistant's *music* path depends on
the HA box's uptime. HA core restarts do not restart add-ons, but an HA OS reboot or
OS update takes MA with it — so "the Dots don't depend on HA" is true for questions
and false for music today.

| case for moving | case against |
|---|---|
| decouples the assistant from HA maintenance windows | you own updates, backups and the data dir |
| MA updates on your schedule, not the add-on store's | HA's snapshots currently cover it for free |
| better hardware (this box vs the HA OS box) | provider re-auth risk if data-dir migration goes wrong |
| MA is infrastructure used by non-HA consumers (this bridge, Sendspin players) | one more service to keep alive; "let HA be HA" is real |

**If it is done later**, the order that keeps it reversible: confirm the installed MA
version offers a backup export (preferred over copying files), stand it up on a
parallel port (`8096:8095`) beside the add-on, verify providers and players import,
then stop the add-on, swap to 8095, repoint HA's integration, the bridge's
`--ma-url` and any Sendspin pairing.

**Recommendation, for what it is worth:** leave it on HA unless the HA box is short
of capacity or its maintenance windows have actually interrupted music. The one
asymmetry that would change the answer is if the assistant's music path needs to stay
up through HA work — that is the only cost worth paying for.

---

## 16. References

**Read while building (authoritative):**

- `~/EchoMuse/docs/device-controller-interface.md` — the device protocol;
  `/data` frame codes, capabilities, link auth, private listening
- `~/EchoMuse/docs/listening.md` — private-listening sessions
- `~/EchoMuse/controller/em_esphome.py` — the satellite server (in the image at
  `/app/em_esphome.py`): the exact HA contract, decision points and comments
- `~/EchoMuse/controller/em_earlytts.py` — why `tts_start_streaming` exists and
  when early playback is dangerous (a backend slower than ~4 tok/s leaves the
  buffer dry — the 7B sits exactly there)
- `~/EchoMuse/controller/esphome/satellite_server.py` — handshake, API 1.10

**Local state:**

- controller data `/srv/compose/echomuse/data/echomuse.db`, `tls/`
- Ollama models `/srv/data/ollama`
- `~/docs/Zion/agent-bus.md`, `~/docs/Zion/home-assistant.md` (context; not used
  by this design)

**Versions at time of writing:** controller v2.26.1 · device fw v2.18.1 /
emos-v0.10 · HA 2026.11.0 (previously bound, now released) · Ollama 0.24.0 ·
aioesphomeapi 46.30.1 · faster-whisper 1.2.1 · piper-tts 1.8.0 · Python 3.13.5.
