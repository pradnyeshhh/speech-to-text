# voice_to_text

Speech to text, two ways:

- **The Mac app** (`src/`): hold a key, speak, and the text is pasted wherever you're typing. It
  runs Whisper on the Mac's own GPU with MLX.
- **The server** (`server/`): the same idea for any computer with a browser. A Linux server
  transcribes on its CPU with NVIDIA's Parakeet, a web page shows the text live as you talk,
  every transcript is saved, Claude writes a summary, and a meeting mode records your
  microphone and a meeting's browser tab as two speakers.

## The Mac app

Needs an Apple Silicon Mac, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run voice_to_text
```

Hold **Right Option**, speak, and let go: the text is pasted where the cursor is. The app
transcribes while you talk, cutting at pauses, so after you let go only the last part is left
(about half a second on an M2).

| Option | |
|---|---|
| `--key right_option \| right_command \| fn` | the hold-to-talk key (default `right_option`) |
| `--device NAME` | the microphone (default: the built-in one) |
| `--model large-v3-turbo-q4 \| large-v3-turbo \| distil-large-v3` | the Whisper model (default `large-v3-turbo-q4`) |
| `--file PATH` | transcribe an audio file, print the text and exit |

The first run asks macOS for **Microphone**, **Input Monitoring** and **Accessibility** for your
terminal app: turn them on in System Settings → Privacy & Security, then reopen the terminal.

Tests: `uv run pytest`. Model and latency measurements: [`BENCHMARKS.md`](BENCHMARKS.md), made with
the scripts in `bench/` (`bench/librispeech.py` fetches the test audio into `bench/data/`).

## The server

### What it does

- **Live dictation in the browser.** Press the button and talk: grey text follows you, and each
  sentence settles when you pause. The text is usually ready within a second of stopping.
- **Meetings.** In Chrome or Edge, share the meeting's tab with its audio. Your microphone and the
  tab are transcribed separately, as "You" and "Meeting" turns with timestamps; speaker echo
  picked up by the microphone is removed.
- **Uploads.** Any common audio file, from the page or `POST /transcribe`.
- **History and summaries.** Every transcript is saved in SQLite, and Claude Haiku 5.5 writes a
  summary with key points, decisions and action items.

```
browser (mic, meeting tab) ── SSH tunnel ──▶ Unix socket (root only)
                                               │
                         voice-to-text service │ aiohttp: /  /live  /transcribe  /api
                                               ├─▶ live session: Silero VAD + pause cutter, per channel
                                               ├─▶ model thread: Parakeet-TDT-0.6B-v2 (int8 ONNX)
                                               ├─▶ SQLite: data/transcripts.db
                                               └─▶ summarizer ──▶ Anthropic API (Claude Haiku 5.5)
```

Speech recognition runs on the server's CPU; the only thing that leaves it is a finished
transcript, sent to Anthropic for its summary.

### Set it up

On a Linux server with Python 3.12 and uv:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r server/requirements.txt
```

Run it by hand to try it (it downloads the model the first time):

```sh
.venv/bin/python server/server.py --model parakeet-tdt-0.6b-v2 --pad 0.4 --port 8095
```

Or install it as a service that starts at boot. It listens on a Unix socket that only root can
open, so other accounts on the server can't reach it:

```sh
install -m 644 server/voice-to-text.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now voice-to-text
journalctl -u voice-to-text -f
```

The unit expects the project at `/root/voice_to_text`; edit `WorkingDirectory` and `ExecStart` if
yours is elsewhere.

For summaries, save an Anthropic API key (it's asked for without being shown, kept in
`/etc/voice-to-text/env`, readable by root only):

```sh
bash server/set-api-key.sh
```

### Use it from your computer

```sh
ssh -N -L 8095:/run/voice-to-text/voice.sock root@your-server
```

Leave that running and open <http://127.0.0.1:8095>. Browsers allow the microphone there because
it's a local address.

### Options

| Option | |
|---|---|
| `--model` | `parakeet-tdt-0.6b-v2` (English, used), `parakeet-tdt-0.6b-v3` (25 European languages), or any faster-whisper model such as `large-v3-turbo` or `small` |
| `--pad SECONDS` | audio kept around the speech in each part (0.4 measured best for Parakeet v2) |
| `--socket PATH` / `--port N` | a root-only Unix socket, and/or a TCP port on 127.0.0.1 |
| `--interim-every`, `--min-part`, `--max-part` | how the live text is cut and how often the grey text updates |
| `--windows stt \| fine`, `--full-window` | Whisper only: the shortened encoder windows |
| `--public-port N` | a second port that only accepts Google sign-ins through Cloudflare Access; see `server/auth.py` and `server/set-google-access.sh` |

### API

| | |
|---|---|
| `GET /` | the page |
| `WS /live` | live audio in, text out (protocol in `LiveSession`'s docstring in `server/server.py`) |
| `POST /transcribe` | body: an audio file; add `?mode=meeting` for a meeting summary |
| `GET /api/transcripts`, `GET` / `DELETE /api/transcripts/<id>` | the saved transcripts and summaries |
| `GET /health` | the model, and whether summaries are on |

```sh
curl --unix-socket /run/voice-to-text/voice.sock --data-binary @memo.m4a http://x/transcribe
```

### How the model was chosen

On a 2-core server, 50 clean and 50 harder LibriSpeech recordings:

| Model | Word errors, clean / hard | Median wait |
|---|---|---|
| Whisper large-v3-turbo, shortened windows | 1.74% / 3.71% | 5.1 s |
| Whisper small | 2.73% / 7.07% | 2.8 s |
| Parakeet v3 | 2.73% / 6.84% | 0.63 s |
| **Parakeet v2, 0.4 s padding** | **1.36% / 4.06%** | **0.70 s** |

Every experiment, including the ones that didn't work, is in
[`server/EXPERIMENTS.md`](server/EXPERIMENTS.md); the scripts and raw results are in
[`server/bench/`](server/bench/).

### Limits

- English only with Parakeet v2: Hindi or Hinglish words are dropped.
- Meetings must run in a browser tab; "Meeting" is everyone else together.
- One model thread: on a small server, two people recording at once share it.

## Layout

```
src/voice_to_text/   the Mac app
tests/               the Mac app's tests
bench/               the Mac app's benchmarks (BENCHMARKS.md)
server/server.py     the server: web server, live sessions, engines, API
server/live.py       voice detection and the pause cutter (copied from the Mac app)
server/summary.py    Claude summaries
server/store.py      the SQLite store
server/auth.py       Google sign-in through Cloudflare Access (optional)
server/web/          the page
server/bench/        the server's benchmarks and tests
```
