# What we tried on the server

A running log of every experiment on the transcription server, what it measured and what we
kept. Newest work is added at the end of each section. Measurements are from this server:
2 vCPUs (AMD EPYC Zen 4, AVX-512 with VNNI), 7.8 GB RAM, a VPS whose CPU is capped when busy,
so timings vary by about ±10% and get slower when other jobs run.

Word error rates (WER) are on 50 LibriSpeech test-clean + 50 test-other utterances from
`bench/data` (median 6.2 s long), normalized with Whisper's English normalizer like
`BENCHMARKS.md`. "Wait" is the time from the end of the audio to the text.

The scripts behind every number, and their raw results, are in `bench/` (see `bench/README.md`).
All four models tried are kept in `/root/.cache/huggingface/hub` (Parakeet v2 and v3 int8,
Whisper large-v3-turbo and small, about 3.3 GB), so any of them can be switched back or
re-measured without a download.

## Where it runs now

| | |
|---|---|
| Speech to text | Parakeet-TDT-0.6B-v2, int8 ONNX (onnx-asr), 0.4 s of audio around each part |
| Live cutting | the Mac app's `Segmenter`: a part ends at a 0.5 s pause, at most 25 s |
| Summaries | Claude Haiku 5.5, effort low (about 1.6 s) |
| Access | SSH tunnel to a root-only Unix socket: `ssh -N -L 8095:/run/voice-to-text/voice.sock root@your-server` |

## Models

| Model | WER clean | WER other | Wait (median) | Verdict |
|---|---|---|---|---|
| Whisper large-v3-turbo, full 30 s window | | | ~11 s for any clip up to 25 s of speech | too slow |
| Whisper large-v3-turbo, short windows (10/20/30 s) | 1.74% | 3.71% | 5.07 s | most accurate, too slow live |
| Whisper small | 2.73% | 7.07% | 2.81 s | replaced by Parakeet |
| Parakeet v3, int8 | 2.73% | 6.84% | 0.63 s | replaced by v2 |
| Parakeet v3, int8, 0.4 s padding | 2.27% | 5.45% | 0.68 s | |
| Parakeet v2 (English only), int8 | 1.29% | 4.40% | 0.81 s | |
| **Parakeet v2, int8, 0.4 s padding** | **1.36%** | **4.06%** | **0.70 s** | **in use** |
| Parakeet v2, int8, 0.6 s padding | 1.36% | 4.52% | 0.66 s | worse than 0.4 s |
| Parakeet v2 and v3, fp32 | | | | won't load: onnxruntime 1.31 refuses weights stored outside the model's folder (the Hugging Face cache does that); would also need ~3 GB of RAM |

On the user's own voice:

| Test | turbo | small | Parakeet v3 | Parakeet v2 + 0.4 s |
|---|---|---|---|---|
| Two voice memos (28 s, 21 s): wait | 11.7 s, 11.4 s | 3.9 s, 3.4 s | 2.4 s, 2.1 s | |
| Two voice memos: wrong words | 0 | 3 | 3, and one memo without punctuation | |
| 68-word test paragraph: wrong words | | | 7 (plurals lost, "site" → "side", "three thirty" → "thirty three. D.", the name) | 1 (the name, "Pradneesh") |
| Hinglish sentence ("Kal meeting hai, toh report ready rakhna") | | | 5 of 7 words lost | not tested; neither Parakeet knows Hindi |

## Making Whisper turbo faster

| What | Result | Verdict |
|---|---|---|
| Encoder time by window size | 10 s: 1.7 s · 12 s: 2.1 s · 16 s: 4.1 s · 20 s: 5.1 s · 25 s: 8.3 s · 30 s: 9.7–11.3 s | the window is the only big lever |
| faster-whisper `chunk_length=10` | no change: faster-whisper pads every window back to 30 s (`transcribe.py:1180`) | call CTranslate2 directly instead |
| Short windows like the Mac app (≤6 s → 10 s, ≤16 s → 20 s) | 4 s clip 11–15 s → 2.5 s; 15 s clip 12.6 s → 6.2 s | kept in the Whisper code path |
| oneDNN (default), 2 threads | 1.7 s (10 s window) / 9.7 s (30 s) | best |
| oneDNN + experimental packed GEMM | 1.7 s / 9.9 s | no gain |
| Intel MKL | 10.9 s / 37.6 s | 4–6× slower on this AMD CPU |
| 1 thread | 2.6 s / 14.9 s | slower |

Live, wait after Stop (47 s clip with pauses / 15 s clip without):

| Setup | 47 s clip | 15 s clip | Wrong words |
|---|---|---|---|
| turbo, as it was | 10.8 s | 9.6 s | 4 / 0 |
| turbo, interim text off | 6.3 s | 6.4 s | 4 / 0 |
| turbo, interim off, parts ≤16 s | 7.0 s | 6.6 s | 4 / 0 |
| turbo, + windows of speech + 4 s | 3.2 s | 12.0 s | 5 / 0 |
| turbo, + interim every 3 s | 5.9 s | 9.2 s | 5 / 0 |
| small, parts ≤16 s, speech + 4 s windows | 2.2 s | 4.2 s | 6 / 0 |
| Parakeet v3 | 0.8 s | 2.1 s | 2 / 1 |

Not finished: whether live parts of at least 3 s or 6 s (more context per part) help Parakeet.
The test was stopped before results to free the CPU for the user's own test.

## Bugs found and fixed

| Problem | Cause | Fix |
|---|---|---|
| A whole sentence missing from the 47 s clip | faster-whisper's own VAD joins all speech and cuts it into 30 s windows | split at pauses into parts of ≤25 s and transcribe each, like the Mac app |
| Grey interim text flickering away | empty interim updates were sent | send only non-empty ones |
| First word wrong ("listening" → "presentation") | the page dropped audio until the server answered | keep that audio and send it once connected |
| "Live" badge and delete button always showing | CSS `display` overrode the `hidden` attribute | `[hidden] { display: none !important }` |
| Meeting: the user's voice missing entirely | with speakers, the mic hears the meeting; whole "You" parts were dropped as echo | remove only runs of 3+ words the meeting said at the same moment (±1 s) |
| Meeting: echo lines showing until Stop | echo was only removed at the end | the server revises "You" lines live |

## Features

| Feature | Status |
|---|---|
| Upload API (`POST /transcribe`) | built first, still there |
| Site with live text, history, summaries | in use |
| Summaries with Claude Opus 5.5 | replaced by Haiku 5.5 at the user's request (about 40× cheaper) |
| Redesigned page (waveform, Space to record, drag-and-drop, light/dark) | in use |
| Meeting mode (mic + a Chrome tab's audio, "You" / "Meeting" turns, meeting summary) | in use; only browser-tab meetings, "Meeting" is everyone else together |
| Per-person transcripts (by Google email) | built; everything made over the SSH tunnel is visible to the tunnel |

## Access

| Option | Status |
|---|---|
| TCP port 127.0.0.1:8095 | dropped: every account on the server could reach it |
| SSH tunnel to a root-only Unix socket | in use |
| Subdomain with one shared password | built and tested, then dropped for Google |
| Subdomain with Google sign-in through Cloudflare Access | built and tested against a fake Cloudflare (14/14 checks), not switched on: the SSH tunnel was chosen instead. `server/auth.py` + `--public-port` + `set-google-access.sh` turn it on |

## Ideas not tried yet

- A corrections list for names (e.g. "Pradneesh" → "Pradnyesh"): Parakeet can't be told which words to expect.
- Parakeet for instant text plus turbo re-checking each sentence a few seconds later.
- Telling the other meeting participants apart (needs a speaker-recognition model).
