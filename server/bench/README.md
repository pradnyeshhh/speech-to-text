# Server benchmarks and tests

The scripts behind the numbers in `../EXPERIMENTS.md`, and their raw results in `results/`.
Run them from `/root/voice_to_text` with the project's venv. They need the audio in
`bench/data` (the Mac app's LibriSpeech sample and clips). Most load a model, so they compete
with the lead pipeline for CPU: run one at a time, and not while someone is using the site.

The models they use are all kept in `/root/.cache/huggingface/hub` (about 3.3 GB):
Parakeet v2 and v3 (int8 ONNX), Whisper large-v3-turbo and Whisper small. Add
`HF_HUB_OFFLINE=1` to run without the network.

## Accuracy and speed of a model

| Script | What it measures | Example |
|---|---|---|
| `wer_bench.py` | Word error rate and wait on 50 test-clean + 50 test-other utterances | `.venv/bin/python -I server/bench/wer_bench.py parakeet-v2-int8-pad400 50 /tmp/out.json` (engines: `parakeet[-v2\|-v3][-int8][-padNNN]`, `small`, `turbo-short`, `turbo-full`) |
| `live_cut_bench.py` | How the live cutting affects Parakeet, on 4-sentence recordings | `.venv/bin/python -I server/bench/live_cut_bench.py v2 int8 0.4` |
| `parakeet_probe.py` | Parakeet on the short / medium / long clips, words wrong | `.venv/bin/python -I server/bench/parakeet_probe.py 2 short medium long` |
| `turbo_bench.py` | Whisper turbo through faster-whisper with one set of options | `.venv/bin/python -I server/bench/turbo_bench.py 2 int8 none 2 bench/data/clips/short.wav` |
| `encode_env.py`, `encode_probe.py` | Whisper turbo's encoder time per window size (set `CT2_USE_MKL=1` etc. to compare math libraries) | `.venv/bin/python -I server/bench/encode_env.py 2 10 20 30` |

## The live site

| Script | What it does | Example |
|---|---|---|
| `live_sweep.sh` | Starts a test server with given options, plays two clips into it live, prints wait after Stop and words wrong | `bash server/bench/live_sweep.sh T1 large-v3-turbo --interim-every 0` |
| `live_measure.py` | One clip into a running server's `/live`, in real time, as one JSON line | `.venv/bin/python -I server/bench/live_measure.py /run/voice-to-text/voice.sock long` |
| `live_client.py` | One clip into `/live`, printing every message | `.venv/bin/python -I server/bench/live_client.py /run/voice-to-text/voice.sock bench/data/clips/medium.wav 16000 1` |
| `meeting_client.py` | A two-channel meeting into `/live`; second argument is how loudly the meeting leaks into the mic, a third makes "you" talk over it | `.venv/bin/python -I server/bench/meeting_client.py SOCKET 0.35 over` |
| `check_clips.py` | Uploads the three clips and silence to `/transcribe` (written for the old TCP port 8095) | |

Anything sent to the real service's socket is saved in its history: delete test transcripts
afterwards (`DELETE /api/transcripts/<id>`), or point the script at a test server started with
`--data` somewhere else.

## Sign-in and the page

| Script | What it does |
|---|---|
| `auth_test.py` | Google sign-in (`../auth.py`) against a fake Cloudflare: its own RSA key, key list and tokens; 14 checks |
| `ui_test.py`, `ui_meeting_stub.py`, `browser_test.py` | The page in headless Chromium with a fake microphone, plus screenshots. They use the lead API's Playwright (`/root/lead-platform/backend/.venv/bin/python`, read-only) and a test server on `http://127.0.0.1:8096` |

## results/

`wer-*.json` hold every utterance's reference, transcript, errors and time for each engine;
`parakeet-variants.txt` and `sweep-results.txt` are the summary lines the tables in
`../EXPERIMENTS.md` came from.
