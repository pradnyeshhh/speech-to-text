# Benchmarks

bolo's speech-to-text, measured on 2026-11-04 by the scripts in `bench/`.
Every number here is a measurement; rerun the scripts to refresh them.

## Hardware

| | |
|---|---|
| Machine | MacBook Air (Mac14,2) |
| Chip | Apple M2, 8 (4 Performance and 4 Efficiency) CPU cores, 8-core GPU |
| Memory | 8 GB |
| Software | macOS 26.6.2, Python 3.12.16, mlx 0.32.4, mlx-whisper 0.4.3 |

A MacBook Air has no fan, so a long run gets slower as the chip heats up; the alternating rounds spread that evenly across models.

## Transcribing while you talk

The app cuts each recording at pauses while you're still talking and transcribes every part as soon as it's complete, so after you release the key only the part since your last pause is left (`bench/streaming.py`).

### Time from the end of speech to text (milliseconds, p50 / p95 / p99)

| Clip | Parts | Transcribed after release (before) | Transcribed while you talk (now) | Median saved |
|---|---|---|---|---|
| short (4.0 s) | 1 | 481 / 505 / 507 | 472 / 497 / 500 | 8 ms |
| medium (15.1 s) | 1 | 1012 / 1030 / 1031 | 1004 / 1022 / 1024 | 9 ms |
| long (46.9 s) | 5 | 3436 / 3712 / 3736 | 949 / 953 / 953 | 2487 ms |

large-v3-turbo-q4, cutting at pauses of 0.5 s or more; measured on 2026-11-04. Each clip is played into the app's pipeline at real-time speed, as 48 kHz microphone audio, 3 times, alternating which method goes first, and timed from the end of the audio to the text. Before is what the app did until now: resample the whole recording, then `Transcriber.transcribe()`. Neither includes the 150 ms post-roll or the paste.

### Accuracy (word error rate, lower is better)

| Method | test-clean | test-other | Change vs whole, clean | Change vs whole, other | Parts per recording |
|---|---|---|---|---|---|
| Whole, after release (before) | 1.53% | 3.97% |  |  | 1 per 25 s |
| In parts, at every pause | 1.97% | 7.71% | +0.45 points (-0.04 to +0.96) | +3.74 points (+1.94 to +6.01) | 6.4 (up to 17) |
| In parts of at least 3 s | 2.91% | 6.36% | +1.38 points (+0.08 to +3.34) | +2.39 points (+1.01 to +4.11) | 4.5 (up to 12) |
| In parts of at least 6 s | 1.60% | 5.14% | +0.07 points (-0.20 to +0.35) | +1.17 points (+0.35 to +1.99) | 3.5 (up to 9) |
| In parts, at every pause, earlier text as a prompt | 1.86% | 8.17% | +0.34 points (-0.20 to +0.99) | +4.20 points (+2.16 to +6.64) | 6.4 (up to 17) |
| In parts, at every pause, full window while talking | 1.79% | 7.06% | +0.26 points (-0.10 to +0.66) | +3.09 points (+1.48 to +5.27) | 6.4 (up to 17) |
| In parts of at least 6 s, full window while talking | 1.64% | 4.90% | +0.11 points (-0.19 to +0.47) | +0.93 points (+0.23 to +1.66) | 3.5 (up to 9) |

25 test-clean and 25 test-other recordings, each four consecutive utterances of the WER sample joined by 0.5-1.0 s pauses, like dictating a few sentences. large-v3-turbo-q4, a part ends at a pause of 0.5 s or more once it has the stated amount of speech; measured on 2026-11-04. Changes come with a 95% paired-bootstrap interval over recordings.

## Models

Each recording transcribed whole, after the key is released (`bench/compare_models.py`).

### Accuracy (word error rate, lower is better)

| Model | test-clean | test-other | Utterances ≤ 6 s | Long clip | Doubled transcripts | Fallback re-decodes |
|---|---|---|---|---|---|---|
| distil-large-v3 | 6.52% | 14.71% | 28.01% | 5.79% | 22 of 200 | 3 |
| large-v3-turbo | 1.90% | 5.72% | 5.45% | 2.48% | 1 of 200 | 1 |
| large-v3-turbo-q4 | 1.90% | 6.13% | 5.92% | 2.48% | 1 of 200 | 0 |

100 LibriSpeech test-clean and 100 test-other utterances, a fixed random sample (seed 0); 96 of them are 6 s or shorter. Both the reference and the transcript go through Whisper's English text normalizer, so formatting like "20" vs "twenty" isn't counted as an error. A doubled transcript has at least 3 extra words and at least half as many as the reference has, usually the sentence written twice. Fallback re-decodes count how often the transcriber decoded again because a result looked like a hallucination.

### Latency (milliseconds, p50 / p95 / p99)

| Model | short (4.0 s) | medium (15.1 s) | long (46.9 s) |
|---|---|---|---|
| distil-large-v3 | 427 / 813 / 849 | 885 / 912 / 914 | 3007 / 3140 / 3240 |
| large-v3-turbo | 424 / 748 / 817 | 975 / 1002 / 1004 | 3367 / 3532 / 3541 |
| large-v3-turbo-q4 | 428 / 586 / 589 | 933 / 949 / 951 | 3184 / 3274 / 3370 |

Time for `Transcriber.transcribe()` on audio already in memory: VAD plus Whisper. 20 runs per clip and model, in 4 rounds that alternate the model order, 0.5 s apart. The clips are LibriSpeech test-clean: one utterance near 4 s, one near 15 s, and consecutive utterances from one chapter joined by 0.5-1.0 s pauses.

### Throughput on the WER set

| Model | Audio | Processing | Faster than real time | Per utterance p50 / p95 |
|---|---|---|---|---|
| distil-large-v3 | 27.2 min | 149.7 s | 11x | 759 / 1394 ms |
| large-v3-turbo | 27.2 min | 173.0 s | 9x | 789 / 2143 ms |
| large-v3-turbo-q4 | 27.2 min | 183.4 s | 9x | 830 / 2130 ms |

### Compared with distil-large-v3

| Model | WER test-clean | WER test-other | p50 short | p50 medium | p50 long |
|---|---|---|---|---|---|
| large-v3-turbo | -4.62 points (-7.78 to -2.12) | -8.99 points (-13.84 to -4.61) | 0.99x | 1.10x | 1.12x |
| large-v3-turbo-q4 | -4.62 points (-7.75 to -2.14) | -8.58 points (-13.35 to -4.28) | 1.00x | 1.05x | 1.06x |

WER: change in percentage points, negative is more accurate, with a 95% paired-bootstrap interval over utterances; an interval that spans zero means no measurable difference. Latency: median time as a multiple of distil-large-v3's, below 1.00x is faster.

### The shortened encoder window vs the full 30 s

To save time the app encodes a clip of up to 6 s in a 10 s window and up to 16 s in a 20 s window, instead of the 30 s window Whisper was trained on. The 30 s rows turn that off (`--full-window`, a separate run on 2026-11-04 with the same settings).

| Model | Encoder window | WER test-clean | WER test-other | WER ≤ 6 s | Doubled transcripts | p50 short | p50 medium | p50 long |
|---|---|---|---|---|---|---|---|---|
| distil-large-v3 | shortened (app) | 6.52% | 14.71% | 28.01% | 22 of 200 | 427 ms | 885 ms | 3007 ms |
| distil-large-v3 | 30 s | 1.94% | 6.07% | 4.51% | 0 of 200 | 1126 ms | 1279 ms | 3766 ms |
| large-v3-turbo | shortened (app) | 1.90% | 5.72% | 5.45% | 1 of 200 | 424 ms | 975 ms | 3367 ms |
| large-v3-turbo | 30 s | 1.71% | 4.20% | 3.20% | 0 of 200 | 1157 ms | 1369 ms | 4028 ms |

### Model size and startup

| Model | Weights on disk | Load + warm-up | Peak GPU memory |
|---|---|---|---|
| distil-large-v3 | 1.51 GB | 4.3 s | 2.16 GB |
| large-v3-turbo | 1.61 GB | 4.0 s | 2.27 GB |
| large-v3-turbo-q4 | 0.46 GB | 4.1 s | 1.36 GB |

Load + warm-up is the median over the run's processes, with the weights already in the OS file cache.

## Rerun

```sh
uv run python bench/streaming.py                      # transcribing while you talk
uv run python bench/compare_models.py                 # the model tables
uv run python bench/compare_models.py --full-window   # the 30 s window rows
uv run python bench/report.py                         # rewrite this file from saved results
```
