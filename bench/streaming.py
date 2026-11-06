"""Transcribing while you talk vs after the key is released. Real measurements only.

    uv run python bench/streaming.py                  # accuracy, then latency at real-time speed
    uv run python bench/streaming.py --skip-latency   # accuracy only
    uv run python bench/streaming.py --skip-accuracy --runs 5 --min-part 3

Accuracy: recordings of four consecutive LibriSpeech utterances joined by pauses, transcribed
whole (as before) and in parts with several cutting options. Latency: each clip is played into
voice_to_text's Pipeline at real-time speed, as 48 kHz microphone audio, and timed from the end of the
audio to the text, against what the app did before: resample the whole recording, then
Transcriber.transcribe(). Saves bench/results-streaming.json and rewrites BENCHMARKS.md.
"""

import argparse
import json
import queue
import sys
import time
from datetime import date

import numpy as np
from scipy.signal import resample_poly

import librispeech
import report

MIC_RATE = 48000  # a Mac's built-in microphone; the pipeline resamples it to 16 kHz

# Options compared for accuracy: key -> (label, options for transcribe_in_parts).
VARIANTS = {
    "every_pause": ("In parts, at every pause", {}),
    "min_3s": ("In parts of at least 3 s", {"min_part_s": 3.0}),
    "min_6s": ("In parts of at least 6 s", {"min_part_s": 6.0}),
    "every_pause_prompt": ("In parts, at every pause, earlier text as a prompt", {"context_chars": 200}),
    "every_pause_full": (
        "In parts, at every pause, full window while talking",
        {"full_window_while_talking": True},
    ),
    "min_6s_full": (
        "In parts of at least 6 s, full window while talking",
        {"min_part_s": 6.0, "full_window_while_talking": True},
    ),
}


def main() -> None:
    from voice_to_text.streaming import FULL_WINDOW_WHILE_TALKING, MIN_PART_S, PAUSE_S
    from voice_to_text.stt import DEFAULT_MODEL, MODELS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--pause", type=float, default=PAUSE_S, help="silence that ends a part, in seconds")
    parser.add_argument("--min-part", type=float, default=MIN_PART_S, help="speech a part needs before a pause ends it (latency)")
    parser.add_argument("--context-chars", type=int, default=0, help="earlier text passed as a prompt (latency)")
    parser.add_argument(
        "--full-window-while-talking",
        action="store_true",
        default=FULL_WINDOW_WHILE_TALKING,
        help="parts finished before release use the full 30 s window (latency)",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), help="accuracy for only these, kept with earlier results"
    )
    parser.add_argument("--runs", type=int, default=20, help="latency runs per clip (default 20)")
    parser.add_argument("--gap", type=float, default=0.5, help="idle seconds between runs")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--limit", type=int, help="accuracy on only the first N recordings per split (quick check)")
    args = parser.parse_args()

    import mlx.core as mx

    from voice_to_text.stt import Transcriber

    librispeech.prepare()
    mx.random.seed(0)  # the temperature fallback samples
    transcriber = Transcriber(args.model)
    transcriber.load()

    # A section that isn't rerun keeps its earlier results.
    results = json.loads(report.STREAMING_JSON.read_text()) if report.STREAMING_JSON.exists() else {}
    results["machine"] = report.machine()
    if not args.skip_accuracy:
        results["accuracy"] = _accuracy(transcriber, args, results.get("accuracy"))
        report.STREAMING_JSON.write_text(json.dumps(results, indent=1))
    if not args.skip_latency:
        results["latency"] = _latency(transcriber, args)
    report.STREAMING_JSON.write_text(json.dumps(results, indent=1))
    print(f"Wrote {report.STREAMING_JSON.relative_to(report.ROOT)}", file=sys.stderr)
    report.write()


def _accuracy(transcriber, args, previous: dict | None) -> dict:
    from voice_to_text.streaming import transcribe_in_parts

    recordings = librispeech.stitched()
    if args.limit:
        recordings = [r for s in librispeech.SPLITS for r in [r for r in recordings if r["split"] == s][: args.limit]]
    # With --variants, keep the other methods from the earlier run (same model, same recordings).
    keep = previous and args.variants and not args.limit and previous["model"] == args.model
    methods = dict(previous["methods"]) if keep else {}
    run = ["after"] if "after" not in methods else []
    run += args.variants or list(VARIANTS)
    for key in run:
        label, options = ("Whole, after release (before)", None) if key == "after" else VARIANTS[key]
        methods[key] = {"label": label, "options": None if key == "after" else {"pause_s": args.pause, **options}, "items": []}
    for number, recording in enumerate(recordings, 1):
        audio = recording["audio"]
        base = {"split": recording["split"], "id": recording["id"], "ref": recording["text"], "audio_s": audio.size / 16000}
        for key in run:
            if key == "after":
                text, parts = transcriber.transcribe(audio), None
            else:
                text, parts = transcribe_in_parts(transcriber, audio, **methods[key]["options"])
            methods[key]["items"].append({**base, "hyp": text, "parts": parts})
        if number % 10 == 0:
            print(f"  accuracy: {number}/{len(recordings)} recordings", file=sys.stderr, flush=True)
    order = ["after", *VARIANTS]
    return {
        "date": date.today().isoformat(),
        "model": args.model,
        "methods": {key: methods[key] for key in order if key in methods},
    }


def _latency(transcriber, args) -> dict:
    from voice_to_text.streaming import Pipeline

    options = {
        "pause_s": args.pause,
        "min_part_s": args.min_part,
        "context_chars": args.context_chars,
        "full_window_while_talking": args.full_window_while_talking,
    }
    clips = librispeech.load_clips()
    microphone = {name: resample_poly(c["audio"], MIC_RATE // 16000, 1).astype(np.float32) for name, c in clips.items()}
    results: queue.Queue = queue.Queue()
    pipeline = Pipeline(transcriber, results.put, **options)
    pipeline.start()  # the model is already loaded; the worker uses it only while a recording is open

    out = {
        name: {"duration_s": c["duration_s"], "stream_s": [], "batch_s": [], "parts": [], "texts": []}
        for name, c in clips.items()
    }

    def stream(name: str, audio: np.ndarray) -> str:
        result = _play(pipeline, results, audio)
        out[name]["stream_s"].append(result.latency_s)
        out[name]["parts"].append(result.parts)
        return result.text

    def batch(name: str, audio: np.ndarray) -> str:
        started = time.perf_counter()  # the old way: resample all of it, then transcribe all of it
        text = transcriber.transcribe(resample_poly(audio, 1, MIC_RATE // 16000).astype(np.float32))
        out[name]["batch_s"].append(time.perf_counter() - started)
        return text

    for run in range(args.runs):
        for name, audio in microphone.items():
            # Alternate which goes first, so neither always meets a warmer chip.
            for kind, measure in (("stream", stream), ("batch", batch))[:: 1 if run % 2 == 0 else -1]:
                time.sleep(args.gap)
                text = measure(name, audio)
                if [kind, text] not in out[name]["texts"]:
                    out[name]["texts"].append([kind, text])
        print(f"  latency: run {run + 1}/{args.runs}", file=sys.stderr, flush=True)
    return {
        "date": date.today().isoformat(),
        "model": args.model,
        "options": options,
        "runs": args.runs,
        "mic_rate": MIC_RATE,
        "clips": out,
    }


def _play(pipeline, results: queue.Queue, audio: np.ndarray):
    """Feed audio to the pipeline in 20 ms blocks at the pace a microphone delivers them."""
    from voice_to_text.recorder import BLOCK_S, Recording

    recording = Recording(MIC_RATE)
    pipeline.begin(recording)
    block = int(MIC_RATE * BLOCK_S)
    started = time.perf_counter()
    for i, offset in enumerate(range(0, audio.size, block)):
        delay = started + (i + 1) * BLOCK_S - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        recording.append(audio[offset : offset + block])
    pipeline.end(recording, time.monotonic())  # the key goes up as the last audio arrives
    recording.close()
    return results.get(timeout=120)


if __name__ == "__main__":
    main()
