"""Compare voice_to_text's speech-to-text models on accuracy (WER) and speed. Real measurements only.

    uv run python bench/compare_models.py
    uv run python bench/compare_models.py --models distil-large-v3 large-v3-turbo --runs 20
    uv run python bench/compare_models.py --full-window   # without the app's shortened encoder window

It times voice_to_text.stt.Transcriber.transcribe() on whole recordings (Silero VAD, then Whisper on the
Apple GPU). Each model runs in its own process so only one is in memory, and the latency rounds
alternate the model order so a Mac heating up doesn't favour whichever model goes first. Saves
every transcript to bench/results.json (or bench/results-full-window.json) and rewrites BENCHMARKS.md.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import librispeech
import report

_RESULT_PREFIX = "RESULT "


def main() -> None:
    from voice_to_text.stt import MODELS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--runs", type=int, default=20, help="latency runs per clip and model (default 20)")
    parser.add_argument("--rounds", type=int, default=4, help="rounds the runs are split into (default 4)")
    parser.add_argument("--gap", type=float, default=0.5, help="idle seconds between transcriptions")
    parser.add_argument("--cooldown", type=float, default=15, help="idle seconds between model processes")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--limit", type=int, help="WER on only the first N utterances per split (quick check)")
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="always encode Whisper's full 30 s window, not the app's shorter ones for short clips",
    )
    parser.add_argument("--worker", choices=["latency", "wer"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    short_windows = not args.full_window

    if args.worker:
        model = args.models[0]
        if args.worker == "latency":
            result = _worker_latency(model, short_windows, args.runs, args.gap)
        else:
            result = _worker_wer(model, short_windows, args.limit)
        print(_RESULT_PREFIX + json.dumps(result), flush=True)
        return

    librispeech.prepare()
    clips = librispeech.load_clips()
    results = {
        "date": date.today().isoformat(),
        "machine": report.machine(),
        "settings": {k: v for k, v in vars(args).items() if k != "worker"},
        "clips": {name: {"duration_s": c["duration_s"], "text": c["text"]} for name, c in clips.items()},
        "models": {m: {"repo": MODELS[m], "latency_rounds": [], "wer": None} for m in args.models},
    }
    window_flag = [] if short_windows else ["--full-window"]
    first = True

    if not args.skip_latency:
        per_round = [args.runs // args.rounds + (r < args.runs % args.rounds) for r in range(args.rounds)]
        for r, runs in enumerate(per_round):
            if not runs:
                continue
            for model in args.models if r % 2 == 0 else args.models[::-1]:
                first = _cooldown(first, args.cooldown)
                print(f"Latency round {r + 1}/{args.rounds}: {model}, {runs} runs", file=sys.stderr, flush=True)
                extra = ["--runs", str(runs), "--gap", str(args.gap), *window_flag]
                results["models"][model]["latency_rounds"].append(_spawn("latency", model, *extra))

    if not args.skip_wer:
        for model in args.models:
            first = _cooldown(first, args.cooldown)
            print(f"WER: {model}", file=sys.stderr, flush=True)
            limit = ["--limit", str(args.limit)] if args.limit else []
            results["models"][model]["wer"] = _spawn("wer", model, *limit, *window_flag)

    for model in args.models:
        results["models"][model]["weights_gb"] = report.weights_gb(MODELS[model])
    path = report.RESULTS_JSON if short_windows else report.FULL_WINDOW_JSON
    path.write_text(json.dumps(results, indent=1))
    print(f"Wrote {path.relative_to(report.ROOT)}", file=sys.stderr)
    report.write()


def _cooldown(first: bool, seconds: float) -> bool:
    if not first:
        time.sleep(seconds)
    return False


def _spawn(kind: str, model: str, *extra: str) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", kind, "--models", model, *extra]
    process = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    if process.returncode:
        raise SystemExit(f"The {kind} run for {model} failed (exit {process.returncode})")
    line = next(l for l in reversed(process.stdout.splitlines()) if l.startswith(_RESULT_PREFIX))
    return json.loads(line[len(_RESULT_PREFIX) :])


# Workers: one model per process.


class _DecodeCounter:
    """Counts the extra decodes Transcriber does when a result looks wrong (its fallbacks)."""

    def __init__(self, transcriber):
        self.chunks = self.decodes = 0
        decode, chunk = transcriber._decode, transcriber._transcribe_chunk

        def counting_decode(*args, **kwargs):
            self.decodes += 1
            return decode(*args, **kwargs)

        def counting_chunk(*args, **kwargs):
            self.chunks += 1
            return chunk(*args, **kwargs)

        transcriber._decode, transcriber._transcribe_chunk = counting_decode, counting_chunk

    def take_redecodes(self) -> int:
        redecodes = self.decodes - self.chunks
        self.chunks = self.decodes = 0
        return redecodes


def _load(model: str, short_windows: bool):
    import mlx.core as mx

    from voice_to_text.stt import Transcriber

    mx.random.seed(0)  # the temperature fallback samples
    started = time.perf_counter()
    transcriber = Transcriber(model, short_windows=short_windows)
    transcriber.load()  # includes the warm-up decodes, as at app start
    return transcriber, time.perf_counter() - started


def _peak_memory_gb() -> float:
    import mlx.core as mx

    get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get_peak() / 1e9


def _worker_latency(model: str, short_windows: bool, runs: int, gap_s: float) -> dict:
    transcriber, load_s = _load(model, short_windows)
    counter = _DecodeCounter(transcriber)
    clips = librispeech.load_clips()
    out = {name: {"seconds": [], "redecodes": 0, "texts": []} for name in clips}
    for run in range(runs):
        for name, clip in clips.items():
            time.sleep(gap_s)
            started = time.perf_counter()
            text = transcriber.transcribe(clip["audio"])
            out[name]["seconds"].append(time.perf_counter() - started)
            out[name]["redecodes"] += counter.take_redecodes()
            if text not in out[name]["texts"]:
                out[name]["texts"].append(text)
        print(f"  {model}: run {run + 1}/{runs}", file=sys.stderr, flush=True)
    return {"load_s": load_s, "peak_memory_gb": _peak_memory_gb(), "clips": out}


def _worker_wer(model: str, short_windows: bool, limit: int | None) -> dict:
    transcriber, load_s = _load(model, short_windows)
    counter = _DecodeCounter(transcriber)
    items = []
    for split in librispeech.SPLITS:
        for utterance in librispeech.utterances(split)[:limit]:  # audio is decoded before timing starts
            started = time.perf_counter()
            hypothesis = transcriber.transcribe(utterance["audio"])
            items.append(
                {
                    "split": split,
                    "id": utterance["id"],
                    "ref": utterance["text"],
                    "hyp": hypothesis,
                    "seconds": time.perf_counter() - started,
                    "audio_s": utterance["audio"].size / librispeech.SAMPLE_RATE,
                    "redecodes": counter.take_redecodes(),
                }
            )
        print(f"  {model}: test-{split} done", file=sys.stderr, flush=True)
    return {"load_s": load_s, "peak_memory_gb": _peak_memory_gb(), "items": items}


if __name__ == "__main__":
    main()
