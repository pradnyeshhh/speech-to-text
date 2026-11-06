"""Scoring, and BENCHMARKS.md from the saved results of compare_models.py and streaming.py.

    uv run python bench/report.py   # rewrite BENCHMARKS.md without measuring anything
"""

import functools
import json
import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import librispeech

ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT / "bench" / "results.json"
FULL_WINDOW_JSON = ROOT / "bench" / "results-full-window.json"
STREAMING_JSON = ROOT / "bench" / "results-streaming.json"
BENCHMARKS_MD = ROOT / "BENCHMARKS.md"
SHORT_S = 6  # the app encodes clips up to this long in a 10 s window


# Scoring


@functools.lru_cache
def _normalizer():
    from whisper_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


def counts(items: list[dict]) -> dict[str, np.ndarray]:
    """Per-utterance word errors, insertions and reference words, after Whisper's English normalizer."""
    import jiwer

    normalize = _normalizer()
    errors, insertions, words = [], [], []
    for item in items:
        out = jiwer.process_words(normalize(item["ref"]), normalize(item["hyp"]))
        errors.append(out.substitutions + out.deletions + out.insertions)
        insertions.append(out.insertions)
        words.append(out.substitutions + out.deletions + out.hits)
    return {"errors": np.array(errors), "insertions": np.array(insertions), "words": np.array(words)}


def wer(items: list[dict]) -> float:
    """Corpus WER (all errors over all reference words), as in Whisper's published numbers."""
    c = counts(items)
    return c["errors"].sum() / c["words"].sum()


def doubled(items: list[dict]) -> int:
    """Transcripts that repeat themselves: at least 3 inserted words, and half as many as the reference has."""
    c = counts(items)
    return int((c["insertions"] >= np.maximum(3, 0.5 * c["words"])).sum())


def wer_difference_ci(base: list[dict], other: list[dict], resamples: int = 2000) -> tuple[float, float]:
    """95% paired-bootstrap interval for other's WER minus base's, resampling utterances."""
    base_counts, other_counts = counts(base), counts(other)
    words = base_counts["words"]
    picks = np.random.default_rng(0).integers(0, len(words), (resamples, len(words)))
    totals = words[picks].sum(axis=1)
    differences = (other_counts["errors"][picks].sum(axis=1) - base_counts["errors"][picks].sum(axis=1)) / totals
    low, high = np.percentile(differences, [2.5, 97.5])
    return low, high


def wer_change(base: list[dict], other: list[dict]) -> str:
    low, high = wer_difference_ci(base, other)
    return f"{(wer(other) - wer(base)) * 100:+.2f} points ({low * 100:+.2f} to {high * 100:+.2f})"


# Facts about the run


def machine() -> dict:
    def run(*command: str) -> str:
        return subprocess.run(command, capture_output=True, text=True).stdout

    hardware = run("system_profiler", "SPHardwareDataType")
    displays = run("system_profiler", "SPDisplaysDataType")

    def field(text: str, name: str) -> str:
        match = re.search(rf"{name}: (.+)", text)
        return match.group(1).strip() if match else "?"

    return {
        "name": field(hardware, "Model Name"),
        "identifier": field(hardware, "Model Identifier"),
        "chip": field(hardware, "Chip"),
        "cpu_cores": field(hardware, "Total Number of Cores"),
        "gpu_cores": field(displays, "Total Number of Cores"),
        "memory": field(hardware, "Memory"),
        "os": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "mlx_whisper": version("mlx-whisper"),
    }


def weights_gb(repo: str) -> float:
    from voice_to_text.stt import local_model_path

    return sum(p.stat().st_size for p in Path(local_model_path(repo)).iterdir() if p.is_file()) / 1e9


# Formatting helpers


def merged_latency(model_result: dict) -> dict:
    merged = {}
    for round_result in model_result["latency_rounds"]:
        for name, clip in round_result["clips"].items():
            entry = merged.setdefault(name, {"seconds": [], "redecodes": 0, "texts": []})
            entry["seconds"] += clip["seconds"]
            entry["redecodes"] += clip["redecodes"]
            entry["texts"] += [t for t in clip["texts"] if t not in entry["texts"]]
    return merged


def _split(items: list[dict], split: str) -> list[dict]:
    return [i for i in items if i["split"] == split]


def _short(items: list[dict]) -> list[dict]:
    return [i for i in items if i["audio_s"] <= SHORT_S]


def _ms(seconds: list[float], q: float) -> str:
    return f"{np.percentile(seconds, q) * 1000:.0f}"


def _pcts(seconds: list[float]) -> str:
    return " / ".join(_ms(seconds, q) for q in (50, 95, 99))


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# BENCHMARKS.md


def write() -> None:
    if not RESULTS_JSON.exists():
        print(f"No {RESULTS_JSON.relative_to(ROOT)} yet, so no report.", file=sys.stderr)
        return
    load = lambda path: json.loads(path.read_text()) if path.exists() else None  # noqa: E731
    BENCHMARKS_MD.write_text(markdown(load(RESULTS_JSON), load(FULL_WINDOW_JSON), load(STREAMING_JSON)))
    print(f"Wrote {BENCHMARKS_MD.relative_to(ROOT)}", file=sys.stderr)


def markdown(results: dict, full_window: dict | None = None, streaming: dict | None = None) -> str:
    machine_info, settings, clips = results["machine"], results["settings"], results["clips"]
    models = results["models"]
    names = list(models)
    baseline = names[0]
    out = [
        "# Benchmarks",
        "",
        f"voice_to_text's speech-to-text on the Mac (`--local`), measured on {results['date']} by the scripts in `bench/`.",
        "Every number here is a measurement; rerun the scripts to refresh them.",
        "",
        "## Hardware",
        "",
        "| | |",
        "|---|---|",
        f"| Machine | {machine_info['name']} ({machine_info['identifier']}) |",
        f"| Chip | {machine_info['chip']}, {machine_info['cpu_cores']} CPU cores, {machine_info['gpu_cores']}-core GPU |",
        f"| Memory | {machine_info['memory']} |",
        f"| Software | {machine_info['os']}, Python {machine_info['python']}, mlx {machine_info['mlx']}, "
        f"mlx-whisper {machine_info['mlx_whisper']} |",
        "",
    ]
    if "Air" in machine_info["name"]:
        out += [
            "A MacBook Air has no fan, so a long run gets slower as the chip heats up; the alternating "
            "rounds spread that evenly across models.",
            "",
        ]

    if streaming:
        out += _streaming_section(streaming)

    wer_ready = all(models[m]["wer"] for m in names)
    latency_ready = all(models[m]["latency_rounds"] for m in names)
    out += ["## Models", "", "Each recording transcribed whole, after the key is released (`bench/compare_models.py`).", ""]
    if wer_ready:
        items0 = models[baseline]["wer"]["items"]
        out += [
            "### Accuracy (word error rate, lower is better)",
            "",
            f"| Model | test-clean | test-other | Utterances ≤ {SHORT_S} s | Long clip | Doubled transcripts "
            "| Fallback re-decodes |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in names:
            items = models[m]["wer"]["items"]
            latency = merged_latency(models[m])
            long_wer = _pct(wer([{"ref": clips["long"]["text"], "hyp": latency["long"]["texts"][0]}])) if latency else "n/a"
            out.append(
                f"| {m} | {_pct(wer(_split(items, 'clean')))} | {_pct(wer(_split(items, 'other')))} "
                f"| {_pct(wer(_short(items)))} | {long_wer} | {doubled(items)} of {len(items)} "
                f"| {sum(i['redecodes'] for i in items)} |"
            )
        out += [
            "",
            f"{len(_split(items0, 'clean'))} LibriSpeech test-clean and {len(_split(items0, 'other'))} test-other "
            f"utterances, a fixed random sample (seed 0); {len(_short(items0))} of them are {SHORT_S} s or shorter. "
            "Both the reference and the transcript go through Whisper's English text normalizer, so formatting "
            "like \"20\" vs \"twenty\" isn't counted as an error. A doubled transcript has at least 3 extra words "
            "and at least half as many as the reference has, usually the sentence written twice. Fallback re-decodes "
            "count how often the transcriber decoded again because a result looked like a hallucination.",
            "",
        ]

    if latency_ready:
        header = " | ".join(f"{name} ({clips[name]['duration_s']:.1f} s)" for name in clips)
        out += [
            "### Latency (milliseconds, p50 / p95 / p99)",
            "",
            f"| Model | {header} |",
            "|---|" + "---|" * len(clips),
        ]
        for m in names:
            latency = merged_latency(models[m])
            out.append(f"| {m} | " + " | ".join(_pcts(latency[c]["seconds"]) for c in clips) + " |")
        runs = len(merged_latency(models[baseline])["short"]["seconds"])
        out += [
            "",
            f"Time for `Transcriber.transcribe()` on audio already in memory: VAD plus Whisper. "
            f"{runs} runs per clip and model, in {settings['rounds']} rounds that alternate the model order, "
            f"{settings['gap']} s apart. The clips are LibriSpeech test-clean: one utterance near 4 s, one near "
            f"15 s, and consecutive utterances from one chapter joined by 0.5-1.0 s pauses.",
            "",
        ]

    if wer_ready:
        out += [
            "### Throughput on the WER set",
            "",
            "| Model | Audio | Processing | Faster than real time | Per utterance p50 / p95 |",
            "|---|---|---|---|---|",
        ]
        for m in names:
            items = models[m]["wer"]["items"]
            audio_s = sum(i["audio_s"] for i in items)
            busy_s = sum(i["seconds"] for i in items)
            per_item = [i["seconds"] for i in items]
            out.append(
                f"| {m} | {audio_s / 60:.1f} min | {busy_s:.1f} s | {audio_s / busy_s:.0f}x "
                f"| {_ms(per_item, 50)} / {_ms(per_item, 95)} ms |"
            )
        out.append("")

    if wer_ready and latency_ready and len(names) > 1:
        out += [
            f"### Compared with {baseline}",
            "",
            "| Model | WER test-clean | WER test-other | " + " | ".join(f"p50 {c}" for c in clips) + " |",
            "|---|---|---|" + "---|" * len(clips),
        ]
        base_latency = merged_latency(models[baseline])
        for m in names[1:]:
            latency = merged_latency(models[m])
            cells = [
                wer_change(_split(models[baseline]["wer"]["items"], s), _split(models[m]["wer"]["items"], s))
                for s in librispeech.SPLITS
            ]
            cells += [
                f"{np.median(latency[c]['seconds']) / np.median(base_latency[c]['seconds']):.2f}x" for c in clips
            ]
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out += [
            "",
            f"WER: change in percentage points, negative is more accurate, with a 95% paired-bootstrap "
            f"interval over utterances; an interval that spans zero means no measurable difference. "
            f"Latency: median time as a multiple of {baseline}'s, below 1.00x is faster.",
            "",
        ]

    if full_window:
        out += _full_window_section(results, full_window)

    out += [
        "### Model size and startup",
        "",
        "| Model | Weights on disk | Load + warm-up | Peak GPU memory |",
        "|---|---|---|---|",
    ]
    for m in names:
        runs = models[m]["latency_rounds"] + ([models[m]["wer"]] if models[m]["wer"] else [])
        load_s = np.median([r["load_s"] for r in runs])
        peak = max(r["peak_memory_gb"] for r in runs)
        out.append(f"| {m} | {models[m]['weights_gb']:.2f} GB | {load_s:.1f} s | {peak:.2f} GB |")
    out += [
        "",
        "Load + warm-up is the median over the run's processes, with the weights already in the OS file cache.",
        "",
        "## Rerun",
        "",
        "```sh",
        "uv run python bench/streaming.py                      # transcribing while you talk",
        "uv run python bench/compare_models.py                 # the model tables",
        "uv run python bench/compare_models.py --full-window   # the 30 s window rows",
        "uv run python bench/report.py                         # rewrite this file from saved results",
        "```",
        "",
    ]
    return "\n".join(out)


def _full_window_section(results: dict, full_window: dict) -> list[str]:
    clips = results["clips"]
    out = [
        "### The shortened encoder window vs the full 30 s",
        "",
        f"To save time the app encodes a clip of up to {SHORT_S} s in a 10 s window and up to 16 s in a 20 s "
        "window, instead of the 30 s window Whisper was trained on. The 30 s rows turn that off "
        f"(`--full-window`, a separate run on {full_window['date']} with the same settings).",
        "",
        f"| Model | Encoder window | WER test-clean | WER test-other | WER ≤ {SHORT_S} s | Doubled transcripts | "
        + " | ".join(f"p50 {c}" for c in clips)
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(clips),
    ]
    for m, full in full_window["models"].items():
        if m not in results["models"]:
            continue
        for label, source in (("shortened (app)", results["models"][m]), ("30 s", full)):
            cells = [m, label]
            if source["wer"]:
                items = source["wer"]["items"]
                cells += [
                    _pct(wer(_split(items, "clean"))),
                    _pct(wer(_split(items, "other"))),
                    _pct(wer(_short(items))),
                    f"{doubled(items)} of {len(items)}",
                ]
            else:
                cells += ["n/a"] * 4
            latency = merged_latency(source)
            cells += [f"{np.median(latency[c]['seconds']) * 1000:.0f} ms" if latency else "n/a" for c in clips]
            out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def _streaming_section(streaming: dict) -> list[str]:
    out = [
        "## Transcribing while you talk",
        "",
        "The app cuts each recording at pauses while you're still talking and transcribes every part as soon "
        "as it's complete, so after you release the key only the part since your last pause is left "
        "(`bench/streaming.py`).",
        "",
    ]
    latency = streaming.get("latency")
    if latency:
        out += [
            "### Time from the end of speech to text (milliseconds, p50 / p95 / p99)",
            "",
            "| Clip | Parts | Transcribed after release (before) | Transcribed while you talk (now) | Median saved |",
            "|---|---|---|---|---|",
        ]
        for name, clip in latency["clips"].items():
            parts = sorted(set(clip["parts"]))
            saved = (np.median(clip["batch_s"]) - np.median(clip["stream_s"])) * 1000
            out.append(
                f"| {name} ({clip['duration_s']:.1f} s) | {'-'.join(map(str, parts)) if len(parts) > 1 else parts[0]} "
                f"| {_pcts(clip['batch_s'])} | {_pcts(clip['stream_s'])} | {saved:.0f} ms |"
            )
        out += [
            "",
            f"{latency['model']}, {_cut_description(latency['options'])}; measured on {latency['date']}. Each clip "
            f"is played into the app's pipeline at real-time speed, as {latency['mic_rate'] // 1000} kHz microphone "
            f"audio, {latency['runs']} times, alternating which method goes first, and timed from the end of the "
            "audio to the text. Before is what the app did until now: resample the whole recording, then "
            "`Transcriber.transcribe()`. Neither includes the 150 ms post-roll or the paste.",
            "",
        ]
    accuracy = streaming.get("accuracy")
    if accuracy:
        methods = accuracy["methods"]
        before = methods["after"]["items"]
        n = {s: len(_split(before, s)) for s in librispeech.SPLITS}
        out += [
            "### Accuracy (word error rate, lower is better)",
            "",
            "| Method | test-clean | test-other | Change vs whole, clean | Change vs whole, other | Parts per recording |",
            "|---|---|---|---|---|---|",
        ]
        for key, method in methods.items():
            items = method["items"]
            parts = [i["parts"] for i in items if i["parts"] is not None]
            changes = (
                ["", ""] if key == "after" else [wer_change(_split(before, s), _split(items, s)) for s in librispeech.SPLITS]
            )
            out.append(
                f"| {method['label']} | {_pct(wer(_split(items, 'clean')))} | {_pct(wer(_split(items, 'other')))} | "
                + " | ".join(changes)
                + f" | {f'{np.mean(parts):.1f} (up to {max(parts)})' if parts else '1 per 25 s'} |"
            )
        pause = next(m["options"]["pause_s"] for m in methods.values() if m["options"])
        out += [
            "",
            f"{n['clean']} test-clean and {n['other']} test-other recordings, each four consecutive utterances of "
            "the WER sample joined by 0.5-1.0 s pauses, like dictating a few sentences. "
            f"{accuracy['model']}, a part ends at a pause of {pause} s or more once it has the stated amount of "
            f"speech; measured on {accuracy['date']}. Changes come with a 95% paired-bootstrap interval over "
            "recordings.",
            "",
        ]
    return out


def _cut_description(options: dict) -> str:
    text = f"cutting at pauses of {options['pause_s']} s or more"
    if options.get("min_part_s"):
        text += f" once a part has {options['min_part_s']:g} s of speech"
    if options.get("context_chars"):
        text += f", with the last {options['context_chars']} characters of text as a prompt"
    if options.get("full_window_while_talking"):
        text += ", parts finished before release in the full 30 s window"
    return text


if __name__ == "__main__":
    write()
