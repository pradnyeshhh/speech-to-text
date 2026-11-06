"""LibriSpeech test audio for the benchmark, fetched once into bench/data/.

Audio comes from the Hugging Face datasets server (openslr/librispeech_asr), which serves the
original FLAC files, so this needs neither the `datasets` library nor pyarrow. Decoding uses
ffmpeg through mlx_whisper.audio.load_audio, the same as `voice_to_text --local --file`.

Delete bench/data/ to fetch a fresh sample.
"""

import itertools
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_RATE = 16000
SPLITS = ("clean", "other")  # LibriSpeech test-clean and the harder test-other

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_DATASET = "openslr/librispeech_asr"
_ID = re.compile(r"\d+-\d+-\d+")  # speaker-chapter-utterance; also keeps file names safe
_CLIPS = ("short", "medium", "long")


def prepare(windows: int = 25, per_window: int = 4, seed: int = 0) -> None:
    """Fetch the WER sample (windows x per_window utterances per split) and build the clips."""
    for split in SPLITS:
        _sample(split, windows, per_window, seed)
    if not (DATA_DIR / "clips" / "clips.json").exists():
        _build_clips(seed)


def utterances(split: str) -> list[dict]:
    """The sampled utterances of a split, each with its 16 kHz audio and reference text."""
    entries = json.loads((DATA_DIR / "librispeech" / f"{split}.json").read_text())
    for entry in entries:
        entry["audio"] = load_audio(DATA_DIR / "librispeech" / split / entry["file"])
    return entries


def stitched(per_recording: int = 4, seed: int = 0) -> list[dict]:
    """Recordings of consecutive sampled utterances joined by pauses, like dictating a few sentences.

    The sample was fetched in windows of consecutive utterances, so each group of per_recording
    is one speaker reading on. Each recording is {"split", "id", "audio", "text"}.
    """
    rng = random.Random(f"stitched-{seed}")
    noise = np.random.default_rng(seed)
    recordings = []
    for split in SPLITS:
        entries = utterances(split)
        for i in range(0, len(entries), per_recording):
            group = entries[i : i + per_recording]
            parts = []
            for entry in group:
                if parts:
                    parts.append(_pause(rng, noise))
                parts.append(entry["audio"])
            recordings.append(
                {
                    "split": split,
                    "id": "+".join(entry["id"] for entry in group),
                    "audio": np.concatenate(parts),
                    "text": " ".join(entry["text"] for entry in group),
                }
            )
    return recordings


def _pause(rng: random.Random, noise: np.random.Generator) -> np.ndarray:
    """0.5-1.0 s of faint noise, a breath between sentences."""
    return noise.normal(0, 3e-4, int(rng.uniform(0.5, 1.0) * SAMPLE_RATE)).astype(np.float32)


def load_clips() -> dict[str, dict]:
    """short, medium and long clips: {"audio", "text", "duration_s"} each."""
    clips_dir = DATA_DIR / "clips"
    manifest = json.loads((clips_dir / "clips.json").read_text())
    clips = {}
    for name in _CLIPS:
        audio = _read_wav(clips_dir / f"{name}.wav")
        clips[name] = {"audio": audio, "text": manifest[name]["text"], "duration_s": audio.size / SAMPLE_RATE}
    return clips


def load_audio(path: Path) -> np.ndarray:
    from mlx_whisper.audio import load_audio as ffmpeg_load

    return np.asarray(ffmpeg_load(str(path)), dtype=np.float32)


# Fetching


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "voice_to_text-bench"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
        except urllib.error.URLError:
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _rows(split: str, offset: int, length: int) -> tuple[list[dict], int]:
    query = urllib.parse.urlencode(
        {"dataset": _DATASET, "config": split, "split": "test", "offset": offset, "length": length}
    )
    page = json.loads(_get(f"{_ROWS_API}?{query}"))
    return [r["row"] for r in page["rows"]], page["num_rows_total"]


def _save_rows(rows: list[dict], directory: Path) -> list[dict]:
    """Download each row's FLAC into directory and return its manifest entry."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        if not _ID.fullmatch(str(row["id"])):
            continue
        path = directory / f"{row['id']}.flac"
        if not path.exists():
            data = _get(row["audio"][0]["src"])
            if not data.startswith(b"fLaC"):
                raise ValueError(f"{row['id']}: the server didn't return a FLAC file")
            path.write_bytes(data)
        entries.append({"id": row["id"], "text": row["text"], "chapter": row["chapter_id"], "file": path.name})
    return entries


def _sample(split: str, windows: int, per_window: int, seed: int) -> None:
    """Random windows of consecutive utterances: many speakers, a fixed and repeatable set."""
    manifest = DATA_DIR / "librispeech" / f"{split}.json"
    if manifest.exists():
        return
    print(f"Fetching {windows * per_window} LibriSpeech test-{split} utterances…", file=sys.stderr, flush=True)
    _, total = _rows(split, 0, 1)
    rng = random.Random(f"{split}-{seed}")
    starts = sorted(rng.sample(range(0, total - per_window, per_window), windows))
    entries = []
    for start in starts:
        rows, _ = _rows(split, start, per_window)
        entries += _save_rows(rows, DATA_DIR / "librispeech" / split)
    manifest.write_text(json.dumps(entries, indent=1))


# Clips for the latency runs


def _build_clips(seed: int) -> None:
    print("Building the short, medium and long clips…", file=sys.stderr, flush=True)
    clean = utterances("clean")
    clips = {}
    for name, low, high, target in (("short", 3.0, 5.0, 4.0), ("medium", 13.0, 17.0, 15.0)):
        candidates = [u for u in clean if low <= u["audio"].size / SAMPLE_RATE <= high]
        if not candidates:
            raise RuntimeError(f"No {low}-{high} s utterance in the sample for the {name} clip")
        best = min(candidates, key=lambda u: abs(u["audio"].size / SAMPLE_RATE - target))
        clips[name] = {"audio": best["audio"], "text": best["text"], "ids": [best["id"]]}
    clips["long"] = _long_clip(seed)

    clips_dir = DATA_DIR / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for name, clip in clips.items():
        _write_wav(clips_dir / f"{name}.wav", clip.pop("audio"))
    (clips_dir / "clips.json").write_text(json.dumps(clips, indent=1))


def _long_clip(seed: int, low_s: float = 42.0, high_s: float = 50.0) -> dict:
    """Consecutive utterances from one chapter, joined by 0.5-1.0 s pauses: one speaker reading on."""
    rng = random.Random(f"long-{seed}")
    noise = np.random.default_rng(seed)
    _, total = _rows("clean", 0, 1)
    directory = DATA_DIR / "librispeech" / "long"
    for _ in range(20):
        rows, _ = _rows("clean", rng.randrange(0, total - 20), 20)
        chapter = rows[0]["chapter_id"]
        run = list(itertools.takewhile(lambda r: r["chapter_id"] == chapter, rows))
        parts, texts, ids, length_s = [], [], [], 0.0
        for entry in _save_rows(run, directory):
            if parts:
                pause = _pause(rng, noise)
                parts.append(pause)
                length_s += pause.size / SAMPLE_RATE
            audio = load_audio(directory / entry["file"])
            parts.append(audio)
            texts.append(entry["text"])
            ids.append(entry["id"])
            length_s += audio.size / SAMPLE_RATE
            if length_s >= low_s:
                break
        if low_s <= length_s <= high_s:
            return {"audio": np.concatenate(parts), "text": " ".join(texts), "ids": ids}
    raise RuntimeError("Couldn't find a chapter with enough consecutive speech for the long clip")


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768
