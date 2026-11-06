"""Transcription server: speech to text on the CPU, a site to use it live, saved transcripts, Claude summaries.

    .venv/bin/python server/server.py                                # 127.0.0.1:8095, large-v3-turbo
    .venv/bin/python server/server.py --model small                  # any faster-whisper model
    .venv/bin/python server/server.py --model parakeet-tdt-0.6b-v3   # NVIDIA Parakeet via onnx-asr
    .venv/bin/python server/server.py --socket /run/voice-to-text/voice.sock   # a root-only Unix socket

GET    /                      the site: record in the browser, watch the text appear, read the summary
GET    /live                  WebSocket the site streams the microphone to (see LiveSession)
POST   /transcribe            body: an audio file in any format PyAV reads (wav, m4a, mp3, webm, ogg, flac)
                              reply: {"id", "text", "audio_s", "speech_s", "seconds", "summary_status"}
GET    /api/transcripts       the latest 100, newest first, with their summaries
GET    /api/transcripts/<id>
DELETE /api/transcripts/<id>
GET    /health                {"ok": true, "model", "summaries", ...}

    curl --data-binary @clip.wav http://127.0.0.1:8095/transcribe

Every transcript is saved in data/transcripts.db, and one with text gets a summary from
Claude in the background when ANTHROPIC_API_KEY is set (summary.py).

Whisper decodes like the Mac app (src/voice_to_text/stt.py): English, greedy, the same VAD
settings, the same shortened encoder windows for short clips (--full-window turns them off),
re-decodes in the full window and then at temperature 0.4 when a result looks like a
hallucination, and the same filter for what Whisper "hears" in a moment of noise. Live audio
is cut at pauses by the app's Segmenter (live.py).

It only listens on this machine: a Unix socket readable by its owner alone (for an SSH
tunnel), and optionally --public-port on 127.0.0.1 behind Google sign-in through Cloudflare
Access (auth.py), which a Cloudflare tunnel publishes as a subdomain.
"""

import argparse
import asyncio
import difflib
import io
import ipaddress
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ctranslate2
import numpy as np
from aiohttp import WSMsgType, web
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from auth import AUTH, OWNER, SiteAuth, guard, logout, security_headers
from live import Resampler, Segmenter
from store import Store
from summary import MODEL as SUMMARY_MODEL
from summary import Summarizer, SummaryError

log = logging.getLogger("voice_to_text")

SAMPLE_RATE = 16000
MAX_BODY_BYTES = 50 * 1024 * 1024
MAX_CHUNK_S = 25  # as in stt.py: speech is split at pauses into parts of at most this

PAD_S = 0.2  # default --pad: audio kept around the speech, as stt.py


def _vad_options(pad_s: float) -> VadOptions:
    return VadOptions(
        threshold=0.5,
        min_speech_duration_ms=200,
        min_silence_duration_ms=500,
        speech_pad_ms=int(pad_s * 1000),
        max_speech_duration_s=MAX_CHUNK_S,
    )


_VAD = _vad_options(PAD_S)  # stt.py's settings

# What Whisper tends to "hear" in silence or noise (stt.py's list).
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye"}


# As in stt.py: (longest clip, window), else the full 30 s. Whisper is trained on 30 s
# windows, but most of its time goes to encoding the window, silence included. On this
# server's 2 cores large-v3-turbo's encoder takes 10.9 s for 30 s, 5.2 s for 20 s and
# 1.7 s for 10 s. faster-whisper's transcribe() always pads to 30 s, so _decode() calls
# its CTranslate2 model directly, the way faster-whisper's generate_with_fallback() does.
_WINDOWS_S = ((6, 10), (16, 20))


def _fine_window(duration_s: float) -> int:
    """--windows fine: the clip plus 4 s, in whole even seconds, 10 to 30.

    stt.py's steps leave at least 4 s after the speech too, but a 7 s clip gets a 20 s
    window there and a 12 s one here (encoder 2.1 s instead of 5.1 s on this server).
    """
    return min(30, max(10, 2 * -(-int(np.ceil(duration_s + 4)) // 2)))
_MAX_LENGTH = 448  # Whisper's decoder context, as faster-whisper's WhisperModel.max_length


@dataclass
class _Decoded:
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    def looks_wrong(self) -> bool:
        """Repetition or low confidence, the usual signs of a Whisper hallucination (stt.py)."""
        if self.compression_ratio > 2.4 or self.avg_logprob < -1.0:
            return True
        sentences = [s.lower() for s in re.split(r"(?<=[.!?])\s+", self.text) if s]
        return len(sentences) > 1 and len(set(sentences)) == 1


PARAKEET = "parakeet-tdt-0.6b-v3"
PARAKEETS = ("parakeet-tdt-0.6b-v2", PARAKEET)  # v2: English only, half v3's word errors here (see _Parakeet)


class Transcriber:
    """One model, one transcription at a time: on 2 cores, parallel runs only slow each other."""

    def __init__(self, model: str, threads: int, short_windows: bool = True, fine_windows: bool = False,
                 pad_s: float = PAD_S):
        self.model_name = model
        self.pad_s = pad_s  # audio kept around the speech in each part, uploads and live alike
        self._vad = _vad_options(pad_s)
        if model in PARAKEETS:
            self._engine = _Parakeet(model, threads)
            self.short_windows = None  # Parakeet encodes the audio it's given, no fixed window
        else:
            self._engine = _Whisper(model, threads, short_windows, fine_windows)
            self.short_windows = short_windows
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        with self._lock:
            self._engine.warm_up()

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Text for a 16 kHz mono clip ("" if there's no speech in it) and seconds of speech.

        Like stt.py, each part of at most MAX_CHUNK_S is decoded on its own. Letting
        faster-whisper join the speech and read it in 30 s windows instead lost a whole
        sentence at a window boundary of the 47 s benchmark clip.
        """
        speech = get_speech_timestamps(audio, self._vad)
        if not speech:
            return "", 0.0
        speech_s = sum(s["end"] - s["start"] for s in speech) / SAMPLE_RATE
        clip = audio[speech[0]["start"] : speech[-1]["end"]]
        if clip.size <= MAX_CHUNK_S * SAMPLE_RATE:
            chunks = [clip]
        else:
            chunks, _ = collect_chunks(audio, speech, max_duration=MAX_CHUNK_S)
        with self._lock:
            text = " ".join(filter(None, (self._engine.transcribe_chunk(chunk) for chunk in chunks)))
        return self._unless_hallucination(text, speech_s), speech_s

    def transcribe_part(self, audio: np.ndarray, speech_s: float) -> str:
        """Text for one part the live Segmenter cut: at most MAX_CHUNK_S, speech already found."""
        with self._lock:
            text = self._engine.transcribe_chunk(audio)
        return self._unless_hallucination(text, speech_s)

    def _unless_hallucination(self, text: str, speech_s: float) -> str:
        """"" for what Whisper tends to hear in a moment of noise (stt.py). Parakeet doesn't do that."""
        if isinstance(self._engine, _Whisper) and speech_s < 1.0 and text.lower().strip(" .!?,") in _HALLUCINATIONS:
            return ""
        return text


class _Parakeet:
    """NVIDIA's Parakeet-TDT-0.6B (v2 or v3), int8 ONNX conversions run by onnx-asr on onnxruntime.

    v3: English plus 24 European languages; v2: English only. Both write punctuation, neither
    knows Hindi. 2026-11-06 on this server, 50 + 50 LibriSpeech test-clean / test-other
    utterances, word errors: v3 2.73% / 6.84%, v3 with --pad 0.4 2.27% / 5.45%,
    v2 1.29% / 4.40%, v2 with --pad 0.4 1.36% / 4.06% (0.6 was worse); median 0.7-0.8 s.
    v3 took 0.4 s for the 4 s clip, 1.2 s for 15 s and 4.4 s for 47 s. The fp32 versions
    won't load from the Hugging Face cache (onnxruntime 1.31 refuses their external weights
    file) and would need ~3 GB of RAM.
    """

    def __init__(self, model: str, threads: int):
        import onnx_asr  # only this engine needs it
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._model = onnx_asr.load_model(
            "nemo-" + model, quantization="int8", sess_options=options, providers=["CPUExecutionProvider"]
        )

    def warm_up(self) -> None:
        self._model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)

    def transcribe_chunk(self, chunk: np.ndarray) -> str:
        return self._model.recognize(chunk, sample_rate=SAMPLE_RATE).strip()


class _Whisper:
    def __init__(self, model: str, threads: int, short_windows: bool, fine_windows: bool = False):
        self.short_windows = short_windows  # False: always encode Whisper's full 30 s window
        self.fine_windows = fine_windows  # True: _fine_window() instead of stt.py's two steps
        self._model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        self._whisper = self._model.model
        self._tokenizer = Tokenizer(self._model.hf_tokenizer, self._whisper.is_multilingual, task="transcribe", language="en")
        self._prompt = self._model.get_prompt(self._tokenizer, [], without_timestamps=True)
        self._suppress = list(get_suppressed_tokens(self._tokenizer, [-1]))

    def warm_up(self) -> None:
        """Decode a second of near-silence in each window size, so no real request is the first."""
        noise = np.random.default_rng(0).normal(0, 1e-3, SAMPLE_RATE).astype(np.float32)
        features = self._model.feature_extractor(noise)
        for window_s in (10, 20, 30) if self.short_windows else (30,):
            self._decode(features, window_s)

    def transcribe_chunk(self, chunk: np.ndarray) -> str:
        """stt.py's steps: the shortest window that fits, then the full window, then
        temperature 0.4, until a result doesn't look like a hallucination."""
        duration_s = chunk.size / SAMPLE_RATE
        window_s = 30
        if self.short_windows:
            if self.fine_windows:
                window_s = _fine_window(duration_s)
            else:
                window_s = next((window for longest, window in _WINDOWS_S if duration_s <= longest), 30)
        features = self._model.feature_extractor(chunk)
        result = self._decode(features, window_s)
        if window_s < 30 and result.looks_wrong():
            result = self._decode(features, 30)
        if result.looks_wrong():
            result = self._decode(features, 30, temperature=0.4)
        if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
            return ""
        return result.text

    def _decode(self, features: np.ndarray, window_s: int, temperature: float = 0.0) -> _Decoded:
        mel = np.ascontiguousarray(pad_or_trim(features, window_s * 100)[None], dtype=np.float32)  # 100 frames a second
        encoder_output = self._whisper.encode(ctranslate2.StorageView.from_array(mel), to_cpu=False)
        if temperature > 0:
            sampling = dict(beam_size=1, num_hypotheses=1, sampling_topk=0, sampling_temperature=temperature)
        else:
            sampling = dict(beam_size=1)
        result = self._whisper.generate(
            encoder_output,
            [self._prompt],
            max_length=_MAX_LENGTH,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=True,
            suppress_tokens=self._suppress,
            **sampling,
        )[0]
        tokens = result.sequences_ids[0]
        text = self._tokenizer.decode(tokens).strip()
        return _Decoded(
            text=text,
            # faster-whisper's: the score is length-normalised (length_penalty 1), make it a mean
            avg_logprob=result.scores[0] * len(tokens) / (len(tokens) + 1),
            compression_ratio=get_compression_ratio(text) if text else 0.0,
            no_speech_prob=result.no_speech_prob,
        )


WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARTIAL_EVERY_S = 0.8  # default --interim-every: how often, at most, the site gets the part still being spoken


class App:
    """What the request handlers share: the model, the transcripts, the summaries."""

    def __init__(self, transcriber: Transcriber, store: Store, summarizer: Summarizer,
                 interim_every_s: float = PARTIAL_EVERY_S, max_part_s: float = MAX_CHUNK_S, min_part_s: float = 0.0):
        self.interim_every_s = interim_every_s  # 0: no interim text, every model run goes to finished parts
        self.max_part_s = max_part_s  # live parts longer than this are cut at their latest short pause
        self.min_part_s = min_part_s  # live: speech a part needs before a pause can end it
        self.transcriber = transcriber
        self.store = store
        self.summarizer = summarizer
        self._model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
        self.model_jobs = 0  # queued or running on the model thread
        self._tasks: set[asyncio.Task] = set()

    async def run_model(self, fn, *args):
        """Run fn on the one model thread, so the event loop keeps serving while it works."""
        self.model_jobs += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._model_thread, fn, *args)
        finally:
            self.model_jobs -= 1

    def save(self, *, source: str, mode: str, owner: str | None, audio_s: float, speech_s: float, seconds: float,
             text: str) -> tuple[int, str]:
        """(id, summary status) of the newly saved transcript."""
        status = "pending" if text and self.summarizer.enabled else ("off" if text else "none")
        transcript_id = self.store.add(
            source=source, mode=mode, owner=owner, model=self.transcriber.model_name, audio_s=audio_s,
            speech_s=speech_s, seconds=seconds, text=text, summary_status=status,
        )
        return transcript_id, status

    async def summarize(self, transcript_id: int, text: str, mode: str = "dictation") -> dict | None:
        """Write the summary into the store; returns the updated transcript (None if it was deleted)."""
        started = time.monotonic()
        try:
            summary, model = await self.summarizer.summarize(text, mode)
            self.store.set_summary(transcript_id, status="done", summary=summary, model=model)
            log.info("summary of transcript %d by %s in %.1f s, %d chars", transcript_id, model,
                     time.monotonic() - started, len(summary))
        except SummaryError as e:
            self.store.set_summary(transcript_id, status="error", error=str(e))
            log.warning("summary of transcript %d failed: %s", transcript_id, e)
        except Exception:
            self.store.set_summary(transcript_id, status="error", error="Unexpected error, see the server log")
            log.exception("summary of transcript %d failed", transcript_id)
        return self.store.get(transcript_id)

    def summarize_later(self, transcript_id: int, text: str, mode: str = "dictation") -> None:
        task = asyncio.create_task(self.summarize(transcript_id, text, mode))
        self._tasks.add(task)  # asyncio keeps only a weak reference to running tasks
        task.add_done_callback(self._tasks.discard)


MEETING_SPEAKERS = ("You", "Meeting")  # channel 0: the microphone; channel 1: the shared tab's audio


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


ECHO_RUN = 3  # words in a row that the meeting also said at that moment: its sound, heard by the microphone
ECHO_SLACK_S = 1.0  # how far apart in time the two copies may be


def _without_echo(text: str, meeting: list[str]) -> str:
    """text minus the runs of ECHO_RUN+ words the meeting said around the same time.

    With speakers instead of headphones the microphone hears the meeting too, so a "You" part
    is often the meeting's words with the speaker's own mixed in. Dropping the whole part
    (the first version) lost the speaker's words as well. Returns "" when what is left is
    under ECHO_RUN words.
    """
    words = text.split()
    other = re.findall(r"[a-z0-9']+", " ".join(meeting).lower())
    if not words or not other:
        return text
    plain = [re.sub(r"[^a-z0-9']", "", w.lower()) for w in words]
    echo = [False] * len(words)
    for block in difflib.SequenceMatcher(a=plain, b=other, autojunk=False).get_matching_blocks():
        if block.size >= ECHO_RUN:
            echo[block.a : block.a + block.size] = [True] * block.size
    if not any(echo):
        return text
    kept = [w for w, e in zip(words, echo) if not e]
    return " ".join(kept) if len(kept) >= ECHO_RUN else ""


class _Channel:
    """One audio stream of a recording, with its own resampler, cuts and interim text."""

    def __init__(self, app: "App", name: str | None, rate: int):
        self.name = name  # None in dictation, else a MEETING_SPEAKERS entry
        self.resampler = Resampler(rate)
        self.segmenter = Segmenter(max_s=app.max_part_s, min_part_s=app.min_part_s, pad_s=app.transcriber.pad_s)
        self.cuts = 0  # parts cut so far; a partial for an older open part is stale
        self.samples = 0  # at 16 kHz
        self.peak = 0.0
        self.last_partial = 0.0


class LiveSession:
    """One recording from the site, over its own WebSocket.

    The site sends {"type": "start", "rate": <Hz>, "mode": "dictation" | "meeting"}, then
    binary messages of 16-bit little-endian PCM at that rate, then {"type": "stop"}. In
    dictation a message is the microphone; in a meeting it is two equal halves, the
    microphone and then the shared tab's audio over the same moment (the site records both
    in one AudioContext, so they line up). The server sends {"type": "partial", "text",
    "speaker"} with a part still being spoken (at most every --interim-every, when the model
    is free), {"type": "segment", "text", "speaker", "start"} each time a pause completes a
    part, {"type": "done", "id", "text", ...} after stop, and finally {"type": "summary", ...}
    once Claude has summarized it. speaker is null in dictation, "You" or "Meeting" in a meeting;
    {"type": "revise", "speaker": "You", "start", "text"} replaces a "You" segment once part of it
    turns out to be the meeting's echo ("" removes it).

    A meeting's text is one line per turn, "[m:ss] You: ..." and "[m:ss] Meeting: ...",
    in time order. With laptop speakers instead of headphones the microphone hears the
    meeting too; runs of words a "You" part shares with the meeting around the same moment
    are taken out of it as that echo (_without_echo).
    """

    def __init__(self, app: App, ws: web.WebSocketResponse, rate: int, mode: str = "dictation",
                 owner: str | None = None):
        self.app = app
        self.ws = ws
        self.mode = mode
        self.owner = owner  # the Google email on the subdomain, None over the SSH tunnel
        names = MEETING_SPEAKERS if mode == "meeting" else (None,)
        self._channels = [_Channel(app, name, rate) for name in names]
        self._parts: asyncio.Queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._transcribe_parts())
        self._results: list[tuple[float, float, int, str]] = []  # (start s, end s, channel, text)
        self._shown: dict[float, str] = {}  # "You" parts as the page shows them, echo removed, by start
        self._echo_trimmed = 0  # "You" parts that lost words to the echo filter (of which some lost all)
        self._echo_dropped = 0
        self._speech_s = 0.0
        self._partial: asyncio.Task | None = None
        self.stopped = False

    def feed(self, data: bytes) -> None:
        """Audio from the site. Resampling and the VAD take ~1 ms per 100 ms of audio: fine on the loop."""
        pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        block = pcm.size // len(self._channels)
        for i, channel in enumerate(self._channels):
            samples = pcm[i * block : (i + 1) * block]
            if samples.size:
                channel.peak = max(channel.peak, float(np.abs(samples).max()))
            self._add(i, channel.resampler.push(samples))
        self._maybe_send_partial()

    async def stop(self) -> None:
        """Transcribe what's left, save the transcript, send it, then the summary."""
        self.stopped = True
        stopped_at = time.monotonic()
        for i, channel in enumerate(self._channels):
            self._add(i, channel.resampler.flush())
            for segment in channel.segmenter.flush():
                self._queue(i, segment)
        self._parts.put_nowait(None)
        await self._worker
        text = self._text()
        seconds = time.monotonic() - stopped_at
        audio_s = self._channels[0].samples / SAMPLE_RATE
        silent = [c.name or "microphone" for c in self._channels if c.samples > 0 and c.peak < 1e-4]
        log.info("live %s: %.1f s of audio in %d parts, %d chars, last text %.2f s after stop",
                 self.mode, audio_s, sum(c.cuts for c in self._channels), len(text), seconds)
        if self.mode == "meeting":  # numbers only, never words: enough to tell a quiet microphone from the echo filter
            for i, c in enumerate(self._channels):
                said = [t for _, _, ch, t in self._results if ch == i]
                log.info("live meeting %s: peak %.3f, %d parts cut, %d with text, %d words",
                         c.name, c.peak, c.cuts, len(said), sum(len(t.split()) for t in said))
            log.info("live meeting echo filter: %d You parts trimmed, %d of them dropped", self._echo_trimmed, self._echo_dropped)
        done = {"type": "done", "id": None, "text": text, "mode": self.mode, "audio_s": round(audio_s, 2),
                "speech_s": round(self._speech_s, 2), "seconds": round(seconds, 2),
                "silent": len(silent) == len(self._channels), "silent_channels": silent, "summary_status": "none"}
        if not text:  # nothing was said: nothing to keep
            await self._send(done)
            return
        transcript_id, status = self.app.save(
            source="live", mode=self.mode, owner=self.owner, audio_s=audio_s, speech_s=self._speech_s,
            seconds=seconds, text=text
        )
        await self._send({**done, "id": transcript_id, "summary_status": status})
        if status == "pending":
            transcript = await self.app.summarize(transcript_id, text, self.mode)
            if transcript:
                await self._send({"type": "summary", **_public(transcript)})

    def _meeting_around(self, start: float, end: float) -> list[str]:
        return [t for s, e, ch, t in sorted(self._results)
                if ch == 1 and s < end + ECHO_SLACK_S and start - ECHO_SLACK_S < e]

    def _text(self) -> str:
        results = sorted(self._results)
        if self.mode != "meeting":
            return " ".join(text for _, _, _, text in results)
        lines: list[list] = []  # [start, speaker, text]
        for start, end, channel, text in results:
            if channel == 0:
                text = _without_echo(text, self._meeting_around(start, end))
                if not text:
                    continue
            speaker = MEETING_SPEAKERS[channel]
            if lines and lines[-1][1] == speaker:
                lines[-1][2] += " " + text
            else:
                lines.append([start, speaker, text])
        return "\n".join(f"[{_clock(start)}] {speaker}: {text}" for start, speaker, text in lines)

    def _add(self, index: int, audio: np.ndarray) -> None:
        channel = self._channels[index]
        channel.samples += audio.size
        if audio.size:
            for segment in channel.segmenter.push(audio):
                self._queue(index, segment)

    def _queue(self, index: int, segment) -> None:
        self._channels[index].cuts += 1
        self._speech_s += segment.speech_s
        self._parts.put_nowait((index, segment))

    async def _transcribe_parts(self) -> None:
        while (item := await self._parts.get()) is not None:
            index, segment = item
            try:
                text = await self.app.run_model(self.app.transcriber.transcribe_part, segment.audio, segment.speech_s)
            except Exception:
                log.exception("live: a part failed to transcribe")
                continue
            if not text:
                continue
            start, end = segment.start / SAMPLE_RATE, segment.end / SAMPLE_RATE
            self._results.append((start, end, index, text))
            if self.mode != "meeting":
                await self._send({"type": "segment", "text": text, "speaker": None, "start": round(start, 2)})
            elif index == 0:  # "You": without what the microphone heard of the meeting
                shown = _without_echo(text, self._meeting_around(start, end))
                self._count_echo(text, shown)
                self._shown[round(start, 2)] = shown
                if shown:
                    await self._send({"type": "segment", "text": shown, "speaker": "You", "start": round(start, 2)})
            else:  # "Meeting": and take its echo out of the "You" parts already shown
                await self._send({"type": "segment", "text": text, "speaker": "Meeting", "start": round(start, 2)})
                for s, e, ch, t in self._results:
                    key = round(s, 2)
                    if ch != 0 or not (s < end + ECHO_SLACK_S and start - ECHO_SLACK_S < e):
                        continue
                    shown = _without_echo(t, self._meeting_around(s, e))
                    if shown != self._shown.get(key):
                        self._count_echo(self._shown.get(key, t), shown)
                        self._shown[key] = shown
                        await self._send({"type": "revise", "speaker": "You", "start": key, "text": shown})

    def _count_echo(self, before: str, after: str) -> None:
        if after != before:
            self._echo_trimmed += 1
            self._echo_dropped += not after

    def _maybe_send_partial(self) -> None:
        now = time.monotonic()
        every = self.app.interim_every_s
        if not every or self._partial or self.app.model_jobs:
            return
        # The channel whose interim text is oldest, among those with a part being spoken.
        for index in sorted(range(len(self._channels)), key=lambda i: self._channels[i].last_partial):
            channel = self._channels[index]
            if now - channel.last_partial < every:
                continue
            part = channel.segmenter.open_part()
            if part is None:
                continue
            channel.last_partial = now
            self._partial = asyncio.create_task(self._send_partial(index, part, channel.cuts))
            return

    async def _send_partial(self, index: int, part: tuple[np.ndarray, float], cuts: int) -> None:
        channel = self._channels[index]
        try:
            text = await self.app.run_model(self.app.transcriber.transcribe_part, *part)
            # Only while its part is still the one being spoken; an empty one would just blank the page's text.
            if text and cuts == channel.cuts and not self.stopped:
                await self._send({"type": "partial", "text": text, "speaker": channel.name})
        except Exception:
            log.exception("live: an interim transcript failed")
        finally:
            self._partial = None

    async def _send(self, payload: dict) -> None:
        if self.ws.closed:
            return
        try:
            await self.ws.send_json(payload)
        except (ConnectionResetError, RuntimeError):
            pass  # the page went away; the transcript is saved regardless


def _public(transcript: dict) -> dict:
    """A transcript as the API returns it."""
    return {key: transcript[key] for key in (
        "id", "created_at", "source", "mode", "owner", "model", "audio_s", "speech_s", "seconds", "text",
        "summary", "summary_status", "summary_model", "summary_error",
    )}


APP = web.AppKey("app", App)


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


async def health(request: web.Request) -> web.Response:
    app = request.app[APP]
    return web.json_response({
        "ok": True,
        "model": app.transcriber.model_name,
        "short_windows": app.transcriber.short_windows,
        "summaries": app.summarizer.enabled,
        "summary_model": SUMMARY_MODEL if app.summarizer.enabled else None,
        "public": request.app[AUTH].is_public(request),  # signed in through the subdomain: the page offers Sign out
        "user": request.get(OWNER),  # the Google email, on the subdomain
    })


async def transcribe(request: web.Request) -> web.Response:
    app = request.app[APP]
    mode = "meeting" if request.query.get("mode") == "meeting" else "dictation"  # a recorded meeting: its summary
    body = await request.read()  # the Application's client_max_size answers 413 above MAX_BODY_BYTES
    if not body:
        return web.json_response({"error": "send the audio file as the request body"}, status=400)
    started = time.monotonic()
    try:
        audio = await asyncio.get_running_loop().run_in_executor(None, decode_audio, io.BytesIO(body), SAMPLE_RATE)
    except Exception as e:
        return web.json_response({"error": f"couldn't read the audio: {e}"}, status=400)
    try:
        text, speech_s = await app.run_model(app.transcriber.transcribe, audio)
    except Exception:
        log.exception("transcription failed")
        return web.json_response({"error": "transcription failed"}, status=500)
    seconds = time.monotonic() - started
    audio_s = audio.size / SAMPLE_RATE
    log.info("upload: transcribed %.1f s of audio in %.2f s, %d chars", audio_s, seconds, len(text))  # never the text
    transcript_id, status = None, "none"
    if text:  # as with live recordings, nothing said means nothing to keep
        transcript_id, status = app.save(source="upload", mode=mode, owner=request[OWNER], audio_s=audio_s,
                                         speech_s=speech_s, seconds=seconds, text=text)
    if status == "pending":
        app.summarize_later(transcript_id, text, mode)
    return web.json_response({
        "id": transcript_id, "text": text, "audio_s": round(audio_s, 2), "speech_s": round(speech_s, 2),
        "seconds": round(seconds, 2), "summary_status": status,
    })


async def list_transcripts(request: web.Request) -> web.Response:
    return web.json_response({"transcripts": [_public(t) for t in request.app[APP].store.recent(request[OWNER])]})


def _transcript_id(request: web.Request) -> int:
    try:
        return int(request.match_info["id"])
    except ValueError:
        raise web.HTTPNotFound() from None


async def get_transcript(request: web.Request) -> web.Response:
    transcript = request.app[APP].store.get(_transcript_id(request), request[OWNER])
    if transcript is None:
        return web.json_response({"error": "no such transcript"}, status=404)
    return web.json_response(_public(transcript))


async def delete_transcript(request: web.Request) -> web.Response:
    if not request.app[APP].store.delete(_transcript_id(request), request[OWNER]):
        return web.json_response({"error": "no such transcript"}, status=404)
    return web.json_response({"deleted": True})


async def live(request: web.Request) -> web.WebSocketResponse:
    app = request.app[APP]
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=1 << 20)
    await ws.prepare(request)
    session: LiveSession | None = None
    try:
        async for message in ws:
            if message.type == WSMsgType.BINARY:
                if session and not session.stopped:
                    session.feed(message.data)
            elif message.type == WSMsgType.TEXT:
                try:
                    data = json.loads(message.data)
                except ValueError:
                    continue
                if data.get("type") == "start" and session is None:
                    rate = int(data.get("rate", SAMPLE_RATE))
                    if not 8000 <= rate <= 192000:
                        await ws.send_json({"type": "error", "error": f"unsupported sample rate {rate}"})
                        break
                    mode = "meeting" if data.get("mode") == "meeting" else "dictation"
                    session = LiveSession(app, ws, rate, mode, request[OWNER])
                    await ws.send_json({"type": "started", "model": app.transcriber.model_name,
                                        "summaries": app.summarizer.enabled})
                elif data.get("type") == "stop" and session and not session.stopped:
                    await session.stop()
                    break
    finally:
        if session and not session.stopped:  # the page closed mid-recording: keep what was said
            await session.stop()
        await ws.close()
    return ws


async def _resume_summaries(application: web.Application) -> None:
    """Summaries a restart interrupted, run again."""
    app = application[APP]
    for transcript_id in app.store.stale_pending():
        transcript = app.store.get(transcript_id)
        if app.summarizer.enabled:
            app.summarize_later(transcript_id, transcript["text"], transcript["mode"])
        else:
            app.store.set_summary(transcript_id, status="off")


def make_app(app: App, site_auth: SiteAuth | None = None) -> web.Application:
    application = web.Application(client_max_size=MAX_BODY_BYTES, middlewares=[guard])
    application[APP] = app
    application[AUTH] = site_auth or SiteAuth(None)
    application.on_startup.append(_resume_summaries)
    application.on_response_prepare.append(security_headers)
    application.add_routes([
        web.post("/logout", logout),
        web.get("/", index),
        web.get("/health", health),
        web.get("/live", live),
        web.post("/transcribe", transcribe),
        web.get("/api/transcripts", list_transcripts),
        web.get("/api/transcripts/{id}", get_transcript),
        web.delete("/api/transcripts/{id}", delete_transcript),
        web.static("/static", WEB_DIR),
    ])
    return application


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a hostname: it could resolve to anything
        return False


async def _serve(application: web.Application, host: str, port: int | None, socket_path: str | None,
                 public_port: int | None = None) -> None:
    runner = web.AppRunner(application, access_log=None)  # our own log lines carry lengths, never text
    await runner.setup()
    if socket_path:
        old_umask = os.umask(0o177)  # the socket is born rw------- : only its owner can connect
        try:
            await web.UnixSite(runner, socket_path).start()
        finally:
            os.umask(old_umask)
        print(f"Listening on unix:{socket_path}", flush=True)
    if port is not None:
        await web.TCPSite(runner, host, port).start()
        print(f"Listening on http://{host}:{port}", flush=True)
    if public_port is not None:
        await web.TCPSite(runner, host, public_port).start()
        state = "Google sign-in set up" if application[AUTH].configured else "NO GOOGLE SIGN-IN SET UP: refusing every request"
        print(f"Listening on http://{host}:{public_port} for the subdomain ({state})", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="a loopback address (default: %(default)s)")
    parser.add_argument("--port", type=int, help="TCP port (default: 8095, unless --socket is given)")
    parser.add_argument("--socket", help="also or instead listen on this Unix socket, readable by its owner only")
    parser.add_argument("--public-port", type=int, metavar="PORT",
                        help="also listen on this TCP port behind Google sign-in, for a Cloudflare tunnel (see auth.py)")
    parser.add_argument(
        "--model", default="large-v3-turbo",
        help=f"a faster-whisper model name, or {' or '.join(PARAKEETS)} (default: %(default)s)"
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="CPU threads (default: %(default)s)")
    parser.add_argument("--full-window", action="store_true", help="always encode Whisper's full 30 s window: slower")
    parser.add_argument("--windows", choices=("stt", "fine"), default="stt",
                        help="Whisper's shortened windows: stt.py's 10/20/30 s steps, or speech + 4 s (default: %(default)s)")
    parser.add_argument("--interim-every", type=float, default=PARTIAL_EVERY_S, metavar="SECONDS",
                        help="live: interim text of the part being spoken at most this often; 0 turns it off (default: %(default)s)")
    parser.add_argument("--pad", type=float, default=PAD_S, metavar="SECONDS",
                        help="audio kept around the speech in each part (default: %(default)s)")
    parser.add_argument("--min-part", type=float, default=0.0, metavar="SECONDS",
                        help="live: speech a part needs before a pause can end it; 0 cuts at every pause (default: %(default)s)")
    parser.add_argument("--max-part", type=float, default=MAX_CHUNK_S, metavar="SECONDS",
                        help="live: cut a part this long at its latest short pause (default: %(default)s)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="where transcripts.db lives (default: %(default)s)")
    args = parser.parse_args()
    if not _is_loopback(args.host):
        print(f"Refusing to listen on {args.host}: use 127.0.0.1 and an SSH tunnel.", file=sys.stderr)
        return 2
    port = args.port if args.port is not None or args.socket or args.public_port else 8095
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading {args.model}…", flush=True)
    transcriber = Transcriber(args.model, args.threads, short_windows=not args.full_window,
                              fine_windows=args.windows == "fine", pad_s=args.pad)
    transcriber.warm_up()
    summarizer = Summarizer()
    print(f"Summaries: {'on, ' + SUMMARY_MODEL if summarizer.enabled else 'off (no ANTHROPIC_API_KEY)'}", flush=True)
    store = Store(args.data / "transcripts.db")
    try:
        app = App(transcriber, store, summarizer, interim_every_s=args.interim_every,
                  max_part_s=args.max_part, min_part_s=args.min_part)
        site_auth = SiteAuth(args.public_port)
        asyncio.run(_serve(make_app(app, site_auth), args.host, port, args.socket, args.public_port))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
