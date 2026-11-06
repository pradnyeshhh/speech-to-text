"""Transcribe a recording while it's still being made.

The audio is cut at pauses as it arrives, and each finished part is transcribed right away. When
the key is released, only the part after the last pause is left to transcribe.
"""

import queue
import threading
import time
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .silero_vad import get_vad_model
from .stt import MAX_CHUNK_S, SAMPLE_RATE

PAUSE_S = 0.5  # silence that ends a part; shorter pauses, inside a sentence, don't
MIN_PART_S = 0.0  # speech a part needs before a pause can end it
FULL_WINDOW_WHILE_TALKING = False  # parts finished before release use Whisper's full 30 s window
WINDOW = 512  # samples per Silero VAD decision at 16 kHz (32 ms)
_CONTEXT = 64  # samples of the previous window Silero sees with each window


class StreamResampler:
    """Resamples audio to 16 kHz as it arrives, matching resample_poly on the whole recording."""

    def __init__(self, rate: int):
        divisor = gcd(SAMPLE_RATE, rate)
        self.up, self.down = SAMPLE_RATE // divisor, rate // divisor
        # resample_poly's filter reaches 10 * max(up, down) upsampled samples to each side.
        reach = -(-10 * max(self.up, self.down) // self.up) + 1  # in input samples
        self._margin = -(-reach // self.down) * self.down  # rounded up to a multiple of down
        self._buffer = np.zeros(0, dtype=np.float32)
        self._start = 0  # input index of _buffer[0], always a multiple of down
        self._done = 0  # output samples produced so far

    def push(self, samples: np.ndarray) -> np.ndarray:
        samples = samples.astype(np.float32, copy=False)
        if self.up == self.down:
            return samples
        self._buffer = np.concatenate([self._buffer, samples])
        received = self._start + self._buffer.size
        # Outputs whose filter lies entirely inside the input received so far.
        return self._produce(max(0, (received - self._margin) * self.up // self.down))

    def flush(self) -> np.ndarray:
        """The rest of the output, now that the input has ended."""
        if self.up == self.down:
            return np.zeros(0, dtype=np.float32)
        received = self._start + self._buffer.size
        return self._produce(-(-received * self.up // self.down))  # resample_poly's output length

    def _produce(self, until: int) -> np.ndarray:
        if until <= self._done:
            return np.zeros(0, dtype=np.float32)
        out = resample_poly(self._buffer, self.up, self.down).astype(np.float32)
        first = self._start * self.up // self.down  # output index of out[0]
        chunk = out[self._done - first : until - first]
        self._done = until
        # Keep the input the next outputs' filter reaches back to, cut at a multiple of down.
        keep_from = max(0, self._done * self.down // self.up - self._margin)
        keep_from -= keep_from % self.down
        if keep_from > self._start:
            self._buffer = self._buffer[keep_from - self._start :]
            self._start = keep_from
        return chunk


class StreamingVad:
    """Silero VAD on audio arriving in pieces, with the same result as on the whole clip."""

    def __init__(self):
        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Speech probabilities for each complete 512-sample window."""
        self._pending = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        count = self._pending.size // WINDOW
        if not count:
            return np.zeros(0, dtype=np.float32)
        windows = self._pending[: count * WINDOW].reshape(count, WINDOW)
        self._pending = self._pending[count * WINDOW :]
        return self._run(windows)

    def flush(self) -> np.ndarray:
        """The last, zero-padded window, as get_speech_timestamps pads a whole clip."""
        window = np.pad(self._pending, (0, WINDOW - self._pending.size))
        self._pending = np.zeros(0, dtype=np.float32)
        return self._run(window[None])

    def _run(self, windows: np.ndarray) -> np.ndarray:
        contexts = np.vstack([self._context[None], windows[:-1, -_CONTEXT:]])
        self._context = windows[-1, -_CONTEXT:].copy()
        inputs = {"input": np.concatenate([contexts, windows], axis=1), "h": self._h, "c": self._c}
        probabilities, self._h, self._c = self._session.run(None, inputs)
        return probabilities


@dataclass
class Segment:
    audio: np.ndarray  # 16 kHz, with up to pad_s of the surrounding silence
    start: int  # where the audio sits in the recording, in 16 kHz samples
    end: int
    speech_s: float  # its speech, without the padding


class Segmenter:
    """Cuts 16 kHz audio into parts as it arrives.

    A part ends once it has min_part_s of speech and its speech is followed by pause_s of
    silence. One that would outgrow max_s is cut early, at its latest pause of at least
    short_pause_s if it had one. The thresholds and padding follow get_speech_timestamps,
    which voice_to_text uses on whole files.
    """

    def __init__(
        self,
        vad=None,
        pause_s: float = PAUSE_S,
        min_part_s: float = MIN_PART_S,
        max_s: float = MAX_CHUNK_S,
        pad_s: float = 0.2,
        min_speech_s: float = 0.2,
        short_pause_s: float = 0.1,
        threshold: float = 0.5,
    ):
        self._vad = vad if vad is not None else StreamingVad()
        self._threshold = threshold
        self._neg_threshold = max(threshold - 0.15, 0.01)
        self._pause = int(pause_s * SAMPLE_RATE)
        self._min_part = int(min_part_s * SAMPLE_RATE)
        self._pad = int(pad_s * SAMPLE_RATE)
        self._max_speech = int(max_s * SAMPLE_RATE) - WINDOW - 2 * self._pad
        self._min_speech = int(min_speech_s * SAMPLE_RATE)
        self._short_pause = int(short_pause_s * SAMPLE_RATE)
        self._audio = np.zeros(0, dtype=np.float32)
        self._offset = 0  # recording index of _audio[0]
        self._position = 0  # where the next VAD window starts
        self._speech_start: int | None = None  # where the open part's speech began
        self._silence_start: int | None = None  # where the current silence began, if inside a part
        self._last_pause: tuple[int, int] | None = None  # latest short pause inside the open part
        self._cut_at = 0  # end of the previous part

    def push(self, samples: np.ndarray) -> list[Segment]:
        """Add audio; returns the parts that are now complete."""
        self._audio = np.concatenate([self._audio, samples.astype(np.float32, copy=False)])
        return self._follow(self._vad.push(samples))

    def flush(self) -> list[Segment]:
        """The recording has ended: returns its last part, if it has speech."""
        segments = self._follow(self._vad.flush())
        if self._speech_start is not None:
            end = self._silence_start if self._silence_start is not None else self._recorded
            segments += self._cut(min(end, self._recorded))
        return segments

    @property
    def _recorded(self) -> int:
        return self._offset + self._audio.size

    def _follow(self, probabilities: np.ndarray) -> list[Segment]:
        segments = []
        for probability in probabilities:
            window_start = self._position
            self._position += WINDOW
            if probability >= self._threshold:
                if self._speech_start is None:
                    self._speech_start = window_start
                elif self._silence_start is not None and window_start - self._silence_start >= self._short_pause:
                    self._last_pause = (self._silence_start, window_start)
                self._silence_start = None
            elif probability < self._neg_threshold and self._speech_start is not None and self._silence_start is None:
                self._silence_start = window_start
            # Between the two thresholds nothing changes: speech or silence carries on.
            if self._speech_start is None:
                continue
            if (
                self._silence_start is not None
                and self._position - self._silence_start >= self._pause
                and self._silence_start - self._speech_start >= self._min_part
            ):
                segments += self._cut(self._silence_start)
            elif self._position - self._speech_start > self._max_speech:
                segments += self._cut_long_part()
        self._forget_old_audio()
        return segments

    def _cut(self, speech_end: int) -> list[Segment]:
        """End the open part after its speech, with padding from the silence that follows."""
        segments = self._segment(self._speech_start, speech_end, min(speech_end + self._pad, self._recorded))
        self._speech_start = self._silence_start = self._last_pause = None
        return segments

    def _cut_long_part(self) -> list[Segment]:
        """The open part is about to outgrow Whisper's window: cut it at a short pause if there is one."""
        if self._silence_start is not None and self._position - self._silence_start >= self._short_pause:
            pause, resume = (self._silence_start, self._position), None  # silent right now
        elif self._last_pause is not None:
            pause, resume = self._last_pause, self._last_pause[1]
        else:  # no pause at all: cut mid-speech, without padding, so nothing is heard twice
            boundary = min(self._position, self._recorded)
            segments = self._segment(self._speech_start, boundary, boundary)
            self._speech_start, self._last_pause = boundary, None
            return segments
        pause_start, pause_end = pause
        boundary = pause_start + min(self._pad, (pause_end - pause_start) // 2)
        segments = self._segment(self._speech_start, pause_start, boundary)
        self._speech_start, self._last_pause = resume, None
        if resume is None:
            self._silence_start = None
        return segments

    def _segment(self, speech_start: int, speech_end: int, end: int) -> list[Segment]:
        end = min(end, self._recorded)
        start = max(self._cut_at, speech_start - self._pad, self._offset)
        self._cut_at = end
        if speech_end - speech_start < self._min_speech:  # a click or a breath
            return []
        audio = self._audio[start - self._offset : end - self._offset].copy()
        return [Segment(audio, start, end, (speech_end - speech_start) / SAMPLE_RATE)]

    def _forget_old_audio(self) -> None:
        open_from = self._speech_start if self._speech_start is not None else self._position
        keep_from = max(self._offset, min(open_from, self._recorded) - self._pad)
        if keep_from - self._offset >= SAMPLE_RATE:
            self._audio = self._audio[keep_from - self._offset :]
            self._offset = keep_from


class Session:
    """One recording on its way through resampling, the VAD, the cuts and Whisper.

    Parts finished while the key is held can take Whisper's full 30 s window
    (full_window_while_talking): slower, more accurate, and nobody is waiting for them yet.
    After release everything uses the fast shortened window. context_chars of earlier text
    go to Whisper as a prompt (0: none); cut_options go to Segmenter.
    """

    def __init__(
        self,
        transcriber,
        rate: int,
        context_chars: int = 0,
        full_window_while_talking: bool = FULL_WINDOW_WHILE_TALKING,
        **cut_options,
    ):
        self._transcriber = transcriber
        self._resampler = StreamResampler(rate)
        self._segmenter = Segmenter(**cut_options)
        self._context_chars = context_chars
        self._full_window_while_talking = full_window_while_talking
        self.released_at: float | None = None  # time.monotonic() of the key release, once it happened
        self.texts: list[str] = []
        self.parts = 0
        self.samples = 0  # at 16 kHz
        self.peak = 0.0

    @property
    def audio_s(self) -> float:
        return self.samples / SAMPLE_RATE

    def feed(self, samples: np.ndarray) -> None:
        """Add audio at the recording's rate, transcribing each part it completes."""
        if samples.size:
            self.peak = max(self.peak, float(np.abs(samples).max()))
        self._add(self._resampler.push(samples))

    def release(self, released_at: float) -> None:
        """The key is up: from now on the speaker is waiting."""
        self.released_at = released_at

    def finish(self) -> str:
        """Transcribe what's left and return the text of the whole recording."""
        if self.released_at is None:
            self.release(time.monotonic())
        self._add(self._resampler.flush())
        self._transcribe(self._segmenter.flush())
        return " ".join(self.texts)

    def _add(self, audio: np.ndarray) -> None:
        self.samples += audio.size
        self._transcribe(self._segmenter.push(audio))

    def _transcribe(self, segments: list[Segment]) -> None:
        full_window = self._full_window_while_talking and self.released_at is None
        for segment in segments:
            self.parts += 1
            prompt = " ".join(self.texts)[-self._context_chars :] if self._context_chars else ""
            text = self._transcriber.transcribe_speech(segment.audio, segment.speech_s, prompt or None, full_window)
            if text:
                self.texts.append(text)


def transcribe_in_parts(transcriber, audio: np.ndarray, **options):
    """(text, parts) the pipeline produces for a 16 kHz clip, fed in at once instead of in real time.

    The key counts as released at the end of the audio; options go to Session.
    """
    session = Session(transcriber, SAMPLE_RATE, **options)
    block = int(0.02 * SAMPLE_RATE)
    for offset in range(0, audio.size, block):
        session.feed(audio[offset : offset + block])
    return session.finish(), session.parts


@dataclass
class Result:
    text: str
    audio_s: float
    parts: int
    silent: bool  # audio arrived, but all of it near zero: likely no microphone permission
    released_at: float  # time.monotonic() when the key was released
    done_at: float

    @property
    def latency_s(self) -> float:
        return self.done_at - self.released_at


class Pipeline:
    """Transcribes recordings on one worker thread while they're being recorded.

    Whisper must stay on one thread, so the worker both cuts and transcribes: every POLL_S it
    reads the new audio of each open recording and transcribes any part a pause completed.
    Call begin() when a recording starts and end() when the key is released, then close the
    recording once it has its last audio; the result comes when both have happened.
    session_options go to Session.
    """

    POLL_S = 0.03

    def __init__(self, transcriber, on_result, on_ready=None, **session_options):
        self.transcriber = transcriber
        self.on_result = on_result  # called on the worker thread with a Result
        self.on_ready = on_ready
        self.session_options = session_options
        self._messages: queue.Queue = queue.Queue()

    def start(self) -> None:
        threading.Thread(target=self._run, name="voice_to_text-transcribe", daemon=True).start()

    def stop(self) -> None:
        """Let the worker exit once it has handled everything sent before this."""
        self._messages.put(("stop", None, 0.0))

    # These are called from the main thread.

    def begin(self, recording) -> None:
        self._messages.put(("begin", recording, 0.0))

    def end(self, recording, released_at: float) -> None:
        """The key was released at released_at (time.monotonic()); the speaker is now waiting."""
        self._messages.put(("end", recording, released_at))

    def cancel(self, recording) -> None:
        self._messages.put(("cancel", recording, 0.0))

    # The worker thread.

    def _run(self) -> None:
        if not self.transcriber.loaded:
            self.transcriber.load()
        if self.on_ready:
            self.on_ready()
        sessions: dict = {}  # recording -> Session, until its result is out
        while True:
            try:
                kind, recording, released_at = self._messages.get(timeout=self.POLL_S)
            except queue.Empty:
                kind = None
            if kind == "stop":
                return
            if kind == "begin":
                sessions[recording] = Session(self.transcriber, recording.rate, **self.session_options)
            elif kind == "cancel":
                sessions.pop(recording, None)
            elif kind == "end" and recording in sessions:
                sessions[recording].release(released_at)
            for open_recording, session in list(sessions.items()):
                try:
                    closed = open_recording.closed  # checked first: a closed recording gets no more audio
                    session.feed(open_recording.read())
                    if closed and session.released_at is not None:
                        del sessions[open_recording]
                        self._finish(session)
                except Exception as exc:  # drop this recording, keep the worker alive
                    sessions.pop(open_recording, None)
                    print(f"transcription error: {exc!r}", flush=True)

    def _finish(self, session: Session) -> None:
        text = session.finish()
        silent = session.samples > 0 and session.peak < 1e-4
        result = Result(text, session.audio_s, session.parts, silent, session.released_at, time.monotonic())
        try:
            self.on_result(result)
        except Exception as exc:  # the app's callback failing mustn't stop the worker
            print(f"result handling error: {exc!r}", flush=True)
