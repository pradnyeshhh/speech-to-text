"""Live transcription: audio arrives in pieces, is cut at pauses, and each part is transcribed.

StreamingVad, Segment and Segmenter are copied unchanged from the Mac app's
src/voice_to_text/streaming.py (keep them in sync), apart from Segmenter.open_part(),
added here so the site can show the part still being spoken. Their VAD is the same
Silero model, from faster-whisper instead of the app's vendored copy.
"""

from dataclasses import dataclass

import av
import numpy as np
from faster_whisper.vad import get_vad_model

SAMPLE_RATE = 16000
MAX_CHUNK_S = 25  # as stt.py and server.py
PAUSE_S = 0.5  # silence that ends a part; shorter pauses, inside a sentence, don't
MIN_PART_S = 0.0  # speech a part needs before a pause can end it
WINDOW = 512  # samples per Silero VAD decision at 16 kHz (32 ms)
_CONTEXT = 64  # samples of the previous window Silero sees with each window


class Resampler:
    """To 16 kHz mono as audio arrives, for browsers that won't record at 16 kHz (PyAV's resampler)."""

    def __init__(self, rate: int):
        self.rate = rate
        self._resampler = None if rate == SAMPLE_RATE else av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    def push(self, samples: np.ndarray) -> np.ndarray:
        if self._resampler is None or not samples.size:
            return samples
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, dtype=np.float32)[None], format="flt", layout="mono")
        frame.sample_rate = self.rate
        return self._collect(self._resampler.resample(frame))

    def flush(self) -> np.ndarray:
        if self._resampler is None:
            return np.zeros(0, dtype=np.float32)
        return self._collect(self._resampler.resample(None))

    @staticmethod
    def _collect(frames) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32, copy=False)


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

    def open_part(self) -> tuple[np.ndarray, float] | None:
        """(audio, seconds of speech) of the part still being spoken, for an interim transcript.

        Not in the Mac app. None when no part is open or it has less speech than a part needs.
        """
        if self._speech_start is None:
            return None
        speech_end = min(self._silence_start if self._silence_start is not None else self._position, self._recorded)
        if speech_end - self._speech_start < self._min_speech:
            return None
        start = max(self._cut_at, self._speech_start - self._pad, self._offset)
        end = min(speech_end + self._pad, self._recorded)
        return self._audio[start - self._offset : end - self._offset].copy(), (speech_end - self._speech_start) / SAMPLE_RATE

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
