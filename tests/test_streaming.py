import threading
import time
from math import gcd

import numpy as np
import pytest
from scipy.signal import resample_poly

from voice_to_text.recorder import Recording
from voice_to_text.silero_vad import get_vad_model
from voice_to_text.streaming import SAMPLE_RATE, WINDOW, Pipeline, Segmenter, StreamingVad, StreamResampler

RNG = np.random.default_rng(0)


def pieces(audio: np.ndarray, seed: int = 1, largest: int = 3000):
    """Audio split into randomly sized pieces, like blocks arriving from a microphone."""
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < audio.size:
        size = int(rng.integers(1, largest))
        yield audio[offset : offset + size]
        offset += size


# Resampling and VAD on arriving audio give what they'd give on the whole clip.


@pytest.mark.parametrize("rate", [48000, 44100, 16000])
def test_resampling_arriving_audio_matches_resampling_it_all_at_once(rate):
    audio = RNG.normal(0, 0.1, 2 * rate).astype(np.float32)
    resampler = StreamResampler(rate)
    streamed = np.concatenate([resampler.push(p) for p in pieces(audio)] + [resampler.flush()])
    divisor = gcd(SAMPLE_RATE, rate)
    expected = audio if rate == SAMPLE_RATE else resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    assert streamed.size == expected.size
    np.testing.assert_allclose(streamed, expected, atol=1e-5)


def test_vad_on_arriving_audio_matches_the_vad_on_the_whole_clip():
    # The content doesn't matter here, only that the model's state carries across the pieces.
    t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    audio = (audio + RNG.normal(0, 0.01, audio.size)).astype(np.float32)
    vad = StreamingVad()
    streamed = np.concatenate([vad.push(p) for p in pieces(audio)] + [vad.flush()])
    expected = get_vad_model()(np.pad(audio, (0, WINDOW - audio.size % WINDOW)))
    np.testing.assert_allclose(streamed, expected, atol=1e-5)


# Cutting, with noise for speech and zeros for silence.


class LoudnessVad:
    """Stands in for Silero: a window is speech when it's loud."""

    def __init__(self):
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, samples):
        self._pending = np.concatenate([self._pending, samples])
        count = self._pending.size // WINDOW
        windows = self._pending[: count * WINDOW].reshape(count, WINDOW)
        self._pending = self._pending[count * WINDOW :]
        return (np.sqrt((windows**2).mean(axis=1)) > 0.01).astype(np.float32)

    def flush(self):
        window = np.pad(self._pending, (0, WINDOW - self._pending.size))
        self._pending = np.zeros(0, dtype=np.float32)
        return self.push(window)


def speech(seconds: float) -> np.ndarray:
    return RNG.normal(0, 0.1, int(seconds * SAMPLE_RATE)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def cut(audio: np.ndarray, **options) -> list[tuple]:
    """Feed audio in 20 ms blocks; returns (part, seconds of audio fed when it came out)."""
    segmenter = Segmenter(vad=LoudnessVad(), **options)
    block = int(0.02 * SAMPLE_RATE)
    parts, fed = [], 0
    for offset in range(0, audio.size, block):
        piece = audio[offset : offset + block]
        fed += piece.size
        parts += [(part, fed / SAMPLE_RATE) for part in segmenter.push(piece)]
    parts += [(part, fed / SAMPLE_RATE) for part in segmenter.flush()]
    return parts


def test_a_pause_ends_a_part_while_the_speaker_goes_on():
    audio = np.concatenate([silence(0.5), speech(2.0), silence(1.0), speech(2.0), silence(0.3)])
    parts = cut(audio, pause_s=0.5)
    assert len(parts) == 2
    (first, out_at), (second, _) = parts
    assert 3.0 <= out_at <= 3.1  # 0.5 s into the pause, long before the second sentence ends
    assert first.speech_s == pytest.approx(2.0, abs=0.07)
    assert second.speech_s == pytest.approx(2.0, abs=0.07)


def test_a_shorter_pause_stays_inside_the_part():
    parts = cut(np.concatenate([speech(1.0), silence(0.3), speech(1.0)]), pause_s=0.5)
    assert len(parts) == 1
    assert parts[0][0].speech_s == pytest.approx(2.3, abs=0.07)


def test_a_pause_ends_a_part_only_once_it_has_min_part_s_of_speech():
    audio = np.concatenate([speech(1.0), silence(0.7), speech(2.5), silence(0.7), speech(1.0)])
    assert len(cut(audio, pause_s=0.5)) == 3
    first, second = [part for part, _ in cut(audio, pause_s=0.5, min_part_s=3.0)]
    assert first.speech_s == pytest.approx(4.2, abs=0.07)  # the 1 s opener was too short to end at its pause
    assert second.speech_s == pytest.approx(1.0, abs=0.07)


def test_the_pause_length_is_configurable():
    audio = np.concatenate([speech(1.0), silence(0.4), speech(1.0)])
    assert len(cut(audio, pause_s=0.3)) == 2
    assert len(cut(audio, pause_s=0.5)) == 1


def test_parts_are_padded_with_the_silence_around_them():
    audio = np.concatenate([silence(1.0), speech(1.0), silence(1.0)])
    (part, _), = cut(audio, pause_s=0.5, pad_s=0.2)
    assert part.start / SAMPLE_RATE == pytest.approx(1.0 - 0.2, abs=0.04)
    assert part.end / SAMPLE_RATE == pytest.approx(2.0 + 0.2, abs=0.04)


def test_a_long_part_is_cut_at_its_last_short_pause():
    audio = np.concatenate([speech(12.0), silence(0.2), speech(8.0), silence(0.2), speech(10.0)])
    parts = [part for part, _ in cut(audio, max_s=25)]
    assert len(parts) == 2
    assert parts[0].end / SAMPLE_RATE == pytest.approx(20.3, abs=0.1)  # in the second short pause
    assert all(part.audio.size <= 25 * SAMPLE_RATE for part in parts)


def test_speech_without_any_pause_is_cut_before_whispers_window_fills():
    audio = speech(40.0)
    parts = [part for part, _ in cut(audio, max_s=25)]
    assert len(parts) == 2
    assert parts[0].audio.size <= 25 * SAMPLE_RATE
    assert parts[0].end == parts[1].start  # nothing lost or repeated at the cut
    assert parts[1].end == audio.size


def test_silence_gives_no_parts():
    assert cut(silence(3.0)) == []


def test_a_click_is_too_short_to_be_a_part():
    assert cut(np.concatenate([silence(1.0), speech(0.06), silence(1.0)])) == []


def test_parts_follow_each_other_and_hold_the_recorded_audio():
    audio = np.concatenate(
        [silence(0.4), speech(1.5), silence(0.8), speech(0.7), silence(0.6), speech(2.2), silence(0.2)]
    )
    parts = [part for part, _ in cut(audio, pause_s=0.5)]
    assert len(parts) == 3
    for earlier, later in zip(parts, parts[1:]):
        assert earlier.end <= later.start
    for part in parts:
        np.testing.assert_array_equal(part.audio, audio[part.start : part.end])


# The pipeline transcribes parts while the recording goes on.


class FakeTranscriber:
    loaded = True

    def __init__(self):
        self.parts: list[float] = []
        self.full_windows: list[bool] = []
        self.first_part = threading.Event()

    def transcribe_speech(self, clip, speech_s, prompt=None, full_window=False):
        self.parts.append(clip.size / SAMPLE_RATE)
        self.full_windows.append(full_window)
        self.first_part.set()
        return f"part{len(self.parts)}"


def wait_for(condition, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_the_first_part_is_transcribed_before_the_key_is_released(say):
    first = np.concatenate([say("Let's meet tomorrow at the coffee shop."), silence(1.0)])
    second = say("Bring the quarterly report.")
    transcriber, results, done = FakeTranscriber(), [], threading.Event()
    pipeline = Pipeline(transcriber, lambda result: (results.append(result), done.set()))
    pipeline.start()

    recording = Recording(SAMPLE_RATE)
    pipeline.begin(recording)
    recording.append(first)
    assert transcriber.first_part.wait(5), "the first part should be transcribed during the recording"
    recording.append(second)
    pipeline.end(recording, time.monotonic())
    recording.close()

    assert done.wait(5)
    assert results[0].text == "part1 part2"
    assert results[0].parts == 2
    pipeline.stop()


def test_parts_finished_while_talking_get_the_full_window_and_the_rest_the_fast_one(say):
    sentence = np.concatenate([say("Let's meet tomorrow at the coffee shop."), silence(1.0)])
    transcriber, results = FakeTranscriber(), []
    pipeline = Pipeline(transcriber, results.append, full_window_while_talking=True)
    pipeline.start()

    recording = Recording(SAMPLE_RATE)
    pipeline.begin(recording)
    recording.append(sentence)
    assert wait_for(lambda: len(transcriber.parts) == 1)
    recording.append(sentence)
    pipeline.end(recording, time.monotonic())  # released during the pause, before it ends the part
    recording.close()
    assert wait_for(lambda: results)
    pipeline.stop()

    assert transcriber.full_windows == [True, False]


def test_the_result_waits_for_the_audio_after_release(say):
    transcriber, results = FakeTranscriber(), []
    pipeline = Pipeline(transcriber, results.append)
    pipeline.start()

    recording = Recording(SAMPLE_RATE)
    pipeline.begin(recording)
    pipeline.end(recording, time.monotonic())  # released...
    time.sleep(0.2)
    recording.append(say("Sounds good."))  # ...but the post-roll still brings audio
    assert not results
    recording.close()
    assert wait_for(lambda: results)
    pipeline.stop()

    assert results[0].text == "part1"


def test_the_pipeline_keeps_going_when_handling_a_result_fails(say):
    audio = say("Sounds good.")
    results = []

    def on_result(result):
        results.append(result)
        if len(results) == 1:
            raise RuntimeError("couldn't paste")

    pipeline = Pipeline(FakeTranscriber(), on_result)
    pipeline.start()
    for _ in range(2):
        recording = Recording(SAMPLE_RATE)
        pipeline.begin(recording)
        recording.append(audio)
        pipeline.end(recording, time.monotonic())
        recording.close()
    assert wait_for(lambda: len(results) == 2)
    pipeline.stop()
