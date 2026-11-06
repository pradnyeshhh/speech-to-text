import numpy as np

from voice_to_text.recorder import BLOCK_S, Recorder

RATE = 48000
BLOCK = int(RATE * BLOCK_S)


def block(value: float = 0.5) -> np.ndarray:
    return np.full(BLOCK, value, dtype=np.float32)


def test_recording_starts_with_the_preroll():
    recorder = Recorder(RATE, preroll_s=0.3)
    for _ in range(50):  # 1 s before the press; only the last 0.3 s should be kept
        recorder.feed(block())
    recording = recorder.start()
    for _ in range(25):  # 0.5 s while held
        recorder.feed(block())
    recorder.stop()

    audio = recording.read()

    assert audio.dtype == np.float32
    assert audio.size == int(0.8 * RATE)


def test_each_read_returns_only_the_new_audio():
    recorder = Recorder(RATE, preroll_s=BLOCK_S)
    recording = recorder.start()
    assert recording.read().size == 0  # no pre-roll recorded yet
    recorder.feed(block(0.1))
    first = recording.read()
    recorder.feed(block(0.2))
    recorder.feed(block(0.2))
    second = recording.read()

    assert first.size == BLOCK and np.all(first == 0.1)
    assert second.size == 2 * BLOCK and np.all(second == 0.2)
    assert recording.read().size == 0


def test_audio_after_stop_goes_to_the_next_preroll():
    recorder = Recorder(RATE, preroll_s=0.1)
    recording = recorder.start()
    recorder.feed(block())
    recorder.stop()
    for _ in range(10):
        recorder.feed(block())

    assert recording.read().size == BLOCK  # nothing after stop reaches the old recording
    assert recorder.start().read().size == int(0.1 * RATE)


def test_stop_without_start_does_nothing():
    recorder = Recorder(RATE)
    recorder.feed(block())
    recorder.stop()
    assert recorder.start().read().size == BLOCK
