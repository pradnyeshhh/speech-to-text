"""Always-open microphone stream that keeps a short pre-roll before each press."""

import collections
import threading

import numpy as np
import sounddevice as sd

BLOCK_S = 0.02


def find_input_device(name: str | None = None) -> int:
    """Index of the input device to record from.

    Without a name this prefers the built-in mic: holding a Bluetooth mic open
    (earbuds, for example) switches the earbuds into low-quality headset audio.
    """
    inputs = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if name:
        for index, device_name in inputs:
            if name.lower() in device_name.lower():
                return index
        available = ", ".join(device_name for _, device_name in inputs)
        raise ValueError(f"No microphone matching {name!r}. Available: {available}")
    for index, device_name in inputs:
        lowered = device_name.lower()
        if "microphone" in lowered and ("macbook" in lowered or "built-in" in lowered):
            return index
    return sd.query_devices(kind="input")["index"]


class Recording:
    """The audio of one hold of the key, readable while it's still being recorded."""

    def __init__(self, rate: int, blocks=()):
        self.rate = rate
        self._blocks = list(blocks)
        self._open = True
        self._lock = threading.Lock()

    def append(self, block: np.ndarray) -> None:
        with self._lock:
            if self._open:
                self._blocks.append(block)

    def close(self) -> None:
        with self._lock:
            self._open = False

    @property
    def closed(self) -> bool:
        """No more audio will arrive (what's already there can still be read)."""
        with self._lock:
            return not self._open

    def read(self) -> np.ndarray:
        """The samples recorded since the last read, at the device's rate."""
        with self._lock:
            blocks, self._blocks = self._blocks, []
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


class Recorder:
    def __init__(self, rate: int, preroll_s: float = 0.3, device: int | None = None):
        self.rate = rate
        self.device = device
        self.device_name = ""
        self._preroll = collections.deque(maxlen=max(1, round(preroll_s / BLOCK_S)))
        self._current: Recording | None = None
        self._lock = threading.Lock()
        self._stream = None

    @classmethod
    def for_input(cls, name: str | None = None, preroll_s: float = 0.3) -> "Recorder":
        index = find_input_device(name)
        info = sd.query_devices(index)
        # Capture at the device's native rate and resample ourselves (voice_to_text.streaming does it):
        # PortAudio's own converter isn't rebuilt if another app changes the device's rate.
        recorder = cls(int(info["default_samplerate"]), preroll_s, index)
        recorder.device_name = info["name"]
        return recorder

    def open(self) -> None:
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.rate * BLOCK_S),
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        self.feed(indata[:, 0].copy())

    def feed(self, block: np.ndarray) -> None:
        with self._lock:
            if self._current is not None:
                self._current.append(block)
            else:
                self._preroll.append(block)

    def start(self) -> Recording:
        """Start a recording; it begins with the pre-roll."""
        with self._lock:
            self._current = Recording(self.rate, self._preroll)
            self._preroll.clear()
            return self._current

    def stop(self) -> None:
        """End the recording in progress. What it recorded stays readable."""
        with self._lock:
            recording, self._current = self._current, None
        if recording is not None:
            recording.close()
