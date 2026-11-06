"""Hold the hotkey, speak, release: the transcript is pasted where you're typing."""

import time

from AppKit import NSSound
from PyObjCTools import AppHelper

from .hotkey import HotkeyListener
from .paste import paste_text
from .recorder import Recorder
from .streaming import Pipeline, Result
from .stt import DEFAULT_MODEL, Transcriber

POSTROLL_S = 0.15  # keep recording briefly after release so the last word isn't clipped


class Dictation:
    def __init__(self, key: str, device: str | None = None, model: str = DEFAULT_MODEL):
        self.key = key
        self.model = model
        self.recorder = Recorder.for_input(device)
        # Transcribes while you talk, so after release only the part since your last pause is left.
        self.pipeline = Pipeline(Transcriber(model), self._on_result, on_ready=self._on_ready)
        self.hotkey = HotkeyListener(key, self._on_start, self._on_finish, self._on_cancel)
        self._recording = None  # the recording in progress
        self._pending_finish = None  # a released recording still in its post-roll

    def run(self) -> None:
        print(f"Microphone: {self.recorder.device_name}")
        self.recorder.open()
        print(f"Loading {self.model}…", flush=True)
        self.pipeline.start()
        self.hotkey.install()
        AppHelper.runConsoleEventLoop(installInterrupt=True)

    # The hotkey callbacks below all run on the main thread.

    def _on_start(self) -> None:
        if self._pending_finish is not None:  # pressed again during the post-roll
            self._finish(self._pending_finish)
        self._recording = self.recorder.start()
        self.pipeline.begin(self._recording)
        _play("Tink")

    def _on_finish(self) -> None:
        self._pending_finish = self._recording
        self.pipeline.end(self._recording, time.monotonic())
        AppHelper.callLater(POSTROLL_S, self._finish, self._recording)

    def _finish(self, recording) -> None:
        if self._pending_finish is not recording:
            return
        self._pending_finish = None
        self.recorder.stop()  # closes the recording; the pipeline then transcribes the rest
        _play("Pop")

    def _on_cancel(self) -> None:
        self.recorder.stop()
        self.pipeline.cancel(self._recording)

    # These run on the pipeline's worker thread.

    def _on_ready(self) -> None:
        print(f"Ready. Hold {self.key.replace('_', ' ')}, speak, release. Ctrl+C quits.", flush=True)

    def _on_result(self, result: Result) -> None:
        if result.silent:
            print("The microphone is silent. Is Microphone access allowed for your terminal?")
            return
        if not result.text:
            print("(no speech)")
            return
        parts = f"{result.parts} part" + ("s" if result.parts != 1 else "")
        print(
            f"{result.text}  [{result.audio_s:.1f} s audio in {parts}, "
            f"{result.latency_s * 1000:.0f} ms after release]",
            flush=True,
        )
        # Trailing space so the next dictation doesn't run into this one.
        AppHelper.callAfter(paste_text, result.text + " ")


def _play(name: str) -> None:
    sound = NSSound.soundNamed_(name)
    if sound is not None:
        sound.stop()
        sound.play()
