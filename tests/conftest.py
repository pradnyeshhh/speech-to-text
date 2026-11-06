import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture
def say(tmp_path):
    """Speech made by macOS `say`, as 16 kHz float32 numpy audio."""
    if not shutil.which("say"):
        pytest.skip("needs macOS `say`")
    from mlx_whisper.audio import load_audio

    def speak(text: str) -> np.ndarray:
        path = tmp_path / "speech.aiff"
        subprocess.run(["say", "-v", "Samantha", "-o", str(path), text], check=True)
        # numpy, not mlx_whisper's lazy MLX array: MLX can't finish another thread's pending work.
        return np.array(load_audio(str(path)), dtype=np.float32)

    return speak
