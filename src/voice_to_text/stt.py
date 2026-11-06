"""English speech-to-text with Whisper models, run on the Apple GPU via MLX."""

import re

import numpy as np

from .silero_vad import VadOptions, collect_chunks, get_speech_timestamps

# MLX conversions that mlx-whisper 0.4.3 can load (it reads weights.safetensors or weights.npz).
MODELS = {
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",  # English only, 2 decoder layers
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",  # OpenAI, multilingual, 4 decoder layers
    "large-v3-turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",  # the same, 4-bit weights
}
# Turbo is as fast as distil on short dictation and far more accurate with the shortened windows
# below; the 4-bit weights match fp16's accuracy in less memory. See BENCHMARKS.md.
DEFAULT_MODEL = "large-v3-turbo-q4"
MODEL_FILES = ["*.json", "*.npz", "*.safetensors"]
SAMPLE_RATE = 16000
MAX_CHUNK_S = 25  # longer recordings are split at pauses into chunks of at most this

_VAD = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=200,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
    max_speech_duration_s=MAX_CHUNK_S,
)

# Whisper normally encodes a full 30 s window, which takes ~1.1 s on an M2 even
# for a 2 s clip. A shorter window is ~2.7x faster on a 4 s clip. With too little
# trailing silence Whisper starts repeating itself, and how much it needs depends on
# the model: on real speech, distil-large-v3 wrote 22 of 200 LibriSpeech utterances
# twice with these windows (0 with 30 s), large-v3-turbo 1. See BENCHMARKS.md.
_WINDOWS_S = ((6, 10), (16, 20))  # (longest clip, window), else the full 30 s

# What Whisper tends to "hear" in silence or noise.
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye"}


def local_model_path(repo: str) -> str:
    """Path of the downloaded model, downloading it the first time."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=repo, allow_patterns=MODEL_FILES, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=repo, allow_patterns=MODEL_FILES)


class Transcriber:
    """Use it from one thread at a time: MLX isn't safe to share across threads."""

    def __init__(self, model: str = DEFAULT_MODEL, short_windows: bool = True):
        self.model_name = model
        self.repo = MODELS[model]
        self.short_windows = short_windows  # False: always encode Whisper's full 30 s window
        self._model = None
        self._positions = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model and run each window size once, so the first dictation is fast."""
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        self._model = load_model(local_model_path(self.repo), dtype=mx.float16)
        self._positions = self._model.encoder._positional_embedding
        for window_s in (10, 20, 30) if self.short_windows else (30,):
            self._decode(np.zeros(SAMPLE_RATE, dtype=np.float32), window_s)

    def transcribe(self, audio: np.ndarray) -> str:
        """Text for a 16 kHz mono clip, or "" if there's no speech in it."""
        speech = get_speech_timestamps(audio, _VAD)
        if not speech:
            return ""
        clip = audio[speech[0]["start"] : speech[-1]["end"]]
        if clip.size <= MAX_CHUNK_S * SAMPLE_RATE:
            chunks = [clip]
        else:
            chunks, _ = collect_chunks(audio, speech, max_duration=MAX_CHUNK_S)
        text = " ".join(filter(None, (self._transcribe_chunk(chunk) for chunk in chunks)))
        speech_s = sum(s["end"] - s["start"] for s in speech) / SAMPLE_RATE
        return _unless_hallucination(text, speech_s)

    def transcribe_speech(
        self, clip: np.ndarray, speech_s: float, prompt: str | None = None, full_window: bool = False
    ) -> str:
        """Text for a clip of at most MAX_CHUNK_S that the caller already found speech in.

        prompt is earlier text from the same recording, to carry on from. full_window encodes
        Whisper's full 30 s window even for a short clip: slower, and more accurate.
        """
        return _unless_hallucination(self._transcribe_chunk(clip, prompt, full_window), speech_s)

    def _transcribe_chunk(self, clip: np.ndarray, prompt: str | None = None, full_window: bool = False) -> str:
        duration_s = clip.size / SAMPLE_RATE
        window_s = 30
        if self.short_windows and not full_window:
            window_s = next((window for longest, window in _WINDOWS_S if duration_s <= longest), 30)
        result = self._decode(clip, window_s, prompt=prompt)
        if window_s < 30 and _looks_wrong(result):
            result = self._decode(clip, 30, prompt=prompt)
        if _looks_wrong(result):
            result = self._decode(clip, 30, temperature=0.4, prompt=prompt)
        if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
            return ""
        return result.text.strip()

    def _decode(self, clip: np.ndarray, window_s: int, temperature: float = 0.0, prompt: str | None = None):
        import mlx.core as mx
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions

        samples = window_s * SAMPLE_RATE
        padded = np.pad(clip, (0, samples - clip.size)) if clip.size < samples else clip[:samples]
        mel = log_mel_spectrogram(padded, n_mels=self._model.dims.n_mels)
        mel = pad_or_trim(mel, window_s * 100, axis=-2).astype(mx.float16)  # 100 mel frames per second

        encoder = self._model.encoder
        encoder._positional_embedding = self._positions[: window_s * 50]  # 50 encoder positions per second
        try:
            options = DecodingOptions(
                language="en",  # skips language detection (both vocabularies are multilingual)
                without_timestamps=True,
                temperature=temperature,
                prompt=prompt,
            )
            return self._model.decode(mel, options)
        finally:
            encoder._positional_embedding = self._positions


def _unless_hallucination(text: str, speech_s: float) -> str:
    """"" for what Whisper tends to hear in a moment of noise, else the text."""
    if speech_s < 1.0 and text.lower().strip(" .!?,") in _HALLUCINATIONS:
        return ""
    return text


def _looks_wrong(result) -> bool:
    """Repetition or low confidence, the usual signs of a Whisper hallucination."""
    if result.compression_ratio > 2.4 or result.avg_logprob < -1.0:
        return True
    sentences = [s.lower() for s in re.split(r"(?<=[.!?])\s+", result.text.strip()) if s]
    return len(sentences) > 1 and len(set(sentences)) == 1
