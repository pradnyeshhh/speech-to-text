"""End-to-end transcription with the real models, on speech made by macOS `say`.

Every test runs once per downloaded model in voice_to_text.stt.MODELS.
"""

import numpy as np
import pytest

from voice_to_text.streaming import transcribe_in_parts
from voice_to_text.stt import MODEL_FILES, MODELS, Transcriber, local_model_path


def _downloaded(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, allow_patterns=MODEL_FILES, local_files_only=True)
        return True
    except Exception:
        return False


DOWNLOADED = [name for name, repo in MODELS.items() if _downloaded(repo)]

pytestmark = pytest.mark.skipif(not DOWNLOADED, reason="needs a downloaded model")


@pytest.fixture(scope="module", params=DOWNLOADED or [None])
def transcriber(request):
    transcriber = Transcriber(request.param)
    transcriber.load()
    return transcriber


def test_transcribes_english_speech(transcriber, say):
    audio = say("Let's meet tomorrow at the coffee shop near the station.")
    text = transcriber.transcribe(audio).lower()
    for word in ("meet", "tomorrow", "coffee", "station"):
        assert word in text


def test_short_clip_is_not_repeated(transcriber, say):
    # Short clips use a shortened encoder window; too short a window makes Whisper repeat itself.
    audio = say("Sounds good.")
    assert transcriber.transcribe(audio).lower().count("sounds good") == 1


def test_long_clip_is_transcribed_in_chunks(transcriber, say):
    items = {
        "meeting room": "First, book a meeting room for Thursday afternoon.",
        "sales figures": "Second, ask the finance team for the updated sales figures.",
        "design review": "Third, move the design review to next week.",
        "invoice": "Fourth, send the invoice to the accounting team.",
        "server upgrade": "Fifth, check whether the server upgrade finished overnight.",
        "shipment": "Sixth, reply to the customer about the delayed shipment.",
        "coffee": "Seventh, order more coffee for the kitchen.",
        "parking": "Eighth, renew the parking permits before they expire.",
        "onboarding": "Ninth, update the onboarding guide for new hires.",
        "planning": "Tenth, schedule the planning session for early March.",
    }
    audio = say(" ".join(items.values()))
    assert audio.size > 25 * 16000  # longer than one chunk
    text = transcriber.transcribe(audio).lower()
    assert [key for key in items if key not in text] == []


def test_repeated_sentences_are_all_kept(transcriber, say, request):
    if transcriber.model_name.startswith("large-v3-turbo"):
        # Turbo writes about 2.5 of 5 identical sentences in a chunk, then stops.
        reason = "large-v3-turbo skips sentences repeated word for word"
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    sentence = "The quarterly numbers look strong, but we should double check the marketing spend. "
    audio = say(sentence * 7)  # longer than one 25 s chunk
    assert audio.size > 25 * 16000
    text = transcriber.transcribe(audio).lower()
    assert text.count("quarterly") == 7


def test_transcribes_in_parts_at_pauses(transcriber, say):
    first = say("Let's meet tomorrow at the coffee shop.")
    second = say("Bring the quarterly report with you.")
    audio = np.concatenate([first, np.zeros(16000, dtype=np.float32), second])
    text, parts = transcribe_in_parts(transcriber, audio)
    assert parts == 2
    for word in ("coffee", "quarterly", "report"):
        assert word in text.lower()


def test_silence_gives_no_text(transcriber):
    assert transcriber.transcribe(np.zeros(2 * 16000, dtype=np.float32)) == ""
    assert transcribe_in_parts(transcriber, np.zeros(2 * 16000, dtype=np.float32)) == ("", 0)


def test_quiet_noise_gives_no_text(transcriber):
    noise = np.random.default_rng(0).normal(0, 0.003, 2 * 16000).astype(np.float32)
    assert transcriber.transcribe(noise) == ""


def test_model_path_is_local(transcriber):
    assert local_model_path(transcriber.repo).startswith("/")
