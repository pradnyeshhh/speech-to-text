"""bolo: hold a key, speak English, and the text is pasted where you're typing."""

import argparse
import sys
import time

from .hotkey import KEYS
from .stt import DEFAULT_MODEL, MODELS


def main() -> int:
    parser = argparse.ArgumentParser(prog="bolo", description=__doc__)
    parser.add_argument("--key", choices=sorted(KEYS), default="right_option", help="hold-to-talk key")
    parser.add_argument("--device", help="microphone name to record from (default: the built-in mic)")
    parser.add_argument(
        "--model", choices=list(MODELS), default=DEFAULT_MODEL, help="speech-to-text model (default: %(default)s)"
    )
    parser.add_argument("--file", help="transcribe an audio file, print the text and exit")
    args = parser.parse_args()

    if args.file:
        return _transcribe_file(args.file, args.model)

    from .app import Dictation
    from .permissions import missing_permissions

    missing = missing_permissions()
    if missing:
        print(
            "Bolo needs these permissions for your terminal app: " + ", ".join(missing) + ".\n"
            "Turn them on in System Settings → Privacy & Security, quit and reopen the terminal, "
            "then run bolo again."
        )
        return 1
    Dictation(args.key, args.device, args.model).run()
    return 0


def _transcribe_file(path: str, model: str) -> int:
    from mlx_whisper.audio import load_audio

    from .stt import Transcriber

    transcriber = Transcriber(model)
    transcriber.load()
    audio = load_audio(path)
    started = time.monotonic()
    text = transcriber.transcribe(audio)
    print(text or "(no speech)")
    print(f"[{audio.size / 16000:.1f} s audio, {time.monotonic() - started:.2f} s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
