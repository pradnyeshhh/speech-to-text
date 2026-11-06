"""Encoder time for large-v3-turbo per window, best of 2. usage: encode_env.py THREADS WINDOW..."""
import sys, time
import ctranslate2, numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=int(sys.argv[1]), local_files_only=True)
feats = model.feature_extractor(decode_audio("/root/voice_to_text/bench/data/clips/short.wav"))
out = []
for w in map(int, sys.argv[2:]):
    f = ctranslate2.StorageView.from_array(np.ascontiguousarray(pad_or_trim(feats, w * 100)[None], dtype=np.float32))
    best = min((lambda t: (model.model.encode(f, to_cpu=True), time.monotonic() - t)[1])(time.monotonic()) for _ in range(2))
    out.append(f"{w} s window {best:5.2f} s")
print(" | ".join(out), flush=True)
