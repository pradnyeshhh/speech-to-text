"""Does CTranslate2's Whisper encoder take fewer than 3000 frames, and how fast is each size?"""
import sys, time
import ctranslate2, numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim

model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=int(sys.argv[1]), local_files_only=True)
audio = decode_audio(sys.argv[2])
feats = model.feature_extractor(audio)  # (128, frames)
print("features", feats.shape, flush=True)
for window_s in (30, 20, 10, 30, 20, 10):
    f = np.ascontiguousarray(pad_or_trim(feats, window_s * 100)[None].astype(np.float32))
    t = time.monotonic()
    try:
        out = model.model.encode(ctranslate2.StorageView.from_array(f), to_cpu=True)
        print(f"window {window_s:2} s: encoder {time.monotonic() - t:5.2f} s, output {list(out.shape)}", flush=True)
    except Exception as e:
        print(f"window {window_s:2} s: rejected: {e}", flush=True)
