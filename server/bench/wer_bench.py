"""WER and latency on the first N LibriSpeech test-clean and test-other utterances of bench/data.

usage: wer_bench.py ENGINE N OUT.json     ENGINE: parakeet | small | turbo-short | turbo-full
"""
import json, statistics, sys, time
import numpy as np
sys.path.insert(0, "/root/voice_to_text/server"); sys.path.insert(0, "/root/voice_to_text/bench")
from faster_whisper import decode_audio
from whisper_normalizer import EnglishTextNormalizer
from server import _VAD, MAX_CHUNK_S, SAMPLE_RATE, Transcriber

engine, n, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if engine.startswith("parakeet"):  # parakeet[-v2|-v3][-int8|-fp32][-padNNN]
    import onnx_asr, onnxruntime as rt
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps
    version = "v2" if "-v2" in engine else "v3"
    quantization = None if "-fp32" in engine else "int8"
    if "-pad" in engine:
        _VAD = VadOptions(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=500,
                          speech_pad_ms=int(engine.split("-pad")[1]), max_speech_duration_s=MAX_CHUNK_S)
    opts = rt.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    model = onnx_asr.load_model(f"nemo-parakeet-tdt-0.6b-{version}", quantization=quantization, sess_options=opts, providers=["CPUExecutionProvider"])
    model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32))
    def transcribe(audio):
        speech = get_speech_timestamps(audio, _VAD)
        if not speech:
            return ""
        clip = audio[speech[0]["start"] : speech[-1]["end"]]
        chunks = [clip] if clip.size <= MAX_CHUNK_S * SAMPLE_RATE else collect_chunks(audio, speech, max_duration=MAX_CHUNK_S)[0]
        return " ".join(filter(None, (model.recognize(c).strip() for c in chunks)))
else:
    name, short = {"small": ("small", False), "turbo-short": ("large-v3-turbo", True), "turbo-full": ("large-v3-turbo", False)}[engine]
    tr = Transcriber(name, 2, short_windows=short); tr.warm_up()
    transcribe = lambda audio: tr.transcribe(audio)[0]

normalize = EnglishTextNormalizer()
def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]

rows = []
for split in ("clean", "other"):
    base = f"/root/voice_to_text/bench/data/librispeech"
    for u in json.load(open(f"{base}/{split}.json"))[:n]:
        audio = decode_audio(f"{base}/{split}/{u['file']}")
        t = time.monotonic(); hyp = transcribe(audio); secs = time.monotonic() - t
        r, h = normalize(u["text"]).split(), normalize(hyp).split()
        rows.append(dict(split=split, id=u["id"], audio_s=audio.size / SAMPLE_RATE, seconds=secs, errors=edits(r, h),
                         ref_words=len(r), doubled=len(h) >= len(r) + 3 and len(h) - len(r) >= len(r) / 2, ref=u["text"], hyp=hyp))
        print(f"{engine} {split} {len(rows)}/{2 * n} {secs:.2f} s", flush=True)
json.dump(rows, open(out_path, "w"), indent=1)

def summary(rs):
    wer = 100 * sum(r["errors"] for r in rs) / sum(r["ref_words"] for r in rs)
    return wer
lat = sorted(r["seconds"] for r in rows)
print(f"RESULT {engine}: WER clean {summary([r for r in rows if r['split']=='clean']):.2f}% | other {summary([r for r in rows if r['split']=='other']):.2f}% | "
      f"doubled {sum(r['doubled'] for r in rows)} | latency p50 {statistics.median(lat):.2f} s p95 {lat[int(0.95 * (len(lat) - 1))]:.2f} s | "
      f"{sum(r['audio_s'] for r in rows) / sum(lat):.1f}x real time | audio p50 {statistics.median(r['audio_s'] for r in rows):.1f} s", flush=True)
