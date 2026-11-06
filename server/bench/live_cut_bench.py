"""How the live cutting affects Parakeet: 4-sentence recordings (bench/librispeech.stitched's recipe)
fed through server/live.py's Segmenter with different pause / minimum-part settings.

usage: live_cut_bench.py VERSION QUANT     e.g. v3 int8
"""
import json, random, sys, time
import numpy as np
sys.path.insert(0, "/root/voice_to_text/server"); sys.path.insert(0, "/root/voice_to_text/bench")
import onnx_asr, onnxruntime as rt
from faster_whisper import decode_audio
from faster_whisper.vad import collect_chunks, get_speech_timestamps
from whisper_normalizer import EnglishTextNormalizer
from live import Segmenter
from server import _VAD, MAX_CHUNK_S, SAMPLE_RATE

version, quant = sys.argv[1], (None if sys.argv[2] == "fp32" else sys.argv[2])
pad = float(sys.argv[3])  # seconds of audio kept around speech
from faster_whisper.vad import VadOptions
_VAD = VadOptions(threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=500, speech_pad_ms=int(pad * 1000), max_speech_duration_s=MAX_CHUNK_S)
opts = rt.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
model = onnx_asr.load_model(f"nemo-parakeet-tdt-0.6b-{version}", quantization=quant, sess_options=opts, providers=["CPUExecutionProvider"])
rec = lambda a: model.recognize(a).strip()

base = "/root/voice_to_text/bench/data/librispeech"
rng, noise = random.Random("stitched-0"), np.random.default_rng(0)
recordings = []
for split in ("clean", "other"):
    entries = json.load(open(f"{base}/{split}.json"))
    for i in range(0, len(entries), 4):
        group, parts = entries[i:i + 4], []
        for e in group:
            if parts:
                parts.append(noise.normal(0, 3e-4, int(rng.uniform(0.5, 1.0) * SAMPLE_RATE)).astype(np.float32))
            parts.append(decode_audio(f"{base}/{split}/{e['file']}"))
        recordings.append({"split": split, "audio": np.concatenate(parts), "text": " ".join(e["text"] for e in group)})

def whole(audio):  # what an upload gets: VAD over the whole recording, parts of at most 25 s
    speech = get_speech_timestamps(audio, _VAD)
    clip = audio[speech[0]["start"]:speech[-1]["end"]]
    chunks = [clip] if clip.size <= MAX_CHUNK_S * SAMPLE_RATE else collect_chunks(audio, speech, max_duration=MAX_CHUNK_S)[0]
    return " ".join(filter(None, map(rec, chunks))), len(chunks)

def live(audio, **cut):
    seg, texts, n = Segmenter(pad_s=pad, **cut), [], 0
    block = int(0.02 * SAMPLE_RATE)
    for o in range(0, audio.size, block):
        for part in seg.push(audio[o:o + block]):
            n += 1; texts.append(rec(part.audio))
    for part in seg.flush():
        n += 1; texts.append(rec(part.audio))
    return " ".join(filter(None, texts)), n

normalize = EnglishTextNormalizer()
def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]

configs = {"whole recording": None, "live, cut at every 0.5 s pause (now)": dict(pause_s=0.5, min_part_s=0.0),
           "live, parts of at least 3 s": dict(pause_s=0.5, min_part_s=3.0), "live, parts of at least 6 s": dict(pause_s=0.5, min_part_s=6.0),
           "live, cut only at 0.8 s pauses": dict(pause_s=0.8, min_part_s=0.0)}
for label, cut in configs.items():
    errs = {"clean": [0, 0], "other": [0, 0]}; parts = 0; t0 = time.monotonic()
    for r in recordings:
        hyp, n = whole(r["audio"]) if cut is None else live(r["audio"], **cut)
        ref_w, hyp_w = normalize(r["text"]).split(), normalize(hyp).split()
        errs[r["split"]][0] += edits(ref_w, hyp_w); errs[r["split"]][1] += len(ref_w); parts += n
    print(f"RESULT {version}-{sys.argv[2]}-pad{pad} | {label:38} | WER clean {100*errs['clean'][0]/errs['clean'][1]:.2f}% other {100*errs['other'][0]/errs['other'][1]:.2f}% "
          f"| parts per recording {parts/len(recordings):.1f} | {time.monotonic()-t0:.0f} s", flush=True)
