"""Parakeet-TDT-0.6B-v3 (int8 ONNX via onnx-asr) through server.py's VAD and splitting, on the benchmark clips."""
import difflib, json, re, sys, time
import numpy as np, onnxruntime as rt, onnx_asr
from faster_whisper import decode_audio
from faster_whisper.vad import collect_chunks, get_speech_timestamps
sys.path.insert(0, "/root/voice_to_text/server")
from server import _VAD, MAX_CHUNK_S, SAMPLE_RATE

opts = rt.SessionOptions(); opts.intra_op_num_threads = int(sys.argv[1]); opts.inter_op_num_threads = 1
t = time.monotonic()
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8", sess_options=opts, providers=["CPUExecutionProvider"])
print(f"load {time.monotonic() - t:.1f} s", flush=True)
model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up

def transcribe(audio):
    speech = get_speech_timestamps(audio, _VAD)
    if not speech:
        return ""
    clip = audio[speech[0]["start"] : speech[-1]["end"]]
    chunks = [clip] if clip.size <= MAX_CHUNK_S * SAMPLE_RATE else collect_chunks(audio, speech, max_duration=MAX_CHUNK_S)[0]
    return " ".join(filter(None, (model.recognize(c).strip() for c in chunks)))

clips = "/root/voice_to_text/bench/data/clips"
ref = json.load(open(f"{clips}/clips.json"))
norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower().replace("-", " ")).split()
for name in sys.argv[2:]:
    audio = decode_audio(f"{clips}/{name}.wav")
    times = []
    for _ in range(2):
        t = time.monotonic(); text = transcribe(audio); times.append(time.monotonic() - t)
    a, b = norm(ref[name]["text"]), norm(text)
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    print(f"{name:6} {audio.size / SAMPLE_RATE:5.1f} s -> best {min(times):5.2f} s   {len(b)}/{len(a)} words  {text[:90]!r}", flush=True)
    for tag, i1, i2, j1, j2 in ops:
        print(f"         {tag}: {' '.join(a[i1:i2])!r} -> {' '.join(b[j1:j2])!r}")
