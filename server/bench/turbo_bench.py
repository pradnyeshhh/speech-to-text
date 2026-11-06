"""Time large-v3-turbo on the benchmark clips with one set of parameters.

usage: turbo_bench.py THREADS COMPUTE_TYPE CHUNK_LENGTH|none REPEATS CLIP...
"""
import statistics, sys, time
from faster_whisper import WhisperModel, decode_audio

threads, compute, chunk, repeats, clips = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
chunk = None if chunk == "none" else int(chunk)
t0 = time.monotonic()
model = WhisperModel("large-v3-turbo", device="cpu", compute_type=compute, cpu_threads=threads, local_files_only=True)
print(f"load {time.monotonic() - t0:.1f} s", flush=True)

def run(audio):
    segments, _ = model.transcribe(
        audio, language="en", beam_size=1, temperature=[0.0, 0.4], compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6, condition_on_previous_text=False,
        without_timestamps=True, vad_filter=False, chunk_length=chunk,
    )
    return " ".join(s.text.strip() for s in segments)

run(decode_audio(clips[0])[:16000])  # warm-up
for path in clips:
    audio = decode_audio(path)
    times, text = [], ""
    for _ in range(repeats):
        t = time.monotonic(); text = run(audio); times.append(time.monotonic() - t)
    print(f"{path.rsplit('/', 1)[-1]:11} {audio.size / 16000:5.1f} s -> median {statistics.median(times):5.2f} s  (all {' '.join(f'{x:.2f}' for x in times)})  {text[:70]!r}", flush=True)
