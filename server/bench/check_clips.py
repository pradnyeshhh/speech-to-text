import difflib, json, re, sys, urllib.request, wave, io
import numpy as np
clips = sys.argv[1]
ref = json.load(open(f"{clips}/clips.json"))
norm = lambda t: re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " ")).split()
def post(data):
    req = urllib.request.Request("http://127.0.0.1:8095/transcribe", data=data, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))
for name in ("short", "medium", "long"):
    r = post(open(f"{clips}/{name}.wav", "rb").read())
    a, b = norm(ref[name]["text"]), norm(r["text"])
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    errs = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    print(f"{name:6} {r['audio_s']:5.1f} s audio -> {r['seconds']:5.1f} s   {len(b)}/{len(a)} words, ~{errs} differ")
    for tag, i1, i2, j1, j2 in ops:
        print(f"         {tag}: {' '.join(a[i1:i2])!r} -> {' '.join(b[j1:j2])!r}")
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((np.random.default_rng(1).normal(0, 30, 48000)).astype(np.int16).tobytes())
print("silence ", post(buf.getvalue()))
