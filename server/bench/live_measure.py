"""Plays a clip into /live in real time; prints one JSON line: arrival times, wait after stop, word accuracy."""
import asyncio, difflib, json, re, sys, time, wave
import aiohttp, numpy as np
sock, name = sys.argv[1], sys.argv[2]
clips = "/root/voice_to_text/bench/data/clips"
with wave.open(f"{clips}/{name}.wav") as w:
    pcm = w.readframes(w.getnframes())
ref = json.load(open(f"{clips}/clips.json"))[name]["text"]
async def main():
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock)) as http:
        async with http.ws_connect("http://x/live") as ws:
            await ws.send_json({"type": "start", "rate": 16000}); await ws.receive_json()
            t0 = time.monotonic(); out = {"segments": [], "partials": 0}
            async def send():
                for i in range(0, len(pcm), 3200):
                    await ws.send_bytes(pcm[i:i + 3200]); await asyncio.sleep(0.1)
                await asyncio.sleep(0.3); out["stop_at"] = time.monotonic() - t0
                await ws.send_json({"type": "stop"})
            task = asyncio.create_task(send())
            async for msg in ws:
                m = json.loads(msg.data); t = round(time.monotonic() - t0, 2)
                if m["type"] == "partial": out["partials"] += 1
                elif m["type"] == "segment": out["segments"].append(t)
                elif m["type"] == "done":
                    out["wait_after_stop"] = round(t - out["stop_at"], 2); out["text"] = m["text"]; break
            await task
    norm = lambda x: re.sub(r"[^a-z' ]", " ", x.lower().replace("-", " ")).split()
    a, b = norm(ref), norm(out["text"])
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    out["word_errors"] = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops); out["words"] = len(a)
    out["diffs"] = [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for _, i1, i2, j1, j2 in ops]
    out["clip"] = name; out["stop_at"] = round(out["stop_at"], 2); del out["text"]
    print(json.dumps(out), flush=True)
asyncio.run(main())
