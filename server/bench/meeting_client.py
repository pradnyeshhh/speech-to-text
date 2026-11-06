"""Plays a two-channel meeting into /live (planar blocks: microphone, then tab) at 4x real time."""
import asyncio, json, sys, wave
import aiohttp, numpy as np
sock, echo = sys.argv[1], float(sys.argv[2])
over = len(sys.argv) > 3  # you also talk over the meeting, at 0:09
def clip(name):
    with wave.open(f"/root/voice_to_text/bench/data/clips/{name}.wav") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
R, total = 16000, 28
you, them = np.zeros(total * R, np.float32), np.zeros(total * R, np.float32)
short, medium = clip("short"), clip("medium")
you[1 * R : 1 * R + short.size] += short
you[22 * R : 22 * R + short.size] += short
them[6 * R : 6 * R + medium.size] += medium
if over:
    you[9 * R : 9 * R + short.size] += short
you += echo * them  # laptop speakers: the microphone hears the meeting too
you += np.random.default_rng(0).normal(0, 3e-4, you.size).astype(np.float32)
to16 = lambda x: (np.clip(x, -1, 1) * 32767).astype("<i2")
async def main():
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock)) as http:
        async with http.ws_connect("http://x/live") as ws:
            await ws.send_json({"type": "start", "rate": R, "mode": "meeting"})
            print("<-", await ws.receive_json())
            n = R // 10
            for i in range(0, you.size, n):
                await ws.send_bytes(to16(you[i:i + n]).tobytes() + to16(them[i:i + n]).tobytes())
                await asyncio.sleep(0.025)
            await ws.send_json({"type": "stop"})
            async for msg in ws:
                m = json.loads(msg.data)
                if m["type"] in ("segment", "revise"): print(m["type"], m["speaker"], m["start"], m.get("text", "")[:60])
                if m["type"] == "done":
                    print("done: mode", m["mode"], "| silent channels", m["silent_channels"], "| summary", m["summary_status"])
                    print(m["text"]); break

asyncio.run(main())
