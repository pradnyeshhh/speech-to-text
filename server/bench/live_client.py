"""Plays a WAV into the server's /live WebSocket like the site does and prints what comes back.

usage: live_client.py SOCKET WAV RATE SPEED     RATE: the rate to send at (resampled here); SPEED: x real time
"""
import asyncio, json, sys, time, wave
import aiohttp, av, numpy as np

sock, path, rate, speed = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
with wave.open(path) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
if rate != 16000:  # what a browser at `rate` would send
    r = av.AudioResampler(format="flt", layout="mono", rate=rate)
    f = av.AudioFrame.from_ndarray(audio[None], format="flt", layout="mono"); f.sample_rate = 16000
    audio = np.concatenate([x.to_ndarray().reshape(-1) for x in r.resample(f) + r.resample(None)])
pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
block = rate // 10

async def main():
    t0 = time.monotonic()
    stamp = lambda: f"{time.monotonic() - t0:6.2f}s"
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock)) as http:
        async with http.ws_connect("http://x/live") as ws:
            await ws.send_json({"type": "start", "rate": rate})
            print(stamp(), "<-", (await ws.receive_json()))
            async def send():
                for i in range(0, pcm.size, block):
                    await ws.send_bytes(pcm[i:i + block].tobytes())
                    await asyncio.sleep(0.1 / speed)
                await asyncio.sleep(0.3)
                print(stamp(), "-> stop", flush=True)
                await ws.send_json({"type": "stop"})
            sender = asyncio.create_task(send())
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                m = json.loads(msg.data)
                short = {k: (v[:70] + "…" if isinstance(v, str) and len(v) > 70 else v) for k, v in m.items()}
                print(stamp(), "<-", short, flush=True)
                if m["type"] == "summary" or (m["type"] == "done" and m["summary_status"] != "pending"):
                    break
            await sender

asyncio.run(main())
