import asyncio
import sys
import time
from pathlib import Path

import ormsgpack
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings  # noqa: E402


async def main():
    headers = {"Authorization": f"Bearer {settings.fish_api_key}",
               "model": "s2.1-pro-free"}
    ws = await websockets.connect(
        "wss://api.fish.audio/v1/tts/live",
        additional_headers=headers, ping_interval=None)
    start = {"event": "start", "request": {
        "text": "", "sample_rate": 24000, "latency": "balanced",
        "format": "pcm", "normalize": True,
        "prosody": {"speed": 1.0, "volume": 0},
        "reference_id": settings.fish_voice_id,
        "temperature": 0.7, "top_p": 0.75}}
    await ws.send(ormsgpack.packb(start))
    await asyncio.sleep(0.3)
    await ws.send(ormsgpack.packb({"event": "text", "text": "hey how are you"}))
    await ws.send(ormsgpack.packb({"event": "flush"}))
    t0 = time.monotonic()
    n = 0
    try:
        while time.monotonic() - t0 < 25:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
            except asyncio.TimeoutError:
                print("recv timeout")
                break
            msg = ormsgpack.unpackb(raw)
            if isinstance(msg, dict) and msg.get("event") == "audio":
                n += len(msg["audio"])
                print(f"audio +{len(msg['audio'])}B @ {time.monotonic()-t0:.2f}s")
            else:
                print("EV:", str(msg)[:200])
            if isinstance(msg, dict) and msg.get("event") == "finish":
                break
    except Exception as e:
        print("closed:", repr(e)[:150])
    print(f"total {n} bytes in {time.monotonic()-t0:.2f}s")


asyncio.run(main())
