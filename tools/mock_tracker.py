#!/usr/bin/env python3
"""Mock eye-tracker for the EyeSync dashboard.

Streams gaze frames such as {"x": 400, "y": 300, "blink": false} over a local
WebSocket so you can test the dashboard without hardware. It also prints the
calibration events the dashboard sends back, and during calibration it "looks"
at the active target so the results table has real numbers in it.

    pip install websockets
    python tools/mock_tracker.py                 # ws://localhost:8765, 800x600 frame, 60 fps
    python tools/mock_tracker.py --rate 30 --width 1920 --height 1080

Only pages from the origins listed with --origin may connect. This stops any
other website you have open from reading your gaze stream.
"""
import argparse
import asyncio
import json
import math
import random
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

DEFAULT_ORIGINS = [
    "https://aniruddh9140.github.io",  # GitHub Pages site for this repository
    "http://localhost:8000",           # python -m http.server
    "http://127.0.0.1:8000",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--width", type=int, default=800, help="tracker frame width in px (match the dashboard)")
    p.add_argument("--height", type=int, default=600, help="tracker frame height in px (match the dashboard)")
    p.add_argument("--rate", type=float, default=60.0, help="frames per second")
    p.add_argument("--origin", action="append", help="allowed page origin; repeat for several (replaces the defaults)")
    p.add_argument("--allow-file", action="store_true", help='also allow pages opened from disk (Origin: "null")')
    return p.parse_args()


async def gaze_stream(ws, args):
    target = None  # (x, y) in frame px while a calibration target is showing

    async def read_events():
        nonlocal target
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "calibration":
                continue
            event = msg.get("event")
            print(f"calibration {event}: {json.dumps({k: v for k, v in msg.items() if k not in ('type', 'event')})}")
            if event == "point":
                src = msg.get("source") or {}
                if isinstance(src.get("x"), (int, float)) and isinstance(src.get("y"), (int, float)):
                    target = (src["x"], src["y"])
            elif event in ("complete", "cancel"):
                target = None

    reader = asyncio.create_task(read_events())
    t0 = next_at = time.monotonic()
    x, y = args.width / 2, args.height / 2
    w, h = args.width, args.height
    try:
        while True:
            t = time.monotonic() - t0
            if target:
                tx, ty = target[0] + w * 0.012, target[1] - h * 0.01
            else:
                tx = w / 2 + w * 0.38 * math.sin(t * 0.7)
                ty = h / 2 + h * 0.36 * math.sin(t * 1.1 + 0.6)
            x += (tx - x) * 0.12
            y += (ty - y) * 0.12
            blink = target is None and (t % 4.2) < 0.16
            jitter = max(w, h) * 0.004
            frame = {"x": round(x + random.gauss(0, jitter)), "y": round(y + random.gauss(0, jitter)), "blink": blink}
            await ws.send(json.dumps(frame))
            # Fixed schedule: Windows timers are coarse, so a plain sleep(1/rate) undershoots the rate.
            next_at += 1 / args.rate
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
    except ConnectionClosed:
        pass
    finally:
        reader.cancel()


async def main():
    args = parse_args()
    origins = list(args.origin or DEFAULT_ORIGINS)
    if args.allow_file:
        origins.append("null")

    async def handler(ws):
        print(f"dashboard connected from {ws.request.headers.get('Origin')}")
        await gaze_stream(ws, args)
        print("dashboard disconnected")

    async with serve(handler, args.host, args.port, origins=origins):
        print(f"Mock tracker on ws://{args.host}:{args.port}  frame {args.width}x{args.height} @ {args.rate:g} fps")
        print("Allowed origins: " + ", ".join(origins))
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
