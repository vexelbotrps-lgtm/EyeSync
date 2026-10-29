#!/usr/bin/env python3
"""Mock EyeSync 1.3 for testing the dashboard without a webcam or Windows.

It plays both programs that answer on EyeSync's ports (see EyeSync's WEBSITE.md):

* the standby, while "EyeSync is closed": ?probe=1 gets {"type": "eyesync", "state": "standby"}
  and close code 4003; ?start=1 gets "starting" and 4002, and EyeSync "starts" --start-delay
  seconds later. Plain connections are refused (HTTP 503), as by the real standby.
* EyeSync itself: ?probe=1 gets "running" with the status and 4003. Any other connection gets
  about 30 cursor updates a second and can send commands; status changes are pushed, and the
  camera view sends small JPEG pictures (needs Pillow). "quit" closes pages with 4000.
  --version 1.2.0 plays an EyeSync without gaze jump, 1.1.4 one without voice typing too.

    pip install websockets pillow
    python tools/mock_eyesync.py                      # starts in standby, as after login
    python tools/mock_eyesync.py --state running
    python tools/mock_eyesync.py --state closed       # as after the user quit EyeSync

Type these while it runs to act "on the PC" (status changes reach the dashboard by themselves):
    smile          pause or resume, like a big smile
    speed N        set the speed (1-5), like the hotkey
    face off|on    lose or find the face (sets the advice text)
    hint [TEXT]    set or clear the advice text
    eyes           close both eyes for 2 s: voice typing on or off
    say [TEXT]     talk while voice typing listens (the words are only printed here, never sent)
    mic off|on     lose or get back the microphone (voice typing shows an error)
    calibrate      start the calibration, like F10 or a head tilt to the right
    calibrate eyes start the eye calibration, like the window's Calibrate eyes button
                   (with "face off" it fails at the end and keeps the previous one)
    quit           quit EyeSync, like the mouth gesture (it then needs Connect)
    idle           close EyeSync as if the dashboard had been closed (auto-connect may restart it)

Only pages from the origins given with --origin may use it, like the real EyeSync.
"""
import argparse
import asyncio
import base64
import io
import json
import math
import random
import sys
import threading
import time
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

try:
    from PIL import Image, ImageDraw
except ImportError:          # the camera view is optional
    Image = None

VERSION = "1.3.0"
DEFAULT_ORIGINS = [
    "https://aniruddh9140.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
STEPS = [("starting", "Get ready: sit as you normally do"), ("centre", "Look at the centre dot"),
         ("left", "Turn your head to the left dot"), ("right", "Turn your head to the right dot"),
         ("up", "Tilt your head up to the top dot"), ("down", "Tilt your head down to the bottom dot"),
         ("blink", "Blink twice, slowly"), ("smile", "Give a big smile")]
GAZE_STEPS = [("starting", "Starting the camera..."),
              ("gaze1", "Look at the dot in the middle - just look, keep your head natural")] +              [(f"gaze{i}", "Follow the dot with your eyes") for i in range(2, 10)]
STEP_S = 1.6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ports", type=int, nargs="+", default=[8765, 8080])
    p.add_argument("--state", choices=["standby", "running", "closed"], default="standby")
    p.add_argument("--start-delay", type=float, default=5.0, help="seconds EyeSync takes to start")
    p.add_argument("--idle-exit", type=float, default=30.0, help="close EyeSync this long after the last page left")
    p.add_argument("--version", default=VERSION, help="version to report (try 1.0.0 to see the update notice)")
    p.add_argument("--origin", action="append", help="allowed page origin; repeat for several (replaces the defaults)")
    return p.parse_args()


class Mock:
    def __init__(self, args):
        self.args = args
        self.origins = set(args.origin or DEFAULT_ORIGINS)
        self.mode = "running" if args.state == "running" else "standby"   # standby | starting | running
        self.closed_by_user = args.state == "closed"
        self.clients, self.controllers, self.previews = set(), set(), set()
        self.last_seen = time.monotonic()
        self.s = {"mouse_enabled": True, "paused": False, "face": True, "speed": 2, "click_mode": "blink",
                  "gestures": True, "calibrating": False, "calibrated": "", "step": "", "text": "",
                  "progress": 0.0, "hint": "", "voice_on": False, "voice_state": "off", "voice_hint": "",
                  "gaze_calibrated": "", "gaze_on": False, "gaze_error_px": 0}
        self.cal_steps = STEPS    # the running (or last) calibration's: STEPS, or GAZE_STEPS for the eyes
        self.mic = True
        self.voice_until = 0.0     # when "hearing"/"working"/"error" ends
        self.cal_started = 0.0
        self.target = None       # the website's accuracy-test target, in frame px
        self.x, self.y, self.t0 = 400.0, 300.0, time.monotonic()

    # ---- what EyeSync reports ----
    def status(self):
        s = self.s
        return {"type": "status", "version": self.args.version, "mouse_enabled": s["mouse_enabled"],
                "paused": s["paused"], "active": bool(self.clients) and s["mouse_enabled"] and not s["paused"],
                "face": s["face"], "camera": True, "speed": s["speed"], "click_mode": s["click_mode"],
                "gestures": s["gestures"], "calibrating": s["calibrating"], "calibrated": s["calibrated"],
                "calibration": {"step": s["step"], "text": s["text"], "progress": round(s["progress"], 2)},
                "hint": s["hint"], "hardware_mouse": False,
                **({"gaze": {"calibrated": s["gaze_calibrated"], "on": s["gaze_on"],
                             "error_px": s["gaze_error_px"]}} if self.has_gaze() else {}),
                **({"voice": {"on": s["voice_on"], "state": s["voice_state"], "hint": s["voice_hint"]}}
                   if self.has_voice() else {})}

    def at_least(self, *version):
        try:
            return tuple(int(n) for n in self.args.version.split(".")[:2]) >= version
        except ValueError:
            return True

    def has_voice(self):          # voice typing came with 1.2.0; older versions send no "voice"
        return self.at_least(1, 2)

    def has_gaze(self):           # gaze jump came with 1.3.0; older versions send no "gaze"
        return self.at_least(1, 3)

    def voice(self, on):
        s = self.s
        if on and not self.mic:
            s.update(voice_on=False, voice_state="error", voice_hint="Can't use the microphone. Check that one is connected.")
            self.voice_until = time.monotonic() + 5
        elif on:
            s.update(voice_on=True, voice_state="starting", voice_hint="")
            self.voice_until = time.monotonic() + 1.0
        else:
            s.update(voice_on=False, voice_state="off", voice_hint="")

    def step_voice(self, now):
        s = self.s
        if now < self.voice_until:
            return
        if s["voice_state"] == "hearing":
            s["voice_state"], self.voice_until = "working", now + 1.0
        elif s["voice_on"] and s["voice_state"] != "listening":
            s["voice_state"] = "listening"
        elif not s["voice_on"] and s["voice_state"] == "error":
            s.update(voice_state="off", voice_hint="")

    def allowed(self, request):
        return request.headers.get("Origin", "") in self.origins

    # ---- the handshake: the standby refuses plain connections with 503 ----
    def process_request(self, connection, request):
        query = parse_qs(urlsplit(request.path).query)
        asks = query.get("probe") == ["1"] or query.get("start") == ["1"]
        if self.mode == "starting" or (self.mode == "standby" and not (asks and self.allowed(request))):
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "EyeSync is closed\n")
        return None

    async def handler(self, ws):
        query = parse_qs(urlsplit(ws.request.path).query)
        if self.mode == "standby":                       # only ?probe=1 / ?start=1 get here
            start = query.get("start") == ["1"]
            await ws.send(json.dumps({"type": "eyesync", "state": "starting" if start else "standby",
                                      "version": self.args.version, "closed_by_user": self.closed_by_user}))
            if start:
                print("standby: the website pressed Connect - starting EyeSync")
                self.mode = "starting"
                asyncio.get_running_loop().call_later(self.args.start_delay, self.started)
                await ws.close(4002, "EyeSync is starting")
            else:
                await ws.close(4003, "EyeSync is installed and waiting - connect with ?start=1 to start it")
            return
        if not self.allowed(ws.request):
            await ws.close(4001, "API key required")
            return
        if query.get("probe") == ["1"]:
            await ws.send(json.dumps({**self.status(), "type": "eyesync", "state": "running", "closed_by_user": False}))
            await ws.close(4003, "EyeSync is running - connect without ?probe=1")
            return
        self.clients.add(ws)
        print(f"website connected ({len(self.clients)} open)")
        try:
            await ws.send(self.payload(False))
            async for message in ws:
                reply = self.command(ws, message)
                if reply is not None:
                    self.controllers.add(ws)
                    await ws.send(json.dumps(reply))
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.controllers.discard(ws)
            self.previews.discard(ws)
            self.last_seen = time.monotonic()
            if not self.clients:
                self.s["mouse_enabled"] = True       # the next page says for itself
            print(f"website disconnected ({len(self.clients)} open)")

    def started(self):
        self.mode, self.closed_by_user = "running", False
        self.last_seen = time.monotonic()
        print("EyeSync is running")

    # ---- commands from the page ----
    def command(self, ws, text):
        try:
            msg = json.loads(text)
        except ValueError:
            return {"type": "error", "error": "messages must be JSON"}
        if isinstance(msg, dict) and msg.get("type") == "calibration":
            event = msg.get("event")
            print(f"website accuracy test: {event}")
            src = msg.get("source") if event == "point" else None
            if isinstance(src, dict) and all(isinstance(src.get(k), (int, float)) for k in "xy"):
                self.target = (src["x"], src["y"])
            elif event in ("complete", "cancel"):
                self.target = None
            return None
        if not isinstance(msg, dict) or not isinstance(msg.get("action"), str):
            return {"type": "error", "error": 'expected {"action": ...}'}
        action, value, s = msg["action"], msg.get("value"), self.s
        print(f"command: {action} {json.dumps(value) if 'value' in msg else ''}")
        if action == "mouse" and isinstance(value, bool):
            s["mouse_enabled"] = value
        elif action == "pause" and isinstance(value, bool):
            s["paused"] = value
        elif action == "recenter":
            self.x, self.y = 400.0, 300.0
        elif action == "speed" and value in (1, 2, 3, 4, 5) and not isinstance(value, bool):
            s["speed"] = value
        elif action == "click_mode" and value in ("blink", "dwell"):
            s["click_mode"] = value
        elif action == "gestures" and isinstance(value, bool):
            s["gestures"] = value
        elif action == "calibrate":
            if value == "cancel":
                s.update(calibrating=False, step="", text="", progress=0.0)      # as EyeSync does
            else:
                self.calibrate(eyes=value == "gaze" and self.has_gaze())
        elif action == "gaze" and self.has_gaze() and isinstance(value, bool):
            s["gaze_on"] = value
        elif action == "voice" and self.has_voice() and (isinstance(value, bool) or "value" not in msg):
            self.voice(value if isinstance(value, bool) else not s["voice_on"])
        elif action == "preview" and isinstance(value, bool):
            (self.previews.add if value else self.previews.discard)(ws)
        elif action == "show_window":
            print("(the EyeSync window would come to the front now)")
        elif action == "quit":
            asyncio.ensure_future(self.quit(by_user=True))      # after the reply, like EyeSync
        elif action != "status":
            return {"type": "error", "error": f"unknown command {msg!r}"[:200]}
        return self.status()

    def calibrate(self, eyes):
        if not self.s["calibrating"]:          # one at a time; both start at "starting"
            self.cal_steps = GAZE_STEPS if eyes else STEPS
            self.s.update(calibrating=True, step="starting", text=self.cal_steps[0][1], progress=0.0)
            self.cal_started = time.monotonic()

    async def quit(self, by_user):
        print("EyeSync closes" + (" (quit by the user)" if by_user else " (no dashboard)"))
        self.mode, self.closed_by_user = "standby", by_user
        self.s.update(calibrating=False, paused=False)
        self.voice(False)
        await asyncio.gather(*(ws.close(4000, "EyeSync was closed - press Connect to start it again")
                               for ws in list(self.clients)), return_exceptions=True)

    # ---- what EyeSync sends ----
    def payload(self, blink):
        return json.dumps({"x": round(self.x, 2), "y": round(self.y, 2), "blink": blink,
                           "confidence": 1.0 if self.s["face"] else 0.0, "face": self.s["face"]})

    def picture(self):
        img = Image.new("RGB", (320, 240), (18, 26, 20))
        d = ImageDraw.Draw(img)
        cx, cy = 320 - self.x / 800 * 320, self.y / 600 * 240          # mirrored, like a selfie
        if self.s["face"]:
            d.ellipse((cx - 46, cy - 60, cx + 46, cy + 60), outline=(143, 227, 168), width=2)
            for i in range(60):                                             # a pretend face mesh
                a, r = i * 2.4, 8 + (i * 7) % 44
                d.point((cx + r * math.cos(a) * 0.9, cy + r * math.sin(a) * 1.2), fill=(143, 227, 168))
            d.ellipse((cx - 24, cy - 18, cx - 10, cy - 10), fill=(240, 240, 240))
            d.ellipse((cx + 10, cy - 18, cx + 24, cy - 10), fill=(240, 240, 240))
        d.text((8, 8), "MOCK CAMERA", fill=(143, 227, 168))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def step_calibration(self, now):
        s = self.s
        if not s["calibrating"]:
            if s["step"] == "done" and now - self.cal_started > len(self.cal_steps) * STEP_S + 6:
                s.update(step="", text="", progress=0.0)      # the result is shown for a few seconds
            return
        t = now - self.cal_started
        i = int(t // STEP_S)
        if i >= len(self.cal_steps) and self.cal_steps is GAZE_STEPS:
            self.finish_eyes()
            return
        if i >= len(self.cal_steps):
            s.update(calibrating=False, step="done", text="Calibration saved. EyeSync now fits you.",
                     progress=1.0, calibrated=time.strftime("%Y-%m-%d"))
            return
        step, text = self.cal_steps[i]
        s.update(step=step, text=text, progress=min(1.0, (t % STEP_S) / STEP_S))

    def finish_eyes(self):
        """Like EyeSync: gaze jump goes on if the result is good enough; a failed one keeps the old one."""
        s = self.s
        if not s["face"]:
            text = "Eye calibration didn't work: it couldn't see your eyes at the middle dot. Try again with your face well lit."
        else:
            err = random.randint(70, 160)
            s.update(gaze_calibrated=time.strftime("%Y-%m-%d"), gaze_error_px=err, gaze_on=True)
            text = (f"Eye calibration done! Your gaze is known to about {err} px. "
                    "Gaze jump is on: look at a spot and nudge your head towards it.")
        s.update(calibrating=False, step="done", text=text, progress=1.0)

    async def run(self):
        next_at = time.monotonic()
        last_status, next_status, next_picture = "", 0.0, 0.0
        while True:
            now = time.monotonic()
            t = now - self.t0
            if self.mode == "running" and not self.clients and now - self.last_seen > self.args.idle_exit:
                await self.quit(by_user=False)
            if self.target:
                tx, ty = self.target[0] + 9, self.target[1] - 6
            else:
                tx, ty = 400 + 300 * math.sin(t * 0.5), 300 + 210 * math.sin(t * 0.8 + 0.6)
            self.x += (tx - self.x) * 0.08
            self.y += (ty - self.y) * 0.08
            self.step_calibration(now)
            self.step_voice(now)
            blink = self.s["click_mode"] == "blink" and self.target is None and (t % 4.5) < 0.2
            for ws in list(self.clients):
                asyncio.ensure_future(self.safe_send(ws, self.payload(blink)))
            if self.controllers and now >= next_status:
                next_status = now + 0.25
                status = json.dumps(self.status())
                if status != last_status:
                    last_status = status
                    for ws in list(self.controllers):
                        asyncio.ensure_future(self.safe_send(ws, status))
            if self.previews and Image is not None and now >= next_picture:
                next_picture = now + 1 / 8
                picture = json.dumps({"type": "preview", "image": self.picture()})
                for ws in list(self.previews):
                    asyncio.ensure_future(self.safe_send(ws, picture))
            next_at += 1 / 30
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))

    @staticmethod
    async def safe_send(ws, text):
        try:
            await ws.send(text)
        except ConnectionClosed:
            pass

    # ---- "on the PC" ----
    def pc(self, line):
        words = line.split()
        if not words:
            return
        cmd, rest, s = words[0].lower(), words[1:], self.s
        if cmd == "smile":
            s["paused"] = not s["paused"]
        elif cmd == "speed" and rest and rest[0] in "12345":
            s["speed"] = int(rest[0])
        elif cmd == "face" and rest:
            s["face"] = rest[0] == "on"
            s["hint"] = "" if s["face"] else "Can't find your face. Move into the camera's view."
        elif cmd == "hint":
            s["hint"] = line.split(None, 1)[1] if rest else ""
        elif cmd == "eyes" and self.has_voice():
            self.voice(not s["voice_on"])
        elif cmd == "say" and s["voice_on"]:
            text = line.split(None, 1)[1].strip() if rest else ""
            if text.lower().rstrip(".!") == "stop listening":
                self.voice(False)
            else:
                s["voice_state"], self.voice_until = "hearing", time.monotonic() + 1.5
                print(f"(typed on the PC, not sent to the page: {text!r})")
        elif cmd == "calibrate":
            self.calibrate(eyes=rest[:1] == ["eyes"] and self.has_gaze())
        elif cmd == "mic" and rest:
            self.mic = rest[0] == "on"
            if not self.mic and s["voice_on"]:
                self.voice(True)                 # the microphone went away while listening
        elif cmd == "quit" or (cmd == "idle" and not self.clients):
            asyncio.ensure_future(self.quit(by_user=cmd == "quit"))
            return
        elif cmd == "idle":
            print("EyeSync only closes by itself once no page is connected: close the dashboard first")
            return
        else:
            print("commands: smile | speed N | face on|off | hint [TEXT] | eyes | say [TEXT] | mic off|on | "
                  "calibrate [eyes] | quit | idle")
            return
        print(f"PC: {line.strip()}")


async def main():
    args = parse_args()
    mock = Mock(args)
    loop = asyncio.get_running_loop()

    def read_stdin():
        for line in sys.stdin:
            loop.call_soon_threadsafe(mock.pc, line)
    threading.Thread(target=read_stdin, daemon=True).start()

    servers = [await serve(mock.handler, "localhost", port, process_request=mock.process_request, max_size=4096)
               for port in args.ports]
    print(f"Mock EyeSync {args.version} on ports {', '.join(map(str, args.ports))}, {mock.mode}"
          f"{' (closed by the user)' if mock.closed_by_user else ''}")
    print("Allowed origins: " + ", ".join(sorted(mock.origins)))
    if Image is None:
        print("Pillow isn't installed, so the camera view stays empty (pip install pillow).")
    try:
        await mock.run()
    finally:
        for server in servers:
            server.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
