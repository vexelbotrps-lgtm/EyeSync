"""
live_inference.py - EyeSync: head-controlled cursor + broadcast.
================================================================

Tracks the user's face with the webcam, moves the cursor with the head (see
HeadPointer) and clicks with blinks and winks, and sends the smoothed screen
position and clicks to:

  A. a mouse - one of:
       * the Seeed XIAO ESP32-S3 (firmware in firmware/xiao_esp32s3_mouse/),
         acting as a real USB mouse, over USB serial at 115200 baud:
             M,SCREEN,<W>,<H>\\n   sent on connect and when the display changes resolution
             M,MOVE,<X>,<Y>\\n     absolute screen pixels
             M,CLICK,0,0\\n        one per left click (blink, or left-eye wink)
             M,RCLICK,0,0\\n       one per right click (long blink, or right-eye wink)
             M,PING,0,0\\n         keep-alive / unplug detection
       * the computer's own cursor (standalone mode, no hardware needed).
     --mouse auto (default) uses the XIAO whenever it is plugged in and falls
     back to software control when it is not - never both at once.

  B. WebSocket servers for the web dashboard, at ~30 messages per second:
       ws://localhost:8080/eye-tracker   <- what the EyeSync website connects to
       ws://localhost:8765               <- generic endpoint
     Other websites and apps must add ?key=<API key> (see ApiKey).
     Payload: {"x": float, "y": float, "blink": bool, "confidence": float,
               "face": bool}
     The dashboard's buttons send commands back (JSON, see handle_command):
       {"action": "mouse", "value": true|false}   Enable / Disable Mouse Control
       {"action": "recenter"}                     cursor to the centre
       {"action": "calibrate"}  (value "cancel" stops it)   see Calibration
       {"action": "calibrate", "value": "gaze"}   the eye calibration (Gaze jump)
       {"action": "gaze", "value": true|false}    Gaze jump on / off
       {"action": "speed", "value": 1-5}, {"action": "click_mode", "value":
       "blink"|"dwell"}, {"action": "gestures", "value": true|false},
       {"action": "voice", "value": true|false} (no value: on <-> off) voice typing,
       {"action": "status"}
     and get {"type": "status", ...} back (only clients that send commands
     get status messages, so an older dashboard never sees one).
     x/y are expressed in an 800x600 reference space by default: the
     website's "Tracker frame" setting, which it scales to its window (use
     --dashboard-size native to send raw screen pixels instead).
     The website's own 9-point calibration sends {"type": "calibration",
     "event": ...} messages; EyeSync logs them and doesn't reply.

Close gesture
-------------
Open your mouth WIDE and hold it for 3 seconds: a warning beep sounds at the
start, the camera view shows a countdown, and EyeSync then shuts down completely
(EyeSync.pyw starts it again only when the website is next opened, see
site_access.wait_for_visit). Close your mouth to cancel. Normal
talking never opens the mouth wide enough, and the eyes must be open when it
starts, so a yawn (which squeezes the eyes) doesn't trigger it. Once it has
started, squinting and brief wavering of the mouth (under 0.4 s) are
tolerated - holding a mouth wide open naturally does both.
--no-close-gesture disables it.

Scrolling
---------
Tilt the head - an ear towards a shoulder - and hold it: left = scroll up,
right = scroll down, faster the further you tilt; straighten up to stop
(TiltScroll). The wheel is Windows' own (scroll_task), also with the XIAO.

Gaze jump: look, then nudge
---------------------------
After the eye calibration (~18 s: follow a dot with your eyes), EyeSync also
knows roughly where you look (gaze_engine - a webcam eye model adapted from
JEOresearch's EyeTracker). Look at a spot far from the cursor and start
moving your head towards it: the cursor jumps there, and your head does the
fine aiming as always. Glancing around, reading or holding the head still
never moves the cursor. See VisionWorker._gaze_jump.

Voice typing
------------
Close both eyes for 1 second (VOICE_EYES_CLOSED_S; a rising beep confirms
it) and say what to type. Close both eyes for 1 second again (falling beep)
to stop. The voice bar's Stop button also works. Silence has no time limit.
See voice_typing.py; the microphone is on only while voice typing is.

Activation
----------
The AI only runs while the EyeSync website's dashboard is connected,
identified by the browser's Origin header (the site's domain), or another
website or app that knows EyeSync's API key (?key=... in the URL, see
ApiKey). Nothing else can connect, switch it on or read its data:
  * not connected  -> camera OFF (released, webcam light off), no face
                      tracking, no cursor control. Only the WebSocket
                      server listens, waiting for the website.
  * website opens  -> the camera starts (~1-3 s); cursor control is on.
  * website closes -> cursor control stops after RECONNECT_GRACE_S; the
                      camera shuts down after --idle-grace seconds (default
                      5), so a page reload doesn't restart everything.
Use --always-active to run without the website at all. The camera also
runs during a calibration the user starts (Calibrate, F10).

Usage:
    python live_inference.py                 # everything automatic
    python live_inference.py --port COM5     # force a serial port
    python live_inference.py --mouse none    # dashboard only, no cursor control

Global hotkeys - work even when the window is minimised or another app is
focused. Ctrl+Alt+Shift + a function key is a combination ordinary apps don't
use; if another program already owns one, EyeSync leaves it alone.
    Ctrl+Alt+Shift+F6   recenter: put the cursor in the middle of the screen
    Ctrl+Alt+Shift+F7   calmer cursor (lower sensitivity)
    Ctrl+Alt+Shift+F8   more responsive cursor
    Ctrl+Alt+Shift+F9   pause / resume cursor control
    Ctrl+Alt+Shift+F10  calibrate (again: cancel) - see Calibration
    Ctrl+Alt+Shift+F11  show the EyeSync window
    Ctrl+Alt+Shift+F12  quit                (Ctrl+C in the console also quits)

Stability design
----------------
* Vision runs in a worker thread; a supervisor restarts it if it crashes or
  stalls (e.g. a webcam driver hanging inside cap.read()).
* Per-frame errors are logged and skipped; the face mesh is rebuilt if they
  persist. The camera is reopened automatically if it is unplugged.
* Serial and WebSocket are independent asyncio tasks: an unplugged XIAO, a
  busy port or a dead browser tab never stalls anything else. The serial
  link reconnects by itself; WebSocket ports retry if they are taken.
* WebSocket messages are always well-formed (finite numbers for x/y, sent
  even before the first face is seen), because the website calls x.toFixed().
* Windows background throttling is disabled for this process, so tracking
  runs at full speed while the window is minimised.
* Everything is logged to eyesync.log (rotating) for post-mortem debugging.
"""

from __future__ import annotations

import os
# Must be set before OpenCV loads: with Media Foundation's hardware transforms
# enabled, opening a webcam takes ~15 s on many Windows PCs (see
# face_engine.open_camera).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import asyncio
import base64
import dataclasses
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from websockets.asyncio.server import broadcast, serve
from websockets.exceptions import ConnectionClosed

import app_paths
import desktop
import face_engine as fe
import gaze_engine as ge
import site_access
import voice_typing
from site_access import (API_KEY_PATH, DEFAULT_SITE_ORIGINS, ApiKey,  # noqa: F401
                         is_website_origin)
from control_backend.mouse import DesktopMouse as SoftwareMouse
from control_backend.adapter import ControlBackend

log = logging.getLogger("eyesync")

GESTURE_DROPOUT_S = 0.7     # face dropouts shorter than this don't cancel the close gesture
GESTURE_KEEP = 0.65         # once started, the mouth must stay above 65% of the start threshold
GESTURE_GRACE_S = 0.6       # ... except for sags shorter than this (a real user needed 6 tries
                            # with 75% / 0.4 s)
GESTURE_CLOSED_MAR = 0.2    # a mouth this closed is a deliberate cancel:
GESTURE_CANCEL_S = 0.25     # ... it cancels after only this long
GESTURE_CONFIRM_S = 0.5     # the mouth must stay wide open this long before the countdown beeps
CLOSE_JAW_MIN = 0.3         # MediaPipe's jawOpen score needed to start the close gesture
# EYESYNC_LOG overrides the log file (used by tests so they never touch the real log).
LOG_PATH = Path(os.environ.get("EYESYNC_LOG", fe.PROJECT_DIR / "eyesync.log"))
VERSION = site_access.VERSION   # shown in the log and to the website; names the release zip

# USB vendor IDs used for serial auto-detection, in priority order:
# Seeed Studio (XIAO), Espressif native USB (S2/S3/C3), Silicon Labs CP210x,
# WCH CH340/CH9102, FTDI.
ESP32_USB_VIDS = (0x2886, 0x303A, 0x10C4, 0x1A86, 0x0403)


# --------------------------------------------------------------------------- #
# Shared state between the vision thread and the async outputs
# --------------------------------------------------------------------------- #

@dataclass
class Snapshot:
    x: float                 # smoothed cursor position, screen pixels
    y: float
    eyes_closed: bool        # eyes currently closed (cursor frozen)
    click_count: int         # monotonically increasing number of (left) clicks
    face: bool               # a face is currently detected
    has_position: bool       # the face has been seen at least once
    confidence: float        # 0..1, see VisionWorker._process
    paused: bool             # user paused cursor control (F9)
    esp32_connected: bool    # XIAO serial link is up
    ws_clients: int          # connected WebSocket clients (any origin)
    site_clients: int        # the website / apps with the key (incl. one reconnecting) -> control
    sensitivity: int         # 1..5, see SENSITIVITY_LEVELS
    preview_on: bool         # preview window wanted (F11)
    click_mode: str = "blink"  # how to click: "blink" or "dwell" (see DwellClicker)
    gestures: bool = True    # face gestures change settings (see GESTURES)
    remote_off: bool = False  # the website's "Disable Mouse Control" (see handle_command)
    hint: str = ""           # advice shown in the window (e.g. "too dark")
    right_click_count: int = 0   # monotonically increasing number of right clicks (right wink)
    scroll_units: int = 0    # running total of wheel turns (TiltScroll; 120 = a notch, + = up)
    pos_time: float = 0.0    # time.monotonic() of the camera frame that gave x, y
    calib_step: str = ""     # calibration: "" (not running), "starting", a Calibration step, "done"
    calib_text: str = ""     # ... the instruction (or the result) shown on screen
    calib_progress: float = 0.0  # ... progress of the current step, 0..1
    calibrated: str = ""     # date of the user's saved calibration ("" = never calibrated)
    remote_preview: bool = False  # a website page shows the camera view
    voice: str = ""          # voice typing: "" (off) or one of voice_typing.STATES
    voice_text: str = ""     # ... the last phrase it typed (shown on this PC only)
    voice_hint: str = ""     # ... a problem or a tip for the user
    voice_level: float = 0.0  # ... microphone loudness 0..1, for the meter
    voice_left: float = 0.0  # ... seconds before it stops by itself (only near the end)
    gaze_jump: bool = True   # Gaze jump wanted (works once the eyes are calibrated, gaze_engine)
    gaze_calibrated: str = ""  # date of the user's eye calibration ("" = never)
    gaze_error_px: int = 0   # ... how accurately it knows where they look (px)
    blink_test: bool = False  # the blink test (Actions.blink_test) has the camera


class SharedState:
    """Thread-safe container for the latest tracker output."""

    def __init__(self, screen: fe.ScreenSize, sensitivity: int, preview_on: bool,
                 click_mode: str = "blink", gestures: bool = True):
        self._lock = threading.Lock()
        self._snap = Snapshot(
            x=screen.width / 2, y=screen.height / 2, eyes_closed=False, click_count=0,
            face=False, has_position=False, confidence=0.0, paused=False,
            esp32_connected=False, ws_clients=0, site_clients=0, sensitivity=sensitivity,
            preview_on=preview_on, click_mode=click_mode, gestures=gestures)
        self.last_frame = time.monotonic()   # heartbeat from the vision thread
        self.stop = threading.Event()        # set to shut everything down
        self.recenter_request = threading.Event()  # F6 -> cursor to the screen centre
        self.calibrate_request = threading.Event() # F10 / Calibrate -> start a calibration
        self.calibrate_cancel = threading.Event()  # ... stop the one running
        self.calibrate_kind = "head"               # ... which one: "head" (Calibration) or "gaze"
        self.voice_requests: queue.SimpleQueue = queue.SimpleQueue()  # voice_typing.request()
        self._preview: tuple[int, np.ndarray | None] = (0, None)

    def update(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self._snap, k, v)

    def add_click(self, right: bool = False) -> None:
        with self._lock:
            if right:
                self._snap.right_click_count += 1
            else:
                self._snap.click_count += 1

    def add_scroll(self, units: int) -> None:
        with self._lock:
            self._snap.scroll_units += int(units)

    def toggle(self, field: str) -> bool:
        """Flip a boolean field atomically and return its new value."""
        with self._lock:
            value = not getattr(self._snap, field)
            setattr(self._snap, field, value)
            return value

    def read(self) -> Snapshot:
        with self._lock:
            return dataclasses.replace(self._snap)

    def set_preview(self, frame: np.ndarray | None) -> None:
        """Latest annotated camera image for the window (BGR), or None."""
        with self._lock:
            self._preview = (self._preview[0] + 1, frame)

    def get_preview(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._preview

    def beat(self) -> None:
        self.last_frame = time.monotonic()


CLOSE_AFTER_SITE_S = 30.0  # dashboard closed this long -> EyeSync closes (EyeSync.pyw waits)
# Exit codes, read by EyeSync.pyw: a quit waits for the website's Connect;
# a closed dashboard may be reopened by the website by itself; others restart.
EXIT_QUIT, EXIT_CRASHED, EXIT_DASHBOARD_CLOSED = 0, 1, 3
RECONNECT_GRACE_S = 4.0   # the website retries every 3 s; a reload or a link inside the site
                          # reconnects within that, without control dropping


def website_connected(snap: Snapshot, args) -> bool:
    return args.always_active or snap.site_clients > 0


def calibrating(snap: Snapshot) -> bool:
    """A calibration is starting or running (the camera is on for it)."""
    return snap.calib_step not in ("", "done")


def status_of(snap: Snapshot, args) -> tuple[str, str]:
    """(key, text) for the window's status line - see main_window.COLORS for the keys."""
    if calibrating(snap):
        return "active", "Calibrating - follow the guide on the screen"
    if not website_connected(snap, args):
        return "off", "Starting the EyeSync camera..."
    if snap.remote_off:
        return "paused", "Mouse control is off on the website"
    if snap.paused:
        return "paused", "Paused"
    if not snap.has_position:
        return "off", "Starting the camera..."
    if not snap.face:
        return "noface", "On - can't see your face"
    how = "blink" if snap.click_mode == "blink" else "hold still"
    return "active", f"On - move your head, {how} to click"


def control_active(snap: Snapshot, args) -> bool:
    """True when the cursor may be moved/clicked (see "Activation" above)."""
    return website_connected(snap, args) and not snap.paused and not snap.remote_off


# --------------------------------------------------------------------------- #
# Sensitivity / smoothing
# --------------------------------------------------------------------------- #

# Sensitivity levels, 1 = calmest ... 5 = most responsive. Change them live
# with Ctrl+Alt+Shift+F7 (calmer) / F8 (more responsive); the choice is saved.
#   min_cutoff  One Euro cutoff (Hz) while the head is still
#   beta        how quickly smoothing relaxes as the head speeds up. Kept
#               small at every level: webcam jitter itself looks like fast
#               movement, so a large beta switches smoothing off exactly
#               when the tracking is noisiest.
#   start_px    dead-zone: the head must move the target this far (px) from
#               where it last rested before the cursor starts moving
#   stop_speed  the cursor stops as soon as the target moves slower than this
#               (px/s) for ~0.13 s, i.e. right after the head stops
#   max_speed   cursor speed limit (px/s). Pixel values are for a
#               1920-px-wide screen and are scaled for other widths.
# Higher levels mainly move faster; all keep the cursor still at rest and
# stop it within ~1 s of the head even with heavy webcam noise (tested in
# tests/test_eyesync.py up to 45 px of jitter on the raw target).
# The head pointer needs a high speed limit: a lower one let the cursor trail
# a normal head turn, so you kept turning and it overshot (measured on the
# user's recorded movements: stops 0.3 s sooner and holds steadier at 3000).
# Lag (head turn -> cursor at 90% of the way, smooth realistic turns with the
# user's measured camera noise): 350 ms with the earlier median-of-5 nose,
# beta 0.001 and slow catch-up; ~170 ms now, still motionless at 3x that noise.
# start_px is small (25-33 px): a 65 px dead zone ignored small aiming
# movements completely, then the cursor jumped - "laggy and too fast". On the
# user's recordings 25 px reacts to a 1-degree aiming move within ~0.5 s and
# starts by itself ~3x a minute (15 px: 9x).
SENSITIVITY_LEVELS = {
    1: (0.8, 0.003, 25, 90, 3000),
    2: (0.8, 0.003, 27, 85, 3300),
    3: (0.8, 0.003, 29, 80, 3600),
    4: (0.8, 0.003, 31, 75, 4000),
    5: (0.8, 0.003, 33, 70, 4500),
}
DEFAULT_SENSITIVITY = 1     # the calmest: steadiest and slowest
SETTINGS_PATH = fe.PROJECT_DIR / "eyesync_settings.json"
SETTINGS_VERSION = 6        # bump when defaults change: older saved settings are reset


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(**updates) -> None:
    settings = load_settings()
    settings.update(updates)
    settings.setdefault("version", SETTINGS_VERSION)
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning(f"[settings] Could not save {SETTINGS_PATH}: {exc}")


class OneEuroFilter:
    """
    One Euro filter (Casiez et al., CHI 2012) - an exponential moving average
    whose smoothing factor adapts to speed:

        alpha = 1 / (1 + tau/dt),   tau = 1 / (2*pi*cutoff)
        cutoff = min_cutoff + beta * |speed|

    When the head is still, the cutoff is low -> heavy smoothing, no jitter.
    When the head moves quickly, the cutoff rises -> little lag. A plain EMA
    has to trade one of those off against the other.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.dx = np.zeros(2)
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, sample: np.ndarray, t: float) -> np.ndarray:
        sample = sample.astype(np.float64)
        if self.x is None or self.t is None:
            self.x, self.t = sample, t
            return self.x
        dt = max(t - self.t, 1e-3)
        self.t = t
        # Smoothed speed (px/s) drives the adaptive cutoff.
        raw_dx = (sample - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (raw_dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (sample - self.x)
        return self.x


class CursorMotion:
    """
    Moves the cursor only while the head is moving, then settles and STOPS
    within a fraction of a second of the head stopping.

    The previous design chased the target position with a speed cap; after
    a big head turn the cursor kept travelling for seconds to close the gap.
    This one is a small state machine driven by the smoothed target:

      REST      The cursor does not move. The head must shift the target more
                than ``start_px`` from where it settled before anything
                happens, so jitter, breathing and small posture shifts are
                ignored completely.
      MOVING    Each frame the cursor moves by the target's own motion plus a
                small pull towards it (CORRECTION), capped at ``max_speed``.
      SETTLING  Entered as soon as the head stops: the cursor quickly closes
                whatever gap the speed cap left (SETTLE_RATE per frame, at
                most SETTLE_SPEEDUP x max_speed) and is back at REST within
                SETTLE_MAX_S.

    "Stopped" is judged from the target's net displacement over the last
    WINDOW frames (~0.2 s) rather than frame-to-frame speed: random tracking
    noise cancels out over the window, real head movement does not.

    Noise adaptation: while resting, the tracker measures how much the target
    jitters on its own (lighting, camera, model) as its scatter around a
    straight line fitted to the last WINDOW frames. A slow, steady head
    movement lies on that line, so it is never mistaken for noise. (Scatter
    around the plain average counted a slow movement as noise and raised the
    dead zone while you aimed: a small, slow movement only moved the cursor
    ~0.25 s after it ended. Distance from where the cursor rests was worse:
    the threshold rose until the cursor could not start at all.) The start and stop
    thresholds are raised to sit safely above that noise (up to NOISE_CAP x
    their configured values, and at least NOISE_FLOOR_PX / NOISE_FLOOR_SPEED),
    so the cursor stays still at rest and stops promptly in a noisy setting,
    without manual tuning.
    """

    CORRECTION = 0.3     # fraction of the cursor-target gap closed per moving frame: the gap
                         # left by the dead zone closes while moving, so the cursor doesn't trail
    CATCHUP_SHARE = 0.35  # ... but at most this share of the head's own movement per frame,
    CATCHUP_MIN_PX = 1.0 # (or this many px) - a smooth start instead of a lurch
    # Share of each frame's new movement taken into the cursor's speed (1.0 =
    # raw, see __call__). Measured with steady turns and the user's webcam
    # noise: 0.4 makes the speed 25-37% more even; following a slowing head at
    # once (1.0) keeps the cursor from coasting past where it stopped (less
    # overshoot than raw), for +6..28 ms of delay on slow moves.
    VELOCITY_SMOOTHING = 0.4
    VELOCITY_SMOOTHING_SLOWING = 1.0
    SETTLE_RATE = 0.45   # fraction of the remaining gap closed per settling frame
    SETTLE_MAX_S = 1.5   # safety net: settling normally ends within ~0.5 s
    SETTLE_DONE_PX = 2.5 # settled when this close to the target
    SETTLE_SPEEDUP = 2.0  # settling glides at up to 2x max_speed, so the peak cursor speed
                          # is 2x max_speed; a higher cap with a lower glide is steadier
                          # and lags less than the other way round (measured on real data)
    WINDOW = 6
    NOISE_CAP = 3.0
    NOISE_FLOOR_PX = 100.0       # the noise-raised thresholds may always reach these
    NOISE_FLOOR_SPEED = 300.0    # (px and px/s on a 1920-px-wide screen)
    NOISE_RATE = 0.1             # noise estimate update per resting frame
    START_PER_JITTER = 30.0      # start / stop thresholds per px of jitter. Start: at rest the
    STOP_PER_JITTER = 33.0       # smoothed target drifts up to ~16x the jitter over seconds
                                 # (p99), more in the first seconds; 30x keeps a 2x noisier
                                 # webcam than the developer's still in ~94% of rests
    START_FRAMES = 2             # frames beyond the start threshold before moving

    def __init__(self, start_px: float, stop_speed: float, max_speed: float,
                 px_scale: float = 1.0):
        self.start_px, self.stop_speed, self.max_speed = start_px, stop_speed, max_speed
        self.px_scale = px_scale
        self.pos: np.ndarray | None = None
        self.jitter = 0.0        # typical scatter around the window's mean while resting (px)
        self.resync()

    def reset(self) -> None:
        """Forget everything; the next target becomes the cursor position."""
        self.pos = None
        self.resync()

    @property
    def _start_cap(self) -> float:
        return max(self.NOISE_CAP * self.start_px, self.NOISE_FLOOR_PX * self.px_scale)

    @property
    def _stop_cap(self) -> float:
        return max(self.NOISE_CAP * self.stop_speed, self.NOISE_FLOOR_SPEED * self.px_scale)

    @property
    def start_threshold(self) -> float:
        return min(max(self.start_px, self.START_PER_JITTER * self.jitter), self._start_cap)

    @property
    def stop_threshold(self) -> float:
        return min(max(self.stop_speed, self.STOP_PER_JITTER * self.jitter), self._stop_cap)

    def resync(self) -> None:
        """Keep the cursor where it is, but restart motion tracking."""
        self.window: deque[tuple[float, np.ndarray]] = deque(maxlen=self.WINDOW)
        self.anchor: np.ndarray | None = None
        self.mode = "rest"
        self.over = 0
        self.settle_since = 0.0
        self.velocity: np.ndarray | None = None   # smoothed movement per frame (see __call__)

    @property
    def moving(self) -> bool:
        return self.mode != "rest"

    def _window_speed(self) -> float:
        (t0, p0), (t1, p1) = self.window[0], self.window[-1]
        return float(np.linalg.norm(p1 - p0)) / max(t1 - t0, 1e-3)

    def _recent(self) -> np.ndarray:
        """Mean of the last 3 targets - a steadier 'where is the head now'."""
        return np.mean([p for _, p in list(self.window)[-3:]], axis=0)

    def __call__(self, target: np.ndarray, now: float) -> np.ndarray:
        target = target.astype(np.float64)
        if self.pos is None:
            self.pos = target.copy()
        prev = self.window[-1] if self.window else None
        self.window.append((now, target.copy()))
        if prev is None:
            self.anchor = target.copy()
            return self.pos
        if len(self.window) < 3:
            return self.pos

        dt = min(max(now - prev[0], 1e-3), 0.1)
        speed = self._window_speed()

        if self.mode == "rest":
            disp = float(np.linalg.norm(self._recent() - self.anchor))
            if disp <= self.start_threshold:
                # Still: learn the noise floor (see the class docstring). Only
                # still frames count - a real head movement would inflate it.
                self.over = 0
                if len(self.window) == self.WINDOW:
                    w = np.array([p for _, p in self.window])
                    t = np.arange(len(w)) - (len(w) - 1) / 2
                    slope = (t[:, None] * w).sum(axis=0) / (t * t).sum()
                    resid = w - w.mean(axis=0) - t[:, None] * slope
                    scatter = float(np.mean(np.linalg.norm(resid, axis=1)))
                    self.jitter += self.NOISE_RATE * (scatter - self.jitter)
                return self.pos
            # A real head movement persists; a noise spike lasts one frame.
            self.over += 1
            if self.over < self.START_FRAMES:
                return self.pos
            self.over = 0
            self.mode, self.velocity = "moving", None

        stop_thr = self.stop_threshold
        if self.mode == "moving" and len(self.window) == self.WINDOW and speed < stop_thr:
            self.mode, self.settle_since = "settling", now

        if self.mode == "settling":
            # Head moving again? (Persisting for START_FRAMES, like a start.)
            self.over = self.over + 1 if speed > 2 * stop_thr else 0
            if self.over >= self.START_FRAMES:
                self.over = 0
                self.mode, self.velocity = "moving", None
            else:
                gap = self._recent() - self.pos
                # Close the gap to the target (a few px). Rest is only entered
                # from here, and at rest a gap below the start threshold would
                # never close - so settle to the target, not to "good enough".
                if float(np.linalg.norm(gap)) < self.SETTLE_DONE_PX or now - self.settle_since > self.SETTLE_MAX_S:
                    self.mode = "rest"
                    self.anchor = self._recent()
                    return self.pos
                step = self.SETTLE_RATE * gap
                max_step = self.SETTLE_SPEEDUP * self.max_speed * dt   # glide, don't jump
                step_len = float(np.linalg.norm(step))
                if step_len > max_step:
                    step *= max_step / step_len
                self.pos = self.pos + step
                return self.pos

        # The head's movement this frame, averaged over the last few frames:
        # moving the cursor by each frame's raw movement passed the webcam
        # noise straight on (the cursor's speed wobbled ~30% during a steady turn).
        raw = target - prev[1]
        if self.velocity is None:
            self.velocity = raw.copy()
        else:
            # Slowing down is followed at once, so the cursor never coasts on
            # past where the head stopped; speeding up and steady movement are smoothed.
            slowing = float(np.dot(raw, self.velocity)) < float(np.dot(self.velocity, self.velocity))
            k = self.VELOCITY_SMOOTHING_SLOWING if slowing else self.VELOCITY_SMOOTHING
            self.velocity = self.velocity + k * (raw - self.velocity)
        motion = self.velocity
        catch = self.CORRECTION * (target - self.pos)
        # Close the gap the dead zone left gradually, at most CATCHUP_SHARE of
        # the head's own movement per frame: catching it up at once made the
        # cursor lurch 3-5x faster at the start of every slow aiming move.
        limit = max(self.CATCHUP_SHARE * float(np.linalg.norm(motion)),
                    self.CATCHUP_MIN_PX * self.px_scale)
        catch_len = float(np.linalg.norm(catch))
        if catch_len > limit:
            catch *= limit / catch_len
        step = motion + catch
        step_len = float(np.linalg.norm(step))
        max_step = self.max_speed * dt
        if step_len > max_step:
            step *= max_step / step_len
        self.pos = self.pos + step
        return self.pos


# --------------------------------------------------------------------------- #
# Head pointer: how the head moves the cursor
# --------------------------------------------------------------------------- #

# How far you move your head to cross the screen width, in face widths
# (outer eye corner to outer eye corner; ~1.2 is about a 35-degree turn),
# per speed level: (moving slowly, turning at a normal pace). Like a mouse,
# slow movements are precise and normal turns go far (see HeadPointer). At
# speed 1 crossing the screen takes ~23 degrees of head turn (15 felt "way
# too fast", 30 "too much turning"); speed 5 ~12.
# HEAD_SPEED is from the user's recorded movements: holding still stays
# below 0.25 face widths/s (90%), real moves peak above 1.15 (90%).
HEAD_RANGE = {1: (2.2, 0.8), 2: (1.9, 0.68), 3: (1.6, 0.57), 4: (1.35, 0.48), 5: (1.1, 0.4)}
# The nose the pointer follows: the average of the last NOSE_FRAMES positions
# (nose_smooth). Until 1.5.11 the median of 3 (it drops one-frame glitches,
# for a frame of lag; 5 added ~35 ms). 1.5.12, the user asking for a quicker
# cursor ("fast fast fast"): the mean of 2 - half a frame of lag instead of
# one, and a steady turn moves the cursor MORE evenly (CursorMotionTests:
# speed spread 0.24 vs 0.24 slow, 0.11 vs 0.13 normal). The newest nose alone
# (28 ms less lag) made slow turns uneven (0.33) - rejected.
NOSE_FRAMES = 2


def nose_smooth(noses) -> np.ndarray:
    """The pointer's nose from the last NOSE_FRAMES nose positions."""
    return np.mean(np.array(noses), axis=0)
SCREEN_CHECK_S = 2.0      # how often the worker checks whether the display's resolution changed
HEAD_SPEED = (0.3, 1.2)   # head speed (face widths/s) where "slow" ends and "normal" begins
HEAD_Y_GAIN = 1.4     # people tilt their head less than they turn it


class HeadPointer:
    """
    The cursor follows the nose tip like a mouse follows your hand: each
    head movement moves the cursor by that movement times a gain that depends
    on how fast you move. Slow, small movements give fine, precise steps;
    a quick turn crosses the screen (see HEAD_RANGE / HEAD_SPEED). Holding
    still holds the cursor still. Movements are
    measured in face widths (eye-corner distance), so sitting closer or
    further away doesn't change the speed.

    Edges: the cursor stops at the screen edge, and turning back moves it at
    once - it can never get "stuck out of range". If your head ends up turned
    uncomfortably far, push the cursor against an edge with a slow movement
    and turn back quickly, or Recenter.
    """

    GAP_S = 0.25    # frames further apart than this (blink, gesture, lost face) start afresh

    def __init__(self, screen_w: int, screen_h: int):
        self.size = np.array([screen_w, screen_h], np.float64)
        # The user's calibrated head range (see Calibration): face widths of
        # head movement that cross the screen's width / height at a normal
        # pace at speed 1. None: the defaults (HEAD_RANGE, HEAD_Y_GAIN).
        self.calib: tuple[float, float] | None = None
        self.reset()

    def reset(self, start: np.ndarray | None = None) -> None:
        """Put the cursor at ``start`` (default: the centre); the head's next moves move it."""
        self.pos = self.size / 2 if start is None else np.asarray(start, np.float64).copy()
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=4)   # (time, nose px)
        self.scale: float | None = None         # eye-corner distance, smoothed

    def gain(self, level: int, speed: float) -> np.ndarray:
        """Cursor px per face width of head movement at ``speed`` face widths/s."""
        slow, quick = HEAD_RANGE.get(level, HEAD_RANGE[DEFAULT_SENSITIVITY])
        u = float(np.clip((speed - HEAD_SPEED[0]) / (HEAD_SPEED[1] - HEAD_SPEED[0]), 0, 1))
        u = u * u * (3 - 2 * u)                  # smooth step between the two
        span = slow + (quick - slow) * u         # face widths to cross the screen width
        if self.calib is not None:
            # Same speed levels and slow/quick feel, scaled to this user's range.
            k = span / HEAD_RANGE[DEFAULT_SENSITIVITY][1]
            return np.array([-self.size[0] / (k * self.calib[0]), self.size[1] / (k * self.calib[1])])
        g = self.size[0] / span
        return np.array([-g, g * HEAD_Y_GAIN])   # image x is mirrored relative to the user

    def target(self, nose_px: np.ndarray, face_px: float, level: int,
               now: float | None = None) -> np.ndarray:
        nose = np.asarray(nose_px, np.float64)
        if now is None:                          # (tests) assume 30 frames per second
            now = self.history[-1][0] + 1 / 30 if self.history else 0.0
        self.scale = face_px if self.scale is None else self.scale + 0.05 * (face_px - self.scale)
        if self.history and now - self.history[-1][0] > self.GAP_S:
            # Head movement while the eyes were closed (a blink-click often
            # nods the head), during a gesture or with the face lost is
            # ignored: the cursor stays where it was.
            self.history.clear()
        if self.history:
            t0, n0 = self.history[0]
            speed = float(np.linalg.norm(nose - n0)) / self.scale / max(now - t0, 1e-3)
            step = (nose - self.history[-1][1]) / self.scale
            self.pos = np.clip(self.pos + step * self.gain(level, speed), 0, self.size - 1)
        self.history.append((now, nose))
        return self.pos.copy()


# --------------------------------------------------------------------------- #
# Dwell -> click (alternative to blinking)
# --------------------------------------------------------------------------- #

CLICK_MODES = ("blink", "dwell")


class DwellClicker:
    """
    Click by holding the cursor still: after the cursor has moved at least
    ``arm_px`` and then rested for ``dwell_s``, one click. It must move
    ``arm_px`` away from that spot before it can click again, so resting on
    a page never fires repeated clicks. For people who can't blink on
    purpose reliably.
    """

    def __init__(self, dwell_s: float = 1.0, arm_px: float = 60.0):
        self.dwell_s, self.arm_px = dwell_s, arm_px
        self.reset()

    def reset(self) -> None:
        """Forget everything: nothing clicks until the cursor has moved."""
        self.anchor: np.ndarray | None = None     # where the last click (or start) was
        self.armed = False
        self.rest_since: float | None = None

    def pause(self) -> None:
        """Restart the dwell timer (e.g. during a blink) but stay armed."""
        self.rest_since = None

    def progress(self, now: float) -> float:
        """0..1 of the way to a click (for on-screen feedback)."""
        if not self.armed or self.rest_since is None:
            return 0.0
        return min(1.0, (now - self.rest_since) / self.dwell_s)

    def update(self, pos: np.ndarray, moving: bool, now: float) -> bool:
        pos = np.asarray(pos, np.float64)
        if self.anchor is None:
            self.anchor = pos.copy()
        if not self.armed and float(np.linalg.norm(pos - self.anchor)) >= self.arm_px:
            self.armed = True
        if moving:
            self.rest_since = None
            return False
        self.rest_since = self.rest_since if self.rest_since is not None else now
        if self.armed and now - self.rest_since >= self.dwell_s:
            self.armed, self.anchor, self.rest_since = False, pos.copy(), None
            return True
        return False


# --------------------------------------------------------------------------- #
# Head direction
# --------------------------------------------------------------------------- #

Pose = tuple[float, float]          # (yaw, pitch) of the head in degrees (fe.head_pose)


def pose_change(a: Pose | None, b: Pose | None) -> float:
    """How far the head turned between two poses (degrees, largest axis)."""
    if a is None or b is None:
        return 0.0
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class PoseBins:
    """
    A resting level learned separately for each head direction. Looking at
    the bottom of the screen lowers the eyelids and raises MediaPipe's blink
    score; turning to a side changes the expression scores. Judged against
    one overall level, a head position alone looked like a blink or a
    gesture (or hid a real one), so each direction - a cell of YAW_DEG x
    PITCH_DEG - keeps its own level, once it has seen MIN_FRAMES.
    """

    YAW_DEG, PITCH_DEG = 8.0, 6.0
    MIN_FRAMES = 15          # ~0.5 s in a direction before its level is trusted
    RATE = 0.03              # then a slow average (~1 s)

    def __init__(self):
        self.cells: dict[tuple[int, int], list[float]] = {}

    @classmethod
    def key(cls, pose: Pose) -> tuple[int, int]:
        return round(pose[0] / cls.YAW_DEG), round(pose[1] / cls.PITCH_DEG)

    def get(self, pose: Pose | None) -> float | None:
        """The learned level for this direction, or None (not learned yet)."""
        if pose is None:
            return None
        cell = self.cells.get(self.key(pose))
        return cell[0] if cell is not None and cell[1] >= self.MIN_FRAMES else None

    def learn(self, pose: Pose | None, value: float) -> None:
        if pose is None:
            return
        cell = self.cells.setdefault(self.key(pose), [value, 0])
        cell[1] += 1
        cell[0] += max(1.0 / cell[1], self.RATE) * (value - cell[0])


# --------------------------------------------------------------------------- #
# Face gestures -> settings
# --------------------------------------------------------------------------- #

# Hold an expression for FaceGestures.HOLD_S to change a setting. Chosen from
# MediaPipe's expression scores that are near zero on a resting face (smile,
# side-mouth); raised eyebrows were rejected -
# with glasses they already scored ~0.6 at rest on the developer's face.
# The minimums are for a BIG smile / a clear pull: a light smile (~0.3-0.5)
# paused EyeSync by mistake with 0.45.
#   channel: (label shown/logged, absolute minimum, margin above your resting level)
GESTURES = {
    "smile": ("Big smile: pause / resume", 0.6, 0.45),
    "side": ("Mouth pulled to one side", 0.45, 0.35),
}
# The "side" channel is two gestures, by which way the mouth went (the face's
# own left and right, fe.mouth_shift - not which expression score rose, whose
# naming may be the camera's): to the user's left = recenter, to their right =
# calibrate. Calibrating takes ~30 s and can't be cancelled hands-free, so
# its hold is long, with a beep and a countdown first. (Head tilts did these
# until 1.4.0; they scroll now - see TiltScroll. "Next speed" left the
# gestures in 1.5.0: speed is in the window, on F7/F8 and on the website.)
SIDE_GESTURES = {1: "mouth_left", -1: "mouth_right"}
GESTURE_LABELS = {"smile": GESTURES["smile"][0],
                  "mouth_left": "Mouth to your left: recenter",
                  "mouth_right": "Mouth to your right: calibrate"}
GESTURE_HOLD_S = {"mouth_right": 3.0}       # (others: FaceGestures.HOLD_S)
MOUTH_SHIFT_MIN_FW = 0.012   # the mouth must be off its resting place by this (face widths)
# A pull seen in the landmarks (FaceGestures._side). Measured on the user's
# face (tools/recordings/blinks-20261020-135539): MediaPipe hardly scores
# their pull to the left as mouthLeft (0.02-0.09) - it looks like a lopsided
# smile, the left corner 0.2-0.5 above the right - and their mouthRight
# wavered 0.1-0.7 through a steady pull to the right, so a score alone never
# held 3 s; the landmarks moved steadily, 0.02-0.04 face widths left and
# 0.03-0.06 right. A head turn moves fe.mouth_shift too (the mouth is nearer
# the camera than the point under the nose): -0.196 x sin(yaw), measured, is
# taken out. A pull is then that shift with a score on the same side agreeing.
MOUTH_PULL_FW = 0.018        # the landmarks' shift for a pull (face widths, turn taken out)
MOUTH_SHIFT_PER_YAW = 0.2    # what a head turn adds to fe.mouth_shift, per sin(yaw)
MOUTH_PULL_LOPSIDED = 0.12   # agreeing score: that side's smile above the other's by this ...
MOUTH_PULL_SCORE = 0.15      # ... or its mouthLeft / mouthRight score
MOUTH_PULL_MAX_YAW = 25.0    # further turned, the turn correction isn't trusted
MOUTH_OPEN_JAW = 0.15        # jawOpen: an open mouth (talking) is not a pull
# Voice typing on / off: both eyes closed this long (VisionWorker._process).
# Not an expression - the user asked for the kiss-lips gesture to be removed,
# and eyebrows don't work with glasses - while the blink detection is already
# fitted to the user's eyes. A click is a closure of at most --max-blink
# (BLINK_MAX_S), always shorter: a closure that switches voice typing never clicks.
# The combined build reserves a two-second bilateral closure for voice typing.
VOICE_EYES_CLOSED_S = 1.0
# While voice typing listens, talking moves the mouth: these don't fire then.
VOICE_IGNORED_GESTURES = frozenset({"smile", "mouth_left", "mouth_right"})


class FaceGestures:
    """
    Detects a held facial expression and reports it once. Each expression's
    threshold adapts to the user's resting level (a slow average while it is
    well below the threshold, per head direction - see PoseBins), a short
    sag (GRACE_S) doesn't cancel the hold, and after a gesture fires the face
    must relax before another can start - so holding a smile toggles pause
    once, not every 1.5 s.

    With the head turned, the camera sees the far half of the mouth badly
    and scores its smile low, so the smile counts the visible side more
    (fully from SMILE_TURN_DEG on): a big smile works at the screen edges too.

    Thresholds and resting levels are per channel (GESTURES); ``active`` and
    what update() returns are gesture names - the "side" channel becomes
    mouth_left or mouth_right by where the mouth went (``mouthShift`` in the
    scores, fe.mouth_shift, against its own resting place). With no shift
    given, or the mouth not clearly to one side, the side gestures don't fire.
    """

    HOLD_S = 1.5
    GRACE_S = 0.3
    SMILE_TURN_DEG = 25.0
    RELAX_MAX_S = 3.0        # after a gesture, blinks wait for the face to rest - at most this

    def __init__(self):
        self.base: dict[str, float] = {}
        self.bins: dict[str, PoseBins] = {n: PoseBins() for n in GESTURES}
        self.minimum = {n: g[1] for n, g in GESTURES.items()}   # calibration may set the smile's
        self.active: str | None = None          # a gesture name (GESTURE_LABELS)
        self.since = 0.0
        self.low_since: float | None = None
        self.released = True
        self.relaxed = True                     # False from a gesture firing until the face rests
        self.fired_at = -1e9
        self.rest_shift: float | None = None    # where the mouth rests (fe.mouth_shift)
        self.side_seen = (0.0, 0.0, 0.0, 0.0, 0.0)  # off its rest, smile L/R, mouthLeft/Right
        self.detail = ""                        # those numbers when the gesture began (for the log)

    def hold_s(self, name: str | None = None) -> float:
        return GESTURE_HOLD_S.get(name or self.active, self.HOLD_S)

    def _side(self, bs: dict[str, float], side_score: float, side_thr: float, yaw: float) -> int:
        """
        Which way the mouth is pulled: +1 the user's left, -1 right, 0 neither
        or unknown. Either the landmarks moved MOUTH_PULL_FW with a score on
        that side agreeing (see MOUTH_PULL_FW), or the side score passed its
        threshold with the landmarks showing the way. The mouth's resting
        place (fe.mouth_shift, the head turn taken out) is learned only while
        no score shows a pull - never in the middle of a long hold.
        """
        shift = bs.get("mouthShift")
        if shift is None:
            return 0
        here = shift + MOUTH_SHIFT_PER_YAW * float(np.sin(np.radians(yaw)))
        if self.rest_shift is None:
            self.rest_shift = here
        sl, sr = bs.get("mouthSmileLeft", 0.0), bs.get("mouthSmileRight", 0.0)
        ml, mr = bs.get("mouthLeft", 0.0), bs.get("mouthRight", 0.0)
        steady = abs(yaw) <= MOUTH_PULL_MAX_YAW and bs.get("jawOpen", 0.0) < MOUTH_OPEN_JAW
        if (steady and abs(sl - sr) < MOUTH_PULL_LOPSIDED / 2 and max(ml, mr) < MOUTH_PULL_SCORE / 2
                and (sl + sr) / 2 < 0.3):
            self.rest_shift += 0.02 * (here - self.rest_shift)
        off = here - self.rest_shift
        self.side_seen = (off, sl, sr, ml, mr)
        if not steady or abs(off) < MOUTH_SHIFT_MIN_FW:
            return 0
        way = 1 if off > 0 else -1
        agrees = (sl - sr if way > 0 else sr - sl) >= MOUTH_PULL_LOPSIDED or \
            (ml if way > 0 else mr) >= MOUTH_PULL_SCORE
        return way if (abs(off) >= MOUTH_PULL_FW and agrees) or side_score >= side_thr else 0

    @classmethod
    def scores(cls, bs: dict[str, float], yaw: float = 0.0) -> dict[str, float]:
        left, right = bs.get("mouthSmileLeft", 0.0), bs.get("mouthSmileRight", 0.0)
        side_weight = min(abs(yaw) / cls.SMILE_TURN_DEG, 1.0)
        return {
            "smile": (left + right) / 2 + side_weight * (max(left, right) - (left + right) / 2),
            "side": max(bs.get("mouthLeft", 0.0), bs.get("mouthRight", 0.0)),
        }

    def threshold(self, name: str, pose: Pose | None = None) -> float:
        margin = GESTURES[name][2]
        base = self.bins[name].get(pose)
        return max(self.minimum[name], (self.base.get(name, 0.0) if base is None else base) + margin)

    def reset(self) -> None:
        self.active, self.low_since = None, None

    def progress(self, now: float) -> float:
        return 0.0 if self.active is None else min(1.0, (now - self.since) / self.hold_s())

    def update(self, bs: dict[str, float] | None, now: float,
               pose: Pose | None = None, ignore=frozenset()) -> str | None:
        """
        Feed one frame's expression scores (with ``mouthShift`` for the side
        gestures). Returns a gesture name when one fires. Gestures in
        ``ignore`` don't fire (their resting levels are still learned).
        """
        if not self.relaxed and now - self.fired_at > self.RELAX_MAX_S:
            self.relaxed = True                     # never block blinks for long
        if bs is None:
            self.reset()
            return None
        yaw = 0.0 if pose is None else float(pose[0])
        sc = self.scores(bs, yaw)
        thr = {n: self.threshold(n, pose) for n in sc}
        side = self._side(bs, sc["side"], thr["side"], yaw)
        # Channels on, as gesture names. A mouth pulled to one side is not a
        # smile, however much its corner rises (the user's pull to the left).
        names = {"smile": "smile", "side": SIDE_GESTURES.get(side)}
        on = {"smile": sc["smile"] >= thr["smile"] and not side, "side": bool(side)}
        above = {names[n] for n in sc if on[n] and names[n] is not None and names[n] not in ignore}
        margin = {names[n]: sc[n] - thr[n] for n in sc if names[n] is not None}
        for n, v in sc.items():
            if v < 0.6 * thr[n]:                   # resting level, away from gestures
                self.base[n] = v if n not in self.base else self.base[n] + 0.01 * (v - self.base[n])
                self.bins[n].learn(pose, v)
        if not side and all(v < 0.6 * thr[n] for n, v in sc.items()):
            self.relaxed = True

        if self.active is not None:
            if self.active in above:
                self.low_since = None
            else:
                self.low_since = self.low_since or now
                if now - self.low_since > self.GRACE_S:
                    self.reset()
                    return None
            if now - self.since >= self.hold_s():
                fired, self.released, self.relaxed = self.active, False, False
                self.fired_at = now
                self.reset()
                return fired
            return None

        if not above:
            self.released = True
            return None
        if self.released:
            name = max(above, key=lambda n: margin[n])
            # The calibrate gesture (the one that beeps, GESTURE_CONFIRM_S)
            # starts only with a mouthLeft / mouthRight score showing a pull (which
            # name depends on the face), not on a lopsided smile alone (1.5.14):
            # 1.5.11 beeped 3 times on 2026-10-28, 2-4 s before a smile pause or a
            # tilt, never calibrating. Every right pull in the user's recording
            # reached it; the hold then goes on as before.
            if (name == "mouth_right"
                    and max(bs.get("mouthLeft", 0.0), bs.get("mouthRight", 0.0)) < MOUTH_PULL_SCORE):
                return None
            self.active = name
            self.since, self.low_since = now, None
            off, sl, sr, ml, mr = self.side_seen
            self.detail = (f"mouth {abs(off):.3f} face widths to your {'left' if off > 0 else 'right'}, "
                           f"mouthLeft {ml:.2f} / mouthRight {mr:.2f}, smile {sl:.2f} / {sr:.2f}")
        return None


# --------------------------------------------------------------------------- #
# Head tilt -> scroll
# --------------------------------------------------------------------------- #

class TiltScroll:
    """
    Scroll by tilting the head - an ear towards a shoulder - and holding it:
    left = up, right = down. The further the tilt, the faster (SLOW to FAST
    wheel notches a second over FULL_DEG past the start); straightening the
    head stops it. Tilt isn't used for pointing, isn't an expression (glasses
    don't matter) and doesn't change when talking, so scrolling never fights
    the cursor, blinks or voice typing.

    The tilt is measured from the user's usual head tilt (learned while the
    head is near it). Scrolling starts once it passes START_DEG for START_S -
    a glance at a screen corner or a quick lean doesn't scroll - and stops
    when it falls below STOP_FRAC of that. ``tilted`` is true from the moment
    it passes START_DEG: the cursor is held from then on - and put back where
    it was when the head was last straight (``calm_at``), as the nose moves
    sideways with the tilt and had moved the cursor on the way.
    """

    # 22 since 1.5.5 (was 15): turning to a screen side leans this user's head
    # too - up to 15.5 degrees in their recordings (19 while talking) - and
    # from 15 the cursor froze, jumped back and blinks were ignored at the
    # corners. Their real tilts pass 15, 20 and 25 within 0.07 s, peaking at
    # 40-52 degrees, so starting later costs no time.
    START_DEG = 22.0
    CALM_DEG = 5.0              # the head counts as straight within this of its usual tilt
    START_S = 0.35
    STOP_FRAC = 0.7
    FULL_DEG = 15.0
    SLOW, FAST = 2.0, 14.0       # wheel notches a second (a notch: ~3 lines of text)
    LEARN_DEG = 9.0              # the usual tilt is learned within this of it ...
    LEARN_RATE = 0.01            # ... slowly (~3 s)
    USUAL_MAX_DEG = 12.0
    WHEEL = 120                  # Windows' wheel units in a notch
    # The tilt is the median of the last frames: a fast head movement can throw
    # the eye corners off for a frame (a real user's: 23 degrees for one frame
    # among 12-18, just after the face was lost for a frame) - and one frame past
    # START_DEG froze the cursor and put it back. A tilt starts only on a full window.
    MEDIAN_FRAMES = 5

    def __init__(self):
        self.usual: float | None = None   # the user's usual head tilt (degrees, + = left)
        self.calm_at = -1e9               # when the head was last straight
        self.recent: deque[float] = deque(maxlen=self.MEDIAN_FRAMES)
        self.reset()

    def reset(self) -> None:
        self.since: float | None = None   # when the tilt passed START_DEG
        self.direction = 0                # +1 scrolling up, -1 down, 0 not scrolling
        self.carry = 0.0                  # notches due but not sent yet
        self.t: float | None = None
        self.notches = 0

    @property
    def active(self) -> bool:
        return self.direction != 0

    @property
    def tilted(self) -> bool:
        return self.since is not None

    def update(self, roll: float | None, now: float) -> int:
        """
        Feed one frame's head tilt (fe.head_roll; None: no face, or gestures
        off). Returns the wheel turn to send now, in Windows' units (+ = up).
        """
        if roll is None:
            self.recent.clear()
            self.reset()
            return 0
        self.recent.append(roll)
        settled = len(self.recent) == self.MEDIAN_FRAMES
        roll = float(np.median(self.recent))
        if self.usual is None:
            self.usual = float(np.clip(roll, -self.USUAL_MAX_DEG, self.USUAL_MAX_DEG))
        rel = roll - self.usual
        dt = 0.0 if self.t is None else min(max(now - self.t, 0.0), 0.1)
        self.t = now
        if abs(rel) < self.CALM_DEG:
            self.calm_at = now
        if self.active:
            if abs(rel) < self.STOP_FRAC * self.START_DEG or np.sign(rel) != self.direction:
                log.info(f"[scroll] Stopped ({self.notches} notches "
                         f"{'up' if self.direction > 0 else 'down'})")
                self.reset()
                self.t = now
                return 0
        elif abs(rel) >= self.START_DEG and settled:
            self.since = self.since if self.since is not None else now
            if now - self.since < self.START_S:
                return 0
            self.direction, self.carry, self.notches = int(np.sign(rel)), 1.0, 0   # one notch at once
            log.info(f"[scroll] Head tilted {'left' if rel > 0 else 'right'} {abs(rel):.0f} deg: "
                     f"scrolling {'up' if rel > 0 else 'down'}")
        else:
            self.since = None
            if abs(rel) < self.LEARN_DEG:
                self.usual = float(np.clip(self.usual + self.LEARN_RATE * rel,
                                           -self.USUAL_MAX_DEG, self.USUAL_MAX_DEG))
            return 0
        past = (abs(rel) - self.START_DEG) / self.FULL_DEG
        self.carry += (self.SLOW + (self.FAST - self.SLOW) * float(np.clip(past, 0.0, 1.0))) * dt
        n = int(self.carry)
        self.carry -= n
        self.notches += n
        return n * self.WHEEL * self.direction


# --------------------------------------------------------------------------- #
# Blink -> click
# --------------------------------------------------------------------------- #

BLINK_SCORE_DELTA = 0.25    # a blink score this far above your open-eye level = closed
# Head movement, from the nose in face widths (1 face width ~ 29 degrees of turn, HEAD_RANGE):
BLINK_MAX_MOVE_FW = 0.15    # moving more than this during a SHALLOW closure: not a blink
BLINK_MOVING_FW_S = 0.5     # eyes closing while the head moves faster: a click only if deep
BLINK_PASSING_MAX_S = 0.5   # ... and never deep this long: not a blink, where the head went
# How far the eyelids must close (closedness units: 1.0 = the close threshold)
# for a closure with head movement to be a click. A real user's calibration
# blinks closed them to 21-44% of open (1.4-1.9); looking at a screen edge
# lowers them only to ~60-80% (below 1.0).
BLINK_DEEP = 1.25
# Click windows (1.5.2): open the eyes after the tick (--min-blink) = left click,
# after the double tick (BLINK_RIGHT_S) = right click, by BLINK_MAX_S. Each
# leaves about a second to react to its tick: the user's recording of
# 2026-10-20 has them opening 0.55-0.9 s after the tick they waited for -
# with the double tick at 0.8 s and resting from 1.5 s (1.4.0-1.5.1), slow
# reactions gave right clicks for left ones and "resting" for right ones.
# 1.5.4: the tick at 0.3 s (was 0.2): natural blinks while reading reached
# 274-286 ms on the user's face and clicked 3-6 times a minute in both
# recordings; their meant clicks wait for the tick (472-1130 ms in 1.5.x).
# Natural blinks come with the head moving (reading): a closure that begins
# while it moves ticks from BLINK_MOVING_MIN_S. The double tick keeps its
# second after the tick, whenever that came.
BLINK_MIN_S = 0.3           # the tick
BLINK_MOVING_MIN_S = 0.4    # ... for a closure that began with the head moving
# 1.5.9: calibration moves the tick to 90% of how long this user's slow blinks
# stay shut. 1.5.10 capped it at 0.45 s (1.5.9: 0.65); 1.5.12 at 0.35 s - the
# user found clicking slow ("fast fast fast") and chose quicker clicks over
# fewer false ones. 1.5.13: never later than the default. The guide's slow,
# half-second blinks put this user's tick at 0.36-0.45 s, while their own
# clicks are quicker: on 2026-10-28 (1.5.11) 366 and 396 ms closures - shut
# ~0.30 and ~0.33 s - opened before the tick, the only meant clicks lost.
# On the 10 GRID readers, false clicks a minute: 1.89 at 0.3 s, 1.48 at
# 0.35, 0.73 at 0.45; the user's recorded quick clicks: 11/20, 2/20, 0/20,
# and their own natural blinks clicked no more at 0.3 s. (A grace for eyes
# opening just as the tick is due costs GRID as much as an earlier tick:
# 1.82 a minute for 1/25 s at 0.35.)
BLINK_CALIBRATED_MAX_S = BLINK_MIN_S
BLINK_RIGHT_S = 1.3         # the double tick: both eyes closed this long = a right click
BLINK_MAX_S = 2.4           # longer is resting (then voice typing, VOICE_EYES_CLOSED_S)
CLICK_REWIND_S = 0.13       # a click lands where the cursor was this long before the eyes closed
DEG_PER_FACE_WIDTH = 29.0   # (only when no nose position is given: from the head angles)
BLINK_SCORE_BASE_MAX = 0.5  # highest open-eye blink score learned (looking down raises it)


class BlinkDetector:
    """
    Turns per-eye EAR (eyelid geometry) and MediaPipe's learned blink scores
    into deliberate-blink clicks.

    Two signals: an eye counts as closed when EITHER says so, each judged
    against the user's own open-eye level. With glasses, the eyelid points
    can be thrown off by the frames while the learned score still rises (on
    the developer's face it only reached ~0.33 in a blink, but from an
    open-eye level of 0.02 - so it is judged relative to that level).

    Signal clean-up
      * Each eye's EAR is median-filtered over 3 frames, removing one-frame
        landmark glitches that used to look like (or break up) a blink.
      * A blink needs BOTH eyes closed; a squint doesn't count.
      * Winks: ONE eye closed while the other stays clearly open. The left
        eye gives a left click, the right eye a right click (both eyes: a
        left click). Not while the head is turned far to a side
        (``winks_ok``), where one eye is partly hidden and looks closed. If
        the other eye closes too during a wink, it is a normal blink.

    Thresholds with hysteresis
      Eyes become CLOSED when both drop below close_ratio x baseline, and are
      OPEN again only once their average rises above open_ratio x baseline.
      The gap stops a blink near the threshold being split into two clicks
      or ending early. The baseline is a slow running average of the user's
      open-eye EAR (~2 s), so looking down - which lowers the eyelids to about
      70-80% of normal - doesn't trigger clicks, while a real blink drives
      EAR to about 20-40%. --ear-threshold sets a fixed close threshold.

    Head position and movement
      How open the eyes look depends on where the head points: looking at
      the bottom of the screen lowers the eyelids and raises the blink score,
      so moving the head there looked like a blink and clicked (a real
      user: "due to the position of my face it registers a blink"). So:
      * the open-eye levels are learned per head direction (``pose``,
        PoseBins); during a closure, those of where it began are used;
      * head MOVEMENT is measured with the nose (``head``, in face widths),
        not with MediaPipe's head angles: those jump ~8 degrees when both
        eyes close, which cancelled every deliberate blink of a real user;
      * real blinks and fake ones (eyes that only LOOK closed where the head
        went) differ in depth: a blink shuts the eyelids to ~20-45% of open,
        a head position lowers them to ~60-80%. With the head moving, only a
        DEEP closure (BLINK_DEEP, eyelids alone) counts - and then it clicks,
        even while moving or tilting the head (a real user blinks that way);
      * eyes that close while the head is moving (> BLINK_MOVING_FW_S) don't
        hold the cursor at first (``deferred``): a natural blink mid-turn
        never does - holding it made turns stutter. Once the closure is deep
        and as long as a click, the cursor is held, back where it began;
      * a SHALLOW closure during which the head moves more than
        BLINK_MAX_MOVE_FW is the head moving, not a blink: no click, and the
        eyes count as open until they look open again - learning that
        direction's open-eye level meanwhile, so blinks work there right after.

    Click timing
      A click fires when the eyes REOPEN after being closed for min_s..max_s:
      shorter closures are involuntary blinks and are ignored; longer ones
      mean the user is resting their eyes. Both eyes closed for right_s or
      more (BLINK_RIGHT_S, ~1.3 s) is a RIGHT click: many people can't wink
      one eye cleanly (a real user's right wink closed both eyes, the right
      only a little more - indistinguishable from a blink), but everyone can
      blink longer. A cooldown blocks double-clicks, and the short natural
      blink that often follows resting the eyes (within AFTER_REST_S of
      opening them) is not a click either.

    Ticks while the eyes are closed (``cue``, played by the worker): "tick"
    the moment opening the eyes would click, "double" the moment it would
    right-click. They wait out the cooldown and the after-rest rule,
    so a tick always means "open now and it clicks" - and teaches how long
    to close the eyes, the most common reason a blink was missed.
    """

    AFTER_REST_S = 1.0
    AFTER_REST_MAX_BLINK_S = 0.4
    MOVING_MIN_S = BLINK_MOVING_MIN_S
    # A tick only while the eyes are shut (average closedness >= SHUT and
    # both eyes still participating), not while they are already opening
    # again: 33 people's
    # videos (the GRID corpus) showed natural blinks that reopen slowly, the
    # eyes looking down - half open for 0.2-1 s, past the tick: 4 false clicks
    # a minute, up to 11 for one person. A meant click keeps them shut.
    TICK_WHEN_SHUT = True
    SHUT = 1.0
    # Once the tick has sounded, the head moving no longer cancels the closure
    # as "eyes that only look closed where the head went": the tick promised
    # a click. The user (1.5.10's log, 2026-10-27/28: "sometimes when I am
    # rotating my head the click is not registering"): 1131 and 1173 ms
    # closures cancelled there, the head having drifted 0.17-0.18 face widths
    # (limit 0.15) - long after their tick. Eyes that only look closed are
    # still caught before the tick, when the head is getting there.
    KEEP_TICKED = True
    HEAD_FRAMES = 12        # head positions kept (~0.4 s): speed and the closure's reference

    def __init__(self, fixed_threshold: float | None, close_ratio: float, open_ratio: float,
                 min_s: float, max_s: float, cooldown_s: float,
                 right_s: float | None = BLINK_RIGHT_S):
        self.fixed = fixed_threshold
        self.close_ratio, self.open_ratio = close_ratio, open_ratio
        self.min_s, self.max_s, self.cooldown_s = min_s, max_s, cooldown_s
        self.right_s = right_s                 # a both-eye closure this long: right click (None: off)
        self.cue = ""                          # "tick" / "double": tick now (see the class docstring)
        self.cued = 0                          # ticks given in this closure (0, 1 or 2)
        self.shut = False                      # the closing eye(s) shut right now (TICK_WHEN_SHUT)
        self.baseline: float | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=3)
        self.score_history: deque[tuple[float, float]] = deque(maxlen=3)
        self.score_base: float | None = None   # open-eye blink score (slow average)
        self.score_delta = BLINK_SCORE_DELTA   # calibration fits it to the user's blinks
        self.ear_bins = PoseBins()            # the same two levels per head direction
        self.score_bins = PoseBins()
        self.pose: Pose | None = None          # head direction for the open-eye levels
        self.heads: deque[tuple[float, np.ndarray]] = deque(maxlen=self.HEAD_FRAMES)
        self.closed_pose: Pose | None = None   # head direction when the closure began
        self.closed_head: np.ndarray | None = None  # head position just before it
        self.moved = 0.0                       # how far the head moved since (face widths)
        self.moved_by = np.zeros(2)            # ... sideways and up-down (for the log)
        self.head_moved = False                # closure cancelled: the head moved (see above)
        self.deep = 0.0                        # how far the eyelids closed (BLINK_DEEP)
        self.moving_start = False              # the closure began while the head moved
        self.deferred = False                  # ... and isn't a click (yet): cursor not held
        self.started_at = 0.0                  # when the current closure began
        self.closed_since: float | None = None
        self.last_click = 0.0
        self.rested_at = -1e9                # when the eyes last opened after resting
        self.wink_eye: str | None = None     # "left" / "right" while one eye is closed
        self.wink_since = 0.0
        self.wink_by_lids = False            # ... seen by the eyelids (see update)
        self.last_hint = -1e9
        self.value: float | None = None      # filtered mean EAR, for the preview

    def _ear_base(self) -> float:
        base = self.ear_bins.get(self.pose)
        return base if base is not None else (self.baseline or 0.28)

    def thresholds(self) -> tuple[float, float]:
        """(close_threshold, open_threshold) for the current head direction."""
        if self.fixed is not None:
            return self.fixed, self.fixed * self.open_ratio / self.close_ratio
        base = self._ear_base()
        return (float(np.clip(self.close_ratio * base, 0.08, 0.28)),
                float(np.clip(self.open_ratio * base, 0.10, 0.32)))

    def closed_for(self, now: float) -> float:
        return now - self.closed_since if self.closed_since is not None else 0.0

    def reset(self) -> None:
        """Forget any in-progress closure (e.g. the face was lost)."""
        self.closed_since = None
        self.wink_eye = None
        self.head_moved = self.deferred = False
        self.history.clear()
        self.score_history.clear()
        self.heads.clear()
        self.value = None

    def closedness(self, l: float, r: float, scores) -> tuple[float, float]:
        """Per eye: 1.0 = at the close threshold, ~0.67 = the reopen threshold."""
        close_thr, _ = self.thresholds()
        base = close_thr / self.close_ratio
        c = [(base - e) / max(base - close_thr, 1e-6) for e in (l, r)]
        if scores is not None:
            sb = self.score_bins.get(self.pose)
            if sb is None:
                sb = self.score_base if self.score_base is not None else min(scores)
            c = [max(ce, (sc - sb) / self.score_delta) for ce, sc in zip(c, scores)]
        return c[0], c[1]

    def eyes_open(self, left: float, right: float, pose: Pose | None = None) -> bool:
        """Both eyes open by their eyelids, for this head direction (close gesture)."""
        self.pose = pose
        return max(self.closedness(left, right, None)) < 1.0

    def learn_open(self, left: float, right: float, scores, pose: Pose | None) -> None:
        """How open eyes look in this head direction (from the calibration)."""
        self.pose = pose
        self._learn((left + right) / 2.0, scores)

    def _learn(self, mean: float, sc, overall: bool = True) -> None:
        """Eyes open: adapt the open-eye levels (overall ~2 s, and this direction's)."""
        m = None if sc is None else min(float(np.mean(sc)), BLINK_SCORE_BASE_MAX)
        if overall:
            self.baseline = mean if self.baseline is None else self.baseline + 0.015 * (mean - self.baseline)
            if m is not None:
                self.score_base = m if self.score_base is None else self.score_base + 0.015 * (m - self.score_base)
        self.ear_bins.learn(self.pose, mean)
        if m is not None:
            self.score_bins.learn(self.pose, m)

    WINK_OTHER_OPEN = 0.6    # during a wink the other eye's closedness stays below this

    def update(self, left: float, right: float, now: float,
               scores: tuple[float, float] | None = None,
               winks_ok: bool = True, pose: Pose | None = None,
               head=None) -> tuple[bool, str, bool]:
        """
        Feed one frame: per-eye EAR, and optionally per-eye blink scores
        (MediaPipe eyeBlinkLeft/Right), the head pose and the head position
        (the nose in face widths, see _process). Returns (eyes_closed,
        click_fired, just_closed). click_fired is "" (none), "left" or "right".
        """
        self.cue = ""
        self.history.append((left, right))
        l, r = np.median(np.array(self.history), axis=0)
        sc = None
        if scores is not None:
            self.score_history.append(scores)
            sc = tuple(np.median(np.array(self.score_history), axis=0))
        mean = float(l + r) / 2.0
        self.value = mean
        closing = self.wink_eye is not None or self.closed_since is not None
        # MediaPipe's head angles jump ~8 degrees when both eyes close (a real
        # user's log: every deliberate blink "moved the head" 7-9 degrees), so
        # during a closure the open-eye levels of where it began are used.
        self.pose = self.closed_pose if closing else pose
        head = self._head(head, pose)
        if head is not None:
            self.heads.append((now, head))
        cl, cr = self.closedness(l, r, sc)
        el, er = self.closedness(l, r, None)     # the eyelids alone: how DEEP (BLINK_DEEP)
        # Reopen level in the same units: 0.67 for the default 0.55 / 0.70 ratios.
        reopen = (1 - self.open_ratio) / max(1 - self.close_ratio, 1e-6)

        if self.head_moved:
            # The eyes only looked closed because of where the head is:
            # treat them as open and learn how open eyes look here.
            if (cl + cr) / 2 < reopen:
                self.head_moved = False
            else:
                self._learn(mean, sc, overall=False)
                return False, "", False

        if closing:
            self.deep = max(self.deep, self._depth(el, er))
            if head is not None and self.closed_head is not None:
                by = head - self.closed_head
                if float(np.linalg.norm(by)) > self.moved:
                    self.moved, self.moved_by = float(np.linalg.norm(by)), by
            if (self.moved > BLINK_MAX_MOVE_FW and self.deep < BLINK_DEEP
                    and not (self.KEEP_TICKED and self.cued)):
                # The head moved and the eyelids never really closed: the eyes
                # only look closed where the head went (e.g. the screen bottom).
                started = self.wink_since if self.wink_eye is not None else self.closed_since
                # Shorter than a click could be: a glance down or a blink mid-turn,
                # not a missed click. (Closures that began with the head moving
                # were never logged before 1.5.11.)
                note = log.info if now - started >= self.min_s else log.debug
                # The depth and movement say which limit stopped it: three of
                # the user's 0.47-0.83 s closures on 2026-10-24 ended here.
                note(f"[blink] Head moved while the eyes only looked half closed "
                     f"({(now - started) * 1000:.0f} ms{', began while turning' if self.deferred else ''}; "
                     f"eyelids {self.deep:.2f} of {BLINK_DEEP:g}, {self._moved_text()}) - not a blink, no click")
                self.closed_since, self.wink_eye, self.deferred = None, None, False
                self.head_moved = True
                return False, "", False

        if self.wink_eye is not None:
            return self._wink(cl, cr, el, er, reopen, now)

        if self.closed_since is None:
            # Which eye is shut is told by the eyelids whenever they show it:
            # one closed, the other open. MediaPipe's blink SCORES rise for
            # both eyes in a wink (the open eye squints a little too), and
            # judged by them a right wink looked like a blink - a LEFT click.
            lid_wink = winks_ok and max(el, er) >= 1.0 and min(el, er) < self.WINK_OTHER_OPEN
            # Both eyes closing - on average, the other eye at least partly: the
            # user's right eye closes later and less than the left (1.5.2's log:
            # blinks began as "left winks" and stayed so - one tick, no double,
            # no voice typing), and a blink with it 0.6-1.0 closed counted as nothing.
            both = ((cl + cr) / 2 >= 1.0 and min(cl, cr) >= self.WINK_OTHER_OPEN) and not lid_wink
            wink = lid_wink or (winks_ok and max(cl, cr) >= 1.0 and min(cl, cr) < self.WINK_OTHER_OPEN)
            if not (both or wink):
                # Open: adapt the baselines slowly (~2 s time constant at 30 FPS).
                self._learn(mean, sc)
                return False, "", False
            moving = self._head_speed(now) > BLINK_MOVING_FW_S
            if both:
                self.closed_since = now
            else:
                self.wink_by_lids = lid_wink
                shut_left = el >= er if lid_wink else cl >= 1.0
                self.wink_eye, self.wink_since = ("left" if shut_left else "right"), now
            self._closing(pose, head, now, moving, self._depth(el, er))
            # With the head still, the cursor is held at once; while it moves,
            # only once this turns out to be a click (_still_closed).
            return not moving, "", not moving

        if (cl + cr) / 2 >= reopen:
            # One reopened eye with an unusually high score on the other must
            # not arm a click through the average alone. Unequal closure still
            # counts: min() uses the participation threshold from blink onset.
            self.shut = ((cl + cr) / 2 >= self.SHUT and
                         min(cl, cr) >= self.WINK_OTHER_OPEN)
            return self._still_closed(now)

        duration = now - self.closed_since
        self.closed_since = None
        if self.deferred:                      # a natural blink while the head moved
            self.deferred = False
            return False, "", False
        click = ""
        # A short blink that starts soon after resting the eyes is the natural
        # one; a long, deliberate closure (>= AFTER_REST_MAX_BLINK_S) still clicks.
        after_rest = ((now - duration) - self.rested_at < self.AFTER_REST_S
                      and duration < self.AFTER_REST_MAX_BLINK_S)
        # A click needs its tick (cued): no tick, no click - a rule the user can hear.
        if (self.cued >= 1 and self.min_s <= duration <= self.max_s
                and now - self.last_click >= self.cooldown_s and not after_rest):
            self.last_click = now
            # The button is what the ticks said (the duration measured at the
            # reopening frame can just pass right_s after one tick).
            click = "right" if self.cued == 2 else "left"
            log.info(f"[blink] {'right click (long blink, ' if click == 'right' else 'click ('}"
                     f"{duration * 1000:.0f} ms closure"
                     f"{', head moving' if self.moving_start or self.moved > BLINK_MAX_MOVE_FW else ''})")
        elif self.min_s <= duration <= self.max_s and after_rest:
            log.info(f"[blink] blink right after resting the eyes ({duration * 1000:.0f} ms) "
                     "- not a click")
        elif duration > self.max_s:
            self.rested_at = now
            log.info(f"[blink] eyes closed {duration:.1f}s - treated as resting, no click")
        elif (0.15 <= duration and self.cued == 0
              and (duration >= BLINK_MIN_S or now - self.last_hint > 20)):
            # Every closure as long as the shortest tick is logged (1.5.13): one
            # hint per 20 s hid how many meant clicks went missing.
            self.last_hint = now
            # Long enough but no tick: the eyes were already opening when it was due
            # (TICK_WHEN_SHUT) - the user's 425 ms closure of 2026-10-23 said "too short".
            first = self._first_tick()
            why = ("the eyes were opening before the tick" if duration >= first
                   else "too short for a click")
            log.info(f"[blink] {duration * 1000:.0f} ms closure - {why} "
                     f"(keep your eyes shut until the tick, about {first:.1f} s)")
        return False, click, False

    def _depth(self, el: float, er: float) -> float:
        """
        How far the closing eye(s) shut by the eyelids alone (closedness
        units). A blink: both eyes on average - many people close one eye
        less (a real user's right eye stopped at 1.2-1.3 while the left
        reached 1.4-1.5, so judged by the less closed eye, firm blinks
        counted as "half closed" and were dropped). A head position that only
        makes the eyes LOOK closed lowers both alike, to about 1.0.
        """
        if self.wink_eye == "left":
            return el
        if self.wink_eye == "right":
            return er
        return (el + er) / 2

    def _closing(self, pose: Pose | None, head: np.ndarray | None, now: float,
                 moving: bool, depth: float) -> None:
        """A closure starts: remember where the head was, and whether it was moving."""
        self.closed_pose = pose
        # From just before the eyelids started to drop (~0.1 s).
        earlier = [h for t, h in self.heads if t <= now - 0.1]
        self.closed_head = earlier[-1] if earlier else head
        self.moved, self.moved_by, self.deep = 0.0, np.zeros(2), depth
        self.moving_start = self.deferred = moving
        self.started_at = now
        self.cued = 0

    def _cue(self, now: float) -> None:
        """
        Tick once opening the eyes now would click (see the class
        docstring): "tick" when it becomes a click, "double" when it becomes
        a right click (both eyes only - a wink's button is its eye).
        """
        wink = self.wink_eye is not None
        started = self.wink_since if wink else self.closed_since
        if started is None or (self.TICK_WHEN_SHUT and not self.shut):
            return
        held = now - started
        after_rest = started - self.rested_at < self.AFTER_REST_S
        first = self._first_tick(after_rest and not wink)
        if (self.cued == 0 and first <= held <= self.max_s
                and now - self.last_click >= self.cooldown_s):
            self.cued, self.cue = 1, "tick"
        elif (self.cued == 1 and not wink and self.right_s is not None
                and self._double_tick(first) <= held <= self.max_s):
            self.cued, self.cue = 2, "double"

    def _first_tick(self, after_rest: bool = False) -> float:
        """When this closure's tick comes: later right after resting the eyes,
        or when it began with the head moving (natural blinks while reading)."""
        return max(self.min_s,
                   self.AFTER_REST_MAX_BLINK_S if after_rest else 0.0,
                   self.MOVING_MIN_S if self.moving_start else 0.0)

    def _double_tick(self, first: float) -> float:
        """When the double tick comes: as long after the first tick as the
        defaults leave (1.0 s), also when a calibrated tick came later - 1.5.9
        kept it at 1.3 s, so a 0.6 s tick left 0.7 s, and the user opens
        0.55-0.9 s after the tick they wait for: left clicks became right ones."""
        return self.right_s + first - min(self.min_s, BLINK_MIN_S)

    def _still_closed(self, now: float) -> tuple[bool, str, bool]:
        """
        A closure goes on. Started while the head moved (``deferred``), the
        cursor keeps moving until it is clearly a click: deep (BLINK_DEEP) and
        as long as a click (min_s) - then it is held, back where the blink
        began (just_closed). A natural blink mid-turn never holds it.
        """
        if not self.deferred:
            self._cue(now)
            return True, "", False
        if self.deep >= BLINK_DEEP and now - self.started_at >= self._first_tick():
            self.deferred = False
            self._cue(now)
            return True, "", True
        if now - self.started_at > BLINK_PASSING_MAX_S:
            # Never deep, this long: the eyes only look closed where the head
            # went - learn how open eyes look here (head_moved). Logged since
            # 1.5.11 (it used to be silent), to tell a missed click from this.
            log.info(f"[blink] Eyes closed while the head moved, but the eyelids never shut fully "
                     f"({(now - self.started_at) * 1000:.0f} ms; eyelids {self.deep:.2f} of "
                     f"{BLINK_DEEP:g}, {self._moved_text()}) - no click")
            self.closed_since, self.wink_eye, self.deferred = None, None, False
            self.head_moved = True
        return False, "", False

    def _moved_text(self) -> str:
        """How the head moved during this closure, for the log: sideways
        (turning) or up-down (e.g. looking at the screen bottom)."""
        x, y = (abs(float(v)) for v in self.moved_by)
        return f"head moved {self.moved:.2f} face widths ({x:.2f} sideways, {y:.2f} up-down)"

    @staticmethod
    def _head(head, pose: Pose | None) -> np.ndarray | None:
        """Head position in face widths: the nose (preferred), else from the angles."""
        if head is not None:
            return np.asarray(head, np.float64)
        if pose is not None:                      # ~29 degrees per face width (HEAD_RANGE)
            return np.array(pose, np.float64) / DEG_PER_FACE_WIDTH
        return None

    def _head_speed(self, now: float) -> float:
        """How fast the head moved over the last ~0.2 s (face widths a second)."""
        recent = [(t, h) for t, h in self.heads if now - t <= 0.25]
        if len(recent) < 3:
            return 0.0
        (t0, h0), (t1, h1) = recent[0], recent[-1]
        return float(np.linalg.norm(h1 - h0)) / max(t1 - t0, 1e-3)

    def _wink(self, cl: float, cr: float, el: float, er: float, reopen: float,
              now: float) -> tuple[bool, str, bool]:
        """
        One eye closed (see the class docstring): left eye = left click, right
        = right. A wink the eyelids showed is followed by the eyelids (the
        blink scores of both eyes rise during it); otherwise by both signals.
        """
        if self.wink_by_lids:
            cl, cr = el, er
        shut, other = (cl, cr) if self.wink_eye == "left" else (cr, cl)
        if other >= self.WINK_OTHER_OPEN:
            # The other eye is closing too (a wink keeps it open): a blink that
            # began on one side - from its start, with its ticks so far.
            self.wink_eye, self.closed_since = None, self.wink_since
            self.shut = ((cl + cr) / 2 >= self.SHUT and
                         min(cl, cr) >= self.WINK_OTHER_OPEN)
            return self._still_closed(now)
        if shut >= reopen:
            self.shut = shut >= self.SHUT
            return self._still_closed(now)     # still winking
        eye, self.wink_eye = self.wink_eye, None
        if self.deferred:                      # a natural wink-like blink while moving
            self.deferred = False
            return False, "", False
        duration = now - self.wink_since
        if (self.cued >= 1 and self.min_s <= duration <= self.max_s
                and now - self.last_click >= self.cooldown_s):
            self.last_click = now
            log.info(f"[blink] {eye} click ({eye} eye closed {duration * 1000:.0f} ms"
                     f"{', head moving' if self.moving_start or self.moved > BLINK_MAX_MOVE_FW else ''})")
            return False, eye, False
        return False, "", False


# --------------------------------------------------------------------------- #
# Calibration: fit the pointer, blink clicks and smile gesture to the user
# --------------------------------------------------------------------------- #

@dataclass
class CalibFrame:
    """What the calibration needs from one camera frame."""
    nose: np.ndarray                         # nose tip, camera pixels
    face_px: float                           # eye-corner distance, pixels
    pose: Pose | None
    ear: tuple[float, float]                 # (left, right) eye aspect ratio
    scores: tuple[float, float] | None       # MediaPipe eyeBlinkLeft / Right
    smile: float | None                      # mean of mouthSmileLeft / Right


class Calibration:
    """
    A ~30 s guided calibration; the user only follows dots on the screen
    (main_window shows them) and it moves on by itself:

      centre     look at the middle and keep still: where the head rests,
                 and how open eyes and a resting mouth look
      left, right, up, down
                 turn the head to look at each screen edge and hold: the head
                 movement the user finds natural for crossing the screen.
                 Speed 1 is then set so that this movement, at a normal pace,
                 crosses it (HeadPointer.calib) - neither more turning than
                 is comfortable, nor a cursor that flies off. The open-eye
                 level of each direction is learned on the way (PoseBins),
                 so looking at the edges never looks like a blink.
      blink      blink slowly 3 times: how far this user's eyes close by
                 their eyelids and by MediaPipe's blink score (with glasses
                 the score may rise only ~0.3), so blink clicks fit them
      smile      a big smile: the smile gesture's threshold is set below it
                 (a light smile stays well under)

    A step that can't be completed in time (no face, too little movement)
    is skipped and that setting keeps its default. ``results()`` gives what
    to save; apply_calibration() puts it to use.
    """

    STEPS = ("centre", "left", "right", "up", "down", "blink", "smile")
    HEAD_STEPS = ("centre", "left", "right", "up", "down")
    TEXT = {
        "centre": "Sit comfortably, look at the dot in the middle and keep still",
        "left": "Turn your HEAD (not just your eyes) towards the dot on the LEFT edge, then hold still",
        "right": "Turn your HEAD (not just your eyes) towards the dot on the RIGHT edge, then hold still",
        "up": "Tilt your HEAD up towards the dot at the TOP, then hold still",
        "down": "Tilt your HEAD down towards the dot at the BOTTOM, then hold still",
        "blink": "Look at the middle dot and blink slowly 3 times - eyes closed for about half a second",
        "smile": "Now give a big smile and hold it",
    }
    DIRECTION = {"left": (0, -1), "right": (0, 1), "up": (1, -1), "down": (1, 1)}

    STILL_S = 0.4          # the head counts as still when the nose stayed within
    STILL_FW = 0.015       # ... this many face widths for this long
    CENTRE_HOLD_S = 1.0    # then held still this long
    EDGE_HOLD_S = 0.8
    # An edge counts only this far from the centre (face widths; left-right,
    # up-down): a real user's head paused on the way at 0.15 (~3 degrees) and
    # that was taken as the left edge.
    EDGE_MIN_FW = (0.2, 0.1)
    STEP_TIMEOUT_S = 20.0  # a step not done by then is skipped
    BLINKS = 3
    BLINK_MIN_S, BLINK_MAX_S = 0.15, 2.0
    SMILE_RISE = 0.2       # a smile: this much above the resting level
    SMILE_HOLD_S = 0.8

    def __init__(self, now: float):
        self.index = 0
        self.step_since = now
        self.recent: deque[tuple[float, np.ndarray]] = deque()   # (time, nose in face widths)
        self.still_since: float | None = None
        self.held: list[CalibFrame] = []     # frames of the current still hold
        self.centre: np.ndarray | None = None
        self.face_px = 0.0
        self.open_ear = self.open_score = None
        self.rest_smile = 0.0
        self.edges: dict[str, tuple[float, Pose | None]] = {}
        # (lowest EAR ratio, highest score rise, detected closed seconds)
        self.blinks: list[tuple[float, float | None, float]] = []
        self.blink_frames: list[list[tuple[float, float, float | None]]] = []   # (t, ratio, rise)
        self.closure: dict | None = None
        self.smile_since: float | None = None
        self.smile_peak: float | None = None
        self.smiles: deque[float] = deque(maxlen=15)
        self.skipped: list[str] = []
        self.face = True
        self.learned: list[CalibFrame] = []  # open-eye frames per head direction, for the worker

    @property
    def step(self) -> str:
        return self.STEPS[self.index] if self.index < len(self.STEPS) else "done"

    @property
    def finished(self) -> bool:
        return self.index >= len(self.STEPS)

    def text(self) -> str:
        if self.finished:
            return ""
        prefix = "" if self.face else "Can't see your face - "
        if self.step == "blink":
            return prefix + self.TEXT["blink"] + f" ({len(self.blinks)} of {self.BLINKS})"
        return prefix + self.TEXT[self.step]

    def progress(self, now: float) -> float:
        step = self.step
        if step == "blink":
            return len(self.blinks) / self.BLINKS
        if step == "smile":
            return 0.0 if self.smile_since is None else min(1.0, (now - self.smile_since) / self.SMILE_HOLD_S)
        if self.still_since is None or step == "done":
            return 0.0
        hold = self.CENTRE_HOLD_S if step == "centre" else self.EDGE_HOLD_S
        return min(1.0, (now - self.still_since) / hold)

    def _next(self, now: float, skipped: bool = False) -> None:
        if skipped:
            self.skipped.append(self.step)
            log.info(f"[calibrate] Step '{self.step}' skipped (not completed in "
                     f"{self.STEP_TIMEOUT_S:.0f}s) - keeping its default")
        self.index += 1
        self.step_since = now
        self.recent.clear()
        self.still_since, self.held, self.closure, self.smile_since = None, [], None, None

    def _offset(self, f: CalibFrame) -> np.ndarray:
        """Nose position in face widths: from the resting centre once known,
        in cursor directions (x mirrored like HeadPointer: + = right)."""
        if self.centre is None:
            return f.nose / max(f.face_px, 1.0)
        return (f.nose - self.centre) / self.face_px * np.array([-1.0, 1.0])

    def _open_frames(self) -> list[CalibFrame]:
        """The held frames without blinks (a blink in the middle of a hold)."""
        ears = np.array([np.mean(f.ear) for f in self.held])
        med = float(np.median(ears))
        return [f for f, e in zip(self.held, ears) if e >= 0.85 * med]

    def update(self, f: CalibFrame | None, now: float) -> None:
        """Feed one frame (None: no face)."""
        if self.finished:
            return
        self.face = f is not None
        if f is None:
            self.recent.clear()
            self.still_since, self.held, self.smile_since = None, [], None
            return
        if self.step != "centre" and now - self.step_since > self.STEP_TIMEOUT_S:
            self._next(now, skipped=True)
            return
        if self.step in self.HEAD_STEPS:
            self._head_step(f, now)
        elif self.step == "blink":
            self._blink_step(f, now)
        else:
            self._smile_step(f, now)

    def _head_step(self, f: CalibFrame, now: float) -> None:
        pos = self._offset(f)
        self.recent.append((now, pos))
        while self.recent and now - self.recent[0][0] > self.STILL_S:
            self.recent.popleft()
        pts = np.array([p for _, p in self.recent])
        still = (now - self.recent[0][0] >= self.STILL_S * 0.8
                 and float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) <= self.STILL_FW)
        if self.step != "centre":
            axis, sign = self.DIRECTION[self.step]
            still = still and sign * float(pts[:, axis].mean()) >= self.EDGE_MIN_FW[axis]
        if not still:
            self.still_since, self.held = None, []
            return
        self.still_since = self.still_since or now
        self.held.append(f)
        hold = self.CENTRE_HOLD_S if self.step == "centre" else self.EDGE_HOLD_S
        if now - self.still_since < hold:
            return
        frames = self._open_frames()
        if self.step == "centre":
            self.centre = np.mean([g.nose for g in self.held], axis=0)
            self.face_px = float(np.mean([g.face_px for g in self.held]))
            self.open_ear = float(np.median([np.mean(g.ear) for g in frames]))
            if all(g.scores is not None for g in frames):
                self.open_score = float(np.median([np.mean(g.scores) for g in frames]))
            smiles = [g.smile for g in frames if g.smile is not None]
            self.rest_smile = float(np.median(smiles)) if smiles else 0.0
            log.info(f"[calibrate] Centre: face {self.face_px:.0f} px wide, open eyes EAR "
                     f"{self.open_ear:.3f}, blink score {self.open_score}")
        else:
            axis, sign = self.DIRECTION[self.step]
            reach = sign * float(np.mean([self._offset(g)[axis] for g in self.held]))
            self.edges[self.step] = (reach, f.pose)
            log.info(f"[calibrate] {self.step.capitalize()} edge: head moved {reach:.3f} face widths"
                     + ("" if f.pose is None else f" (yaw {f.pose[0]:.0f}, pitch {f.pose[1]:.0f} deg)"))
        self.learned.extend(frames)
        self._next(now)

    def _blink_step(self, f: CalibFrame, now: float) -> None:
        ratio = float(np.mean(f.ear)) / max(self.open_ear or 0.28, 1e-6)
        rise = None
        if f.scores is not None and self.open_score is not None:
            rise = float(np.mean(f.scores)) - self.open_score
        closed = ratio < 0.75 or (rise is not None and rise > 0.15)
        opened = ratio > 0.85 and (rise is None or rise < 0.08)
        if self.closure is None:
            if closed:
                self.closure = {"since": now, "ratio": ratio, "rise": rise, "frames": [(now, ratio, rise)]}
            return
        c = self.closure
        c["frames"].append((now, ratio, rise))
        c["ratio"] = min(c["ratio"], ratio)
        if rise is not None:
            c["rise"] = rise if c["rise"] is None else max(c["rise"], rise)
        if not opened:
            return
        self.closure = None
        duration = now - c["since"]
        if self.BLINK_MIN_S <= duration <= self.BLINK_MAX_S:
            self.blinks.append((c["ratio"], c["rise"], duration))
            self.blink_frames.append(c["frames"])
            log.info(f"[calibrate] Blink {len(self.blinks)}: eyelids down to {c['ratio']:.0%} "
                     f"of open for {duration:.2f} s"
                     + ("" if c["rise"] is None else f", blink score +{c['rise']:.2f}"))
            if len(self.blinks) >= self.BLINKS:
                self._next(now)

    def _smile_step(self, f: CalibFrame, now: float) -> None:
        if f.smile is None:                      # no expression scores: nothing to fit
            self._next(now, skipped=True)
            return
        self.smiles.append(f.smile)
        level = float(np.median(self.smiles))
        if level < self.rest_smile + self.SMILE_RISE:
            self.smile_since = None
            return
        self.smile_since = self.smile_since or now
        self.smile_peak = max(self.smile_peak or 0.0, level)
        if now - self.smile_since >= self.SMILE_HOLD_S:
            log.info(f"[calibrate] Big smile: {self.smile_peak:.2f} (resting {self.rest_smile:.2f})")
            self._next(now)

    @staticmethod
    def _shut_s(frames: list[tuple[float, float, float | None]], close: float, delta: float) -> float:
        """How long a calibration blink was shut the way BlinkDetector sees it:
        eyelids at the close ratio (closedness 1.0) or the score risen by delta."""
        shut = [t for t, ratio, rise in frames
                if ratio <= close or (rise is not None and rise >= delta)]
        return shut[-1] - shut[0] if shut else 0.0

    def results(self) -> dict:
        """What to save (see apply_calibration); only the steps completed."""
        res: dict = {"date": time.strftime("%Y-%m-%d")}
        e = self.edges

        def span(a: str, b: str, lo: float, hi: float) -> float | None:
            if a not in e and b not in e:
                return None
            one, two = e.get(a, e.get(b))[0], e.get(b, e.get(a))[0]
            return round(float(np.clip(one + two, lo, hi)), 3)

        x, y = span("left", "right", 0.3, 1.6), span("up", "down", 0.15, 1.0)
        if x is not None:
            res["head_x"] = x
        if y is not None:
            res["head_y"] = y
        if "left" in e and "right" in e and e["left"][1] is not None and e["right"][1] is not None:
            res["turn_deg"] = round(abs(e["left"][1][0] - e["right"][1][0]))
        if self.blinks:
            ratio = float(np.median([b[0] for b in self.blinks]))
            if ratio <= 0.65:
                # Eyelids close clearly: "closed" 40% of the way from this
                # user's closed level to open (default 0.55 / 0.70).
                close = float(np.clip(ratio + 0.4 * (1 - ratio), 0.45, 0.72))
                res["blink_close_ratio"] = round(close, 3)
                res["blink_open_ratio"] = round(min(close + 0.15, 0.85), 3)
            rises = [b[1] for b in self.blinks if b[1] is not None]
            if rises and float(np.median(rises)) > 0.05:
                res["blink_score_delta"] = round(float(np.clip(0.5 * np.median(rises), 0.12, 0.4)), 3)
        if len(self.blink_frames) >= self.BLINKS:
            # The tick: 90% of how long these blinks stayed SHUT by the
            # thresholds just fitted - the tick comes only while the eyes are
            # shut (TICK_WHEN_SHUT). 1.5.9 took 90% of the whole closure, from
            # the lids a quarter down until fully open again, which is up to
            # ~0.2 s longer: the tick could fall after the demonstrated blink
            # had begun to open, and that very blink didn't click.
            close = res.get("blink_close_ratio", 0.55)
            delta = res.get("blink_score_delta", BLINK_SCORE_DELTA)
            shut = float(np.median([self._shut_s(f, close, delta) for f in self.blink_frames]))
            res["blink_tick_s"] = round(float(np.clip(0.9 * shut, BLINK_MIN_S, BLINK_CALIBRATED_MAX_S)), 3)
        if self.smile_peak is not None:
            res["smile_min"] = round(float(np.clip(0.75 * self.smile_peak, 0.45, 0.85)), 3)
        return res


CALIB_RANGE = (0.7, 1.3)   # a calibrated head range stays within these factors of the default


def apply_calibration(cal: dict | None, head: "HeadPointer", blink: BlinkDetector,
                      gestures: FaceGestures, timing: bool = True) -> None:
    """Put a saved calibration (Calibration.results) to use; None: the defaults."""
    cal = cal if isinstance(cal, dict) else {}
    default_x = HEAD_RANGE[DEFAULT_SENSITIVITY][1]
    default_y = default_x * (head.size[1] / head.size[0]) / HEAD_Y_GAIN
    if "head_x" in cal or "head_y" in cal:
        # A calibration fine-tunes the speed, within CALIB_RANGE of the
        # defaults: a real user's measured 2.2x the default up-down range (one
        # edge rushed, the other exaggerated), which would have made the
        # cursor crawl vertically.
        lo, hi = CALIB_RANGE
        x, y = float(cal.get("head_x", default_x)), float(cal.get("head_y", default_y))
        cx, cy = float(np.clip(x, lo * default_x, hi * default_x)), float(np.clip(y, lo * default_y, hi * default_y))
        if (cx, cy) != (x, y):
            log.info(f"[calibrate] Head range {x:.2f} x {y:.2f} face widths limited to "
                     f"{cx:.2f} x {cy:.2f} (within {lo:.0%}-{hi:.0%} of the defaults)")
        head.calib = (cx, cy)
    else:
        head.calib = None
    if blink.fixed is None and "blink_close_ratio" in cal:
        blink.close_ratio = float(cal["blink_close_ratio"])
        blink.open_ratio = float(cal.get("blink_open_ratio", blink.close_ratio + 0.15))
    if "blink_score_delta" in cal:
        blink.score_delta = float(cal["blink_score_delta"])
    # An explicit --min-blink is an accessibility override, not replaced by
    # calibration. Otherwise use the tick this user demonstrated they can reach.
    # 1.5.9's "blink_min_s" isn't used: it was measured over the whole closure
    # and may come after the eyes begin to open (see Calibration.results) -
    # those users keep the default tick until they calibrate again.
    if timing and "blink_tick_s" in cal:
        blink.min_s = float(np.clip(cal["blink_tick_s"], BLINK_MIN_S, BLINK_CALIBRATED_MAX_S))
    gestures.minimum["smile"] = float(cal.get("smile_min", GESTURES["smile"][1]))


def calibration_summary(cal: dict, skipped: list[str]) -> str:
    """One or two sentences for the screen when a calibration ends."""
    parts = []
    if "head_x" in cal:
        turn = f" (about {cal['turn_deg']} degrees across)" if "turn_deg" in cal else ""
        parts.append(f"Speed 1 now fits your natural head movement{turn}.")
    if "blink_close_ratio" in cal or "blink_score_delta" in cal or "blink_tick_s" in cal:
        tick = f": the tick comes at {cal['blink_tick_s']:.2f} s" if "blink_tick_s" in cal else ""
        parts.append(f"Blink clicks are fitted to your blinks{tick}.")
    if "smile_min" in cal:
        parts.append("The smile gesture is fitted to your smile.")
    if skipped:
        parts.append(f"Skipped: {', '.join(skipped)} (kept the defaults).")
    return "Calibration done! " + (" ".join(parts) if parts else "Nothing could be measured.")


# --------------------------------------------------------------------------- #
# Vision worker (runs in its own thread)
# --------------------------------------------------------------------------- #

class MeshHolder:
    """
    Keeps one face mesh loaded while EyeSync is idle. Creating it takes ~5 s
    on a laptop (more on battery), which would otherwise be added to every
    website connect; the camera itself opens in ~2 s. A worker takes the
    mesh, and hands it back when it stops cleanly - a worker abandoned
    because its camera hung keeps its mesh, so two threads never share one.
    """

    PRELOAD_WAIT_S = 15.0    # a worker waits this long for a preload under way

    def __init__(self):
        self._lock = threading.Lock()
        self._mesh: fe.FaceMeshBackend | None = None
        self._preloaded = threading.Event()
        self._preloaded.set()                  # no preload under way

    def expect_preload(self) -> None:
        """A preload is about to start (on another thread): take() waits for it."""
        self._preloaded.clear()

    def preload(self) -> None:
        self._preloaded.clear()
        try:
            self.give(fe.FaceMeshBackend())
        except Exception:
            log.exception("[mediapipe] Preloading the face mesh failed; it will load on demand")
        finally:
            self._preloaded.set()

    def take(self) -> fe.FaceMeshBackend:
        # The camera opens in ~2 s, the first mesh takes ~3 s: wait for the
        # preload rather than build a second mesh beside it (1.5.11 did, and
        # the two loads slowed each other down at every Connect).
        self._preloaded.wait(self.PRELOAD_WAIT_S)
        with self._lock:
            mesh, self._mesh = self._mesh, None
        return mesh if mesh is not None else fe.FaceMeshBackend()

    def give(self, mesh: fe.FaceMeshBackend) -> None:
        with self._lock:
            if self._mesh is None:
                self._mesh = mesh
                return
        mesh.close()                       # already holding one

    def close(self) -> None:
        with self._lock:
            mesh, self._mesh = self._mesh, None
        if mesh is not None:
            mesh.close()


class VisionWorker(threading.Thread):
    def __init__(self, args, screen: fe.ScreenSize, state: SharedState,
                 meshes: MeshHolder | None = None):
        super().__init__(name="vision", daemon=True)
        self.args = args
        self.screen = screen
        self.state = state
        self.meshes = meshes or MeshHolder()
        self.halt = threading.Event()      # stops just this worker (supervisor)
        self.error: BaseException | None = None
        self.started_at = time.monotonic()
        self.preview_shown = False
        self.last_preview = 0.0

        self.sensitivity = state.read().sensitivity
        min_cutoff, beta, start_px, stop_speed, speed = SENSITIVITY_LEVELS[self.sensitivity]
        self.px_scale = screen.width / 1920.0
        self.filter = OneEuroFilter(min_cutoff, beta)
        self.motion = CursorMotion(start_px * self.px_scale, stop_speed * self.px_scale,
                                   speed * self.px_scale, self.px_scale)
        self.blink = BlinkDetector(args.ear_threshold, args.close_ratio, args.open_ratio,
                                   args.min_blink, min(args.max_blink, VOICE_EYES_CLOSED_S - 0.1), args.click_cooldown,
                                   getattr(args, "right_blink", BLINK_RIGHT_S) or None)
        self.dwell = DwellClicker(args.dwell_time, 60.0 * self.px_scale)
        self.head = HeadPointer(screen.width, screen.height)
        self.control_backend = ControlBackend(screen.width, screen.height)
        self.was_active = False                  # cursor control was on last frame
        self.low_light = fe.LowLight()           # keeps tracking working in a dark room
        self.dark_since: float | None = None
        self.gestures = FaceGestures()
        self.gesture_warned = False              # the long-hold gesture in progress has beeped
        self.scroll = TiltScroll()
        self.tilt_held = False                   # the cursor is held for a head tilt (scroll)
        self.nose_history: deque[np.ndarray] = deque(maxlen=NOSE_FRAMES)
        self.click_mode = state.read().click_mode
        # Recent cursor positions, to put a click where the cursor was just
        # before the eyes began to close (see CLICK_REWIND_S).
        self.pos_history: deque[tuple[float, float, float]] = deque(maxlen=20)   # (time, x, y) ~0.6 s
        self.hold_until = 0.0
        self.screen_checked = time.monotonic()      # last look at the display's size
        self.fps = 0.0
        self.last_landmarks: np.ndarray | None = None
        self.last_mar: float | None = None
        self.gesture_since: float | None = None     # close gesture in progress
        self.gesture_low_since: float | None = None # mouth dipped below the keep level
        self.gesture_announced = False              # ... and its countdown has beeped
        self.face_lost_since: float | None = None
        self.calib: Calibration | None = None       # a calibration in progress
        self.calib_shown_until = 0.0                # its result stays on screen until then
        self.voice_closure: float | None = None     # the eye closure that switched voice typing
        # Eye gaze (gaze_engine): the eye model, its calibrated mapping to the
        # screen and Gaze jump - all None until the eyes are calibrated.
        self.gaze = ge.GazeTracker()
        self.gaze_model: ge.GazeModel | None = None
        self.gaze_jump: ge.GazeJump | None = None
        self.last_gaze: ge.GazeSample | None = None
        self.prev_nose: tuple[float, np.ndarray] | None = None   # for the head's speed
        self.jumped_at = -1e9
        self.jumps = 0
        settings = load_settings()
        apply_calibration(settings.get("calibration"), self.head, self.blink, self.gestures,
                          timing=getattr(self.args, "min_blink", BLINK_MIN_S) == BLINK_MIN_S)
        self._apply_gaze(settings.get("gaze"))
        if calibrating(state.read()) and state.read().calib_step != "starting":
            state.calibrate_request.set()           # the previous worker died mid-way: start over

    @property
    def halted(self) -> bool:
        return self.halt.is_set() or self.state.stop.is_set()

    # -- thread entry -------------------------------------------------------
    def run(self) -> None:
        cap = mesh = None
        try:
            cap = self._open_camera()
            if cap is None:
                return
            mesh = self.meshes.take()
            consecutive_errors = 0
            t_prev = time.monotonic()
            # Health statistics, logged every --health-interval seconds so a
            # long session leaves a record of how tracking was doing.
            stats = {"frames": 0, "faces": 0, "proc": 0.0, "errors": 0, "since": t_prev,
                     "moving": 0, "starts": 0, "jumps": 0}
            was_moving = False

            while not self.halted:
                ok, frame = cap.read()
                if not ok or frame is None:
                    log.warning("[camera] Frame grab failed; reopening camera...")
                    cap.release()
                    cap = self._open_camera()
                    if cap is None:
                        break
                    continue
                self.state.beat()

                t_proc = time.perf_counter()
                try:
                    self._process(frame, mesh)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    stats["errors"] += 1
                    if consecutive_errors in (1, 10) or consecutive_errors % 100 == 0:
                        log.exception(f"[vision] Frame processing error (#{consecutive_errors})")
                    if consecutive_errors == 30:
                        log.warning("[vision] Persistent errors; rebuilding the face mesh")
                        mesh.close()
                        mesh = fe.FaceMeshBackend()

                now = time.monotonic()
                try:
                    self._follow_screen(now)
                except Exception:
                    log.exception("[screen] Could not check the display size")
                stats["frames"] += 1
                stats["faces"] += self.last_landmarks is not None
                # How restless the cursor is: share of time moving, and how
                # often it starts a movement (jitter shows up as many starts).
                moving = self.motion.moving
                stats["moving"] += moving
                stats["starts"] += moving and not was_moving
                was_moving = moving
                stats["proc"] += time.perf_counter() - t_proc
                dt, t_prev = now - t_prev, now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt
                if now - stats["since"] >= self.args.health_interval and stats["frames"]:
                    n = stats["frames"]
                    log.info(f"[health] {n / (now - stats['since']):4.1f} fps, face tracked in "
                             f"{100 * stats['faces'] / n:3.0f}% of frames, "
                             f"{1000 * stats['proc'] / n:4.1f} ms/frame processing, "
                             f"{stats['errors']} errors, sensitivity "
                             f"{self.sensitivity}/5, cursor moving "
                             f"{100 * stats['moving'] / n:.0f}% of the time, "
                             f"{60 * stats['starts'] / (now - stats['since']):.0f} moves/min"
                             + ("" if self.gaze_model is None else
                                f", {self.jumps - stats['jumps']} gaze jumps"))
                    stats = {"frames": 0, "faces": 0, "proc": 0.0, "errors": 0, "since": now,
                             "moving": 0, "starts": 0, "jumps": self.jumps}

                self._update_preview(frame)
        except BaseException as exc:
            self.error = exc
            log.exception("[vision] Worker crashed")
        finally:
            if cap is not None:
                cap.release()
            if mesh is not None:
                if self.error is None:
                    self.meshes.give(mesh)         # keep it loaded for next time
                else:
                    mesh.close()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self.state.update(face=False, eyes_closed=False, confidence=0.0, hint="")
            self.state.set_preview(None)

    DARK_HINT = ("It's too dark for the camera. Turn on a light, or turn up your screen "
                 "brightness - the screen lights your face.")

    def _dark_hint(self, face: bool, now: float) -> None:
        """Explain a lost face in a dark room (after 2 s), and clear it again."""
        dark = not face and self.low_light.mean < 45
        self.dark_since = (self.dark_since or now) if dark else None
        hint = self.state.read().hint
        if self.dark_since is not None and now - self.dark_since > 2.0 and hint != self.DARK_HINT:
            self.state.update(hint=self.DARK_HINT)
            log.info(f"[camera] Too dark to see a face (picture brightness "
                     f"{self.low_light.mean:.0f}/255)")
        elif face and hint == self.DARK_HINT:
            self.state.update(hint="")

    CAMERA_HINT = ("Can't open the camera. Close other apps that use it (Zoom, Teams, a "
                   "browser tab), or allow camera access in Windows Settings > Privacy.")

    CAMERA_LOG_S = 60.0

    def _open_camera(self):
        """
        Open the camera, retrying every 2 s until it works or we're halted.
        While it can't be opened, that's logged once a minute (CAMERA_LOG_S):
        an app holding the camera for an hour used to write ~4 lines every few
        seconds, pushing the useful history out of the log.
        """
        failing_since = logged_at = None
        while not self.halted:
            now = time.monotonic()
            quiet = logged_at is not None and now - logged_at < self.CAMERA_LOG_S
            try:
                cap = fe.open_any_camera(self.args.camera, quiet=quiet)
                if self.state.read().hint == self.CAMERA_HINT:
                    self.state.update(hint="")
                if failing_since is not None:
                    log.info(f"[camera] Working again after {time.monotonic() - failing_since:.0f}s")
                return cap
            except RuntimeError as exc:
                if not quiet:
                    log.error(f"[camera] {exc}" if failing_since is None
                              else f"[camera] Still can't open the camera "
                                   f"({time.monotonic() - failing_since:.0f}s) - retrying every 2s")
                    logged_at = now
                failing_since = failing_since if failing_since is not None else now
                self.state.update(hint=self.CAMERA_HINT)
                self.halt.wait(2.0)
        return None

    def _apply_sensitivity(self, level: int) -> None:
        min_cutoff, beta, start_px, stop_speed, speed = SENSITIVITY_LEVELS[level]
        self.filter.min_cutoff, self.filter.beta = min_cutoff, beta
        self.motion.start_px = start_px * self.px_scale
        self.motion.stop_speed = stop_speed * self.px_scale
        self.motion.max_speed = speed * self.px_scale
        self.sensitivity = level

    def _follow_screen(self, now: float, probe=None) -> None:
        """
        Every SCREEN_CHECK_S: has the primary display's resolution changed (a
        monitor plugged in or out, Win+P, new display settings)? Then the
        cursor's range, speeds and Gaze jump follow it at once - the screen
        object is shared, so the mouse, the website and the XIAO do too.
        """
        if now - self.screen_checked < SCREEN_CHECK_S:
            return
        self.screen_checked = now
        new = (probe or fe.primary_screen_size)()
        old = self.screen
        if new is None or (new.width, new.height) == (old.width, old.height):
            return
        log.info(f"[screen] Display changed: {old.width}x{old.height} -> {new.width}x{new.height}")
        snap = self.state.read()
        scale = np.array([new.width / old.width, new.height / old.height])
        old.width, old.height = new.width, new.height
        self.px_scale = new.width / 1920.0
        self._apply_sensitivity(self.sensitivity)
        self.motion.px_scale = self.px_scale
        self.dwell.arm_px = 60.0 * self.px_scale
        self.head.size = np.array([new.width, new.height], np.float64)
        if self.gaze_jump is not None:
            self.gaze_jump.resize(new.width, new.height)
        if isinstance(self.calib, ge.GazeCalibration):
            # Its dots and error are in the old screen's pixels: start it again.
            log.info("[calibrate] The display changed during the eye calibration - "
                     "press Calibrate eyes again")
            self.state.calibrate_cancel.set()
            self.state.update(calib_step="", calib_text="", calib_progress=0.0)   # the dots go
        self._restart_motion(np.clip(np.array([snap.x, snap.y]) * scale, 0, self.head.size - 1))

    # -- per-frame pipeline -------------------------------------------------
    def _freeze(self) -> None:
        """Hold the cursor still; motion tracking restarts afterwards."""
        self.filter.reset()
        self.motion.resync()
        self.dwell.pause()
        self.nose_history.clear()

    def _close_gesture(self, mar: float, eyes_open: bool, now: float,
                       jaw: float | None = None) -> bool:
        """
        Mouth wide open + eyes open, held for --close-gesture-hold seconds ->
        shut EyeSync down. Returns True while the gesture is in progress.
        ``jaw`` is MediaPipe's jawOpen score, when available.
        """
        if self.args.no_close_gesture:
            return False
        if self.state.stop.is_set():
            return True                    # already shutting down
        if self.gesture_since is None:
            # Starting needs a clearly wide-open mouth with the eyes open (a
            # yawn squeezes the eyes shut, so it can't start the gesture),
            # and MediaPipe agreeing that the jaw is dropped.
            if (mar >= self.args.close_gesture_mar and eyes_open
                    and (jaw is None or jaw >= CLOSE_JAW_MIN)):
                self.gesture_since = now
                self.gesture_low_since = None
                self.gesture_announced = False
                return True
            return False
        # In progress: holding a wide-open mouth makes the eyes squint and the
        # opening waver, so only the mouth counts now, with a lower bar
        # (GESTURE_KEEP of the start threshold) and GESTURE_GRACE_S of
        # tolerance before a dip cancels it.
        if mar >= self.args.close_gesture_mar * GESTURE_KEEP:
            self.gesture_low_since = None
        else:
            self.gesture_low_since = self.gesture_low_since or now
            grace = GESTURE_CANCEL_S if mar < GESTURE_CLOSED_MAR else GESTURE_GRACE_S
            if now - self.gesture_low_since > grace:
                if self.gesture_announced:
                    log.info(f"[gesture] Close gesture cancelled (mouth closed to {mar:.2f} "
                             f"after {now - self.gesture_since:.1f}s)")
                self.gesture_since = None
                return False
        if (not self.gesture_announced and self.gesture_low_since is None
                and now - self.gesture_since >= GESTURE_CONFIRM_S):
            # Only a mouth kept wide open is announced: opening it for a
            # moment (talking, a breath) never even beeps.
            self.gesture_announced = True
            log.info(f"[gesture] Mouth wide open - hold {self.args.close_gesture_hold:.0f}s "
                     "to close EyeSync (close your mouth to cancel)")
            desktop.play_warning_sound()
        if now - self.gesture_since >= self.args.close_gesture_hold:
            log.info("[gesture] Close gesture completed - shutting EyeSync down")
            desktop.play_click_sound()
            self.state.stop.set()
        return True

    def _process(self, frame, mesh) -> None:
        snap = self.state.read()
        if snap.sensitivity != self.sensitivity:
            self._apply_sensitivity(snap.sensitivity)
        if snap.click_mode != self.click_mode:
            self.click_mode = snap.click_mode
            self.dwell.reset()                 # no surprise click right after switching

        landmarks = self.low_light.process(mesh, frame)
        self.last_landmarks = landmarks
        now = time.monotonic()
        self._dark_hint(landmarks is not None, now)
        self._calibration_control(now)
        if landmarks is None:
            # No face: hold the cursor where it is. Opening the mouth wide
            # often makes the tracker drop a frame or two, so the close
            # gesture survives short dropouts (GESTURE_DROPOUT_S).
            self.face_lost_since = self.face_lost_since or now
            if self.calib is not None:
                self.calib.update(None, now)
                self._publish_calibration(now)
            self.blink.reset()
            self.control_backend.reset()
            self.last_mar = None
            self.last_gaze = None
            if (self.gesture_since is not None
                    and now - self.face_lost_since > GESTURE_DROPOUT_S):
                if self.gesture_announced:
                    log.info("[gesture] Close gesture cancelled (face lost)")
                self.gesture_since = None
            self._freeze()
            self.state.update(face=False, eyes_closed=False, confidence=0.0)
            return
        self.face_lost_since = None

        h, w = frame.shape[:2]
        if self.calib is None:
            self._process_desktop_controls(landmarks, mesh, w, h, snap, now)
            return
        # Where the head points: blinks and gestures are judged against how
        # the face looks in that direction (looking at a screen edge changes
        # the eyelids, the mouth's shape and the expression scores).
        pose = mesh.last_pose
        bs = mesh.last_blendshapes
        left, right = fe.eye_aspect_ratios(landmarks, w, h)
        mar = fe.mouth_aspect_ratio(landmarks, w, h, pose)
        self.last_mar = mar
        eyes_open = self.blink.eyes_open(left, right, pose)
        xyz = getattr(mesh, "last_xyz", None)        # the landmarks with depth, for the gaze

        if self.calib is not None:
            # Calibrating: the cursor stays still and nothing clicks or fires.
            if isinstance(self.calib, ge.GazeCalibration):
                self.calib.update(ge.GazeFrame(xyz, w, h, (left + right) / 2, eyes_open), now)
            else:
                nose, face_px = fe.nose_pointer(landmarks, w, h)
                smile = None if bs is None else (bs.get("mouthSmileLeft", 0.0)
                                                 + bs.get("mouthSmileRight", 0.0)) / 2
                scores = None if bs is None else (bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
                self.calib.update(CalibFrame(nose, face_px, pose, (left, right), scores, smile), now)
                for g in self.calib.learned:      # how open eyes look in each head direction
                    self.blink.learn_open(g.ear[0], g.ear[1], g.scores, g.pose)
                self.calib.learned.clear()
            self._publish_calibration(now)
            if self.calib.finished:
                self._finish_calibration(now)
            self.gestures.reset()
            self._freeze()
            self.hold_until = now + 0.3
            self.state.update(face=True, eyes_closed=False)
            return
        self._track_gaze(xyz, w, h, (left + right) / 2, eyes_open, now)

        # Mouth wide open (close gesture, or a yawn): freeze the cursor and
        # cancel any blink in progress so it can't turn into a click.
        gesture = self._close_gesture(mar, eyes_open, now, None if bs is None else bs.get("jawOpen"))
        if gesture or mar >= self.args.close_gesture_mar * 0.85:
            self.blink.reset()
            self.gestures.reset()
            self._freeze()
            self.hold_until = now + 0.2
            self.state.update(face=True, eyes_closed=False)
            return

        # Head tilted: scroll (TiltScroll). The cursor stays put and nothing clicks.
        # Paused or inactive, a tilt is just a tilt (no scrolling, nothing held).
        tilt = snap.gestures and control_active(snap, self.args)
        wheel = self.scroll.update(fe.head_roll(landmarks, w, h) if tilt else None, now)
        if self.scroll.tilted and not self.tilt_held:
            # The tilt's sideways nose movement moved the cursor on the way: put
            # it back where it was while the head was straight (within 1 s).
            self.tilt_held = True
            if self.pos_history and now - self.scroll.calm_at <= 1.0:
                back = [(x, y) for t, x, y in self.pos_history if t <= self.scroll.calm_at]
                self._restart_motion(np.array(back[-1] if back else self.pos_history[0][1:],
                                              np.float64))
        self.tilt_held = self.scroll.tilted
        if self.scroll.tilted:
            if wheel:
                self.state.add_scroll(wheel)
            self.blink.reset()
            self.gestures.reset()
            self._freeze()
            self.hold_until = now + 0.2
            self.state.update(face=True, eyes_closed=False)
            return

        expressions = None
        if snap.gestures and bs is not None:
            expressions = {**bs, "mouthShift": fe.mouth_shift(landmarks, w, h)}   # the side gestures
        fired = self.gestures.update(expressions, now, pose,
                                     VOICE_IGNORED_GESTURES if snap.voice else frozenset())
        holding = self.gestures.active
        if (holding in GESTURE_HOLD_S and not self.gesture_warned
                and now - self.gestures.since >= GESTURE_CONFIRM_S):
            # A long hold (calibrate): beep once it's clearly meant, then count down.
            self.gesture_warned = True
            desktop.play_warning_sound()
            log.info(f"[gesture] {GESTURE_LABELS[holding]} - hold {self.gestures.hold_s():.0f}s "
                     f"(relax to cancel; {self.gestures.detail})")
        if holding is None:
            self.gesture_warned = False
        if fired:
            self._gesture_action(fired)
        if self.gestures.active is not None:
            # Holding an expression: keep the cursor still, and a smile's
            # squint must not count as a blink.
            self.blink.reset()
            self._freeze()
            self.hold_until = now + 0.2
            self.state.update(face=True, eyes_closed=False)
            return

        scores = None if bs is None else (bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        nose_px, face_px = fe.nose_pointer(landmarks, w, h)
        head_speed = self._head_speed(nose_px, face_px, now)
        if self.gestures.relaxed:
            closed, click, just_closed = self.blink.update(left, right, now, scores,
                                                           fe.eyes_level_ratio(landmarks, w, h) > 0.7,
                                                           pose, nose_px / face_px)
        else:
            # A gesture just fired and the face is still easing out of it: its
            # squint is not a blink (1.5.2's log: a "left wink" click the moment
            # a smile resumed EyeSync). No closure starts; the cursor moves.
            self.blink.reset()
            self.blink.cue = ""
            closed, click, just_closed = False, "", False
        self._voice_gesture(closed, snap, now)
        if self.blink.cue and snap.click_mode == "blink" and not self.args.no_click_sound:
            # Eyes still closed: open now and it clicks (tick) / right-clicks (double tick).
            desktop.play_tick_sound(double=self.blink.cue == "double")
        if just_closed and self.pos_history:
            # The eyelids start dropping a few frames before the closure is
            # detected, and that slightly shifts the face mesh. Rewind the
            # cursor to where it was ~130 ms before the eyes began to close,
            # so the click lands on target - also for a blink made while the
            # head moved, which holds the cursor only once it is clearly a click.
            back_to = self.blink.started_at - CLICK_REWIND_S
            earlier = [(x, y) for t, x, y in self.pos_history if t <= back_to]
            rx, ry = earlier[-1] if earlier else self.pos_history[0][1:]
            self.motion.pos = np.array([rx, ry])
            self.head.pos = np.array([rx, ry])
            self.state.update(x=rx, y=ry)
        if click and snap.click_mode == "blink":
            self.state.add_click(right=click == "right")
            if not self.args.no_click_sound:
                desktop.play_click_sound(right=click == "right")   # you hear every click

        if closed:
            # Freeze the cursor while the eyes are shut and briefly after.
            self.hold_until = now + 0.15
        if closed or now < self.hold_until:
            self._freeze()
            self.state.update(face=True, eyes_closed=closed)
            return

        active = control_active(snap, self.args)
        if active and not self.was_active:
            # Control just switched on: carry on from where the cursor is
            # (you may have used the mouse meanwhile) - never jump.
            here = desktop.cursor_position()
            self._restart_motion(None if here is None else np.array(here, np.float64))
        self.was_active = active
        nose, face_px = fe.nose_pointer(landmarks, w, h)
        self.nose_history.append(nose)
        nose = nose_smooth(self.nose_history)
        if self.state.recenter_request.is_set():
            self.state.recenter_request.clear()
            self._restart_motion(np.array([self.screen.width / 2, self.screen.height / 2]))
            log.info("[recenter] Cursor moved to the screen centre")
        target = self.head.target(nose, face_px, self.sensitivity, now)
        if self.gaze_jump is not None and self.gaze_jump.absorbing(head_speed, now):
            # Just jumped: the rest of the head's nudge doesn't carry the cursor on.
            self.head.reset(self.motion.pos)
            self._freeze()
            self.state.update(face=True, eyes_closed=False)
            return
        resting = not self.motion.moving
        before = None if self.motion.pos is None else self.motion.pos.copy()
        self._move_cursor(target, 1.0, snap, now)
        if resting and self.motion.moving and before is not None:
            self._gaze_jump(before, target, snap, active, now)

    def _process_desktop_controls(self, landmarks, mesh, width, height, snap, now) -> None:
        """Desktop project's gaze, smile left click and mouth-open right click.

        Keep the website project's blink recognizer solely for its long-eye-close
        voice typing switch. Its blink clicks and head pointer are not used.
        """
        if snap.paused:
            self.blink.reset()
            self.voice_closure = None
            closed = False
        else:
            pose = mesh.last_pose
            left, right = fe.eye_aspect_ratios(landmarks, width, height)
            scores = mesh.last_blendshapes
            blink_scores = None if scores is None else (
                scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
            nose, face_px = fe.nose_pointer(landmarks, width, height)
            closed, _old_click, _just_closed = self.blink.update(
                left, right, now, blink_scores,
                fe.eyes_level_ratio(landmarks, width, height) > 0.7,
                pose, nose / face_px)
            self._voice_gesture(closed, snap, now)
        active = control_active(snap, self.args)
        target, click, held = self.control_backend.update(
            landmarks, width, height,
            click_enabled=active and snap.click_mode == "blink",
            voice_active=bool(snap.voice), now=now)
        if click and active:
            self.state.add_click(right=click == "right")
            if not self.args.no_click_sound:
                desktop.play_click_sound(right=click == "right")
        if closed or held or target is None:
            self.state.update(face=True, eyes_closed=closed)
            return
        x, y = float(target[0]), float(target[1])
        self.motion.pos = target
        self.state.update(x=x, y=y, pos_time=now, has_position=True,
                          face=True, eyes_closed=False, confidence=1.0)

    def _voice_gesture(self, closed: bool, snap: Snapshot, now: float) -> None:
        """
        Both eyes closed for VOICE_EYES_CLOSED_S: toggle voice typing once
        per closure. Only a real closure counts: both eyes (not a wink) and
        deep by the eyelids (BLINK_DEEP), not eyes that look closed because of
        where the head points. A click needs the eyes open again by
        --max-blink (BLINK_MAX_S, shorter), so this never clicks.
        """
        b = self.blink
        if snap.paused:
            self.voice_closure = None
            return
        if not closed or b.closed_since is None:
            self.voice_closure = None
            return
        if (self.voice_closure != b.closed_since
                and now - b.closed_since >= VOICE_EYES_CLOSED_S and b.deep >= BLINK_DEEP):
            self.voice_closure = b.closed_since
            turn_on = snap.voice in ("", "error")
            log.info(f"[gesture] Eyes closed {VOICE_EYES_CLOSED_S:g}s: voice typing "
                     f"{'on' if turn_on else 'off'}")
            desktop.play_voice_sound(turn_on)
            voice_typing.request(self.state, turn_on, "eyes closed")

    # -- eye gaze (see gaze_engine) ------------------------------------------
    def _apply_gaze(self, cal: dict | None) -> bool:
        """Use a saved eye calibration (GazeCalibration.results); False if it can't be used."""
        if not isinstance(cal, dict):
            return False
        tracker = ge.GazeTracker.from_dict(cal.get("tracker"))
        model = ge.GazeModel.from_dict(cal.get("model"))
        if tracker is None or model is None:
            return False
        self.gaze, self.gaze_model = tracker, model
        self.gaze_jump = ge.GazeJump(self.screen.width, self.screen.height,
                                     float(cal.get("error_px", 0)), float(cal.get("noise_px", 0)))
        return True

    def _track_gaze(self, xyz, w: int, h: int, lid: float, eyes_open: bool, now: float) -> None:
        """This frame's gaze -> Gaze jump and the camera view. Closed eyes give none."""
        self.last_gaze = None
        if self.gaze_model is None or self.gaze_jump is None:
            return
        sample = self.gaze.sample(xyz, w, h, lid) if eyes_open else None
        self.last_gaze = sample
        spot = None
        if sample is not None:
            spot = self.gaze_model.predict(sample.features()) * np.array(
                [self.screen.width, self.screen.height], np.float64)
        self.gaze_jump.add(spot, now)

    def _head_speed(self, nose: np.ndarray, face_px: float, now: float) -> float:
        """How fast the head moves, face widths per second (frame to frame)."""
        prev, self.prev_nose = self.prev_nose, (now, np.asarray(nose, np.float64).copy())
        if prev is None or not 0 < now - prev[0] <= HeadPointer.GAP_S:
            return 0.0
        return float(np.linalg.norm(self.prev_nose[1] - prev[1])) / max(face_px, 1.0) / (now - prev[0])

    def _gaze_jump(self, cursor: np.ndarray, target: np.ndarray, snap: Snapshot,
                   active: bool, now: float) -> None:
        """
        The head just started moving the cursor (from ``cursor`` towards
        ``target``): if the eyes rest on a spot far away in that direction,
        the cursor jumps there (see gaze_engine.GazeJump).
        """
        if not (active and snap.gaze_jump and self.gaze_jump is not None):
            return
        spot = self.gaze_jump.jump(cursor, target - cursor, now)
        if spot is None:
            return
        self._restart_motion(spot)
        self.state.update(pos_time=now)
        self.jumped_at = now
        self.jumps += 1
        log.debug(f"[gaze] Jump {float(np.linalg.norm(spot - cursor)):.0f} px to "
                  f"({spot[0]:.0f}, {spot[1]:.0f})")

    # -- calibration (see Calibration) ---------------------------------------
    CALIB_RESULT_S = 6.0      # the result stays on the screen this long

    def _calibration_control(self, now: float) -> None:
        """Start / cancel a calibration on request; clear its result later."""
        if self.state.calibrate_request.is_set():
            self.state.calibrate_request.clear()
            self.state.calibrate_cancel.clear()
            if self.state.calibrate_kind == "gaze":
                self.calib = ge.GazeCalibration(now, self.screen.width, self.screen.height)
                log.info("[calibrate] Eye calibration started")
            else:
                self.calib = Calibration(now)
                log.info("[calibrate] Calibration started")
            self._publish_calibration(now)
        if self.state.calibrate_cancel.is_set():
            self.state.calibrate_cancel.clear()
            if self.calib is not None:
                self.calib = None
                self.blink.reset()
                log.info("[calibrate] Cancelled")
        if self.calib is None and self.calib_shown_until and now > self.calib_shown_until:
            self.calib_shown_until = 0.0
            if self.state.read().calib_step == "done":
                self.state.update(calib_step="", calib_text="", calib_progress=0.0)

    def _publish_calibration(self, now: float) -> None:
        c = self.calib
        if c is not None and not c.finished:
            self.state.update(calib_step=c.step, calib_text=c.text(),
                              calib_progress=round(c.progress(now), 2))

    def _finish_calibration(self, now: float) -> None:
        """Save and use the result; speed 1 is the calibrated speed."""
        if isinstance(self.calib, ge.GazeCalibration):
            self._finish_gaze_calibration(now)
            return
        cal, skipped = self.calib.results(), self.calib.skipped
        self.calib = None
        apply_calibration(cal, self.head, self.blink, self.gestures,
                          timing=getattr(self.args, "min_blink", BLINK_MIN_S) == BLINK_MIN_S)
        self.blink.reset()
        save_settings(calibration=cal, sensitivity=DEFAULT_SENSITIVITY)
        summary = calibration_summary(cal, skipped)
        log.info(f"[calibrate] Done: {cal}")
        self.state.update(calib_step="done", calib_text=summary, calib_progress=1.0,
                          calibrated=cal["date"], sensitivity=DEFAULT_SENSITIVITY)
        self.calib_shown_until = now + self.CALIB_RESULT_S
        desktop.play_click_sound()

    def _finish_gaze_calibration(self, now: float) -> None:
        """Save and use the eye calibration; Gaze jump goes on if it is accurate enough.
        One that failed keeps the previous calibration."""
        res, skipped = self.calib.results(), self.calib.skipped
        self.calib = None
        self.blink.reset()
        if "failed" not in res and not self._apply_gaze(res):
            res = {**res, "failed": "its result couldn't be used"}
        if "failed" not in res:
            good = res["error_px"] <= ge.GOOD_ERROR_FRAC * self.screen.width
            save_settings(gaze=res, gaze_jump=good)
            self.state.update(gaze_jump=good, gaze_calibrated=res["date"],
                              gaze_error_px=int(res["error_px"]))
            log.info(f"[calibrate] Eyes done: about {res['error_px']} px ({res['model']['form']} fit, "
                     f"{res['points']} dots, {res['noise_px']} px frame-to-frame); gaze jump "
                     f"{'on' if good else 'off (too rough)'}")
        else:
            log.info(f"[calibrate] Eye calibration failed: {res.get('failed', 'unusable')}")
        self.state.update(calib_step="done", calib_progress=1.0,
                          calib_text=ge.calibration_summary(res, skipped, self.screen.width))
        self.calib_shown_until = now + self.CALIB_RESULT_S
        desktop.play_click_sound()

    def _gesture_action(self, name: str) -> None:
        """Carry out a face gesture (see GESTURES)."""
        desktop.play_warning_sound()
        if name == "smile":
            paused = self.state.toggle("paused")
            self.state.update(hint="Paused - smile for 1.5 s again to resume." if paused else "")
            log.info(f"[gesture] Smile: cursor control {'PAUSED' if paused else 'RESUMED'}")
        elif name == "mouth_left":
            self.state.recenter_request.set()
            log.info("[gesture] Mouth to the left: recenter")
        elif name == "mouth_right":
            log.info("[gesture] Mouth to the right: calibrate")
            request_calibration(self.state)

    def _restart_motion(self, start: np.ndarray | None = None) -> None:
        """Start cursor tracking afresh at ``start`` (default: where the cursor is)."""
        snap = self.state.read()
        start = np.array([snap.x, snap.y], np.float64) if start is None else start
        self.filter.reset()
        self.motion.reset()
        self.motion.pos = start.copy()
        self.head.reset(start)
        self.nose_history.clear()
        self.pos_history.clear()
        self.state.update(x=float(start[0]), y=float(start[1]))

    def _move_cursor(self, target_px: np.ndarray, confidence: float, snap: Snapshot,
                     now: float) -> None:
        """Smooth (One Euro), gate (CursorMotion) and publish a cursor target."""
        scale = np.array([self.screen.width, self.screen.height], np.float64)
        target = self.filter(target_px, now)
        pos = np.clip(self.motion(target, now), 0, scale - 1)
        self.motion.pos = pos
        sx, sy = float(pos[0]), float(pos[1])
        self.pos_history.append((now, sx, sy))
        self.state.update(x=sx, y=sy, pos_time=now, has_position=True, face=True,
                          eyes_closed=False, confidence=round(confidence, 3))
        if snap.click_mode == "dwell" and self.dwell.update(pos, self.motion.moving, now):
            log.info("[dwell] click")
            self.state.add_click()
            desktop.play_click_sound()       # dwell clicks always confirm audibly

    # -- camera view for the window ----------------------------------------
    def _update_preview(self, frame) -> None:
        """Hand an annotated, mirrored camera image to the window while it is shown."""
        snap = self.state.read()
        if not (snap.preview_on or snap.remote_preview):
            if self.preview_shown:
                self.state.set_preview(None)
                self.preview_shown = False
            return
        # ~15 images a second is plenty for the window
        now = time.monotonic()
        if now - self.last_preview < 1 / 15:
            return
        self.last_preview = now
        self.state.set_preview(self._render_preview(frame, snap, now))
        self.preview_shown = True

    def _render_preview(self, frame, snap: Snapshot, now: float) -> np.ndarray:
        import main_window
        view = frame.copy()
        if self.last_landmarks is not None:
            fe.draw_landmark_overlay(view, self.last_landmarks)
        if self.last_gaze is not None:
            # Each eye's gaze, as in JEOresearch's tracker: a ray out of the iris.
            for (x, y), d in zip(self.last_gaze.irises, self.last_gaze.dirs):
                cv2.circle(view, (int(x), int(y)), 3, (255, 200, 0), -1)
                cv2.line(view, (int(x), int(y)), (int(x + 150 * d[0]), int(y + 150 * d[1])),
                         (255, 200, 0), 2, cv2.LINE_AA)
        view = cv2.flip(view, 1)            # mirror view; features use the raw frame
        view = cv2.resize(view, (main_window.PREVIEW_W, main_window.PREVIEW_H))
        vh, vw = view.shape[:2]
        green, red, orange, grey = (0, 200, 0), (60, 60, 230), (0, 165, 255), (210, 210, 210)

        def banner(text, color, y):
            cv2.rectangle(view, (0, y - 22), (vw, y + 8), (0, 0, 0), -1)
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if self.last_landmarks is None:
            banner("Can't see your face", red, vh // 2)
        if self.gesture_since is not None and self.gesture_announced:
            left_s = max(0.0, self.args.close_gesture_hold - (now - self.gesture_since))
            banner(f"Closing EyeSync in {left_s:.1f}s - close mouth to cancel", red, vh // 2)
        held = self.blink.closed_for(now) if snap.eyes_closed else 0.0
        if snap.eyes_closed and held > self.blink.max_s:
            # The eye gesture toggles voice typing.
            if held < VOICE_EYES_CLOSED_S:
                text, color = ("keep closed: stop voice typing..." if snap.voice
                               else "keep closed: start voice typing..."), orange
                frac = (held - self.blink.max_s) / max(VOICE_EYES_CLOSED_S - self.blink.max_s, 1e-3)
            else:
                text, color, frac = ("Voice typing stopping - open your eyes" if snap.voice
                                     else "Voice typing starting - open your eyes"), green, 1.0
            banner(text, color, vh - 30)
            cv2.rectangle(view, (vw - 130, vh - 40), (vw - 130 + int(120 * frac), vh - 28),
                          color, -1)
        elif snap.eyes_closed:
            banner("Keep both eyes closed for voice typing", orange, vh - 30)
        if self.scroll.active:
            banner(f"Scrolling {'up' if self.scroll.direction > 0 else 'down'} - "
                   "straighten your head to stop", green, 40)
        elif self.scroll.tilted:
            banner("Head tilted: hold to scroll...", orange, 40)
        elif self.gestures.active is not None:
            active = self.gestures.active
            frac = self.gestures.progress(now)
            if active in GESTURE_HOLD_S:            # a long hold: count down
                left_s = max(0.0, self.gestures.hold_s() - (now - self.gestures.since))
                banner(f"Calibrating in {left_s:.1f}s - relax your mouth to cancel", red, 40)
            else:
                banner(f"{GESTURE_LABELS[active]} - hold...", orange, 40)
            cv2.rectangle(view, (10, 48), (10 + int((vw - 20) * frac), 56), orange, -1)
        elif snap.voice and snap.voice != "error":
            banner("Voice typing: " + ("say what to type" if snap.voice != "working"
                                       else "typing..."), green, 40)
        dwell = self.dwell.progress(now) if snap.click_mode == "dwell" else 0.0
        if dwell > 0:
            banner("hold still to click...", orange, vh - 30)
            cv2.rectangle(view, (vw - 130, vh - 40), (vw - 130 + int(120 * dwell), vh - 28),
                          orange, -1)
        cv2.putText(view, f"{self.fps:.0f} fps", (vw - 60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    grey, 1, cv2.LINE_AA)
        if now - self.jumped_at < 0.8:
            cv2.putText(view, "gaze jump", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1,
                        cv2.LINE_AA)
        return view

async def supervise_vision(args, screen, state) -> None:
    """
    Run the AI (camera + face mesh) only while it is needed - i.e. while the
    website is connected - and keep it healthy while it runs:
      * start it when the website connects; stop it (releasing the camera)
        once the website has been gone for --idle-grace seconds;
      * restart it (with back-off) if it crashes;
      * replace it if it stops delivering frames - some webcam drivers hang
        forever inside cap.read() when unplugged, and a hung thread cannot be
        killed, so we abandon it (it is a daemon) and start a fresh one.
    """
    backoff = 2.0
    worker: VisionWorker | None = None
    gone_since: float | None = None
    waiting_logged = False
    meshes = MeshHolder()
    meshes.expect_preload()                    # a worker starting first waits for it
    preload = asyncio.create_task(asyncio.to_thread(meshes.preload), name="mesh-preload")

    def start() -> VisionWorker:
        w = VisionWorker(args, screen, state, meshes)
        w.start()
        return w

    while not state.stop.is_set():
        now = time.monotonic()
        snap = state.read()
        # The camera runs for the website, and for a calibration the user started -
        # but not during the blink test, which needs it (Actions.blink_test).
        wanted = ((website_connected(snap, args) or calibrating(snap)
                   or state.calibrate_request.is_set()) and not snap.blink_test)
        gone_since = None if wanted else (gone_since or now)

        if worker is None:
            if wanted:
                log.info("[ai] Starting the camera"
                         + ("" if args.always_active else " (website connected)"))
                state.beat()
                worker = start()
                waiting_logged = False
            elif not waiting_logged and not snap.blink_test:
                log.info("[ai] Idle - camera off. Waiting for the EyeSync website dashboard "
                         "to connect...")
                waiting_logged = True
        elif snap.blink_test or now - (gone_since or now) >= args.idle_grace:
            log.info("[ai] The blink test needs the camera - stopping it here" if snap.blink_test
                     else "[ai] Not needed any more (website disconnected) - stopping the camera")
            worker.halt.set()
            await asyncio.to_thread(worker.join, 5.0)
            worker = None
            state.update(face=False, eyes_closed=False, confidence=0.0)
        elif not worker.is_alive():
            if now - worker.started_at > 60:
                backoff = 2.0          # it ran fine for a while; reset back-off
            log.warning(f"[vision] Worker stopped; restarting in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            worker = start() if not state.stop.is_set() else None
        elif (now - state.last_frame > args.stall_timeout
              and now - worker.started_at > args.stall_timeout):
            log.error(f"[vision] No frames for {args.stall_timeout:.0f}s; replacing worker")
            worker.halt.set()
            state.beat()
            worker = start()

        await asyncio.sleep(0.25)

    if worker is not None:              # quitting: don't keep the user waiting (daemon thread)
        worker.halt.set()
        await asyncio.to_thread(worker.join, 2.0)
    await asyncio.gather(preload, return_exceptions=True)
    meshes.close()


# --------------------------------------------------------------------------- #
# Fixed-rate timer
# --------------------------------------------------------------------------- #

OUTPUT_HZ = 120     # cursor updates per second (the camera gives ~30 positions)


class CursorGlide:
    """
    Smooth cursor output. The camera gives a new cursor position ~30 times a
    second; moving the cursor straight there made it jump 30-100 px at a
    time during a head turn - visibly jerky. Instead the cursor glides from
    where it is to each new position over exactly the time between the two
    camera frames, at OUTPUT_HZ, so a turn becomes one continuous movement
    (at most one frame, ~33 ms, behind the camera). At rest nothing changes.
    """

    MIN_S, MAX_S = 0.012, 0.05     # glide durations (a frame interval, clamped)

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget the motion: the next position is taken as is."""
        self.start = self.end = None
        self.stamp = self.t0 = 0.0
        self.duration = self.MAX_S

    def target(self, x: float, y: float, stamp: float, now: float) -> None:
        """A (possibly) new position from the camera frame taken at ``stamp``."""
        end = np.array([x, y], np.float64)
        if self.end is None:
            self.start = self.end = end
            self.stamp, self.t0 = stamp, now
            return
        if np.array_equal(end, self.end):
            self.stamp = max(self.stamp, stamp)   # resting: the next move glides one frame
            return
        interval = stamp - self.stamp if stamp > self.stamp else self.MIN_S
        self.duration = float(np.clip(interval, self.MIN_S, self.MAX_S))
        self.start, self.end = self.at(now), end
        self.stamp, self.t0 = stamp, now

    def at(self, now: float) -> np.ndarray:
        """Where the cursor should be now."""
        u = min(1.0, max(0.0, (now - self.t0) / self.duration))
        return self.start + (self.end - self.start) * u

    def finish(self) -> None:
        """Arrive at once (before a click, so it lands on the target)."""
        if self.end is not None:
            self.start, self.t0 = self.end, 0.0


class Ticker:
    """
    Fixed-rate loop timer. Sleeps until the next deadline rather than for a
    fixed interval, so the *average* rate stays at ``fps`` even though Windows
    rounds asyncio sleeps to ~15.6 ms (a plain sleep(1/30) gives ~25 Hz).
    """

    def __init__(self, fps: float):
        self.period = 1.0 / fps
        self.deadline = time.monotonic()

    async def wait(self) -> None:
        self.deadline += self.period
        delay = self.deadline - time.monotonic()
        if delay < -self.period:          # fell far behind: resync, don't burst
            self.deadline = time.monotonic()
            delay = 0.0
        await asyncio.sleep(max(0.0, delay))


# --------------------------------------------------------------------------- #
# A1. Hardware mouse: XIAO ESP32-S3 over USB serial
# --------------------------------------------------------------------------- #

def find_esp32_port() -> str | None:
    """Return the best serial port whose USB vendor ID looks like an ESP32 board."""
    ports = list(serial.tools.list_ports.comports())
    for vid in ESP32_USB_VIDS:
        for p in ports:
            if p.vid == vid:
                return p.device
    return None


def open_serial(port: str, baud: int) -> serial.Serial:
    """
    Open the port without resetting the board.

    Classic ESP32 dev boards wire DTR/RTS to EN/GPIO0 for auto-flashing, and
    pyserial asserts both on open by default, which can reboot the board or
    park it in the bootloader. De-asserting them before open() avoids that.
    (The XIAO's native USB ignores the lines for resetting, so it's harmless.)
    """
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0
    ser.write_timeout = 0.2   # never let a stuck device block us for long
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


async def serial_task(args, state: SharedState, screen: fe.ScreenSize) -> None:
    """
    Stream MOVE (only when the position changes, gliding between camera
    frames - see CursorGlide) and CLICK commands to the XIAO. A PING every
    2 s while idle detects unplugging even when the cursor isn't moving. Any
    error closes the port and the loop reconnects every 2 s; the rest of the
    pipeline is unaffected.
    """
    RETRY_S, PING_S = 2.0, 2.0
    ticker = Ticker(max(args.fps, OUTPUT_HZ))
    glide = CursorGlide()
    ser: serial.Serial | None = None
    last_pos: tuple[int, int] | None = None
    last_clicks = state.read().click_count
    last_rclicks = state.read().right_click_count
    next_retry = last_tx = 0.0
    warned_no_port = False
    rx_buf = b""
    told = (screen.width, screen.height)     # the screen size the firmware has

    def drop(reason: str) -> None:
        nonlocal ser
        log.warning(f"[serial] Disconnected ({reason}); retrying every {RETRY_S:.0f}s")
        try:
            ser.close()
        except Exception:
            pass
        ser = None
        state.update(esp32_connected=False)

    while not state.stop.is_set():
        now = time.monotonic()

        # --- (re)connect -------------------------------------------------
        if ser is None and now >= next_retry:
            next_retry = now + RETRY_S
            port = args.port or find_esp32_port()
            if port is None:
                if not warned_no_port:
                    fallback = (" - using software mouse meanwhile" if args.mouse == "auto" else "")
                    log.info(f"[serial] No XIAO ESP32-S3 found; will keep looking{fallback}")
                    warned_no_port = True
            else:
                try:
                    ser = await asyncio.to_thread(open_serial, port, args.baud)
                    log.info(f"[serial] Connected to {port} @ {args.baud} baud")
                    warned_no_port = False
                    last_pos = None
                    last_clicks = state.read().click_count   # don't replay old clicks
                    last_rclicks = state.read().right_click_count
                    # Tell the firmware the screen size for absolute positioning.
                    told = (screen.width, screen.height)
                    await asyncio.to_thread(ser.write, f"M,SCREEN,{told[0]},{told[1]}\n".encode())
                    last_tx = now
                    state.update(esp32_connected=True)
                except (serial.SerialException, OSError) as exc:
                    if ser is not None:
                        drop(str(exc))
                    else:
                        log.warning(f"[serial] Cannot open {port}: {exc}")

        # --- send --------------------------------------------------------
        if ser is not None:
            snap = state.read()
            out = []
            if (screen.width, screen.height) != told:       # the display changed resolution
                told = (screen.width, screen.height)
                out.append(f"M,SCREEN,{told[0]},{told[1]}\n")
                last_pos = None
            if not control_active(snap, args):
                last_clicks = snap.click_count   # swallow clicks while inactive
                last_rclicks = snap.right_click_count
                glide.reset()
            else:
                glide.target(snap.x, snap.y, snap.pos_time, now)
                clicking = snap.click_count != last_clicks or snap.right_click_count != last_rclicks
                if clicking:
                    glide.finish()
                x, y = glide.at(now)
                pos = (int(round(x)), int(round(y)))
                if snap.has_position and (snap.face or clicking) and pos != last_pos:
                    out.append(f"M,MOVE,{pos[0]},{pos[1]}\n")
                    last_pos = pos
                out.extend("M,CLICK,0,0\n" for _ in range(snap.click_count - last_clicks))
                out.extend("M,RCLICK,0,0\n" for _ in range(snap.right_click_count - last_rclicks))
                last_clicks, last_rclicks = snap.click_count, snap.right_click_count
            if not out and now - last_tx >= PING_S:
                out.append("M,PING,0,0\n")

            if out:
                try:
                    await asyncio.to_thread(ser.write, "".join(out).encode("ascii"))
                    last_tx = now
                except (serial.SerialException, OSError) as exc:
                    drop(str(exc))

            # Read (and log) whatever the firmware prints, so the OS buffer
            # never fills up and its messages are visible in eyesync.log.
            if ser is not None:
                try:
                    waiting = ser.in_waiting
                    if waiting:
                        rx_buf += ser.read(waiting)
                        *lines, rx_buf = rx_buf.split(b"\n")
                        for line in lines:
                            text = line.decode("ascii", "replace").strip()
                            if text and not text.endswith("PONG"):
                                log.info(f"[xiao] {text}")
                        rx_buf = rx_buf[-256:]
                except (serial.SerialException, OSError) as exc:
                    drop(str(exc))

        await ticker.wait()

    if ser is not None:
        ser.close()
        log.info("[serial] Closed")


# --------------------------------------------------------------------------- #
# A2. Standalone mouse: move the computer's own cursor
# --------------------------------------------------------------------------- #

async def software_mouse_task(args, state: SharedState) -> None:
    """
    Drive the OS cursor directly. In --mouse auto this is only active while no
    XIAO is connected, so the two never fight over the cursor.
    """
    mouse = SoftwareMouse()
    if not mouse.available:
        return
    log.info("[mouse] Software mouse ready")
    ticker = Ticker(max(args.fps, OUTPUT_HZ))
    glide = CursorGlide()
    last_pos: tuple[int, int] | None = None
    last_clicks = state.read().click_count
    last_rclicks = state.read().right_click_count
    was_active = False

    while not state.stop.is_set():
        snap = state.read()
        active = (args.mouse == "software"
                  or (args.mouse == "auto" and not snap.esp32_connected))
        active = active and control_active(snap, args) and snap.has_position

        if active != was_active:
            log.info(f"[mouse] Software cursor control {'ON' if active else 'OFF'}")
            was_active = active

        if not active:
            last_clicks = snap.click_count   # don't replay clicks on activation
            last_rclicks = snap.right_click_count
            last_pos = None
            glide.reset()
        else:
            try:
                now = time.monotonic()
                glide.target(snap.x, snap.y, snap.pos_time, now)
                clicking = snap.click_count != last_clicks or snap.right_click_count != last_rclicks
                if clicking:
                    glide.finish()
                x, y = glide.at(now)
                pos = (int(round(x)), int(round(y)))
                if (snap.face or clicking) and pos != last_pos:
                    mouse.move(*pos)
                    last_pos = pos
                for _ in range(snap.click_count - last_clicks):
                    mouse.click()
                for _ in range(snap.right_click_count - last_rclicks):
                    mouse.click(right=True)
                last_clicks, last_rclicks = snap.click_count, snap.right_click_count
            except Exception:
                log.exception("[mouse] Software mouse error")
        await ticker.wait()


async def scroll_task(args, state: SharedState, mouse: SoftwareMouse | None = None) -> None:
    """
    The mouse wheel (TiltScroll), as Windows' own wheel input - also with the
    XIAO in use: its firmware has no wheel, and turning the wheel needs no
    cursor position, so the two never fight. Whole notches only: some
    programs ignore smaller wheel steps.
    """
    mouse = mouse or SoftwareMouse()
    if not mouse.available:
        return
    last = state.read().scroll_units
    while not state.stop.is_set():
        snap = state.read()
        if snap.scroll_units != last:
            if control_active(snap, args):
                try:
                    mouse.scroll(snap.scroll_units - last)
                except Exception:
                    log.exception("[mouse] Scrolling failed")
            last = snap.scroll_units
        await asyncio.sleep(1 / 60)


# --------------------------------------------------------------------------- #
# B. WebSocket output -> web dashboard
# --------------------------------------------------------------------------- #

def parse_size(text: str) -> tuple[int, int] | None:
    if text.lower() == "native":
        return None
    w, h = text.lower().split("x")
    return int(w), int(h)


def status_message(state: SharedState, args) -> dict:
    """What the dashboard shows about EyeSync (sent after every command)."""
    snap = state.read()
    return {"type": "status", "version": VERSION,
            "edition": "standalone" if app_paths.STANDALONE else "original",
            "mouse_enabled": not snap.remote_off, "paused": snap.paused,
            "active": control_active(snap, args), "face": snap.face,
            "camera": snap.has_position and time.monotonic() - state.last_frame < 3.0,
            "speed": snap.sensitivity, "click_mode": snap.click_mode,
            "gestures": snap.gestures, "calibrating": calibrating(snap),
            "calibrated": snap.calibrated,
            "calibration": {"step": snap.calib_step, "text": snap.calib_text,
                            "progress": snap.calib_progress},
            "hint": snap.hint, "hardware_mouse": snap.esp32_connected,
            "gaze": None,  # the separate gaze-jump feature is replaced by model control
            # What is said is never sent here - only whether voice typing listens.
            "voice": {"on": snap.voice not in ("", "error"), "state": snap.voice or "off",
                      "hint": snap.voice_hint}}


STATUS_PUSH_S = 0.25   # how often status changes are pushed to pages that send commands
PREVIEW_FPS = 8        # camera view frames a second for the website (~15 KB each)


def _preview_request(text: str) -> bool | None:
    """{"action": "preview", "value": true|false} -> the value; anything else -> None."""
    try:
        msg = json.loads(text)
    except ValueError:
        return None
    if isinstance(msg, dict) and msg.get("action") == "preview" and isinstance(msg.get("value"), bool):
        return msg["value"]
    return None


def handle_command(text: str, state: SharedState, actions: "Actions", args) -> dict | None:
    """
    One message from the dashboard -> carry it out and return the reply
    (the status, or an error), or None for a message that needs no reply
    (the website's calibration progress). Commands are documented at the top
    of this file; only the website and apps with the API key can send them.
    """
    try:
        msg = json.loads(text)
    except ValueError:
        return {"type": "error", "error": "messages must be JSON"}
    if isinstance(msg, dict) and msg.get("type") == "calibration":
        # The website's own calibration measures how well the cursor reaches
        # its 9 targets; head control needs nothing from it, so just log it.
        event = msg.get("event")
        if event == "start":
            log.info("[website] Calibration started on the website")
        elif event == "complete":
            log.info(f"[website] Calibration complete: mean error "
                     f"{msg.get('mean_error_px')} px on the website")
        elif event == "cancel":
            log.info("[website] Calibration cancelled on the website")
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("action"), str):
        return {"type": "error", "error": 'expected {"action": ...}'}
    action, value = msg["action"], msg.get("value")
    if action == "mouse" and isinstance(value, bool):
        actions.set_remote_enabled(value)
    elif action == "recenter":
        actions.recenter()
    elif action == "calibrate":
        if value == "cancel":
            actions.cancel_calibration()
        elif value != "gaze" and not calibrating(state.read()):
            actions.calibrate()
    elif action == "gaze" and isinstance(value, bool):
        actions.set_gaze_jump(value)
    elif action == "speed" and isinstance(value, int) and value in SENSITIVITY_LEVELS:
        actions.set_sensitivity(value)
    elif action == "click_mode" and value in CLICK_MODES:
        actions.set_click_mode(value)
    elif action == "gestures" and isinstance(value, bool):
        actions.set_gestures(value)
    elif action == "pause" and isinstance(value, bool):
        if state.read().paused != value:
            actions.pause()
    elif action == "voice":
        return {"type": "error", "error": "Close both eyes for 1 second to toggle voice typing; the voice bar's Stop button also works."}
    elif action == "show_window":
        actions.show_window()
    elif action == "quit":
        actions.quit()              # the connection then closes with code 4000
    elif action != "status":
        return {"type": "error", "error": f"unknown command {msg!r}"[:200]}
    return status_message(state, args)


async def websocket_task(args, state: SharedState, screen: fe.ScreenSize,
                         api_key: ApiKey | None = None,
                         actions: "Actions | None" = None) -> None:
    """
    Serve every port in --ws-ports (any URL path is accepted, so both
    ws://localhost:8080/eye-tracker and ws://localhost:8765 work) and push the
    payload to all connected dashboards at ``args.fps``.

    Who may connect: the EyeSync website (by its Origin) and anyone who
    adds ?key=<API key> to the URL (see ApiKey). Both switch EyeSync on.
    Everyone else is closed with code 4001 and a reason.

    ``broadcast`` writes to each client without awaiting it, so one slow or
    dead browser tab can't slow the others down. If a port is already in use
    we keep retrying it instead of crashing, and the other ports still work.
    The website itself reconnects every 3 s on its own, so as long as this
    process (or a restarted one) is listening, it reconnects automatically.
    """
    clients: set = set()
    ref = parse_size(args.dashboard_size)
    last_clicks = state.read().click_count + state.read().right_click_count

    def payload(snap: Snapshot, blink: bool) -> str:
        # Per message: the screen may change resolution (VisionWorker._follow_screen).
        sx = ref[0] / screen.width if ref else 1.0
        sy = ref[1] / screen.height if ref else 1.0
        x, y = snap.x * sx, snap.y * sy
        if not (math.isfinite(x) and math.isfinite(y)):   # never send NaN to JSON.parse
            x, y = screen.width * sx / 2, screen.height * sy / 2
        return json.dumps({
            "x": round(x, 2),
            "y": round(y, 2),
            "blink": bool(blink),
            "confidence": round(snap.confidence if snap.face else 0.0, 3),
            "face": bool(snap.face),
        })

    site_clients: set = set()
    controllers: set = set()        # pages that sent commands: they get status changes
    preview_clients: set = set()    # pages that asked for the camera view
    last_site_seen = -math.inf      # see RECONNECT_GRACE_S
    key_clients: dict = {}          # ws -> ApiKey.version it connected with
    key_version = api_key.version if api_key else 0
    last_refusal_log = -math.inf

    def publish_counts() -> None:
        nonlocal last_site_seen
        now = time.monotonic()
        if site_clients:
            last_site_seen = now
        # A website that just left (reload, another page of the site) still
        # counts for RECONNECT_GRACE_S, so control doesn't drop meanwhile.
        shown = len(site_clients) or int(now - last_site_seen < RECONNECT_GRACE_S)
        before = state.read().site_clients
        state.update(ws_clients=len(clients), site_clients=shown)
        if bool(before) != bool(shown) and not args.always_active:
            if shown:
                log.info("[main] Website or app connected - cursor control ACTIVE")
            else:
                log.info("[main] Website or app disconnected - cursor control INACTIVE")

    async def handler(ws):
        nonlocal last_refusal_log
        from urllib.parse import parse_qs, urlsplit

        url = urlsplit(getattr(ws.request, "path", "/"))
        path = url.path or "/"                   # without the query: never log the key
        given = parse_qs(url.query).get("key", [None])[0]
        origin = ws.request.headers.get("Origin", "") or ""
        by_key = given is not None and api_key is not None and api_key.matches(given)
        is_site = is_website_origin(origin, args.site_origin)
        if not (by_key or is_site):
            reason = "API key required" if given is None else "invalid API key"
            if time.monotonic() - last_refusal_log > 60:      # a page retrying can't flood the log
                last_refusal_log = time.monotonic()
                log.warning(f"[ws] Refused a connection on {path} (origin {origin or 'none'}): "
                            f"{reason}")
            await ws.close(4001, reason)
            return
        who = "Website" if is_site else "App (API key)"
        if parse_qs(url.query).get("probe") == ["1"]:
            # "Is EyeSync installed?" (WEBSITE.md): yes, and running - without
            # connecting (a probe never switches cursor control on).
            await ws.send(json.dumps({**status_message(state, args), "type": "eyesync",
                                      "state": "running", "closed_by_user": False}))
            await ws.close(4003, "EyeSync is running - connect without ?probe=1")
            return
        clients.add(ws)
        site_clients.add(ws)
        if by_key:
            key_clients[ws] = api_key.version
        publish_counts()
        log.info(f"[ws] {who} connected on {path} (origin {origin or 'none'}) - "
                 f"{len(clients)} total")
        try:
            # Send a message immediately so the dashboard shows data at once.
            await ws.send(payload(state.read(), False))
            async for message in ws:         # the dashboard's buttons (handle_command)
                if actions is None or not isinstance(message, str):
                    continue
                preview = _preview_request(message)
                if preview is not None:      # the camera view, for this page only
                    (preview_clients.add if preview else preview_clients.discard)(ws)
                    state.update(remote_preview=bool(preview_clients))
                    log.info(f"[ws] Camera view on the website {'ON' if preview else 'OFF'}")
                    reply = status_message(state, args)
                else:
                    reply = handle_command(message, state, actions, args)
                if reply is not None:
                    controllers.add(ws)      # from now on it also gets status changes
                    await ws.send(json.dumps(reply))
        except ConnectionClosed:
            pass
        finally:
            clients.discard(ws)
            site_clients.discard(ws)
            key_clients.pop(ws, None)
            controllers.discard(ws)
            if ws in preview_clients:
                preview_clients.discard(ws)
                state.update(remote_preview=bool(preview_clients))
            if not site_clients and state.read().remote_off:
                # The next dashboard says for itself whether mouse control is on.
                state.update(remote_off=False)
            publish_counts()
            log.info(f"[ws] {who} disconnected ({len(clients)} total)")

    # "localhost" may resolve to only ::1 or only 127.0.0.1 depending on the
    # OS, while browsers may try either. Bind both loopbacks explicitly so the
    # website connects every time; fall back to IPv4 alone if IPv6 is off.
    if args.ws_host == "localhost":
        host_options = [["127.0.0.1", "::1"], ["127.0.0.1"]]
    else:
        host_options = [[args.ws_host]]

    async def bind(port: int):
        last_exc: OSError | None = None
        for hosts in host_options:
            try:
                return await serve(handler, hosts, port, ping_interval=20, ping_timeout=20,
                                   max_size=4096)     # commands are tiny
            except OSError as exc:
                last_exc = exc
        raise last_exc

    async def serve_port(port: int):
        """Bind one port, retrying every 5 s while it is busy."""
        warned = False
        while not state.stop.is_set():
            try:
                server = await bind(port)
            except OSError as exc:
                if not warned:
                    log.error(f"[ws] Port {port} unavailable ({exc}); retrying every 5s. "
                              "Is another program (or another EyeSync) using it?")
                    warned = True
                await asyncio.sleep(5)
                continue
            log.info(f"[ws] Listening on ws://{args.ws_host}:{port}")
            try:
                while not state.stop.is_set():
                    await asyncio.sleep(0.2)
            finally:
                server.close()
                await server.wait_closed()
            return

    servers = [asyncio.create_task(serve_port(p)) for p in args.ws_ports]
    ticker = Ticker(args.fps)
    last_status, next_status, next_preview, last_frame_id = "", 0.0, 0.0, -1
    try:
        while not state.stop.is_set():
            snap = state.read()
            now = time.monotonic()
            if controllers and now >= next_status:
                # Changes made elsewhere (a gesture paused EyeSync, a hotkey
                # changed the speed, calibration progress) show on the page too.
                next_status = now + STATUS_PUSH_S
                status = json.dumps(status_message(state, args))
                if status != last_status:
                    last_status = status
                    broadcast(controllers, status)
            if preview_clients and now >= next_preview:
                next_preview = now + 1 / PREVIEW_FPS
                frame_id, frame = state.get_preview()
                if frame is not None and frame_id != last_frame_id:
                    last_frame_id = frame_id
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        broadcast(preview_clients, json.dumps({
                            "type": "preview",
                            "image": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}))
            # "blink" is true while the eyes are closed, and for at least one
            # message after every click so a short blink is never missed.
            clicks = snap.click_count + snap.right_click_count
            blink = snap.eyes_closed or clicks != last_clicks
            last_clicks = clicks
            publish_counts()                           # ends a reconnect grace on time
            if api_key is not None and api_key.version != key_version:
                key_version = api_key.version          # "New key": drop the old key's users
                for ws, version in list(key_clients.items()):
                    if version != key_version:
                        key_clients.pop(ws, None)
                        asyncio.create_task(ws.close(4001, "API key changed"))
            if clients:
                try:
                    broadcast(clients, payload(snap, blink))
                except Exception:
                    log.exception("[ws] Broadcast error")
            await ticker.wait()
    finally:
        # Tell the dashboard why (it shows the close code): after a quit only
        # its Connect button starts EyeSync again (site_access.wait_for_visit).
        closing = [ws.close(4000, "EyeSync was closed - press Connect to start it again")
                   for ws in list(clients)]
        if closing:
            await asyncio.wait_for(asyncio.gather(*closing, return_exceptions=True), 2.0)
        await asyncio.gather(*servers, return_exceptions=True)
        log.info("[ws] Servers stopped")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def run(args, on_ready=None) -> int:
    """
    The engine: WebSocket servers, mouse output, hotkeys and the vision
    supervisor. ``on_ready(state, actions, status)`` is called once
    everything is set up (the window uses it to bind its controls).
    """
    screen = fe.get_screen_size()
    log.info(f"[screen] {screen.width}x{screen.height}")

    settings = load_settings()
    if settings.get("version", 1) < SETTINGS_VERSION:
        # Improved defaults (slower, steadier pointer...) replace settings
        # saved by an older EyeSync, so every update takes effect by itself.
        log.info("[settings] New EyeSync defaults applied")
        # ... except the user's own calibrations, which are about them, not defaults.
        settings = {"version": SETTINGS_VERSION,
                    **{k: settings[k] for k in ("calibration", "gaze") if k in settings}}
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning(f"[settings] Could not save {SETTINGS_PATH}: {exc}")
    sensitivity = args.sensitivity or settings.get("sensitivity", DEFAULT_SENSITIVITY)
    if sensitivity not in SENSITIVITY_LEVELS:
        sensitivity = DEFAULT_SENSITIVITY
    click_mode = "blink"  # the only click gestures are smile and mouth-open
    state = SharedState(screen, sensitivity, preview_on=False, click_mode=click_mode,
                        gestures=bool(settings.get("gestures", True)))
    state.update(calibrated="Saved desktop gaze model")
    calibration = settings.get("calibration")
    if isinstance(calibration, dict):
        state.update(calibrated=str(calibration.get("date", "yes")))
        log.info(f"[main] Using your calibration from {calibration.get('date', '?')}")
    state.update(gaze_jump=bool(settings.get("gaze_jump", True)))
    gaze = settings.get("gaze")
    if isinstance(gaze, dict) and "model" in gaze:
        state.update(gaze_calibrated=str(gaze.get("date", "yes")),
                     gaze_error_px=int(gaze.get("error_px", 0)))
        log.info(f"[main] Eyes calibrated on {gaze.get('date', '?')} (about "
                 f"{gaze.get('error_px', '?')} px): gaze jump "
                 f"{'on' if state.read().gaze_jump else 'off'}")

    desktop.keep_running_in_background()
    api_key = ApiKey(API_KEY_PATH)
    actions = Actions(state, api_key)
    hotkeys = desktop.GlobalHotkeys(
        {} if app_paths.STANDALONE else make_hotkey_actions(actions))
    hotkeys.start()
    if hotkeys.registered:
        log.info("[hotkeys] Active: " + ", ".join(desktop.HOTKEY_PREFIX + k
                                                   for k in hotkeys.registered))
    log.info(f"[main] Sensitivity {sensitivity}/5 (Ctrl+Alt+Shift+F7 calmer, F8 faster)")
    voice = voice_typing.VoiceTyping(state, sounds=False).start()
    panel_script = app_paths.APP_DIR / "control_backend" / "wispr_command_panel.py"
    panel_process = None
    if not args.no_gui and (app_paths.FROZEN or panel_script.exists()):
        if app_paths.FROZEN:
            command = [sys.executable, "--command-panel"]
        else:
            pythonw = Path(sys.executable).with_name("pythonw.exe")
            command = [str(pythonw if pythonw.exists() else sys.executable),
                       "-m", "control_backend.wispr_command_panel"]
        panel_process = subprocess.Popen(
            [*command, "--parent-pid", str(os.getpid())],
            cwd=str(app_paths.DATA_DIR))
    problem = voice_typing.available()
    log.info(f"[main] Voice typing: close both eyes for {VOICE_EYES_CLOSED_S:g}s "
             "to toggle with a beep; the bar's Stop button also works" +
             (f" - UNAVAILABLE: {problem}" if problem else ""))
    tasks = [
        asyncio.create_task(websocket_task(args, state, screen, api_key, actions),
                            name="websocket"),
    ]
    if args.mouse in ("auto", "esp32"):
        tasks.append(asyncio.create_task(serial_task(args, state, screen), name="serial"))
    if args.mouse in ("auto", "software"):
        tasks.append(asyncio.create_task(software_mouse_task(args, state), name="software-mouse"))
    if args.mouse != "none":
        tasks.append(asyncio.create_task(scroll_task(args, state), name="scroll"))

    vision = asyncio.create_task(supervise_vision(args, screen, state), name="vision")
    if on_ready is not None:
        on_ready(state, actions, lambda: status_of(state.read(), args))

    try:
        # Idle until something (F12 hotkey, Ctrl+C) requests a stop. Any output
        # task that dies unexpectedly is logged and restarted by design of its
        # own retry loop; here we just watch for crashes worth reporting.
        site_seen, site_gone = False, None
        code = EXIT_QUIT
        while not state.stop.is_set():
            await asyncio.sleep(0.25)
            snap = state.read()
            if snap.site_clients or calibrating(snap):
                site_seen, site_gone = site_seen or bool(snap.site_clients), None
            elif site_seen and not (args.always_active or args.stay_open):
                # Open only while the dashboard is: close, and EyeSync.pyw
                # waits for the next visit.
                site_gone = site_gone or time.monotonic()
                if time.monotonic() - site_gone > CLOSE_AFTER_SITE_S:
                    log.info("[main] The website dashboard was closed - EyeSync closes and "
                             "opens again with the dashboard")
                    code = EXIT_DASHBOARD_CLOSED
                    state.stop.set()
            for t in tasks + [vision]:
                if t.done() and not t.cancelled() and t.exception() is not None:
                    log.error(f"[main] Task {t.get_name()} crashed: {t.exception()!r}")
                    code = EXIT_CRASHED              # the launcher restarts EyeSync
                    state.stop.set()
    finally:
        state.stop.set()
        hotkeys.stop()
        await asyncio.gather(vision, *tasks, asyncio.to_thread(voice.close),
                             return_exceptions=True)
    return code


def request_calibration(state: SharedState, kind: str = "head") -> None:
    """
    Ask the vision worker to start a calibration (the button, F10, the
    website, a tilt): ``kind`` "head" (Calibration) or "gaze" (the eyes,
    gaze_engine.GazeCalibration).
    """
    state.calibrate_cancel.clear()
    state.calibrate_kind = kind
    state.update(calib_step="starting", calib_text="Starting the camera...", calib_progress=0.0)
    state.calibrate_request.set()
    log.info("[control] " + ("Eye calibration" if kind == "gaze" else "Calibration") + " requested")


class Actions:
    """
    Everything the user can ask EyeSync to do. Shared by the global hotkeys
    and the window; each runs on the caller's thread, so keep them quick.
    """

    def __init__(self, state: SharedState, api_key: ApiKey | None = None):
        self.state = state
        self.api_key = api_key

    def set_sensitivity(self, level: int) -> None:
        level = int(np.clip(level, 1, 5))
        self.state.update(sensitivity=level)
        save_settings(sensitivity=level)
        log.info(f"[control] Sensitivity {level}/5")

    def calmer(self) -> None:
        self.set_sensitivity(self.state.read().sensitivity - 1)

    def faster(self) -> None:
        self.set_sensitivity(self.state.read().sensitivity + 1)

    def set_click_mode(self, mode: str) -> None:
        if mode != "blink":
            return
        self.state.update(click_mode=mode)
        save_settings(click_mode=mode)
        log.info(f"[control] Click by {'blinking' if mode == 'blink' else 'holding still (dwell)'}")

    def set_gestures(self, on: bool) -> None:
        self.state.update(gestures=bool(on))
        save_settings(gestures=bool(on))
        log.info(f"[control] Face gestures {'on' if on else 'off'}")

    def set_remote_enabled(self, on: bool) -> None:
        """The website's Enable / Disable Mouse Control button."""
        if self.state.read().remote_off == (not on):
            return
        self.state.update(remote_off=not on)
        log.info(f"[website] Mouse control {'ENABLED' if on else 'DISABLED'} on the website")

    def recenter(self) -> None:
        self.state.recenter_request.set()
        log.info("[control] Recenter requested")

    def calibrate(self) -> None:
        """Release the camera, train the active pointer model, then reopen EyeSync."""
        if app_paths.FROZEN:
            command = [sys.executable, "--calibrate-portable"]
        else:
            pythonw = Path(sys.executable).with_name("pythonw.exe")
            command = [str(pythonw if pythonw.exists() else sys.executable),
                       str(app_paths.APP_DIR / "control_backend" / "calibrate.py")]
        command += ["--wait-pid", str(os.getpid()), "--restart"]
        try:
            subprocess.Popen(command, cwd=str(app_paths.DATA_DIR))
        except OSError:
            log.exception("[control] Could not open pointer calibration")
            self.state.update(hint="Could not open calibration; see EyeSync log.")
            return
        log.info("[control] Starting pointer calibration")
        self.quit()

    def set_gaze_jump(self, on: bool) -> None:
        self.state.update(gaze_jump=bool(on))
        save_settings(gaze_jump=bool(on))
        log.info(f"[control] Gaze jump {'on' if on else 'off'}")

    def cancel_calibration(self) -> None:
        if not calibrating(self.state.read()):
            return
        self.state.calibrate_request.clear()
        self.state.calibrate_cancel.set()
        self.state.update(calib_step="", calib_text="", calib_progress=0.0)
        log.info("[control] Calibration cancelled - keeping the previous settings")

    def pause(self) -> None:
        paused = self.state.toggle("paused")
        log.info(f"[control] Cursor control {'PAUSED' if paused else 'RESUMED'}")

    def voice(self, on: bool | None = None, source: str = "button") -> None:
        """Voice typing on (True), off (False) or the other way round (None)."""
        voice_typing.request(self.state, on, source)

    def new_api_key(self) -> None:
        if self.api_key is not None:
            self.api_key.regenerate()

    def show_window(self) -> None:
        desktop.bring_window_to_front(
            "EyeSync", "TkTopLevel")

    def quit(self) -> None:
        log.info("[control] Quit requested")
        self.state.stop.set()

    BLINK_TEST_HINT = "Blink test running - EyeSync goes on when it ends."

    def blink_test(self, run=subprocess.run) -> threading.Thread | None:
        """
        The blink test (tools/record_blinks.py) in a window of its own: a few
        minutes of guided blinks and gestures, saved as numbers only (no
        pictures) for whoever tunes EyeSync to a new face. It needs the
        camera, so EyeSync's stops meanwhile (supervise_vision) and starts
        again when the test ends. ``run``: for the tests.
        """
        if self.state.read().blink_test:
            return None
        self.state.update(blink_test=True, hint=self.BLINK_TEST_HINT)

        def go():
            cmd = app_paths.blink_test_command()
            log.info("[test] Blink test started")
            try:
                code = run(cmd).returncode
                log.info(f"[test] Blink test ended (exit code {code})")
            except Exception:
                log.exception("[test] The blink test couldn't run")
            finally:
                if self.state.read().hint == self.BLINK_TEST_HINT:
                    self.state.update(hint="")
                self.state.update(blink_test=False)

        thread = threading.Thread(target=go, name="blink-test", daemon=True)
        thread.start()
        return thread


def make_hotkey_actions(actions: Actions) -> dict:
    """Ctrl+Alt+Shift+<key> -> action."""
    return {"F6": actions.recenter, "F7": actions.calmer, "F8": actions.faster,
            "F9": actions.pause, "F10": actions.calibrate, "F11": actions.show_window,
            "F12": actions.quit}


def parse_args():
    p = argparse.ArgumentParser(description="EyeSync live inference + mouse/WebSocket output")
    p.add_argument("--camera", type=int, default=None,
                   help="webcam index (default: the first working one of 0, 1, 2)")

    g = p.add_argument_group("mouse output")
    g.add_argument("--mouse", choices=("auto", "esp32", "software", "none"), default="auto",
                   help="auto = XIAO when connected, else the computer's own cursor (default)")
    g.add_argument("--port", help="XIAO serial port, e.g. COM5 (default: auto-detect)")
    g.add_argument("--baud", type=int, default=115200)

    g = p.add_argument_group("web dashboard")
    g.add_argument("--ws-host", default="localhost",
                   help="'localhost' binds both 127.0.0.1 and ::1 (default)")
    g.add_argument("--ws-ports", type=int, nargs="+", default=list(site_access.DEFAULT_PORTS),
                   help="WebSocket ports (default: 8080 for the website, 8765)")
    g.add_argument("--dashboard-size", default="800x600",
                   help="coordinate space for WebSocket x/y: WxH or 'native' (default 800x600, "
                        "the website's default 'Tracker frame' size)")
    g.add_argument("--fps", type=float, default=30.0, help="output rate (default 30)")
    g.add_argument("--site-origin", action="append", default=None,
                   help="website origin allowed to connect without the API key; repeatable "
                        "(default: the EyeSync Wix site)")
    g.add_argument("--always-active", action="store_true",
                   help="run the AI and control the cursor even without the website")
    g.add_argument("--stay-open", action="store_true",
                   help="keep EyeSync open after the website dashboard closes (default: it "
                        f"closes {CLOSE_AFTER_SITE_S:.0f} s later and opens again with it)")
    g.add_argument("--idle-grace", type=float, default=5.0,
                   help="seconds after the website disconnects before the camera "
                        "stops (default 5)")

    g = p.add_argument_group("cursor feel")
    g.add_argument("--sensitivity", type=int, choices=sorted(SENSITIVITY_LEVELS),
                   help="1 = calmest ... 5 = most responsive (default: last used, else "
                        f"{DEFAULT_SENSITIVITY}); change live with Ctrl+Alt+Shift+F7/F8")

    g = p.add_argument_group("blink click")
    g.add_argument("--ear-threshold", type=float, default=None,
                   help="fixed EAR close threshold (e.g. 0.18); default is adaptive")
    g.add_argument("--close-ratio", type=float, default=0.55,
                   help="eyes count as closed below this x open-eye EAR (default 0.55)")
    g.add_argument("--open-ratio", type=float, default=0.70,
                   help="...and open again above this x open-eye EAR (default 0.70)")
    g.add_argument("--min-blink", type=float, default=BLINK_MIN_S,
                   help=f"shortest closure (s) that counts as a click (default {BLINK_MIN_S})")
    g.add_argument("--max-blink", type=float, default=BLINK_MAX_S,
                   help=f"longest closure (s) that counts as a click (default {BLINK_MAX_S}; "
                        f"keep it below {VOICE_EYES_CLOSED_S} s, voice typing)")
    g.add_argument("--right-blink", type=float, default=BLINK_RIGHT_S,
                   help="both eyes closed this long (up to --max-blink) = right click; a double "
                        f"tick says when (default {BLINK_RIGHT_S}; 0 = off, right-eye winks only)")
    g.add_argument("--click-cooldown", type=float, default=0.6,
                   help="minimum seconds between clicks (default 0.6)")
    g.add_argument("--click", choices=CLICK_MODES, default=None,
                   help="how to click: blink (default) or dwell = hold the cursor still "
                        "(the choice is saved; also in the window)")
    g.add_argument("--dwell-time", type=float, default=1.0,
                   help="seconds the cursor must rest for a dwell click (default 1.0)")
    g.add_argument("--no-click-sound", action="store_true",
                   help="don't play the short system sound on every blink-click")

    g = p.add_argument_group("close gesture")
    g.add_argument("--no-close-gesture", action="store_true",
                   help="disable closing EyeSync by holding the mouth wide open")
    g.add_argument("--close-gesture-mar", type=float, default=0.6,
                   help="mouth opening ratio that counts as 'wide open' (default 0.6)")
    g.add_argument("--close-gesture-hold", type=float, default=3.0,
                   help="seconds to hold the mouth open to close EyeSync (default 3)")

    g = p.add_argument_group("misc")
    g.add_argument("--health-interval", type=float, default=300.0,
                   help="seconds between [health] log lines (fps, tracking rate)")
    g.add_argument("--stall-timeout", type=float, default=10.0,
                   help="restart the camera pipeline after this many seconds without frames")
    g.add_argument("--no-gui", action="store_true",
                   help="run without the window (for tests and servers)")
    g.add_argument("--minimized", action="store_true",
                   help="start with the window minimised (used when starting with Windows)")
    return p.parse_args()


_crash_log = None          # kept open for faulthandler


def main() -> int:
    global _crash_log
    fe.setup_logging(LOG_PATH)
    # A crash inside a native library (camera driver, MediaPipe) ends the
    # process without a Python error; faulthandler still writes where it was.
    import faulthandler
    try:
        # Its own file: holding eyesync.log open would stop its rotation.
        _crash_log = open(LOG_PATH.with_name("eyesync_crash.log"), "a", encoding="utf-8")
        faulthandler.enable(_crash_log)
    except OSError:
        pass
    args = parse_args()
    try:
        parse_size(args.dashboard_size)
    except ValueError:
        log.error("--dashboard-size must look like 1920x1080 or 'native'")
        return 2
    if not args.site_origin:
        args.site_origin = list(DEFAULT_SITE_ORIGINS)
    instance_name = ("EyeSync.Standalone.LiveInference" if app_paths.STANDALONE
                     else "EyeSync.LiveInference")
    if not desktop.acquire_single_instance(instance_name):
        # Already running (e.g. started with Windows): show its window
        # instead, and exit cleanly (code 0) so launchers don't retry.
        log.info("[main] EyeSync is already running - bringing its window to the front")
        desktop.bring_window_to_front(
            "EyeSync", "TkTopLevel")
        return 0
    log.info(f"=== EyeSync {VERSION} live session starting ===")
    if not args.always_active:
        log.info("[main] The AI starts when the EyeSync website dashboard connects")
    try:
        if args.no_gui:
            return asyncio.run(run(args))
        import main_window
        args.log_path = LOG_PATH
        return main_window.run_with_window(args, run)
    except KeyboardInterrupt:
        log.info("[main] Stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
