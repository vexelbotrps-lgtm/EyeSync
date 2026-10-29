"""
face_engine.py - Shared vision core for EyeSync.
=================================================

What lives here:
  * open_camera()        - robust cv2.VideoCapture initialisation.
  * LowLight             - keeps face tracking working in a dark room.
  * FaceMeshBackend      - wraps MediaPipe's 468/478-point face mesh.
  * nose_pointer()       - the nose tip and face size the head pointer follows.
  * eye_aspect_ratios()  - EAR blink metric from the eyelid landmarks.
  * get_screen_size()    - physical screen resolution (DPI-aware on Windows).

About the MediaPipe API
-----------------------
The classic API is ``mediapipe.solutions.face_mesh.FaceMesh``. Newer MediaPipe
releases (0.10.3x and the 1.x line) removed ``mp.solutions`` entirely and only
ship the "Tasks" API (``FaceLandmarker``). FaceLandmarker runs the same face
mesh model and returns the same landmark indices, so the backend below uses
``solutions.face_mesh.FaceMesh`` when it exists and falls back to
FaceLandmarker otherwise. Everything downstream is identical either way.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
import threading
import time
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path


def _defer_matplotlib() -> None:
    """
    MediaPipe's package imports its drawing helpers, and they import
    matplotlib.pyplot: ~0.6 s of every start, for plotting EyeSync never
    does (it draws its own overlay). Stand-ins load the real matplotlib the
    moment anything does use it. (1.5.12; EyeSync.exe leaves it out.)
    """
    if "matplotlib" in sys.modules:
        return

    class Deferred(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            for name in ("matplotlib.pyplot", "matplotlib"):
                if isinstance(sys.modules.get(name), Deferred):
                    del sys.modules[name]
            return getattr(importlib.import_module(self.__name__), attr)

    parent, pyplot = Deferred("matplotlib"), Deferred("matplotlib.pyplot")
    parent.pyplot = pyplot                  # "import matplotlib.pyplot as plt" reads this
    sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = parent, pyplot


os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # see open_camera
import cv2
import numpy as np
_defer_matplotlib()
import mediapipe as mp

import app_paths

log = logging.getLogger("eyesync")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_DIR = app_paths.DATA_DIR   # the user's files: settings, log (see app_paths)

# The Tasks API needs a .task model file. It is downloaded once on first use
# (EyeSync.exe and the zip package ship it).
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = app_paths.APP_DIR / "models" / "face_landmarker.task"

# Eye landmark indices in the MediaPipe face mesh, ordered for the EAR formula
# (p1..p6 in Soukupova & Cech, 2016):
#
#            p2   p3
#     p1 ----------------- p4        p1/p4 = eye corners (horizontal)
#            p6   p5                 p2/p6, p3/p5 = upper/lower eyelid pairs
#
# "Left"/"right" are from the subject's point of view.
LEFT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)

# Eyelid polylines and iris rings used by the dataset/closure diagnostics.
# The order follows the upper/lower lid from one corner to the other. Iris
# points are predictions too, so these measurements are features for a
# conservative classifier, never proof by themselves that an eye is open.
EYE_IRIS_IDX = {
    "left": {
        "corners": (362, 263),
        "upper": (362, 398, 384, 385, 386, 387, 388, 466, 263),
        "lower": (362, 382, 381, 380, 374, 373, 390, 249, 263),
        "centre": 473,
        "ring": (474, 475, 476, 477),
    },
    "right": {
        "corners": (33, 133),
        "upper": (33, 246, 161, 160, 159, 158, 157, 173, 133),
        "lower": (33, 7, 163, 144, 145, 153, 154, 155, 133),
        "centre": 468,
        "ring": (469, 470, 471, 472),
    },
}

# Inner-lip and mouth-corner landmarks, for the mouth aspect ratio (MAR) used
# by the close gesture.
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

class LatestFrameCamera:
    """
    A cv2.VideoCapture wrapper whose background thread keeps grabbing frames
    and holds only the newest one.

    Why: a plain read() -> process -> read() loop runs capture and face
    tracking one after the other, and read() often waits for the *next*
    frame - on the developer's laptop that capped tracking at ~25 fps. With
    a grabber thread they overlap, so processing never waits and always gets
    the freshest frame (lowest cursor latency).

    Stability: read() never blocks for more than ``timeout`` seconds. If the
    camera driver hangs (it happens when a USB webcam is unplugged), read()
    reports failure and the caller reopens the camera instead of freezing.
    Same interface as VideoCapture for what EyeSync uses: read(),
    release(), isOpened(), get(), set().
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0            # frames grabbed so far
        self._read_seq = 0       # last frame handed out by read()
        self._failed = False
        self._running = True
        self._thread = threading.Thread(target=self._grab, name="camera-grabber", daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        failures = 0
        try:
            while self._running:
                ok, frame = self._cap.read()
                with self._cond:
                    if ok and frame is not None:
                        self._frame, failures = frame, 0
                        self._seq += 1
                    else:
                        failures += 1
                        if failures >= 30:       # ~1 s of failed reads: camera is gone
                            self._failed = True
                    self._cond.notify_all()
                if not ok:
                    time.sleep(0.03)
        finally:
            self._cap.release()

    def read(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        """Newest frame not returned before; (False, None) on failure/timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == self._read_seq and not self._failed and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(remaining)
            if self._seq == self._read_seq:
                return False, None
            self._read_seq = self._seq
            return True, self._frame

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self._running and not self._failed

    def get(self, prop: int) -> float:
        return self._cap.get(prop)

    def set(self, prop: int, value: float) -> bool:
        return self._cap.set(prop, value)

    def release(self) -> None:
        """Stop grabbing; the device is released by the grabber thread."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        # Wait briefly so a quick reopen doesn't find the device still busy;
        # if the driver is hung, don't hang with it.
        self._thread.join(timeout=1.0)


def _measure_fps(cap: cv2.VideoCapture, frames: int = 8) -> float:
    """Frames per second actually delivered (a quick sample, for the log)."""
    t0 = time.perf_counter()
    got = sum(bool(cap.read()[0]) for _ in range(frames))
    dt = time.perf_counter() - t0
    return got / dt if dt > 0 else 0.0


class LowLight:
    """
    Keeps face tracking working in a dark room.

    Measured on the developer's frames darkened like a dim/dark room: the
    face is lost because of camera NOISE, not darkness - brightening alone
    made it worse (0% found), while a blur (denoise) and then brightening
    found the face in 88% of nearly black frames (plain: 0%) and 100% of very
    noisy ones (plain: 47%). In normal light the plain frame is better, so:

    * the plain frame is used first;
    * if it loses the face in a dark picture, the enhanced one is tried for
      that frame; after SWITCH_FRAMES such frames in a row, the enhanced
      picture is used from then on (``active``);
    * back to plain the same way, or at once when the room gets bright.
    A single frame where the face is briefly lost (a wide-open mouth, a
    hand) doesn't switch: switching back and forth every second nudged the
    tracking points (seen live).
    """

    DARK_MEAN = 90          # picture mean (0-255) below which enhancing can help
    BRIGHT_MEAN = 110       # lights on again: stop enhancing
    SWITCH_FRAMES = 5       # frames in a row where only the other picture finds the face

    def __init__(self):
        self.active = False
        self.mean = 128.0   # brightness of the last frame (0-255)
        self.other_wins = 0
        self.last_log = -1e9

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        out = cv2.GaussianBlur(frame, (5, 5), 0)
        g = min(1.0, np.log(110 / 255) / np.log(max(self.mean, 1.0) / 255))
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
        return cv2.LUT(out, lut)

    def process(self, mesh: "FaceMeshBackend", frame: np.ndarray,
                ts_ms: int | None = None) -> np.ndarray | None:
        """Face landmarks for ``frame`` (see the class docstring), or None.

        ``ts_ms`` is used for recorded videos; live capture keeps the default.
        """
        self.mean = float(cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA).mean())
        if self.active and self.mean > self.BRIGHT_MEAN:
            self.active = False
        def run(image):
            return mesh.process(image) if ts_ms is None else mesh.process(image, ts_ms)

        landmarks = run(self.enhance(frame) if self.active else frame)
        if landmarks is not None:
            self.other_wins = 0
        elif self.active or self.mean < self.DARK_MEAN:
            landmarks = run(frame if self.active else self.enhance(frame))
            self.other_wins = self.other_wins + 1 if landmarks is not None else 0
            if self.other_wins >= self.SWITCH_FRAMES:
                self.active, self.other_wins = not self.active, 0
                if time.monotonic() - self.last_log > 60:
                    self.last_log = time.monotonic()
                    log.info("[camera] Low light - enhancing the camera picture" if self.active
                             else "[camera] Enough light again - plain camera picture")
        return landmarks


def open_any_camera(index: int | None = None, width: int = 640,
                    height: int = 480, quiet: bool = False) -> "LatestFrameCamera":
    """
    ``index`` None: the first webcam of 0, 1, 2 that delivers frames - on
    some PCs the usable webcam isn't number 0 (an external USB camera, a
    virtual camera from another app). Otherwise that camera only.
    ``quiet``: failed attempts are logged at debug level only.
    """
    if index is not None:
        return open_camera(index, width, height, quiet=quiet)
    first_error = None
    for i in (0, 1, 2):
        try:
            return open_camera(i, width, height, retries=1, quiet=quiet)
        except RuntimeError as exc:
            first_error = first_error or exc
    raise first_error


def open_camera(index: int = 0, width: int = 640, height: int = 480,
                retries: int = 3, quiet: bool = False) -> "LatestFrameCamera":
    """
    Open a webcam and verify it actually delivers frames.

    ``VideoCapture`` happily returns an object even when the camera is missing
    or held by another application, so we read a test frame before trusting it.

    Windows backend choice: Media Foundation (MSMF) first, DirectShow as the
    fallback. On the developer's laptop DirectShow delivered only 15 fps
    while MSMF delivered 30 fps from the same camera - twice the tracking
    rate, which halves cursor latency. MSMF's notoriously slow open (~15 s)
    is avoided by OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0, set before
    OpenCV loads (top of this module and of the entry scripts): ~1 s.

    Raises RuntimeError with a human-readable message if no camera works.
    """
    if sys.platform == "win32":
        backends = [(cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_DSHOW, "DirectShow")]
    else:
        backends = [(cv2.CAP_ANY, "default")]

    for attempt in range(1, retries + 1):
        for backend, backend_name in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Keep the driver buffer tiny so we always process the newest frame
            # (stale frames = laggy cursor). Not every backend honours this.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Some cameras return a few empty frames while warming up.
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    log.info(f"[camera] Opened camera {index} via {backend_name} at {w}x{h}, "
                             f"~{_measure_fps(cap):.0f} fps")
                    return LatestFrameCamera(cap)
                time.sleep(0.05)
            cap.release()

        (log.debug if quiet else log.warning)(
            f"[camera] Attempt {attempt}/{retries} failed for camera {index}"
            + ("; retrying..." if attempt < retries else ""))
        time.sleep(1.0)

    raise RuntimeError(
        f"Could not open camera index {index}. Check that a webcam is connected, "
        "that no other app (Zoom, Teams, browser tab) is using it, and that "
        "camera access is allowed in Windows Settings > Privacy > Camera. "
        "You can pick a different camera with --camera N."
    )


# --------------------------------------------------------------------------- #
# Face mesh backend
# --------------------------------------------------------------------------- #

def _ensure_face_landmarker_model() -> Path:
    """Download the FaceLandmarker .task file the first time it is needed."""
    if FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 0:
        return FACE_LANDMARKER_PATH

    FACE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[mediapipe] Downloading face landmarker model to {FACE_LANDMARKER_PATH} ...")
    tmp_path = FACE_LANDMARKER_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp_path)
        os.replace(tmp_path, FACE_LANDMARKER_PATH)
    except Exception as exc:  # network errors, permission errors, ...
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the MediaPipe model ({exc}). Download it manually "
            f"from {FACE_LANDMARKER_URL} and save it as {FACE_LANDMARKER_PATH}."
        ) from exc
    return FACE_LANDMARKER_PATH


class FaceMeshBackend:
    """
    Thin wrapper that turns a BGR frame into a (478, 2) array of normalised
    landmark coordinates, or None when no face is visible.

    Coordinates are MediaPipe's native normalised space: x in [0, 1] across the
    frame width, y in [0, 1] down the frame height.

    After each call, ``last_blendshapes`` holds MediaPipe's expression
    scores (blinks, smile, ...), ``last_pose`` the head's (yaw, pitch) in
    degrees (see head_pose) and ``last_xyz`` the landmarks with their depth
    (N, 3) for the eye gaze (gaze_engine), or None when they aren't available.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self.last_blendshapes: dict[str, float] | None = None   # expression scores 0..1
        self.last_pose: tuple[float, float] | None = None        # (yaw, pitch) degrees
        self.last_xyz: np.ndarray | None = None                  # (N, 3): x, y and depth z

        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError:
            vision = None

        if vision is not None:
            # ---- Tasks API: FaceLandmarker - preferred, because it also
            # returns the expression scores (blinks, gestures).
            self.kind = "tasks.FaceLandmarker"
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_ensure_face_landmarker_model())),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                # Expression scores (eyeBlinkLeft, mouthSmileLeft, browInnerUp,
                # ...): a learned blink signal that copes with glasses far
                # better than eyelid geometry, and the face gestures.
                output_face_blendshapes=True,
                # The head's rotation: blinks and gestures are judged
                # against how the face looks in each head direction.
                output_facial_transformation_matrixes=True,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)
        else:
            # ---- Classic API: mediapipe.solutions.face_mesh.FaceMesh (old
            # MediaPipe without Tasks). Same landmarks, no expression scores.
            self.kind = "solutions.face_mesh"
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,      # track between video frames
                max_num_faces=1,              # one user drives the cursor
                refine_landmarks=True,        # iris points + better eyelids
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        log.debug(f"[mediapipe] Using {self.kind} backend")

    def process(self, frame_bgr: np.ndarray, ts_ms: int | None = None) -> np.ndarray | None:
        """Run the face mesh on one frame. Returns (N, 2) float array or None.
        ``ts_ms``: the frame's time in a video file (tools/video_features.py);
        live, the clock's."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.last_blendshapes = None
        self.last_pose = None
        self.last_xyz = None

        if self.kind == "solutions.face_mesh":
            rgb.flags.writeable = False  # lets MediaPipe avoid a copy
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            # VIDEO mode requires strictly increasing millisecond timestamps.
            if ts_ms is None:
                ts_ms = int((time.monotonic() - self._t0) * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            result = self._mesh.detect_for_video(image, ts_ms)
            if not result.face_landmarks:
                return None
            lms = result.face_landmarks[0]
            if getattr(result, "face_blendshapes", None):
                self.last_blendshapes = {c.category_name: float(c.score)
                                         for c in result.face_blendshapes[0]}
            if getattr(result, "facial_transformation_matrixes", None):
                self.last_pose = head_pose(np.asarray(result.facial_transformation_matrixes[0]))

        self.last_xyz = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
        return np.ascontiguousarray(self.last_xyz[:, :2])

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

NOSE_TIP = 4                     # landmark followed by the head pointer
EYE_OUTER = (33, 263)            # outer eye corners: face scale for the head pointer


def nose_pointer(landmarks: np.ndarray, frame_w: int, frame_h: int) -> tuple[np.ndarray, float]:
    """
    Head-pointer input: the nose tip in pixels, and the distance between the
    outer eye corners (pixels) as the face's scale. On the developer's
    frames the nose tip jittered 2-3x less, relative to its range, than the
    head angles - the steadiest signal for moving a cursor.
    """
    size = np.array([frame_w, frame_h], np.float64)
    nose = landmarks[NOSE_TIP, :2] * size
    eyes = float(np.linalg.norm((landmarks[EYE_OUTER[0], :2] - landmarks[EYE_OUTER[1], :2]) * size))
    return nose, max(eyes, 1.0)


def head_pose(matrix: np.ndarray) -> tuple[float, float]:
    """
    (yaw, pitch) of the head in degrees from MediaPipe's facial
    transformation matrix: 0, 0 facing the camera; the sign says which side.
    """
    rot = np.asarray(matrix, np.float64)[:3, :3]
    forward = rot[:, 2] / max(float(np.linalg.norm(rot[:, 2])), 1e-9)   # out of the face
    yaw = math.degrees(math.atan2(forward[0], abs(forward[2])))
    pitch = math.degrees(math.atan2(forward[1], math.hypot(forward[0], forward[2])))
    return yaw, pitch


def head_roll(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    """
    Sideways head tilt in degrees, from the line between the outer eye
    corners: 0 upright, positive = ear towards the user's LEFT shoulder,
    negative = towards the right. (The camera picture isn't mirrored: the
    user's left eye, landmark 263, is on the picture's right and drops when
    they tilt to the left.)
    """
    size = np.array([frame_w, frame_h], np.float64)
    right_eye, left_eye = landmarks[EYE_OUTER[0], :2] * size, landmarks[EYE_OUTER[1], :2] * size
    dx, dy = left_eye - right_eye
    return math.degrees(math.atan2(dy, dx))


MOUTH_CORNERS = (61, 291)        # outer mouth corners
SUBNASALE = 2                    # just below the nose: the face's midline above the mouth


def mouth_shift(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    """
    How far the mouth sits to one side of the nose, in face widths, along
    the line between the eyes: positive = towards the user's LEFT. Tells a
    mouth pulled to the left from one pulled to the right by the face's own
    left and right (the eye landmarks, as head_roll), whichever way an
    expression score names them.
    """
    size = np.array([frame_w, frame_h], np.float64)
    right_eye, left_eye = landmarks[EYE_OUTER[0], :2] * size, landmarks[EYE_OUTER[1], :2] * size
    across = left_eye - right_eye
    width = float(np.linalg.norm(across))
    if width < 1e-6:
        return 0.0
    mouth = landmarks[list(MOUTH_CORNERS), :2].mean(axis=0) * size
    nose = landmarks[SUBNASALE, :2] * size
    return float(np.dot(mouth - nose, across / width)) / width


def mouth_aspect_ratio(landmarks: np.ndarray, frame_w: int, frame_h: int,
                       pose: tuple[float, float] | None = None) -> float:
    """
    Inner-lip opening divided by mouth width (pixel units). About 0-0.1 with
    the mouth closed, 0.1-0.45 while talking, 0.6+ with the mouth wide open.
    With the head ``pose`` (head_pose), the foreshortening is undone: a
    turned head makes the mouth look narrower (an ordinary opening would
    read as wide open) and a tilted one makes the opening look shorter.
    """
    scale = np.array([frame_w, frame_h], dtype=np.float32)
    p = landmarks[[UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT]] * scale
    width = float(np.linalg.norm(p[2] - p[3]))
    opening = float(np.linalg.norm(p[0] - p[1]))
    if pose is not None:
        width /= max(math.cos(math.radians(pose[0])), 0.5)
        opening /= max(math.cos(math.radians(pose[1])), 0.5)
    return opening / width if width > 1e-6 else 0.0


def _single_eye_ear(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) for one eye (pixel coords)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return float(vertical / horizontal) if horizontal > 1e-6 else 0.0


def eye_aspect_ratios(landmarks: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float]:
    """EAR of the (left, right) eye separately - see eye_aspect_ratio()."""
    scale = np.array([frame_w, frame_h], dtype=np.float32)
    left = landmarks[list(LEFT_EYE_EAR_IDX)] * scale
    right = landmarks[list(RIGHT_EYE_EAR_IDX)] * scale
    return _single_eye_ear(left), _single_eye_ear(right)


def iris_apertures(landmarks: np.ndarray, frame_w: int, frame_h: int,
                   ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return geometric iris/eyelid features for the (left, right) eyes.

    Each tuple is ``(visible, down_offset, radius, opening)``. ``visible`` is
    the fraction of the predicted iris's vertical diameter between the lids;
    ``down_offset`` is its centre relative to the lid midpoint, positive down;
    ``radius`` and ``opening`` are normalised by eye-corner distance.

    MediaPipe still predicts iris points when an eye is shut. Consequently
    these values describe geometry only and must be combined with EAR,
    expression scores and time. They are recorded by the public-dataset tools
    so a learned guard can be tested without changing live blink behaviour.
    """
    if len(landmarks) < 478:
        empty = (float("nan"),) * 4
        return empty, empty
    scale = np.array([frame_w, frame_h], dtype=np.float64)
    points = np.asarray(landmarks, dtype=np.float64) * scale
    answers = []
    for spec in EYE_IRIS_IDX.values():
        a, b = points[list(spec["corners"])]
        if b[0] < a[0]:                    # local +x always goes left-to-right
            a, b = b, a
        across = b - a
        width = float(np.linalg.norm(across))
        if width < 1e-6:
            answers.append((float("nan"),) * 4)
            continue
        ex = across / width
        ey = np.array([-ex[1], ex[0]])     # down relative to the eye corners
        origin = a

        def local(indices):
            q = points[list(indices)] - origin
            return np.column_stack((q @ ex, q @ ey))

        upper, lower = local(spec["upper"]), local(spec["lower"])
        centre = local((spec["centre"],))[0]
        ring = local(spec["ring"])
        radius_px = float(np.mean(np.linalg.norm(ring - centre, axis=1)))
        # Interpolate the two lid polylines at the iris centre. np.interp also
        # clamps a slightly outlying iris prediction to the nearest corner.
        ux = np.argsort(upper[:, 0])
        lx = np.argsort(lower[:, 0])
        top = float(np.interp(centre[0], upper[ux, 0], upper[ux, 1]))
        bottom = float(np.interp(centre[0], lower[lx, 0], lower[lx, 1]))
        top, bottom = min(top, bottom), max(top, bottom)
        cy = float(centre[1])
        if radius_px > 1e-6:
            visible = (min(bottom, cy + radius_px) - max(top, cy - radius_px)) / (2 * radius_px)
            visible = float(np.clip(visible, 0.0, 1.0))
        else:
            visible = 0.0
        answers.append((visible, (cy - (top + bottom) / 2) / width,
                        radius_px / width, max(0.0, bottom - top) / width))
    return answers[0], answers[1]


def eyes_level_ratio(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    """
    Narrower / wider eye width (corner to corner) as the camera sees them:
    ~1.0 facing the camera, smaller as the head turns sideways (the far eye
    is foreshortened and partly hidden; ~0.7 at about 20 degrees).
    """
    scale = np.array([frame_w, frame_h], dtype=np.float32)
    left = np.linalg.norm((landmarks[263] - landmarks[362]) * scale)
    right = np.linalg.norm((landmarks[33] - landmarks[133]) * scale)
    return float(min(left, right) / max(left, right, 1e-6))


def eye_aspect_ratio(landmarks: np.ndarray, frame_w: int, frame_h: int) -> float:
    """
    Average Eye Aspect Ratio of both eyes.

    Landmarks are converted back to pixel units first: in normalised space x
    and y have different scales (a 640x480 frame stretches y by 4/3), which
    would distort the ratio. Open eyes typically give ~0.25-0.35; a closed eye
    drops towards ~0.05-0.15.
    """
    left, right = eye_aspect_ratios(landmarks, frame_w, frame_h)
    return (left + right) / 2.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@dataclass
class ScreenSize:
    width: int
    height: int


def get_screen_size() -> ScreenSize:
    """
    Physical resolution of the primary monitor.

    On Windows we first opt into per-monitor DPI awareness; otherwise a 4K
    screen at 150% scaling would report 2560x1440 and every predicted pixel
    would be off by the scaling factor.
    """
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        user32 = ctypes.windll.user32
        return ScreenSize(user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))

    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = ScreenSize(root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        log.warning("[screen] Could not detect screen size; assuming 1920x1080")
        return ScreenSize(1920, 1080)


def primary_screen_size() -> ScreenSize | None:
    """
    The primary monitor's resolution now (after get_screen_size made the
    process DPI aware), cheap enough to ask every few seconds; None where it
    can't be asked cheaply (not Windows) or Windows gives no answer.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    user32 = ctypes.windll.user32
    w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return ScreenSize(w, h) if w > 0 and h > 0 else None


def setup_logging(logfile: Path | None = None) -> None:
    """
    Log to the console and (optionally) to a rotating file, so a long-running
    session leaves a trail when something goes wrong overnight.
    """
    from logging.handlers import RotatingFileHandler

    handlers: list[logging.Handler] = []
    if sys.stdout is not None:      # None when running windowless (pythonw)
        handlers.append(logging.StreamHandler(sys.stdout))
    if logfile is not None:
        handlers.append(RotatingFileHandler(logfile, maxBytes=1_000_000, backupCount=3,
                                            encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def draw_landmark_overlay(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw the face box, the nose tip (the pointer) and the eyelid points on a preview frame."""
    h, w = frame.shape[:2]
    face = landmarks[:468]
    (min_x, min_y), (max_x, max_y) = face.min(axis=0), face.max(axis=0)
    cv2.rectangle(frame, (int(min_x * w), int(min_y * h)),
                  (int(max_x * w), int(max_y * h)), (0, 255, 0), 2)
    for idx in LEFT_EYE_EAR_IDX + RIGHT_EYE_EAR_IDX:
        x, y = landmarks[idx]
        cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 200, 255), -1)
    x, y = landmarks[NOSE_TIP]
    cv2.circle(frame, (int(x * w), int(y * h)), 4, (255, 0, 255), -1)
