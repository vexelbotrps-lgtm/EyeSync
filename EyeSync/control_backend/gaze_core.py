"""
gaze_core.py - shared building blocks for the face-gaze "Brain".

calibrate.py (training) and live.py (inference) both import from here so the
features used at training time are IDENTICAL to those used at inference time.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pickle

# ----------------------------- shared configuration ----------------------------- #
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 320, 240       # request the webcam's faster low-latency mode
MODEL_PATH = Path(__file__).with_name("gaze_model.joblib")
CLICK_PROFILE_PATH = Path(__file__).with_name("click_profile.joblib")
BLINK_PROFILE_PATH = Path(__file__).with_name("blink_profile.json")
MOUTH_TEETH_MODEL_PATH = Path(__file__).with_name("mouth_teeth_model.joblib")
MOUTH_OPEN_PROFILE_PATH = Path(__file__).with_name("mouth_open_profile.json")
TAB_PROFILE_PATH = Path(__file__).with_name("tab_profile.json")
CLICK_PROFILE_VERSION = 11
BLINK_PROFILE_VERSION = 4
MOUTH_OPEN_PROFILE_VERSION = 1
TAB_PROFILE_VERSION = 1
MOUTH_OPEN_THRESHOLD = 0.142
MOUTH_OPEN_RELEASE_THRESHOLD = 0.109
EAR_THRESHOLD = 0.18     # average EAR below this => eyes closed
BLINK_MIN_FRAMES = 5   # ~170 ms at 30 FPS: deliberate blink, not a natural one     # consecutive closed frames required (rejects landmark glitches)
BLINK_COOLDOWN_S = 0.40  # minimum time between two clicks
EMA_ALPHA = 0.25         # smoothing: lower = smoother but laggier

FEATURE_NAMES = ("face_min_x", "face_min_y", "face_max_x", "face_max_y")
_MOUTH_HOG = cv2.HOGDescriptor((96, 48), (16, 16), (8, 8), (8, 8), 9)
_MOUTH_LANDMARKS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
)

# Eyelid landmarks of the standard 468-point FaceMesh (NO iris points; those are
# 468-477 and only exist with refine_landmarks=True, which we never enable).
#   p1,p4 = eye corners; p2,p3 = upper lid; p5,p6 = lower lid
#   EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)


# ---------------------------------- camera ---------------------------------- #
class CameraError(RuntimeError):
    """Raised when no working camera can be opened."""


def open_camera(index: int = CAMERA_INDEX,
                width: int = FRAME_WIDTH,
                height: int = FRAME_HEIGHT) -> cv2.VideoCapture:
    """Open a webcam and verify it really delivers frames.

    On Windows the default MSMF backend is often slow or reports "opened" but
    never yields frames, so we try DirectShow first and fall back. Each
    candidate is validated by reading an actual frame.
    """
    backends = ([cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
                if sys.platform.startswith("win") else [cv2.CAP_ANY])

    problem = "no backend could open the device"
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # compressed stream: 30 FPS
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # low latency: avoid queued stale frames
        for _ in range(10):                      # drivers need a few warm-up frames
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        problem = f"backend {backend} opened the device but returned no frames"
        cap.release()

    raise CameraError(
        f"Could not open camera index {index}: {problem}. Check it is plugged in, "
        "not used by another app (Zoom/Teams/browser), and that Windows camera "
        "privacy settings allow it. Try another index with --camera 1."
    )


def camera_frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Return the actual capture dimensions negotiated by the camera driver."""
    return (int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# ------------------------------ face features + EAR ------------------------------ #
@dataclass
class FrameFeatures:
    features: np.ndarray      # (4,) float32 [min_x, min_y, max_x, max_y], each in [0, 1]
    ear: float                # average Eye Aspect Ratio of both eyes
    ear_right: float = 0.0    # subject's right eye
    ear_left: float = 0.0     # subject's left eye
    smile: float = 0.0        # mouth-corner width / inter-ocular distance
    smile_features: Optional[np.ndarray] = None
    mouth: float = 0.0        # 3D outer-lip gap / 3D inter-ocular distance
    mouth_patch: Optional[np.ndarray] = None
    lm: Optional[np.ndarray] = None   # (468,3) landmarks centred on the nose, scaled by iod


# Eye mesh: corner pair + THREE upper/lower eyelid pairs per eye, measured in 3D (x, y, z)
# so the ratio stays stable when the head is turned or tilted.
EYE_MESH = {
    "right": ((33, 133), ((160, 144), (159, 145), (158, 153))),
    "left":  ((362, 263), ((385, 380), (386, 374), (387, 373))),
}


def _eye_aspect_ratio(pts3d: np.ndarray, eye: str) -> float:
    # pts3d is in PIXEL units (x*w, y*h, z*w); normalised coords would distort the ratio.
    (c1, c2), pairs = EYE_MESH[eye]
    horizontal = np.linalg.norm(pts3d[c1] - pts3d[c2])
    vertical = np.mean([np.linalg.norm(pts3d[u] - pts3d[l]) for u, l in pairs])
    return float(vertical / (horizontal + 1e-9))


def mouth_patch_hog(patch: np.ndarray) -> np.ndarray:
    """Normalize a mouth crop into shared HOG features for click classifiers."""
    gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 48), interpolation=cv2.INTER_AREA)
    return _MOUTH_HOG.compute(gray).reshape(-1).astype(np.float32)


def _crop_mouth(frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    left, right = points_px[61, :2], points_px[291, :2]
    mouth_width = float(np.linalg.norm(right - left))
    if mouth_width < 2:
        return np.zeros((48, 96), dtype=np.uint8)
    center = 0.5 * (left + right)
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    transform = cv2.getRotationMatrix2D(center, angle, 64.0 / mouth_width)
    transform[0, 2] += 48.0 - center[0]
    transform[1, 2] += 24.0 - center[1]
    return cv2.warpAffine(frame_gray, transform, (96, 48),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)





class FaceFeatureExtractor:
    """MediaPipe FaceMesh wrapper producing the 4-float whole-face feature vector."""

    def __init__(self, static_image_mode: bool = False) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,         # <-- omits the iris landmarks entirely
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[FrameFeatures]:
        """Return features for this frame, or None if no face is visible."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe wants RGB
        rgb.flags.writeable = False                        # lets MediaPipe skip a copy
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lms = result.multi_face_landmarks[0].landmark
        pts3 = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)  # (468, 3)
        pts_norm = pts3[:, :2]                                                 # 0..1

        # Global bounding box of the ENTIRE mesh, normalised to the frame boundary.
        mins = pts_norm.min(axis=0)
        maxs = pts_norm.max(axis=0)
        features = np.clip(np.array([mins[0], mins[1], maxs[0], maxs[1]],
                                    dtype=np.float32), 0.0, 1.0)

        # EAR from eyelid landmarks only.
        pts_px = pts3 * np.array([w, h, w], dtype=np.float32)   # z is scaled like x
        ear_r = _eye_aspect_ratio(pts_px, "right")
        ear_l = _eye_aspect_ratio(pts_px, "left")
        eye_left = 0.5 * (pts_px[33, :2] + pts_px[133, :2])
        eye_right = 0.5 * (pts_px[362, :2] + pts_px[263, :2])
        iod_2d = float(np.linalg.norm(eye_left - eye_right)) + 1e-6
        image_left_eye, image_right_eye = sorted(
            (eye_left, eye_right), key=lambda point: point[0]
        )
        # Use image-plane mouth width: 3-D landmark depth estimates shift with
        # head pitch and made a real smile score much lower when looking down.
        smile = float(abs(pts_px[61, 0] - pts_px[291, 0])) / iod_2d
        iod = float(np.linalg.norm(pts_px[33] - pts_px[263])) + 1e-6
        inner_mouth = float(abs(pts_px[13, 1] - pts_px[14, 1])) / iod_2d
        mouth_points = pts_px[list(_MOUTH_LANDMARKS), :2]
        mouth = float(np.ptp(mouth_points[:, 1])) / iod_2d
        eye_mid_y = 0.5 * (image_left_eye[1] + image_right_eye[1])
        mouth_mid_y = 0.5 * (pts_px[61, 1] + pts_px[291, 1])
        mouth_pose = float((mouth_mid_y - eye_mid_y) / iod_2d)
        eye_roll = float(np.arctan2(image_right_eye[1] - image_left_eye[1],
                                    image_right_eye[0] - image_left_eye[0]))
        nose_yaw = float((pts_px[1, 0] - 0.5 * (
            image_left_eye[0] + image_right_eye[0]
        )) / iod_2d)
        smile_features = np.array(
            [smile, inner_mouth, mouth_pose, eye_roll, nose_yaw],
            dtype=np.float32,
        )
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mouth_patch = _crop_mouth(frame_gray, pts_px)
        return FrameFeatures(features=features, ear=0.5 * (ear_r + ear_l),
                             ear_right=ear_r, ear_left=ear_l, smile=smile,
                             smile_features=smile_features,
                             mouth=mouth,
                             mouth_patch=mouth_patch,
                             lm=((pts_px - pts_px[1]) / iod).astype(np.float32))

    def close(self) -> None:
        self._mesh.close()


# ------------------------------------ blink ------------------------------------ #
class BlinkDetector:
    """EAR stream -> one-shot click events.

    Fires once after BLINK_MIN_FRAMES consecutive closed frames, then latches:
    holding the eyes shut does not auto-repeat; they must re-open first.
    A cooldown also suppresses double-clicks from a flutter blink.
    """

    def __init__(self, threshold: float = EAR_THRESHOLD,
                 min_frames: int = BLINK_MIN_FRAMES,
                 cooldown_s: float = BLINK_COOLDOWN_S) -> None:
        self.threshold, self.min_frames, self.cooldown_s = threshold, min_frames, cooldown_s
        self._closed_frames = 0
        self._latched = False
        self._last_click = 0.0

    @property
    def eyes_closed(self) -> bool:
        return self._closed_frames > 0

    def reset(self) -> None:
        """Call when the face is lost so stale state cannot cause a phantom click."""
        self._closed_frames = 0
        self._latched = False

    def update(self, ear: float) -> bool:
        if ear < self.threshold:
            self._closed_frames += 1
        else:
            self._closed_frames = 0
            self._latched = False            # eyes re-opened -> armed again

        now = time.monotonic()
        if (self._closed_frames >= self.min_frames and not self._latched
                and now - self._last_click >= self.cooldown_s):
            self._latched = True
            self._last_click = now
            return True
        return False


# ---------------------------------- smoothing ---------------------------------- #
class EmaFilter:
    """Exponential moving average of a 2-D point: s = a*new + (1-a)*s_prev."""

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        self._state = p if self._state is None else self.alpha * p + (1 - self.alpha) * self._state
        return float(self._state[0]), float(self._state[1])


# -------------------------------- model helpers -------------------------------- #
def load_model_bundle(path: Path = MODEL_PATH) -> dict:
    """Load the dict written by calibrate.py."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Run `python calibrate.py` first.")
    bundle = joblib.load(path)
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Model was trained on a different feature layout; recalibrate.")
    if "camera_size" not in bundle:
        raise ValueError("Gaze model has no camera-mode metadata; run `python calibrate.py` again.")
    return bundle


def predict_xy(bundle: dict, features: np.ndarray) -> tuple[float, float]:
    """4-float features -> screen pixels, clamped to the screen."""
    sw, sh = bundle["screen_w"], bundle["screen_h"]
    scaled = bundle["x_scaler"].transform(features.reshape(1, -1))
    if "z_lo" in bundle:   # never extrapolate far outside the calibrated range
        scaled = np.clip(scaled, bundle["z_lo"], bundle["z_hi"])
    nx, ny = bundle["model"].predict(scaled)[0]      # network outputs 0..1 coordinates
    return (float(np.clip(nx * sw, 0, sw - 1)),
            float(np.clip(ny * sh, 0, sh - 1)))


# ---------------------------- extra smoothing (v2) ---------------------------- #
class VectorEma:
    """EMA over a feature vector; smooths landmark noise before the network sees it."""

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        self._state = v if self._state is None else self.alpha * v + (1 - self.alpha) * self._state
        return self._state.astype(np.float32)


class OneEuroFilter2D:
    """One-Euro filter (Casiez et al.): heavy smoothing when the point is nearly
    still (kills jitter), lighter smoothing when it moves fast (low lag).
    min_cutoff: lower = steadier at rest.  beta: higher = less lag when moving."""

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.004,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: Optional[np.ndarray] = None
        self._dx = np.zeros(2)
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._t = p, t
            return float(p[0]), float(p[1])
        dt = max(t - self._t, 1e-3)
        dx = (p - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(dt, cutoff)
        self._x = a * p + (1 - a) * self._x
        self._t = t
        return float(self._x[0]), float(self._x[1])


# ------------------------- stable hybrid gaze model -------------------------- #
class HybridGazeModel:
    """Ridge linear map plus a strongly regularised MLPRegressor on its residual.
    The linear part extrapolates predictably (no sideways swaying when the head
    moves between calibration dots); the MLP only adds gentle non-linear correction."""

    def __init__(self, mlp_alpha: float = 10.0) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        self.linear = Ridge(alpha=1.0)
        self.mlp = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh",
                                alpha=mlp_alpha, max_iter=800, random_state=0)
        self.n_iter_ = 0

    def fit(self, X, Y):
        self.linear.fit(X, Y)
        self.mlp.fit(X, Y - self.linear.predict(X))
        self.n_iter_ = self.mlp.n_iter_
        return self

    def predict(self, X):
        return self.linear.predict(X) + self.mlp.predict(X)


def load_click_profile(path: Path = CLICK_PROFILE_PATH) -> Optional[dict]:
    """Personal pose-specific smile classifier (None if not recorded)."""
    try:
        profile = joblib.load(path)
    except (OSError, ValueError, EOFError, AttributeError, ImportError,
            TypeError, pickle.UnpicklingError):
        return None
    if not isinstance(profile, dict):
        return None
    if profile.get("version") != CLICK_PROFILE_VERSION:
        return None
    try:
        models = profile["models"]
        for pose in ("center", "down", "left", "right"):
            model = models[pose]
            classifier = model["classifier"]
            values = (
                float(model["threshold"]),
                float(model["release_threshold"]),
                float(model["pose"]),
                float(model["yaw"]),
                float(model["validation_accuracy"]),
            )
            if (not np.all(np.isfinite(values))
                    or model["release_threshold"] >= model["threshold"]
                    or not callable(getattr(classifier, "decision_function", None))
                    or int(classifier.n_features_in_) != _MOUTH_HOG.getDescriptorSize()
                    or not np.array_equal(np.asarray(classifier.classes_), [0, 1])):
                return None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return profile


def smile_visibility_score(profile: dict, mouth_features: np.ndarray,
                           smile_features: np.ndarray) -> tuple[float, float, float]:
    """Return the nearest-pose smile margin, activation threshold, and release threshold."""
    pose, yaw = float(smile_features[2]), float(smile_features[4])
    models = profile["models"]
    selected = min(
        models,
        key=lambda key: ((pose - float(models[key]["pose"])) ** 2
                         + (yaw - float(models[key]["yaw"])) ** 2),
    )
    model = models[selected]
    features = mouth_features.reshape(1, -1)
    score = float(model["classifier"].decision_function(features)[0])
    release_threshold = float(model["release_threshold"])
    calibrated_threshold = float(model["threshold"])
    threshold = release_threshold + 0.25 * (
        calibrated_threshold - release_threshold
    )
    return score, threshold, release_threshold


def load_blink_profile(path: Path = BLINK_PROFILE_PATH) -> Optional[dict]:
    """Load a legacy pose-specific eye-closure profile, if present."""
    import json
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(profile, dict) or profile.get("version") != BLINK_PROFILE_VERSION:
        return None
    try:
        for pose in ("center", "down", "left", "right"):
            model = profile["models"][pose]
            values = np.asarray([
                model["baseline_l"], model["baseline_r"],
                model["close_l"], model["close_r"],
                model["open_l"], model["open_r"],
                model["pose"], model["yaw"],
            ], dtype=np.float64)
            if (not np.all(np.isfinite(values))
                    or np.any(values[:2] <= 0)
                    or model["close_l"] >= model["open_l"]
                    or model["close_r"] >= model["open_r"]):
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return profile


def load_mouth_open_profile(path: Path = MOUTH_OPEN_PROFILE_PATH) -> Optional[dict]:
    """Hysteresis thresholds for the sustained-mouth-open right-click gesture."""
    import json
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(profile, dict) or profile.get("version") != MOUTH_OPEN_PROFILE_VERSION:
        return None
    try:
        release = float(profile["release_threshold"])
        opened = float(profile["open_threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.isfinite(release) or not np.isfinite(opened)
            or release < 0.0 or opened <= release):
        return None
    return {"version": MOUTH_OPEN_PROFILE_VERSION,
            "release_threshold": release, "open_threshold": opened}


def load_tab_profile(path: Path = TAB_PROFILE_PATH) -> Optional[dict]:
    """Load personalized, normalized bilateral eye-closure thresholds."""
    import json

    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return load_tab_profile_from_data(profile)


def load_tab_profile_from_data(profile: object) -> Optional[dict]:
    """Validate a decoded Tab profile before using or saving it."""
    if not isinstance(profile, dict) or profile.get("version") != TAB_PROFILE_VERSION:
        return None
    try:
        values = np.asarray([
            profile["close_ratio_left"], profile["open_ratio_left"],
            profile["close_ratio_right"], profile["open_ratio_right"],
        ], dtype=np.float64)
        if (not np.all(np.isfinite(values))
                or np.any(values <= 0)
                or np.any(values > 1.5)
                or values[0] >= values[1]
                or values[2] >= values[3]):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return profile


def eye_closure_signal(profile: dict, ear_left: float, ear_right: float,
                       smile_features: np.ndarray
                       ) -> tuple[float, float, float, float, float, float]:
    """Normalize eye aspect ratios to the nearest calibrated head pose."""
    pose, yaw = float(smile_features[2]), float(smile_features[4])
    models = profile["models"]
    selected = min(
        models,
        key=lambda key: ((pose - float(models[key]["pose"])) ** 2
                         + (yaw - float(models[key]["yaw"])) ** 2),
    )
    model = models[selected]
    baseline_l = float(model["baseline_l"])
    baseline_r = float(model["baseline_r"])
    return (
        float(ear_left) / baseline_l,
        float(ear_right) / baseline_r,
        float(model["close_l"]),
        float(model["close_r"]),
        float(model["open_l"]),
        float(model["open_r"]),
    )


class BilateralCloseClickDetector:
    """Legacy detector for a sustained closure of both eyes."""

    HOLD_S = 1.0
    COOLDOWN_S = 0.8
    AMBIGUOUS_GAP_S = 0.20
    FALLBACK_CLOSE_RATIO = 0.72
    FALLBACK_OPEN_RATIO = 0.86
    FALLBACK_BASELINE_SAMPLES = 30

    def __init__(self, threshold: float = EAR_THRESHOLD,
                 profile: Optional[dict] = None) -> None:
        self.baseline_l = self.baseline_r = 1.0
        self.normalize_ear = False
        self.adaptive_fallback = profile is None
        self.fallback_baseline_ready = not self.adaptive_fallback
        self._fallback_open_l: list[float] = []
        self._fallback_open_r: list[float] = []
        self.last_left = self.last_right = 0.0
        self.baseline_alpha = 0.03
        self.close_l = self.close_r = threshold
        self.open_l = self.open_r = threshold + 0.02
        if profile:
            self.baseline_l = float(profile.get("baseline_l", 1.0))
            self.baseline_r = float(profile.get("baseline_r", 1.0))
            self.normalize_ear = "baseline_l" in profile and "baseline_r" in profile
            self.close_l = min(0.55, float(profile["close_l"]))
            self.close_r = min(0.55, float(profile["close_r"]))
            self.open_l = max(self.close_l + 0.08, min(0.68, float(profile["open_l"])))
            self.open_r = max(self.close_r + 0.08, min(0.68, float(profile["open_r"])))
        self._since: Optional[float] = None
        self._gap_since: Optional[float] = None
        self._latched = False
        self._last_click = 0.0

    @property
    def active(self) -> bool:
        return self._since is not None

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def reset(self) -> None:
        self._since = None
        self._gap_since = None
        self._latched = False

    def update(self, ear_left: float, ear_right: float,
               now: Optional[float] = None,
               closed_probabilities: Optional[tuple[float, float]] = None,
               probability_threshold: float = 0.9,
               thresholds: Optional[tuple[float, float, float, float]] = None
               ) -> bool:
        if closed_probabilities is not None:
            closed_left, closed_right = closed_probabilities
            left_closed = closed_left >= probability_threshold
            right_closed = closed_right >= probability_threshold
            left_open = closed_left < probability_threshold
            right_open = closed_right < probability_threshold
        elif thresholds is not None:
            self.close_l, self.close_r, self.open_l, self.open_r = thresholds
            left_closed, right_closed = ear_left < self.close_l, ear_right < self.close_r
            left_open, right_open = ear_left >= self.open_l, ear_right >= self.open_r
        elif self.adaptive_fallback:
            raw_left, raw_right = ear_left, ear_right
            if self.baseline_l == 1.0 and self.baseline_r == 1.0:
                if raw_left >= EAR_THRESHOLD and raw_right >= EAR_THRESHOLD:
                    if len(self._fallback_open_l) < self.FALLBACK_BASELINE_SAMPLES:
                        self._fallback_open_l.append(raw_left)
                        self._fallback_open_r.append(raw_right)
                    if len(self._fallback_open_l) == self.FALLBACK_BASELINE_SAMPLES:
                        self.baseline_l = float(np.median(self._fallback_open_l))
                        self.baseline_r = float(np.median(self._fallback_open_r))
                        self.fallback_baseline_ready = True
                normalized_left, normalized_right = raw_left, raw_right
                self.close_l = self.close_r = EAR_THRESHOLD
                self.open_l = self.open_r = EAR_THRESHOLD + 0.02
            else:
                normalized_left = raw_left / self.baseline_l
                normalized_right = raw_right / self.baseline_r
                self.close_l = self.close_r = self.FALLBACK_CLOSE_RATIO
                self.open_l = self.open_r = self.FALLBACK_OPEN_RATIO
                safe_open = (normalized_left >= self.open_l
                             and normalized_right >= self.open_r)
                if safe_open:
                    self.baseline_l = (
                        (1.0 - self.baseline_alpha) * self.baseline_l
                        + self.baseline_alpha * raw_left
                    )
                    self.baseline_r = (
                        (1.0 - self.baseline_alpha) * self.baseline_r
                        + self.baseline_alpha * raw_right
                    )
            ear_left, ear_right = normalized_left, normalized_right
            self.last_left, self.last_right = ear_left, ear_right
            left_closed, right_closed = ear_left < self.close_l, ear_right < self.close_r
            left_open, right_open = ear_left >= self.open_l, ear_right >= self.open_r
        elif self.normalize_ear:
            raw_left, raw_right = ear_left, ear_right
            normalized_left = raw_left / self.baseline_l
            normalized_right = raw_right / self.baseline_r
            safe_open = (normalized_left >= self.open_l
                         and normalized_right >= self.open_r)
            if safe_open:
                self.baseline_l = ((1.0 - self.baseline_alpha) * self.baseline_l
                                   + self.baseline_alpha * raw_left)
                self.baseline_r = ((1.0 - self.baseline_alpha) * self.baseline_r
                                   + self.baseline_alpha * raw_right)
            ear_left = raw_left / self.baseline_l
            ear_right = raw_right / self.baseline_r
            left_closed, right_closed = ear_left < self.close_l, ear_right < self.close_r
            left_open, right_open = ear_left >= self.open_l, ear_right >= self.open_r
        else:
            left_closed, right_closed = ear_left < self.close_l, ear_right < self.close_r
            left_open, right_open = ear_left >= self.open_l, ear_right >= self.open_r
        if thresholds is not None or not self.adaptive_fallback:
            self.last_left, self.last_right = ear_left, ear_right

        both_closed = left_closed and right_closed
        both_open = left_open and right_open
        now = time.monotonic() if now is None else now

        if both_open:
            self._since = None
            self._gap_since = None
            self._latched = False
        elif both_closed and self._since is None:
            self._since = now
            self._gap_since = None
        elif both_closed:
            self._gap_since = None
        elif self._since is not None and not self._latched:
            if self._gap_since is None:
                self._gap_since = now
            elif now - self._gap_since > self.AMBIGUOUS_GAP_S:
                self._since = None
                self._gap_since = None

        if (self._since is not None and not self._latched
                and now - self._since >= self.HOLD_S
                and now - self._last_click >= self.COOLDOWN_S
                and self.fallback_baseline_ready):
            self._latched = True
            self._last_click = now
            return True
        return False


class LandmarkSmileClickDetector:
    """A sustained closed-mouth smile, measured from face landmarks, clicks once."""

    HOLD_S = 1.2
    BASELINE_SAMPLES = 24
    SMILE_RATIO = 1.15
    RELEASE_RATIO = 1.08

    def __init__(self) -> None:
        self._baseline_samples: list[float] = []
        self._baseline: Optional[float] = None
        self._since: Optional[float] = None
        self._fired = False
        self.last_ratio = 1.0

    @property
    def active(self) -> bool:
        return self._since is not None

    @property
    def ready(self) -> bool:
        return self._baseline is not None

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def reset(self) -> None:
        self._since = None
        self._fired = False

    def update(self, mouth_width: float, mouth_open: float,
               max_mouth_open: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if (not np.isfinite(mouth_width) or mouth_width <= 0
                or not np.isfinite(mouth_open) or not np.isfinite(max_mouth_open)):
            self.reset()
            return False

        if self._baseline is None:
            if mouth_open < max_mouth_open:
                self._baseline_samples.append(float(mouth_width))
                if len(self._baseline_samples) >= self.BASELINE_SAMPLES:
                    self._baseline = float(np.median(self._baseline_samples))
                    self._baseline_samples.clear()
            return False

        self.last_ratio = float(mouth_width / self._baseline)
        if mouth_open >= max_mouth_open:
            self.reset()
            return False

        if self._since is None:
            if self.last_ratio >= self.SMILE_RATIO:
                self._since = now
                self._fired = False
            elif self.last_ratio <= self.RELEASE_RATIO:
                self._baseline = (
                    0.998 * self._baseline + 0.002 * float(mouth_width)
                )
            return False

        if self.last_ratio < self.RELEASE_RATIO:
            self.reset()
            return False

        if not self._fired and now - self._since >= self.HOLD_S:
            self._fired = True
            return True
        return False


class BilateralEyeClosureDetector:
    """Recognize a sustained bilateral eye closure using personal EAR baselines."""

    HOLD_S = 1.0
    RELEASE_S = 0.25
    AMBIGUOUS_GAP_S = 0.20
    BASELINE_SAMPLES = 30
    BASELINE_MIN_EAR = 0.05
    CLOSE_RATIO = 0.84
    OPEN_RATIO = 0.90
    BASELINE_ALPHA = 0.03

    def __init__(self, profile: Optional[dict] = None) -> None:
        self._baseline_samples_l: list[float] = []
        self._baseline_samples_r: list[float] = []
        self._baseline_l: Optional[float] = None
        self._baseline_r: Optional[float] = None
        self.close_ratio_l = float(
            profile["close_ratio_left"] if profile else self.CLOSE_RATIO
        )
        self.open_ratio_l = float(
            profile["open_ratio_left"] if profile else self.OPEN_RATIO
        )
        self.close_ratio_r = float(
            profile["close_ratio_right"] if profile else self.CLOSE_RATIO
        )
        self.open_ratio_r = float(
            profile["open_ratio_right"] if profile else self.OPEN_RATIO
        )
        self._since: Optional[float] = None
        self._open_since: Optional[float] = None
        self._gap_since: Optional[float] = None
        self._latched = False
        self.last_left = 1.0
        self.last_right = 1.0

    @property
    def ready(self) -> bool:
        return self._baseline_l is not None and self._baseline_r is not None

    @property
    def baseline_ears(self) -> Optional[tuple[float, float]]:
        if not self.ready:
            return None
        return self._baseline_l, self._baseline_r

    @property
    def active(self) -> bool:
        return self._since is not None or self._latched

    @property
    def awaiting_reopen(self) -> bool:
        return self._latched

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def cancel_hold(self) -> None:
        self._since = None
        self._gap_since = None

    def update(self, ear_left: float, ear_right: float,
               now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if (not np.isfinite(ear_left) or ear_left <= 0
                or not np.isfinite(ear_right) or ear_right <= 0):
            self.cancel_hold()
            return False

        if not self.ready:
            if (ear_left >= self.BASELINE_MIN_EAR
                    and ear_right >= self.BASELINE_MIN_EAR):
                self._baseline_samples_l.append(float(ear_left))
                self._baseline_samples_r.append(float(ear_right))
                if len(self._baseline_samples_l) >= self.BASELINE_SAMPLES:
                    self._baseline_l = float(np.median(self._baseline_samples_l))
                    self._baseline_r = float(np.median(self._baseline_samples_r))
                    self._baseline_samples_l.clear()
                    self._baseline_samples_r.clear()
            return False

        self.last_left = float(ear_left / self._baseline_l)
        self.last_right = float(ear_right / self._baseline_r)
        both_closed = (
            self.last_left <= self.close_ratio_l
            and self.last_right <= self.close_ratio_r
        )
        both_open = (
            self.last_left >= self.open_ratio_l
            and self.last_right >= self.open_ratio_r
        )

        if both_open:
            self._since = None
            self._gap_since = None
            if self._latched:
                if self._open_since is None:
                    self._open_since = now
                elif now - self._open_since >= self.RELEASE_S:
                    self._latched = False
                    self._open_since = None
            else:
                self._baseline_l = (
                    (1.0 - self.BASELINE_ALPHA) * self._baseline_l
                    + self.BASELINE_ALPHA * float(ear_left)
                )
                self._baseline_r = (
                    (1.0 - self.BASELINE_ALPHA) * self._baseline_r
                    + self.BASELINE_ALPHA * float(ear_right)
                )
            return False

        self._open_since = None
        if self._latched:
            self._since = None
            return False

        if not both_closed:
            if self._since is not None:
                if self._gap_since is None:
                    self._gap_since = now
                elif now - self._gap_since > self.AMBIGUOUS_GAP_S:
                    self.cancel_hold()
            return False

        if self._since is None:
            self._since = now
            self._gap_since = None
            return False
        self._gap_since = None
        if now - self._since >= self.HOLD_S:
            self._since = None
            self._latched = True
            return True
        return False


class MouthOpenClickDetector:
    """A sustained, calibrated mouth opening produces one right click."""

    HOLD_S = 1.0
    COOLDOWN_S = 0.8
    AMBIGUOUS_GAP_S = 0.15

    def __init__(self, profile: Optional[dict] = None) -> None:
        self.open_threshold = (
            float(profile["open_threshold"]) if profile
            else MOUTH_OPEN_THRESHOLD
        )
        self.release_threshold = (
            float(profile["release_threshold"]) if profile
            else MOUTH_OPEN_RELEASE_THRESHOLD
        )
        self._since: Optional[float] = None
        self._gap_since: Optional[float] = None
        self._latched = False
        self._last_click = 0.0

    @property
    def active(self) -> bool:
        return self._since is not None

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def reset(self) -> None:
        self._since = None
        self._gap_since = None
        self._latched = False

    def update(self, mouth_open: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if not np.isfinite(mouth_open):
            self.reset()
            return False

        if self._since is None:
            if mouth_open >= self.open_threshold:
                self._since = now
            return False

        if mouth_open >= self.release_threshold:
            self._gap_since = None
        elif not self._latched:
            if self._gap_since is None:
                self._gap_since = now
            elif now - self._gap_since > self.AMBIGUOUS_GAP_S:
                self.reset()
                return False
        else:
            self.reset()
            return False

        if (not self._latched and now - self._since >= self.HOLD_S
                and now - self._last_click >= self.COOLDOWN_S):
            self._latched = True
            self._last_click = now
            return True
        return False


class LatestFrameCamera:
    """Reads the camera on its own thread and always hands out the NEWEST frame, so a slow
    processing step never leaves stale frames queued inside the driver (lag)."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        import threading
        self._cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._seen = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._seen, timeout):
                return False, None
            self._seen = self._seq
            return True, self._frame

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)
        self._cap.release()


class BlinkClickDetector:
    """Legacy classifier-based smile click detector.

    A gesture must be held for HOLD_S, fires once, and must be released before it can fire
    again."""

    HOLD_S = 1.1
    COOLDOWN_S = 0.6
    def __init__(self, profile: Optional[dict] = None) -> None:
        self.profile = profile
        self._since: Optional[float] = None
        self._which: Optional[str] = None
        self._fired = False
        self._last = 0.0

    @property
    def configured(self) -> bool:
        return self.profile is not None

    @property
    def eyes_closed(self) -> bool:
        return self._which is not None

    def reset(self) -> None:
        self._since, self._which, self._fired = None, None, False

    @property
    def hold_progress(self) -> float:
        if self._since is None:
            return 0.0
        return float(np.clip(
            (time.monotonic() - self._since) / self.HOLD_S, 0.0, 1.0
        ))

    def update(self, smile_score: float, smile_threshold: float,
               release_threshold: float) -> Optional[str]:
        if not self.configured:
            return None
        now = time.monotonic()
        if (not np.isfinite(smile_score)
                or not np.isfinite(smile_threshold)
                or not np.isfinite(release_threshold)):
            self.reset()
            return None
        if self._which is None:
            if smile_score >= smile_threshold:
                self._which, self._since, self._fired = "left", now, False
            return None
        if smile_score < release_threshold:
            self.reset()
            return None
        if self._fired or now - self._since < self.HOLD_S:
            return None
        self._fired = True
        if now - self._last < self.COOLDOWN_S:
            return None
        self._last = now
        return "left"
