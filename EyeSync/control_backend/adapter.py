"""Use the calibrated gaze and facial click controls from the desktop project.

The website EyeSync owns the camera and face mesh. This adapter consumes those
same landmarks, so voice typing and the dashboard continue to use one camera.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import app_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaze_core as gc  # noqa: E402; also resolves classes in the saved joblib model


class ControlBackend:
    def __init__(self, screen_width: int, screen_height: int):
        personal_model = app_paths.DATA_DIR / "gaze_model.joblib"
        self.bundle = gc.load_model_bundle(
            personal_model if personal_model.is_file() else gc.MODEL_PATH)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.features = gc.VectorEma(gc.EMA_ALPHA)
        self.smoother = gc.OneEuroFilter2D(1.0, 0.01)
        self.smile = gc.LandmarkSmileClickDetector()
        self.mouth = gc.MouthOpenClickDetector(gc.load_mouth_open_profile())
        self.last_target = None

    def reset(self):
        self.smile.reset()
        self.mouth.reset()
        self.features = gc.VectorEma(gc.EMA_ALPHA)
        self.smoother = gc.OneEuroFilter2D(1.0, 0.01)

    def update(self, landmarks, width: int, height: int, *, click_enabled: bool,
               voice_active: bool, now: float | None = None):
        """Return (screen target, left/right click, gesture hold)."""
        now = time.monotonic() if now is None else now
        face = np.asarray(landmarks[:468], dtype=np.float32)
        bounds_min = np.min(face[:, :2], axis=0)
        bounds_max = np.max(face[:, :2], axis=0)
        box = np.clip(np.array([*bounds_min, *bounds_max], dtype=np.float32), 0, 1)

        # These are the same image-plane measurements as gaze_core's
        # FaceFeatureExtractor, using the mesh EyeSync has already computed.
        px = face[:, :2] * np.array([width, height], dtype=np.float32)
        eye_a = (px[33] + px[133]) / 2
        eye_b = (px[362] + px[263]) / 2
        iod = float(np.linalg.norm(eye_a - eye_b)) + 1e-6
        smile_width = float(abs(px[61, 0] - px[291, 0]) / iod)
        mouth_open = float(abs(px[13, 1] - px[14, 1]) / iod)

        click = None
        if click_enabled:
            # Keep smile left click available to press the voice bar's Stop
            # button. Talking opens the mouth, so suppress right click while
            # the microphone is on.
            if voice_active:
                self.mouth.reset()
                right = False
            else:
                right = self.mouth.update(mouth_open, now)
            left = self.smile.update(smile_width, mouth_open,
                                     self.mouth.release_threshold, now)
            if self.mouth.active:
                self.smile.reset()
                left = False
            click = "right" if right else "left" if left else None
        else:
            self.smile.reset()
            self.mouth.reset()
        held = self.smile.active or self.mouth.active
        if held:
            return self.last_target, click, True

        filtered = self.features.update(box)
        x, y = gc.predict_xy(self.bundle, filtered)
        x *= self.screen_width / self.bundle["screen_w"]
        y *= self.screen_height / self.bundle["screen_h"]
        self.last_target = np.asarray(self.smoother.update(x, y, now), dtype=np.float64)
        return self.last_target, click, False
