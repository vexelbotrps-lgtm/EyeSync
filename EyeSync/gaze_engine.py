"""
gaze_engine.py - where the eyes look, from an ordinary webcam.
==============================================================

EyeSync moves the cursor with the head (live_inference.HeadPointer): precise,
but crossing a big screen takes a big head turn. This module adds the eyes:

  * GazeTracker      the eye model: each eye is a ball fixed in the head, and
                     the gaze runs from its centre through the iris.
  * GazeModel        turns that gaze into a screen position, fitted to the
                     user by a 9-point eye calibration (GazeCalibration).
  * GazeJump         "look, then nudge": when the head starts moving the
                     cursor towards where the eyes rest, the cursor jumps
                     there, and the head does the fine aiming from there.

Where it comes from
-------------------
The eye model is the webcam 3D eye tracker of JEOresearch's EyeTracker
(Webcam3DTracker/MonitorTracking.py, MIT licence, Copyright (c) 2024
JEOresearch - see THIRD_PARTY_NOTICES.md), reworked for EyeSync:

  * There, an eyeball centre is locked for each eye while the user looks at
    the screen centre - behind the iris by the eyeball radius - and held in
    a head frame from the nose landmarks (PCA, with axis flips caught
    against a reference). Here the head is a least-squares fit of rigid
    face landmarks (nose, forehead, eye corners, temples) to the ones at
    the lock (``similarity``, Umeyama 1991): no axes to flip, and it
    measures the head's scale, so leaning in or back keeps the eyeballs in
    place.
  * There, gaze angles map to the screen with fixed spans (15 degrees to
    the left/right edge, 5 up/down) and a centre offset; a multi-point
    calibration was on its to-do list. Here a 9-point calibration fits the
    mapping (GazeModel), which also uses where the eyes are in front of the
    camera (a sideways lean moves the gaze point too) and how open the
    eyelids are (they follow the eyes up and down, which the iris alone
    shows poorly).

Why "look, then nudge" rather than moving the cursor with the eyes: webcam
gaze is good to a few degrees - roughly 100-250 px on a laptop screen - which
is not enough to hit a button. MAGIC pointing (Zhai, Morimoto & Ihde,
CHI 1999) showed that gaze for the long jump plus the hand (here: the head)
for the last bit beats either alone, and the cursor doesn't chase every
glance: it jumps only when the head asks it to.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Landmarks: MediaPipe's face mesh with its iris points (478 in all)
# --------------------------------------------------------------------------- #

N_LANDMARKS = 478
IRISES = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))   # centre + 4 on the rim, per eye
# Points that move with the skull, not with expressions: nose bridge and
# tip, forehead, eye corners and temples. The original used the nose alone;
# points spread over the face hold the head's rotation steadier.
RIGID = (1, 4, 5, 6, 168, 197, 195, 8, 9, 151, 10, 33, 133, 362, 263, 127, 356)
_OUTER = (RIGID.index(33), RIGID.index(263))      # outer eye corners: the face's width

EYE_RADIUS_FW = 0.11      # eyeball centre -> iris, in face widths (~10 of ~95 mm)
VIEW_DISTANCE_FW = 6.0    # eyes to screen at the lock (~57 cm), in face widths
MAX_FIT_RMS_FW = 0.05     # a face mesh that fits the head worse than this is a glitch
GOOD_ERROR_FRAC = 0.15    # calibrated error up to this share of the screen width: gaze jump on


def points_px(xyz: np.ndarray, w: int, h: int) -> np.ndarray:
    """MediaPipe's normalised (x, y, z) -> pixels; z (depth, + = away from the camera) on x's scale."""
    return np.asarray(xyz, np.float64)[:, :3] * np.array([w, h, w], np.float64)


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Scale s, rotation R and shift t with dst ~ s * R @ src + t, by least
    squares (Umeyama 1991), and the RMS distance left over.
    """
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    var = float((a * a).sum()) / len(a)
    u, sv, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.array([1.0, 1.0, -1.0 if np.linalg.det(u) * np.linalg.det(vt) < 0 else 1.0])
    rot = u @ np.diag(d) @ vt
    s = float((sv * d).sum()) / var if var > 1e-12 else 1.0
    t = mu_d - s * rot @ mu_s
    rms = float(np.sqrt(np.mean(np.sum((s * src @ rot.T + t - dst) ** 2, axis=1))))
    return s, rot, t, rms


# --------------------------------------------------------------------------- #
# The eye model
# --------------------------------------------------------------------------- #

@dataclass
class GazeSample:
    """One frame's gaze."""
    u: float                 # where the gaze meets the screen's plane (at the camera), in face
    v: float                 # widths from the camera: u + = picture right, v + = up
    lid: float               # eyelid opening (mean eye aspect ratio)
    irises: np.ndarray       # (2, 2) iris centres, camera pixels (for the camera view)
    dirs: np.ndarray         # (2, 3) each eye's gaze direction

    def features(self) -> np.ndarray:
        return np.array([self.u, self.v, self.lid])


class GazeTracker:
    """
    lock() fixes each eyeball's centre in the head while the user looks at
    the middle of the screen; sample() then follows the head and gives the
    gaze of every frame (GazeSample).
    """

    def __init__(self):
        self.ref: np.ndarray | None = None       # RIGID landmarks at the lock, pixels
        self.centres: np.ndarray | None = None   # (2, 3) eyeball centres at the lock, pixels

    @property
    def locked(self) -> bool:
        return self.ref is not None

    @property
    def face_px(self) -> float:
        """Face width (outer eye corners) at the lock, pixels."""
        return float(np.linalg.norm(self.ref[_OUTER[0]] - self.ref[_OUTER[1]]))

    def lock(self, frames: list[np.ndarray], w: int, h: int) -> bool:
        """
        ``frames``: landmark arrays (N, 3) of a still look at the screen
        centre. Each is aligned to the first (the head moves a little) and the
        median taken; each eyeball centre is then the eyeball radius behind its
        iris, away from the camera.
        """
        pts = [points_px(f, w, h) for f in frames if f is not None and len(f) >= N_LANDMARKS]
        if not pts:
            return False
        rigid = list(RIGID)
        base = pts[0][rigid]
        aligned, irises = [], []
        for p in pts:
            s, rot, t, _ = similarity(p[rigid], base)
            q = s * p @ rot.T + t
            aligned.append(q[rigid])
            irises.append([q[list(i)].mean(axis=0) for i in IRISES])
        self.ref = np.median(np.array(aligned), axis=0)
        iris = np.median(np.array(irises), axis=0)
        self.centres = iris + np.array([0.0, 0.0, EYE_RADIUS_FW * self.face_px])
        return True

    def sample(self, xyz: np.ndarray | None, w: int, h: int, lid: float) -> GazeSample | None:
        """This frame's gaze, or None (not locked, no iris points, a glitch)."""
        if self.ref is None or xyz is None or len(xyz) < N_LANDMARKS:
            return None
        p = points_px(xyz, w, h)
        s, rot, t, rms = similarity(self.ref, p[list(RIGID)])
        face = s * self.face_px
        if face < 1.0 or rms > MAX_FIT_RMS_FW * face:
            return None
        centres = s * self.centres @ rot.T + t                  # the eyeballs, where the head is now
        irises = np.array([p[list(i)].mean(axis=0) for i in IRISES])
        rays = irises - centres
        lengths = np.linalg.norm(rays, axis=1)
        radius = EYE_RADIUS_FW * face
        if np.any(lengths < 0.3 * radius) or np.any(lengths > 3.0 * radius):
            return None
        dirs = rays / lengths[:, None]
        g = dirs.mean(axis=0)
        g /= np.linalg.norm(g)
        ahead = max(-g[2], 0.2)                    # towards the camera is -z; ~78 degrees at most
        eyes = centres.mean(axis=0)
        dist = VIEW_DISTANCE_FW * self.face_px / face   # leaning in makes the face bigger
        # The gaze ray from the eyes to the plane of the screen (pinhole camera
        # at the screen): where the eyes are, plus how far the ray travels sideways.
        u = (eyes[0] - w / 2) / face + dist * g[0] / ahead
        v = -(eyes[1] - h / 2) / face - dist * g[1] / ahead
        return GazeSample(float(u), float(v), float(lid), irises[:, :2], dirs)

    def to_dict(self) -> dict:
        return {"points": list(RIGID), "ref": np.round(self.ref, 2).tolist(),
                "centres": np.round(self.centres, 2).tolist()}

    @classmethod
    def from_dict(cls, data) -> "GazeTracker | None":
        try:
            if list(data["points"]) != list(RIGID):
                return None                        # saved by a version with other landmarks
            ref, centres = np.array(data["ref"], np.float64), np.array(data["centres"], np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if (ref.shape != (len(RIGID), 3) or centres.shape != (2, 3)
                or not (np.all(np.isfinite(ref)) and np.all(np.isfinite(centres)))):
            return None
        tracker = cls()
        tracker.ref, tracker.centres = ref, centres
        return tracker if tracker.face_px > 1.0 else None


# --------------------------------------------------------------------------- #
# Gaze -> screen
# --------------------------------------------------------------------------- #

class GazeModel:
    """
    Gaze features (GazeSample.features: u, v, eyelids) -> screen position,
    as fractions of the screen's width and height (a new resolution keeps
    working). Least squares on the calibration points, in one of three forms;
    fit() keeps the one that predicts left-out points best:
      linear      x, y from u, v
      lids        ... and the eyelid opening
      quadratic   x, y from u, v, uv, u^2, v^2 (the classic 9-point eye-tracker fit)
    """

    FORMS = ("linear", "lids", "quadratic")
    RIDGE = 0.01          # a little regularisation: 9 points, each a bit noisy
    BETTER = 0.95         # a more complex form must predict at least 5% better
    MAX_DROPPED = 2       # calibration points that may be dropped as glances away

    def __init__(self, form: str, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray):
        self.form, self.mean, self.scale, self.coef = form, mean, scale, coef

    @staticmethod
    def design(form: str, z: np.ndarray) -> np.ndarray:
        u, v, lid = z[:, 0], z[:, 1], z[:, 2]
        one = np.ones(len(z))
        if form == "linear":
            cols = [one, u, v]
        elif form == "lids":
            cols = [one, u, v, lid]
        else:
            cols = [one, u, v, u * v, u * u, v * v]
        return np.stack(cols, axis=1)

    @classmethod
    def solve(cls, form: str, feats: np.ndarray, targets: np.ndarray) -> "GazeModel":
        mean = feats.mean(axis=0)
        scale = feats.std(axis=0)
        scale[scale < 1e-9] = 1.0
        a = cls.design(form, (feats - mean) / scale)
        reg = cls.RIDGE * len(a) * np.eye(a.shape[1])
        reg[0, 0] = 0.0                            # the offset isn't held back
        coef = np.linalg.solve(a.T @ a + reg, a.T @ targets)
        return cls(form, mean, scale, coef)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """Screen fractions (x, y) for one feature vector, or (n, 2) for n."""
        f = np.atleast_2d(np.asarray(feats, np.float64))
        out = self.design(self.form, (f - self.mean) / self.scale) @ self.coef
        return out[0] if np.ndim(feats) == 1 else out

    @classmethod
    def _loo(cls, form: str, feats: np.ndarray, targets: np.ndarray, screen: np.ndarray) -> np.ndarray:
        """Each point's error (px) when the fit is made without it."""
        errs = []
        for i in range(len(feats)):
            rest = np.arange(len(feats)) != i
            m = cls.solve(form, feats[rest], targets[rest])
            errs.append(float(np.linalg.norm((m.predict(feats[i]) - targets[i]) * screen)))
        return np.array(errs)

    @classmethod
    def fit(cls, feats: np.ndarray, targets: np.ndarray, screen: np.ndarray,
            min_points: int = 6) -> tuple["GazeModel", float, list[int]]:
        """
        The best form, its mean error on left-out points (px: an honest
        accuracy estimate) and which points it used: a point far worse than
        the others (the user looked away) is dropped, up to MAX_DROPPED.
        """
        keep = list(range(len(feats)))
        screen = np.asarray(screen, np.float64)
        while True:
            best = None
            for form in cls.FORMS:
                errs = cls._loo(form, feats[keep], targets[keep], screen)
                if best is None or float(errs.mean()) < cls.BETTER * float(best[1].mean()):
                    best = (form, errs)
            form, errs = best
            worst = int(np.argmax(errs))
            if (len(keep) > min_points and len(feats) - len(keep) < cls.MAX_DROPPED
                    and errs[worst] > 2.5 * float(np.median(errs)) and errs[worst] > 0.1 * screen[0]):
                del keep[worst]
                continue
            return cls.solve(form, feats[keep], targets[keep]), float(errs.mean()), keep

    def to_dict(self) -> dict:
        return {"form": self.form, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, data) -> "GazeModel | None":
        try:
            form = data["form"]
            mean, scale = np.array(data["mean"], np.float64), np.array(data["scale"], np.float64)
            coef = np.array(data["coef"], np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if form not in cls.FORMS or mean.shape != (3,) or scale.shape != (3,):
            return None
        if coef.shape != (cls.design(form, np.zeros((1, 3))).shape[1], 2):
            return None
        if not all(np.all(np.isfinite(a)) for a in (mean, scale, coef)) or np.any(scale <= 0):
            return None
        return cls(form, mean, scale, coef)


# --------------------------------------------------------------------------- #
# Eye calibration: 9 dots
# --------------------------------------------------------------------------- #

GAZE_POINTS = ((0.5, 0.5), (0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.9, 0.5),
               (0.9, 0.9), (0.5, 0.9), (0.1, 0.9), (0.1, 0.5))      # screen fractions
GAZE_STEPS = tuple(f"gaze{i + 1}" for i in range(len(GAZE_POINTS)))  # Snapshot.calib_step


@dataclass
class GazeFrame:
    """What the eye calibration needs from one camera frame."""
    xyz: np.ndarray | None   # face landmarks (N, 3) in MediaPipe's normalised coordinates
    w: int
    h: int
    lid: float               # mean eye aspect ratio
    eyes_open: bool


class GazeCalibration:
    """
    ~18 s: a dot visits 9 places on the screen (main_window draws it) and the
    user just looks at it - moving the head is fine, the model follows the
    head. At each place it waits SETTLE_S for the eyes to land, then takes
    COLLECT_S of open-eye frames. The first place is the centre, where the
    eyeballs are locked into the head (GazeTracker.lock). A place not done
    within POINT_TIMEOUT_S (eyes closed, face lost) is skipped; the fit needs
    the centre and MIN_POINTS places in all.

    Same interface as live_inference.Calibration for the vision worker and
    the screen guide: step, finished, text(), progress(), update(), skipped.
    """

    SETTLE_S = 0.9
    COLLECT_S = 1.0
    MIN_FRAMES = 12
    POINT_TIMEOUT_S = 8.0
    MIN_POINTS = 6
    TEXT_CENTRE = "Look at the dot in the middle - just look, keep your head natural"
    TEXT = "Follow the dot with your eyes"

    def __init__(self, now: float, screen_w: int, screen_h: int):
        self.screen = np.array([screen_w, screen_h], np.float64)
        self.tracker = GazeTracker()
        self.index = 0
        self.point_since = now
        self.collect_since: float | None = None
        self.raw: list[tuple[np.ndarray, float]] = []       # centre: (landmarks, lid) until the lock
        self.frames: list[np.ndarray] = []                  # this place's features, per frame
        self.points: list[tuple[int, np.ndarray]] = []      # (place, its features (n, 3))
        self.skipped: list[str] = []
        self.failed = ""
        self.face = True
        self.learned: list = []      # (live_inference.Calibration's open-eye frames: none here)

    @property
    def step(self) -> str:
        return GAZE_STEPS[self.index] if not self.finished else "done"

    @property
    def finished(self) -> bool:
        return self.index >= len(GAZE_POINTS)

    def text(self) -> str:
        if self.finished:
            return ""
        prefix = "" if self.face else "Can't see your face - "
        return prefix + (self.TEXT_CENTRE if self.index == 0 else self.TEXT)

    def progress(self, now: float) -> float:
        if self.collect_since is None or self.finished:
            return 0.0
        return min(1.0, (now - self.collect_since) / self.COLLECT_S)

    def _next(self, now: float) -> None:
        self.index += 1
        self.point_since, self.collect_since = now, None
        self.frames, self.raw = [], []

    def update(self, f: GazeFrame | None, now: float) -> None:
        """Feed one frame (None: no face)."""
        if self.finished:
            return
        self.face = f is not None and f.xyz is not None
        if now - self.point_since > self.POINT_TIMEOUT_S:
            if self.index == 0:
                self.failed = "it couldn't see your eyes at the middle dot"
                self.index = len(GAZE_POINTS)    # nothing works without the lock
                return
            self.skipped.append(f"dot {self.index + 1}")
            self._next(now)
            return
        if not self.face or not f.eyes_open or now - self.point_since < self.SETTLE_S:
            return
        if self.index == 0:
            self.raw.append((np.array(f.xyz, np.float64), f.lid))
            n = len(self.raw)
        else:
            s = self.tracker.sample(f.xyz, f.w, f.h, f.lid)
            if s is None:
                return
            self.frames.append(s.features())
            n = len(self.frames)
        if self.collect_since is None:
            self.collect_since = now
        if n < self.MIN_FRAMES or now - self.collect_since < self.COLLECT_S:
            return
        if self.index == 0:
            if not self.tracker.lock([x for x, _ in self.raw], f.w, f.h):
                return
            samples = [self.tracker.sample(x, f.w, f.h, lid) for x, lid in self.raw]
            self.frames = [s.features() for s in samples if s is not None]
            if len(self.frames) < self.MIN_FRAMES // 2:
                self.tracker = GazeTracker()     # a jumpy lock: try the centre again
                self.raw, self.frames, self.collect_since = [], [], None
                return
        self.points.append((self.index, np.array(self.frames)))
        self._next(now)

    def results(self) -> dict:
        """What to save; "failed" says why when there is nothing to use."""
        res: dict = {"date": time.strftime("%Y-%m-%d")}
        if self.failed:
            return {**res, "failed": self.failed}
        if len(self.points) < self.MIN_POINTS:
            return {**res, "failed": f"only {len(self.points)} of {len(GAZE_POINTS)} dots could "
                                     "be measured (eyes closed or face not seen)"}
        feats = np.array([np.median(fr, axis=0) for _, fr in self.points])
        targets = np.array([GAZE_POINTS[i] for i, _ in self.points], np.float64)
        model, error, keep = GazeModel.fit(feats, targets, self.screen, self.MIN_POINTS)
        # How much the gaze point wanders frame to frame while looking at one
        # spot (px): GazeJump tells a resting look from a moving one by it.
        noise = float(np.median([
            np.median(np.linalg.norm((model.predict(fr) - model.predict(np.median(fr, axis=0)))
                                     * self.screen, axis=1))
            for _, fr in self.points]))
        return {**res, "tracker": self.tracker.to_dict(), "model": model.to_dict(),
                "error_px": round(error), "noise_px": round(noise), "points": len(keep)}


def calibration_summary(res: dict, skipped: list[str], screen_w: int) -> str:
    """One or two sentences for the screen when the eye calibration ends."""
    if "failed" in res:
        return f"Eye calibration didn't work: {res['failed']}. Try again with your face well lit."
    err = int(res["error_px"])
    text = f"Eye calibration done! Your gaze is known to about {err} px."
    if err <= GOOD_ERROR_FRAC * screen_w:
        text += " Gaze jump is on: look at a spot and nudge your head towards it."
    else:
        text += (" That's too rough for gaze jump, so it stays off - more light on your face "
                 "or sitting closer usually helps.")
    if skipped:
        text += f" Skipped: {', '.join(skipped)}."
    return text


# --------------------------------------------------------------------------- #
# Look, then nudge
# --------------------------------------------------------------------------- #

class GazeJump:
    """
    "Look, then nudge" (conservative MAGIC pointing). Fed the gaze point of
    every frame, it keeps track of where the eyes rest. When the head starts
    moving the cursor (CursorMotion leaves its rest), jump() says whether -
    and where - the cursor jumps:
      * the eyes have rested on one spot (their points within ``spread_px``)
        for the last FIX_S, until just now (not across a blink);
      * that spot is far from the cursor - more than ``min_px``: nearer, the
        head is quicker and more precise than the gaze;
      * the head is moving the cursor towards it (within MAX_ANGLE_DEG).
    After a jump the rest of that head movement is absorbed (absorbing()),
    so the cursor doesn't shoot past the spot. Reading, glancing around and
    keeping the head still never move the cursor.
    """

    FIX_S = 0.3
    FIX_MIN = 5              # gaze points needed in FIX_S
    STALE_S = 0.2            # the newest at most this old
    MAX_ANGLE_DEG = 45.0
    COOLDOWN_S = 0.8
    ABSORB_MAX_S = 0.5
    ABSORB_STILL = 0.35      # head speed (face widths/s) that counts as stopped
    MIN_JUMP_FRAC = 0.12     # min_px: at least this share of the screen width ...
    MIN_JUMP_ERRORS = 2.5    # ... and this many times the calibrated error
    MIN_SPREAD_FRAC = 0.03   # spread_px: at least this share of the screen width ...
    SPREAD_NOISES = 2.5      # ... and this many times the calibrated frame-to-frame noise

    def __init__(self, screen_w: int, screen_h: int, error_px: float, noise_px: float):
        self.size = np.array([screen_w, screen_h], np.float64)
        self.min_px = max(self.MIN_JUMP_ERRORS * error_px, self.MIN_JUMP_FRAC * screen_w)
        self.spread_px = max(self.SPREAD_NOISES * noise_px, self.MIN_SPREAD_FRAC * screen_w)
        self.points: deque[tuple[float, np.ndarray]] = deque(maxlen=30)
        self.last_jump = -1e9
        self.absorb_since: float | None = None
        self.still = 0

    def resize(self, screen_w: int, screen_h: int) -> None:
        """The display changed resolution: the same distances, in the new pixels."""
        k = screen_w / self.size[0]
        self.size = np.array([screen_w, screen_h], np.float64)
        self.min_px *= k
        self.spread_px *= k
        self.points.clear()

    def add(self, gaze_px: np.ndarray | None, now: float) -> None:
        """This frame's gaze point (screen px), or None (eyes closed, no face)."""
        if gaze_px is not None:
            self.points.append((now, np.asarray(gaze_px, np.float64)))

    def fixation(self, now: float) -> np.ndarray | None:
        """Where the eyes rest right now (screen px), or None."""
        if not self.points or now - self.points[-1][0] > self.STALE_S:
            return None
        recent = np.array([p for t, p in self.points if now - t <= self.FIX_S])
        if len(recent) < self.FIX_MIN:
            return None
        centre = np.median(recent, axis=0)
        if float(np.median(np.linalg.norm(recent - centre, axis=1))) > self.spread_px:
            return None
        return centre

    def jump(self, cursor: np.ndarray, head_dir: np.ndarray, now: float) -> np.ndarray | None:
        """The head starts moving the cursor (``head_dir``, screen px): the place to jump to, or None."""
        if now - self.last_jump < self.COOLDOWN_S:
            return None
        spot = self.fixation(now)
        if spot is None:
            return None
        gap = spot - np.asarray(cursor, np.float64)
        dist, push = float(np.linalg.norm(gap)), float(np.linalg.norm(head_dir))
        if dist < self.min_px or push < 1e-6:
            return None
        if float(np.dot(gap, head_dir)) / (dist * push) < math.cos(math.radians(self.MAX_ANGLE_DEG)):
            return None
        self.last_jump, self.absorb_since, self.still = now, now, 0
        return np.clip(spot, 0, self.size - 1)

    def absorbing(self, head_speed: float, now: float) -> bool:
        """True while the head's nudge that made a jump is still going (the cursor stays put)."""
        if self.absorb_since is None:
            return False
        self.still = self.still + 1 if head_speed < self.ABSORB_STILL else 0
        if self.still >= 2 or now - self.absorb_since > self.ABSORB_MAX_S:
            self.absorb_since = None
            return False
        return True
