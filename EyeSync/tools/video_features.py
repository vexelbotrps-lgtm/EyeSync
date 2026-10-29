"""Extract compact EyeSync measurements from public videos, without images.

The output NPZ contains only per-frame numbers, timestamps and column names.
It never contains pictures, audio, settings or API keys. ``dataset_report.py``
replays those numbers through the application's own gesture/blink classes.

Example::

    python tools/video_features.py /datasets/grid/s1 --max-videos 500 -o s1.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import face_engine as fe  # noqa: E402
import site_access  # noqa: E402

log = logging.getLogger("eyesync-dataset")
VIDEO_SUFFIXES = {".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".mkv"}
BASE_COLUMNS = (
    "dt", "t", "face",
    "ear_left", "ear_right", "head_x", "head_y", "face_px", "winks_ok",
    "mar", "mouth_shift", "roll", "yaw", "pitch",
    "iris_left_visible", "iris_left_down", "iris_left_radius", "iris_left_opening",
    "iris_right_visible", "iris_right_down", "iris_right_radius", "iris_right_opening",
)


class FeatureExtractor:
    def __init__(self, gap_s: float = 0.35):
        self.gap_s = gap_s
        self.mesh = fe.FaceMeshBackend()
        self.low_light = fe.LowLight()
        self.rows: list[list[float]] = []
        self.blend_names: list[str] = []
        self.videos: list[dict] = []
        self.frames = 0
        self.faces = 0

    def close(self) -> None:
        self.mesh.close()

    def _row(self, dt: float, t: float, face: bool, values=(), scores=None) -> None:
        if scores and not self.blend_names:
            self.blend_names = sorted(scores)
        row = [dt, t, float(face), *values]
        row.extend(float("nan") if scores is None else scores.get(name, float("nan"))
                   for name in self.blend_names)
        self.rows.append(row)
        self.frames += face
        self.faces += face

    def video(self, path: Path, offset: float) -> tuple[float, float]:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.warning(f"cannot open {path}; skipped")
            cap.release()
            return offset, 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= fps <= 240.0:
            fps = 25.0
        count = found = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = offset + count / fps
                count += 1
                landmarks = self.low_light.process(self.mesh, frame, int(round(t * 1000)))
                if landmarks is None:
                    self._row(1.0 / fps, t, False)
                    continue
                found += 1
                h, w = frame.shape[:2]
                left, right = fe.eye_aspect_ratios(landmarks, w, h)
                nose, face_px = fe.nose_pointer(landmarks, w, h)
                pose = self.mesh.last_pose
                yaw, pitch = pose if pose is not None else (float("nan"), float("nan"))
                iris = fe.iris_apertures(landmarks, w, h)
                values = (
                    left, right, float(nose[0] / max(face_px, 1.0)),
                    float(nose[1] / max(face_px, 1.0)), face_px,
                    float(fe.eyes_level_ratio(landmarks, w, h) > 0.7),
                    fe.mouth_aspect_ratio(landmarks, w, h, pose),
                    fe.mouth_shift(landmarks, w, h), fe.head_roll(landmarks, w, h),
                    yaw, pitch, *iris[0], *iris[1],
                )
                self._row(1.0 / fps, t, True, values, self.mesh.last_blendshapes)
        finally:
            cap.release()
        duration = count / fps
        self.videos.append({"name": path.name, "frames": count, "faces": found,
                            "fps": fps, "seconds": duration})
        if self.gap_s:
            # A video cut is not a several-second eye closure. A no-face row
            # resets temporal state; dt=0 means the gap is not counted as video.
            self._row(0.0, offset + duration + self.gap_s / 2, False)
        return offset + duration + self.gap_s, duration

    def save(self, output: Path) -> dict:
        columns = [*BASE_COLUMNS, *(f"bs_{name}" for name in self.blend_names)]
        data = np.full((len(self.rows), len(columns)), np.nan, dtype=np.float32)
        for i, row in enumerate(self.rows):
            data[i, :len(row)] = row
        seconds = sum(v["seconds"] for v in self.videos)
        meta = {
            "format": 1,
            "gap_s": self.gap_s,
            "eyesync_version": site_access.VERSION,
            "created_unix": time.time(),
            "seconds": seconds,
            "frames": int(sum(v["frames"] for v in self.videos)),
            "faces": int(sum(v["faces"] for v in self.videos)),
            "videos": self.videos,
            "contains_images": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, data=data, columns=np.array(columns),
                                meta=np.array(json.dumps(meta, separators=(",", ":"))))
        tmp.replace(output)
        return {key: value for key, value in meta.items() if key != "videos"}


def video_paths(inputs: list[str], limit: int) -> list[Path]:
    paths: list[Path] = []
    for text in inputs:
        path = Path(text)
        if path.is_dir():
            paths.extend(p for p in sorted(path.rglob("*")) if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            paths.append(path)
    return paths[:limit] if limit > 0 else paths


def extract(inputs: list[str], output: Path, limit: int = 0, gap_s: float = 0.35) -> dict:
    paths = video_paths(inputs, limit)
    if not paths:
        raise RuntimeError("no video files found")
    extractor = FeatureExtractor(gap_s)
    offset = 0.0
    started = time.monotonic()
    try:
        for i, path in enumerate(paths, 1):
            offset, _ = extractor.video(path, offset)
            if i % 25 == 0 or i == len(paths):
                log.info(f"{i}/{len(paths)} videos, {len(extractor.rows):,} rows, "
                         f"{time.monotonic() - started:.0f}s elapsed")
        return extractor.save(output)
    finally:
        extractor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="video files or folders")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all")
    parser.add_argument("--gap", type=float, default=0.35, help="no-face gap between clips")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = extract(args.inputs, args.output, args.max_videos, args.gap)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
