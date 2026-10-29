"""Tests for the public-dataset tools (the tools are not shipped in the exe)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))

import dataset_report
import event_crops
import face_engine as fe
import grid_dataset
import timing_study
import video_features


class IrisApertureTests(unittest.TestCase):
    def landmarks(self):
        w, h = 100, 100
        points = np.zeros((478, 2), dtype=np.float64)
        for spec in fe.EYE_IRIS_IDX.values():
            def put(indices, values):
                points[list(indices)] = np.asarray(values) / (w, h)

            xs = np.linspace(10, 30, len(spec["upper"]))
            put(spec["upper"], np.column_stack((xs, np.full(len(xs), 5))))
            put(spec["lower"], np.column_stack((xs, np.full(len(xs), 15))))
            put(spec["corners"], ((10, 10), (30, 10)))
            put((spec["centre"],), ((20, 10),))
            put(spec["ring"], ((24, 10), (20, 14), (16, 10), (20, 6)))
        return points

    def test_open_downcast_and_closed_geometry(self):
        points = self.landmarks()
        for eye in fe.iris_apertures(points, 100, 100):
            visible, down, radius, opening = eye
            self.assertAlmostEqual(visible, 1.0)
            self.assertAlmostEqual(down, 0.0, places=5)
            self.assertAlmostEqual(radius, 0.2)
            self.assertAlmostEqual(opening, 0.5)

        spec = fe.EYE_IRIS_IDX["right"]
        points[spec["centre"]] = (0.20, 0.18)
        points[list(spec["ring"])] = np.array(((24, 18), (20, 22), (16, 18), (20, 14))) / 100
        visible, down, _, _ = fe.iris_apertures(points, 100, 100)[1]
        self.assertLess(visible, 0.2)
        self.assertGreater(down, 0.3)

        for spec in fe.EYE_IRIS_IDX.values():
            points[list(spec["upper"]), 1] = 0.10
            points[list(spec["lower"]), 1] = 0.10
        for eye in fe.iris_apertures(points, 100, 100):
            self.assertEqual(eye[3], 0.0)

    def test_mesh_without_irises_is_safe(self):
        for eye in fe.iris_apertures(np.zeros((468, 2)), 100, 100):
            self.assertTrue(all(np.isnan(eye)))


class DatasetReportTests(unittest.TestCase):
    def write(self, path: Path, closed_ranges=(), gap=False):
        rows = []
        closed_frames = {i for a, b in closed_ranges for i in range(a, b)}
        total = 90
        for i in range(total):
            closed = i in closed_frames
            face = not (gap and i == 45)
            ear = 0.07 if closed else 0.28
            score = 0.9 if closed else 0.05
            rows.append((1 / 30, i / 30, float(face), ear, ear, 0, 0, 200, 1,
                         0, 0, 0, 0, 0, *([0.0] * 8), score, score, 0, 0, 0, 0, 0))
        columns = [*video_features.BASE_COLUMNS, "bs_eyeBlinkLeft", "bs_eyeBlinkRight",
                   "bs_jawOpen", "bs_mouthSmileLeft", "bs_mouthSmileRight",
                   "bs_mouthLeft", "bs_mouthRight"]
        meta = {"format": 1, "frames": total, "eyesync_version": "test"}
        with path.open("wb") as stream:
            np.savez_compressed(stream, data=np.asarray(rows, dtype=np.float32),
                                columns=np.array(columns), meta=np.array(json.dumps(meta)))

    def test_deliberate_and_natural_closures_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.npz"
            self.write(path, ((32, 47), (70, 75)))
            result = dataset_report.report(path, include_events=True)
        self.assertEqual(result["clicks"], 1)
        self.assertEqual(result["right"], 0)
        self.assertEqual(result["gestures"], {})
        self.assertEqual(result["notches"], 0)
        self.assertEqual([event["click"] for event in result["events"]], ["left", ""])

    def test_timing_override_and_public_appearance_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.npz"
            self.write(path, ((32, 47), (70, 75)))
            delayed = dataset_report.report(path, blink_min_s=0.6)
            probe = timing_study.recall(path, 0.6, hold_s=0.65)
        self.assertEqual(delayed["clicks"], 0)       # the original closure was only 0.5 s
        self.assertEqual(delayed["blink_min_s"], 0.6)
        self.assertEqual((probe["passed"], probe["tested"]), (2, 2))

    def test_a_video_cut_cancels_a_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cut.npz"
            self.write(path, ((42, 50),), gap=True)
            result = dataset_report.report(path, include_events=True)
        self.assertEqual(result["clicks"], 0)
        self.assertTrue(result["events"][0]["cancelled"])


class EventReviewTests(unittest.TestCase):
    def test_selection_is_balanced_and_row_mapping_skips_gaps(self):
        events = []
        for i in range(30):
            events.append({"click": "left", "cancelled": False, "duration": 0.5})
            events.append({"click": "", "cancelled": False, "duration": 0.2})
        selected = event_crops.choose(events, 12)
        self.assertLessEqual(len(selected), 16)
        self.assertEqual(sum(group == "C" for group, _, _ in selected), 12)
        self.assertEqual(sum(group == "N" for group, _, _ in selected), 4)
        meta = {"gap_s": 0.35, "videos": [{"name": "a", "frames": 5},
                                           {"name": "b", "frames": 5}]}
        self.assertEqual(event_crops.locate(meta, 0), (meta["videos"][0], 0))
        self.assertEqual(event_crops.locate(meta, 4), (meta["videos"][0], 4))
        self.assertEqual(event_crops.locate(meta, 6), (meta["videos"][1], 0))
        with self.assertRaises(IndexError):
            event_crops.locate(meta, 5)


class GridDownloadTests(unittest.TestCase):
    def test_selection_covers_the_whole_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "s1.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for i in range(10):
                    z.writestr(f"video/{i}.mpg", bytes([i]))
            count = grid_dataset.extract_selected(archive, root / "videos", 3)
            names = sorted(p.name for p in (root / "videos").iterdir())
        self.assertEqual(count, 3)
        self.assertEqual(len(names), 3)
        self.assertTrue(names[0].endswith("0.mpg"))
        self.assertTrue(names[-1].endswith("9.mpg"))


if __name__ == "__main__":
    unittest.main()
