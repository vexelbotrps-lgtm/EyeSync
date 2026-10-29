"""Make small public GRID eye-region review sheets for closure-event labelling.

The source videos are public CC BY 4.0 material. Sheets are temporary review
artifacts, not application training data to ship. A JSON sidecar maps every
 tile to its numerical event, video and frame; no unidentified image is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))

import dataset_report  # noqa: E402
import face_engine as fe  # noqa: E402

log = logging.getLogger("eyesync-crops")
EYE_INDICES = set()
for _spec in fe.EYE_IRIS_IDX.values():
    for _key in ("corners", "upper", "lower", "centre", "ring"):
        _value = _spec[_key]
        EYE_INDICES.update((_value,) if isinstance(_value, int) else _value)


def choose(events: list[dict], count: int) -> list[tuple[str, int, dict]]:
    groups = [
        ("C", [i for i, e in enumerate(events) if e["click"]]),
        ("N", [i for i, e in enumerate(events)
               if not e["click"] and not e["cancelled"] and e["duration"] < 0.25]),
    ]
    selected = []
    # Twelve false-click candidates and four natural short blinks per 16-tile sheet.
    wanted = (min(12, count), min(4, count))
    for (label, indices), number in zip(groups, wanted):
        if not indices or not number:
            continue
        positions = np.unique(np.linspace(0, len(indices) - 1, min(number, len(indices))).round().astype(int))
        selected.extend((label, indices[p], events[indices[p]]) for p in positions)
    return selected[:16]


def locate(meta: dict, row_index: int) -> tuple[dict, int]:
    cursor = 0
    gap_rows = 1 if meta.get("gap_s", 0.35) > 0 else 0
    for video in meta["videos"]:
        frames = int(video["frames"])
        if cursor <= row_index < cursor + frames:
            return video, row_index - cursor
        cursor += frames + gap_rows
    raise IndexError(f"row {row_index} is outside the recorded videos")


def eye_crop(frame: np.ndarray, mesh: fe.FaceMeshBackend, timestamp: int) -> np.ndarray:
    h, w = frame.shape[:2]
    landmarks = mesh.process(frame, timestamp)
    if landmarks is None:
        return frame[:max(1, int(h * 0.65)), :]
    points = landmarks[sorted(EYE_INDICES)] * np.array([w, h])
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    ew, eh = x1 - x0, y1 - y0
    left = max(0, int(x0 - 0.45 * ew))
    right = min(w, int(x1 + 0.45 * ew))
    top = max(0, int(y0 - 1.25 * eh))
    bottom = min(h, int(y1 + 1.65 * eh))
    return frame[top:bottom, left:right] if right > left and bottom > top else frame


def make(features: Path, report_path: Path, videos: Path, output_dir: Path) -> dict:
    data, columns, meta = dataset_report.load(features)
    report = json.loads(report_path.read_text(encoding="utf-8"))[0]
    events = report["events"]
    selected = choose(events, 12)
    if not selected:
        raise RuntimeError("no reviewable closure events")
    mesh = fe.FaceMeshBackend()
    tiles = []
    labels = []
    try:
        for tile_index, (group, event_id, event) in enumerate(selected):
            a, b = event["start"], event["end"] + 1
            score_columns = [columns[name] for name in ("bs_eyeBlinkLeft", "bs_eyeBlinkRight")
                             if name in columns]
            scores = data[a:b, score_columns]
            peak_offset = int(np.nanargmax(np.nanmean(scores, axis=1))) if scores.size else 0
            row_index = min(a + peak_offset, b - 1)
            video, frame_index = locate(meta, row_index)
            cap = cv2.VideoCapture(str(videos / video["name"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                log.warning(f"could not read {video['name']} frame {frame_index}")
                continue
            crop = eye_crop(frame, mesh, 1000 + tile_index * 1000)
            crop = cv2.resize(crop, (360, 140), interpolation=cv2.INTER_AREA)
            tile = np.zeros((180, 360, 3), dtype=np.uint8)
            tile[:140] = crop
            label = f"{tile_index}: {group}{event_id}  {event['duration']:.2f}s  {video['name']}"
            cv2.putText(tile, label, (6, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(tile, f"frame {frame_index}  GRID CC BY 4.0", (6, 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            tiles.append(tile)
            labels.append({"tile": tile_index, "group": group, "event_id": event_id,
                           "video": video["name"], "frame": frame_index,
                           "duration": event["duration"], "click": event["click"]})
    finally:
        mesh.close()
    while len(tiles) < 16:
        tiles.append(np.zeros((180, 360, 3), dtype=np.uint8))
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, 16, 4)]
    sheet = np.concatenate(rows, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{features.stem}-events.jpg"
    cv2.imwrite(str(image_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    sidecar = {"dataset": "GRID", "license": "CC BY 4.0",
               "attribution": "Cooke, Barker, Cunningham & Shao (2006)",
               "source": "https://zenodo.org/records/3625687",
               "features": features.name, "labels": labels}
    (output_dir / f"{features.stem}-events.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return {"image": str(image_path), "tiles": len(labels)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(make(args.features, args.report, args.videos, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
