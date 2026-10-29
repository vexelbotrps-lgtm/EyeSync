"""Development/held-out evaluation of calibration-derived blink timing.

False clicks come from ordinary GRID readers. Deliberate-recall probes reuse
real, clearly closed public blink onsets and extend the closed portion to a
known hold; they test detector sensitivity, never intent recognition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import dataset_report
import live_inference as li


def signals(row: np.ndarray, columns: dict[str, int]):
    ear = (float(row[columns["ear_left"]]), float(row[columns["ear_right"]]))
    score_pair = (float(row[columns["bs_eyeBlinkLeft"]]),
                  float(row[columns["bs_eyeBlinkRight"]]))
    scores = score_pair if all(math.isfinite(v) for v in score_pair) else None
    pose_pair = (float(row[columns["yaw"]]), float(row[columns["pitch"]]))
    pose = pose_pair if all(math.isfinite(v) for v in pose_pair) else None
    return ear, scores, pose


def eligible_events(data: np.ndarray, columns: dict[str, int], events: list[dict]) -> list[tuple[int, int]]:
    """Clearly closed two-eye events: (event index, deepest frame)."""
    eligible = []
    for event_id, event in enumerate(events):
        if event["cancelled"] or event["kind"] != "both":
            continue
        a, b = event["start"], event["end"] + 1
        rows = data[a:b]
        left, right = rows[:, columns["ear_left"]], rows[:, columns["ear_right"]]
        sl = rows[:, columns["bs_eyeBlinkLeft"]]
        sr = rows[:, columns["bs_eyeBlinkRight"]]
        closed = (np.maximum(left, right) < 0.16) & (np.minimum(sl, sr) > 0.45)
        if not closed.any():
            continue
        quality = np.minimum(sl, sr) - np.maximum(left, right)
        quality[~closed] = -np.inf
        eligible.append((event_id, a + int(np.argmax(quality))))
    return eligible


def open_rows(data: np.ndarray, columns: dict[str, int], events: list[dict], count: int = 90) -> np.ndarray:
    allowed = (data[:, columns["face"]] > 0.5)
    allowed &= np.isfinite(data[:, columns["ear_left"]]) & np.isfinite(data[:, columns["ear_right"]])
    allowed &= (data[:, columns["ear_left"]] > 0.12) & (data[:, columns["ear_right"]] > 0.12)
    for event in events:
        allowed[max(0, event["start"] - 3):event["end"] + 4] = False
    indices = np.flatnonzero(allowed)
    if len(indices) < 30:
        raise RuntimeError("not enough open-eye frames for a public-appearance probe")
    positions = np.linspace(0, len(indices) - 1, min(count, len(indices))).round().astype(int)
    return data[indices[positions]]


def one_probe(baseline: np.ndarray, event_rows: np.ndarray, peak: int, start: int,
              columns: dict[str, int], minimum: float, hold_s: float) -> bool:
    detector = li.BlinkDetector(None, 0.55, 0.70, minimum, li.BLINK_MAX_S,
                                0.6, li.BLINK_RIGHT_S)
    now = 1000.0
    dt = 1.0 / 30.0
    clicks = []

    def feed(row):
        nonlocal now
        now += dt
        ear, scores, pose = signals(row, columns)
        # A stable explicit nose prevents poses sampled from different clips
        # from being mistaken for a fast physical head movement.
        click = detector.update(*ear, now, scores, True, pose, np.zeros(2))[1]
        if click:
            clicks.append(click)

    for row in baseline:
        feed(row)
    onset = event_rows[:peak - start + 1]
    for row in onset:
        feed(row)
    peak_row = event_rows[peak - start]
    # The 3-frame median can delay recognition beyond the event's onset rows.
    # First sustain the real closed appearance until the detector starts (with
    # a bounded wait for malformed probes), then hold from ITS closed_since.
    wait_started = now
    while not clicks and detector.closed_since is None and now - wait_started < 0.5:
        feed(peak_row)
    while not clicks and detector.closed_since is not None and now - detector.closed_since < hold_s:
        feed(peak_row)
    # Release after the requested hold (or immediately after the audible tick).
    for row in event_rows[peak - start + 1:]:
        feed(row)
    for row in baseline[:12]:
        feed(row)
    return clicks == ["left"]


def recall(path: Path, minimum: float, hold_s: float = 0.65, limit: int = 80) -> dict:
    data, columns, _ = dataset_report.load(path)
    baseline_report = dataset_report.report(path, include_events=True)
    eligible = eligible_events(data, columns, baseline_report["events"])
    if len(eligible) > limit:
        positions = np.linspace(0, len(eligible) - 1, limit).round().astype(int)
        eligible = [eligible[i] for i in positions]
    baseline = open_rows(data, columns, baseline_report["events"])
    passed = 0
    misses = []
    for event_id, peak in eligible:
        event = baseline_report["events"][event_id]
        rows = data[event["start"]:event["end"] + 1]
        if one_probe(baseline, rows, peak, event["start"], columns, minimum, hold_s):
            passed += 1
        else:
            misses.append(event_id)
    return {"speaker": path.stem, "tested": len(eligible), "passed": passed,
            "recall": passed / max(1, len(eligible)), "misses": misses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--min", dest="minima", type=float, nargs="+", required=True)
    parser.add_argument("--hold", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"hold_s": args.hold, "note": "extended real closures test sensitivity, not intent",
              "candidates": []}
    for minimum in args.minima:
        speakers = [recall(path, minimum, args.hold, args.limit) for path in args.features]
        tested = sum(s["tested"] for s in speakers)
        passed = sum(s["passed"] for s in speakers)
        row = {"blink_min_s": minimum, "tested": tested, "passed": passed,
               "recall": passed / max(1, tested), "speakers": speakers}
        result["candidates"].append(row)
        print(f"min={minimum:.2f}: {passed}/{tested} = {row['recall']:.1%}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
