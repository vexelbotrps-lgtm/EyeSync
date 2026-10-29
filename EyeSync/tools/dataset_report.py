"""Replay video_features.py numbers through EyeSync's own temporal classes.

People in the supported public videos are not operating EyeSync. Therefore
any click, gesture or scroll in their report is a false positive. No images
are needed for the replay and none are written by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import live_inference as li  # noqa: E402


def load(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        data = np.asarray(archive["data"], dtype=np.float64)
        names = [str(n) for n in archive["columns"]]
        meta = json.loads(str(archive["meta"].item()))
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(f"invalid feature table in {path}")
    return data, {name: i for i, name in enumerate(names)}, meta


def _finite(row, columns, name) -> bool:
    i = columns.get(name)
    return i is not None and math.isfinite(float(row[i]))


def _value(row, columns, name, default=0.0):
    i = columns.get(name)
    if i is None:
        return default
    value = float(row[i])
    return value if math.isfinite(value) else default


def report(path: Path, include_events: bool = False,
           blink_min_s: float | None = None) -> dict:
    path = Path(path)
    data, columns, meta = load(path)
    minimum = li.BLINK_MIN_S if blink_min_s is None else float(blink_min_s)
    if not 0.05 <= minimum <= 2.0:
        raise ValueError("blink_min_s must be between 0.05 and 2 seconds")
    blink = li.BlinkDetector(None, 0.55, 0.70, minimum, li.BLINK_MAX_S,
                             0.6, li.BLINK_RIGHT_S)
    gestures = li.FaceGestures()
    scroll = li.TiltScroll()
    clicks = Counter()
    gesture_counts = Counter()
    notches = 0
    mouth_open_s = 0.0
    mouth_since = None
    events: list[dict] = []
    active_event = None
    tracked = int(data[:, columns["face"]].sum()) if len(data) else 0

    def closing() -> bool:
        return blink.closed_since is not None or blink.wink_eye is not None

    def finish_event(index: int, click: str = "", cancelled: bool = False) -> None:
        nonlocal active_event
        if active_event is None:
            return
        event = dict(active_event)
        event.update(end=int(index), click=click, cancelled=bool(cancelled),
                     cued=int(blink.cued), deep=float(blink.deep), moved=float(blink.moved))
        event["duration"] = max(0.0, event["t1"] - event["t0"])
        events.append(event)
        active_event = None

    def cancel_event(index: int) -> None:
        finish_event(index, cancelled=True)
        blink.reset()

    for index, row in enumerate(data):
        t = _value(row, columns, "t")
        now = t + 1000.0                    # avoid an artificial start-up cooldown
        if not _value(row, columns, "face"):
            if mouth_since is not None:
                mouth_open_s = max(mouth_open_s, now - mouth_since)
            mouth_since = None
            cancel_event(index)
            gestures.reset()
            scroll.reset()
            continue

        ear = (_value(row, columns, "ear_left"), _value(row, columns, "ear_right"))
        scores = None
        if _finite(row, columns, "bs_eyeBlinkLeft") and _finite(row, columns, "bs_eyeBlinkRight"):
            scores = (_value(row, columns, "bs_eyeBlinkLeft"),
                      _value(row, columns, "bs_eyeBlinkRight"))
        pose = None
        if _finite(row, columns, "yaw") and _finite(row, columns, "pitch"):
            pose = (_value(row, columns, "yaw"), _value(row, columns, "pitch"))
        head = None
        if _finite(row, columns, "head_x") and _finite(row, columns, "head_y"):
            head = np.array([_value(row, columns, "head_x"), _value(row, columns, "head_y")])
        mar = _value(row, columns, "mar")
        jaw = _value(row, columns, "bs_jawOpen")

        # This is the maximum continuous wide-open mouth interval, not the
        # cumulative time spent talking. A quit needs three continuous seconds.
        if mar >= 0.6 and jaw >= li.CLOSE_JAW_MIN and blink.eyes_open(*ear, pose):
            mouth_since = mouth_since if mouth_since is not None else now
        else:
            if mouth_since is not None:
                mouth_open_s = max(mouth_open_s, now - mouth_since)
            mouth_since = None

        if mar >= 0.6 * 0.85:
            cancel_event(index)
            gestures.reset()
            scroll.reset()
            continue

        roll = _value(row, columns, "roll", float("nan"))
        wheel = scroll.update(roll if math.isfinite(roll) else None, now)
        notches += abs(int(wheel))
        if scroll.tilted:
            cancel_event(index)
            gestures.reset()
            continue

        expressions = {name[3:]: _value(row, columns, name)
                       for name in columns if name.startswith("bs_") and _finite(row, columns, name)}
        expressions["mouthShift"] = _value(row, columns, "mouth_shift")
        fired = gestures.update(expressions, now, pose)
        if fired:
            gesture_counts[fired] += 1
        if gestures.active is not None or not gestures.relaxed:
            cancel_event(index)
            continue

        was_closing = closing()
        closed, click, _ = blink.update(*ear, now, scores,
                                        bool(_value(row, columns, "winks_ok", 1.0)), pose, head)
        if not was_closing and closing():
            active_event = {"start": index, "t0": t, "t1": t,
                            "kind": "wink" if blink.wink_eye is not None else "both"}
        if active_event is not None:
            active_event["t1"] = t
        if click:
            clicks[click] += 1
        if was_closing and not closing():
            finish_event(index, click, cancelled=False)

    if active_event is not None:
        finish_event(len(data) - 1, cancelled=True)
    seconds = float(data[:, columns["dt"]].sum()) if len(data) else 0.0
    answer = {
        "name": path.stem,
        "minutes": seconds / 60.0,
        "frames": int(len(data)),
        "tracked": tracked,
        "tracked_pct": 100.0 * tracked / max(1, int(meta.get("frames", len(data)))),
        "clicks": int(clicks["left"] + clicks["right"]),
        "left": int(clicks["left"]),
        "right": int(clicks["right"]),
        "gestures": dict(gesture_counts),
        "notches": int(notches),
        "mouth_open_s": float(mouth_open_s),
        "closures": len(events),
        "eyesync_version": meta.get("eyesync_version", "unknown"),
    }
    if blink_min_s is not None:
        answer["blink_min_s"] = minimum
    if include_events:
        answer["events"] = events
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--events", action="store_true", help="include closure event indices")
    parser.add_argument("--blink-min", type=float, help="development-only tick-time override")
    args = parser.parse_args()
    reports = [report(path, args.events, args.blink_min) for path in args.features]
    for item in reports:
        print(f"{item['name']}: {item['minutes']:.2f} min, {item['clicks']} false clicks "
              f"({item['right']} right), {item['closures']} closures, "
              f"{item['tracked_pct']:.1f}% tracked, gestures={item['gestures']}, "
              f"scroll={item['notches']}")
    total_minutes = sum(item["minutes"] for item in reports)
    total_clicks = sum(item["clicks"] for item in reports)
    print(f"TOTAL: {total_clicks} clicks / {total_minutes:.2f} min = "
          f"{total_clicks / max(total_minutes, 1e-9):.3f} per minute")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
