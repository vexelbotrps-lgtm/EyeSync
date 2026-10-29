"""
record_blinks.py - a 4.5-minute blink and gesture test: what EyeSync sees of your face.
=======================================================================================

    python tools/record_blinks.py          (from the EyeSync folder; quit EyeSync first -
                                            the test needs the camera)

In EyeSync itself (since 1.5.6): the window's Blink test button - EyeSync
hands the camera over while it runs - or EyeSync.exe --blink-test. Recordings
then go to the recordings folder next to the installed EyeSync.exe, and the
file is shown in Explorer at the end, to send to whoever tunes EyeSync.

A window guides you through nine short parts - reading silently (natural
blinks), blinking to click on a cue, long blinks (right click), moving your
head around, clicking in the screen's corners, head tilts, the mouth pulled to your left (recenter) and to
your right (calibrate), reading out loud - and saves the measurements to
tools/recordings/blinks-<date-time>.npz (see below for EyeSync.exe), with
the calibration it was made with. EyeSync's blink detector runs
along with your calibration and plays its real ticks and click sounds, so
it feels as in EyeSync.
tools/replay_blinks.py then runs them through EyeSync's blink detector and
says how many cues it caught, with the right button, and how many clicks it
made when you didn't mean one.

Only numbers are saved - how open each eye is, MediaPipe's blink, squint
and mouth scores, where the head and mouth are and how the head is turned -
never a picture.
Esc stops early (what was recorded so far is kept).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2               # noqa: E402
import numpy as np       # noqa: E402

import app_paths           # noqa: E402
import desktop             # noqa: E402
import face_engine as fe   # noqa: E402
import live_inference as li  # noqa: E402
import site_access         # noqa: E402

# (name, seconds, cue every N s or None, instruction). Since 2026-10-20 there are
# no wink parts (the user can't wink; blinks-2026-10-18 has them) but three
# mouth parts: the recenter / calibrate gestures, and talking (no gesture).
# Since 1.5.2 the click cues are further apart (a click takes ~1 s: the user
# reacts 0.5-0.9 s after the tick), and "natural" is long enough to count
# natural blinks that click by mistake. Since 1.5.5 "corners": a click in each
# screen corner (the user: "at extreme corners the eye blink is not working").
PARTS = [
    ("natural", 45, None, "Read the text below to yourself, silently. Blink normally. "
                          "Don't try to click."),
    ("blink", 24, 3.5, "Each time the circle turns GREEN: close both eyes, and open them after "
                       "the single TICK (a left click)."),
    ("long", 27, 4.5, "Each time it turns GREEN: close BOTH eyes and keep them closed until you "
                      "hear the DOUBLE tick, then open them (a right click)."),
    ("move", 12, None, "Move your head around and look at all four corners of the screen. "
                       "Don't click."),
    ("corners", 32, 7.0, "When GREEN: turn your head to the screen corner shown below, as far "
                         "as you go to click there, and click: close both eyes, open after "
                         "the TICK. Then come back to the middle."),
    ("tilt", 16, 4.0, "When GREEN: tilt your head - ear towards the shoulder the arrow shows - "
                      "hold it about 2 s, then straighten up."),
    ("mouth_left", 18, 5.5, "When GREEN: pull your mouth to YOUR LEFT (towards your left ear) "
                            "and hold it until you hear the beep, then relax (recenter)."),
    ("mouth_right", 24, 7.5, "When GREEN: pull your mouth to YOUR RIGHT and hold it - a beep, "
                             "a countdown - until the second beep, then relax (calibrate)."),
    ("talk", 20, None, "Relax your mouth, then READ THE TEXT BELOW OUT LOUD, at your normal "
                       "pace. Don't pull your mouth to a side."),
]
# Big title per part (the 2026-10-20 test's "read out loud" was missed) and
# text to read in the parts that have no cue circle.
TITLES = {"natural": "READ SILENTLY", "blink": "CLICK: open after the TICK",
          "long": "RIGHT CLICK: open after the DOUBLE tick", "move": "MOVE YOUR HEAD",
          "corners": "CLICK IN THE CORNERS",
          "tilt": "TILT TO SCROLL", "mouth_left": "MOUTH TO YOUR LEFT",
          "mouth_right": "MOUTH TO YOUR RIGHT, hold 3 s", "talk": "READ OUT LOUD"}
READING = {
    "natural": "EyeSync moves the mouse with your head and clicks when you blink. It learns how "
               "your eyes look when they are open, and how they look when you close them on "
               "purpose. A quick blink while you read or think should never click. This test "
               "counts how often that still happens, so it can be fixed. Keep reading at your "
               "own pace; if you reach the end, start again from the top.",
    "talk": "The quick brown fox jumps over the lazy dog. I would like a cup of tea, please. "
            "My name is on the list, and I will be there at seven. Could you open the window? "
            "It is very warm in here today.",
}
READY_S = 3.0
CUE_LIT_S = 0.6
SCORES = ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
          "browDownLeft", "browDownRight", "cheekSquintLeft", "cheekSquintRight")
MOUTH_SCORES = ("mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight", "jawOpen")
# Where the eyes look (since 1.5.7): eyes looking far down look closed to the
# camera - the GRID corpus's readers clicked that way (dataset_report.py).
LOOK_SCORES = ("eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight")
# The mouth columns come last, so older recordings (without them) still replay.
COLS = ("t", "part", "face", "ear_l", "ear_r", *SCORES, "nose_x", "nose_y", "face_px",
        "yaw", "pitch", "roll", "level", *MOUTH_SCORES, "mouth_shift", "mar", *LOOK_SCORES)
GESTURE_SHOWN_S = 1.5
CORNERS = ("top-left", "top-right", "bottom-right", "bottom-left")
W, H = 1100, 620


def wrap(text: str, width: int = 46) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def beep() -> None:
    try:
        import winsound
        winsound.Beep(880, 70)
    except Exception:          # noqa: BLE001 - only a cue
        pass


def recordings_dir() -> Path:
    """Where recordings go: tools/recordings, or next to the installed EyeSync.exe."""
    return (app_paths.DATA_DIR if app_paths.FROZEN else HERE) / "recordings"


def open_camera(wait_s: float = 15.0):
    """The camera - EyeSync may still be handing it over (its Blink test button)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return fe.open_any_camera(None, quiet=True)
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def recording_meta(cues: list, stopped: bool, settings: dict) -> dict:
    """What the recording says about itself: its parts and cues, and the calibration
    the test ran with - a replay of another person's recording needs theirs."""
    return {"columns": COLS, "parts": [p[:3] for p in PARTS], "cues": cues,
            "complete": not stopped, "date": time.strftime("%Y-%m-%d %H:%M"),
            "version": site_access.VERSION, "calibration": settings.get("calibration")}


def show_saved(out: Path) -> None:
    """The end: where the file is, shown in Explorer (in EyeSync.exe) to send it."""
    screen = np.full((H, W, 3), 24, np.uint8)
    lines = ["Done - thank you!", "", "Saved (numbers only, no pictures):", str(out.name), "",
             "It opens in Explorer now: send this file to the", "person who asked you to do the test."]
    for i, line in enumerate(lines):
        cv2.putText(screen, line, (60, 120 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 if i == 0 else 0.8, (90, 230, 255) if i == 0 else (240, 240, 240), 2,
                    cv2.LINE_AA)
    cv2.imshow("EyeSync blink test", screen)
    cv2.waitKey(4000)
    if app_paths.FROZEN and sys.platform == "win32":
        import subprocess
        subprocess.Popen(["explorer", f"/select,{out}"])


def main() -> int:
    out_dir = recordings_dir()
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"blinks-{time.strftime('%Y%m%d-%H%M%S')}.npz"
    print("Opening the camera (quit EyeSync first if this fails) ...")
    cam = open_camera()
    mesh, low = fe.FaceMeshBackend(), fe.LowLight()
    # EyeSync's own blink detector, with the user's calibration: its ticks and click
    # sounds play as they would in EyeSync (the recording itself is the raw signals).
    blink = li.BlinkDetector(None, 0.55, 0.70, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
    gestures = li.FaceGestures()          # ... and its face gestures (nothing is carried out)
    settings = li.load_settings()
    li.apply_calibration(settings.get("calibration"), li.HeadPointer(1920, 1080), blink, gestures)
    scroll = li.TiltScroll()
    warned = False
    shown, shown_until = "", 0.0          # a gesture that fired, on the guide for a moment
    rows, cues = [], []                # cues: (time, part, which: "", "left", "right", a corner)
    cv2.namedWindow("EyeSync blink test", cv2.WINDOW_AUTOSIZE)
    t0 = time.monotonic()
    stopped = False
    try:
        for p, (name, secs, every, text) in enumerate(PARTS):
            start = time.monotonic()
            next_cue = start + READY_S + 1.0 if every else None
            n_cue = 0
            lit_until = 0.0
            while not stopped:
                now = time.monotonic()
                if now - start > READY_S + secs:
                    break
                ok, frame = cam.read()
                if not ok:
                    continue
                recording = now - start >= READY_S
                if recording and next_cue is not None and now >= next_cue:
                    which = (("left" if n_cue % 2 == 0 else "right") if name == "tilt" else
                             CORNERS[n_cue % len(CORNERS)] if name == "corners" else "")
                    cues.append((now - t0, name, which))
                    n_cue += 1
                    lit_until = now + CUE_LIT_S
                    next_cue = now + every if now + every < start + READY_S + secs - 1.0 else None
                    beep()
                lm = low.process(mesh, frame)
                h, w = frame.shape[:2]
                if lm is None:
                    blink.reset()
                    scroll.update(None, now)
                    gestures.update(None, now)
                else:
                    el, er = fe.eye_aspect_ratios(lm, w, h)
                    nose, face_px = fe.nose_pointer(lm, w, h)
                    bs = mesh.last_blendshapes or {}
                    blink.eyes_open(el, er, mesh.last_pose)
                    scroll.update(fe.head_roll(lm, w, h), now)
                    fired = None
                    if scroll.tilted:
                        blink.reset()
                        gestures.reset()
                    elif bs:
                        # The face gestures as in the vision worker: a beep when a long
                        # hold is clearly meant, a beep when it fires.
                        fired = gestures.update({**bs, "mouthShift": fe.mouth_shift(lm, w, h)},
                                                now, mesh.last_pose)
                        holding = gestures.active
                        if (holding in li.GESTURE_HOLD_S and not warned
                                and now - gestures.since >= li.GESTURE_CONFIRM_S):
                            warned = True
                            desktop.play_warning_sound()
                        if holding is None:
                            warned = False
                        if fired:
                            desktop.play_warning_sound()
                            shown, shown_until = li.GESTURE_LABELS[fired], now + GESTURE_SHOWN_S
                    if gestures.active is not None or fired or not gestures.relaxed:
                        blink.reset()                 # a smile's squint is not a blink
                    elif not scroll.tilted:
                        sc = (bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) if bs else None
                        click = blink.update(el, er, now, sc, fe.eyes_level_ratio(lm, w, h) > 0.7,
                                             mesh.last_pose, nose / face_px)[1]
                        if blink.cue:
                            desktop.play_tick_sound(double=blink.cue == "double")
                        if click:
                            desktop.play_click_sound(right=click == "right")
                if lm is not None and recording:
                    left, right = fe.eye_aspect_ratios(lm, w, h)
                    nose, face_px = fe.nose_pointer(lm, w, h)
                    bs = mesh.last_blendshapes or {}
                    pose = mesh.last_pose or (np.nan, np.nan)
                    rows.append([now - t0, p, 1, left, right, *[bs.get(s, np.nan) for s in SCORES],
                                 nose[0] / face_px, nose[1] / face_px, face_px, pose[0], pose[1],
                                 fe.head_roll(lm, w, h), fe.eyes_level_ratio(lm, w, h),
                                 *[bs.get(s, np.nan) for s in MOUTH_SCORES],
                                 fe.mouth_shift(lm, w, h),
                                 fe.mouth_aspect_ratio(lm, w, h, mesh.last_pose),
                                 *[bs.get(s, np.nan) for s in LOOK_SCORES]])
                elif recording:
                    rows.append([now - t0, p, 0] + [np.nan] * (len(COLS) - 3))
                # The guide.
                screen = np.full((H, W, 3), 24, np.uint8)
                view = cv2.resize(cv2.flip(frame, 1), (320, 240))
                if lm is None:
                    cv2.putText(view, "Can't see your face", (40, 125), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (60, 60, 230), 2, cv2.LINE_AA)
                screen[20:260, 20:340] = view
                cv2.putText(screen, f"Part {p + 1} of {len(PARTS)}", (370, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(screen, TITLES.get(name, name.upper()), (370, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 230, 255), 2, cv2.LINE_AA)
                for i, line in enumerate(wrap(text, 52)):
                    cv2.putText(screen, line, (370, 128 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (240, 240, 240), 2, cv2.LINE_AA)
                for i, line in enumerate(wrap(READING.get(name, ""), 86)):
                    cv2.putText(screen, line, (40, 400 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (230, 230, 200), 2, cv2.LINE_AA)
                if not recording:
                    msg = f"Get ready... {READY_S - (now - start):.0f}"
                elif every:
                    msg = f"{n_cue} done"
                else:
                    msg = f"{READY_S + secs - (now - start):.0f} s left"
                cv2.putText(screen, msg, (370, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (150, 200, 255), 2, cv2.LINE_AA)
                if every:
                    lit = now < lit_until
                    cv2.circle(screen, (W // 2, 470), 70, (80, 200, 60) if lit else (70, 70, 70), -1)
                    if name == "tilt" and cues and cues[-1][1] == "tilt":
                        arrow = "<- LEFT" if cues[-1][2] == "left" else "RIGHT ->"
                        cv2.putText(screen, arrow, (W // 2 - 70, 590), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (240, 240, 240), 2, cv2.LINE_AA)
                    if name == "corners":
                        corner = cues[-1][2] if cues and cues[-1][1] == "corners" else CORNERS[0]
                        cv2.putText(screen, corner.upper().replace("-", " "), (W // 2 - 95, 590),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
                # A gesture being held (with its progress), or one that just fired.
                if gestures.active is not None:
                    label = li.GESTURE_LABELS[gestures.active]
                    cv2.putText(screen, f"Holding: {label}", (370, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (120, 220, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(screen, (370, 395), (370 + int(360 * gestures.progress(now)), 410),
                                  (120, 220, 255), -1)
                elif now < shown_until:
                    cv2.putText(screen, f"Done: {shown}", (370, 380), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (80, 220, 80), 2, cv2.LINE_AA)
                cv2.putText(screen, "Esc = stop", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (120, 120, 120), 1, cv2.LINE_AA)
                cv2.imshow("EyeSync blink test", screen)
                if cv2.waitKey(1) & 0xFF == 27:
                    stopped = True
            if stopped:
                break
    finally:
        cam.release()
        mesh.close()
    data = np.array(rows, np.float64) if rows else np.zeros((0, len(COLS)))
    meta = recording_meta(cues, stopped, settings)
    np.savez_compressed(out, data=data, meta=json.dumps(meta))
    print(f"Saved {len(rows)} frames, {len(cues)} cues to {out}")
    try:
        show_saved(out)
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
