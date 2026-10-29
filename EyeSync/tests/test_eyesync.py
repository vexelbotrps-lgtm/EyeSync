"""
EyeSync regression tests - no camera, website or hardware needed.

    python -m unittest discover -s tests -v        (from the EyeSync folder)

Covers the parts whose behaviour users feel directly: blink and wink clicks,
the head pointer, the close gesture, cursor motion (stops when the head
stops, lands on target), low light, the standby and the WebSocket output
the website consumes. Nothing here touches your real settings, API key or
log: the WebSocket tests use ports 8799, 8798 and 8796.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("EYESYNC_LOG", os.path.join(tempfile.gettempdir(), "eyesync_test.log"))

import face_engine as fe           # noqa: E402
import live_inference as li        # noqa: E402
import voice_typing as vt          # noqa: E402

FPS = 30
li.desktop.play_warning_sound = li.desktop.play_click_sound = lambda: None   # silent tests
li.desktop.play_tick_sound = lambda double=False: None
_log = logging.getLogger("eyesync")          # the app's expected warnings would look like
_log.addHandler(logging.NullHandler())       # test failures in the installer's output
_log.propagate = False


# --------------------------------------------------------------------------- #
# Blink -> click
# --------------------------------------------------------------------------- #

class BlinkTests(unittest.TestCase):
    O, C = 0.28, 0.07            # open / closed eye aspect ratio

    def buttons(self, seq, winks_ok=True) -> list[str]:
        """seq of (left eye EAR, right eye EAR) at 30 FPS -> the clicks, e.g. ["left", "right"]."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.25, 1.5, 0.6)
        b.baseline = self.O
        t, out = 0.0, []
        for left, right in seq:
            t += 1 / FPS
            click = b.update(left, right, t, None, winks_ok)[1]
            if click:
                out.append(click)
        return out

    def clicks(self, seq) -> int:
        return len(self.buttons(seq))

    def test_cases(self):
        O, C = self.O, self.C
        cases = {
            "involuntary blink 100 ms": ([(O, O)] * 20 + [(C, C)] * 3 + [(O, O)] * 20, 0),
            "deliberate blink 400 ms": ([(O, O)] * 20 + [(C, C)] * 12 + [(O, O)] * 20, 1),
            "deliberate blink with a 1-frame glitch":
                ([(O, O)] * 20 + [(C, C)] * 6 + [(O, O)] + [(C, C)] * 6 + [(O, O)] * 20, 1),
            "flicker at the threshold":
                ([(O, O)] * 20 + [(C, C)] * 5 + [(0.17, 0.17), (0.14, 0.14)] * 4 + [(O, O)] * 20, 1),
            "quick look down": ([(O, O)] * 20 + [(0.21, 0.21)] * 15 + [(O, O)] * 20, 0),
            "resting eyes 3 s": ([(O, O)] * 20 + [(C, C)] * 90 + [(O, O)] * 20, 0),
            "single-frame glitch": ([(O, O)] * 20 + [(C, C)] + [(O, O)] * 20, 0),
        }
        for name, (seq, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.clicks(seq), expected)

    def test_blink_seen_only_by_blink_score(self):
        """With glasses the eyelid points can miss a blink; the learned score still rises."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, clicks = 0.0, 0
        # Developer's face: open-eye score ~0.02, a real blink only reached ~0.33.
        seq = [(0.02, 0.02)] * 30 + [(0.35, 0.33)] * 10 + [(0.02, 0.02)] * 20
        for sc in seq:
            t += 1 / FPS
            clicks += bool(b.update(self.O, self.O, t, sc)[1])   # eyelid geometry says open
        self.assertEqual(clicks, 1)

    def test_looking_down_is_not_a_blink(self):
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, clicks = 0.0, 0
        seq = [(self.O, 0.03)] * 30 + [(0.22, 0.15)] * 20 + [(self.O, 0.03)] * 20
        for ear, sc in seq:
            t += 1 / FPS
            clicks += bool(b.update(ear, ear, t, (sc, sc))[1])
        self.assertEqual(clicks, 0)

    def test_winks(self):
        O, C = self.O, self.C
        pad = [(O, O)] * 20
        cases = {
            "both eyes 400 ms = left click": ([(C, C)] * 12, ["left"]),
            "left eye 400 ms = left click": ([(C, O)] * 12, ["left"]),
            "right eye 400 ms = right click": ([(O, C)] * 12, ["right"]),
            "right wink, other eye squinting a little": ([(0.21, C)] * 12, ["right"]),
            "involuntary one-eye twitch 100 ms": ([(O, C)] * 3, []),
            "wink that turns into a blink = left click": ([(O, C)] * 4 + [(C, C)] * 10, ["left"]),
            "one eye resting 3 s": ([(O, C)] * 90, []),
        }
        for name, (closure, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.buttons(pad + closure + pad), expected)
        # Head turned far to a side: the far eye looks closed - never a click.
        self.assertEqual(self.buttons(pad + [(O, C)] * 12 + pad, winks_ok=False), [])

    def scored(self, seq) -> list[str]:
        """seq of ((left EAR, right EAR), (left score, right score)) at 30 FPS -> the clicks."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        b.baseline = self.O
        t, out = 0.0, []
        for ears, scores in seq:
            t += 1 / FPS
            click = b.update(ears[0], ears[1], t, scores, True)[1]
            if click:
                out.append(click)
        return out

    def test_winks_when_both_blink_scores_rise(self):
        """
        MediaPipe's blink score rises for the OPEN eye too during a wink (it
        squints a little). Judged by the scores, a right wink was a blink: a
        LEFT click. The eyelids tell which eye winked.
        """
        O, C = self.O, self.C
        pad = [((O, O), (0.03, 0.03))] * 30
        cases = {
            "right wink": ([((O, C), (0.45, 0.9))] * 12, ["right"]),
            "left wink": ([((C, O), (0.9, 0.45))] * 12, ["left"]),
            "right wink, the left eye squinting": ([((0.21, C), (0.5, 0.9))] * 12, ["right"]),
            "blink, one eyelid a frame behind": ([((C, 0.2), (0.9, 0.6))] + [((C, C), (0.9, 0.9))] * 11,
                                                 ["left"]),
            "blink": ([((C, C), (0.9, 0.9))] * 12, ["left"]),
            "short right twitch": ([((O, C), (0.45, 0.9))] * 3, []),
        }
        for name, (closure, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.scored(pad + closure + pad), expected)

    def ticked(self, seq, **kw):
        """seq of (left EAR, right EAR) at 30 FPS -> (clicks, [(time, tick)]) - see BlinkDetector.cue."""
        b = li.BlinkDetector(None, 0.55, 0.70, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6, **kw)
        b.baseline = self.O
        t, clicks, ticks = 0.0, [], []
        for left, right in seq:
            t += 1 / FPS
            click = b.update(left, right, t, None, True)[1]
            if b.cue:
                ticks.append((t, b.cue))
            if click:
                clicks.append(click)
        return clicks, ticks

    def test_long_blink_is_a_right_click(self):
        """A real user's right wink closed both eyes: a longer blink is the right click."""
        O, C = self.O, self.C
        pad = [(O, O)] * 30
        # 1.5.2's windows (BLINK_RIGHT_S): about a second to react to each tick -
        # the user opened 0.55-0.9 s after the tick they waited for.
        cases = {
            "0.4 s: click": (12, ["left"], ["tick"]),
            "1.0 s: a slow reaction to the tick, still a click": (30, ["left"], ["tick"]),
            "1.4 s: right click": (42, ["right"], ["tick", "double"]),
            "2.2 s: a slow reaction to the double tick, still a right click": (66, ["right"], ["tick", "double"]),
            "2.6 s: resting, no click": (78, [], ["tick", "double"]),
            "0.1 s natural blink: nothing": (3, [], []),
        }
        for name, (frames, clicks, ticks) in cases.items():
            with self.subTest(name):
                got_clicks, got_ticks = self.ticked(pad + [(C, C)] * frames + pad)
                self.assertEqual(got_clicks, clicks)
                self.assertEqual([k for _, k in got_ticks], ticks)
        # Off (--right-blink 0): a long blink is an ordinary click again.
        self.assertEqual(self.ticked(pad + [(C, C)] * 45 + pad, right_s=None),
                         (["left"], [(self.ticked(pad + [(C, C)] * 45 + pad)[1][0][0], "tick")]))
        # A right-eye wink is still a right click, with a single tick.
        self.assertEqual([k for _, k in self.ticked(pad + [(O, C)] * 30 + pad)[1]], ["tick"])
        self.assertEqual(self.ticked(pad + [(O, C)] * 30 + pad)[0], ["right"])

    def test_the_double_tick_keeps_its_second_after_a_calibrated_tick(self):
        """1.5.9's calibrated tick (up to 0.65 s) kept the double tick at 1.3 s:
        0.65 s to react, and the user opens 0.55-0.9 s after the tick - a left
        click came out as a right click."""
        O, C = self.O, self.C
        for min_s in (li.BLINK_MIN_S, 0.45):                # 0.45: 1.5.10's cap, or --min-blink
            with self.subTest(min_s):
                def run(frames):
                    b = li.BlinkDetector(None, 0.55, 0.70, min_s, li.BLINK_MAX_S, 0.6)
                    b.baseline = O
                    t, clicks, ticks = 0.0, [], []
                    for left, right in [(O, O)] * 30 + [(C, C)] * frames + [(O, O)] * 30:
                        t += 1 / FPS
                        click = b.update(left, right, t, None, True)[1]
                        if b.cue:
                            ticks.append(t)
                        if click:
                            clicks.append(click)
                    return clicks, ticks
                _, ticks = run(60)
                self.assertAlmostEqual(ticks[1] - ticks[0], 1.0, delta=1.5 / FPS)
                # Opened 0.85 s after the tick: still the left click it was meant as.
                react = int(round((min_s + 0.85) * FPS))
                self.assertEqual(run(react)[0], ["left"])

    def ear(self, closedness: float) -> float:
        """The EAR at this closedness (1.0 = the close threshold; see BlinkDetector.closedness)."""
        return self.O * (1 - closedness * (1 - 0.55))

    def test_a_blink_that_starts_on_one_side_is_a_blink(self):
        """1.5.2's log: the user's right eye closes later and less - blinks began as
        "left winks" and stayed so (one tick, no double, no voice typing)."""
        O, C = self.O, self.C
        pad = [(O, O)] * 30
        lag = [(C, self.ear(0.5))] * 3                  # the right eye still half open ...
        for name, right, frames, want, ticks in (
                ("... then closes only to 0.7", 0.7, 45, ["right"], ["tick", "double"]),
                ("... then closes fully", 1.5, 45, ["right"], ["tick", "double"]),
                ("... a short one", 0.7, 15, ["left"], ["tick"])):
            with self.subTest(name):
                clicks, got = self.ticked(pad + lag + [(C, self.ear(right))] * frames + pad)
                self.assertEqual(clicks, want)
                self.assertEqual([k for _, k in got], ticks)
        # Both eyes closing, the right only to 0.7 from the start: a blink too
        # (it used to count as nothing at all - no tick).
        clicks, got = self.ticked(pad + [(C, self.ear(0.7))] * 15 + pad)
        self.assertEqual((clicks, [k for _, k in got]), (["left"], ["tick"]))
        # A real wink keeps the other eye open: still a wink.
        self.assertEqual(self.ticked(pad + [(C, self.ear(0.3))] * 15 + pad)[0], ["left"])
        self.assertEqual(self.ticked(pad + [(self.ear(0.3), C)] * 15 + pad)[0], ["right"])

    def test_a_blink_that_reopens_slowly_is_not_a_click(self):
        """33 people's videos (the GRID corpus): natural blinks that reopen slowly,
        the eyes looking down - half open for 0.2-1 s - passed the tick and
        clicked (4 times a minute; one person's half-shut eye made false right
        clicks). The tick comes only while the eyes are shut (TICK_WHEN_SHUT)."""
        O, C = self.O, self.C
        pad = [(O, O)] * 30
        half = (self.ear(0.8), self.ear(0.8))           # between reopening and closed
        slow = pad + [(C, C)] * 5 + [half] * 15 + pad   # shut 0.17 s, then half open 0.5 s
        self.assertEqual(self.ticked(slow), ([], []))
        self.assertEqual(self.ticked(pad + [(O, C)] * 4 + [(O, self.ear(0.8))] * 15 + pad), ([], []))
        # Meant: shut through the tick (then it may open slowly) - a click, as before.
        meant = pad + [(C, C)] * 12 + [half] * 6 + pad
        clicks, ticks = self.ticked(meant)
        self.assertEqual((clicks, [k for _, k in ticks]), (["left"], ["tick"]))
        # Without the rule, the slow blink clicked.
        li.BlinkDetector.TICK_WHEN_SHUT = False
        try:
            self.assertEqual(self.ticked(slow)[0], ["left"])
        finally:
            li.BlinkDetector.TICK_WHEN_SHUT = True

    def test_every_missed_click_is_logged(self):
        """One hint per 20 s hid how many meant clicks went missing (2026-10-28)."""
        O, C = self.O, self.C
        half = (self.ear(0.8), self.ear(0.8))
        missed = [(C, C)] * 6 + [half] * 8 + [(O, O)] * 30      # ~0.47 s, shut only ~0.17 s
        with self.assertLogs("eyesync", "INFO") as logs:
            self.assertEqual(self.ticked([(O, O)] * 30 + missed * 2), ([], []))
        self.assertEqual(sum("opening before the tick" in m for m in logs.output), 2)

    def test_tick_requires_both_eyes_to_stay_closed(self):
        """One eye reopening must not let the other eye's high score arm a click.

        Asymmetric deliberate blinks still work: both eyes participate, even when
        one closes only partly. The failing pattern is one eye open while the
        other's score is so high that their average alone looks shut.
        """
        O, C = self.O, self.C
        pad = [(O, O)] * 30
        one_open = [(self.ear(0.2), self.ear(2.0))] * 15
        self.assertEqual(self.ticked(pad + [(C, C)] * 4 + one_open + pad), ([], []))

        # Both eyes closing unequally is still a blink, not a wink or a miss.
        partial = [(C, self.ear(0.7))] * 15
        clicks, ticks = self.ticked(pad + partial + pad)
        self.assertEqual(clicks, ["left"])
        self.assertEqual([kind for _, kind in ticks], ["tick"])

    def drifting(self, drift_from: float, frames: int = 36) -> list[str]:
        """A closure the eyelids show only partly (0.9 - MediaPipe's blink score
        says shut) while the head drifts 0.18 face widths sideways from
        ``drift_from`` s on, then opens: the clicks."""
        b = li.BlinkDetector(None, 0.55, 0.70, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
        b.baseline = self.O
        half = self.ear(0.9)
        seq = ([(self.O, (0.03, 0.03))] * 30 + [(half, (0.9, 0.9))] * frames
               + [(self.O, (0.03, 0.03))] * 30)
        t, out = 0.0, []
        for k, (ear, sc) in enumerate(seq):
            t += 1 / FPS
            held = (k - 30) / FPS
            x = 0.18 * float(np.clip((held - drift_from) / 0.4, 0, 1)) if k >= 30 else 0.0
            click = b.update(ear, ear, t, sc, True, None, np.array([x, 0.0]))[1]
            if click:
                out.append(click)
        return out

    def test_a_ticked_click_survives_the_head_drifting(self):
        """1.5.10's log (2026-10-27/28): 1131 and 1173 ms closures, eyelids 0.86 and
        1.10 of 1.25, cancelled as "half closed" when the head had drifted 0.17-0.18
        face widths - long after their tick. The tick promised a click."""
        self.assertEqual(self.drifting(0.6), ["left"])
        li.BlinkDetector.KEEP_TICKED = False
        try:
            self.assertEqual(self.drifting(0.6), [])          # 1.5.10
        finally:
            li.BlinkDetector.KEEP_TICKED = True
        # The head getting there before the tick: eyes that only look closed
        # where it went (the screen bottom) - still no click.
        self.assertEqual(self.drifting(-0.2), [])

    def test_ticks_come_when_opening_would_click(self):
        O, C = self.O, self.C
        pad = [(O, O)] * 30
        _, ticks = self.ticked(pad + [(C, C)] * 40 + pad)
        (t1, _), (t2, _) = ticks
        self.assertAlmostEqual(t1 - 30 / FPS, li.BLINK_MIN_S, delta=0.07)
        self.assertAlmostEqual(t2 - t1, li.BLINK_RIGHT_S - li.BLINK_MIN_S, delta=0.04)
        # Whatever the length, the button is what the ticks said.
        for frames in range(4, 50):
            clicks, ticks = self.ticked(pad + [(C, C)] * frames + pad)
            if clicks:
                self.assertEqual(clicks, ["right" if ticks[-1][1] == "double" else "left"], frames)
        # No tick while a click wouldn't count: right after a click (cooldown) ...
        blink = [(C, C)] * 12 + [(O, O)] * 3
        clicks, ticks = self.ticked(pad + blink * 2 + pad)
        self.assertEqual((clicks, [k for _, k in ticks]), (["left"], ["tick"]))
        # ... and right after resting the eyes the tick waits for a deliberate length.
        rest = [(C, C)] * 75 + [(O, O)] * 9
        clicks, ticks = self.ticked(pad + rest + [(C, C)] * 10 + pad)
        self.assertEqual(clicks, [])
        self.assertEqual([k for t, k in ticks if t > 4.0], [])      # (the rest itself ticked)

    def test_cooldown(self):
        O, C = self.O, self.C
        blink = [(C, C)] * 12 + [(O, O)] * 3        # two blinks 0.5 s apart
        self.assertEqual(self.clicks([(O, O)] * 20 + blink * 2 + [(O, O)] * 20), 1)

    # -- head position (a real user: "due to the position of my face it registers a blink") --
    def run_posed(self, frames):
        """frames of (EAR, blink score, pitch) at 30 FPS -> (clicks, frames reported closed)."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        t, clicks, closed = 0.0, [], 0
        for ear, sc, pitch in frames:
            t += 1 / FPS
            shut, click = b.update(ear, ear, t, (sc, sc), True, (0.0, pitch))[:2]
            closed += shut
            if click:
                clicks.append(click)
        return clicks, closed

    @staticmethod
    def move(frm, to, frames):
        """Head turning from pitch frm to pitch to: (EAR, score, pitch). Looking further
        down lowers the eyelids (EAR 0.28 -> 0.15) and raises the blink score (to 0.55)."""
        out = []
        for i in range(1, frames + 1):
            p = frm + (to - frm) * i / frames
            k = min(max(-p / 20.0, 0.0), 1.0)          # 0 at the centre, 1 at -20 deg
            out.append((0.28 - 0.13 * k, 0.03 + 0.52 * k, p))
        return out

    def test_moving_the_head_down_and_back_is_not_a_blink(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        at_bottom = self.move(-20, -20, 20)                 # 0.67 s looking at the screen bottom
        seq = centre + self.move(0, -20, 9) + at_bottom + self.move(-20, 0, 9) + centre
        self.assertEqual(self.run_posed(seq)[0], [])

    def test_blinks_work_while_looking_at_the_screen_bottom(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        bottom = self.move(-20, -20, 1)[0]
        blink = [(0.05, 0.95, -20.0)] * 12                  # a real 0.4 s blink down there
        seq = centre + self.move(0, -20, 9) + [bottom] * 60 + blink + [bottom] * 30
        clicks, closed = self.run_posed(seq)
        self.assertEqual(clicks, ["left"])
        # The cursor is held only during the real blink, not while looking down.
        self.assertLess(closed, 12 + 10)

    def test_blink_with_a_small_nod_still_clicks(self):
        centre = [(0.28, 0.03, 0.0)] * 60
        nod = [(0.05, 0.95, -3.0 * i / 6) for i in range(1, 7)] + [(0.05, 0.95, -3.0)] * 6
        clicks, _ = self.run_posed(centre + nod + centre)
        self.assertEqual(clicks, ["left"])

    # -- seen in a real log (EyeSync 1.1.1): every deliberate blink was cancelled --
    def run_nose(self, frames, min_s=0.2, max_s=1.5):
        """frames of (EAR, score, nose x in face widths, head-angle reading) -> (clicks,
        frames reported closed, frames that rewound the cursor)."""
        b = li.BlinkDetector(None, 0.55, 0.70, min_s, max_s, 0.6)
        t, clicks, closed, rewinds = 0.0, [], 0, 0
        for ear, sc, nose_x, pitch in frames:
            t += 1 / FPS
            shut, click, just = b.update(ear, ear, t, (sc, sc), True, (0.0, pitch),
                                         np.array([nose_x, 0.0]))
            closed += shut
            rewinds += just
            if click:
                clicks.append(click)
        return clicks, closed, rewinds

    def test_still_blink_clicks_although_the_head_angle_reading_jumps(self):
        """MediaPipe's head angles jump ~8 degrees when both eyes close; the nose stays."""
        still = [(0.28, 0.03, 0.0, 0.0)] * 60
        blink = [(0.05, 0.95, 0.002 * (i % 2), -8.5) for i in range(14)]    # ~0.47 s
        self.assertEqual(self.run_nose(still + blink + still)[0], ["left"])

    def test_natural_blink_during_a_head_turn_does_not_stop_the_cursor(self):
        turn = [(0.28, 0.03, i / 30, 0.0) for i in range(30)]        # 1 face width a second
        blink = [(0.05, 0.95, (30 + i) / 30, 0.0) for i in range(5)]  # a 170 ms blink mid-turn
        after = [(0.28, 0.03, (35 + i) / 30, 0.0) for i in range(30)]
        clicks, closed, rewinds = self.run_nose(turn + blink + after)
        self.assertEqual((clicks, closed, rewinds), ([], 0, 0))

    def test_deliberate_blink_right_after_slow_aiming_clicks(self):
        aim = [(0.28, 0.03, 0.25 * i / 30, 0.0) for i in range(45)]  # aiming at 0.25 face widths/s
        x = 0.25 * 45 / 30
        blink = [(0.05, 0.95, x, -8.0)] * 12
        still = [(0.28, 0.03, x, 0.0)] * 30
        self.assertEqual(self.run_nose(aim + blink + still)[0], ["left"])

    def test_turning_the_head_with_eyes_only_half_closed_is_not_a_click(self):
        """Eyelids at ~55% of open + a high blink score (looking down): a head position."""
        still = [(0.28, 0.03, 0.0, 0.0)] * 60
        half = [(0.15, 0.6, 0.4 * i / 12, 0.0) for i in range(1, 13)]           # 0.4 fw (~12 deg)
        self.assertEqual(self.run_nose(still + half + [(0.28, 0.03, 0.4, 0.0)] * 30)[0], [])
        moving_half = [(0.15, 0.6, 0.03 * i, 0.0) for i in range(1, 25)]        # starts mid-turn
        self.assertEqual(self.run_nose(still + moving_half + [(0.28, 0.03, 0.72, 0.0)] * 30)[0], [])

    # -- a real user: "when I am moving my head, tilting it, the blink is not registering" --
    def test_deliberate_blink_while_moving_the_head_clicks(self):
        move = [(0.28, 0.03, i / 30, 0.0) for i in range(30)]          # 1 face width a second
        blink = [(0.07, 0.9, (30 + i) / 30, 0.0) for i in range(16)]  # a 0.53 s blink, still moving
        after = [(0.28, 0.03, (46 + i) / 30, 0.0) for i in range(30)]
        clicks, closed, rewinds = self.run_nose(move + blink + after)
        self.assertEqual(clicks, ["left"])
        self.assertEqual(rewinds, 1)              # held (once clearly a click), back where it began
        self.assertLess(closed, 16)               # not held for the first part of the blink

    def test_natural_blinks_while_reading_do_not_click(self):
        """Both recordings of a real user: natural blinks while reading lasted up to
        274-286 ms and clicked 3-6 times a minute with the tick at 0.2 s (1.5.3)."""
        def blink(frames, speed):
            still = [(0.28, 0.03, 0.0, 0.0)] * 60
            moving = [(0.28, 0.03, speed * i / 30, 0.0) for i in range(30)]
            x = speed
            shut = [(0.07, 0.9, x + speed * i / 30, 0.0) for i in range(frames)]
            x += speed * frames / 30
            return self.run_nose(still + moving + shut + [(0.28, 0.03, x + speed * i / 30, 0.0)
                                                          for i in range(40)],
                                 li.BLINK_MIN_S, li.BLINK_MAX_S)[0]
        for speed in (0.0, 0.7):                   # head still / moving as when reading (0.5-0.75 fw/s)
            for frames in (4, 6, 8, 9):            # 133-300 ms
                with self.subTest(speed=speed, ms=round(frames * 1000 / 30)):
                    self.assertEqual(blink(frames, speed), [])
        self.assertEqual(blink(11, 0.0), ["left"])          # 0.37 s, head still: past the tick
        self.assertEqual(blink(11, 0.7), [])                # ... head moving: before its tick
        self.assertEqual(blink(15, 0.7), ["left"])          # 0.5 s, head moving: past it

    def test_ticks_wait_for_a_moving_head(self):
        """A closure that begins with the head moving ticks from BLINK_MOVING_MIN_S, and
        its double tick keeps the second after it."""
        b = li.BlinkDetector(None, 0.55, 0.70, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
        t, ticks = 0.0, []
        for i in range(160):
            t += 1 / FPS
            ear = 0.07 if 30 <= i < 90 else 0.28
            b.update(ear, ear, t, (0.03, 0.03), True, None, np.array([0.7 * min(i, 60) / 30, 0.0]))
            if b.cue:
                ticks.append((t - 31 / FPS, b.cue))
        (t1, k1), (t2, k2) = ticks
        self.assertEqual((k1, k2), ("tick", "double"))
        self.assertAlmostEqual(t1, li.BLINK_MOVING_MIN_S, delta=0.05)
        self.assertAlmostEqual(t2 - t1, li.BLINK_RIGHT_S - li.BLINK_MIN_S, delta=0.05)

    def test_wink_while_tilting_the_head_clicks(self):
        """Tilting moves the nose sideways too; a right-eye wink is still a right click."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        t, clicks = 0.0, []
        for i in range(90):
            t += 1 / FPS
            right = 0.06 if 40 <= i < 56 else 0.28                    # right eye shut for 0.53 s
            sc = (0.03, 0.9 if 40 <= i < 56 else 0.03)
            click = b.update(0.28, right, t, sc, True, (0.0, 0.0), np.array([0.025 * i, 0.01 * i]))[1]
            if click:
                clicks.append(click)
        self.assertEqual(clicks, ["right"])

    def test_click_lands_where_the_blink_began(self):
        """The detector says when the eyes began to close, for putting the click there."""
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        t, held_at = 0.0, None
        for i in range(70):
            t += 1 / FPS
            ear = 0.07 if 30 <= i < 46 else 0.28
            shut, click, just = b.update(ear, ear, t, (0.03, 0.03), True, None, np.array([i / 30, 0]))
            if just:
                held_at = t
        # Noticed a frame after the eyes began to close (the eyelid signal is the
        # median of 3 frames); CLICK_REWIND_S reaches back past that.
        self.assertTrue(31 / FPS <= b.started_at <= 33 / FPS, b.started_at)
        self.assertLess(b.started_at - 31 / FPS, li.CLICK_REWIND_S)
        # Held once it's a click: the head was moving, so from BLINK_MOVING_MIN_S.
        self.assertGreaterEqual(held_at - b.started_at, li.BLINK_MOVING_MIN_S - 1e-6)

    def test_blink_after_resting_eyes_is_not_a_click(self):
        O, C = self.O, self.C
        rest = [(C, C)] * 75 + [(O, O)] * 9            # eyes closed 2.5 s, open 0.3 s...
        blink = [(C, C)] * 10 + [(O, O)] * 30          # ...then the natural blink
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + blink), 0)
        later = [(O, O)] * 45 + blink                   # a deliberate blink 1.5 s later still clicks
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + later), 1)
        long_blink = [(C, C)] * 20 + [(O, O)] * 30      # 0.67 s right away: deliberate, clicks
        self.assertEqual(self.clicks([(O, O)] * 20 + rest + long_blink), 1)


# --------------------------------------------------------------------------- #
# Head pointer
# --------------------------------------------------------------------------- #

class HeadPointerTests(unittest.TestCase):
    FACE = 95.0                      # eye-corner distance in pixels (~40 cm from the camera)

    def pointer(self):
        hp = li.HeadPointer(1920, 1080)
        hp.reset()
        return hp

    def test_starts_where_asked(self):
        hp = li.HeadPointer(1920, 1080)
        hp.reset(np.array([300.0, 200.0]))
        np.testing.assert_allclose(hp.target(np.array([320.0, 240.0]), self.FACE, 2), [300, 200])

    def test_directions(self):
        hp = self.pointer()
        base = np.array([320.0, 240.0])
        hp.target(base, self.FACE, 2)
        right = hp.target(base + [-10, 0], self.FACE, 2)       # user turns right: nose moves image-left
        self.assertGreater(right[0], 960 + 50)
        down = hp.target(base + [0, 8], self.FACE, 2)           # user tilts down
        self.assertGreater(down[1], 540 + 50)

    def test_edges_never_get_stuck(self):
        hp = self.pointer()
        base = np.array([320.0, 240.0])
        hp.target(base, self.FACE, 2)
        self.assertEqual(hp.target(base + [-200, 0], self.FACE, 2)[0], 1919)  # far past the right edge
        back = hp.target(base + [-190, 0], self.FACE, 2)        # turn back a little
        self.assertLess(back[0], 1919 - 50, "cursor stuck at the edge")

    def test_slow_is_precise_quick_goes_far(self):
        """The same 0.3-face-width turn: slowly (2 s) moves the cursor a little, quickly far."""
        turn = 0.3 * self.FACE

        def moved(seconds):
            hp = self.pointer()
            n = int(seconds * FPS)
            for i in range(n + 1):
                pos = hp.target(np.array([320.0 - turn * i / n, 240.0]), self.FACE, 1, i / FPS)
            return pos[0] - 960
        slow = moved(2.0)
        quick = moved(0.1)                     # 3 face widths/s: a typical quick turn (measured)
        self.assertLess(slow, 0.3 * 1920 / li.HEAD_RANGE[1][0] + 5)
        self.assertGreater(quick, 2.5 * slow)

    def test_blink_nod_does_not_move_the_cursor(self):
        hp = self.pointer()
        hp.target(np.array([320.0, 240.0]), self.FACE, 2, 0.0)
        hp.target(np.array([320.0, 240.0]), self.FACE, 2, 1 / FPS)
        # eyes closed for 0.4 s (no frames used) while the head nods down 6 px
        after = hp.target(np.array([320.0, 246.0]), self.FACE, 2, 0.45)
        np.testing.assert_allclose(after, [960, 540])
        self.assertGreater(hp.target(np.array([320.0, 252.0]), self.FACE, 2, 0.45 + 1 / FPS)[1], 545)

    def test_distance_does_not_change_speed(self):
        near, far = self.pointer(), self.pointer()
        near.target(np.array([320.0, 240.0]), self.FACE * 1.5, 2)
        far.target(np.array([320.0, 240.0]), self.FACE, 2)
        a = near.target(np.array([320.0 - 15, 240.0]), self.FACE * 1.5, 2)   # same head turn,
        b = far.target(np.array([320.0 - 10, 240.0]), self.FACE, 2)         # closer = bigger in image
        self.assertAlmostEqual(a[0], b[0], delta=1)

    def test_steady_and_precise_with_real_jitter(self):
        """Nose jitter measured on the developer's recordings: ~0.45 px. Rest, small move, rest."""
        rng = np.random.default_rng(5)
        for level in li.SENSITIVITY_LEVELS:
            with self.subTest(level=level):
                hp = self.pointer()
                mc, beta, start, stop, vmax = li.SENSITIVITY_LEVELS[level]
                f, m = li.OneEuroFilter(mc, beta), li.CursorMotion(start, stop, vmax)
                hist, pos = deque(maxlen=li.NOSE_FRAMES), []
                for i in range(240):
                    t = i / FPS
                    u = min(max(t - 3.0, 0.0), 1.0)
                    nose = np.array([320.0 - 12 * (3 * u * u - 2 * u * u * u), 240.0])
                    hist.append(nose + rng.normal(0, 0.45, 2))
                    tgt = hp.target(li.nose_smooth(hist), self.FACE, level)
                    pos.append(m(f(tgt, t), t).copy())
                pos = np.array(pos)
                # Judge the distance moved: the first (noisy) frame sets the
                # reference, which shifts every position by the same few px.
                expected = 12 / self.FACE * 1920 / li.HEAD_RANGE[level][0]   # a slow move
                self.assertLess(np.ptp(pos[30:89], axis=0).max(), 3, "cursor wobbles at rest")
                self.assertLess(np.ptp(pos[-40:], axis=0).max(), 3, "cursor wobbles after moving")
                # It may stop short by less than the dead zone: the slow end
                # of a movement is ignored like any small movement.
                self.assertLess(abs(pos[-1][0] - pos[88][0] - expected), start,
                                "cursor did not follow the head")

    def test_noisier_webcam_rarely_nudges_at_rest(self):
        """A webcam twice as noisy (1 px): the dead zone grows with the measured noise."""
        for level in (1, 3, 5):
            moved = 0
            for seed in range(30):
                rng = np.random.default_rng(seed)
                hp = self.pointer()
                mc, beta, start, stop, vmax = li.SENSITIVITY_LEVELS[level]
                f, m = li.OneEuroFilter(mc, beta), li.CursorMotion(start, stop, vmax)
                hist, pos = deque(maxlen=li.NOSE_FRAMES), []
                for i in range(90):
                    t = i / FPS
                    hist.append(np.array([320.0, 240.0]) + rng.normal(0, 1.0, 2))
                    tgt = hp.target(li.nose_smooth(hist), self.FACE, level)
                    pos.append(m(f(tgt, t), t).copy())
                moved += np.ptp(np.array(pos)[30:89], axis=0).max() >= 3
            self.assertLessEqual(moved, 4, f"level {level}: nudged in {moved}/30 rests")


# --------------------------------------------------------------------------- #
# Face gestures
# --------------------------------------------------------------------------- #

class FaceGestureTests(unittest.TestCase):
    # Resting levels measured on the developer's face (with glasses).
    REST = {"mouthSmileLeft": 0.001, "mouthSmileRight": 0.001, "mouthPucker": 0.17,
            "mouthLeft": 0.003, "mouthRight": 0.003, "mouthShift": 0.0}
    TO_LEFT = {"mouthRight": 0.6, "mouthShift": 0.06}    # pulled to the user's left (fe.mouth_shift > 0)
    TO_RIGHT = {"mouthLeft": 0.6, "mouthShift": -0.06}

    def run_seq(self, segments):
        """segments: (seconds, {score overrides}) -> list of fired gestures."""
        g, t, fired = li.FaceGestures(), 0.0, []
        for secs, over in segments:
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                f = g.update({**self.REST, **over}, t)
                if f:
                    fired.append(f)
        return fired

    def test_each_gesture_fires_once(self):
        smile = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.75}
        self.assertEqual(self.run_seq([(3, {}), (1.7, smile), (1, {})]), ["smile"])
        self.assertEqual(self.run_seq([(3, {}), (1.7, {"mouthPucker": 0.8}), (1, {})]), [])  # removed
        self.assertEqual(self.run_seq([(3, {}), (1.7, self.TO_LEFT), (1, {})]), ["mouth_left"])
        self.assertEqual(self.run_seq([(3, {}), (3.3, self.TO_RIGHT), (1, {})]), ["mouth_right"])

    def test_mouth_side_gestures(self):
        # Calibrate needs the long hold; 1.7 s to the right is nothing.
        self.assertEqual(self.run_seq([(3, {}), (1.7, self.TO_RIGHT), (1, {})]), [])
        # The way the mouth went decides, not which expression score rose.
        self.assertEqual(self.run_seq([(3, {}), (1.7, {"mouthLeft": 0.6, "mouthShift": 0.06})]),
                         ["mouth_left"])
        # No measured shift (or too small to tell): no side gesture at all.
        self.assertEqual(self.run_seq([(3, {}), (3.3, {"mouthRight": 0.6, "mouthShift": 0.005})]), [])
        self.assertEqual(self.run_seq([(3, {"mouthShift": None}), (3.3, {"mouthRight": 0.6})]), [])
        # A face whose mouth rests off-centre: the shift is from where it rests.
        crooked = {"mouthShift": 0.04}
        self.assertEqual(self.run_seq([(3, crooked), (1.7, {**self.TO_LEFT, "mouthShift": 0.045})]), [])
        self.assertEqual(self.run_seq([(3, crooked), (1.7, {**self.TO_LEFT, "mouthShift": 0.1})]),
                         ["mouth_left"])
        # Swinging from one side to the other doesn't finish the first gesture.
        self.assertEqual(self.run_seq([(3, {}), (1.0, self.TO_LEFT), (0.6, self.TO_RIGHT), (1, {})]), [])

    def test_this_users_pulls(self):
        """Measured 2026-10-20 (MOUTH_PULL_FW): MediaPipe scores this user's pull to
        the left as a lopsided smile, and wavers on their steady pull to the right."""
        lopsided = {"mouthSmileLeft": 0.78, "mouthSmileRight": 0.45, "mouthShift": 0.03}
        self.assertEqual(self.run_seq([(3, {}), (1.7, lopsided), (1, {})]), ["mouth_left"])  # no pause
        wavering = [(0.2, {"mouthRight": v, "mouthSmileRight": 0.1, "mouthShift": -0.04})
                    for v in (0.6, 0.12, 0.5, 0.18, 0.7, 0.1, 0.45, 0.3) * 2]
        self.assertEqual(self.run_seq([(3, {})] + wavering + [(1, {})]), ["mouth_right"])
        # A big smile is even, even if the mouth shifts a little: still pause.
        even = {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.88, "mouthShift": 0.015}
        self.assertEqual(self.run_seq([(3, {}), (1.7, even), (1, {})]), ["smile"])
        # The landmarks alone (no score on that side) are not a pull: a face shifting.
        self.assertEqual(self.run_seq([(3, {}), (3.3, {"mouthShift": -0.05})]), [])
        # An open mouth (talking) is not a pull, whatever the scores say.
        self.assertEqual(self.run_seq([(3, {}), (3.3, {"mouthRight": 0.6, "mouthShift": -0.05,
                                                       "jawOpen": 0.3})]), [])

    def test_a_lopsided_smile_never_starts_calibrate(self):
        """1.5.11's log (2026-10-28): "Mouth to your right: calibrate" beeped 3 times,
        2-4 s before a smile pause or a tilt, never meant. A smile rising more on
        one side, the mouth drifting with it, is no pull without a pull score."""
        lopsided = {"mouthSmileLeft": 0.15, "mouthSmileRight": 0.45, "mouthShift": -0.03}
        g, t, seen = li.FaceGestures(), 0.0, set()
        for secs, over in ((3, {}), (3.3, lopsided), (1, {})):
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                self.assertIsNone(g.update({**self.REST, **over}, t))
                seen.add(g.active)
        self.assertNotIn("mouth_right", seen)
        even = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.75}
        self.assertEqual(self.run_seq([(3, {}), (1.0, lopsided), (1.7, even), (1, {})]), ["smile"])
        # A real pull still starts it, and the log says what was seen.
        g = li.FaceGestures()
        for k in range(int(4 * FPS)):
            g.update({**self.REST, **(self.TO_RIGHT if k >= 3 * FPS else {})}, (k + 1) / FPS)
        self.assertEqual(g.active, "mouth_right")
        self.assertIn("to your right, mouthLeft 0.60", g.detail)

    def test_the_face_eases_out_of_a_gesture_before_blinks_count(self):
        """1.5.2's log: a "left wink" click the moment a smile resumed EyeSync."""
        g, t = li.FaceGestures(), 0.0
        smile = {**self.REST, "mouthSmileLeft": 0.8, "mouthSmileRight": 0.75}
        easing = {**self.REST, "mouthSmileLeft": 0.45, "mouthSmileRight": 0.4}
        relaxed = []
        for secs, bs in ((3, self.REST), (1.7, smile), (1.0, easing), (0.5, self.REST)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                g.update(bs, t)
                relaxed.append(g.relaxed)
        n = int(3 * FPS)
        self.assertTrue(all(relaxed[:n + 40]))                 # resting, then holding the smile
        self.assertFalse(any(relaxed[n + 46:n + 81]))          # fired: easing out of it
        self.assertTrue(relaxed[-1])                           # rested again
        # A face that never quite rests doesn't block blinks for long.
        g.update(smile, t + 10)
        self.assertTrue(g.relaxed)

    def test_a_head_turn_is_not_a_pull(self):
        # Turning moves fe.mouth_shift (-0.196 x sin(yaw) on this user's face).
        g, t, fired = li.FaceGestures(), 0.0, []
        for secs, yaw in ((3, 0.0), (3.3, 20.0), (3.3, -20.0), (1, 0.0)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                turned = {**self.REST, "mouthShift": -0.196 * np.sin(np.radians(yaw)),
                          "mouthRight": 0.2 if yaw < 0 else 0.0, "mouthLeft": 0.2 if yaw > 0 else 0.0}
                if f := g.update(turned, t, (yaw, 0.0)):
                    fired.append(f)
        self.assertEqual(fired, [])

    def test_mouth_shift_measurement(self):
        lm = np.zeros((478, 2), np.float32)
        lm[33], lm[263] = (0.40, 0.40), (0.60, 0.40)       # the user's right / left outer eye corner
        lm[fe.SUBNASALE] = (0.50, 0.55)
        for off, want in ((0.0, 0.0), (0.02, 0.1), (-0.03, -0.15)):
            lm[61], lm[291] = (0.45 + off, 0.62), (0.55 + off, 0.62)
            # (x is scaled by 640 and the face is 0.2 * 640 = 128 px wide)
            self.assertAlmostEqual(fe.mouth_shift(lm, 640, 480), want, places=4)

    def test_holding_on_fires_only_once(self):
        smile = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
        self.assertEqual(self.run_seq([(3, {}), (5, smile)]), ["smile"])
        self.assertEqual(self.run_seq([(3, {}), (1.7, smile), (0.5, {}), (1.7, smile)]),
                         ["smile", "smile"])

    def test_short_expressions_do_nothing(self):
        smile = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
        self.assertEqual(self.run_seq([(3, {}), (1.0, smile), (1, {}), (0.8, {"mouthPucker": 0.8})]), [])

    def test_talking_does_nothing(self):
        rng = np.random.default_rng(2)
        segs = [(3, {})] + [(0.2, {"mouthPucker": float(rng.uniform(0.1, 0.66)),
                                   "mouthSmileLeft": float(rng.uniform(0, 0.3)),
                                   "mouthLeft": float(rng.uniform(0, 0.2))}) for _ in range(60)]
        self.assertEqual(self.run_seq(segs), [])

    def test_brief_sag_does_not_cancel(self):
        smile = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}
        self.assertEqual(self.run_seq([(3, {}), (0.8, smile), (0.2, {}), (0.9, smile)]), ["smile"])

    def test_head_tilts_are_not_hold_gestures_any_more(self):
        """Tilts scroll now (TiltScroll); they no longer recenter or calibrate."""
        self.assertNotIn("tilt_left", li.GESTURES)
        self.assertEqual(self.run_seq([(3, {}), (1.7, {"headRoll": 20.0}), (1, {})]), [])

    def test_head_roll_measurement(self):
        lm = np.zeros((478, 2), np.float32)
        for angle in (0.0, 15.0, -20.0):
            a = np.radians(angle)
            lm[fe.EYE_OUTER[0]] = (0.40, 0.50)                                   # user's right eye
            lm[fe.EYE_OUTER[1]] = (0.40 + 0.2 * np.cos(a), 0.50 + 0.2 * np.sin(a))   # left eye lower
            self.assertAlmostEqual(fe.head_roll(lm, 640, 640), angle, places=3)

    def test_light_smile_does_nothing(self):
        light = {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.45, "mouthRight": 0.3}
        self.assertEqual(self.run_seq([(3, {}), (4, light), (1, {})]), [])

    def test_big_smile_works_with_the_head_turned(self):
        """At a screen edge the far half of the mouth is scored low."""
        g, t, fired = li.FaceGestures(), 0.0, []
        smile = {**self.REST, "mouthSmileLeft": 0.85, "mouthSmileRight": 0.3}
        for secs, bs in ((3, self.REST), (1.7, smile), (1, self.REST)):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                if f := g.update(bs, t, (24.0, -5.0)):
                    fired.append(f)
        self.assertEqual(fired, ["smile"])
        # Facing the camera, a lopsided half-smile like that is not a big smile.
        self.assertEqual(self.run_seq([(3, {}), (1.7, {"mouthSmileLeft": 0.85,
                                                          "mouthSmileRight": 0.3})]), [])

    def test_resting_levels_are_learned_per_head_direction(self):
        """Turned to a side, this face's resting side-mouth score is 0.25."""
        g, t, fired = li.FaceGestures(), 0.0, []
        side_rest = {**self.REST, "mouthRight": 0.25}
        pull = {**self.REST, "mouthRight": 0.5, "mouthShift": 0.06}
        for secs, bs, pose in ((3, self.REST, (0.0, 0.0)), (3, side_rest, (24.0, 0.0)),
                               (1.7, pull, (24.0, 0.0)), (1.0, self.REST, (0.0, 0.0)),
                               (1.7, pull, (0.0, 0.0))):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                if f := g.update(bs, t, pose):
                    fired.append((f, pose[0]))
        # 0.5 is only 0.25 above this face's level when turned (not a pull), a clear one ahead.
        self.assertEqual(fired, [("mouth_left", 0.0)])


# --------------------------------------------------------------------------- #
# Dwell -> click
# --------------------------------------------------------------------------- #

class DwellTests(unittest.TestCase):
    """Paths are (seconds, x, moving) segments; the cursor starts resting at x=100."""

    def setUp(self):
        self.t = 0.0
        self.d = li.DwellClicker(1.0, 60.0)

    def run_path(self, segments, start=True) -> int:
        if start:
            segments = [(0.3, 100, False)] + segments
        clicks = 0
        for secs, x, moving in segments:
            for _ in range(int(round(secs * FPS))):
                self.t += 1 / FPS
                clicks += self.d.update(np.array([x, 500.0]), moving, self.t)
        return clicks

    def test_move_then_rest_clicks_once(self):
        self.assertEqual(self.run_path([(0.3, 300, True), (4.0, 300, False)]), 1)

    def test_never_moved_never_clicks(self):
        self.assertEqual(self.run_path([(5.0, 100, False)]), 0)

    def test_small_moves_do_not_rearm(self):
        self.assertEqual(self.run_path([(0.3, 300, True), (1.2, 300, False),
                                        (0.2, 330, True), (2.0, 330, False)]), 1)

    def test_moving_away_rearms(self):
        self.assertEqual(self.run_path([(0.3, 300, True), (1.2, 300, False),
                                        (0.3, 450, True), (1.2, 450, False)]), 2)

    def test_short_rest_does_not_click(self):
        self.assertEqual(self.run_path([(0.3, 300, True), (0.8, 300, False), (0.3, 500, True)]), 0)

    def test_blink_restarts_timer(self):
        self.assertEqual(self.run_path([(0.3, 300, True), (0.7, 300, False)]), 0)
        self.d.pause()                                     # a blink froze the cursor
        self.assertEqual(self.run_path([(0.7, 300, False)], start=False), 0)
        self.assertEqual(self.run_path([(0.4, 300, False)], start=False), 1)


# --------------------------------------------------------------------------- #
# The EyeSync window
# --------------------------------------------------------------------------- #

class WindowTests(unittest.TestCase):
    def test_status_keys(self):
        import main_window
        screen = fe.ScreenSize(1920, 1080)
        args = types.SimpleNamespace(always_active=False)
        st = li.SharedState(screen, 2, False)
        self.assertEqual(li.status_of(st.read(), args)[0], "off")
        st.update(site_clients=1)
        self.assertEqual(li.status_of(st.read(), args)[0], "off")         # camera starting
        st.update(has_position=True, face=True)
        self.assertEqual(li.status_of(st.read(), args)[0], "active")
        st.update(face=False)
        self.assertEqual(li.status_of(st.read(), args)[0], "noface")
        st.update(paused=True)
        self.assertEqual(li.status_of(st.read(), args)[0], "paused")
        for key in ("off", "active", "noface", "paused"):
            self.assertIn(key, main_window.COLORS)

    def test_controls_drive_the_engine(self):
        import tkinter as tk

        import main_window
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        saved = li.save_settings
        li.save_settings = lambda **kw: None              # don't touch the real settings file
        tmp = tempfile.TemporaryDirectory()
        try:
            root.withdraw()
            win = main_window.MainWindow(root, False, PROJECT / "README.md", PROJECT)
            st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
            args = types.SimpleNamespace(always_active=False)
            api_key = li.ApiKey(Path(tmp.name) / "key.txt")
            win.attach(st, li.Actions(st, api_key), lambda: li.status_of(st.read(), args))

            key = api_key.get()
            self.assertNotIn(key, win.key_var.get())       # shown masked
            self.assertTrue(win.key_var.get().endswith(key[-4:]))
            copied = []
            root.clipboard_clear = lambda: None           # leave the user's clipboard alone
            root.clipboard_append = copied.append
            win.copy_key_btn.invoke()
            self.assertEqual(copied, [key])

            win.pause_btn.invoke()
            self.assertTrue(st.read().paused)
            win.refresh()
            self.assertEqual(win.pause_btn.cget("text"), "Resume")
            win.recenter_btn.invoke()
            self.assertTrue(st.recenter_request.is_set())
            win.sens_var.set(5)
            win._set_sensitivity()
            self.assertEqual(st.read().sensitivity, 5)
            win.click_var.set("dwell")
            win._set_click()
            self.assertEqual(st.read().click_mode, "dwell")

            st.update(hint="look at the centre")
            st.set_preview(np.zeros((main_window.PREVIEW_H, main_window.PREVIEW_W, 3), np.uint8))
            win.refresh()
            self.assertEqual(win.hint_var.get(), "look at the centre")
            self.assertIsNotNone(win._photo)               # camera image shown
            self.assertEqual(win.status_var.get(), li.status_of(st.read(), args)[1])

            # Calibrate: the guide opens (hidden here, not over the real screen),
            # follows the steps, and Cancel closes it.
            ov = win.overlay

            def hidden_show():
                ov.top = tk.Toplevel(root)
                ov.top.withdraw()
                ov.canvas = tk.Canvas(ov.top)
                ov.drawn = None
            ov._show = hidden_show
            self.assertIn("Not calibrated", win.calib_var.get())
            win.calibrate_btn.invoke()
            self.assertTrue(st.calibrate_request.is_set())
            win.refresh()
            self.assertIsNotNone(ov.top)
            st.update(calib_step="left", calib_text="Turn your head...", calib_progress=0.5)
            win.refresh()
            self.assertEqual(ov.drawn, ("left", "Turn your head...", 0.5))
            win.calibrate_btn.invoke()                     # pressed again: cancel
            self.assertTrue(st.calibrate_cancel.is_set())
            win.refresh()
            self.assertIsNone(ov.top)
            st.update(calib_step="done", calib_text="Calibration done!", calibrated="2026-10-15")
            win.refresh()
            self.assertIn("2026-10-15", win.calib_var.get())
            ov.done_since = time.monotonic() - ov.RESULT_S - 1   # the result was shown long enough
            win.refresh()
            win.refresh()
            self.assertEqual(st.read().calib_step, "")
            self.assertIsNone(ov.top)
        finally:
            li.save_settings = saved
            root.destroy()
            tmp.cleanup()


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

class CalibrationTests(unittest.TestCase):
    FACE, CX, CY = 100.0, 320.0, 240.0          # face 100 px wide, resting nose position
    OPEN = ((0.28, 0.28), (0.03, 0.03))         # open eyes: EAR, blink scores

    def frame(self, dx=0.0, dy=0.0, eyes=None, smile=0.01, yaw=0.0, pitch=0.0):
        """Head moved dx/dy face widths in CURSOR directions (+x = right)."""
        ear, sc = eyes or self.OPEN
        nose = np.array([self.CX - dx * self.FACE, self.CY + dy * self.FACE])   # image x mirrored
        return li.CalibFrame(nose, self.FACE, (yaw, pitch), ear, sc, smile)

    def run_cal(self, script, seed=4):
        """script: (seconds, frame kwargs or a list of them per frame) -> the Calibration."""
        rng = np.random.default_rng(seed)
        cal, t = li.Calibration(0.0), 0.0
        for secs, kw in script:
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                f = self.frame(**kw)
                if f is not None:
                    f.nose = f.nose + rng.normal(0, 0.4, 2)          # the user's camera noise
                cal.update(f, t)
        return cal

    def head_script(self, x=0.3, y=0.12, down_eyes=((0.2, 0.2), (0.4, 0.4))):
        return [(2.5, {}),
                (2.0, dict(dx=-x, yaw=15)), (2.0, dict(dx=x, yaw=-15)),
                (2.0, dict(dy=-y, pitch=8)), (2.0, dict(dy=y, pitch=-8, eyes=down_eyes))]

    BLINK = [(0.6, {}), (0.4, dict(eyes=((0.06, 0.06), (0.9, 0.9))))] * 3 + [(0.6, {})]
    SMILE = [(1.5, dict(smile=0.8))]

    def test_full_calibration(self):
        cal = self.run_cal(self.head_script() + self.BLINK + self.SMILE)
        self.assertTrue(cal.finished, cal.step)
        res = cal.results()
        self.assertAlmostEqual(res["head_x"], 0.6, delta=0.03)
        self.assertAlmostEqual(res["head_y"], 0.24, delta=0.02)
        self.assertEqual(res["turn_deg"], 30)
        self.assertAlmostEqual(res["blink_close_ratio"], 0.06 / 0.28 + 0.4 * (1 - 0.06 / 0.28), delta=0.02)
        self.assertEqual(res["blink_score_delta"], 0.4)      # deep, clear blinks: harder to fake
        self.assertGreaterEqual(res["blink_tick_s"], li.BLINK_MIN_S)
        self.assertLess(res["blink_tick_s"], 0.4)            # below the demonstrated 0.4 s holds
        self.assertAlmostEqual(res["smile_min"], 0.6, delta=0.01)
        self.assertEqual(cal.skipped, [])
        # Every head direction taught the blink detector how open eyes look there.
        self.assertEqual({tuple(f.pose) for f in cal.learned},
                         {(0.0, 0.0), (15, 0.0), (-15, 0.0), (0.0, 8), (0.0, -8)})

    def test_slow_calibration_blinks_never_delay_the_tick(self):
        """The guide asks for slow, half-second blinks; the user's own clicks are
        quicker (2026-10-28: 366 and 396 ms closures opened before a 0.36-0.45 s
        calibrated tick). Since 1.5.13 the tick stays at the default."""
        closed = dict(eyes=((0.06, 0.06), (0.9, 0.9)))
        for secs in (0.4, 0.7):
            with self.subTest(secs):
                slow = [(0.6, {}), (secs, closed)] * 3 + [(0.6, {})]
                cal = self.run_cal(self.head_script() + slow + self.SMILE)
                self.assertEqual(len(cal.blinks), 3)
                self.assertEqual(cal.results()["blink_tick_s"], li.BLINK_MIN_S)

    # A real blink: the lids pass "a quarter down" (ratio 0.7) on the way, stay
    # shut, then reopen slowly through ratio 0.8 - all part of the closure.
    RAMP_DOWN = dict(eyes=((0.196, 0.196), (0.3, 0.3)))
    SHUT = dict(eyes=((0.06, 0.06), (0.9, 0.9)))
    RAMP_UP = dict(eyes=((0.224, 0.224), (0.3, 0.3)))

    def test_the_tick_comes_while_the_demonstrated_blink_is_shut(self):
        """1.5.9 set the tick to 90% of the whole closure (lids a quarter down until
        fully open), up to ~0.2 s longer than the eyes are shut - and the tick
        comes only while they are (TICK_WHEN_SHUT): the very blink shown didn't
        click. Now 90% of the time shut, so it does."""
        blink = [(0.6, {}), (0.1, self.RAMP_DOWN), (0.5, self.SHUT), (0.167, self.RAMP_UP)]
        cal = self.run_cal(self.head_script() + blink * 3 + [(0.6, {})] + self.SMILE)
        res = cal.results()
        self.assertAlmostEqual(res["blink_tick_s"], min(0.9 * 14 / FPS, li.BLINK_CALIBRATED_MAX_S), delta=0.01)
        old = float(np.clip(0.9 * np.median([b[2] for b in cal.blinks]), li.BLINK_MIN_S, 0.65))
        self.assertGreater(old, 0.6)                          # what 1.5.9 saved

        def clicks(min_s):
            b = li.BlinkDetector(None, 0.55, 0.70, li.BLINK_MIN_S, li.BLINK_MAX_S, 0.6)
            li.apply_calibration(res, li.HeadPointer(1920, 1080), b, li.FaceGestures())
            if min_s is not None:
                b.min_s = min_s
            b.baseline = 0.28
            seq = ([0.28] * 30 + [0.196] * 3 + [0.06] * 15 + [0.224] * 5 + [0.28] * 30)
            out, t = [], 0.0
            for ear in seq:
                t += 1 / FPS
                click = b.update(ear, ear, t, None, True)[1]
                if click:
                    out.append(click)
            return out

        self.assertEqual(clicks(None), ["left"])
        self.assertEqual(clicks(old), [])                     # 1.5.9: no tick, no click

    def test_glasses_blinks_fit_the_blink_score(self):
        """Eyelid points barely move behind glasses; the score rises only ~0.3."""
        weak = [(0.6, {}), (0.4, dict(eyes=((0.25, 0.25), (0.33, 0.33))))] * 3 + [(0.6, {})]
        res = self.run_cal(self.head_script() + weak + self.SMILE).results()
        self.assertNotIn("blink_close_ratio", res)               # eyelids: keep the default
        self.assertAlmostEqual(res["blink_score_delta"], 0.15, delta=0.01)

    def test_steps_not_done_are_skipped(self):
        script = [(2.5, {}), (2.0, dict(dx=-0.3)), (2.0, dict(dx=0.3)),
                  (21.0, {}),                                    # never looks up...
                  (2.0, dict(dy=0.12))] + self.BLINK + [(21.0, {})]   # ...nor smiles
        cal = self.run_cal(script)
        self.assertTrue(cal.finished, cal.step)
        res = cal.results()
        self.assertEqual(cal.skipped, ["up", "smile"])
        self.assertAlmostEqual(res["head_y"], 0.24, delta=0.02)  # twice the bottom
        self.assertNotIn("smile_min", res)

    def test_face_lost_waits(self):
        cal = li.Calibration(0.0)
        for i in range(90):
            cal.update(None, (i + 1) / FPS)
        self.assertEqual(cal.step, "centre")
        self.assertIn("Can't see your face", cal.text())

    def test_calibrated_pointer_and_settings(self):
        screen = fe.ScreenSize(1920, 1080)
        hp = li.HeadPointer(screen.width, screen.height)
        blink = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        gestures = li.FaceGestures()
        default = [hp.gain(lvl, s) for lvl in (1, 3) for s in (0.1, 2.0)]
        # A saved calibration with the defaults' own range changes nothing...
        li.apply_calibration({"head_x": 0.8, "head_y": 0.8 * (1080 / 1920) / li.HEAD_Y_GAIN},
                             hp, blink, gestures)
        np.testing.assert_allclose([hp.gain(lvl, s) for lvl in (1, 3) for s in (0.1, 2.0)], default)
        # ...a user whose natural range is 0.65 x 0.28 face widths crosses the
        # screen with exactly that at speed 1, at a normal pace.
        li.apply_calibration({"head_x": 0.65, "head_y": 0.28, "blink_close_ratio": 0.6,
                              "blink_open_ratio": 0.75, "blink_score_delta": 0.15,
                              "smile_min": 0.5}, hp, blink, gestures)
        g = hp.gain(1, 2.0)
        self.assertAlmostEqual(abs(g[0]) * 0.65, 1920)
        self.assertAlmostEqual(g[1] * 0.28, 1080)
        self.assertGreater(abs(hp.gain(3, 2.0)[0]), abs(g[0]))   # higher speeds still go faster
        self.assertEqual((blink.close_ratio, blink.open_ratio, blink.score_delta), (0.6, 0.75, 0.15))
        self.assertEqual(gestures.threshold("smile"), 0.5)
        li.apply_calibration(None, hp, blink, gestures)              # back to the defaults
        self.assertIsNone(hp.calib)
        self.assertEqual(gestures.threshold("smile"), li.GESTURES["smile"][1])

    def test_explicit_blink_timing_overrides_calibration(self):
        hp = li.HeadPointer(1920, 1080)
        blink = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        gestures = li.FaceGestures()
        cal = {"blink_tick_s": 0.361}                    # the user's saved 1.5.11 calibration
        li.apply_calibration(cal, hp, blink, gestures, timing=False)
        self.assertEqual(blink.min_s, 0.2)
        li.apply_calibration(cal, hp, blink, gestures)
        self.assertEqual(blink.min_s, li.BLINK_MIN_S)    # used as it is: no new calibration needed
        li.apply_calibration({"blink_tick_s": 0.6}, hp, blink, gestures)
        self.assertEqual(blink.min_s, li.BLINK_CALIBRATED_MAX_S)
        # 1.5.9's value (measured over the whole closure) isn't used.
        blink.min_s = li.BLINK_MIN_S
        li.apply_calibration({"blink_min_s": 0.6}, hp, blink, gestures)
        self.assertEqual(blink.min_s, li.BLINK_MIN_S)

    def test_a_lopsided_calibration_only_fine_tunes(self):
        """A real user's calibration: up-down 0.69 face widths, 2.2x the default (one
        edge rushed, one exaggerated). It may change the speed by 30% at most."""
        hp = li.HeadPointer(1920, 1080)
        li.apply_calibration({"head_x": 0.712, "head_y": 0.691}, hp,
                             li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6), li.FaceGestures())
        default_y = 0.8 * (1080 / 1920) / li.HEAD_Y_GAIN
        self.assertAlmostEqual(hp.calib[0], 0.712)                   # within range: kept
        self.assertAlmostEqual(hp.calib[1], 1.3 * default_y)         # limited

    def test_a_pause_on_the_way_is_not_the_edge(self):
        """The head pausing at 0.15 face widths on its way left isn't the left edge."""
        script = [(2.5, {}), (2.0, dict(dx=-0.15)), (2.0, dict(dx=-0.35)), (2.0, dict(dx=0.3)),
                  (2.0, dict(dy=-0.12)), (2.0, dict(dy=0.12))] + self.BLINK + self.SMILE
        cal = self.run_cal(script)
        self.assertTrue(cal.finished, cal.step)
        self.assertAlmostEqual(cal.edges["left"][0], 0.35, delta=0.02)

    def test_summary(self):
        text = li.calibration_summary({"head_x": 0.5, "turn_deg": 28, "smile_min": 0.6}, ["blink"])
        self.assertIn("28 degrees", text)
        self.assertIn("Skipped: blink", text)
        self.assertIn("the tick comes at 0.42 s",
                      li.calibration_summary({"blink_close_ratio": 0.5, "blink_tick_s": 0.42}, []))


# --------------------------------------------------------------------------- #
# Smooth cursor output
# --------------------------------------------------------------------------- #

class CursorGlideTests(unittest.TestCase):
    def run_glide(self, positions, cam_fps=30.0, jitter_s=0.004, seed=1):
        """Camera positions at cam_fps (arriving a little late, at random) ->
        the cursor at li.OUTPUT_HZ: list of (time, x)."""
        rng = np.random.default_rng(seed)
        g, out = li.CursorGlide(), []
        frames = [(i / cam_fps, x, i / cam_fps + rng.uniform(0, jitter_s))
                  for i, x in enumerate(positions)]
        t, k, latest = 0.0, 0, None
        while t < len(positions) / cam_fps + 0.1:
            while k < len(frames) and frames[k][2] <= t:
                latest = frames[k]
                k += 1
            if latest is not None:
                g.target(latest[1], 500.0, latest[0], t)
                out.append((t, float(g.at(t)[0])))
            t += 1 / li.OUTPUT_HZ
        return out

    def test_a_head_turn_moves_the_cursor_evenly(self):
        speed = 2400.0                                       # px/s, a normal turn
        xs = [100.0] * 10 + [100 + speed * i / 30 for i in range(1, 31)] + [100 + speed] * 10
        out = self.run_glide(xs)
        steps = np.diff([x for _, x in out])
        moving = steps[np.abs(steps) > 0.01]
        per_tick = speed / li.OUTPUT_HZ                      # 20 px
        # Before: one 80 px jump per camera frame. Now ~20 px every 8 ms.
        self.assertLess(moving.max(), per_tick * 1.8)
        self.assertLess(np.percentile(moving, 90), per_tick * 1.3)
        self.assertAlmostEqual(out[-1][1], 100 + speed, places=3)   # arrives exactly

    def test_still_cursor_stays_still(self):
        out = self.run_glide([700.0] * 60)
        self.assertEqual({x for _, x in out}, {700.0})

    def test_lag_is_at_most_a_frame(self):
        xs = [100.0] * 5 + [900.0] * 10
        out = self.run_glide(xs, jitter_s=0.0)
        arrived = next(t for t, x in out if x == 900.0)
        self.assertLessEqual(arrived - 5 / 30, 0.05 + 1 / li.OUTPUT_HZ)

    def test_click_lands_on_the_target(self):
        g = li.CursorGlide()
        g.target(100, 100, 0.0, 0.0)
        g.target(400, 100, 1 / 30, 1 / 30)
        g.finish()
        self.assertEqual(list(g.at(1 / 30 + 0.001)), [400.0, 100.0])


# --------------------------------------------------------------------------- #
# Close gesture
# --------------------------------------------------------------------------- #

class CloseGestureTests(unittest.TestCase):
    def worker(self):
        w = li.VisionWorker.__new__(li.VisionWorker)
        w.args = types.SimpleNamespace(no_close_gesture=False, close_gesture_mar=0.6,
                                       close_gesture_hold=3.0)
        w.gesture_since = w.gesture_low_since = None
        w.gesture_announced = False
        w.state = types.SimpleNamespace(stop=threading.Event())
        return w

    def quits(self, seq, jaw=None) -> bool:
        """seq of (mouth aspect ratio, eyes open) at 30 FPS -> did EyeSync shut down?"""
        w, t = self.worker(), 0.0
        for mar, eyes_open in seq:
            t += 1 / FPS
            w._close_gesture(mar, eyes_open, t, jaw)
        return w.state.stop.is_set()

    def test_brief_wide_opening_does_not_even_beep(self):
        w, t, beeps = self.worker(), 0.0, []
        old, li.desktop.play_warning_sound = li.desktop.play_warning_sound, lambda: beeps.append(1)
        try:
            for mar in [0.7] * 12 + [0.05] * 30:       # 0.4 s wide open, e.g. a breath
                t += 1 / FPS
                w._close_gesture(mar, True, t)
        finally:
            li.desktop.play_warning_sound = old
        self.assertEqual(beeps, [])
        self.assertFalse(w.state.stop.is_set())

    def test_jaw_not_dropped_does_not_start(self):
        """The ratio reads wide open but MediaPipe sees the jaw closed (e.g. head turned)."""
        self.assertFalse(self.quits([(0.7, True)] * 120, jaw=0.15))
        self.assertTrue(self.quits([(0.7, True)] * 96, jaw=0.6))

    def test_head_pose_and_turned_mouth(self):
        def rot(yaw, pitch):
            a, b = np.radians(yaw), np.radians(pitch)
            ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            m = np.eye(4)
            m[:3, :3] = ry @ rx
            return m
        for yaw, pitch in ((0, 0), (20, 0), (-15, 10), (25, -12)):
            got = fe.head_pose(rot(yaw, pitch))
            self.assertAlmostEqual(abs(got[0]), abs(yaw), delta=1.5)
            self.assertAlmostEqual(abs(got[1]), abs(pitch), delta=1.5)
        # An ordinary opening (0.45) seen with the head turned 30 degrees: the
        # mouth looks narrower, so the plain ratio reads ~0.52; the pose undoes it.
        lm = np.zeros((478, 2), np.float32)
        lm[fe.MOUTH_LEFT], lm[fe.MOUTH_RIGHT] = (0.45, 0.6), (0.45 + 0.1 * np.cos(np.radians(30)), 0.6)
        lm[fe.UPPER_LIP], lm[fe.LOWER_LIP] = (0.5, 0.6), (0.5, 0.6 + 0.045)
        self.assertGreater(fe.mouth_aspect_ratio(lm, 640, 640), 0.51)
        self.assertAlmostEqual(fe.mouth_aspect_ratio(lm, 640, 640, (30.0, 0.0)), 0.45, places=2)

    def test_talking_does_not_quit(self):
        self.assertFalse(self.quits([(0.45 * abs(np.sin(i / 3)), True) for i in range(600)]))

    def test_yawn_does_not_start(self):
        self.assertFalse(self.quits([(0.8, False)] * 150))       # eyes squeezed shut

    def test_short_open_cancels(self):
        self.assertFalse(self.quits([(0.7, True)] * 60 + [(0.05, True)] * 90))

    def test_held_open_quits(self):
        self.assertTrue(self.quits([(0.7, True)] * 96))

    def test_wavering_and_squinting_still_quits(self):
        # What a real held-open mouth looks like: dips below the start level,
        # brief deeper dips, eyes squinting after the start.
        seq = [(0.7, True)] * 10
        for i in range(90):
            mar = 0.5 if i % 15 < 9 else (0.35 if i % 15 < 12 else 0.65)
            seq.append((mar, i % 4 != 0))
        self.assertTrue(self.quits(seq))

    def test_half_second_sag_still_quits(self):
        # Seen in a real log: tiring after ~1-2 s and the opening sagging
        # for half a second cancelled the gesture six times in a row.
        seq = [(0.7, True)] * 40 + [(0.30, True)] * 15 + [(0.55, False)] * 50
        self.assertTrue(self.quits(seq))

    def test_closing_mouth_cancels(self):
        self.assertFalse(self.quits([(0.7, True)] * 45 + [(0.1, True)] * 15 + [(0.7, False)] * 60))


# --------------------------------------------------------------------------- #
# Cursor motion
# --------------------------------------------------------------------------- #

class CursorMotionTests(unittest.TestCase):
    def simulate(self, level: int, noise: float, seed: int = 3):
        """Head still 3 s, turns 800 px over 1 s, then still for 3.5 s."""
        rng = np.random.default_rng(seed)
        mc, beta, start, stop, vmax = li.SENSITIVITY_LEVELS[level]
        f, m = li.OneEuroFilter(mc, beta), li.CursorMotion(start, stop, vmax)
        pos = []
        for i in range(225):
            t = i / FPS
            u = min(max(t - 3.0, 0.0), 1.0)
            true_x = 400 + 800 * (3 * u * u - 2 * u * u * u)
            pos.append(m(f(np.array([true_x, 540.0]) + rng.normal(0, noise, 2), t), t).copy())
        return np.array(pos)

    def steady_turn(self, speed_fw, seed, level=1):
        """The whole pointer (nose -> HeadPointer -> One Euro -> CursorMotion) during a
        steady head turn with the user's webcam noise -> (unevenness, overshoot px)."""
        rng = np.random.default_rng(seed)
        mc, beta, start, stop, vmax = li.SENSITIVITY_LEVELS[level]
        f, m = li.OneEuroFilter(mc, beta), li.CursorMotion(start, stop, vmax)
        hp = li.HeadPointer(1920, 1080)
        hp.reset(np.array([200.0, 540.0]))
        hist, xs = deque(maxlen=li.NOSE_FRAMES), []
        secs = min(0.4 / speed_fw, 2.5)
        for i in range(int((4 + secs) * FPS)):
            t = i / FPS
            nose = np.array([400.0 - min(max(t - 2, 0), secs) * speed_fw * 95, 240.0])
            hist.append(nose + rng.normal(0, 0.45, 2))
            pos = m(f(hp.target(li.nose_smooth(hist), 95.0, level, t), t), t).copy()
            m.pos = pos
            xs.append(pos[0])
        xs = np.array(xs)
        v = np.abs(np.diff(xs))[int((2 + 0.4 * secs) * FPS):int((2 + secs) * FPS)]
        return float(np.std(v) / np.mean(v)), float(max(0.0, xs.max() - xs[-1]))

    def test_steady_turn_moves_the_cursor_evenly(self):
        """Webcam noise is as big as a slow turn's per-frame movement; the cursor's speed
        must still be even (it wobbled ~30% before the movement was averaged)."""
        uneven = np.mean([self.steady_turn(0.15, s)[0] for s in range(4)])
        self.assertLess(uneven, 0.27)
        uneven = np.mean([self.steady_turn(0.6, s)[0] for s in range(4)])
        self.assertLess(uneven, 0.15)

    def test_cursor_does_not_coast_past_where_the_head_stopped(self):
        self.assertLess(max(self.steady_turn(0.6, s)[1] for s in range(4)), 12.0)

    def test_lands_and_stops(self):
        # Noise = jitter (px, std) of the head target; the head pointer's own
        # is ~4-10 px, 25 px is a very noisy webcam.
        for level in li.SENSITIVITY_LEVELS:
            for noise in (8, 25):
                for seed in (1, 2, 3):
                    with self.subTest(level=level, noise=noise, seed=seed):
                        pos = self.simulate(level, noise, seed)
                        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
                        moving = np.flatnonzero(speed > 1)
                        stop_s = (moving[-1] + 1) / FPS - 4.0
                        self.assertLess(stop_s, 1.3, "cursor kept moving after the head stopped")
                        self.assertLess(abs(pos[-1][0] - 1200), 40, "cursor stopped off target")
                        self.assertLess(np.ptp(pos[-40:], axis=0).max(), 5, "cursor wobbles at rest")

    def test_cursor_keeps_up_and_higher_levels_need_less_turning(self):
        # A cursor that trails the head makes you keep turning, then overshoots:
        # at every level it must arrive with the head (the turn ends at 4.0 s).
        for level in sorted(li.SENSITIVITY_LEVELS):
            pos = self.simulate(level, 15)
            arrive = np.flatnonzero(np.abs(pos[:, 0] - 1200) < 50)[0] / FPS
            self.assertLess(arrive, 4.1, f"cursor trails the head at level {level}")
        ranges = [li.HEAD_RANGE[level] for level in sorted(li.HEAD_RANGE)]
        self.assertEqual(ranges, sorted(ranges, reverse=True))


# --------------------------------------------------------------------------- #
# Website protocol
# --------------------------------------------------------------------------- #

class WebsiteTests(unittest.TestCase):
    def test_origin_matching(self):
        allowed = li.DEFAULT_SITE_ORIGINS
        site = allowed[0]
        self.assertTrue(li.is_website_origin(site, allowed))
        self.assertTrue(li.is_website_origin(site.replace("https", "http") + ":443", allowed))
        self.assertFalse(li.is_website_origin("https://evil.example.com", allowed))
        self.assertFalse(li.is_website_origin("", allowed))
        self.assertFalse(li.is_website_origin("null", allowed))

    def test_websocket_payload_and_activation(self):
        from websockets.asyncio.client import connect

        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 2, False)
        state.update(x=float("nan"), y=100.0, face=True, confidence=0.9)
        args = types.SimpleNamespace(dashboard_size="1920x1080", site_origin=li.DEFAULT_SITE_ORIGINS,
                                     always_active=False, ws_host="localhost", ws_ports=[8799],
                                     fps=30)

        async def scenario():
            server = asyncio.create_task(li.websocket_task(args, state, screen))
            try:
                for _ in range(50):                       # wait for the port to open
                    try:
                        ws = await connect("ws://localhost:8799/eye-tracker",
                                           origin=li.DEFAULT_SITE_ORIGINS[0])
                        break
                    except OSError:
                        await asyncio.sleep(0.1)
                else:
                    self.fail("WebSocket server did not start")
                msgs = [json.loads(await asyncio.wait_for(ws.recv(), 2)) for _ in range(15)]
                active = state.read().site_clients
                await ws.close()                          # page reload / another page
                await asyncio.sleep(0.2)
                reloading = state.read().site_clients
                await asyncio.sleep(li.RECONNECT_GRACE_S)  # ...but it never came back
                return msgs, active, reloading, state.read().site_clients
            finally:
                state.stop.set()
                await server

        saved, li.RECONNECT_GRACE_S = li.RECONNECT_GRACE_S, 0.5
        try:
            msgs, active, reloading, after = asyncio.run(scenario())
        finally:
            li.RECONNECT_GRACE_S = saved
        self.assertEqual(active, 1, "website connection did not activate EyeSync")
        self.assertEqual(reloading, 1, "control dropped while the website reconnects")
        self.assertEqual(after, 0, "disconnect not noticed")
        for m in msgs:
            self.assertEqual(set(m), {"x", "y", "blink", "confidence", "face"})
            self.assertTrue(all(np.isfinite([m["x"], m["y"]])), "NaN sent to the website")
        self.assertEqual(msgs[0]["x"], 960.0)             # NaN replaced by the centre

    def test_quitting_tells_the_dashboard_to_press_connect(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 1, False)
        args = types.SimpleNamespace(dashboard_size="800x600", site_origin=li.DEFAULT_SITE_ORIGINS,
                                     always_active=False, ws_host="localhost", ws_ports=[8795],
                                     fps=30)

        async def scenario():
            server = asyncio.create_task(li.websocket_task(args, state, screen))
            for _ in range(50):
                try:
                    ws = await connect("ws://localhost:8795/", origin=li.DEFAULT_SITE_ORIGINS[0])
                    break
                except OSError:
                    await asyncio.sleep(0.1)
            await asyncio.wait_for(ws.recv(), 2)
            state.stop.set()                              # mouth wide open / Quit / F12
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), 3)
            except ConnectionClosed as exc:
                code, reason = exc.rcvd.code, exc.rcvd.reason
            await server
            return code, reason

        code, reason = asyncio.run(scenario())
        self.assertEqual(code, 4000)
        self.assertIn("press Connect", reason)

    def test_dashboard_buttons_control_eyesync(self):
        """The dashboard's buttons (commands over the same WebSocket) and their status replies."""
        from websockets.asyncio.client import connect

        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 1, False)
        actions = li.Actions(state)
        args = types.SimpleNamespace(dashboard_size="1920x1080", site_origin=li.DEFAULT_SITE_ORIGINS,
                                     always_active=False, ws_host="localhost", ws_ports=[8797],
                                     fps=30)

        async def command(ws, msg, want=None, until=lambda m: True):
            await ws.send(json.dumps(msg))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:        # skip the 30/s data messages
                m = json.loads(await asyncio.wait_for(ws.recv(), 2))
                # (a status pushed because something changed can arrive before the
                # reply: the 1.5.10 build's installed copy read one as calibrate's)
                if "type" in m and (want is None or m["type"] == want) and until(m):
                    return m
            self.fail(f"no reply to {msg}")

        async def scenario():
            server = asyncio.create_task(li.websocket_task(args, state, screen, None, actions))
            out = {}
            try:
                for _ in range(50):
                    try:
                        ws = await connect("ws://localhost:8797/eye-tracker",
                                           origin=li.DEFAULT_SITE_ORIGINS[0])
                        break
                    except OSError:
                        await asyncio.sleep(0.1)
                else:
                    self.fail("WebSocket server did not start")
                plain = [json.loads(await asyncio.wait_for(ws.recv(), 2)) for _ in range(5)]
                out["no_status_unasked"] = all("type" not in m for m in plain)
                out["active_at_start"] = li.control_active(state.read(), args)
                r = await command(ws, {"action": "mouse", "value": False})
                out["disabled"] = (r["mouse_enabled"], r["active"], li.control_active(state.read(), args))
                r = await command(ws, {"action": "mouse", "value": True})
                out["enabled"] = (r["mouse_enabled"], r["active"])
                await command(ws, {"action": "recenter"})
                out["recenter"] = state.recenter_request.is_set()
                r = await command(ws, {"action": "calibrate"}, until=lambda m: m.get("calibrating"))
                out["calibrate"] = (r["calibrating"], state.calibrate_request.is_set())
                r = await command(ws, {"action": "calibrate", "value": "cancel"},
                                  until=lambda m: not m.get("calibrating"))
                out["calibrate_cancel"] = (r["calibrating"], state.calibrate_cancel.is_set())
                out["speed"] = (await command(ws, {"action": "speed", "value": 3}))["speed"]
                out["click"] = (await command(ws, {"action": "click_mode", "value": "dwell"}))["click_mode"]
                out["gestures"] = (await command(ws, {"action": "gestures", "value": False}))["gestures"]
                out["bad"] = (await command(ws, {"action": "speed", "value": 9}, "error"))["type"]
                out["junk"] = (await command(ws, "hello", "error"))["type"]
                # the website's calibration progress needs no reply (not even an error)
                await ws.send(json.dumps({"type": "calibration", "event": "start", "points": 9}))
                out["after_calibration"] = (await command(ws, {"action": "status"}))["type"]
                await command(ws, {"action": "mouse", "value": False})
                await ws.close()
                await asyncio.sleep(0.3)
                out["reset_after_leaving"] = state.read().remote_off
                return out
            finally:
                state.stop.set()
                await server

        saved, li.save_settings = li.save_settings, lambda **kw: None   # leave real settings alone
        try:
            out = asyncio.run(scenario())
        finally:
            li.save_settings = saved
        self.assertTrue(out["no_status_unasked"], "an older dashboard would get status messages")
        self.assertTrue(out["active_at_start"])
        self.assertEqual(out["disabled"], (False, False, False))
        self.assertEqual(out["enabled"], (True, True))
        self.assertTrue(out["recenter"])
        self.assertEqual(out["calibrate"], (True, True))
        self.assertEqual(out["calibrate_cancel"], (False, True))
        self.assertEqual((out["speed"], out["click"], out["gestures"]), (3, "dwell", False))
        self.assertEqual((out["bad"], out["junk"]), ("error", "error"))
        self.assertEqual(out["after_calibration"], "status")
        self.assertFalse(out["reset_after_leaving"], "Disable must not outlive the dashboard")

    def test_website_controls_everything(self):
        """WEBSITE.md: probe, live status, pause, the camera view, quit."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 1, False)
        actions = li.Actions(state)
        actions.show_window = lambda: None                 # never touch the real window
        args = types.SimpleNamespace(dashboard_size="800x600", site_origin=li.DEFAULT_SITE_ORIGINS,
                                     always_active=False, ws_host="localhost", ws_ports=[8794],
                                     fps=30)
        site = li.DEFAULT_SITE_ORIGINS[0]

        async def until(ws, kind, ok=lambda m: True):
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 3))
                if m.get("type") == kind and ok(m):
                    return m

        async def scenario():
            server = asyncio.create_task(li.websocket_task(args, state, screen, None, actions))
            out = {}
            try:
                for _ in range(50):
                    try:
                        async with connect("ws://localhost:8794/?probe=1", origin=site) as ws:
                            first = json.loads(await asyncio.wait_for(ws.recv(), 2))
                            try:
                                await asyncio.wait_for(ws.recv(), 2)
                            except ConnectionClosed:
                                pass
                            out["probe"] = (first["type"], first["state"], first["version"],
                                            ws.close_code, state.read().site_clients)
                        break
                    except OSError:
                        await asyncio.sleep(0.1)
                ws = await connect("ws://localhost:8794/", origin=site)
                await ws.send(json.dumps({"action": "pause", "value": True}))
                out["paused"] = (await until(ws, "status"))["paused"]
                state.update(sensitivity=4)                # e.g. a hotkey: pushed to the page
                out["pushed"] = (await until(ws, "status", lambda m: m["speed"] == 4))["speed"]
                await ws.send(json.dumps({"action": "preview", "value": True}))
                state.set_preview(np.zeros((240, 320, 3), np.uint8))
                image = (await until(ws, "preview"))["image"]
                out["preview"] = (image.startswith("data:image/jpeg;base64,"),
                                  state.read().remote_preview)
                await ws.send(json.dumps({"action": "preview", "value": False}))
                await until(ws, "status")
                out["preview_off"] = state.read().remote_preview
                await ws.send(json.dumps({"action": "quit"}))
                try:
                    while True:
                        await asyncio.wait_for(ws.recv(), 3)
                except ConnectionClosed:
                    pass
                out["quit"] = (state.stop.is_set(), ws.close_code)
                return out
            finally:
                state.stop.set()
                await server

        saved, li.save_settings = li.save_settings, lambda **kw: None
        try:
            out = asyncio.run(scenario())
        finally:
            li.save_settings = saved
        self.assertEqual(out["probe"], ("eyesync", "running", li.VERSION, 4003, 0))
        self.assertTrue(out["paused"])
        self.assertEqual(out["pushed"], 4)
        self.assertEqual(out["preview"], (True, True))
        self.assertFalse(out["preview_off"])
        self.assertEqual(out["quit"], (True, 4000))

    def test_api_key_gate(self):
        """Only the website or the API key get in; a new key drops the old key's apps."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 2, False)
        args = types.SimpleNamespace(dashboard_size="1920x1080", site_origin=li.DEFAULT_SITE_ORIGINS,
                                     always_active=False, ws_host="localhost", ws_ports=[8798],
                                     fps=30)
        url = "ws://localhost:8798/eye-tracker"

        async def closed_with(ws):
            """Read until the server closes us -> (code, reason); None if it never does."""
            try:
                async with asyncio.timeout(2):
                    while True:
                        await ws.recv()
            except ConnectionClosed as exc:
                return exc.rcvd.code, exc.rcvd.reason
            except TimeoutError:
                return None

        async def scenario(api_key):
            server = asyncio.create_task(li.websocket_task(args, state, screen, api_key))
            out = {}
            try:
                for _ in range(50):                       # wait for the port to open
                    try:
                        app = await connect(f"{url}?key={api_key.get()}")
                        break
                    except OSError:
                        await asyncio.sleep(0.1)
                else:
                    self.fail("WebSocket server did not start")
                json.loads(await asyncio.wait_for(app.recv(), 2))
                out["app_active"] = state.read().site_clients
                site = await connect(url, origin=li.DEFAULT_SITE_ORIGINS[0])
                await asyncio.wait_for(site.recv(), 2)
                out["no_key"] = await closed_with(await connect(url))
                out["wrong_key"] = await closed_with(await connect(f"{url}?key=esk_not_the_key"))
                out["other_site"] = await closed_with(
                    await connect(url, origin="https://evil.example.com"))
                old = api_key.get()
                api_key.regenerate()
                out["app_after_new_key"] = await closed_with(app)
                out["old_key"] = await closed_with(await connect(f"{url}?key={old}"))
                await asyncio.wait_for(site.recv(), 2)    # the website is not affected
                fresh = await connect(f"{url}?key={api_key.get()}")
                await asyncio.wait_for(fresh.recv(), 2)
                out["active_at_end"] = state.read().site_clients
                await fresh.close()
                await site.close()
                return out
            finally:
                state.stop.set()
                await server

        with tempfile.TemporaryDirectory() as tmp:
            out = asyncio.run(scenario(li.ApiKey(Path(tmp) / "key.txt")))
        self.assertEqual(out["app_active"], 1, "a correct API key did not switch EyeSync on")
        self.assertEqual(out["no_key"], (4001, "API key required"))
        self.assertEqual(out["wrong_key"], (4001, "invalid API key"))
        self.assertEqual(out["other_site"], (4001, "API key required"))
        self.assertEqual(out["app_after_new_key"], (4001, "API key changed"))
        self.assertEqual(out["old_key"], (4001, "invalid API key"))
        self.assertEqual(out["active_at_end"], 2)     # the website + the app with the new key


class LowLightMesh:
    SWITCH = fe.LowLight.SWITCH_FRAMES


class LowLightTests(unittest.TestCase):
    """fe.LowLight with a stand-in face mesh that only 'sees' a face in a clean picture."""

    class Mesh:
        def __init__(self):
            self.calls = 0

        def process(self, img):
            self.calls += 1
            # noisy dark frames have big pixel-to-pixel jumps; the enhanced one is smooth
            ok = np.abs(np.diff(img[:, :, 0].astype(int), axis=1)).mean() < 8 and img.mean() > 20
            return np.zeros((478, 2)) if ok else None

    def frames(self, mean, noise, n=20):
        rng = np.random.default_rng(0)
        return [np.clip(rng.normal(mean, noise, (120, 160, 3)), 0, 255).astype(np.uint8)
                for _ in range(n)]

    def test_brief_face_loss_does_not_flip_modes(self):
        ll, mesh = fe.LowLight(), self.Mesh()
        for fr in self.frames(30, 25):                  # dark room: switches to enhanced
            ll.process(mesh, fr)
        self.assertTrue(ll.active)
        real = mesh.process
        for i, fr in enumerate(self.frames(30, 25, 40)):
            # every 4th frame the face is lost in the enhanced picture (a hand, a wide mouth)
            mesh.process = (lambda img: None) if i % 4 == 0 else real
            ll.process(mesh, fr)
            mesh.process = real
            self.assertTrue(ll.active, f"flipped to plain at frame {i}")

    def test_normal_light_never_enhances(self):
        ll, mesh = fe.LowLight(), self.Mesh()
        for fr in self.frames(130, 2):
            self.assertIsNotNone(ll.process(mesh, fr))
        self.assertFalse(ll.active)
        self.assertEqual(mesh.calls, 20)                # one face-mesh run per frame

    def test_dark_noisy_room_switches_to_enhanced_and_back(self):
        ll, mesh = fe.LowLight(), self.Mesh()
        found = [ll.process(mesh, fr) is not None for fr in self.frames(30, 25)]
        self.assertTrue(all(found), "face lost in the dark")
        self.assertTrue(ll.active)
        self.assertLess(mesh.calls, 20 + LowLightMesh.SWITCH + 1)   # then enhanced only: no double work
        for fr in self.frames(140, 2, 3):               # lights on
            self.assertIsNotNone(ll.process(mesh, fr))
        self.assertFalse(ll.active)


class StandbyTests(unittest.TestCase):
    """After a quit, EyeSync.pyw waits in site_access.wait_for_visit (timings scaled down)."""

    PORT = 8796

    def run_standby(self, visits, wake_at=None, key_path=None, after_quit=True):
        """visits: list of (seconds after start, origin, query). Returns (result, seconds)."""
        import socket
        import site_access

        t0 = time.monotonic()
        clock = time.monotonic
        woken = threading.Event()
        done = threading.Event()        # the standby returned: stop visiting (next test!)

        def visitor():
            for at, origin, query in visits:
                while clock() - t0 < at and not done.is_set():
                    time.sleep(0.01)
                if done.is_set():
                    return
                try:
                    with socket.create_connection(("127.0.0.1", self.PORT), timeout=1) as c:
                        c.sendall(f"GET /eye-tracker{query} HTTP/1.1\r\nHost: localhost\r\n"
                                  f"Origin: {origin}\r\nUpgrade: websocket\r\n\r\n".encode())
                        c.recv(100)
                except OSError:
                    pass
            if wake_at is not None:
                while clock() - t0 < wake_at and not done.is_set():
                    time.sleep(0.01)
                woken.set()

        thread = threading.Thread(target=visitor, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            kp = Path(tmp) / "key.txt"
            if key_path:
                kp.write_text(key_path, encoding="utf-8")
            got = site_access.wait_for_visit(ports=(self.PORT,), key_path=kp, quiet_s=0.6,
                                             confirm_s=0.5, wake=woken.is_set,
                                             after_quit=after_quit)
        secs = clock() - t0
        done.set()
        thread.join(2)
        return got, secs

    SITE = li.DEFAULT_SITE_ORIGINS[0]

    def test_website_opened_starts_eyesync(self):
        got, secs = self.run_standby([(1.0, self.SITE, ""), (1.3, self.SITE, "")], wake_at=3)
        self.assertEqual(got, "website")
        self.assertLess(secs, 2)

    def test_dashboard_already_open_at_login_starts_eyesync(self):
        # At login (not after a quit) a dashboard the browser reopened counts at once.
        tries = [(0.1 + 0.3 * i, self.SITE, "") for i in range(8)]
        self.assertEqual(self.run_standby(tries, wake_at=2.8, after_quit=False)[0], "website")

    # The website's automatic retries after a lost connection: gaps of ~1, 2, 4 s
    # (+-20%, so at least 0.8, 1.6, 3.2 s). Its Connect button: a try, and ~0.5 s later another.
    RETRIES = [(0.1, "", ""), (0.9, "", ""), (2.5, "", "")]

    def tries(self, schedule):
        return [(at, self.SITE, query) for at, _, query in schedule]

    def test_website_retrying_after_quit_does_not_restart(self):
        # Quit with the mouth (or Quit / F12) while the website is connected: it keeps
        # retrying by itself - EyeSync must stay closed.
        self.assertEqual(self.run_standby(self.tries(self.RETRIES), wake_at=3.0)[0], "shortcut")

    def test_connect_click_after_quit_restarts(self):
        click = [(3.0, "", ""), (3.5, "", "")]
        got, secs = self.run_standby(self.tries(self.RETRIES + click), wake_at=5.0)
        self.assertEqual(got, "website")
        self.assertLess(secs, 4.0)

    def test_background_tab_does_not(self):
        # Browsers slow a hidden tab's retries to one a minute: single tries never start it.
        self.assertEqual(self.run_standby([(1.0, self.SITE, ""), (2.0, self.SITE, "")],
                                          wake_at=2.8)[0], "shortcut")

    def test_only_the_website_or_the_key(self):
        evil = [(1.0, "https://evil.example.com", ""), (1.3, "https://evil.example.com", "")]
        self.assertEqual(self.run_standby(evil, wake_at=2)[0], "shortcut")
        key = "esk_" + "k" * 32
        wrong = [(1.0, "", "?key=esk_wrong"), (1.3, "", "?key=esk_wrong")]
        self.assertEqual(self.run_standby(wrong, wake_at=2, key_path=key)[0], "shortcut")
        right = [(1.0, "", f"?key={key}"), (1.3, "", f"?key={key}")]
        self.assertEqual(self.run_standby(right, wake_at=3, key_path=key)[0], "website")

    def ws_visit(self, query, origin=None, delay=0.3):
        """A real browser-like WebSocket visit: (first message, close code)."""
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect
        time.sleep(delay)
        with connect(f"ws://127.0.0.1:{self.PORT}/{query}", origin=origin or self.SITE,
                     open_timeout=2) as ws:
            first = json.loads(ws.recv(timeout=2))
            try:
                ws.recv(timeout=2)
            except ConnectionClosed:
                pass
            return first, ws.close_code

    def standby_with(self, query, after_quit=True, wake_after=1.5):
        import site_access
        out, woken = {}, threading.Event()

        def visit():
            try:
                out["visit"] = self.ws_visit(query)
            finally:
                time.sleep(wake_after)
                woken.set()
        th = threading.Thread(target=visit, daemon=True)
        th.start()
        with tempfile.TemporaryDirectory() as tmp:
            out["got"] = site_access.wait_for_visit(ports=(self.PORT,), key_path=Path(tmp) / "k",
                                                    wake=woken.is_set, after_quit=after_quit)
        th.join(5)
        return out

    def test_website_probe_learns_eyesync_is_installed(self):
        out = self.standby_with("?probe=1")
        first, code = out["visit"]
        self.assertEqual((first["type"], first["state"], first["version"], first["closed_by_user"]),
                         ("eyesync", "standby", li.VERSION, True))
        self.assertEqual(code, 4003)
        self.assertEqual(out["got"], "shortcut")              # a probe never starts EyeSync

    def test_connect_with_start_starts_eyesync_at_once(self):
        out = self.standby_with("?start=1")                   # even after a quit
        self.assertEqual(out["visit"][0]["state"], "starting")
        self.assertEqual(out["visit"][1], 4002)
        self.assertEqual(out["got"], "website")

    def test_other_sites_cannot_probe(self):
        import site_access
        from websockets.exceptions import InvalidStatus
        out, woken = {}, threading.Event()

        def visit():
            time.sleep(0.3)
            try:
                self.ws_visit("?probe=1", origin="https://evil.example.com", delay=0)
            except InvalidStatus as exc:
                out["status"] = exc.response.status_code
            woken.set()
        threading.Thread(target=visit, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            got = site_access.wait_for_visit(ports=(self.PORT,), key_path=Path(tmp) / "k",
                                             wake=woken.is_set)
        self.assertEqual((got, out.get("status")), ("shortcut", 503))

    def test_launcher_knows_why_eyesync_closed(self):
        """Quit -> only Connect restarts; dashboard closed -> it may; a crash -> restart."""
        import subprocess

        import launcher
        saved = launcher.time, launcher.subprocess
        launcher.time = types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda s: None)
        try:
            for codes, after_quit in (([li.EXIT_QUIT], True), ([li.EXIT_DASHBOARD_CLOSED], False),
                                      ([li.EXIT_CRASHED, 0xC0000005, li.EXIT_QUIT], True)):
                calls, seen = iter(codes), []

                def call(cmd, **kw):
                    seen.append(cmd)
                    return next(calls)
                launcher.subprocess = types.SimpleNamespace(
                    call=call, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
                self.assertEqual(launcher.run_until_quit(["--minimized"]), after_quit, codes)
                self.assertIsNone(next(calls, None), "every crash is restarted")
                self.assertEqual(seen[0][-1], "--minimized")
        finally:
            launcher.time, launcher.subprocess = saved

    def test_opened_from_the_website_link(self):
        import launcher
        self.assertEqual(launcher.parse_args(["--standby"]), (True, [], False))
        self.assertEqual(launcher.parse_args([]), (False, [], False))
        # eyesync://start from the site: start now, minimised; the link text is dropped
        self.assertEqual(launcher.parse_args(["--url", "eyesync://start?x=--mouse none"]),
                         (False, ["--minimized"], True))
        self.assertEqual(launcher.parse_args(["--standby", "--url", "eyesync:"]),
                         (False, ["--minimized"], True))

    def test_engine_command_from_source_and_as_exe(self):
        import app_paths
        saved = app_paths.FROZEN
        app_paths.FROZEN = False
        try:
            cmd = app_paths.engine_command(["--minimized"])
        finally:
            app_paths.FROZEN = saved
        self.assertTrue(cmd[1].endswith("live_inference.py") and cmd[-1] == "--minimized", cmd)
        app_paths.FROZEN = True
        try:                                     # EyeSync.exe runs itself as the engine
            self.assertEqual(app_paths.engine_command(["-x"]), [sys.executable, "--engine", "-x"])
        finally:
            app_paths.FROZEN = saved

    def test_port_is_free_for_eyesync_afterwards(self):
        import site_access
        self.run_standby([], wake_at=0.3)
        socks = site_access._listen(self.PORT)
        self.assertTrue(socks, "standby did not release the port")
        for s in socks:
            s.close()


class AutostartTests(unittest.TestCase):
    """The login start, against a fake registry (the real one is never touched)."""

    def test_switched_off_in_windows_shows_off_and_enable_switches_it_on(self):
        import autostart
        reg = {(autostart.RUN_KEY, "EyeSync"): '"pythonw.exe" "EyeSync.pyw" --standby',
               (autostart.APPROVED_KEY, "EyeSync"): bytes([3] + [0] * 11)}   # Task Manager: off

        class Key:
            def __init__(self, path): self.path = path
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        def query(k, name):
            if (k.path, name) not in reg:
                raise FileNotFoundError(name)
            return reg[(k.path, name)], 0

        fake = types.SimpleNamespace(
            HKEY_CURRENT_USER=0, KEY_SET_VALUE=2, KEY_QUERY_VALUE=1, REG_SZ=1, REG_BINARY=3,
            OpenKey=lambda root, path, *a: Key(path), CreateKeyEx=lambda root, path, *a: Key(path),
            QueryValueEx=query,
            SetValueEx=lambda k, name, _r, _t, value: reg.__setitem__((k.path, name), value))
        saved = sys.modules.get("winreg")
        sys.modules["winreg"] = fake
        try:
            self.assertTrue(autostart.turned_off_in_windows())
            self.assertFalse(autostart.enabled())             # the checkbox shows it off
            autostart.enable()                                # the user ticks it
            self.assertFalse(autostart.turned_off_in_windows())
            self.assertTrue(autostart.enabled())
            del reg[(autostart.APPROVED_KEY, "EyeSync")]      # never touched in Windows
            self.assertTrue(autostart.enabled())
            # eyesync: links from the website open EyeSync for it
            autostart.make_icon = lambda: None
            autostart.register_url()
            self.assertEqual(reg[(autostart.URL_KEY, "URL Protocol")], "")
            cmd = reg[(autostart.URL_KEY + r"\shell\open\command", "")]
            self.assertTrue(cmd.endswith('--url "%1"'), cmd)       # EyeSync.pyw or EyeSync.exe
            self.assertIn("EyeSync.exe" if getattr(sys, "frozen", False) else "EyeSync.pyw", cmd)
        finally:
            import importlib
            importlib.reload(autostart)                       # undo the make_icon stub
            if saved is not None:
                sys.modules["winreg"] = saved


class InstallerTests(unittest.TestCase):
    """EyeSync.exe installing itself (only copying - nothing is registered)."""

    def test_installs_as_eyesync_exe_and_keeps_the_users_files(self):
        import installer
        with tempfile.TemporaryDirectory() as tmp:
            # A second download is saved as "EyeSync (1).exe" by the browser.
            download = Path(tmp) / "Downloads" / "EyeSync (1).exe"
            download.parent.mkdir()
            download.write_bytes(b"new version" * 1000)
            dest = Path(tmp) / "Programs" / "EyeSync"
            dest.mkdir(parents=True)
            (dest / "EyeSync.exe").write_bytes(b"old version")
            (dest / "eyesync_settings.json").write_text('{"calibration": {}}')
            (dest / "eyesync_api_key.txt").write_text("esk_" + "k" * 32)
            (dest / "eyesync.log").write_text("log")
            (dest / ".venv").mkdir()                       # an older zip install
            (dest / "live_inference.py").write_text("# old")
            (dest / "notes.txt").write_text("not ours")
            saved = installer.sys
            installer.sys = types.SimpleNamespace(executable=str(download))
            try:
                target = installer.install(dest, register=False)
            finally:
                installer.sys = saved
            self.assertEqual(target.name, "EyeSync.exe")
            self.assertEqual(target.read_bytes(), download.read_bytes())
            self.assertEqual(sorted(p.name for p in dest.iterdir()),
                             ["EyeSync.exe", "eyesync.log", "eyesync_api_key.txt",
                              "eyesync_settings.json", "notes.txt"])

    def test_refuses_the_source_folder(self):
        import installer
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "build_exe.py").write_text("# source")
            with self.assertRaises(RuntimeError):
                installer.remove_old(Path(tmp))


# --------------------------------------------------------------------------- #
# Voice typing - never the real keyboard or microphone: a recording plays the
# microphone and a fake keyboard records what would have been typed.
# --------------------------------------------------------------------------- #

def drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


class FakeMic:
    """Plays a recording as if spoken into the microphone (faster than real time)."""

    def __init__(self, audio: np.ndarray, noise: float = 0.002):
        rng = np.random.default_rng(1)
        self.audio = (audio + rng.standard_normal(audio.size) * noise).astype(np.float32)
        self.rng, self.noise, self.pos = rng, noise, 0
        self.name, self.opened, self.closed = "test recording", False, False

    def open(self):
        self.opened = True

    def read(self, timeout):
        time.sleep(0.002)
        block = self.audio[self.pos:self.pos + 800]
        self.pos += 800
        if block.size < 800:                       # after the recording: a quiet room
            block = (self.rng.standard_normal(800) * self.noise).astype(np.float32)
        return block

    def close(self):
        self.closed = True


class FakeTypist:
    def __init__(self):
        self.ops = []

    def run(self, ops):
        self.ops.extend(ops)
        return len(vt.key_events(ops))


class FakeTracker:
    """The window in front: always the same text box."""

    def poll(self):
        pass

    def peek(self):
        return 1234

    target = peek

    def elevated(self, hwnd):
        return False


def speechlike(secs: float, rng, level: float = 0.05) -> np.ndarray:
    """Noise in the speech band, pulsing like syllables (~4 a second)."""
    n = int(secs * vt.RATE)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / vt.RATE)
    spec[(f < 300) | (f > 3000)] = 0
    x = np.fft.irfft(spec, n)
    x *= 0.55 + 0.45 * np.sin(2 * np.pi * 4 * np.arange(n) / vt.RATE)
    return (x / np.std(x) * level).astype(np.float32)


def room(secs: float, rng, level: float = 0.001) -> np.ndarray:
    return (rng.standard_normal(int(secs * vt.RATE)) * level).astype(np.float32)


def loudness_vad(chunk) -> float:
    """A stand-in voice detector for the splitter's own logic: loud = speech."""
    return 0.9 if float(np.sqrt(np.mean(chunk * chunk))) > 0.01 else 0.05


def segment(audio: np.ndarray, seg=None) -> list[np.ndarray]:
    seg = seg or vt.SpeechSegmenter(loudness_vad)
    phrases = []
    for i in range(0, audio.size, 800):
        phrases += seg.feed(audio[i:i + 800])
    return phrases


def recording() -> np.ndarray:
    """tests/data/voice_phrases.wav: four phrases with 1.2 s pauses."""
    import wave
    with wave.open(str(PROJECT / "tests" / "data" / "voice_phrases.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def silero_phrases(audio: np.ndarray) -> list[float]:
    """Phrase lengths (s) the real voice detection finds."""
    import _pywhispercpp
    _pywhispercpp.whisper_log_set(vt._whisper_log)     # as during voice typing: no console spam
    vad = vt.SileroVad()
    try:
        return [round(p.size / vt.RATE, 2) for p in segment(audio, vt.SpeechSegmenter(vad))]
    finally:
        vad.close()
        _pywhispercpp.whisper_log_set(None)            # (left set, it crashes Python's exit)


class VoiceTypingTests(unittest.TestCase):
    def test_phrases_join_like_writing(self):
        c = vt.Composer()
        self.assertEqual(c.plan(" Hello"), [("text", "Hello")])
        # A sentence carried on after a pause: no capital, a space between.
        self.assertEqual(c.plan(" How are you doing today?"), [("text", " how are you doing today?")])
        self.assertEqual(c.plan("I'm fine."), [("text", " I'm fine.")])
        # Whisper ended it with a full stop, but it went on: the full stop is taken back.
        self.assertEqual(c.plan("and you?"), [("back", 1), ("text", " and you?")])
        self.assertEqual(c.plan("Great."), [("text", " Great.")])
        c = vt.Composer()
        c.plan("I met")
        self.assertEqual(c.plan("John today."), [("text", " John today.")])     # names stay
        c = vt.Composer()
        c.plan("I would like to...")
        self.assertEqual(c.plan("schedule a meeting."), [("back", 3), ("text", " schedule a meeting.")])
        self.assertEqual(c.plan("Okay, thanks"), [("text", " Okay, thanks")])
        self.assertEqual(c.plan(", bye"), [("text", ", bye")])
        # Lower case after a full stop because Whisper swallowed the first word
        # ("The"): a new sentence, not a reason to take the full stop back.
        c = vt.Composer()
        c.plan("Please send the report.")
        self.assertEqual(c.plan("quick brown fox."), [("text", " Quick brown fox.")])

    def test_voice_commands(self):
        c = vt.Composer()
        self.assertEqual(c.plan("Delete that."), [])                  # nothing typed yet
        c.plan("Dear John,")
        self.assertEqual(c.plan("New line."), [("key", "newline")])
        self.assertEqual(c.plan("Thanks for the update."), [("text", "Thanks for the update.")])
        self.assertEqual(c.plan("Question mark."), [("back", 1), ("text", "?")])  # replaces the "."
        self.assertEqual(c.plan("Delete that."), [("back", 1)])
        self.assertEqual(c.plan("Scratch that"), [("back", len("Thanks for the update"))])
        self.assertEqual(c.plan("Thanks."), [("text", "Thanks.")])
        self.assertEqual(c.plan('"Kama"'), [("back", 1), ("text", ",")])    # a misheard "comma"
        self.assertEqual(c.plan("New paragraph"), [("key", "newline"), ("key", "newline")])
        self.assertEqual(c.plan("Press enter."), [("key", "enter")])
        self.assertEqual(c.plan("Tab."), [("key", "tab")])
        self.assertEqual(c.plan("Backspace"), [("key", "backspace")])
        self.assertEqual(c.plan("space"), [("text", " ")])
        self.assertEqual(c.plan("Stop listening."), [("stop", "")])
        # At the start or end of a longer phrase they still count...
        c = vt.Composer()
        self.assertEqual(c.plan("Best wishes, new line."), [("text", "Best wishes,"), ("key", "newline")])
        self.assertEqual(c.plan("New line. See you soon."), [("key", "newline"), ("text", "See you soon.")])
        self.assertEqual(c.plan("That's all, stop listening."), [("text", " That's all"), ("stop", "")])
        # ... but not in the middle of a sentence.
        self.assertEqual(vt.Composer().plan("We launched a new line of products."),
                         [("text", "We launched a new line of products.")])

    def test_sound_labels_and_video_credits_are_not_typed(self):
        for raw in ("[BLANK_AUDIO]", "(buzzing)", "[MUSIC PLAYING]", "*laughs*", "♪ ♪", " ...",
                    "Thanks for watching!", "Subtitles by the Amara.org community"):
            with self.subTest(raw):
                self.assertEqual(vt.Composer().plan(raw), [])
        self.assertEqual(vt.clean(" Hello (laughs) there."), "Hello there.")
        self.assertEqual(vt.clean("[BLANK_AUDIO] Hello [BLANK_AUDIO]"), "Hello")

    def test_key_presses(self):
        U, UP = vt.KEYEVENTF_UNICODE, vt.KEYEVENTF_KEYUP
        ev = vt.key_events([("text", "Hi"), ("key", "newline"), ("back", 2), ("text", "\U0001F600")])
        self.assertEqual(ev[:4], [(0, ord("H"), U), (0, ord("H"), U | UP),
                                  (0, ord("i"), U), (0, ord("i"), U | UP)])
        # A new line is Shift+Enter: in a chat box, Enter alone would send the message.
        self.assertEqual(ev[4:8], [(vt.VK_SHIFT, 0, 0), (vt.VK_RETURN, 0, 0),
                                   (vt.VK_RETURN, 0, UP), (vt.VK_SHIFT, 0, UP)])
        self.assertEqual(ev[8:12], [(vt.VK_BACK, 0, 0), (vt.VK_BACK, 0, UP)] * 2)
        self.assertEqual([e[1] for e in ev[12:]], [0xD83D, 0xD83D, 0xDE00, 0xDE00])  # 2 UTF-16 units
        self.assertEqual(vt.key_events([("key", "enter")]),
                         [(vt.VK_RETURN, 0, 0), (vt.VK_RETURN, 0, UP)])

    def test_phrases_are_cut_at_pauses(self):
        """The real voice detection on the test recording: its four phrases, at every noise level."""
        rng = np.random.default_rng(3)
        audio = recording()
        want = [3.1, 1.15, 2.75, 1.5]               # with ~0.3 s kept before and after each
        for name, noisy in (("clean", audio),
                            ("a fan", audio + room(audio.size / vt.RATE, rng, 0.02))):
            with self.subTest(name):
                got = silero_phrases(noisy)
                self.assertEqual(len(got), 4, got)
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, delta=0.25, msg=got)

    def test_a_microphone_that_mutes_the_room(self):
        """
        A real user's laptop microphone: near-silence between words, with
        noise bursts let through. A loudness splitter heard those as talking
        and made 15-20 s phrases; the voice detection still finds the four.
        """
        rng = np.random.default_rng(6)
        audio = recording() * 0.5
        audio = np.where(np.abs(audio) < 0.0015, 1e-5 * rng.standard_normal(audio.size), audio)
        for start in range(0, audio.size - 800, 3200):          # a burst every 0.2 s
            audio[start:start + 480] += rng.standard_normal(480) * 0.02
        got = silero_phrases(audio.astype(np.float32))
        self.assertEqual(len(got), 4, got)
        self.assertTrue(all(g < 3.6 for g in got), got)

    def test_noise_is_not_a_phrase(self):
        rng = np.random.default_rng(4)
        fan = room(6.0, rng, 0.02)                              # a loud fan...
        for start in range(8000, fan.size - 800, 6400):         # ... and keyboard clicks
            fan[start:start + 480] += (rng.standard_normal(480) * 0.3).astype(np.float32)
        hum = (0.05 * np.sin(2 * np.pi * 100 * np.arange(4 * vt.RATE) / vt.RATE)).astype(np.float32)
        for name, noise in (("fan and clicks", fan), ("hum", hum), ("silence", np.zeros(vt.RATE * 3, np.float32))):
            with self.subTest(name):
                self.assertEqual(silero_phrases(noise), [])

    def test_splitter_logic(self):
        """With a stand-in detector: pauses inside a phrase, the end, the countdown, a cough."""
        rng = np.random.default_rng(3)
        audio = np.concatenate([room(1.0, rng), speechlike(1.5, rng), room(0.3, rng),
                                speechlike(1.0, rng),             # a short pause: one phrase
                                room(1.2, rng), speechlike(0.8, rng), room(1.5, rng),
                                speechlike(0.15, rng), room(1.0, rng)])   # a cough: nothing
        seg = vt.SpeechSegmenter(loudness_vad)
        phrases = segment(audio, seg)
        self.assertEqual(len(phrases), 2)
        self.assertTrue(2.8 < phrases[0].size / vt.RATE < 3.6, phrases[0].size / vt.RATE)
        self.assertTrue(0.9 < phrases[1].size / vt.RATE < 1.6, phrases[1].size / vt.RATE)
        self.assertFalse(seg.speaking)
        self.assertAlmostEqual(seg.quiet_s, 1.0, delta=0.15)     # counts towards stopping

    def test_long_talk_without_a_pause_is_cut(self):
        rng = np.random.default_rng(5)
        phrases = segment(np.concatenate([room(0.5, rng), speechlike(26.0, rng), room(1.5, rng)]))
        self.assertEqual(len(phrases), 2)
        self.assertTrue(all(p.size / vt.RATE <= vt.MAX_PHRASE_S + 0.5 for p in phrases))

    def test_resample(self):
        for rate in (48000, 44100):
            x = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
            y = vt.resample(x, rate, vt.RATE)
            self.assertEqual(y.size, vt.RATE)
            peak = np.argmax(np.abs(np.fft.rfft(y))) * vt.RATE / y.size
            self.assertAlmostEqual(peak, 440, delta=2)

    def test_eyes_closed_2s_switches_voice_typing(self):
        O, C = 0.28, 0.07
        state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
        b = li.BlinkDetector(None, 0.55, 0.70, 0.2, 1.5, 0.6)
        b.baseline = O
        worker = types.SimpleNamespace(blink=b, state=state, voice_closure=None)
        t, clicks = 0.0, []

        def run(seq):
            nonlocal t
            for left, right in seq:
                t += 1 / FPS
                closed, click, _ = b.update(left, right, t, None, True)
                if click:
                    clicks.append(click)
                li.VisionWorker._voice_gesture(worker, closed, state.read(), t)
            return drain(state.voice_requests)

        pad = [(O, O)] * 40
        # 2.5 s since 1.5.2 (VOICE_EYES_CLOSED_S), after the longest click (BLINK_MAX_S).
        self.assertLess(li.BLINK_MAX_S, li.VOICE_EYES_CLOSED_S)
        b.max_s = li.BLINK_MAX_S
        self.assertEqual(run(pad + [(C, C)] * 81 + pad), [(None, "eyes closed")])   # 2.7 s
        self.assertEqual(clicks, [])                                 # ... and no click
        self.assertEqual(run(pad + [(C, C)] * 150 + pad), [(None, "eyes closed")])  # 5 s: once
        self.assertEqual(run(pad + [(C, C)] * 66 + pad), [])         # 2.2 s: not yet
        self.assertEqual(clicks, ["right"])                          # (a slow right click)
        clicks.clear()
        self.assertEqual(run(pad + [(C, C)] * 12 + pad), [])         # a blink clicks...
        self.assertEqual(clicks, ["left"])
        # The user's blink: the right eye closes later and less (1.5.2's log).
        lag = [(C, O * (1 - 0.5 * 0.45))] * 3 + [(C, O * (1 - 0.9 * 0.45))] * 78
        self.assertEqual(run(pad + lag + pad), [(None, "eyes closed")])
        self.assertEqual(run(pad + [(O, C)] * 75 + pad), [])         # one eye only: no
        state.update(gestures=False)
        self.assertEqual(run(pad + [(C, C)] * 75 + pad), [])         # gestures off: no

    def test_camera_view_shows_the_voice_typing_gesture(self):
        import main_window
        args = types.SimpleNamespace(ear_threshold=None, close_ratio=0.55, open_ratio=0.70,
                                     min_blink=0.2, max_blink=1.5, click_cooldown=0.6,
                                     dwell_time=1.0, close_gesture_hold=3.0)
        state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
        w = li.VisionWorker(args, fe.ScreenSize(1920, 1080), state, meshes=types.SimpleNamespace())
        frame = np.zeros((480, 640, 3), np.uint8)
        w.blink.closed_since = 100.0
        for voice, held in (("", 0.5), ("", 1.8), ("", 2.2), ("listening", 1.8), ("working", 0.0),
                            ("error", 0.0)):
            with self.subTest(voice=voice, held=held):
                state.update(eyes_closed=held > 0, voice=voice)
                view = w._render_preview(frame, state.read(), 100.0 + held)
                self.assertEqual(view.shape[:2], (main_window.PREVIEW_H, main_window.PREVIEW_W))

    def test_talking_while_voice_typing_does_not_fire_mouth_gestures(self):
        rest = FaceGestureTests.REST
        g, t, fired = li.FaceGestures(), 0.0, []
        for secs, over in ((3, {}), (1.7, {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}), (1, {}),
                           (1.7, FaceGestureTests.TO_LEFT), (1, {}), (3.3, FaceGestureTests.TO_RIGHT),
                           (1, {})):
            for _ in range(int(secs * FPS)):
                t += 1 / FPS
                if f := g.update({**rest, **over}, t, None, li.VOICE_IGNORED_GESTURES):
                    fired.append(f)
        self.assertEqual(fired, [])                                  # (head tilts still scroll)

    def test_website_hotkey_and_status(self):
        state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
        actions = li.Actions(state)
        args = types.SimpleNamespace(always_active=False)
        r = li.handle_command(json.dumps({"action": "voice"}), state, actions, args)
        self.assertEqual(r["voice"], {"on": False, "state": "off", "hint": ""})
        li.handle_command(json.dumps({"action": "voice", "value": True}), state, actions, args)
        bad = li.handle_command(json.dumps({"action": "voice", "value": "yes"}), state, actions, args)
        self.assertEqual(bad["type"], "error")
        li.make_hotkey_actions(actions)["F5"]()
        self.assertEqual(drain(state.voice_requests),
                         [(None, "website"), (True, "website"), (None, "hotkey")])
        state.update(voice="hearing", voice_text="my secret words")
        msg = li.status_message(state, args)
        self.assertEqual(msg["voice"]["on"], True)
        self.assertNotIn("secret", json.dumps(msg))            # what is said never leaves the PC

    def test_speech_is_typed_end_to_end(self):
        """The test recording through voice typing: two sentences, "new line", "stop listening"."""
        import wave
        self.assertEqual(vt.available(), "", "voice typing is not completely installed")
        with wave.open(str(PROJECT / "tests" / "data" / "voice_phrases.wav")) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
        mic, typist = FakeMic(audio), FakeTypist()
        voice = vt.VoiceTyping(state, mic_factory=lambda: mic, typist=typist, tracker=FakeTracker(),
                               sounds=False, check=lambda: "")
        messages = []
        handler = logging.Handler()
        handler.emit = lambda record: messages.append(record.getMessage())
        _log.addHandler(handler)
        level = _log.level
        _log.setLevel(logging.INFO)
        try:
            voice.start()
            vt.request(state, True, "test")
            deadline = time.monotonic() + 90
            while not mic.closed and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            state.stop.set()
            voice.close()
            _log.removeHandler(handler)
            _log.setLevel(level)
        self.assertTrue(mic.closed, "voice typing didn't stop")
        self.assertEqual([op for op, _ in typist.ops], ["text", "key", "text"], typist.ops)
        self.assertEqual(typist.ops[1], ("key", "newline"))
        first, second = typist.ops[0][1], typist.ops[2][1]
        self.assertFalse(second.startswith(" "), "a space after a new line")
        want = "please send me the report by friday afternoon thank you so much for your help yesterday"
        got = vt.normalise(first + " " + second).split()
        wrong = sum(a != b for a, b in zip(got, want.split())) + abs(len(got) - len(want.split()))
        self.assertLessEqual(wrong, 2, f"recognised {first!r} {second!r}")
        self.assertEqual(state.read().voice, "")
        self.assertTrue(any('stop listening' in m for m in messages), messages)
        logged = " ".join(messages).lower()
        self.assertNotIn("friday", logged)                     # the words never go in the log
        self.assertNotIn("yesterday", logged)

    def test_window_button_and_bar(self):
        import tkinter as tk

        import main_window
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        try:
            root.withdraw()
            win = main_window.MainWindow(root, False, PROJECT / "README.md", PROJECT)
            st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
            args = types.SimpleNamespace(always_active=False)
            win.attach(st, li.Actions(st), lambda: li.status_of(st.read(), args))
            win.voice_btn.invoke()
            self.assertEqual(drain(st.voice_requests), [(None, "button")])

            bar = win.voice_bar

            def hidden_show():                     # not over the real screen
                bar.top = tk.Toplevel(root)
                bar.top.withdraw()
                bar.zoom = 1.0
                bar.canvas = tk.Canvas(bar.top, width=600, height=64)
                bar.stop_btn = tk.Button(bar.top, command=bar.on_stop)
                bar.drawn = None
            bar._show = hidden_show
            st.update(voice="hearing", voice_level=0.5, voice_text="Hello there.")
            win.refresh()
            self.assertIsNotNone(bar.top)
            self.assertEqual(bar.drawn[:3], ("hearing", "Hello there.", ""))
            self.assertEqual(win.voice_btn.cget("text"), "Stop voice typing")
            bar.stop_btn.invoke()
            self.assertEqual(drain(st.voice_requests), [(False, "the bar's Stop button")])
            st.update(voice="")
            win.refresh()
            self.assertIsNone(bar.top)
            self.assertEqual(win.voice_btn.cget("text"), "Voice typing")

            # The real bar never takes the keyboard focus (shown off the screen here).
            top = tk.Toplevel(root)
            top.withdraw()
            top.overrideredirect(True)
            top.geometry("200x40+-4000+-4000")
            main_window.show_without_focus(top)
            if sys.platform == "win32":
                import ctypes
                hwnd = int(top.wm_frame(), 16)
                self.assertTrue(ctypes.windll.user32.GetWindowLongW(hwnd, -20) & 0x08000000)
                self.assertTrue(vt.ForegroundTracker().ours(hwnd))      # EyeSync's own window
            top.destroy()
        finally:
            root.destroy()


class ApiKeyTests(unittest.TestCase):
    def test_made_once_kept_and_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.txt"
            a = li.ApiKey(path)
            key = a.get()
            self.assertTrue(key.startswith("esk_") and len(key) >= 36)
            self.assertEqual(li.ApiKey(path).get(), key)            # same key after a restart
            self.assertNotEqual(li.ApiKey(Path(tmp) / "b.txt").get(), key)   # random per install
            self.assertTrue(a.matches(key))
            for wrong in (key[:-1], key + "x", "", "é" * 36):
                self.assertFalse(a.matches(wrong), wrong)
            a.regenerate()
            self.assertNotEqual(a.get(), key)
            self.assertEqual(a.version, 1)
            self.assertFalse(a.matches(key))
            self.assertEqual(li.ApiKey(path).get(), a.get())        # the new key is saved

    def test_damaged_file_gets_a_new_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.txt"
            path.write_text("short", encoding="utf-8")
            key = li.ApiKey(path).get()
            self.assertTrue(key.startswith("esk_"))
            self.assertEqual(path.read_text(encoding="utf-8").strip(), key)


# --------------------------------------------------------------------------- #
# Eye gaze and Gaze jump (gaze_engine)
# --------------------------------------------------------------------------- #

import gaze_engine as ge          # noqa: E402


def _rot(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    a, b, c = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


class SyntheticFace:
    """
    A face for the gaze tests, in millimetres: a head with two eyeballs
    whose irises turn to look at a point on the screen, seen by a pinhole
    webcam at the screen's top edge (camera coordinates: x = picture right,
    y = down, z = away from the camera). landmarks() gives what MediaPipe
    would: (478, 3) normalised x, y and depth. Open eyes (EAR 0.33), a
    closed mouth; positive yaw turns the head towards the screen's right,
    positive roll tilts it towards the user's left shoulder.
    """

    W, H, F = 640, 480, 600.0
    SCREEN_MM, SCREEN_PX = (344.0, 194.0), (1920, 1080)     # a 15.6" laptop
    EYE_R = 11.0

    def __init__(self, seed=0):
        p = np.random.default_rng(seed).normal(0, 1, (478, 3)) * [45, 60, 25] + [0, 20, 0]
        for i, x in ((33, -47), (133, -17), (362, 17), (263, 47)):          # eye corners
            p[i] = (x, 0, 5)
        for up, low, x in ((160, 144, -37), (158, 153, -27), (385, 380, 27), (387, 373, 37)):
            p[up], p[low] = (x, -5, 2), (x, 5, 2)                           # eyelids
        p[4] = (0, 40, -25)                                                # nose tip
        p[fe.SUBNASALE] = (0, 52, -14)
        p[13], p[14], p[78], p[308] = (0, 70, -8), (0, 72, -8), (-22, 71, -6), (22, 71, -6)
        p[61], p[291] = (-25, 71, -5), (25, 71, -5)                        # outer mouth corners
        self.head = p
        self.eyes = np.array([[-32.0, 0, 15], [32.0, 0, 15]])

    def screen_mm(self, px, py) -> np.ndarray:
        # The user's right (the screen's right) is the camera picture's left.
        return np.array([-(px / self.SCREEN_PX[0] - 0.5) * self.SCREEN_MM[0],
                         10 + py / self.SCREEN_PX[1] * self.SCREEN_MM[1], 0.0])

    MOUTH = (13, 14, 78, 308, 61, 291)

    LIDS = (160, 144, 158, 153, 385, 380, 387, 373)

    def landmarks(self, look_px, yaw=0.0, pitch=0.0, shift=(0.0, 0.0, 0.0), noise=0.0, rng=None,
                  roll=0.0, mouth=0.0, eyes=1.0):
        """... ``mouth``: the mouth pulled this many mm towards the user's left (- = right);
        ``eyes``: how open the eyelids are (1 = open, EAR 0.33; 0.15 = a blink)."""
        rot = _rot(yaw, pitch, roll)
        pos = np.array([0, 0, 600.0]) + np.asarray(shift, np.float64)
        head = self.head.copy()
        head[list(self.MOUTH), 0] += mouth                 # head x: + = the user's left
        head[list(self.LIDS), 1] *= eyes
        cam = head @ rot.T + pos
        target = self.screen_mm(*look_px)
        for eye, idx in zip(self.eyes, ge.IRISES):
            centre = rot @ eye + pos
            iris = centre + self.EYE_R * (target - centre) / np.linalg.norm(target - centre)
            cam[idx[0]] = iris
            for k, off in zip(idx[1:], ((5.5, 0, 0), (-5.5, 0, 0), (0, 5.5, 0), (0, -5.5, 0))):
                cam[k] = iris + rot @ np.array(off)
        px = np.stack([self.W / 2 + self.F * cam[:, 0] / cam[:, 2],
                       self.H / 2 + self.F * cam[:, 1] / cam[:, 2],
                       self.F * (cam[:, 2] - pos[2]) / pos[2]], axis=1)
        if noise:
            px = px + rng.normal(0, noise, px.shape)
        return px / np.array([self.W, self.H, self.W])


def calibrate_gaze(face, noise=0.0, rng=None, look=None, eyes_open=None, head=None):
    """Run a GazeCalibration on the synthetic face at 30 FPS -> (calibration, results).
    look(i) / eyes_open(i) / head(i): where the user looks, whether the eyes are
    open and the (yaw, pitch) at dot i (defaults: the dot, open, still)."""
    cal, t = ge.GazeCalibration(0.0, 1920, 1080), 0.0
    while not cal.finished and t < 90:
        i = cal.index
        spot = look(i) if look else (ge.GAZE_POINTS[i][0] * 1920, ge.GAZE_POINTS[i][1] * 1080)
        yaw, pitch = head(i) if head else (0.0, 0.0)
        t += 1 / FPS
        xyz = face.landmarks(spot, yaw, pitch, noise=noise, rng=rng)
        cal.update(ge.GazeFrame(xyz, face.W, face.H, 0.33, eyes_open(i) if eyes_open else True), t)
    return cal, cal.results()


class GazeTests(unittest.TestCase):
    def predict(self, res, xyz):
        tracker, model = ge.GazeTracker.from_dict(res["tracker"]), ge.GazeModel.from_dict(res["model"])
        s = tracker.sample(xyz, SyntheticFace.W, SyntheticFace.H, 0.33)
        return None if s is None else model.predict(s.features()) * [1920, 1080]

    def test_similarity_recovers_the_head(self):
        rng = np.random.default_rng(1)
        src = rng.normal(0, 50, (17, 3))
        rot = _rot(14, -9)
        dst = 1.3 * src @ rot.T + [20, -5, 300]
        s, r, t, rms = ge.similarity(src, dst)
        self.assertAlmostEqual(s, 1.3, places=6)
        np.testing.assert_allclose(r, rot, atol=1e-9)
        np.testing.assert_allclose(t, [20, -5, 300], atol=1e-6)
        self.assertLess(rms, 1e-6)

    def test_gaze_follows_the_eyes_not_the_head(self):
        face, rng = SyntheticFace(), np.random.default_rng(2)
        cal, res = calibrate_gaze(face)
        self.assertNotIn("failed", res)
        self.assertEqual(res["points"], 9)
        self.assertLess(res["error_px"], 40)
        errs = []
        for _ in range(40):
            spot = np.array([rng.uniform(100, 1820), rng.uniform(80, 1000)])
            # The head turns up to 8 degrees and tilts up to 5 while the eyes stay on the spot.
            xyz = face.landmarks(spot, rng.uniform(-8, 8), rng.uniform(-5, 5))
            errs.append(float(np.linalg.norm(self.predict(res, xyz) - spot)))
        self.assertLess(np.mean(errs), 50, errs)
        self.assertLess(np.max(errs), 120, errs)
        # Without the eye model's head tracking, a turn alone would move the gaze point
        # by hundreds of px; with it, the same spot stays the same spot.
        a = self.predict(res, face.landmarks((1500, 300), 0, 0))
        b = self.predict(res, face.landmarks((1500, 300), 8, 0))
        self.assertLess(float(np.linalg.norm(a - b)), 60)
        c = self.predict(res, face.landmarks((500, 300), 0, 0))
        self.assertGreater(float(np.linalg.norm(a - c)), 800)          # ... but the eyes count

    def test_noise_and_a_glance_away(self):
        face, rng = SyntheticFace(), np.random.default_rng(3)
        away = (1700.0, 900.0)
        _, res = calibrate_gaze(face, noise=0.2, rng=rng,
                                look=lambda i: away if i == 3 else (ge.GAZE_POINTS[i][0] * 1920,
                                                                     ge.GAZE_POINTS[i][1] * 1080))
        self.assertNotIn("failed", res)
        self.assertEqual(res["points"], 8)                 # the dot looked away from is dropped
        self.assertLess(res["error_px"], 80)
        self.assertGreater(res["noise_px"], 5)             # the camera noise is measured ...
        self.assertLess(res["noise_px"], 80)

    def test_skipped_dots_and_failures(self):
        face = SyntheticFace()
        cal, res = calibrate_gaze(face, eyes_open=lambda i: i != 2)   # eyes shut at dot 3
        self.assertEqual(cal.skipped, ["dot 3"])
        self.assertNotIn("failed", res)
        self.assertIn("Skipped: dot 3", ge.calibration_summary(res, cal.skipped, 1920))
        cal, res = calibrate_gaze(face, eyes_open=lambda i: i != 0)   # never seen at the centre
        self.assertIn("failed", res)
        self.assertIn("didn't work", ge.calibration_summary(res, cal.skipped, 1920))
        cal, res = calibrate_gaze(face, eyes_open=lambda i: i < 4)    # only 4 dots
        self.assertIn("only 4 of 9", res["failed"])

    def test_saved_calibration_round_trip(self):
        face = SyntheticFace()
        _, res = calibrate_gaze(face)
        saved = json.loads(json.dumps(res))                 # as in eyesync_settings.json
        xyz = face.landmarks((300, 800), 3, -2)
        np.testing.assert_allclose(self.predict(saved, xyz), self.predict(res, xyz), atol=0.5)
        self.assertIn("Gaze jump is on", ge.calibration_summary(res, [], 1920))
        self.assertIn("too rough", ge.calibration_summary({**res, "error_px": 400}, [], 1920))
        for bad in (None, {}, {"ref": [[1, 2, 3]], "centres": [[0, 0, 0]] * 2, "points": []},
                    {**saved["tracker"], "points": [1, 2, 3]}, {**saved["tracker"], "ref": "x"}):
            self.assertIsNone(ge.GazeTracker.from_dict(bad), bad)
        for bad in (None, {}, {**saved["model"], "form": "cubic"},
                    {**saved["model"], "coef": [[1, 2]]}, {**saved["model"], "scale": [0, 1, 1]}):
            self.assertIsNone(ge.GazeModel.from_dict(bad), bad)

    def test_tracker_ignores_glitches(self):
        face = SyntheticFace()
        t = ge.GazeTracker()
        self.assertIsNone(t.sample(face.landmarks((960, 540)), 640, 480, 0.3))   # not locked
        self.assertTrue(t.lock([face.landmarks((960, 540))] * 3, 640, 480))
        self.assertIsNotNone(t.sample(face.landmarks((960, 540)), 640, 480, 0.3))
        self.assertIsNone(t.sample(face.landmarks((960, 540))[:468], 640, 480, 0.3))  # no irises
        bent = face.landmarks((960, 540))
        bent[list(ge.RIGID[:6])] += 0.05                   # a mesh that doesn't fit the head
        self.assertIsNone(t.sample(bent, 640, 480, 0.3))


class GazeJumpTests(unittest.TestCase):
    CURSOR = np.array([960.0, 540.0])

    def feed(self, j, spot, t0, secs, spread=0.0, rng=None):
        t = t0
        for _ in range(int(secs * FPS)):
            t += 1 / FPS
            p = np.asarray(spot, np.float64)
            j.add(p + (rng.normal(0, spread, 2) if spread else 0), t)
        return t

    def test_look_then_nudge(self):
        j = ge.GazeJump(1920, 1080, error_px=60, noise_px=20)
        t = self.feed(j, (1700, 300), 0.0, 0.5)
        to = j.jump(self.CURSOR, np.array([30.0, -8.0]), t)        # head nudges right, towards it
        np.testing.assert_allclose(to, (1700, 300))
        self.assertTrue(j.absorbing(1.0, t + 0.03))                  # the nudge goes on ...
        self.assertTrue(j.absorbing(0.1, t + 0.06))
        self.assertFalse(j.absorbing(0.1, t + 0.1))                  # ... and stops
        self.assertFalse(j.absorbing(1.0, t + 0.13))                 # nothing more to absorb
        self.assertIsNone(j.jump(self.CURSOR, np.array([30.0, 0]), t + 0.2))  # cooldown

    def test_resize_keeps_the_same_distances(self):
        j = ge.GazeJump(1920, 1080, error_px=60, noise_px=20)
        min_px, spread_px = j.min_px, j.spread_px
        self.feed(j, (1700, 300), 0.0, 0.5)
        j.resize(3840, 2160)                                  # the same screen at 4K
        self.assertAlmostEqual(j.min_px, 2 * min_px)
        self.assertAlmostEqual(j.spread_px, 2 * spread_px)
        self.assertIsNone(j.fixation(0.5))                    # old-size gaze points are gone
        t = self.feed(j, (3400, 600), 0.5, 0.5)
        np.testing.assert_allclose(j.jump(self.CURSOR * 2, np.array([30.0, -8.0]), t), (3400, 600))

    def test_no_jump_when_it_would_not_help(self):
        rng = np.random.default_rng(4)
        j = ge.GazeJump(1920, 1080, error_px=60, noise_px=20)
        t = self.feed(j, (1100, 560), 0.0, 0.5)
        self.assertIsNone(j.jump(self.CURSOR, np.array([30.0, 0]), t))       # the spot is near
        t = self.feed(j, (1700, 300), t, 0.5)
        self.assertIsNone(j.jump(self.CURSOR, np.array([-30.0, 0]), t))      # head goes the other way
        self.assertIsNone(j.jump(self.CURSOR, np.array([0.0, 30.0]), t))     # ... or down
        self.assertIsNone(j.jump(self.CURSOR, np.array([30.0, -8.0]), t + 0.3))   # eyes closed since
        t = self.feed(j, (1700, 300), t + 0.3, 0.5, spread=300, rng=rng)    # eyes wander (reading)
        self.assertIsNone(j.jump(self.CURSOR, np.array([30.0, -8.0]), t))
        t = self.feed(j, (1700, 300), t, 0.1)                               # only just arrived
        self.assertIsNone(j.jump(self.CURSOR, np.array([30.0, -8.0]), t))
        t = self.feed(j, (1700, 300), t, 0.3)
        self.assertIsNotNone(j.jump(self.CURSOR, np.array([30.0, -8.0]), t))

    def test_thresholds_follow_the_calibration(self):
        rough = ge.GazeJump(1920, 1080, error_px=200, noise_px=90)
        self.assertEqual(rough.min_px, 500)
        self.assertEqual(rough.spread_px, 225)
        fine = ge.GazeJump(1920, 1080, error_px=10, noise_px=2)
        self.assertAlmostEqual(fine.min_px, 0.12 * 1920)          # never jump a short way
        self.assertAlmostEqual(fine.spread_px, 0.03 * 1920)


class RecordedMouthTests(unittest.TestCase):
    """
    The same user's 3-minute test of 2026-10-20 (tools/record_blinks.py): the
    mouth pulled to their left (recenter) and right (calibrate) on cue, after
    natural use, blinks, long blinks, head movement and tilts - replayed
    through the worker's order (tools/replay_blinks.py). Not in release
    packages, so this skips there.
    """
    RECORDING = PROJECT / "tests" / "data" / "blinks_user_2026-10-20.npz"

    def setUp(self):
        if not self.RECORDING.exists():
            self.skipTest("the recorded test isn't here (not in release packages)")
        sys.path.insert(0, str(PROJECT / "tools"))
        import replay_blinks
        self.data, cols, self.meta = replay_blinks.load(self.RECORDING)
        cal = {"blink_close_ratio": 0.525, "blink_open_ratio": 0.675, "blink_score_delta": 0.312,
               "smile_min": 0.697}
        _, _, _, self.fired = replay_blinks.replay(self.data, cols, {"calibration": cal}, 0.2)
        parts = [p[0] for p in self.meta["parts"]]
        t = self.data[:, cols["t"]]
        self.part_at = lambda tf: parts[int(self.data[np.argmin(np.abs(t - tf)), cols["part"]])]

    def test_pulls_recenter_and_calibrate(self):
        cues = [(t, p) for t, p, _ in self.meta["cues"] if p.startswith("mouth")]
        answers = []
        for t, part in cues:
            got = [n for tf, n in self.fired if t <= tf <= t + (4.5 if part == "mouth_left" else 6.5)]
            answers.append(got[0] if got else None)
        # Before the landmark rule: 0 of 6 (the left pull looked like a smile, the
        # right's score never held 3 s). The second left cue was answered with a
        # real, even big smile (both corners 0.9): pausing was right. The third
        # right cue came in the middle of one 28 s pull, already answered.
        self.assertEqual(answers, ["mouth_left", "smile", "mouth_left",
                                   "mouth_right", "mouth_right", None])

    def test_no_gestures_without_a_pull(self):
        stray = [(round(tf, 1), n, self.part_at(tf)) for tf, n in self.fired
                 if self.part_at(tf) in ("natural", "blink", "long", "move", "tilt")]
        self.assertEqual(stray, [])


class RecordedBlinkTests(unittest.TestCase):
    """
    A real user's 2-minute blink test (tools/record_blinks.py, 2026-10-18):
    natural blinking, 8 blinks and 6 winks per eye on cue, head movement and
    4 head tilts - replayed through BlinkDetector and TiltScroll the way the
    vision worker calls them. Only measurements, no pictures; kept out of the
    release packages, so this skips there.
    """
    RECORDING = PROJECT / "tests" / "data" / "blinks_user_2026-10-18.npz"
    # This user's calibration (head + blinks) from the same evening.
    CALIBRATION = {"blink_close_ratio": 0.525, "blink_open_ratio": 0.675, "blink_score_delta": 0.312}

    def setUp(self):
        if not self.RECORDING.exists():
            self.skipTest("the recorded blink test isn't here (not in release packages)")
        sys.path.insert(0, str(PROJECT / "tools"))
        import replay_blinks
        self.rb = replay_blinks
        self.data, self.cols, self.meta = replay_blinks.load(self.RECORDING)

    def answers(self):
        clicks, wheel, _, _ = self.rb.replay(self.data, self.cols, {"calibration": self.CALIBRATION}, 0.2)
        cue_clicks, used = {}, set()
        for t, part, _ in self.meta["cues"]:
            if part == "tilt":
                continue
            hit = [(i, b) for i, (tc, b) in enumerate(clicks) if t <= tc <= t + 2.0 and i not in used]
            if hit:
                used.add(hit[0][0])
            cue_clicks.setdefault(part, []).append(hit[0][1] if hit else None)
        return cue_clicks, clicks, wheel

    def test_blinks_on_cue_click(self):
        cue_clicks, _, _ = self.answers()
        # 7 of 8: the one missed was 158 ms, shorter than a click (min 0.2 s). Two firm
        # blinks used to be dropped as "half closed" (the right eye closes less).
        self.assertGreaterEqual(cue_clicks["blink"].count("left"), 7, cue_clicks["blink"])
        # This user's left winks close both eyes: a blink, which is a left click too.
        self.assertEqual(cue_clicks["left"], ["left"] * 6)
        self.assertNotIn("right", cue_clicks["blink"] + cue_clicks["left"])

    def test_tilts_scroll_the_right_way_and_only_then(self):
        _, _, wheel = self.answers()
        tilts = [(t, w) for t, p, w in self.meta["cues"] if p == "tilt"]
        for k, (t, which) in enumerate(tilts):
            until = tilts[k + 1][0] if k + 1 < len(tilts) else t + 5.0
            got = sum(u for tw, u in wheel if t <= tw < until)
            self.assertEqual(np.sign(got), 1 if which == "left" else -1, (t, which, got))
        first = tilts[0][0]
        self.assertEqual([u for tw, u in wheel if tw < first], [])   # no scrolling before

    def test_head_movement_is_never_a_tilt(self):
        """Both recordings: turning to the screen's sides leaned the head up to 15.5
        degrees (19 while talking, and a one-frame 23 in a fast move) - never a
        tilt. Only the tilt part's are, and one real 46-degree tilt in 10-20's mouth part (124-128 s)."""
        for name in ("blinks_user_2026-10-18.npz", "blinks_user_2026-10-20.npz"):
            data, cols, meta = self.rb.load(PROJECT / "tests" / "data" / name)
            parts = [p[0] for p in meta["parts"]]
            s, wrong, part = li.TiltScroll(), [], None
            for row in data:
                t = float(row[cols["t"]])
                if row[cols["part"]] != part:            # the recorder's pause between parts:
                    part = row[cols["part"]]             # no face seen, as live
                    s.update(None, t)
                s.update(float(row[cols["roll"]]) if row[cols["face"]] else None, t)
                real = parts[int(part)] == "tilt" or ("10-20" in name and 124 <= t <= 128.5)
                if s.tilted and not real:
                    wrong.append(round(t, 1))
            self.assertEqual(wrong, [], name)

    def test_natural_blinks_while_reading_do_not_click(self):
        """Both recordings' natural parts: blinks of up to 274-286 ms clicked with the
        tick at 0.2 s (1.5.3); none of them reaches the tick now."""
        for name in ("blinks_user_2026-10-18.npz", "blinks_user_2026-10-20.npz"):
            with self.subTest(name):
                data, cols, meta = self.rb.load(PROJECT / "tests" / "data" / name)
                clicks, _, _, _ = self.rb.replay(data, cols, {"calibration": self.CALIBRATION},
                                                 li.BLINK_MIN_S)
                parts = [p[0] for p in meta["parts"]]
                nat = data[data[:, cols["part"]] == parts.index("natural"), cols["t"]]
                # (10-20 opens with a one-second closure, before the reading: a click.)
                short = [t for t, _ in clicks if nat[0] + 5 <= t <= nat[-1]]
                self.assertEqual(short, [])


class TiltScrollTests(unittest.TestCase):
    def run_tilts(self, segments, s=None):
        """segments: (seconds, head tilt in degrees, + = left) -> (wheel turns per frame, scroller)."""
        s, t, out = s or li.TiltScroll(), 0.0, []
        for secs, roll in segments:
            for _ in range(int(round(secs * FPS))):
                t += 1 / FPS
                out.append(s.update(roll, t))
        return out, s

    def test_tilt_left_scrolls_up_right_scrolls_down(self):
        up, _ = self.run_tilts([(2, 0.0), (1.0, 30.0), (1, 0.0)])
        down, _ = self.run_tilts([(2, 0.0), (1.0, -30.0), (1, 0.0)])
        self.assertTrue(all(u >= 0 for u in up) and sum(up) > 0)
        self.assertTrue(all(d <= 0 for d in down) and sum(down) < 0)
        self.assertEqual(sum(up), -sum(down))
        self.assertTrue(all(u % li.TiltScroll.WHEEL == 0 for u in up))   # whole notches
        first = next(i for i, u in enumerate(up) if u)
        delay = li.TiltScroll.START_S + (li.TiltScroll.MEDIAN_FRAMES // 2) / FPS   # (the median)
        self.assertAlmostEqual((first + 1) / FPS - 2.0, delay, delta=0.07)

    def test_further_is_faster_and_straightening_stops(self):
        gentle, _ = self.run_tilts([(2, 0.0), (3, 24.0)])
        far, _ = self.run_tilts([(2, 0.0), (3, 40.0)])
        self.assertGreater(sum(far), 3 * sum(gentle))
        notches = sum(far) / li.TiltScroll.WHEEL
        delay = li.TiltScroll.START_S + (li.TiltScroll.MEDIAN_FRAMES // 2) / FPS
        self.assertAlmostEqual(notches, 1 + li.TiltScroll.FAST * (3 - delay), delta=2)
        out, s = self.run_tilts([(2, 0.0), (1, 30.0), (0.5, 8.0), (1, 8.0)])
        self.assertFalse(s.active)
        self.assertEqual(sum(out[int(3.1 * FPS):]), 0)               # nothing after straightening

    def test_brief_or_small_tilts_do_nothing(self):
        out, s = self.run_tilts([(2, 0.0), (0.25, 30.0), (1, 0.0), (3, 12.0), (3, -12.0)])
        self.assertEqual(sum(abs(u) for u in out), 0)
        self.assertFalse(s.tilted)

    def test_a_head_turned_to_a_screen_side_is_not_a_tilt(self):
        """A real user's head leans as it turns to a screen side: up to 15.5 degrees
        in their recordings, 19 while talking. Not a tilt: no scrolling, and
        the cursor and blinks carry on (``tilted`` stays false)."""
        for lean in (15.5, -15.5, 19.0, -19.0, 21.0):
            with self.subTest(lean=lean):
                out, s = self.run_tilts([(2, 0.0), (0.3, lean / 2), (3, lean), (1, 0.0)])
                self.assertEqual(sum(abs(u) for u in out), 0)
                self.assertFalse(s.tilted)

    def test_a_usual_head_tilt_is_learned(self):
        """Someone who sits with the head tilted 8 degrees tilts 22 more to scroll."""
        out, s = self.run_tilts([(6, 8.0), (2, 27.0)])
        self.assertEqual(sum(out), 0)
        self.assertAlmostEqual(s.usual, 8.0, delta=1.0)
        out, _ = self.run_tilts([(6, 8.0), (2, 32.0)])
        self.assertGreater(sum(out), 0)

    def test_losing_the_face_stops(self):
        s = li.TiltScroll()
        self.run_tilts([(2, 0.0), (1, 30.0)], s)
        self.assertTrue(s.active)
        self.assertEqual(s.update(None, 10.0), 0)
        self.assertFalse(s.active)

    def test_wheel_output(self):
        class FakeMouse:
            available = True

            def __init__(self):
                self.turns = []

            def scroll(self, units):
                self.turns.append(units)

        async def run():
            state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
            args = types.SimpleNamespace(always_active=True)
            mouse = FakeMouse()
            task = asyncio.create_task(li.scroll_task(args, state, mouse))
            await asyncio.sleep(0.05)
            state.add_scroll(240)
            await asyncio.sleep(0.05)
            state.update(paused=True)                  # paused: the wheel isn't turned
            state.add_scroll(-120)
            await asyncio.sleep(0.05)
            state.update(paused=False)
            state.add_scroll(-360)
            await asyncio.sleep(0.05)
            state.stop.set()
            await task
            return mouse.turns
        self.assertEqual(asyncio.run(run()), [240, -360])


class FakeGazeMesh:
    """A face mesh that sees whatever SyntheticFace frame the test puts in ``xyz``."""
    last_pose = last_blendshapes = None

    def __init__(self):
        self.xyz = self.last_xyz = None

    def process(self, frame):
        self.last_xyz = self.xyz
        return None if self.xyz is None else self.xyz[:, :2].astype(np.float32)


class WorkerTests(unittest.TestCase):
    """The vision worker end to end with SyntheticFace: gaze jump, scrolling."""
    FRAME = np.full((480, 640, 3), 128, np.uint8)

    def setUp(self):
        self.saved = (li.load_settings, li.save_settings, li.time, li.desktop.cursor_position)
        self.clock, self.settings = 100.0, {}
        li.load_settings = lambda: {}
        li.save_settings = lambda **kw: self.settings.update(kw)
        li.time = types.SimpleNamespace(monotonic=lambda: self.clock, perf_counter=time.perf_counter,
                                        strftime=time.strftime, sleep=lambda s: None)
        li.desktop.cursor_position = lambda: None

    def tearDown(self):
        li.load_settings, li.save_settings, li.time, li.desktop.cursor_position = self.saved

    def run_frames(self, secs, look, shift=lambda u: (0.0, 0.0, 0.0), roll=lambda u: 0.0, mouth=0.0,
                   eyes=1.0):
        """``secs`` of frames looking at ``look``; ``shift(u)`` / ``roll(u)``: the head's
        position and tilt, u = 0..1 through the frames; ``mouth``, ``eyes``: see SyntheticFace."""
        n = int(round(secs * FPS))
        for k in range(n):
            self.clock += 1 / FPS
            u = (k + 1) / n
            self.mesh.xyz = self.face.landmarks(look, shift=shift(u), roll=roll(u), mouth=mouth,
                                                eyes=eyes)
            self.worker._process(self.FRAME, self.mesh)

    def cursor(self) -> np.ndarray:
        snap = self.state.read()
        return np.array([snap.x, snap.y])

    def start(self, calibrate_eyes=True):
        args = types.SimpleNamespace(ear_threshold=None, close_ratio=0.55, open_ratio=0.70,
                                     min_blink=0.2, max_blink=1.5, click_cooldown=0.6,
                                     dwell_time=1.0, close_gesture_hold=3.0, close_gesture_mar=0.6,
                                     no_close_gesture=False, no_click_sound=True, always_active=True)
        screen = fe.ScreenSize(1920, 1080)
        self.state = li.SharedState(screen, 1, False)
        self.worker = li.VisionWorker(args, screen, self.state, meshes=types.SimpleNamespace())
        self.face, self.mesh = SyntheticFace(), FakeGazeMesh()
        if not calibrate_eyes:
            self.run_frames(1.0, (960, 540))
            return
        # The eye calibration, through the worker: look at each dot it asks for.
        li.Actions(self.state).calibrate_gaze()
        self.assertEqual(self.state.calibrate_kind, "gaze")
        for _ in range(40 * FPS):
            step = self.state.read().calib_step
            if step == "done":
                break
            i = ge.GAZE_STEPS.index(step) if step in ge.GAZE_STEPS else 0
            self.run_frames(1 / FPS, (ge.GAZE_POINTS[i][0] * 1920, ge.GAZE_POINTS[i][1] * 1080))
        snap = self.state.read()
        self.assertEqual(snap.calib_step, "done")
        self.assertIn("Gaze jump is on", snap.calib_text)
        self.assertTrue(snap.gaze_jump and snap.gaze_calibrated)
        self.assertIn("model", self.settings["gaze"])
        self.assertTrue(self.settings["gaze_jump"])
        self.run_frames(0.5, (960, 540))                   # settle back at the centre
        self.state.recenter_request.set()
        self.run_frames(0.3, (960, 540))

    def nudge_right(self, look, start=0.0):
        # The head moves 12 mm to the user's right (picture left) in 0.25 s, then holds.
        self.run_frames(0.25, look, lambda u: (start - 12.0 * u, 0.0, 0.0))
        self.run_frames(0.6, look, lambda u: (start - 12.0, 0.0, 0.0))

    def test_tilting_the_head_scrolls_and_holds_the_cursor(self):
        self.start(calibrate_eyes=False)
        here = self.cursor()
        self.run_frames(0.3, (960, 540), roll=lambda u: 30.0 * u)      # tilt left ...
        self.run_frames(1.5, (960, 540), roll=lambda u: 30.0)          # ... and hold
        self.assertGreater(self.state.read().scroll_units, 0)          # scrolled up
        self.assertLess(float(np.linalg.norm(self.cursor() - here)), 5)   # the cursor stayed
        view = self.worker._render_preview(self.FRAME, self.state.read(), self.clock)
        self.assertEqual(view.ndim, 3)
        up = self.state.read().scroll_units
        self.run_frames(0.3, (960, 540), roll=lambda u: 30.0 * (1 - u))   # straighten up
        self.run_frames(0.5, (960, 540))
        self.assertFalse(self.worker.scroll.active)
        self.run_frames(0.3, (960, 540), roll=lambda u: -30.0 * u)     # tilt right ...
        self.run_frames(1.5, (960, 540), roll=lambda u: -30.0)
        self.assertLess(self.state.read().scroll_units, up)            # ... scrolled down
        self.state.update(gestures=False)                              # face gestures off: none
        before = self.state.read().scroll_units
        self.run_frames(1.5, (960, 540), roll=lambda u: -30.0)
        self.assertEqual(self.state.read().scroll_units, before)

    def test_a_tilt_while_paused_does_nothing(self):
        # The 1.5.0 log: tilts after a smile-pause were logged as scrolling.
        self.start(calibrate_eyes=False)
        self.state.update(paused=True)
        with self.assertNoLogs("eyesync", level="INFO"):
            self.run_frames(0.3, (960, 540), roll=lambda u: 30.0 * u)
            self.run_frames(1.5, (960, 540), roll=lambda u: 30.0)
        self.assertEqual(self.state.read().scroll_units, 0)
        self.assertFalse(self.worker.scroll.tilted)
        self.state.update(paused=False)                                 # resumed, still tilted:
        self.run_frames(1.0, (960, 540), roll=lambda u: 30.0)           # now it scrolls
        self.assertGreater(self.state.read().scroll_units, 0)

    def test_a_blink_at_a_screen_corner_clicks(self):
        """The user (1.5.4): "at extreme corners the eye blink is not working" and
        "sometimes it is taking my moving head for scroll". Turning to a screen
        side leans their head up to 15.5 degrees; from 15 (until 1.5.4) that
        counted as a tilt: the cursor froze, went back, and blinks were ignored."""
        self.start(calibrate_eyes=False)
        corner = (-70.0, -25.0, 0.0)                                     # towards the top right
        self.run_frames(0.8, (960, 540), lambda u: tuple(u * c for c in corner), roll=lambda u: -19.0 * u)
        self.run_frames(0.8, (960, 540), lambda u: corner, roll=lambda u: -19.0)
        self.assertFalse(self.worker.scroll.tilted)
        there = self.cursor()
        self.assertGreater(there[0], 1300)                               # it got out there
        self.run_frames(0.7, (960, 540), lambda u: corner, roll=lambda u: -19.0, eyes=0.15)   # a blink
        self.run_frames(0.5, (960, 540), lambda u: corner, roll=lambda u: -19.0)
        snap = self.state.read()
        self.assertEqual((snap.click_count, snap.right_click_count, snap.scroll_units), (1, 0, 0))
        np.testing.assert_allclose(self.cursor(), there, atol=40)       # and stayed there
        # A real tilt there still scrolls.
        self.run_frames(0.2, (960, 540), lambda u: corner, roll=lambda u: -19.0 - 21.0 * u)
        self.run_frames(1.0, (960, 540), lambda u: corner, roll=lambda u: -40.0)
        self.assertLess(self.state.read().scroll_units, 0)

    def test_the_cursor_follows_a_display_change(self):
        self.start(calibrate_eyes=False)
        self.run_frames(0.5, (960, 540), lambda u: (-20.0 * u, 0.0, 0.0))  # to the right of centre
        before = self.cursor()
        span = self.worker.motion.max_speed
        bigger = fe.ScreenSize(2560, 1440)
        with self.assertLogs("eyesync", level="INFO") as logs:
            self.worker._follow_screen(self.clock + li.SCREEN_CHECK_S, probe=lambda: bigger)
        self.assertIn("1920x1080 -> 2560x1440", "\n".join(logs.output))
        self.assertEqual((self.worker.screen.width, self.worker.screen.height), (2560, 1440))
        np.testing.assert_allclose(self.cursor(), before * [2560 / 1920, 1440 / 1080], atol=1)
        self.assertAlmostEqual(self.worker.motion.max_speed, span * 2560 / 1920)
        # The whole new screen is reachable: a big turn reaches its far edge.
        self.run_frames(2.0, (960, 540), lambda u: (-20.0 - 150.0 * u, 0.0, 0.0))
        self.assertGreater(self.cursor()[0], 1920 + 300)
        # The same size again, or no answer: nothing changes.
        self.worker._follow_screen(self.clock + 2 * li.SCREEN_CHECK_S, probe=lambda: None)
        self.assertEqual(self.worker.screen.width, 2560)
        # During the eye calibration: it's cancelled (its dots were for the old size).
        li.Actions(self.state).calibrate_gaze()
        self.run_frames(0.5, (960, 540))
        self.assertIsInstance(self.worker.calib, ge.GazeCalibration)
        self.worker._follow_screen(self.clock + 3 * li.SCREEN_CHECK_S,
                                   probe=lambda: fe.ScreenSize(1920, 1080))
        self.run_frames(0.1, (960, 540))
        self.assertIsNone(self.worker.calib)
        self.assertFalse(li.calibrating(self.state.read()))           # the window hides the dots

    def test_mouth_left_recenters_and_mouth_right_calibrates(self):
        self.start(calibrate_eyes=False)
        rest = {"mouthLeft": 0.003, "mouthRight": 0.003, "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0}
        # The "wrong" score rises each time: which way the mouth MOVED decides.
        self.mesh.last_blendshapes = rest
        self.run_frames(2.0, (960, 540))
        self.mesh.last_blendshapes = {**rest, "mouthRight": 0.6}
        self.run_frames(1.7, (960, 540), mouth=6.0)                 # to the user's left
        self.assertTrue(self.state.recenter_request.is_set() or self.worker.gestures.active is None)
        self.mesh.last_blendshapes = rest
        self.run_frames(1.0, (960, 540))
        self.assertFalse(self.state.recenter_request.is_set())     # taken: it recentered
        self.assertFalse(self.state.calibrate_request.is_set())
        self.mesh.last_blendshapes = {**rest, "mouthLeft": 0.6}
        self.run_frames(1.7, (960, 540), mouth=-6.0)                # to the right: still counting down
        self.assertEqual(self.worker.gestures.active, "mouth_right")
        self.assertTrue(self.worker.gesture_warned)                 # it beeped
        view = self.worker._render_preview(self.FRAME, self.state.read(), self.clock)
        self.assertEqual(view.ndim, 3)                              # the countdown is drawn
        self.assertFalse(self.state.calibrate_request.is_set())
        self.run_frames(1.5, (960, 540), mouth=-6.0)
        # Held 3 s: the (head) calibration has started (the worker takes the request at once).
        self.assertIsInstance(self.worker.calib, li.Calibration)
        self.assertEqual(self.state.calibrate_kind, "head")

    def test_look_then_nudge_jumps(self):
        self.start()
        spot = (1700.0, 300.0)
        self.run_frames(0.6, spot)                          # just looking: the cursor stays
        self.assertLess(float(np.linalg.norm(self.cursor() - (960, 540))), 5)
        self.nudge_right(spot)
        self.assertEqual(self.worker.jumps, 1)
        self.assertLess(float(np.linalg.norm(self.cursor() - spot)), 150, self.cursor())
        view = self.worker._render_preview(self.FRAME, self.state.read(), self.clock)
        self.assertEqual(view.ndim, 3)                      # gaze rays drawn without trouble

    def test_no_jump_when_off_or_looking_elsewhere(self):
        self.start()
        self.state.update(gaze_jump=False)
        self.run_frames(0.6, (1700, 300))
        self.nudge_right((1700, 300))
        self.assertEqual(self.worker.jumps, 0)
        self.assertLess(self.cursor()[0], 1300)             # only the head moved it
        self.state.update(gaze_jump=True)
        self.state.recenter_request.set()
        held = lambda u: (-12.0, 0.0, 0.0)                  # noqa: E731 - where the nudge left it
        self.run_frames(0.6, (200, 540), held)              # looking left ...
        self.assertLess(float(np.linalg.norm(self.cursor() - (960, 540))), 5)
        self.nudge_right((200, 540), start=-12.0)           # ... while the head goes right
        self.assertEqual(self.worker.jumps, 0)
        self.assertGreater(self.cursor()[0], 960)           # the head moved it, to the right


class GazeControlTests(unittest.TestCase):
    def test_commands_and_status(self):
        state = li.SharedState(fe.ScreenSize(1920, 1080), 1, False)
        actions, args = li.Actions(state), types.SimpleNamespace(always_active=False)
        saved, li.save_settings = li.save_settings, lambda **kw: None
        try:
            r = li.handle_command(json.dumps({"action": "gaze", "value": False}), state, actions, args)
            self.assertEqual(r["gaze"], {"calibrated": "", "on": False, "error_px": 0})
            self.assertEqual(li.handle_command(json.dumps({"action": "gaze", "value": "x"}),
                                               state, actions, args)["type"], "error")
            li.handle_command(json.dumps({"action": "calibrate", "value": "gaze"}), state, actions, args)
            self.assertTrue(state.calibrate_request.is_set())
            self.assertEqual(state.calibrate_kind, "gaze")
            li.handle_command(json.dumps({"action": "calibrate", "value": "cancel"}), state, actions, args)
            self.assertTrue(state.calibrate_cancel.is_set())
            li.handle_command(json.dumps({"action": "calibrate"}), state, actions, args)
            self.assertEqual(state.calibrate_kind, "head")
        finally:
            li.save_settings = saved

    def test_window_gaze_controls(self):
        import tkinter as tk

        import main_window
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        saved = li.save_settings
        li.save_settings = lambda **kw: None
        try:
            root.withdraw()
            win = main_window.MainWindow(root, False, PROJECT / "README.md", PROJECT)
            st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
            args = types.SimpleNamespace(always_active=False)
            win.attach(st, li.Actions(st), lambda: li.status_of(st.read(), args))
            win.refresh()
            self.assertIn("Eyes not calibrated", win.gaze_info_var.get())
            win.gaze_var.set(False)
            win._set_gaze()
            self.assertFalse(st.read().gaze_jump)
            ov = win.overlay

            def hidden_show():
                ov.top = tk.Toplevel(root)
                ov.top.withdraw()
                ov.canvas = tk.Canvas(ov.top)
                ov.drawn = None
            ov._show = hidden_show
            win.gaze_btn.invoke()
            self.assertEqual(st.calibrate_kind, "gaze")
            st.update(calib_step="gaze4", calib_text="Follow the dot", calib_progress=0.5)
            win.refresh()
            self.assertEqual(ov.drawn, ("gaze4", "Follow the dot", 0.5))
            win.gaze_btn.invoke()                           # pressed again: cancel
            self.assertTrue(st.calibrate_cancel.is_set())
            win.refresh()
            self.assertIsNone(ov.top)
            st.update(gaze_calibrated="2026-10-18", gaze_error_px=90, gaze_jump=True)
            win.refresh()
            self.assertIn("about 90 px", win.gaze_info_var.get())
            self.assertTrue(win.gaze_var.get())
        finally:
            li.save_settings = saved
            root.destroy()


class SerialTaskTests(unittest.TestCase):
    """serial_task with a fake XIAO."""

    def test_the_xiao_is_told_a_new_screen_size(self):
        written = []

        class FakePort:
            in_waiting = 0

            def write(self, data):
                written.append(data.decode())

            def close(self):
                pass

        saved = li.find_esp32_port, li.open_serial
        li.find_esp32_port, li.open_serial = (lambda: "COM9"), (lambda port, baud: FakePort())
        screen = fe.ScreenSize(1920, 1080)
        state = li.SharedState(screen, 1, False)
        args = types.SimpleNamespace(port=None, baud=115200, mouse="auto", fps=30, always_active=True)

        async def scenario():
            task = asyncio.create_task(li.serial_task(args, state, screen))
            await asyncio.sleep(0.2)
            screen.width, screen.height = 2560, 1440        # VisionWorker._follow_screen
            await asyncio.sleep(0.2)
            state.stop.set()
            await asyncio.wait_for(task, 2)

        try:
            with self.assertLogs("eyesync", level="INFO"):
                asyncio.run(scenario())
        finally:
            li.find_esp32_port, li.open_serial = saved
        text = "".join(written)
        self.assertIn("M,SCREEN,1920,1080\n", text)
        self.assertIn("M,SCREEN,2560,1440\n", text)
        self.assertLess(text.index("M,SCREEN,1920,1080"), text.index("M,SCREEN,2560,1440"))


# --------------------------------------------------------------------------- #
# Release readiness (1.5.6): the blink test for new faces, camera retries, the build
# --------------------------------------------------------------------------- #

class DatasetTests(unittest.TestCase):
    """
    Other people's faces: tools/video_features.py's numbers from the GRID
    audiovisual corpus (Cooke, Barker, Cunningham & Shao 2006, CC BY 4.0 -
    people reading short sentences aloud, not using EyeSync), kept outside
    EyeSync (EYESYNC_DATASETS, default ..\\EyeSync-datasets); skipped without
    them. Everything EyeSync would do there is a false click, gesture or scroll.
    """
    FOLDER = Path(os.environ.get("EYESYNC_DATASETS", PROJECT.parent / "EyeSync-datasets")) / "grid" / "features2"

    reports: list = []

    @classmethod
    def setUpClass(cls):
        files = sorted(cls.FOLDER.glob("s*.npz")) if cls.FOLDER.exists() else []
        if files:
            sys.path.insert(0, str(PROJECT / "tools"))
            import dataset_report
            cls.reports = [dataset_report.report(p) for p in files]   # replayed once: ~20 s

    def setUp(self):
        if not self.reports:
            self.skipTest(f"no dataset numbers in {self.FOLDER}")

    def test_talking_fires_no_scroll_quit_or_mouth_side_gesture(self):
        for r in self.reports:
            with self.subTest(r["name"]):
                self.assertEqual(r["notches"], 0)
                self.assertLess(r["mouth_open_s"], 1.0)               # the quit gesture needs 3 s
                self.assertLessEqual(set(r["gestures"]), {"smile"})   # never recenter / calibrate

    def test_real_smiles_pause_rarely(self):
        """A big smile held 1.5 s is the pause gesture, whoever smiles: s11
        smiled for real while reading (score 0.7-0.9 held 1.5 s and more, the
        mouth nearly shut) and paused 3 times in 25 minutes; nine never did."""
        for r in self.reports:
            with self.subTest(r["name"]):
                self.assertLessEqual(r["gestures"].get("smile", 0) / r["minutes"], 0.2)

    def test_natural_blinks_click_rarely(self):
        """On s1-s4, where the tick rule was tuned: 4.2 false clicks a minute
        with the tick at any closure state (1.5.6), 3.0 with it only while the
        eyes are shut (1.5.7). Held out (s5, s6, s7, s11, s15, s16): 1.16; all
        ten 1.89. Most are one reader's long closures (s1: 9.4 a minute, median
        520 ms) and two readers whose heads move while reading (s11, s15)."""
        minutes = clicks = rights = 0
        for r in self.reports:
            minutes, clicks, rights = minutes + r["minutes"], clicks + r["clicks"], rights + r["right"]
        self.assertLess(clicks / minutes, 3.2)
        self.assertEqual(rights, 0)                                  # no half-shut "winks"


class BlinkTestButtonTests(unittest.TestCase):
    """The window's Blink test: the camera goes to the test while it runs, then back."""

    def test_the_action_runs_the_test_once_and_hands_the_camera_back(self):
        st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
        actions = li.Actions(st)
        started, release = [], threading.Event()

        def run(cmd):                                   # the test's window, standing in
            started.append(cmd)
            self.assertTrue(st.read().blink_test)       # the camera is the test's meanwhile
            release.wait(5)
            return types.SimpleNamespace(returncode=0)

        with self.assertLogs("eyesync", level="INFO") as logs:
            thread = actions.blink_test(run=run)
            self.assertIsNotNone(thread)
            self.assertIsNone(actions.blink_test(run=run))      # a second press: still the one
            self.assertTrue(st.read().blink_test)
            self.assertEqual(st.read().hint, li.Actions.BLINK_TEST_HINT)
            release.set()
            thread.join(5)
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0], li.app_paths.blink_test_command())
        self.assertFalse(st.read().blink_test)
        self.assertEqual(st.read().hint, "")
        self.assertIn("Blink test ended (exit code 0)", "\n".join(logs.output))

    def test_a_test_that_cannot_start_still_gives_the_camera_back(self):
        st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)

        def run(cmd):
            raise OSError("no such file")

        with self.assertLogs("eyesync", level="ERROR"):
            li.Actions(st).blink_test(run=run).join(5)
        self.assertFalse(st.read().blink_test)

    def test_the_supervisor_stops_the_camera_for_the_test(self):
        class FakeWorker(threading.Thread):
            made = []

            def __init__(self, args, screen, state, meshes):
                super().__init__(daemon=True)
                self.halt, self.started_at = threading.Event(), time.monotonic()
                FakeWorker.made.append(self)

            def run(self):
                self.halt.wait(10)

        class FakeMeshes:
            def expect_preload(self):
                pass

            def preload(self):
                pass

            def close(self):
                pass

        st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
        args = types.SimpleNamespace(always_active=True, idle_grace=30.0, stall_timeout=60.0)
        saved = li.VisionWorker, li.MeshHolder
        li.VisionWorker, li.MeshHolder = FakeWorker, FakeMeshes

        async def scenario():
            task = asyncio.create_task(li.supervise_vision(args, fe.ScreenSize(1920, 1080), st))
            await asyncio.sleep(0.6)
            self.assertEqual(len(FakeWorker.made), 1)               # the camera is on
            st.update(blink_test=True)
            await asyncio.sleep(0.8)
            self.assertTrue(FakeWorker.made[0].halt.is_set())       # ... off for the test
            self.assertEqual(len(FakeWorker.made), 1)               # ... and stays off
            st.update(blink_test=False)
            await asyncio.sleep(0.8)
            self.assertEqual(len(FakeWorker.made), 2)               # on again after it
            st.stop.set()
            await asyncio.wait_for(task, 5)

        try:
            with self.assertLogs("eyesync", level="INFO") as logs:
                asyncio.run(scenario())
        finally:
            li.VisionWorker, li.MeshHolder = saved
        self.assertIn("The blink test needs the camera", "\n".join(logs.output))

    def test_the_window_has_the_button(self):
        import tkinter as tk

        import main_window
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        try:
            root.withdraw()
            win = main_window.MainWindow(root, False, PROJECT / "README.md", PROJECT)
            st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
            actions, pressed = li.Actions(st), []
            actions.blink_test = lambda: pressed.append(1)
            args = types.SimpleNamespace(always_active=False)
            win.attach(st, actions, lambda: li.status_of(st.read(), args))
            win.blink_test_btn.invoke()
            self.assertEqual(pressed, [1])
            st.update(blink_test=True)                 # running: the button waits for it
            win.refresh()
            self.assertTrue(win.blink_test_btn.instate(["disabled"]))
            st.update(blink_test=False)
            win.refresh()
            self.assertFalse(win.blink_test_btn.instate(["disabled"]))
        finally:
            root.destroy()


class CameraRetryTests(unittest.TestCase):
    def test_a_camera_held_by_another_app_is_logged_once_a_minute(self):
        """An app holding the camera for an hour wrote ~4 lines every few seconds."""
        clock = [0.0]
        st = li.SharedState(fe.ScreenSize(1920, 1080), 2, False)
        tries, quiet = [], []                          # each try, and whether it was quiet

        class Halt:
            def wait(self, s):
                clock[0] += s + 3.0                    # a round: the retry wait + three cameras
                return False

        worker = types.SimpleNamespace(halted=False, state=st, halt=Halt(),
                                       args=types.SimpleNamespace(camera=None),
                                       CAMERA_HINT=li.VisionWorker.CAMERA_HINT,
                                       CAMERA_LOG_S=li.VisionWorker.CAMERA_LOG_S)

        def open_any_camera(index, quiet=False):
            tries.append(clock[0])
            if clock[0] < 600:                         # ten minutes held by another app
                flags.append(quiet)
                raise RuntimeError("Could not open camera")
            return "camera"

        flags = quiet
        saved = li.fe.open_any_camera, li.time
        li.fe.open_any_camera = open_any_camera
        li.time = types.SimpleNamespace(monotonic=lambda: clock[0])
        try:
            with self.assertLogs("eyesync", level="INFO") as logs:
                got = li.VisionWorker._open_camera(worker)
        finally:
            li.fe.open_any_camera, li.time = saved
        self.assertEqual(got, "camera")
        self.assertGreater(len(tries), 100)                       # it kept trying every few s
        loud = [line for line in logs.output if "camera" in line.lower()]
        self.assertLessEqual(len(loud), 12)                       # ~once a minute, + "working"
        self.assertEqual(quiet.count(False), len(loud) - 1)       # the camera's own lines too
        self.assertIn("Working again after", loud[-1])
        self.assertIn("Could not open camera", loud[0])
        self.assertEqual(st.read().hint, "")                      # the hint is gone once it works


class StartupTests(unittest.TestCase):
    """What happens between the website's Connect and a moving cursor."""

    def test_a_worker_waits_for_the_preload(self):
        """1.5.11 built a second face mesh when the camera opened (~2 s) before the
        preload (~3 s) was done; the two loads slowed each other down."""
        made = []

        class SlowMesh:
            def __init__(self):
                time.sleep(0.3)
                made.append(self)

            def close(self):
                pass

        saved, li.fe.FaceMeshBackend = li.fe.FaceMeshBackend, SlowMesh
        try:
            holder = li.MeshHolder()
            holder.expect_preload()
            threading.Thread(target=holder.preload, daemon=True).start()
            mesh = holder.take()
        finally:
            li.fe.FaceMeshBackend = saved
        self.assertEqual(len(made), 1)
        self.assertIs(mesh, made[0])

    def test_matplotlib_waits_until_something_uses_it(self):
        """MediaPipe's drawing helpers import matplotlib.pyplot (~0.6 s of every
        start); EyeSync never plots, so a stand-in takes its place."""
        mine = lambda k: k == "matplotlib" or k.startswith("matplotlib.")   # noqa: E731
        saved = {k: v for k, v in sys.modules.items() if mine(k)}
        for k in saved:
            del sys.modules[k]
        try:
            fe._defer_matplotlib()
            import matplotlib.pyplot as plt                     # what the helpers do
            self.assertEqual(type(plt).__name__, "Deferred")
            self.assertEqual(type(sys.modules["matplotlib"]).__name__, "Deferred")
        finally:
            for k in [k for k in sys.modules if mine(k)]:
                del sys.modules[k]
            sys.modules.update(saved)


class RecordingFileTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(PROJECT / "tools"))
        try:
            import record_blinks
        except ImportError as exc:
            self.skipTest(f"the blink test isn't here: {exc}")
        self.rec = record_blinks

    def test_a_recording_carries_its_calibration_and_version(self):
        cal = {"blink_close_ratio": 0.5, "date": "2026-10-22"}
        meta = self.rec.recording_meta([(1.0, "blink", "")], False, {"calibration": cal, "x": 1})
        self.assertEqual(meta["calibration"], cal)
        self.assertEqual(meta["version"], li.site_access.VERSION)
        self.assertTrue(meta["complete"])
        self.assertEqual([p[0] for p in meta["parts"]][:2], ["natural", "blink"])
        self.assertIn("corners", [p[0] for p in meta["parts"]])
        json.dumps(meta)                                           # saved as JSON
        self.assertNotIn("x", meta)                                # only the calibration

    def test_recordings_go_next_to_the_program_or_in_tools(self):
        where = self.rec.recordings_dir()
        if li.app_paths.FROZEN:
            self.assertEqual(where, li.app_paths.DATA_DIR / "recordings")
        else:
            self.assertEqual(where, PROJECT / "tools" / "recordings")


class BuildTests(unittest.TestCase):
    """build_exe.py's version information and signing commands (not in the exe)."""

    def setUp(self):
        try:
            import build_exe
        except ImportError:
            self.skipTest("build_exe.py isn't here (inside EyeSync.exe)")
        self.b = build_exe

    def test_version_information(self):
        import ast
        with tempfile.TemporaryDirectory() as tmp:
            saved = self.b.BUILD
            self.b.BUILD = Path(tmp)
            try:
                text = self.b.version_file("1.5.6").read_text(encoding="utf-8")
            finally:
                self.b.BUILD = saved
        ast.parse(text)                                            # PyInstaller evaluates it
        self.assertIn("filevers=(1, 5, 6, 0)", text)
        self.assertIn("StringStruct('ProductVersion', '1.5.6')", text)
        self.assertIn("StringStruct('OriginalFilename', 'EyeSync.exe')", text)

    def test_signing_commands_and_no_signing(self):
        sign, verify = self.b.sign_commands(r"C:\SDK\signtool.exe", '/fd SHA256 /f "my cert.pfx"',
                                            Path(r"E:\Release dir\EyeSync.exe"))
        self.assertEqual(sign, r'"C:\SDK\signtool.exe" sign /fd SHA256 /f "my cert.pfx" '
                               r'"E:\Release dir\EyeSync.exe"')
        self.assertEqual(verify, r'"C:\SDK\signtool.exe" verify /pa "E:\Release dir\EyeSync.exe"')
        saved = os.environ.pop("EYESYNC_SIGN", None)
        try:
            self.assertTrue(self.b.sign(Path("nothing.exe")))     # not asked for: fine, unsigned
            os.environ["EYESYNC_SIGN"] = "/fd SHA256"
            saved_find = self.b.find_signtool
            self.b.find_signtool = lambda: None
            try:
                self.assertFalse(self.b.sign(Path("nothing.exe")))  # asked for, can't: stop
            finally:
                self.b.find_signtool = saved_find
        finally:
            os.environ.pop("EYESYNC_SIGN", None)
            if saved is not None:
                os.environ["EYESYNC_SIGN"] = saved


if __name__ == "__main__":
    unittest.main()
