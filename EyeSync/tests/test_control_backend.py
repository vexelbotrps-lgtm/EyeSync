"""Regression checks for the imported gaze and facial click controls."""
import sys
import unittest
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_backend.adapter import ControlBackend  # noqa: E402
from control_backend import voice_commands  # noqa: E402
import live_inference as li  # noqa: E402
import voice_typing as vt  # noqa: E402


def face(smile=1.0, mouth=0.02):
    points = np.full((468, 2), 0.5, dtype=np.float32)
    points[33], points[133] = [0.3, 0.3], [0.4, 0.3]
    points[362], points[263] = [0.6, 0.3], [0.7, 0.3]
    points[61], points[291] = [0.5 - 0.1 * smile, 0.65], [0.5 + 0.1 * smile, 0.65]
    points[13], points[14] = [0.5, 0.65 - mouth / 2], [0.5, 0.65 + mouth / 2]
    return points


class ControlBackendTests(unittest.TestCase):
    def test_model_moves_and_smile_left_clicks(self):
        backend = ControlBackend(1366, 768)
        for i in range(25):
            target, click, _held = backend.update(
                face(), 640, 480, click_enabled=True, voice_active=False, now=i / 30)
            self.assertEqual(target.shape, (2,))
            self.assertIsNone(click)
        backend.update(face(smile=1.3), 640, 480,
                       click_enabled=True, voice_active=False, now=1.0)
        _, click, held = backend.update(face(smile=1.3), 640, 480,
                                        click_enabled=True, voice_active=False, now=2.3)
        self.assertEqual(click, "left")
        self.assertTrue(held)

    def test_mouth_open_right_click_and_voice_suppression(self):
        backend = ControlBackend(1366, 768)
        backend.update(face(mouth=0.12), 640, 480,
                       click_enabled=True, voice_active=False, now=1.0)
        _, click, _held = backend.update(face(mouth=0.12), 640, 480,
                                          click_enabled=True, voice_active=False, now=2.1)
        self.assertEqual(click, "right")
        _, click, held = backend.update(face(mouth=0.12), 640, 480,
                                        click_enabled=True, voice_active=True, now=3.3)
        self.assertIsNone(click)
        self.assertFalse(held)

    def test_smile_can_press_voice_bar_stop_button(self):
        backend = ControlBackend(1366, 768)
        for i in range(25):
            backend.update(face(), 640, 480,
                           click_enabled=True, voice_active=True, now=i / 30)
        backend.update(face(smile=1.3), 640, 480,
                       click_enabled=True, voice_active=True, now=1.0)
        _, click, _held = backend.update(face(smile=1.3), 640, 480,
                                          click_enabled=True, voice_active=True, now=2.3)
        self.assertEqual(click, "left")

    def test_two_eye_closures_toggle_voice_with_distinct_beeps(self):
        requests = queue.SimpleQueue()
        worker = SimpleNamespace(
            blink=SimpleNamespace(closed_since=1.0, deep=li.BLINK_DEEP),
            voice_closure=None,
            state=SimpleNamespace(voice_requests=requests),
        )
        with patch.object(li.desktop, "play_voice_sound") as beep:
            self.assertEqual(li.VOICE_EYES_CLOSED_S, 1.0)
            li.VisionWorker._voice_gesture(worker, True, SimpleNamespace(voice="", paused=False), 1.99)
            self.assertTrue(requests.empty())
            li.VisionWorker._voice_gesture(worker, True, SimpleNamespace(voice="", paused=False), 2.01)
            self.assertEqual(requests.get_nowait(), (True, "eyes closed"))
            li.VisionWorker._voice_gesture(worker, True, SimpleNamespace(voice="", paused=False), 2.2)
            self.assertTrue(requests.empty())
            worker.blink.closed_since = 4.0
            li.VisionWorker._voice_gesture(
                worker, True, SimpleNamespace(voice="listening", paused=False), 5.01)
            self.assertEqual(requests.get_nowait(), (False, "eyes closed"))
            self.assertEqual(beep.call_args_list, [call(True), call(False)])

    def test_paused_voice_gesture_does_nothing(self):
        requests = queue.SimpleQueue()
        worker = SimpleNamespace(
            blink=SimpleNamespace(closed_since=1.0, deep=li.BLINK_DEEP),
            voice_closure=None,
            state=SimpleNamespace(voice_requests=requests),
        )
        with patch.object(li.desktop, "play_voice_sound") as beep:
            li.VisionWorker._voice_gesture(
                worker, True, SimpleNamespace(voice="", paused=True), 3.0)
            self.assertTrue(requests.empty())
            beep.assert_not_called()

    def test_voice_session_has_no_silence_or_duration_timeout(self):
        class SilentSegmenter:
            quiet_s = 100000.0
            speaking = False
            level = 0.0
            t = 0.0

            def __init__(self, _vad):
                pass

            def feed(self, _block):
                return []

            def flush(self):
                return None

        state = SimpleNamespace(stop=threading.Event(), update=lambda **_fields: None)
        session = vt.VoiceTyping(state, recognizer=object(), sounds=False)

        class Mic:
            calls = 0

            def read(self, _seconds):
                self.calls += 1
                if self.calls == 4:
                    session.stop_session.set()
                return np.ones(1600, dtype=np.float32)

        times = iter(range(0, 100000, 1000))
        with patch.object(vt, "SpeechSegmenter", SilentSegmenter), patch.object(
                vt.time, "monotonic", side_effect=lambda: next(times)):
            self.assertEqual(session._listen(Mic(), object()), "stopped")

    def test_command_panel_only_accepts_listed_phrases(self):
        self.assertEqual(voice_commands.wispr_command_action(
            "Open the on-screen keyboard"), "open_keyboard")
        self.assertIsNone(voice_commands.wispr_command_action("delete every file"))


if __name__ == "__main__":
    unittest.main()
