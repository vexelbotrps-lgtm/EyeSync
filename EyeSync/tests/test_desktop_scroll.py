import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import live_inference as li

class DesktopScrollTests(unittest.TestCase):
    def worker(self):
        return SimpleNamespace(scroll=li.TiltScroll(), args=SimpleNamespace(always_active=True),
            state=Mock(), control_backend=Mock(), blink=Mock(), voice_closure=None)

    def run_roll(self, w, roll, start, seconds, paused=False):
        snap = SimpleNamespace(gestures=True, paused=paused, remote_off=False)
        with patch.object(li.fe, 'head_roll', return_value=roll):
            for i in range(round(seconds * 30)):
                li.VisionWorker._desktop_scroll(w, None, 640, 480, snap, start+i/30)

    def test_both_directions_and_straighten(self):
        for roll, sign in ((35, 1), (-35, -1)):
            w = self.worker()
            self.run_roll(w, 0, 0, 1)
            self.run_roll(w, roll, 1, 1)
            turns = [c.args[0] for c in w.state.add_scroll.call_args_list]
            self.assertTrue(turns)
            self.assertTrue(all(t * sign > 0 for t in turns))
            w.control_backend.reset.assert_called()
            self.run_roll(w, 0, 2, .5)
            self.assertFalse(w.tilt_held)

    def test_pause_cancels_scroll_and_requires_a_new_hold(self):
        w = self.worker()
        self.run_roll(w, 0, 0, 1)
        self.run_roll(w, 35, 1, 1)
        w.state.add_scroll.reset_mock()
        self.run_roll(w, 35, 2, 1, paused=True)
        w.state.add_scroll.assert_not_called()
        self.assertFalse(w.tilt_held)
