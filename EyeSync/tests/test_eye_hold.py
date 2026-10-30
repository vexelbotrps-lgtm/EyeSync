import unittest
from control_backend.eye_hold import EyeHold

class EyeHoldTests(unittest.TestCase):
    def armed(self):
        h = EyeHold()
        h.update(.25, .25, (.1, .1), 0)
        h.update(.25, .25, (.1, .1), .26)
        return h

    def test_downward_gaze_scores_alone_cannot_start_hold(self):
        h = self.armed()
        self.assertFalse(h.update(.24, .24, (.95, .95), .3))
        self.assertIsNone(h.closed_since)

    def test_wink_cannot_start_hold(self):
        self.assertFalse(self.armed().update(.08, .25, (.95, .1), .3))

    def test_moderate_scores_can_register_a_real_closure(self):
        h = self.armed()
        for i in range(12):
            self.assertTrue(h.update(.13, .13, (.5, .48), .3 + i * .1))
        self.assertAlmostEqual(h.closed_since, .3)

    def test_squint_without_enough_geometry_change_is_rejected(self):
        self.assertFalse(self.armed().update(.22, .22, (.55, .55), .3))

    def test_continuous_closure_and_reopening(self):
        h = self.armed()
        for i in range(12):
            self.assertTrue(h.update(.08, .08, (.95, .95), .3 + i * .1))
        self.assertAlmostEqual(h.closed_since, .3)
        self.assertFalse(h.update(.25, .25, (.1, .1), 1.5))
        self.assertIsNone(h.closed_since)

    def test_tracking_gap_cancels_hold(self):
        h = self.armed()
        h.update(.08, .08, (.95, .95), .3)
        self.assertFalse(h.update(.08, .08, (.95, .95), 1.0))

    def test_different_eye_geometry_at_different_directions(self):
        # Simulated per-eye foreshortening: require an actual reduction from
        # open-eye geometry in every case, not just a high blink score.
        for left, right in ((.32, .30), (.14, .15), (.20, .30), (.30, .20)):
            with self.subTest(open_ears=(left, right)):
                h = EyeHold()
                h.update(left, right, (.1, .1), 0)
                h.update(left, right, (.1, .1), .26)
                self.assertFalse(h.update(left, right, (.95, .95), .3))
                self.assertTrue(h.update(left * .4, right * .4, (.95, .95), .4))

if __name__ == '__main__':
    unittest.main()
