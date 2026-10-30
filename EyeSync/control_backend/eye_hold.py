"""Conservative, bilateral eye hold for the standalone voice switch."""
class EyeHold:
    def __init__(self):
        self.reset()

    def reset(self):
        self.closed_since = None
        self.open_since = None
        self.last_frame = None
        self.armed = False
        self.deep = 0.0
        self.baseline = None
        self.score_base = (0.0, 0.0)

    def update(self, left, right, scores, now):
        # Never carry a hold across missing camera frames.
        if self.last_frame is not None and now - self.last_frame > 0.5:
            self.reset()
        self.last_frame = now
        if scores is None:
            self.closed_since = self.open_since = None
            self.armed = False
            return False
        # Learn each eye separately from confident open-eye frames. This
        # accommodates narrower eyes and foreshortening when looking sideways
        # or vertically; never learn from a suspected closure.
        base = self.baseline or (.28, .28)
        ratios = (left / max(base[0], .07), right / max(base[1], .07))
        if (max(scores) < 0.45 and min(left, right) > 0.07
                and (self.baseline is None or min(ratios) > 0.85)):
            if self.baseline is None:
                self.baseline = (left, right)
            else:
                self.baseline = tuple(.8 * b + .2 * e
                                      for b, e in zip(self.baseline, (left, right)))
            self.score_base = tuple(.8 * b + .2 * s
                                    for b, s in zip(self.score_base, scores))
        base = self.baseline or (.28, .28)
        ratios = (left / max(base[0], .07), right / max(base[1], .07))
        # Require independent bilateral evidence, continuously for the hold.
        delta = min(s - b for s, b in zip(scores, self.score_base))
        holding = self.closed_since is not None
        shut = (max(ratios) < (0.82 if holding else 0.75)
                and delta >= (0.15 if holding else 0.25))
        opened = min(ratios) > 0.75 and max(scores) < 0.45
        if opened:
            self.closed_since = None
            self.open_since = now if self.open_since is None else self.open_since
            if now - self.open_since >= 0.25:
                self.armed = True
        else:
            self.open_since = None
        if not shut:
            self.closed_since = None
            self.deep = 0.0
            return False
        if not self.armed:
            return False
        self.closed_since = now if self.closed_since is None else self.closed_since
        self.deep = 2.0
        return True
