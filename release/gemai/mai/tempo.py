import numpy as np
from bisect import bisect_right
class NpTempo:
    """vectorized beat->sec for a tempo map (beats, bpms) ; sec relative to beat 0"""
    def __init__(self, beats, bpms):
        self.b = np.asarray(beats, dtype=np.float64)
        self.p = np.asarray(bpms, dtype=np.float64)
        secs = [0.0]
        for i in range(1, len(self.b)):
            secs.append(secs[-1] + (self.b[i] - self.b[i - 1]) * 60.0 / self.p[i - 1])
        self.s = np.asarray(secs)
        # plain-float copies for the scalar fast path (same IEEE operations as to_sec, no numpy call overhead)
        self._bl, self._pl, self._sl = self.b.tolist(), self.p.tolist(), self.s.tolist()
    def to_sec(self, beat):
        beat = np.asarray(beat, dtype=np.float64)
        i = np.clip(np.searchsorted(self.b, beat, side="right") - 1, 0, len(self.b) - 1)
        return self.s[i] + (beat - self.b[i]) * 60.0 / self.p[i]
    def sec_scalar(self, beat):
        """to_sec of one float: bit-identical to ``float(self.to_sec(beat))``."""
        i = bisect_right(self._bl, beat) - 1
        if i < 0:
            i = 0
        return self._sl[i] + (beat - self._bl[i]) * 60.0 / self._pl[i]
    def to_beat(self, sec):
        sec = np.asarray(sec, dtype=np.float64)
        i = np.clip(np.searchsorted(self.s, sec, side="right") - 1, 0, len(self.s) - 1)
        return self.b[i] + (sec - self.s[i]) * self.p[i] / 60.0
    def bpm_at(self, beat):
        i = int(np.clip(np.searchsorted(self.b, beat, side="right") - 1, 0, len(self.b) - 1))
        return float(self.p[i])
