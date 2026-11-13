"""Fast randomized cross-check of V against the brute-force segmentation."""

import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_infer.engine import run_inference  # noqa: E402
from test_acceptance import brute_best  # noqa: E402


def in_region(intervals, u):
    for iv in intervals:
        lo = -math.inf if iv.lower.__class__.__name__ == "_Infinity" \
            else iv.lower.to_float()
        hi = math.inf if iv.upper.__class__.__name__ == "_Infinity" \
            else iv.upper.to_float()
        if lo < u < hi:
            return True
        if u == lo and not iv.lower_open:
            return True
        if u == hi and not iv.upper_open:
            return True
    return False


def main():
    rng = random.Random(2026)
    mismatches = 0
    checked = 0
    for trial in range(60):
        n = rng.randint(2, 6)
        k = rng.randint(2, n)
        y = [Fraction(rng.randint(-4, 4), rng.randint(1, 2)) for _ in range(n)]
        result = run_inference(
            y, k, Fraction(1), Fraction(0), Fraction(1, 20))
        contrast = result.contrast
        norm = contrast.eta_norm_sq
        probes = [Fraction(rng.randint(-150, 150), 50) for _ in range(24)]
        probes.append(contrast.t)
        for u in probes:
            observed = [
                contrast.z[i] + contrast.eta[i] * u / norm
                for i in range(n)
            ]
            winner, _ = brute_best(observed, k)
            expected = winner == result.cuts
            got = in_region(result.intervals, float(u))
            checked += 1
            if expected != got:
                mismatches += 1
                print("MISMATCH", trial, float(u), expected, got, y, k)
    print(f"probes checked: {checked}, mismatches: {mismatches}")
    assert mismatches == 0
    print("fast stress check passed")


if __name__ == "__main__":
    main()
