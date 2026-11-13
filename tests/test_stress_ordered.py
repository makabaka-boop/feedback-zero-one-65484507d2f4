"""Fast randomized cross-check of the ordered region against brute-force PAVA.

Every probe is tested with exact rational membership (no floating-point
comparison), and every rational kink boundary is probed directly so that
open/closed inclusion is exercised as well.
"""

import os
import random
import sys
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_infer.engine import run_inference  # noqa: E402
from seg_infer.endpoints import RationalEP, compare_endpoints  # noqa: E402
from seg_infer.ordered import ordered_levels  # noqa: E402
from seg_infer.ordered_region import collect_kinks  # noqa: E402
from seg_infer.regions import INF  # noqa: E402


def brute_winner(values, k):
    best = None
    best_cost = None
    for cuts in combinations(range(1, len(values)), k - 1):
        cuts = tuple(cuts)
        cost = ordered_levels(values, cuts)[0]
        if best_cost is None or cost < best_cost or (
            cost == best_cost and cuts < best
        ):
            best_cost = cost
            best = cuts
    return best


def contains(intervals, q):
    rep = RationalEP(q)
    for interval in intervals:
        if interval.lower is not INF:
            left = compare_endpoints(interval.lower, rep)
            if left > 0 or (left == 0 and interval.lower_open):
                continue
        if interval.upper is not INF:
            right = compare_endpoints(rep, interval.upper)
            if right > 0 or (right == 0 and interval.upper_open):
                continue
        return True
    return False


def main():
    rng = random.Random(20261115)
    checked = 0
    mismatches = 0
    for trial in range(70):
        n = rng.randint(2, 8)
        k = rng.randint(2, n)
        y = [Fraction(rng.randint(-5, 5), rng.randint(1, 3)) for _ in range(n)]
        result = run_inference(
            y, k, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        contrast = result.contrast
        norm = contrast.eta_norm_sq
        candidates = [tuple(c) for c in combinations(range(1, n), k - 1)]
        kinks = collect_kinks(candidates, contrast.z, contrast.eta, norm)
        probes = [contrast.t] + [
            Fraction(rng.randint(-400, 400), 40) for _ in range(20)
        ] + kinks
        for u in probes:
            values = [
                contrast.z[i] + contrast.eta[i] * u / norm for i in range(n)
            ]
            expected = brute_winner(values, k) == result.cuts
            got = contains(result.intervals, u)
            checked += 1
            if expected != got:
                mismatches += 1
                print("MISMATCH", trial, u, expected, got, y, k, result.cuts)
    print(f"ordered probes checked: {checked}, mismatches: {mismatches}")
    assert mismatches == 0
    print("ordered fast stress check passed")


if __name__ == "__main__":
    main()
