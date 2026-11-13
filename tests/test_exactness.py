"""Standalone checks for exact boundary handling and inversion accuracy."""

import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_infer.engine import run_inference  # noqa: E402
from seg_infer.endpoints import AlgEP, RationalEP  # noqa: E402
from seg_infer.probability import conditional_cdf  # noqa: E402
from seg_infer.rational import frac_float  # noqa: E402
from seg_infer.segmentation import best_segmentation  # noqa: E402


def main():
    rng = random.Random(990011)
    boundary_checks = 0
    residuals = []
    for _ in range(30):
        n = rng.randint(2, 7)
        k = rng.randint(2, n)
        y = [Fraction(rng.randint(-6, 6), rng.randint(1, 3)) for _ in range(n)]
        alpha = Fraction(rng.randint(1, 19), 20)
        result = run_inference(
            y, k, Fraction(rng.randint(1, 3)),
            Fraction(rng.randint(-2, 2)), alpha,
        )
        contrast = result.contrast
        norm = contrast.eta_norm_sq

        for interval in result.intervals:
            for endpoint, is_open in (
                (interval.lower, interval.lower_open),
                (interval.upper, interval.upper_open),
            ):
                if isinstance(endpoint, RationalEP):
                    u = endpoint.value
                    observed = [
                        contrast.z[i] + contrast.eta[i] * u / norm
                        for i in range(n)
                    ]
                    winner, _ = best_segmentation(observed, k)
                    boundary_checks += 1
                    # A closed endpoint must keep the optimum under tie rules.
                    if not is_open:
                        assert winner == result.cuts or winner < result.cuts
                elif isinstance(endpoint, AlgEP):
                    lo, hi = endpoint.lo, endpoint.hi
                    a, b, c = endpoint.poly
                    # The isolating bracket genuinely straddles one root.
                    assert (a * lo * lo + b * lo + c) * (
                        a * hi * hi + b * hi + c) < 0
                    boundary_checks += 1

        if result.confidence.theta_intervals:
            low, high = result.confidence.theta_intervals[0]
            sd = math.sqrt(frac_float(contrast.v))
            t = frac_float(contrast.t)
            target = frac_float(alpha / 2)
            residuals.append(abs(
                conditional_cdf(result.intervals, t, low, sd) - (1 - target)))
            residuals.append(abs(
                conditional_cdf(result.intervals, t, high, sd) - target))

    worst = max(residuals)
    print(f"boundary checks: {boundary_checks}")
    print(f"worst inversion residual: {worst:.3e}")
    assert boundary_checks > 0 and worst < 1e-9
    print("exactness checks passed")


if __name__ == "__main__":
    main()
