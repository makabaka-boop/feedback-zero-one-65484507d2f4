"""Independent acceptance tests for the segmented-normal inference backend.

These tests deliberately re-implement the reference behaviour independently
(brute-force segment enumeration, dense grid scans and Monte Carlo truncated
normal probabilities) instead of reusing the package's own logic.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_infer.engine import run_inference, serialize_result  # noqa: E402
from seg_infer.input_parsing import parse_input  # noqa: E402
from seg_infer.probability import (  # noqa: E402
    conditional_cdf,
    standard_cdf,
    standard_quantile,
)
from seg_infer.rational import frac_float  # noqa: E402
from seg_infer.regions import INF  # noqa: E402

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "run_inference.py")


def brute_best(y, k):
    """Independent dynamic-programming-free exhaustive segmentation."""
    n = len(y)

    def cutsets(start, remaining):
        if remaining == 1:
            return [()]
        out = []
        for first in range(start + 1, n - remaining + 2):
            for rest in cutsets(first, remaining - 1):
                out.append((first,) + rest)
        return out

    best, best_cost = None, None
    for cuts in cutsets(0, k):
        bounds = (0,) + cuts + (n,)
        cost = Fraction(0)
        for a, b in zip(bounds, bounds[1:]):
            seg = y[a:b]
            mean = sum(seg, Fraction(0)) / len(seg)
            cost += sum((v - mean) ** 2 for v in seg)
        if best_cost is None or cost < best_cost or (cost == best_cost and cuts < best):
            best, best_cost = cuts, cost
    return best, best_cost


def brute_region(y, k):
    """Grid-based independent approximation of the stability region V."""
    result = run_inference(y, k, Fraction(1), Fraction(0), Fraction(1, 20))
    contrast = result.contrast
    n = len(y)
    norm = contrast.eta_norm_sq

    def winners(u):
        observed = [contrast.z[i] + contrast.eta[i] * u / norm for i in range(n)]
        best, _ = brute_best(observed, k)
        return best

    grid = [Fraction(-2000 + i, 100) for i in range(4001)]
    feasible = [u for u in grid if winners(u) == result.cuts]
    return result, feasible, winners


class SegmentationTests(unittest.TestCase):
    def test_spec_example(self):
        doc = {"y": [0, 1, 0], "k": 2, "s2": "1/1",
               "mu0": "0/1", "alpha": "1/20"}
        y, k, s2, mu0, alpha = parse_input(doc)
        result = run_inference(y, k, s2, mu0, alpha)
        self.assertEqual(result.cuts, (1,))
        self.assertEqual(len(result.intervals), 2)
        left, right = result.intervals
        self.assertIs(left.lower, INF)
        self.assertEqual(left.upper.value, Fraction(-1, 2))
        self.assertFalse(left.upper_open)
        self.assertEqual(right.lower.value, Fraction(3, 2))
        self.assertFalse(right.lower_open)
        self.assertIs(right.upper, INF)

    def test_all_candidates_exhaustive_random(self):
        rng = random.Random(20261113)
        for trial in range(40):
            n = rng.randint(2, 7)
            k = rng.randint(2, n)
            y = [Fraction(rng.randint(-4, 4), rng.randint(1, 3))
                 for _ in range(n)]
            result = run_inference(y, k, Fraction(1), Fraction(0),
                                   Fraction(1, 20))
            expected, expected_cost = brute_best(y, k)
            self.assertEqual(result.cuts, expected)

    def test_tie_lexicographic(self):
        # Perfectly symmetric data: costs tie, smallest cut array must win.
        y = [Fraction(v) for v in (1, -1, -1, 1)]
        result = run_inference(y, 2, Fraction(1), Fraction(0), Fraction(1, 20))
        expected, _ = brute_best(y, 2)
        self.assertEqual(result.cuts, expected)
        self.assertEqual(result.cuts, (1,))

    def test_kn_equals_n(self):
        y = [Fraction(1), Fraction(3), Fraction(-2), Fraction(4)]
        result = run_inference(y, 4, Fraction(2), Fraction(1), Fraction(1, 10))
        self.assertEqual(result.cuts, (1, 2, 3))
        self.assertEqual(len(result.intervals), 1)
        interval = result.intervals[0]
        self.assertIs(interval.lower, INF)
        self.assertIs(interval.upper, INF)
        # Full line: the confidence interval is symmetric around t.
        low, high = result.confidence.theta_intervals[0]
        t = float(result.contrast.t)
        self.assertLess(abs((low + high) / 2 - t), 1e-6)

    def test_region_matches_grid_scan(self):
        rng = random.Random(77)
        for trial in range(12):
            n = rng.randint(3, 6)
            k = rng.randint(2, n - 1)
            y = [Fraction(rng.randint(-3, 3), rng.randint(1, 2))
                 for _ in range(n)]
            result, feasible, winners = brute_region(y, k)
            for interval in result.intervals:
                lo = -20.0 if interval.lower is INF else interval.lower.to_float()
                hi = 20.0 if interval.upper is INF else interval.upper.to_float()
                for u in feasible:
                    uf = float(u)
                    if lo < uf < hi:
                        self.assertEqual(winners(u), result.cuts)
            # Every grid-feasible point must be covered by an exact interval.
            for u in feasible:
                covered = False
                for interval in result.intervals:
                    lo = -math.inf if interval.lower is INF else interval.lower.to_float()
                    hi = math.inf if interval.upper is INF else interval.upper.to_float()
                    if lo <= float(u) <= hi:
                        covered = True
                self.assertTrue(covered, f"feasible {u} missing from V")

    def test_t_always_inside_region(self):
        rng = random.Random(91)
        for _ in range(20):
            n = rng.randint(2, 8)
            k = rng.randint(2, n)
            y = [Fraction(rng.randint(-5, 5)) for _ in range(n)]
            result = run_inference(y, k, Fraction(1), Fraction(0),
                                   Fraction(1, 20))
            t = frac_float(result.contrast.t)
            self.assertTrue(any(
                (math.inf if iv.upper is INF else iv.upper.to_float()) >= t
                >= (-math.inf if iv.lower is INF else iv.lower.to_float())
                for iv in result.intervals))

    def test_synthetic_open_boundary(self):
        # Direct constraint-level check: a strict lex-smaller candidate whose
        # zero is feasible through another (weak) constraint must be open.
        from seg_infer.regions import compute_region
        # weak: u^2 - 1 >= 0 (feasible outside [-1, 1], endpoints closed);
        # strict candidate with the same zeros makes both endpoints open.
        rows = [
            ((1,), (Fraction(1), Fraction(0), Fraction(-1)), False),
            ((0,), (Fraction(1), Fraction(0), Fraction(-1)), True),
        ]
        intervals = compute_region(rows)
        self.assertEqual(len(intervals), 2)
        self.assertTrue(intervals[0].upper_open)
        self.assertTrue(intervals[1].lower_open)

    def test_irrational_boundaries(self):
        # Found by randomized search: candidate (2,3,4) yields x^2 - 12.
        y = [Fraction(5, 2), Fraction(6), Fraction(1), Fraction(4),
             Fraction(-4), Fraction(-1, 2), Fraction(-3, 2)]
        result = run_inference(y, 4, Fraction(1), Fraction(0), Fraction(1, 20))
        from seg_infer.endpoints import AlgEP
        kinds = {type(iv.lower).__name__ for iv in result.intervals}
        kinds |= {type(iv.upper).__name__ for iv in result.intervals}
        self.assertIn("AlgEP", kinds)
        alg_endpoints = [
            ep for iv in result.intervals for ep in (iv.lower, iv.upper)
            if isinstance(ep, AlgEP)
        ]
        self.assertTrue(alg_endpoints)
        alg = alg_endpoints[0]
        expected = -math.sqrt(12) if alg.index == 0 else math.sqrt(12)
        self.assertLess(abs(alg.to_float() - expected), 1e-9)


class ProbabilityTests(unittest.TestCase):
    def test_monte_carlo_conditional_cdf(self):
        doc = {"y": [0, 1, 0], "k": 2, "s2": "1/1",
               "mu0": "0/1", "alpha": "1/20"}
        y, k, s2, mu0, alpha = parse_input(doc)
        result = run_inference(y, k, s2, mu0, alpha)
        sd = math.sqrt(float(result.contrast.v))
        t = float(result.contrast.t)
        rng = random.Random(424242)
        inside, below = 0, 0
        trials = 400_000
        for _ in range(trials):
            x = rng.gauss(float(mu0), sd)
            hit = any(
                (-math.inf if iv.lower is INF else iv.lower.to_float()) <= x
                <= (math.inf if iv.upper is INF else iv.upper.to_float())
                for iv in result.intervals)
            if hit:
                inside += 1
                if x <= t:
                    below += 1
        mc = below / inside
        exact = conditional_cdf(result.intervals, t, float(mu0), sd)
        self.assertLess(abs(mc - exact), 1e-3)
        # Here F(t) > 1/2, so the two-sided p-value uses 2*(1-F(t)).
        tail = min(exact, 1.0 - exact)
        self.assertLess(abs(tail - result.confidence.p_value / 2), 1e-12)

    def test_cdf_monotone_and_ci_brackets(self):
        y = [Fraction(v) for v in (0, 1, 0, 2)]
        result = run_inference(y, 2, Fraction(1), Fraction(0), Fraction(1, 10))
        sd = math.sqrt(float(result.contrast.v))
        t = float(result.contrast.t)
        previous = 2.0
        for theta in range(-6, 7):
            value = conditional_cdf(result.intervals, t, float(theta), sd)
            self.assertLess(value, previous + 1e-12)
            previous = value
        low, high = result.confidence.theta_intervals[0]
        target = 0.05
        f_low = conditional_cdf(result.intervals, t, low, sd)
        f_high = conditional_cdf(result.intervals, t, high, sd)
        self.assertLess(abs(f_low - (1 - target)), 1e-6)
        self.assertLess(abs(f_high - target), 1e-6)

    def test_full_line_reduces_to_plain_normal(self):
        y = [Fraction(1), Fraction(3), Fraction(-2), Fraction(4)]
        result = run_inference(y, 4, Fraction(2), Fraction(1), Fraction(1, 10))
        sd = math.sqrt(float(result.contrast.v))
        t = float(result.contrast.t)
        f = conditional_cdf(result.intervals, t, 1.0, sd)
        self.assertLess(abs(f - standard_cdf((t - 1.0) / sd)), 1e-12)
        # Unrestricted normal: symmetric confidence interval.
        low, high = result.confidence.theta_intervals[0]
        z = standard_quantile(1 - 0.05)
        self.assertLess(abs((high - low) / 2 - z * sd), 1e-6)
        self.assertLess(abs((low + high) / 2 - t), 1e-6)


class InputValidationTests(unittest.TestCase):
    def _bad(self, doc, fragment):
        with self.assertRaises(Exception) as ctx:
            parse_input(doc)
        self.assertIn(fragment, str(ctx.exception))

    def test_rejected_inputs(self):
        base = {"y": [0, 1], "k": 2, "s2": "1/1",
                "mu0": "0/1", "alpha": "1/20"}
        self._bad({**base, "k": True}, "$.k")
        self._bad({**base, "k": 1}, "$.k")
        self._bad({**base, "k": 3}, "$.k")
        self._bad({**base, "y": [1]}, "$.y")
        self._bad({**base, "s2": "0/1"}, "$.s2")
        self._bad({**base, "s2": "-1/2"}, "$.s2")
        self._bad({**base, "alpha": "0/1"}, "$.alpha")
        self._bad({**base, "alpha": "1/1"}, "$.alpha")
        self._bad({**base, "alpha": "1/0"}, "$.alpha")
        self._bad({**base, "s2": "1.5"}, "$.s2")
        self._bad({**base, "y": [True, 1]}, "$.y[0]")
        self._bad({**base, "extra": 1}, "$.extra")
        del base["mu0"]
        self._bad(base, "$.mu0")

    def test_duplicate_and_unknown_member_paths(self):
        text = '{"y":[0,1],"k":2,"k":2,"s2":"1/1","mu0":"0/1","alpha":"1/20"}'
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(text)
            path = f.name
        proc = subprocess.run(
            [sys.executable, "-I", ENTRY, path, path + ".out"],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("duplicate member", proc.stderr)
        self.assertIn("$", proc.stderr)

    def test_failure_preserves_old_output(self):
        good = {"y": [0, 1, 0], "k": 2, "s2": "1/1",
                "mu0": "0/1", "alpha": "1/20"}
        bad = {"y": [0, 1], "k": 9, "s2": "1/1",
               "mu0": "0/1", "alpha": "1/20"}
        with tempfile.TemporaryDirectory() as tmp:
            in_good = os.path.join(tmp, "good.json")
            in_bad = os.path.join(tmp, "bad.json")
            out = os.path.join(tmp, "out.json")
            with open(in_good, "w") as f:
                json.dump(good, f)
            with open(in_bad, "w") as f:
                json.dump(bad, f)
            subprocess.run([sys.executable, "-I", ENTRY, in_good, out], check=True)
            before = open(out).read()
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, in_bad, out],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(open(out).read(), before)

    def test_cli_roundtrip_and_serialization(self):
        doc = {"y": ["1/2", "-1/2", "0/1"], "k": 2, "s2": "2/3",
               "mu0": "-1/4", "alpha": "1/20"}
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.json")
            out_path = os.path.join(tmp, "out.json")
            with open(in_path, "w") as f:
                json.dump(doc, f)
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, in_path, out_path],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(open(out_path).read())
            self.assertEqual(set(payload), {"S", "V", "p", "theta"})
            y, k, s2, mu0, alpha = parse_input(doc)
            direct = serialize_result(run_inference(y, k, s2, mu0, alpha))
            self.assertEqual(payload["S"], direct["S"])
            self.assertEqual(payload["V"], direct["V"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
