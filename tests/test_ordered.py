"""Independent acceptance tests for ``mode = "ordered"`` isotonic inference.

The checks re-implement the reference behaviour independently (exhaustive
PAVA segmentation over every cut set, direct point-wise region membership and
exact boundary evaluation) rather than reusing the production region code.
Coverage follows the specification:

* the canonical ``y=[0,1,0]`` example (cuts, reduced ``p/q`` fit, full-line V);
* strict/weak constraint open/closed boundary inclusion;
* several equal adjacent platforms while every cut point is retained;
* cross-segmentation ties resolved by lexicographically smallest cut array;
* irrational region boundaries with exact brackets;
* the ``k = n`` degenerate segmentation with the whole real line as V;
* complete region acceptance (not just the component containing observations);
* input validation with the ``$.mode`` path and old-file preservation.
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
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_infer.engine import run_inference, serialize_result  # noqa: E402
from seg_infer.endpoints import (  # noqa: E402
    AlgEP,
    RationalEP,
    compare_endpoints,
)
from seg_infer.errors import InputError  # noqa: E402
from seg_infer.input_parsing import parse_input, parse_input_full  # noqa: E402
from seg_infer.ordered import (  # noqa: E402
    best_ordered_segmentation,
    fit_values,
    ordered_levels,
    pava_blocks,
)
from seg_infer.regions import INF  # noqa: E402

ENTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "run_inference.py",
)


def brute_ordered_winner(values, k):
    """Exhaustive ordered segmentation independent of the production solver."""
    n = len(values)
    best = None
    best_cost = None
    for cuts in combinations(range(1, n), k - 1):
        cuts = tuple(cuts)
        cost, _ = ordered_levels(values, cuts)
        if best_cost is None or cost < best_cost or (
            cost == best_cost and cuts < best
        ):
            best_cost = cost
            best = cuts
    return best


def region_contains(intervals, q):
    """Exact rational membership test using endpoint comparisons."""
    rep = RationalEP(q)
    for interval in intervals:
        if interval.lower is not INF:
            cmp_left = compare_endpoints(interval.lower, rep)
            if cmp_left > 0 or (cmp_left == 0 and interval.lower_open):
                continue
        if interval.upper is not INF:
            cmp_right = compare_endpoints(rep, interval.upper)
            if cmp_right > 0 or (cmp_right == 0 and interval.upper_open):
                continue
        return True
    return False


class CanonicalExampleTests(unittest.TestCase):
    def test_spec_example(self):
        doc = {
            "y": [0, 1, 0], "k": 2, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20", "mode": "ordered",
        }
        y, k, s2, mu0, alpha, mode = parse_input_full(doc)
        self.assertEqual(mode, "ordered")
        result = run_inference(y, k, s2, mu0, alpha, mode)
        self.assertEqual(result.cuts, (1,))
        self.assertEqual(result.fit, (Fraction(0), Fraction(1, 2), Fraction(1, 2)))
        # V is the entire real line: a single unbounded interval.
        self.assertEqual(len(result.intervals), 1)
        interval = result.intervals[0]
        self.assertIs(interval.lower, INF)
        self.assertIs(interval.upper, INF)

        payload = serialize_result(result)
        self.assertEqual(payload["S"], [1])
        self.assertEqual(payload["fit"], ["0/1", "1/2", "1/2"])
        self.assertEqual(payload["V"], [{"lower": None, "upper": None}])
        self.assertEqual(set(payload), {"S", "fit", "V", "p", "theta"})

    def test_omitted_mode_keeps_default_behaviour(self):
        doc = {
            "y": [0, 1, 0], "k": 2, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20",
        }
        y, k, s2, mu0, alpha = parse_input(doc)
        result = run_inference(y, k, s2, mu0, alpha)
        self.assertIsNone(result.fit)
        self.assertNotIn("fit", serialize_result(result))
        # Default V has two half-lines, unlike the ordered full line.
        self.assertEqual(len(result.intervals), 2)


class PAVAFitTests(unittest.TestCase):
    def test_equal_platforms_retain_all_cuts(self):
        y = [Fraction(3), Fraction(1), Fraction(0), Fraction(2)]
        result = run_inference(
            y, 3, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        self.assertEqual(result.cuts, (1, 3))
        # The first three points share one platform although two cuts survive.
        self.assertEqual(
            result.fit,
            (Fraction(4, 3), Fraction(4, 3), Fraction(4, 3), Fraction(2)),
        )
        self.assertTrue(
            all(result.fit[i] <= result.fit[i + 1] for i in range(len(y) - 1))
        )

    def test_fit_is_global_isotonic_minimum(self):
        rng = random.Random(31337)
        for _ in range(50):
            n = rng.randint(2, 8)
            k = rng.randint(2, n)
            y = [Fraction(rng.randint(-5, 5), rng.randint(1, 2)) for _ in range(n)]
            result = run_inference(
                y, k, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
            )
            sse, levels = ordered_levels(y, result.cuts)
            # Per-segment levels expand to the per-observation fit vector.
            expanded = fit_values(y, result.cuts)
            self.assertEqual(expanded, result.fit)
            self.assertEqual(len(levels), k)
            # No other feasible monotone level vector on these segments can
            # improve the SSE: check the residual orthogonality identity
            # SSE = ||y||^2 - ||m||^2 holds for the PAVA projection exactly.
            explained = sum(level * level for level in result.fit)
            square_total = sum(value * value for value in y)
            self.assertEqual(sse, square_total - explained)
            # Independent exhaustive winner agrees on the segmentation.
            self.assertEqual(brute_ordered_winner(y, k), result.cuts)

    def test_pava_violation_pooling_examples(self):
        # Strictly decreasing segment means must pool into one platform.
        sums = [Fraction(9), Fraction(4), Fraction(1)]
        lengths = [3, 2, 1]
        blocks = pava_blocks(sums, lengths)
        self.assertEqual(len(blocks), 1)
        block_sum, block_len, first, last = blocks[0]
        self.assertEqual((block_len, first, last), (6, 0, 2))
        # The block stores the raw observation sum; its level is the mean.
        self.assertEqual(block_sum, Fraction(14))
        self.assertEqual(block_sum / block_len, Fraction(7, 3))
        # Already non-decreasing means are left untouched.
        blocks = pava_blocks([Fraction(1), Fraction(4)], [1, 2])
        self.assertEqual(len(blocks), 2)


class TieAndBoundaryTests(unittest.TestCase):
    def test_cross_segmentation_tie_lexicographic(self):
        # Data where several ordered segmentations reach the same minimum SSE;
        # the lexicographically smallest internal cut array must be selected.
        y = [Fraction(v) for v in (-1, 3, 2, 3, 2)]
        result = run_inference(
            y, 4, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        costs = {
            cuts: ordered_levels(y, cuts)[0]
            for cuts in combinations(range(1, len(y)), 3)
        }
        winners = sorted(
            cuts for cuts, cost in costs.items() if cost == min(costs.values())
        )
        self.assertGreater(len(winners), 1)
        self.assertEqual(result.cuts, winners[0])
        self.assertEqual(result.cuts, brute_ordered_winner(y, 4))

    def test_open_boundary_excludes_lex_smaller_tie(self):
        # Discovered by randomized search: at u=-5 a lex-smaller candidate
        # ties with the optimum (1,5), so the boundary must be open.
        y = [Fraction(v) for v in (-3, 3, 0, -2, -3, 4, -4)]
        result = run_inference(
            y, 3, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        contrast = result.contrast
        u = Fraction(-5)
        values = [
            contrast.z[i] + contrast.eta[i] * u / contrast.eta_norm_sq
            for i in range(len(y))
        ]
        costs = {
            cuts: ordered_levels(values, cuts)[0]
            for cuts in combinations(range(1, len(y)), 2)
        }
        minimum = min(costs.values())
        winners = sorted(cuts for cuts, cost in costs.items() if cost == minimum)
        self.assertIn((1, 5), winners)
        self.assertLess(winners[0], (1, 5))
        # The exact point must be excluded because a lex-smaller candidate ties.
        self.assertFalse(region_contains(result.intervals, u))
        # The exhaustive winner at the point is that lexicographically smaller
        # candidate, never the optimum segmentation.
        self.assertNotEqual(brute_ordered_winner(values, 3), result.cuts)

    def test_closed_boundary_keeps_lex_smallest_optimum(self):
        y = [Fraction(v) for v in (-2, 1, 4)]
        result = run_inference(
            y, 2, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        contrast = result.contrast
        u = Fraction(-9, 2)
        values = [
            contrast.z[i] + contrast.eta[i] * u / contrast.eta_norm_sq
            for i in range(3)
        ]
        costs = {
            cuts: ordered_levels(values, cuts)[0]
            for cuts in combinations(range(1, 3), 1)
        }
        winners = sorted(
            cuts for cuts, cost in costs.items() if cost == min(costs.values())
        )
        self.assertEqual(winners[0], result.cuts)
        self.assertTrue(region_contains(result.intervals, u))

    def test_irrational_boundaries_exact(self):
        rng = random.Random(909)
        seen = 0
        for _ in range(300):
            n = rng.randint(3, 8)
            k = rng.randint(2, n - 1)
            y = [
                Fraction(rng.randint(-5, 5), rng.randint(1, 2))
                for _ in range(n)
            ]
            result = run_inference(
                y, k, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
            )
            algs = [
                ep
                for interval in result.intervals
                for ep in (interval.lower, interval.upper)
                if isinstance(ep, AlgEP)
            ]
            for endpoint in algs:
                a, b, c = endpoint.poly
                discriminant = b * b - 4 * a * c
                numeric_roots = sorted(
                    [
                        (-b + math.sqrt(discriminant)) / (2 * a),
                        (-b - math.sqrt(discriminant)) / (2 * a),
                    ]
                )
                self.assertLess(
                    abs(endpoint.to_float() - numeric_roots[endpoint.index]),
                    1e-9,
                )
                # The isolating bracket genuinely straddles the unique root.
                self.assertLess(
                    (a * endpoint.lo ** 2 + b * endpoint.lo + c)
                    * (a * endpoint.hi ** 2 + b * endpoint.hi + c),
                    0,
                )
                seen += 1
            if seen >= 4:
                break
        self.assertGreaterEqual(seen, 4)


class RegionAcceptanceTests(unittest.TestCase):
    def test_full_region_matches_brute_force_on_dense_probes(self):
        rng = random.Random(20261114)
        for _ in range(25):
            n = rng.randint(3, 8)
            k = rng.randint(2, n)
            y = [
                Fraction(rng.randint(-4, 4), rng.randint(1, 2))
                for _ in range(n)
            ]
            result = run_inference(
                y, k, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
            )
            contrast = result.contrast
            norm = contrast.eta_norm_sq
            probes = [
                Fraction(rng.randint(-400, 400), 50) for _ in range(30)
            ]
            probes.append(contrast.t)
            for u in probes:
                values = [
                    contrast.z[i] + contrast.eta[i] * u / norm
                    for i in range(n)
                ]
                expected = brute_ordered_winner(values, k) == result.cuts
                self.assertEqual(region_contains(result.intervals, u), expected)

    def test_complete_region_not_only_observed_component(self):
        # The canonical example's region must cover the whole real line, and a
        # generic case must accept probes far outside the observed t value.
        y = [Fraction(v) for v in (0, 1, 0)]
        result = run_inference(
            y, 2, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        for u in (Fraction(-1000), Fraction(1000), contrast_t(result)):
            self.assertTrue(region_contains(result.intervals, u))

    def test_kn_degenerate_region_is_full_line(self):
        y = [Fraction(1), Fraction(3), Fraction(-2), Fraction(4)]
        result = run_inference(
            y, 4, Fraction(2), Fraction(1), Fraction(1, 10), "ordered"
        )
        self.assertEqual(result.cuts, (1, 2, 3))
        self.assertEqual(len(result.intervals), 1)
        interval = result.intervals[0]
        self.assertIs(interval.lower, INF)
        self.assertIs(interval.upper, INF)
        self.assertTrue(
            all(result.fit[i] <= result.fit[i + 1] for i in range(3))
        )

    def test_contrast_built_from_original_y_and_segment_lengths(self):
        y = [Fraction(3), Fraction(1), Fraction(0), Fraction(2)]
        result = run_inference(
            y, 3, Fraction(1), Fraction(0), Fraction(1, 20), "ordered"
        )
        contrast = result.contrast
        # Even though the first three fitted values are equal, eta uses the
        # lengths of the first two selected segments rather than merged blocks.
        self.assertEqual(contrast.eta, (Fraction(1), Fraction(-1, 2),
                                        Fraction(-1, 2), Fraction(0)))
        norm = sum(e * e for e in contrast.eta)
        self.assertEqual(contrast.t, sum(contrast.eta[i] * y[i] for i in range(4)))
        self.assertEqual(contrast.v, norm)
        self.assertEqual(
            contrast.z,
            tuple(
                y[i] - contrast.eta[i] * contrast.t / norm
                for i in range(4)
            ),
        )


def contrast_t(result):
    return result.contrast.t


class InputValidationTests(unittest.TestCase):
    def test_illegal_modes_rejected_at_mode_path(self):
        base = {
            "y": [0, 1], "k": 2, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20",
        }
        for bad in ("default", "foo", "", 1, True, None, ["ordered"], {}):
            with self.assertRaises(InputError) as ctx:
                parse_input_full({**base, "mode": bad})
            self.assertEqual(ctx.exception.path, "$.mode")

    def test_other_field_paths_unchanged(self):
        base = {
            "y": [0, 1], "k": 2, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20", "mode": "ordered",
        }
        with self.assertRaises(InputError) as ctx:
            parse_input_full({**base, "k": True})
        self.assertEqual(ctx.exception.path, "$.k")
        with self.assertRaises(InputError) as ctx:
            parse_input_full({**base, "y": [True, 1]})
        self.assertEqual(ctx.exception.path, "$.y[0]")
        with self.assertRaises(InputError) as ctx:
            parse_input_full({**base, "extra": 1})
        self.assertEqual(ctx.exception.path, "$.extra")

    def test_cli_failure_preserves_old_output(self):
        good = {
            "y": [0, 1, 0], "k": 2, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20", "mode": "ordered",
        }
        bad = {
            "y": [0, 1], "k": 9, "s2": "1/1",
            "mu0": "0/1", "alpha": "1/20", "mode": "ordered",
        }
        with tempfile.TemporaryDirectory() as tmp:
            good_path = os.path.join(tmp, "good.json")
            bad_path = os.path.join(tmp, "bad.json")
            out_path = os.path.join(tmp, "out.json")
            with open(good_path, "w") as handle:
                json.dump(good, handle)
            with open(bad_path, "w") as handle:
                json.dump(bad, handle)
            subprocess.run(
                [sys.executable, "-I", ENTRY, good_path, out_path], check=True
            )
            before = open(out_path).read()
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, bad_path, out_path],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(open(out_path).read(), before)

    def test_cli_ordered_roundtrip(self):
        doc = {
            "y": ["1/2", "-1/2", "0/1"], "k": 2, "s2": "2/3",
            "mu0": "-1/4", "alpha": "1/20", "mode": "ordered",
        }
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.json")
            out_path = os.path.join(tmp, "out.json")
            with open(in_path, "w") as handle:
                json.dump(doc, handle)
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, in_path, out_path],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(open(out_path).read())
            self.assertEqual(set(payload), {"S", "fit", "V", "p", "theta"})
            y, k, s2, mu0, alpha, mode = parse_input_full(doc)
            direct = serialize_result(
                run_inference(y, k, s2, mu0, alpha, mode)
            )
            self.assertEqual(payload, direct)


if __name__ == "__main__":
    unittest.main(verbosity=2)
