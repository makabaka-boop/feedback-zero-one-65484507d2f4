"""Independent acceptance tests for ``mode=trend`` (L1 trend filtering).

The reference behaviour here is re-derived from scratch rather than reusing
the package's solver: every candidate KKT state of the box-constrained dual
(each interior center is free, pinned at +1 or pinned at -1) is enumerated,
its linear stationarity system is solved independently, and feasibility of
the full KKT conditions is checked.  Stability regions are checked by
refitting from this independent solver on dense probes, including exact
endpoint neighborhoods, so open versus closed boundaries are exercised.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_infer.engine import (  # noqa: E402
    run_job,
    serialize_job_result,
    serialize_trend_result,
)
from seg_infer.errors import InputError  # noqa: E402
from seg_infer.input_parsing import parse_job  # noqa: E402
from seg_infer.trend import fit_trend, run_trend_inference  # noqa: E402

ENTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "run_inference.py",
)


# ---------------------------------------------------------------------------
# Independent reference implementation
# ---------------------------------------------------------------------------

def _d_matrix(n):
    m = n - 2
    matrix = [[Fraction(0)] * n for _ in range(m)]
    for i in range(m):
        matrix[i][i] = Fraction(1)
        matrix[i][i + 1] = Fraction(-2)
        matrix[i][i + 2] = Fraction(1)
    return matrix


def _gauss(matrix, rhs):
    m = len(matrix)
    tableau = [list(matrix[i]) + [rhs[i]] for i in range(m)]
    for col in range(m):
        pivot = next(r for r in range(col, m) if tableau[r][col] != 0)
        tableau[col], tableau[pivot] = tableau[pivot], tableau[col]
        for row in range(m):
            if row == col:
                continue
            factor = tableau[row][col] / tableau[col][col]
            for k in range(col, m + 1):
                tableau[row][k] -= factor * tableau[col][k]
    return [tableau[i][m] / tableau[i][i] for i in range(m)]


def reference_fit(y, penalty):
    """Brute-force KKT enumeration; returns ``(fit_tuple, knot_tuple)``."""
    n = len(y)
    m = n - 2
    d = _d_matrix(n)
    gram = [
        [sum(d[k][i] * d[l][i] for i in range(n)) for l in range(m)]
        for k in range(m)
    ]
    dy = [sum(d[c][i] * y[i] for i in range(n)) for c in range(m)]

    solutions = []
    for state in product((0, 1, -1), repeat=m):
        free = [c for c in range(m) if state[c] == 0]
        bound = [c for c in range(m) if state[c] != 0]
        dual = [Fraction(state[c]) for c in range(m)]
        if free:
            matrix = [
                [penalty * gram[c][e] for e in free] for c in free
            ]
            rhs = [
                dy[c]
                - sum(penalty * gram[c][a] * Fraction(state[a]) for a in bound)
                for c in free
            ]
            values = _gauss(matrix, rhs)
            if any(not (-1 <= value <= 1) for value in values):
                continue
            for center, value in zip(free, values):
                dual[center] = value
        diffs = [
            dy[c] - penalty * sum(gram[c][k] * dual[k] for k in range(m))
            for c in range(m)
        ]
        if any(state[c] != 0 and state[c] * diffs[c] <= 0 for c in bound):
            continue
        if any(state[c] == 0 and diffs[c] != 0 for c in free):
            continue
        dt = [[d[k][i] for k in range(m)] for i in range(n)]
        fit = tuple(
            y[i] - penalty * sum(dt[i][k] * dual[k] for k in range(m))
            for i in range(n)
        )
        solutions.append(fit)

    # Strong convexity: every feasible KKT state yields the same unique x.
    assert solutions and all(s == solutions[0] for s in solutions)
    fit = solutions[0]
    knots = tuple(
        c + 1 for c in range(m)
        if fit[c] - 2 * fit[c + 1] + fit[c + 2] != 0
    )
    return fit, knots


def _in_region(intervals, u):
    for interval in intervals:
        if interval.lower.__class__.__name__ == "_Infinity":
            low_ok = True
        elif u == interval.lower.value:
            low_ok = not interval.lower_open
        else:
            low_ok = u > interval.lower.value
        if interval.upper.__class__.__name__ == "_Infinity":
            high_ok = True
        elif u == interval.upper.value:
            high_ok = not interval.upper_open
        else:
            high_ok = u < interval.upper.value
        if low_ok and high_ok:
            return True
    return False


def make_trend_doc(y, penalty, s2="1/1", mu0="0/1", alpha="1/20"):
    return {
        "y": [str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
              for v in y],
        "mode": "trend",
        "lambda": (
            str(penalty.numerator) if penalty.denominator == 1
            else f"{penalty.numerator}/{penalty.denominator}"
        ),
        "s2": s2,
        "mu0": mu0,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class SpecExampleTests(unittest.TestCase):
    def test_worked_example(self):
        y = [Fraction(0), Fraction(0), Fraction(12)]
        result = run_trend_inference(
            y, Fraction(1), Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.knots, (1,))
        self.assertEqual(result.fit, (Fraction(-1), Fraction(2), Fraction(11)))
        self.assertEqual(len(result.intervals), 2)
        left, right = result.intervals
        from seg_infer.regions import INF
        self.assertIs(left.lower, INF)
        self.assertEqual(left.upper.value, Fraction(-6))
        self.assertTrue(left.upper_open)
        self.assertEqual(right.lower.value, Fraction(6))
        self.assertTrue(right.lower_open)
        self.assertIs(right.upper, INF)

    def test_threshold_equality_is_no_change(self):
        # y2 = 6 is exactly the threshold: |s| reaches 1 with zero fitted
        # difference, so S is empty and only the two mandated keys appear.
        y = [Fraction(0), Fraction(0), Fraction(6)]
        result = run_trend_inference(
            y, Fraction(1), Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.knots, ())
        payload = serialize_trend_result(result)
        self.assertEqual(payload, {"S": [], "status": "no_change"})

    def test_just_above_threshold_changes(self):
        y = [Fraction(0), Fraction(0), Fraction(6001, 1000)]
        result = run_trend_inference(
            y, Fraction(1), Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.knots, (1,))


class FitReferenceTests(unittest.TestCase):
    def test_matches_independent_kkt_enumeration(self):
        import random
        rng = random.Random(20261114)
        for _ in range(120):
            n = rng.randint(3, 8)
            y = [
                Fraction(rng.randint(-20, 20), rng.randint(1, 3))
                for _ in range(n)
            ]
            penalty = Fraction(rng.randint(1, 10), rng.randint(1, 3))
            expected_fit, expected_knots = reference_fit(y, penalty)
            got = fit_trend(y, penalty)
            self.assertEqual(got.fit, expected_fit)
            self.assertEqual(got.knots, expected_knots)

    def test_multi_knot_linkage(self):
        # Three coupled interior centers carry knots simultaneously; the
        # independent reference and the solver must agree coordinate by
        # coordinate, including the non-integer reduced rational fit entries.
        y = [Fraction(15), Fraction(-11), Fraction(-3),
             Fraction(5), Fraction(-14), Fraction(-13)]
        penalty = Fraction(1)
        expected_fit, expected_knots = reference_fit(y, penalty)
        result = run_trend_inference(
            y, penalty, Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertEqual(result.knots, (1, 3, 4))
        self.assertEqual(result.fit, expected_fit)
        self.assertEqual(
            serialize_trend_result(result)["fit"],
            [14, "-53/6", "-10/3", "13/6", -11, -14],
        )


class RegionReferenceTests(unittest.TestCase):
    def _assert_region_by_refit(self, y, penalty):
        result = run_trend_inference(
            y, penalty, Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertTrue(result.changed)
        contrast = result.contrast
        norm = contrast.eta_norm_sq
        endpoints = sorted({
            edge.value
            for interval in result.intervals
            for edge in (interval.lower, interval.upper)
            if edge.__class__.__name__ != "_Infinity"
        })
        probes = [Fraction(i, 2) for i in range(-160, 161)]
        for value in endpoints:
            probes += [
                value - Fraction(1, 10 ** 6),
                value,
                value + Fraction(1, 10 ** 6),
            ]
        for u in probes:
            observed = [
                contrast.z[i] + contrast.eta[i] * u / norm
                for i in range(len(y))
            ]
            _, knots = reference_fit(observed, penalty)
            self.assertEqual(
                _in_region(result.intervals, u),
                tuple(knots) == result.knots,
                f"region mismatch at u={u}",
            )
        # The observed point u = t is always feasible.
        self.assertTrue(_in_region(result.intervals, contrast.t))
        return result

    def test_disconnected_multi_knot_region(self):
        # Four all-open components: the knot set survives only inside
        # disjoint intervals as the active dual signs flip along the line.
        y = [Fraction(-3), Fraction(-10), Fraction(-12),
             Fraction(15), Fraction(-14)]
        result = self._assert_region_by_refit(y, Fraction(2))
        self.assertEqual(result.knots, (1, 2, 3))
        self.assertEqual(len(result.intervals), 4)
        for interval in result.intervals:
            # Every finite boundary of this case is a sign-flip root.
            if interval.lower.__class__.__name__ != "_Infinity":
                self.assertTrue(interval.lower_open)
            if interval.upper.__class__.__name__ != "_Infinity":
                self.assertTrue(interval.upper_open)

    def test_sign_flip_across_gap(self):
        # Single knot whose dual sign is -1 on the left component and +1 on
        # the right component; the gap (where the knot disappears) is open at
        # both sides.  Refits independently confirm the flipped signs.
        y = [Fraction(-12), Fraction(3), Fraction(-8), Fraction(-10)]
        result = self._assert_region_by_refit(y, Fraction(4))
        self.assertEqual(result.knots, (1,))
        self.assertEqual(len(result.intervals), 2)
        left, right = result.intervals
        self.assertTrue(left.upper_open)
        self.assertTrue(right.lower_open)
        contrast = result.contrast
        left_probe = left.lower.value if left.lower.__class__.__name__ != "_Infinity" \
            else left.upper.value - 1
        right_probe = right.upper.value if right.upper.__class__.__name__ != "_Infinity" \
            else right.lower.value + 1
        left_y = [
            contrast.z[i] + contrast.eta[i] * left_probe / contrast.eta_norm_sq
            for i in range(4)
        ]
        right_y = [
            contrast.z[i] + contrast.eta[i] * right_probe / contrast.eta_norm_sq
            for i in range(4)
        ]
        fit_left, _ = reference_fit(left_y, Fraction(4))
        fit_right, _ = reference_fit(right_y, Fraction(4))
        self.assertLess(fit_left[0] - 2 * fit_left[1] + fit_left[2], 0)
        self.assertGreater(fit_right[0] - 2 * fit_right[1] + fit_right[2], 0)

    def test_randomized_region_scan(self):
        import random
        rng = random.Random(424242)
        checked = 0
        for _ in range(25):
            n = rng.randint(3, 7)
            y = [Fraction(rng.randint(-12, 12)) for _ in range(n)]
            penalty = Fraction(rng.randint(1, 5))
            fit = fit_trend(y, penalty)
            if not fit.knots:
                continue
            self._assert_region_by_refit(y, penalty)
            checked += 1
        self.assertGreater(checked, 5)

    def test_linear_data_has_no_knots(self):
        # Every second difference of an affine sequence vanishes: the fitted
        # sequence equals the observations and the job reports no_change.
        y = [Fraction(i) for i in range(5)]
        result = run_trend_inference(
            y, Fraction(3), Fraction(1), Fraction(0), Fraction(1, 20)
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.knots, ())

    def test_interval_union_semantics(self):
        from seg_infer.endpoints import RationalEP
        from seg_infer.regions import INF, Interval
        from seg_infer.trend import _union_intervals

        def iv(lo, hi, lo_open=False, hi_open=False):
            return Interval(
                INF if lo is None else RationalEP(Fraction(lo)),
                INF if hi is None else RationalEP(Fraction(hi)),
                lo_open, hi_open,
            )

        # Overlapping intervals merge into one spanning component.
        merged = _union_intervals([iv(0, 2), iv(1, 3)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].lower.value, Fraction(0))
        self.assertEqual(merged[0].upper.value, Fraction(3))

        # Two open components touching at one point stay separate (the shared
        # point belongs to neither), but a closed side closes the boundary.
        open_pair = _union_intervals(
            [iv(None, 1, False, True), iv(1, None, True, False)]
        )
        self.assertEqual(len(open_pair), 2)
        closed_pair = _union_intervals(
            [iv(None, 1, False, False), iv(1, None, True, False)]
        )
        self.assertEqual(len(closed_pair), 1)
        self.assertIs(closed_pair[0].lower, INF)
        self.assertIs(closed_pair[0].upper, INF)

        # Nested interval sharing an open lower bound merges (interiors
        # overlap), rather than being treated as touching components.
        nested = _union_intervals(
            [iv(1, 5, True, False), iv(1, 3, True, False)]
        )
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0].lower.value, Fraction(1))
        self.assertTrue(nested[0].lower_open)
        self.assertEqual(nested[0].upper.value, Fraction(5))


class InputAndCliTests(unittest.TestCase):
    def _bad(self, document, fragment):
        with self.assertRaises(InputError) as ctx:
            parse_job(document)
        self.assertIn(fragment, str(ctx.exception))

    def test_trend_validation_paths(self):
        base = make_trend_doc(
            [Fraction(0), Fraction(0), Fraction(12)], Fraction(1)
        )
        self._bad({**base, "lambda": "0/1"}, "$.lambda")
        self._bad({**base, "lambda": "-2"}, "$.lambda")
        self._bad({**base, "lambda": "1.5"}, "$.lambda")
        self._bad({**base, "lambda": True}, "$.lambda")
        self._bad({**base, "y": [0, 12]}, "$.y")
        self._bad({**base, "mode": "seg"}, "$.mode")
        self._bad({**base, "mode": 3}, "$.mode")
        self._bad({**base, "k": 2}, "$.k")
        del base["lambda"]
        self._bad(base, "$.lambda")
        self._bad(
            {**make_trend_doc([Fraction(0)], Fraction(1))},
            "$.y",
        )

    def test_legacy_documents_remain_unchanged(self):
        legacy = {"y": [0, 1, 0], "k": 2, "s2": "1/1",
                  "mu0": "0/1", "alpha": "1/20"}
        job = parse_job(legacy)
        self.assertEqual(job.mode, "segment")
        self.assertEqual(job.k, 2)
        self._bad({**legacy, "lambda": "1/1"}, "$.lambda")

    def test_cli_roundtrip_and_failure_preserves_file(self):
        good = make_trend_doc(
            [Fraction(0), Fraction(0), Fraction(12)], Fraction(1)
        )
        no_change = make_trend_doc(
            [Fraction(0), Fraction(0), Fraction(6)], Fraction(1)
        )
        bad = make_trend_doc([Fraction(0), Fraction(12)], Fraction(1))
        with tempfile.TemporaryDirectory() as tmp:
            good_path = os.path.join(tmp, "good.json")
            no_change_path = os.path.join(tmp, "no_change.json")
            bad_path = os.path.join(tmp, "bad.json")
            out_path = os.path.join(tmp, "out.json")
            for path, doc in (
                (good_path, good),
                (no_change_path, no_change),
                (bad_path, bad),
            ):
                with open(path, "w") as handle:
                    json.dump(doc, handle)

            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, good_path, out_path],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out_path) as handle:
                payload = json.load(handle)
            self.assertEqual(set(payload), {"S", "V", "p", "theta", "fit"})
            self.assertEqual(payload["fit"], [-1, 2, 11])
            self.assertEqual(payload["S"], [1])

            with open(out_path) as handle:
                before = handle.read()
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, bad_path, out_path],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("$.y", proc.stderr)
            with open(out_path) as handle:
                self.assertEqual(handle.read(), before)

            nc_path = os.path.join(tmp, "nc.json")
            proc = subprocess.run(
                [sys.executable, "-I", ENTRY, no_change_path, nc_path],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(nc_path) as handle:
                self.assertEqual(
                    json.load(handle),
                    {"S": [], "status": "no_change"},
                )

            # The serialization route used by the CLI equals the direct route.
            direct = serialize_job_result(run_job(parse_job(good)))
            self.assertEqual(payload, direct)


if __name__ == "__main__":
    unittest.main(verbosity=2)
