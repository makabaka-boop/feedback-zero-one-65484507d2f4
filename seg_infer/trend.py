"""``mode=trend``: exact L1 second-difference (trend) filtering.

The fit minimizes

    1/2 * sum_i (y_i - x_i)^2  +  lambda * sum_c |x_{c-1} - 2 x_c + x_{c+1}|

with the second sum running over the interior centers c = 1 .. n-2.  The
quadratic term makes the objective strongly convex, so the primal fit ``x`` is
unique.  Everything here is exact rational arithmetic (``fractions.Fraction``).

Dual formulation.  With ``D`` the (n-2) x n second-difference operator the
dual of the primal problem is

    min_{|s_c| <= 1}  1/2 * lambda * s' (D D') s - (D y)' s

(after dividing the dual objective by lambda > 0).  The Gram matrix ``D D'``
is pentadiagonal with entries 6 / -4 / 1 and positive definite, so the box QP
is solved by a finite primal active-set procedure; the primal fit is then
``x = y - lambda D' s``.

Stability region.  Along the contrast line ``y(u) = z + eta u / (eta'eta)``
every KKT quantity is affine in ``u`` for a fixed active set ``S`` and fixed
sign pattern ``sigma`` on ``S``.  The KKT inequalities are therefore linear
in ``u``: free dual variables give weak bounds (closed endpoints) while the
active sign-consistency conditions are strict (open endpoints).  The region
is the union, over all 2^|S| sign patterns, of the resulting intervals, so
every finite boundary is an exact rational.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import List, Optional, Sequence, Tuple

from .endpoints import RationalEP, compare_endpoints
from .probability import ConfidenceResult, invert_confidence
from .regions import INF, Interval
from .segmentation import Contrast

Center = int  # zero-based interior index, in 1 .. n-2


# ---------------------------------------------------------------------------
# Operator helpers
# ---------------------------------------------------------------------------

def second_differences(values: Sequence[Fraction]) -> List[Fraction]:
    """``D v``: one second difference per interior center (length n-2)."""
    return [
        values[c - 1] - 2 * values[c] + values[c + 1]
        for c in range(1, len(values) - 1)
    ]


def transpose_dual(dual: Sequence[Fraction], n: int) -> List[Fraction]:
    """``D' s`` for a dual vector of length ``n - 2``."""
    out = [Fraction(0)] * n
    for offset, value in enumerate(dual):
        center = offset + 1
        out[center - 1] += value
        out[center] -= 2 * value
        out[center + 1] += value
    return out


def gram_matrix(m: int) -> List[List[Fraction]]:
    """Positive-definite Gram matrix ``D D'`` (m x m, m = n - 2)."""
    matrix = [[Fraction(0)] * m for _ in range(m)]
    for i in range(m):
        matrix[i][i] = Fraction(6)
        if i >= 1:
            matrix[i][i - 1] = Fraction(-4)
            matrix[i - 1][i] = Fraction(-4)
        if i >= 2:
            matrix[i][i - 2] = Fraction(1)
            matrix[i - 2][i] = Fraction(1)
    return matrix


# ---------------------------------------------------------------------------
# Exact linear algebra and box-constrained QP
# ---------------------------------------------------------------------------

def solve_linear(matrix: Sequence[Sequence[Fraction]],
                 rhs: Sequence[Fraction]) -> List[Fraction]:
    """Gauss-Jordan elimination with exact rational pivoting."""
    m = len(matrix)
    if m == 0:
        return []
    tableau = [list(matrix[row]) + [rhs[row]] for row in range(m)]
    for col in range(m):
        pivot_row = next(row for row in range(col, m) if tableau[row][col] != 0)
        if pivot_row != col:
            tableau[col], tableau[pivot_row] = tableau[pivot_row], tableau[col]
        pivot = tableau[col][col]
        for row in range(m):
            if row == col:
                continue
            factor = tableau[row][col] / pivot
            if factor == 0:
                continue
            for k in range(col, m + 1):
                tableau[row][k] -= factor * tableau[col][k]
    return [tableau[row][m] / tableau[row][row] for row in range(m)]


def box_qp(hessian: Sequence[Sequence[Fraction]],
           linear: Sequence[Fraction]) -> List[Fraction]:
    """Minimize ``1/2 s' H s - f' s`` subject to ``-1 <= s_i <= 1``.

    Primal active-set method: each iteration either pins a variable to the
    bound it reaches, or releases the bound with the most wrong-signed
    multiplier.  The positive-definite Hessian guarantees finite termination.
    """
    m = len(linear)
    solution = [Fraction(0)] * m
    upper: set = set()  # indices pinned at +1
    lower: set = set()  # indices pinned at -1
    pinned = upper | lower

    for _iteration in range(100 * m + 100):
        free = [i for i in range(m) if i not in pinned]
        proposal = solution[:]
        if free:
            sub_matrix = [
                [hessian[i][j] for j in free] for i in free
            ]
            sub_rhs = [
                linear[i]
                - sum(hessian[i][j] * solution[j] for j in pinned)
                for i in free
            ]
            sub_solution = solve_linear(sub_matrix, sub_rhs)
            for index, component in zip(free, sub_solution):
                proposal[index] = component

        # Longest feasible step from `solution` towards the free minimizer.
        alpha = Fraction(1)
        hits: List[Tuple[int, int]] = []
        for i in free:
            target = proposal[i]
            if target > 1:
                step = (1 - solution[i]) / (target - solution[i])
                if step < alpha:
                    alpha, hits = step, [(i, 1)]
                elif step == alpha:
                    hits.append((i, 1))
            elif target < -1:
                step = (-1 - solution[i]) / (target - solution[i])
                if step < alpha:
                    alpha, hits = step, [(i, -1)]
                elif step == alpha:
                    hits.append((i, -1))

        if alpha < 1:
            for i in free:
                solution[i] += alpha * (proposal[i] - solution[i])
            for index, bound in hits:
                solution[index] = Fraction(bound)
                if bound == 1:
                    upper.add(index)
                else:
                    lower.add(index)
            pinned = upper | lower
            continue

        # The free minimizer is feasible: check the bound multipliers.
        solution = proposal
        gradient = [
            sum(hessian[i][j] * solution[j] for j in range(m)) - linear[i]
            for i in range(m)
        ]
        # At an upper pin stationarity needs g_i <= 0; at a lower pin g_i >= 0.
        worst_index: Optional[int] = None
        worst_side = 0
        worst_violation = Fraction(0)
        for i in upper:
            if gradient[i] > 0 and gradient[i] > worst_violation:
                worst_index, worst_side, worst_violation = i, 1, gradient[i]
        for i in lower:
            if gradient[i] < 0 and -gradient[i] > worst_violation:
                worst_index, worst_side, worst_violation = i, -1, -gradient[i]
        if worst_index is None:
            return solution
        if worst_side == 1:
            upper.remove(worst_index)
        else:
            lower.remove(worst_index)
        pinned = upper | lower

    raise RuntimeError("trend box QP active-set solver did not converge")


# ---------------------------------------------------------------------------
# Trend fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendFit:
    fit: Tuple[Fraction, ...]
    knots: Tuple[Center, ...]   # centers with a nonzero fitted second difference
    dual: Tuple[Fraction, ...]  # optimal dual variables s


def fit_trend(y: Sequence[Fraction], penalty: Fraction) -> TrendFit:
    """Exact unique L1 second-difference fit and its knot set S."""
    n = len(y)
    m = n - 2
    gram = gram_matrix(m)
    hessian = [[penalty * gram[i][j] for j in range(m)] for i in range(m)]
    dual = box_qp(hessian, second_differences(y))
    fitted = [
        y[i] - penalty * transpose_dual(dual, n)[i] for i in range(n)
    ]
    fitted_diff = second_differences(fitted)
    knots = tuple(c + 1 for c in range(m) if fitted_diff[c] != 0)
    return TrendFit(tuple(fitted), knots, tuple(dual))


def build_trend_contrast(
    y: Sequence[Fraction], first_knot: Center, s2: Fraction
) -> Contrast:
    """Contrast vector eta = (1, -2, 1) centered on the first knot."""
    n = len(y)
    eta = [Fraction(0)] * n
    eta[first_knot - 1] = Fraction(1)
    eta[first_knot] = Fraction(-2)
    eta[first_knot + 1] = Fraction(1)
    eta_tuple = tuple(eta)
    norm_sq = sum(component * component for component in eta_tuple)
    t = sum(eta_tuple[i] * y[i] for i in range(n))
    v = s2 * norm_sq
    z = tuple(y[i] - eta_tuple[i] * t / norm_sq for i in range(n))
    return Contrast(eta_tuple, t, v, z, norm_sq)


# ---------------------------------------------------------------------------
# Exact stability region V
# ---------------------------------------------------------------------------

def _lower_compare(a: Interval, b: Interval) -> int:
    """Order intervals by lower bound; the INF sentinel means -infinity."""
    if a.lower is INF:
        return 0 if b.lower is INF else -1
    if b.lower is INF:
        return 1
    return compare_endpoints(a.lower, b.lower)


def invert_matrix(matrix: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    """Exact inverse of a positive-definite rational matrix via Gauss-Jordan."""
    m = len(matrix)
    identity = [
        [Fraction(1) if row == col else Fraction(0) for col in range(m)]
        for row in range(m)
    ]
    tableau = [list(matrix[row]) + identity[row] for row in range(m)]
    for col in range(m):
        pivot_row = next(row for row in range(col, m) if tableau[row][col] != 0)
        if pivot_row != col:
            tableau[col], tableau[pivot_row] = tableau[pivot_row], tableau[col]
        pivot = tableau[col][col]
        for row in range(m):
            if row == col:
                continue
            factor = tableau[row][col] / pivot
            if factor == 0:
                continue
            for k in range(col, 2 * m):
                tableau[row][k] -= factor * tableau[col][k]
    return [
        [tableau[row][m + col] / tableau[row][row] for col in range(m)]
        for row in range(m)
    ]


@dataclass(frozen=True)
class _RegionBasis:
    """Quantities shared by every sign pattern of one region computation.

    Knot signs are parameterized by bits ``w_j in {0, 1}`` with
    ``sigma_j = -1 + 2 w_j``.  Every KKT quantity is affine in ``w``:

      * ``s0_f(w) = s0_base_f - 2 sum_j coupling[f, j] w_j`` on free centers;
      * ``d0_c(w) = d0_base_c + sum_j delta_d[c, j] w_j``;
      * the slopes ``s1`` and ``d1`` do not depend on the pattern at all.

    Gray-code enumeration flips one bit at a time so a pattern is reached
    from the previous one in O(m) exact rational updates.
    """

    m: int
    gram: Tuple[Tuple[Fraction, ...], ...]
    penalty: Fraction
    knots: Tuple[Center, ...]
    free: Tuple[Center, ...]
    s1: Tuple[Fraction, ...]
    d1: Tuple[Fraction, ...]
    s0_base: Tuple[Fraction, ...]       # free-center constants (all bits 0)
    d0_base: Tuple[Fraction, ...]       # fitted-diff constants (all bits 0)
    coupling: Tuple[Tuple[Fraction, ...], ...]  # [free local][knot]
    delta_d: Tuple[Tuple[Fraction, ...], ...]   # [center][knot]


def _build_region_basis(
    m: int,
    gram: Sequence[Sequence[Fraction]],
    penalty: Fraction,
    knots: Sequence[Center],
    dz: Sequence[Fraction],
    dw: Sequence[Fraction],
) -> _RegionBasis:
    """Precompute the affine KKT forms shared by all sign patterns.

    The free centers are the same for every pattern, so the stationarity
    matrix ``A = lambda G_free,free`` is inverted once and the free dual
    components are expressed as affine functions of the knot signs.
    """
    knot_set = set(knots)
    free = tuple(center for center in range(1, m + 1) if center not in knot_set)
    knot_count = len(knots)

    # Pattern-independent free dual slopes s1 = A^{-1} dw and the constant
    # part of the free dual stationarity system.
    slope_free: List[Fraction] = [Fraction(0)] * len(free)
    base_free: List[Fraction] = [Fraction(0)] * len(free)
    coupling: List[List[Fraction]] = [
        [Fraction(0)] * knot_count for _ in free
    ]
    if free:
        matrix = [
            [penalty * gram[c - 1][d - 1] for d in free] for c in free
        ]
        inverse = invert_matrix(matrix)
        for local in range(len(free)):
            slope_free[local] = sum(
                inverse[local][j] * dw[free[j] - 1]
                for j in range(len(free))
            )
            base_free[local] = sum(
                inverse[local][j] * dz[free[j] - 1]
                for j in range(len(free))
            )
            for j, knot in enumerate(knots):
                coupling[local][j] = sum(
                    inverse[local][q] * penalty
                    * gram[free[q] - 1][knot - 1]
                    for q in range(len(free))
                )

    s1 = [Fraction(0)] * m
    for local, center in enumerate(free):
        s1[center - 1] = slope_free[local]
    d1 = [
        dw[c] - penalty * sum(gram[c][k] * s1[k] for k in range(m))
        for c in range(m)
    ]

    # Free dual constants at the all-minus-one pattern (w = 0).
    s0_base = [
        base_free[local] + sum(coupling[local])
        for local in range(len(free))
    ]

    # Fitted differences d0(w) = A_c + sum_j B[c, j] sigma_j, then rewritten
    # in bits: d0_base (w = 0) and delta_d columns for flipping bit j.
    constant_a = [Fraction(0)] * m
    matrix_b = [
        [Fraction(0)] * knot_count for _ in range(m)
    ]
    for c in range(m):
        constant_a[c] = dz[c] - penalty * sum(
            gram[c][f - 1] * base_free[local]
            for local, f in enumerate(free)
        )
        for j, knot in enumerate(knots):
            matrix_b[c][j] = -penalty * gram[c][knot - 1] + penalty * sum(
                gram[c][f - 1] * coupling[local][j]
                for local, f in enumerate(free)
            )
    d0_base = [
        constant_a[c] - sum(matrix_b[c]) for c in range(m)
    ]
    delta_d = tuple(
        tuple(2 * matrix_b[c][j] for j in range(knot_count))
        for c in range(m)
    )
    return _RegionBasis(
        m,
        tuple(tuple(row) for row in gram),
        penalty,
        tuple(knots),
        free,
        tuple(s1),
        tuple(d1),
        tuple(s0_base),
        tuple(d0_base),
        tuple(tuple(row) for row in coupling),
        delta_d,
    )


def _interval_from_state(
    basis: _RegionBasis,
    bits: Sequence[int],
    s0_free: Sequence[Fraction],
    d0: Sequence[Fraction],
) -> Optional[Interval]:
    """Derive the feasible u-interval of one fully specified KKT state.

    Free-center box conditions are weak inequalities (closed endpoints);
    knot sign-consistency conditions are strict (open endpoints).
    """
    lower: Optional[Fraction] = None
    upper: Optional[Fraction] = None
    lower_open = False
    upper_open = False

    def impose(slope: Fraction, constant: Fraction, strict: bool) -> bool:
        """Require ``slope * u + constant >= 0`` (strictly if ``strict``)."""
        nonlocal lower, upper, lower_open, upper_open
        if slope == 0:
            if constant < 0 or (constant == 0 and strict):
                return False
            return True
        root = -constant / slope
        if slope > 0:  # u >= root
            if lower is None or root > lower:
                lower, lower_open = root, strict
            elif root == lower and strict:
                lower_open = True
        else:  # slope < 0: u <= root
            if upper is None or root < upper:
                upper, upper_open = root, strict
            elif root == upper and strict:
                upper_open = True
        return True

    for local, center in enumerate(basis.free):
        k = center - 1
        # -1 <= s0 + s1 u <= 1; touching a bound keeps the knot set S.
        if not impose(basis.s1[k], 1 + s0_free[local], False):
            return None
        if not impose(-basis.s1[k], 1 - s0_free[local], False):
            return None
    for j, knot in enumerate(basis.knots):
        k = knot - 1
        sigma = -1 + 2 * bits[j]
        # sigma * d_k(u) > 0: a zero fitted difference removes the knot.
        if not impose(sigma * basis.d1[k], sigma * d0[k], True):
            return None

    if lower is not None and upper is not None:
        if lower > upper:
            return None
        if lower == upper and (lower_open or upper_open):
            # Open bounds meeting at one root describe an empty cell.
            return None
    low_ep = INF if lower is None else RationalEP(lower)
    up_ep = INF if upper is None else RationalEP(upper)
    return Interval(low_ep, up_ep, lower_open, upper_open)


def enumerate_trend_intervals(basis: _RegionBasis) -> List[Interval]:
    """Enumerate all sign patterns in Gray-code order, updating incrementally."""
    knot_count = len(basis.knots)
    bits = [0] * knot_count
    s0_free = list(basis.s0_base)
    d0 = list(basis.d0_base)

    intervals: List[Interval] = []
    for pattern in range(1 << knot_count):
        interval = _interval_from_state(basis, bits, s0_free, d0)
        if interval is not None:
            intervals.append(interval)
        if pattern == (1 << knot_count) - 1:
            break

        # Gray-code transition: the flipped bit is the low set bit of
        # pattern + 1; delta = +1 for 0 -> 1 and -1 for 1 -> 0.
        flip = (pattern + 1) & -(pattern + 1)
        j = flip.bit_length() - 1
        delta = 1 if bits[j] == 0 else -1
        bits[j] = 1 - bits[j]
        for c in range(basis.m):
            d0[c] += delta * basis.delta_d[c][j]
        for local in range(len(basis.free)):
            s0_free[local] -= 2 * delta * basis.coupling[local][j]
    return intervals


def _union_intervals(intervals: List[Interval]) -> List[Interval]:
    """Union of rational-endpoint intervals, merging overlaps and touches.

    The INF sentinel means -infinity as a lower bound and +infinity as an
    upper bound, so comparisons handle it explicitly rather than through a
    single endpoint ordering.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=cmp_to_key(_lower_compare))
    merged = [ordered[0]]
    for nxt in ordered[1:]:
        prev = merged[-1]
        if prev.upper is INF:
            relation = -1  # prev already reaches +infinity: nxt is nested
        else:
            relation = compare_endpoints(nxt.lower, prev.upper)
        if relation > 0:
            merged.append(nxt)
            continue
        if relation == 0 and prev.upper_open and nxt.lower_open:
            # The shared point is missing from both components.
            merged.append(nxt)
            continue
        # Overlap, or touch with at least one closed side: merge.
        if nxt.upper is INF:
            new_upper, new_upper_open = INF, False
        elif prev.upper is INF:
            new_upper, new_upper_open = prev.upper, prev.upper_open
        elif compare_endpoints(nxt.upper, prev.upper) <= 0:
            new_upper = prev.upper
            new_upper_open = (
                prev.upper_open
                if compare_endpoints(nxt.upper, prev.upper) < 0
                else prev.upper_open and nxt.upper_open
            )
        else:
            new_upper, new_upper_open = nxt.upper, nxt.upper_open
        merged[-1] = Interval(
            prev.lower, new_upper, prev.lower_open, new_upper_open
        )
    return merged


def trend_stability_region(
    y: Sequence[Fraction],
    penalty: Fraction,
    fit: TrendFit,
    contrast: Contrast,
) -> List[Interval]:
    """All u for which refitting y(u) keeps exactly the same knot set S."""
    n = len(y)
    m = n - 2
    gram = gram_matrix(m)
    knots = fit.knots
    direction = [contrast.eta[i] / contrast.eta_norm_sq for i in range(n)]
    dz = second_differences(list(contrast.z))
    dw = second_differences(direction)
    basis = _build_region_basis(m, gram, penalty, knots, dz, dw)
    intervals = enumerate_trend_intervals(basis)
    return _union_intervals(intervals)


# ---------------------------------------------------------------------------
# End-to-end trend job
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendResult:
    changed: bool
    knots: Tuple[Center, ...]
    fit: Optional[Tuple[Fraction, ...]]
    intervals: Optional[List[Interval]]
    confidence: Optional[ConfidenceResult]
    contrast: Optional[Contrast]


def run_trend_inference(
    y: Sequence[Fraction],
    penalty: Fraction,
    s2: Fraction,
    mu0: Fraction,
    alpha: Fraction,
) -> TrendResult:
    """Full ``mode=trend`` pipeline with exact arithmetic."""
    fit = fit_trend(y, penalty)
    if not fit.knots:
        return TrendResult(False, (), None, None, None, None)

    first_knot = fit.knots[0]
    contrast = build_trend_contrast(y, first_knot, s2)
    intervals = trend_stability_region(y, penalty, fit, contrast)
    if not intervals:  # defensive: u = t must always remain feasible
        raise RuntimeError("empty trend stability region")
    confidence = invert_confidence(
        intervals, contrast.t, contrast.v, mu0, alpha
    )
    return TrendResult(
        True, fit.knots, fit.fit, intervals, confidence, contrast
    )
