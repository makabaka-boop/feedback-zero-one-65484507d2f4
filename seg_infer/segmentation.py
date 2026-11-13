"""Exact 1-D piecewise-constant segmentation and the contrast vector."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import List, Sequence, Tuple

# A cut tuple holds the k-1 internal break points; each break point equals the
# number of observations preceding it, so cuts are increasing ints in 1..n-1.
Cuts = Tuple[int, ...]


def all_cuts(n: int, k: int) -> List[Cuts]:
    """Enumerate every way to split n points into k non-empty segments."""
    return [tuple(cuts) for cuts in combinations(range(1, n), k - 1)]


def segment_bounds(n: int, cuts: Cuts) -> List[Tuple[int, int]]:
    """Return half-open index ranges ``[start, end)`` of every segment."""
    boundaries = (0,) + cuts + (n,)
    return [
        (boundaries[j], boundaries[j + 1])
        for j in range(len(boundaries) - 1)
    ]


def segment_sse(y: Sequence[Fraction], start: int, end: int) -> Fraction:
    """Within-segment sum of squared deviations from the segment mean."""
    length = end - start
    total = sum(y[start:end], Fraction(0))
    square_sum = sum((value * value for value in y[start:end]), Fraction(0))
    return square_sum - total * total / length


def total_sse(y: Sequence[Fraction], cuts: Cuts) -> Fraction:
    """Exact within-segment SSE of a candidate segmentation."""
    n = len(y)
    return sum(
        (segment_sse(y, start, end) for start, end in segment_bounds(n, cuts)),
        Fraction(0),
    )


def best_segmentation(y: Sequence[Fraction], k: int) -> Tuple[Cuts, List[Cuts]]:
    """Find the minimizer of the within-segment SSE.

    Ties are broken by choosing the lexicographically smallest internal
    break-point array, exactly as mandated.  All candidates are inspected.
    Returns ``(best_cuts, candidates)`` with candidates in enumeration order.
    """
    n = len(y)
    candidates = all_cuts(n, k)
    best_cuts = None
    best_cost = None
    for cuts in candidates:
        cost = total_sse(y, cuts)
        if best_cost is None or cost < best_cost or (cost == best_cost and cuts < best_cuts):
            best_cost = cost
            best_cuts = cuts
    return best_cuts, candidates


@dataclass(frozen=True)
class Contrast:
    """The contrast object built from the first two segments of the optimum."""

    eta: Tuple[Fraction, ...]
    t: Fraction
    v: Fraction
    z: Tuple[Fraction, ...]
    eta_norm_sq: Fraction


def build_contrast(y: Sequence[Fraction], cuts: Cuts, s2: Fraction) -> Contrast:
    """Construct eta, t, v and the residual z with exact rational arithmetic.

    ``eta_i`` is ``1/L1`` on the first segment, ``-1/L2`` on the second and
    zero elsewhere; ``t = eta' y``, ``v = s2 * eta' eta`` and
    ``z = y - eta * t / (eta' eta)``.
    """
    n = len(y)
    bounds = segment_bounds(n, cuts)
    (start1, end1), (start2, end2) = bounds[0], bounds[1]
    length1, length2 = end1 - start1, end2 - start2

    eta: List[Fraction] = [Fraction(0)] * n
    for i in range(start1, end1):
        eta[i] = Fraction(1, length1)
    for i in range(start2, end2):
        eta[i] = Fraction(-1, length2)
    eta_tuple = tuple(eta)

    norm_sq = sum((component * component for component in eta_tuple), Fraction(0))
    t = sum((eta[i] * y[i] for i in range(n)), Fraction(0))
    v = s2 * norm_sq
    z = tuple(y[i] - eta[i] * t / norm_sq for i in range(n))
    return Contrast(eta_tuple, t, v, z, norm_sq)


def parametric_cost_coefficients(
    z: Sequence[Fraction], eta: Sequence[Fraction], cuts: Cuts
) -> Tuple[Fraction, Fraction, Fraction]:
    """Coefficients ``(c2, c1, c0)`` of the candidate's SSE along the line.

    Observations are parametrized as ``y(u) = z + w * u`` where
    ``w = eta / (eta'eta)``.  For segment j of length L with sums
    ``Z = sum z`` and ``W = sum w`` the segment SSE is
    ``sum(z+wu)^2 - (Z+Wu)^2/L``, which is a convex quadratic in u:

      * ``c2 = sum w^2 - W^2/L``
      * ``c1 = 2 * (sum z*w - Z*W/L)``
      * ``c0 = sum z^2 - Z^2/L``.
    """
    n = len(z)
    norm_sq = sum((component * component for component in eta), Fraction(0))
    c2 = c1 = c0 = Fraction(0)
    for start, end in segment_bounds(n, cuts):
        length = end - start
        z_sum = eta_z_sum = Fraction(0)
        z_sq = cross = eta_sq = Fraction(0)
        for i in range(start, end):
            w_i = eta[i] / norm_sq
            z_sum += z[i]
            eta_z_sum += w_i
            z_sq += z[i] * z[i]
            cross += z[i] * w_i
            eta_sq += w_i * w_i
        c2 += eta_sq - eta_z_sum * eta_z_sum / length
        c1 += 2 * (cross - z_sum * eta_z_sum / length)
        c0 += z_sq - z_sum * z_sum / length
    return c2, c1, c0
