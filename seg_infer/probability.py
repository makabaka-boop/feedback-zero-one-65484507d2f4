"""Truncated-normal probabilities and exact-set confidence inversion.

The normal CDF is evaluated from the standard-library error function.  The
region handling (open/closed boundaries, interval unions, root inversion and
bisection) is implemented here rather than delegated to a distribution library.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, List, Optional, Sequence

from .rational import frac_float
from .regions import INF, Interval

_EPS = 1e-14


def standard_cdf(x: float) -> float:
    """Standard normal CDF via :func:`math.erf`."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def standard_quantile(p: float) -> float:
    """Inverse standard normal CDF via bisection on :func:`math.erf`."""
    if not 0.0 < p < 1.0:
        raise ValueError("quantile argument must lie strictly in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if standard_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def interval_mass(interval: Interval, mean: float, sd: float) -> float:
    """Unnormalized Gaussian mass of one region component."""
    def cdf(point: Optional[EndpointLike]) -> float:
        if point is INF or point is None:
            return 1.0
        return standard_cdf((point.to_float() - mean) / sd)

    left_cdf = 0.0 if interval.lower is INF else cdf(interval.lower)
    right_cdf = 1.0 if interval.upper is INF else cdf(interval.upper)
    return right_cdf - left_cdf


# Structural typing would add a dependency-free Protocol; keep a loose alias.
EndpointLike = object


def conditional_cdf(
    intervals: Sequence[Interval], t: float, mean: float, sd: float
) -> float:
    """F_theta(t) = P(X <= t | X in V) for X ~ N(mean, sd^2).

    Boundary inclusions do not change continuous probabilities.
    """
    numerator = 0.0
    denominator = 0.0
    for interval in intervals:
        mass = interval_mass(interval, mean, sd)
        denominator += mass
        left = -math.inf if interval.lower is INF else interval.lower.to_float()
        right = math.inf if interval.upper is INF else interval.upper.to_float()
        if t >= right:
            numerator += mass
        elif t > left:
            numerator += (
                standard_cdf((t - mean) / sd)
                - (0.0 if interval.lower is INF
                   else standard_cdf((left - mean) / sd))
            )
    if denominator <= 0.0:
        # At extreme means the truncated Gaussian mass underflows to zero.
        # The limiting conditional CDF is 0 (mean far right) or 1 (far left).
        if mean > t:
            return 0.0
        return 1.0
    return min(1.0, max(0.0, numerator / denominator))


@dataclass(frozen=True)
class ConfidenceResult:
    p_value: float
    theta_intervals: List[List[float]]  # always zero or one closed interval


def _side_masses(
    intervals: Sequence[Interval], t_float: float, sd: float
) -> tuple:
    """Unnormalized Gaussian mass of V below and above t at probe mean 0."""
    below = above = 0.0
    probe_mean = 0.0
    for interval in intervals:
        left = -math.inf if interval.lower is INF else interval.lower.to_float()
        right = math.inf if interval.upper is INF else interval.upper.to_float()
        full = interval_mass(interval, probe_mean, sd)
        if right <= t_float:
            below += full
        elif left >= t_float:
            above += full
        else:
            cdf_t = standard_cdf((t_float - probe_mean) / sd)
            cdf_left = 0.0 if interval.lower is INF else standard_cdf(
                (left - probe_mean) / sd
            )
            piece = cdf_t - cdf_left
            below += piece
            above += full - piece
    return below, above


def _bracket_root(
    f_at: Callable[[float], float], start: float, sd: float, target: float
) -> tuple:
    """Find ``(lo, hi)`` with F(lo) >= target >= F(hi) around one root.

    The starting point need not straddle the target; the search first walks
    in whichever direction restores the bracket, then geometrically expands.
    """
    x = start
    value = f_at(x)
    step = max(sd, 1e-12)
    limit = max(abs(start) + 40.0 * sd, 40.0 * sd)
    lo = hi = None
    for _ in range(200):
        if value >= target:
            lo = x
        if value <= target:
            hi = x
        if lo is not None and hi is not None:
            return lo, hi
        if abs(x) > limit:
            # The target is unattainable before the 0/1 plateau; clamp at the
            # reached side so bisection returns the limiting boundary.
            if lo is None:
                lo = x
            if hi is None:
                hi = x
            return lo, hi
        if value > target:
            x += step  # decreasing F: move right toward the target
        elif value < target:
            x -= step  # move left toward the target
        else:
            # Exact hit: a zero-width bracket still converges to x.
            return x, x
        new_value = f_at(x)
        if new_value == value:
            # Reached the 0/1 plateau without crossing: target unattainable.
            if lo is None:
                lo = x
            if hi is None:
                hi = x
            return lo, hi
        value = new_value
        step *= 2.0
    return (lo if lo is not None else x), (hi if hi is not None else x)


def _bisect(f_at: Callable[[float], float], lo: float, hi: float,
            target: float) -> float:
    """Bisection for a decreasing F, F(lo) >= target >= F(hi)."""
    f_lo, f_hi = f_at(lo), f_at(hi)
    if f_lo < target or f_hi > target:  # pragma: no cover - defensive
        return lo if abs(f_lo - target) < abs(f_hi - target) else hi
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if f_at(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def invert_confidence(
    intervals: Sequence[Interval],
    t: Fraction,
    v: Fraction,
    mu0: Fraction,
    alpha: Fraction,
) -> ConfidenceResult:
    """Compute the two-sided p-value and the confidence set of theta."""
    sd = math.sqrt(max(frac_float(v), 0.0))
    t_float = frac_float(t)
    mu0_float = frac_float(mu0)

    f_null = conditional_cdf(intervals, t_float, mu0_float, sd)
    p_value = min(1.0, 2.0 * min(f_null, 1.0 - f_null))

    target_low = frac_float(alpha / 2)
    target_high = 1.0 - target_low

    below, above = _side_masses(intervals, t_float, sd)
    if below <= _EPS or above <= _EPS:
        # V lies on one side of t: the two-sided confidence set is empty.
        return ConfidenceResult(p_value, [])

    def f_at(mean: float) -> float:
        return conditional_cdf(intervals, t_float, mean, sd)

    low_bracket = _bracket_root(f_at, t_float, sd, target_high)
    high_bracket = _bracket_root(f_at, t_float, sd, target_low)
    theta_low = _bisect(f_at, low_bracket[0], low_bracket[1], target_high)
    theta_high = _bisect(f_at, high_bracket[0], high_bracket[1], target_low)
    return ConfidenceResult(p_value, [[theta_low, theta_high]])
