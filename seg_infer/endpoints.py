"""Exact algebraic endpoints of the stability region.

Finite boundaries of :math:`V` are roots of integer polynomials of degree one
or two.  Rational roots are stored as :class:`fractions.Fraction`; irrational
roots of an integer quadratic are stored with the primitive polynomial, the
zero-based root index and an open rational bracket that is refined on demand.
All comparisons and sign tests are therefore exact: floating point is only
used for the final numerical probability calculations.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Tuple

from .rational import frac_float

# Integer quadratic represented as (a, b, c) meaning a*x^2 + b*x + c.
IntPoly = Tuple[int, int, int]


class Endpoint:
    """Base class for ordered exact boundary points."""

    def to_float(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class RationalEP(Endpoint):
    value: Fraction

    def to_float(self) -> float:
        return frac_float(self.value)


@dataclass(frozen=True)
class AlgEP(Endpoint):
    """An irrational real root of the primitive integer quadratic ``poly``."""

    poly: IntPoly
    index: int  # 0 = smaller distinct real root, 1 = larger one
    lo: Fraction
    hi: Fraction

    def to_float(self) -> float:
        """Numerical value refined until the bracket is ~1e-13 wide."""
        key = (self.poly, self.index)
        cached = _FLOAT_CACHE.get(key)
        if cached is not None:
            return cached
        endpoint = self
        for _ in range(200):
            mid = (endpoint.lo + endpoint.hi) / 2
            width = abs(endpoint.hi - endpoint.lo)
            if width <= Fraction(1, 10 ** 13) * max(Fraction(1), abs(mid)):
                break
            endpoint = refine(endpoint)
        value = frac_float((endpoint.lo + endpoint.hi) / 2)
        _FLOAT_CACHE[key] = value
        return value

    def bracket(self) -> Tuple[Fraction, Fraction]:
        return self.lo, self.hi


_FLOAT_CACHE: dict = {}


def _discriminant(poly: IntPoly) -> int:
    a, b, c = poly
    return b * b - 4 * a * c


def make_alg_endpoint(poly: IntPoly, index: int, allow_negative: bool = False) -> AlgEP:
    """Create an algebraic endpoint with a rational bracket around its root.

    By default the primitive polynomial is expected to have a positive leading
    coefficient (the convention used by the unconstrained pipeline).  Setting
    ``allow_negative`` admits quadratics with a negative leading coefficient,
    which arise as ordered (PAVA) cost differences; the root indexing still
    orders roots ascending, only the bracket sign logic is generalized.
    """
    a, b, c = poly
    disc = _discriminant(poly)
    if disc <= 0:
        raise ValueError("algebraic endpoint requires a positive discriminant")
    if isqrt(disc) ** 2 == disc:
        raise ValueError("rational roots must not be stored as algebraic endpoints")
    if not allow_negative and a <= 0:
        raise ValueError("quadratic leading coefficient must be positive")
    # Roots are (-b +/- sqrt(disc)) / (2a), a != 0.
    root_sq = Fraction(disc, 4 * a * a)
    sqrt_lo, sqrt_hi = _sqrt_bracket(root_sq)
    center = Fraction(-b, 2 * a)
    if index == 0:
        lo = center - sqrt_hi
        hi = center - sqrt_lo
    elif index == 1:
        lo = center + sqrt_lo
        hi = center + sqrt_hi
    else:  # pragma: no cover - defensive
        raise ValueError("quadratic root index must be 0 or 1")
    if lo > hi:  # a < 0 flips the bracket orientation
        lo, hi = hi, lo
    return AlgEP(poly, index, lo, hi)


def _sqrt_bracket(value: Fraction) -> Tuple[Fraction, Fraction]:
    """Strict rational bracket ``lo < sqrt(value) < hi`` with integer ends."""
    if value <= 0:
        raise ValueError("sqrt bracket requires a positive value")
    floor_int = isqrt(value.numerator // value.denominator)
    lo = Fraction(floor_int)
    if lo * lo == value:  # pragma: no cover - guarded by callers
        return lo, lo + 1
    hi = lo + 1
    while hi * hi <= value:
        lo = hi
        hi *= 2
    while lo * lo >= value:  # pragma: no cover - defensive
        lo -= 1
    return lo, hi


def refine(endpoint: AlgEP) -> AlgEP:
    """Halve the bracket while keeping the unique root inside it."""
    a, b, c = endpoint.poly
    mid = (endpoint.lo + endpoint.hi) / 2
    value = a * mid * mid + b * mid + c
    if a > 0:
        # a > 0: polynomial is negative strictly between its two roots.
        if endpoint.index == 0:
            # On the smaller root the sign changes from positive to negative.
            if value > 0:
                return AlgEP(endpoint.poly, endpoint.index, mid, endpoint.hi)
            return AlgEP(endpoint.poly, endpoint.index, endpoint.lo, mid)
        if value < 0:
            return AlgEP(endpoint.poly, endpoint.index, mid, endpoint.hi)
        return AlgEP(endpoint.poly, endpoint.index, endpoint.lo, mid)
    # a < 0: the sign pattern is the mirror image of the a > 0 case.
    if endpoint.index == 0:
        if value < 0:
            return AlgEP(endpoint.poly, endpoint.index, mid, endpoint.hi)
        return AlgEP(endpoint.poly, endpoint.index, endpoint.lo, mid)
    if value > 0:
        return AlgEP(endpoint.poly, endpoint.index, mid, endpoint.hi)
    return AlgEP(endpoint.poly, endpoint.index, endpoint.lo, mid)


def _eval_poly(poly: IntPoly, x: Fraction) -> Fraction:
    a, b, c = poly
    return a * x * x + b * x + c


def _eval_at_alg_poly(poly: IntPoly, endpoint: AlgEP) -> Fraction:
    """Evaluate integer polynomial ``poly`` at another quadratic's root.

    The root is represented exactly in the basis ``p0 + p1*sqrt(disc)``; the
    value of any integer quadratic at that root is again a pair
    ``(constant, sqrt_coefficient)`` in that basis.  The result is zero only
    when both components vanish, i.e. the polynomials genuinely share a root.
    """
    qa, qb, qc = endpoint.poly
    disc = _discriminant(endpoint.poly)
    p0 = Fraction(-qb, 2 * qa)
    p1 = Fraction(1 if endpoint.index == 1 else -1, 2 * qa)
    pa, pb, pc = (Fraction(coef) for coef in poly)
    # r^2 expressed in the basis {1, sqrt(disc)}.
    r_sq_const = p0 * p0 + p1 * p1 * disc
    r_sq_sqrt = 2 * p0 * p1
    const = pa * r_sq_const + pb * p0 + pc
    sqrt_coef = pa * r_sq_sqrt + pb * p1
    if const == 0 and sqrt_coef == 0:
        return Fraction(0)
    # Nonzero basis element a + b*sqrt(D): multiplying by the conjugate gives
    # a^2 - b^2*D, which has the same real sign as the original element.
    discriminant_value = const * const - sqrt_coef * sqrt_coef * disc
    if discriminant_value != 0:
        return discriminant_value
    # Conjugate vanishes only when a=0; then the sign is that of b*sqrt(D).
    return sqrt_coef if sqrt_coef != 0 else const


def sign_at(poly: IntPoly, point: Endpoint) -> int:
    """Sign (-1/0/1) of an integer polynomial at an exact endpoint."""
    if isinstance(point, RationalEP):
        value = _eval_poly(poly, point.value)
    else:
        value = _eval_at_alg_poly(poly, point)
    if value == 0:
        return 0
    return 1 if value > 0 else -1


def endpoints_equal(left: Endpoint, right: Endpoint) -> bool:
    """Exact equality test covering rational/quadratic boundary points."""
    if isinstance(left, RationalEP) and isinstance(right, RationalEP):
        return left.value == right.value
    if isinstance(left, AlgEP) and isinstance(right, AlgEP):
        if left.poly == right.poly:
            return left.index == right.index
        # Distinct primitive quadratics of degree at most two cannot share a
        # root without being proportional, so different polynomials mean the
        # endpoints are different points (even if one evaluates to zero at a
        # point that merely happens to be the other's root).
        return False
    if isinstance(left, AlgEP) and isinstance(right, RationalEP):
        return _eval_poly(left.poly, right.value) == 0
    return _eval_poly(right.poly, left.value) == 0  # type: ignore[union-attr]


def compare_endpoints(left: Endpoint, right: Endpoint) -> int:
    """Exact ordering: return -1, 0 or 1."""
    if endpoints_equal(left, right):
        return 0
    if isinstance(left, RationalEP) and isinstance(right, RationalEP):
        return -1 if left.value < right.value else 1
    if isinstance(left, RationalEP):
        return _compare_rational_alg(left.value, right)  # type: ignore[arg-type]
    if isinstance(right, RationalEP):
        result = _compare_rational_alg(right.value, left)
        return -result
    return _compare_alg(left, right)


def _compare_rational_alg(q: Fraction, endpoint: AlgEP) -> int:
    lo, hi = endpoint.lo, endpoint.hi
    if not (lo < q < hi):
        if q <= lo:
            return -1
        return 1
    alg = endpoint
    for _ in range(200):
        if not (alg.lo < q < alg.hi):
            return -1 if q <= alg.lo else 1
        alg = refine(alg)
    raise RuntimeError("rational/quadratic comparison failed to converge")  # pragma: no cover


def _compare_alg(left: AlgEP, right: AlgEP) -> int:
    a, b = left, right
    for _ in range(200):
        if a.hi <= b.lo:
            return -1
        if b.hi <= a.lo:
            return 1
        a = refine(a)
        b = refine(b)
    raise RuntimeError("quadratic/quadratic comparison failed to converge")  # pragma: no cover
