"""Construction of the exact stability region V.

Every candidate segmentation contributes one constraint on u.  A candidate
that is lexicographically smaller than the optimum may never tie, hence its
constraint is strict; a lexicographically larger candidate is allowed to tie.
The feasible region is obtained by collecting all constraint roots, sorting
the exact algebraic endpoints and testing each cell and each root point.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from math import gcd, isqrt
from typing import List, Optional, Sequence, Tuple

from .endpoints import (
    AlgEP,
    Endpoint,
    RationalEP,
    compare_endpoints,
    make_alg_endpoint,
    sign_at,
)
from .segmentation import Cuts

IntPoly = Tuple[int, int, int]


class _Infinity:
    """Sentinel for an unbounded interval end (serialized as JSON null)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "INF"


INF = _Infinity()


@dataclass(frozen=True)
class Constraint:
    poly: IntPoly          # primitive integer quadratic, leading coefficient > 0
    strict: bool           # True when equality is forbidden (lex-smaller tie)
    origin: Cuts           # candidate that produced the constraint


@dataclass(frozen=True)
class Atom:
    """One element of the ordered decomposition of the real line."""

    kind: str              # "ray", "cell", "point"
    sample: Optional[Fraction]  # rational probe for open cells/rays
    endpoint: Optional[Endpoint] = None  # defined for point atoms only


@dataclass(frozen=True)
class Interval:
    lower: object          # Endpoint or None for -infinity
    upper: object          # Endpoint or None for +infinity
    lower_open: bool
    upper_open: bool


def normalize_quadratic(
    c2: Fraction, c1: Fraction, c0: Fraction
) -> Optional[IntPoly]:
    """Scale rational coefficients to a primitive integer polynomial.

    Returns ``None`` when the polynomial is identically zero.  The leading
    coefficient is normalized to a positive integer.
    """
    if c2 == 0 and c1 == 0 and c0 == 0:
        return None
    scale_lcm = _lcm(c2.denominator, _lcm(c1.denominator, c0.denominator))
    ai = c2.numerator * (scale_lcm // c2.denominator)
    bi = c1.numerator * (scale_lcm // c1.denominator)
    ci = c0.numerator * (scale_lcm // c0.denominator)
    common = gcd(gcd(abs(ai), abs(bi)), abs(ci))
    ai, bi, ci = ai // common, bi // common, ci // common
    if ai < 0:
        ai, bi, ci = -ai, -bi, -ci
    return ai, bi, ci


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def build_constraints(
    coefficients: Sequence[Tuple[Cuts, Tuple[Fraction, Fraction, Fraction], bool]],
) -> Tuple[List[Constraint], List[Tuple[bool, int]]]:
    """Turn candidate cost differences into integer polynomial constraints.

    Each input item is ``(cuts, (c2, c1, c0), strict)`` for
    ``cost(candidate) - cost(optimum)``.  Constant constraints are returned
    separately as ``(is_violation, candidate_index_hint)`` pairs.
    """
    constraints: List[Constraint] = []
    constants: List[Tuple[bool, int]] = []
    for index, (cuts, (c2, c1, c0), strict) in enumerate(coefficients):
        poly = normalize_quadratic(c2, c1, c0)
        if poly is None:
            constants.append((False, index))
            continue
        if poly[0] == 0 and poly[1] == 0:
            # Constant integer polynomial, leading coefficient positive.
            constants.append((poly[2] < 0, index))
            if poly[2] == 0 and strict:
                # Strict constraint with zero polynomial: infeasible point set.
                raise InfeasibleRegionError(cuts)
            continue
        constraints.append(Constraint(poly, strict, cuts))
    return constraints, constants


class InfeasibleRegionError(RuntimeError):
    """Raised if a strict constraint makes the stability region empty."""


def polynomial_roots(poly: IntPoly) -> List[Endpoint]:
    """Return the distinct real roots of a degree-one or degree-two polynomial.

    A double root is returned once: the quadratic has one sign everywhere
    except at the root itself, which is therefore a closed boundary whose two
    adjacent cells share the same feasibility status.  Quadratics with a
    negative leading coefficient (used by the ordered PAVA pipeline) are
    supported as well.
    """
    a, b, c = poly
    if a == 0:
        # Linear: b x + c = 0 with b != 0.
        return [RationalEP(Fraction(-c, b))]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []  # one sign everywhere, no boundary
    root_base = isqrt(disc)
    if root_base * root_base == disc:
        if disc == 0:
            return [RationalEP(Fraction(-b, 2 * a))]
        return [
            RationalEP(Fraction(-b - root_base, 2 * a)),
            RationalEP(Fraction(-b + root_base, 2 * a)),
        ]
    return [
        make_alg_endpoint(poly, 0, allow_negative=True),
        make_alg_endpoint(poly, 1, allow_negative=True),
    ]


def _deduplicate(endpoints: List[Endpoint]) -> List[Endpoint]:
    unique: List[Endpoint] = []
    for endpoint in endpoints:
        for existing in unique:
            if _same_endpoint(endpoint, existing):
                break
        else:
            unique.append(endpoint)
    return unique


def _same_endpoint(left: Endpoint, right: Endpoint) -> bool:
    from .endpoints import endpoints_equal

    return endpoints_equal(left, right)


def _sorted_endpoints(endpoints: List[Endpoint]) -> List[Endpoint]:
    return sorted(endpoints, key=cmp_to_key(compare_endpoints))


def _bounds_of(endpoint: Endpoint) -> Tuple[Fraction, Fraction]:
    """Tight rational bounds ``(lo, hi)`` with lo <= endpoint <= hi."""
    if isinstance(endpoint, AlgEP):
        return endpoint.lo, endpoint.hi
    return endpoint.value, endpoint.value


def _interior_probe(left: Endpoint, right: Endpoint) -> Fraction:
    """Rational point lying strictly between two adjacent distinct endpoints.

    Algebraic brackets are refined locally (only this pair is touched) until
    their inner bounds admit a strict midpoint.  This keeps the overall cost
    linear in the number of endpoints instead of pairwise.
    """
    from .endpoints import compare_endpoints, refine

    left_alg = left if isinstance(left, AlgEP) else None
    right_alg = right if isinstance(right, AlgEP) else None
    for _ in range(300):
        l_lo, l_hi = _bounds_of(left_alg if left_alg else left)
        r_lo, r_hi = _bounds_of(right_alg if right_alg else right)
        low = l_hi
        high = r_lo
        if low < high:
            return (low + high) / 2
        # Brackets still overlap or touch: refine the algebraic neighbour(s).
        if left_alg is not None:
            left_alg = refine(left_alg)
        if right_alg is not None:
            right_alg = refine(right_alg)
    raise RuntimeError("could not build an interior cell probe")  # pragma: no cover


def build_atoms(endpoints: List[Endpoint]) -> List[Atom]:
    """Decompose the real line into open cells and singleton point atoms."""
    ordered = _sorted_endpoints(_deduplicate(endpoints))
    atoms: List[Atom] = []
    if not ordered:
        atoms.append(Atom("ray", Fraction(0)))
        return atoms
    first = ordered[0]
    first_bound = first.lo if isinstance(first, AlgEP) else first.value
    atoms.append(Atom("ray", first_bound - 1))
    atoms.append(Atom("point", None, first))
    for prev, current in zip(ordered, ordered[1:]):
        atoms.append(Atom("cell", _interior_probe(prev, current)))
        atoms.append(Atom("point", None, current))
    last = ordered[-1]
    last_bound = last.hi if isinstance(last, AlgEP) else last.value
    atoms.append(Atom("ray", last_bound + 1))
    return atoms


def _constraint_ok(constraint: Constraint, atom: Atom) -> bool:
    if atom.kind == "point":
        sign = sign_at(constraint.poly, atom.endpoint)  # type: ignore[arg-type]
        if sign < 0:
            return False
        if sign == 0 and constraint.strict:
            return False
        return True
    sign = _rational_sign(constraint.poly, atom.sample)  # type: ignore[arg-type]
    return sign > 0


def _rational_sign(poly: IntPoly, q: Fraction) -> int:
    a, b, c = poly
    value = a * q * q + b * q + c
    if value == 0:  # pragma: no cover - cell samples are never roots
        return 0
    return 1 if value > 0 else -1


def compute_region(
    coefficients: Sequence[Tuple[Cuts, Tuple[Fraction, Fraction, Fraction], bool]],
) -> List[Interval]:
    """Compute the maximal disjoint intervals forming V."""
    constraints, constants = build_constraints(coefficients)
    for violation, _ in constants:
        if violation:
            return []
    endpoints: List[Endpoint] = []
    for constraint in constraints:
        endpoints.extend(polynomial_roots(constraint.poly))
    atoms = build_atoms(endpoints)

    included: List[bool] = []
    for atom in atoms:
        ok = True
        for constraint in constraints:
            if not _constraint_ok(constraint, atom):
                ok = False
                break
        included.append(ok)

    return _merge_atoms(atoms, included)


def _merge_atoms(atoms: List[Atom], included: List[bool]) -> List[Interval]:
    """Turn the binary atom walk into maximal disjoint intervals.

    The atom walk alternates open cells/rays and singleton point atoms.
    An included run starting at a point atom gives a closed boundary; a run
    starting at a cell gives an open boundary whose location is the adjacent
    point.  Runs separated by a single excluded point produce two intervals
    meeting at that open boundary and are intentionally kept apart.
    """
    intervals: List[Interval] = []
    index = 0
    count = len(atoms)
    while index < count:
        if not included[index]:
            index += 1
            continue
        start = index
        while index < count and included[index]:
            index += 1
        end = index - 1
        first, last = atoms[start], atoms[end]

        if first.kind == "ray":
            if start == 0:
                lower, lower_open = INF, False
            else:
                # Right-side ray whose run starts after an excluded point.
                lower, lower_open = atoms[start - 1].endpoint, True
        elif first.kind == "point":
            lower, lower_open = first.endpoint, False
        else:
            lower, lower_open = atoms[start - 1].endpoint, True
        if last.kind == "ray":
            if end == count - 1:
                upper, upper_open = INF, False
            else:
                # Left-side ray whose run ends before an excluded point.
                upper, upper_open = atoms[end + 1].endpoint, True
        elif last.kind == "point":
            upper, upper_open = last.endpoint, False
        else:
            upper, upper_open = atoms[end + 1].endpoint, True
        intervals.append(Interval(lower, upper, lower_open, upper_open))
    return _merge_touching(intervals)


def _merge_touching(intervals: List[Interval]) -> List[Interval]:
    """Merge components that meet without a gap at a shared endpoint.

    Two components are joined when the upper endpoint of the first equals the
    lower endpoint of the second and at least one side includes that point.
    If both sides are open the point is missing and the components stay apart.
    """
    if not intervals:
        return intervals
    from .endpoints import endpoints_equal

    merged = [intervals[0]]
    for nxt in intervals[1:]:
        prev = merged[-1]
        touch = (
            prev.upper is not INF
            and nxt.lower is not INF
            and endpoints_equal(prev.upper, nxt.lower)
        )
        if touch and (not prev.upper_open or not nxt.lower_open):
            merged[-1] = Interval(
                prev.lower, nxt.upper, prev.lower_open, nxt.upper_open
            )
        else:
            merged.append(nxt)
    return merged
