"""Exact stability region for the isotonic (``mode = "ordered"``) fit.

For a fixed segmentation the ordered cost is the minimum SSE over monotone
segment levels, i.e. a PAVA fit.  Along the line
``y(u) = z + eta * u / (eta' eta)`` every segment sum is affine in ``u``, so
two segment means become equal at a rational "kink" value of ``u``.  Between
consecutive kinks the PAVA block partition of every candidate is fixed and the
candidate cost is an ordinary convex quadratic, whose coefficients are computed
exactly.  The real line is therefore decomposed into:

* open cells/rays sampled at rational points, where quadratic constraints
  decide feasibility *cell by cell* (a constraint polynomial is only valid
  inside the cell whose PAVA partition produced it);
* rational kink points, evaluated directly with the exact PAVA fit;
* quadratic constraint roots (possibly irrational), evaluated exactly through
  the algebraic endpoint sign tests.

Open versus closed inclusion follows the same lexicographic tie rule as the
unconstrained pipeline: a candidate with a lexicographically smaller cut array
may never tie with the optimum, while larger arrays may.
"""

from bisect import bisect_right
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .endpoints import (
    Endpoint,
    RationalEP,
    compare_endpoints,
    sign_at,
)
from .ordered import best_ordered_segmentation, pava_blocks
from .regions import (
    INF,
    Atom,
    Interval,
    _merge_atoms,
    _rational_sign,
    build_atoms,
    polynomial_roots,
)
from .segmentation import Cuts, segment_bounds


def normalize_signed_quadratic(
    c2: Fraction, c1: Fraction, c0: Fraction
) -> Optional[Tuple[int, int, int]]:
    """Scale rational coefficients to a primitive integer polynomial.

    Unlike the unconstrained pipeline, the sign of the leading coefficient is
    preserved: ordered (PAVA) cost differences can be concave quadratics and
    flipping the sign would invert every feasibility test.  Returns ``None``
    only for the identically zero polynomial.
    """
    if c2 == 0 and c1 == 0 and c0 == 0:
        return None
    from math import gcd

    def _lcm(a: int, b: int) -> int:
        return a // gcd(a, b) * b

    scale_lcm = _lcm(c2.denominator, _lcm(c1.denominator, c0.denominator))
    ai = c2.numerator * (scale_lcm // c2.denominator)
    bi = c1.numerator * (scale_lcm // c1.denominator)
    ci = c0.numerator * (scale_lcm // c0.denominator)
    common = gcd(gcd(abs(ai), abs(bi)), abs(ci))
    return ai // common, bi // common, ci // common


def _segment_line_aggregates(
    z: Sequence[Fraction],
    eta: Sequence[Fraction],
    cuts: Cuts,
) -> Tuple[List[Fraction], List[Fraction], List[int]]:
    """Per-segment sums ``Z_j``, ``W_j`` and lengths along ``y(u) = z + w u``.

    Here ``w = eta / (eta' eta)``; the common norm factor is applied by the
    caller where needed.
    """
    n = len(z)
    z_sums: List[Fraction] = []
    w_sums: List[Fraction] = []
    lengths: List[int] = []
    for start, end in segment_bounds(n, cuts):
        z_sums.append(sum(z[start:end], Fraction(0)))
        w_sums.append(sum(eta[start:end], Fraction(0)))
        lengths.append(end - start)
    return z_sums, w_sums, lengths


def collect_kinks(
    candidates: Sequence[Cuts],
    z: Sequence[Fraction],
    eta: Sequence[Fraction],
    eta_norm_sq: Fraction,
) -> List[Fraction]:
    """All rational values of u at which two block means can become equal.

    Every block used by PAVA is a union of consecutive *original* segments, so
    a merge can occur whenever the means of two adjacent segment ranges
    coincide -- not merely when two single adjacent segment means cross.  The
    kink set therefore contains the crossing of the means of every pair of
    adjacent non-empty segment ranges; between consecutive collected kinks the
    PAVA block partition of every candidate is provably fixed.
    """
    kinks: List[Fraction] = []
    seen = set()

    def add_root(const: Fraction, coef: Fraction) -> None:
        if coef == 0:
            return
        root = -const * eta_norm_sq / coef
        if root not in seen:
            seen.add(root)
            kinks.append(root)

    for cuts in candidates:
        z_sums, w_sums, lengths = _segment_line_aggregates(z, eta, cuts)
        seg_count = len(lengths)
        # Prefix sums let us aggregate any consecutive segment range in O(1).
        prefix_z = [Fraction(0)]
        prefix_w = [Fraction(0)]
        prefix_l = [0]
        for j in range(seg_count):
            prefix_z.append(prefix_z[-1] + z_sums[j])
            prefix_w.append(prefix_w[-1] + w_sums[j])
            prefix_l.append(prefix_l[-1] + lengths[j])
        for split in range(1, seg_count):
            # Left range [left, split), right range [split, right); both
            # ranges are contiguous unions of original segments.
            for left in range(split):
                left_z = prefix_z[split] - prefix_z[left]
                left_w = prefix_w[split] - prefix_w[left]
                left_l = prefix_l[split] - prefix_l[left]
                for right in range(split + 1, seg_count + 1):
                    right_z = prefix_z[right] - prefix_z[split]
                    right_w = prefix_w[right] - prefix_w[split]
                    right_l = prefix_l[right] - prefix_l[split]
                    # Equality of the two range means along the line:
                    # (Z_l + W_l u/N)/L_l = (Z_r + W_r u/N)/L_r.
                    const = left_z * right_l - right_z * left_l
                    coef = left_w * right_l - right_w * left_l
                    add_root(const, coef)
    kinks.sort()
    return kinks


def ordered_cost_coefficients(
    z: Sequence[Fraction],
    eta: Sequence[Fraction],
    eta_norm_sq: Fraction,
    cuts: Cuts,
    probe: Fraction,
) -> Tuple[Fraction, Fraction, Fraction]:
    """Quadratic coefficients ``(c2, c1, c0)`` of the candidate's ordered cost.

    The PAVA block partition is taken from the fit at the rational ``probe``
    point, which by construction lies strictly inside a cell where no segment
    means cross, so that partition is constant throughout the cell.  Each
    block then behaves like an ordinary segment SSE in
    ``y(u) = z + (eta/N) u``.
    """
    n = len(z)
    z_sums, w_sums, lengths = _segment_line_aggregates(z, eta, cuts)
    probe_values = [
        z[i] + eta[i] * probe / eta_norm_sq for i in range(n)
    ]
    probe_sums = [
        sum(probe_values[start:end], Fraction(0))
        for start, end in segment_bounds(n, cuts)
    ]
    blocks = pava_blocks(probe_sums, lengths)

    c2 = c1 = c0 = Fraction(0)
    z_sq_total = sum((value * value for value in z), Fraction(0))
    w_sq_total = sum((component * component for component in eta), Fraction(0))
    cross_total = sum((z[i] * eta[i] for i in range(n)), Fraction(0))
    # Per-point totals are independent of the block partition.
    c0 += z_sq_total
    c2 += w_sq_total / (eta_norm_sq * eta_norm_sq)
    c1 += 2 * cross_total / eta_norm_sq
    for block_sum, block_len, first, last in blocks:
        block_z = sum(z_sums[first:last + 1], Fraction(0))
        block_w = sum(w_sums[first:last + 1], Fraction(0))
        c2 -= (block_w * block_w) / (eta_norm_sq * eta_norm_sq * block_len)
        c1 -= 2 * block_z * block_w / (eta_norm_sq * block_len)
        c0 -= block_z * block_z / block_len
    return c2, c1, c0


def _cell_samples(kinks: Sequence[Fraction]) -> List[Fraction]:
    """One strict rational sample per open cell/ray of the kink partition."""
    if not kinks:
        return [Fraction(0)]
    samples = [kinks[0] - 1]
    for left, right in zip(kinks, kinks[1:]):
        samples.append((left + right) / 2)
    samples.append(kinks[-1] + 1)
    return samples


def _locate_cell_at_sample(kinks: Sequence[Fraction], point: Fraction) -> int:
    """Index of the open cell containing a non-kink rational point."""
    return bisect_right(kinks, point)


def compute_ordered_region(
    candidates: Sequence[Cuts],
    optimum: Cuts,
    z: Sequence[Fraction],
    eta: Sequence[Fraction],
    eta_norm_sq: Fraction,
) -> List[Interval]:
    """Build the exact stability region for the ordered optimum segmentation."""
    kinks = collect_kinks(candidates, z, eta, eta_norm_sq)
    samples = _cell_samples(kinks)
    cell_count = len(samples)
    n = len(z)

    # Per-cell quadratic constraint rows: list per cell of
    # (primitive poly, strict), plus a per-cell feasibility flag.
    cell_rows: List[List[Tuple[Tuple[int, int, int], bool]]] = [
        [] for _ in range(cell_count)
    ]
    cell_feasible = [True] * cell_count
    optimum_coeffs = [
        ordered_cost_coefficients(z, eta, eta_norm_sq, optimum, sample)
        for sample in samples
    ]
    for candidate in candidates:
        if candidate == optimum:
            continue
        strict = candidate < optimum
        candidate_coeffs = [
            ordered_cost_coefficients(z, eta, eta_norm_sq, candidate, sample)
            for sample in samples
        ]
        for cell in range(cell_count):
            c2, c1, c0 = (
                candidate_coeffs[cell][j] - optimum_coeffs[cell][j]
                for j in range(3)
            )
            poly = normalize_signed_quadratic(c2, c1, c0)
            if poly is None:
                if strict:
                    cell_feasible[cell] = False
                continue
            if poly[0] == 0 and poly[1] == 0:
                if poly[2] < 0 or (poly[2] == 0 and strict):
                    cell_feasible[cell] = False
                continue
            cell_rows[cell].append((poly, strict))

    # Global endpoint set: kink singletons plus every constraint root that
    # actually falls inside the cell owning its polynomial.  A quadratic built
    # from one cell's PAVA partition has no meaning past the adjacent kinks:
    # outside the cell the correct formula is the *neighbouring* cell's, so an
    # escaping root never creates a boundary there.  The kink itself stays an
    # atom and is evaluated directly with the exact PAVA fit.
    endpoints: List[Endpoint] = [RationalEP(kink) for kink in kinks]
    owner: Dict[Endpoint, int] = {}
    for cell, rows in enumerate(cell_rows):
        cell_left = INF if cell == 0 else RationalEP(kinks[cell - 1])
        cell_right = INF if cell == cell_count - 1 else RationalEP(kinks[cell])
        for poly, _strict in rows:
            for root in polynomial_roots(poly):
                if cell_left is not INF:
                    if compare_endpoints(cell_left, root) >= 0:
                        continue
                if cell_right is not INF:
                    if compare_endpoints(root, cell_right) >= 0:
                        continue
                endpoints.append(root)
                owner[root] = cell
    atoms = build_atoms(endpoints)

    rational_cache: Dict[Fraction, Cuts] = {}

    def winner_at(point: Fraction) -> Cuts:
        if point not in rational_cache:
            values = [
                z[i] + eta[i] * point / eta_norm_sq for i in range(n)
            ]
            winner, _ = best_ordered_segmentation(values, len(optimum) + 1)
            rational_cache[point] = winner
        return rational_cache[point]

    def feasible(atom: Atom) -> bool:
        if atom.kind == "point":
            endpoint = atom.endpoint
            if isinstance(endpoint, RationalEP):
                # Kinks and rational constraint roots are evaluated directly
                # with the exact PAVA fit; the lex tie rule settles inclusion.
                return winner_at(endpoint.value) == optimum
            # Algebraic root: only its owning cell's constraints apply.
            cell = owner[endpoint]
            if not cell_feasible[cell]:
                return False
            for poly, strict in cell_rows[cell]:
                sign = sign_at(poly, endpoint)
                if sign < 0 or (sign == 0 and strict):
                    return False
            return True

        # Open cell or ray: locate the owning cell from the rational sample.
        # Every polynomial here is the valid cost difference on this whole
        # cell, so its sign at the sample determines feasibility throughout.
        cell = _locate_cell_at_sample(kinks, atom.sample)
        if not cell_feasible[cell]:
            return False
        for poly, strict in cell_rows[cell]:
            sign = _rational_sign(poly, atom.sample)
            if sign < 0 or (sign == 0 and strict):
                return False
        return True

    included = [feasible(atom) for atom in atoms]
    return _merge_atoms(atoms, included)
