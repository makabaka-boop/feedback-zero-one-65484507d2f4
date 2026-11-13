"""Isotonic (monotone non-decreasing) piecewise-constant fitting.

For ``mode = "ordered"`` every segmentation still splits the observations into
``k`` contiguous non-empty segments, but the fitted segment levels
``m1 <= m2 <= ... <= mk`` are the minimizers of the sum of squared residuals
under that order constraint.  The optimum levels are obtained exactly with the
pool-adjacent-violators algorithm (PAVA): adjacent violating segments are
merged into a block whose level is the block mean, so that equal adjacent
levels are allowed while every internal cut point is retained.

All arithmetic is exact :class:`fractions.Fraction` arithmetic.
"""

from fractions import Fraction
from typing import List, Sequence, Tuple

from .segmentation import Cuts, all_cuts, segment_bounds

# A PAVA block is (observation_sum, length, first_segment, last_segment).
Block = Tuple[Fraction, int, int, int]


def pava_blocks(
    segment_sums: Sequence[Fraction], segment_lengths: Sequence[int]
) -> List[Block]:
    """Run PAVA over fixed consecutive segments.

    Each input segment carries its observation sum and length.  Violating
    adjacent blocks (left mean strictly greater than the right mean) are
    pooled until all block means are non-decreasing.  Mean comparisons use
    exact cross multiplication, never floating point.
    """
    blocks: List[Block] = []
    for index, (total, length) in enumerate(zip(segment_sums, segment_lengths)):
        block: Block = (Fraction(total), int(length), index, index)
        blocks.append(block)
        while len(blocks) >= 2:
            left_sum, left_len, left_first, _ = blocks[-2]
            right_sum, right_len, _, right_last = blocks[-1]
            # Pool while left.mean > right.mean, equivalently
            # left_sum * right_len > right_sum * left_len.
            if left_sum * right_len <= right_sum * left_len:
                break
            blocks[-2] = (
                left_sum + right_sum,
                left_len + right_len,
                left_first,
                right_last,
            )
            blocks.pop()
    return blocks


def _segment_aggregates(
    values: Sequence[Fraction], cuts: Cuts
) -> Tuple[List[Fraction], List[int]]:
    n = len(values)
    sums: List[Fraction] = []
    lengths: List[int] = []
    for start, end in segment_bounds(n, cuts):
        sums.append(sum(values[start:end], Fraction(0)))
        lengths.append(end - start)
    return sums, lengths


def ordered_levels(
    values: Sequence[Fraction], cuts: Cuts
) -> Tuple[Fraction, Tuple[Fraction, ...]]:
    """Return ``(sse, levels)`` of the isotonic fit for one segmentation."""
    sums, lengths = _segment_aggregates(values, cuts)
    blocks = pava_blocks(sums, lengths)
    square_total = sum((value * value for value in values), Fraction(0))
    explained = Fraction(0)
    levels: List[Fraction] = [Fraction(0)] * len(lengths)
    for block_sum, block_len, first, last in blocks:
        explained += block_sum * block_sum / block_len
        mean = block_sum / block_len
        for segment_index in range(first, last + 1):
            levels[segment_index] = mean
    sse = square_total - explained
    return sse, tuple(levels)


def fit_values(values: Sequence[Fraction], cuts: Cuts) -> Tuple[Fraction, ...]:
    """Isotonic fitted value per observation, in the original point order."""
    _, levels = ordered_levels(values, cuts)
    n = len(values)
    fit: List[Fraction] = [Fraction(0)] * n
    for segment_index, (start, end) in enumerate(segment_bounds(n, cuts)):
        for point in range(start, end):
            fit[point] = levels[segment_index]
    return tuple(fit)


def best_ordered_segmentation(
    values: Sequence[Fraction], k: int
) -> Tuple[Cuts, List[Cuts]]:
    """Minimize the isotonic SSE over every k-segment segmentation.

    Ties use the lexicographically smallest internal cut array.  Returns
    ``(best_cuts, candidates)`` with candidates in enumeration order.
    """
    candidates = all_cuts(len(values), k)
    best_cuts = None
    best_cost = None
    for cuts in candidates:
        cost, _ = ordered_levels(values, cuts)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cuts = cuts
        elif cost == best_cost and cuts < best_cuts:
            best_cuts = cuts
    return best_cuts, candidates
