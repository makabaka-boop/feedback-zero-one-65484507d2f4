"""End-to-end inference pipeline: segmentation, stability region, p-value."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .endpoints import AlgEP, Endpoint, RationalEP
from .ordered import best_ordered_segmentation, fit_values
from .ordered_region import compute_ordered_region
from .probability import ConfidenceResult, invert_confidence
from .regions import INF, Interval, compute_region
from .rational import rational_to_string
from .segmentation import (
    Contrast,
    best_segmentation,
    build_contrast,
    parametric_cost_coefficients,
)


@dataclass(frozen=True)
class InferenceResult:
    cuts: Tuple[int, ...]
    intervals: List[Interval]
    confidence: ConfidenceResult
    contrast: Contrast
    fit: Optional[Tuple[Fraction, ...]] = None


def run_inference(
    y: Sequence[Fraction],
    k: int,
    s2: Fraction,
    mu0: Fraction,
    alpha: Fraction,
    mode: str = "default",
) -> InferenceResult:
    """Execute the complete backend inference with exact arithmetic."""
    if mode == "ordered":
        return _run_ordered(y, k, s2, mu0, alpha)
    if mode != "default":  # pragma: no cover - rejected at the parsing layer
        raise ValueError(f"unsupported mode {mode!r}")

    cuts, candidates = best_segmentation(y, k)
    contrast = build_contrast(y, cuts, s2)

    optimum_coeffs = parametric_cost_coefficients(
        contrast.z, contrast.eta, cuts
    )
    coefficient_rows = []
    for candidate in candidates:
        if candidate == cuts:
            continue
        coeffs = parametric_cost_coefficients(
            contrast.z, contrast.eta, candidate
        )
        difference = tuple(coeffs[j] - optimum_coeffs[j] for j in range(3))
        strict = candidate < cuts
        coefficient_rows.append((candidate, difference, strict))

    intervals = compute_region(coefficient_rows)
    confidence = invert_confidence(
        intervals, contrast.t, contrast.v, mu0, alpha
    )
    return InferenceResult(cuts, intervals, confidence, contrast)


def _run_ordered(
    y: Sequence[Fraction],
    k: int,
    s2: Fraction,
    mu0: Fraction,
    alpha: Fraction,
) -> InferenceResult:
    """Ordered (isotonic) variant: monotone segment levels and exact region."""
    cuts, candidates = best_ordered_segmentation(y, k)
    # eta is still built from the lengths of the first two selected segments,
    # never from the fitted platforms; t, v and z come from the original y.
    contrast = build_contrast(y, cuts, s2)
    fit = fit_values(y, cuts)

    intervals = compute_ordered_region(
        candidates, cuts, contrast.z, contrast.eta, contrast.eta_norm_sq
    )
    confidence = invert_confidence(
        intervals, contrast.t, contrast.v, mu0, alpha
    )
    return InferenceResult(cuts, intervals, confidence, contrast, fit)



def serialize_endpoint(endpoint: Any) -> Any:
    if endpoint is INF:
        return None
    if isinstance(endpoint, RationalEP):
        return {"rational": rational_to_string(endpoint.value)}
    if isinstance(endpoint, AlgEP):
        a, b, c = endpoint.poly
        return {
            "poly": [a, b, c],
            "root_index": endpoint.index,
        }
    raise TypeError(f"unsupported endpoint {endpoint!r}")  # pragma: no cover


def serialize_intervals(intervals: Sequence[Interval]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for interval in intervals:
        item: Dict[str, Any] = {
            "lower": serialize_endpoint(interval.lower),
            "upper": serialize_endpoint(interval.upper),
        }
        if interval.lower is not INF:
            item["lower_open"] = interval.lower_open
        if interval.upper is not INF:
            item["upper_open"] = interval.upper_open
        serialized.append(item)
    return serialized


def serialize_result(result: InferenceResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "S": list(result.cuts),
        "V": serialize_intervals(result.intervals),
        "p": result.confidence.p_value,
        "theta": result.confidence.theta_intervals,
    }
    if result.fit is not None:
        payload["fit"] = [rational_to_string(value) for value in result.fit]
    return payload
