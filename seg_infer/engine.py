"""End-to-end inference pipeline: segmentation, stability region, p-value."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple

from .endpoints import AlgEP, Endpoint, RationalEP
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


def run_inference(
    y: Sequence[Fraction], k: int, s2: Fraction, mu0: Fraction, alpha: Fraction
) -> InferenceResult:
    """Execute the complete backend inference with exact arithmetic."""
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
    return {
        "S": list(result.cuts),
        "V": serialize_intervals(result.intervals),
        "p": result.confidence.p_value,
        "theta": result.confidence.theta_intervals,
    }
