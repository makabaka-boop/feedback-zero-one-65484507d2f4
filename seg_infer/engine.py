"""End-to-end inference pipeline: segmentation, stability region, p-value."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple

from .endpoints import AlgEP, Endpoint, RationalEP
from .probability import ConfidenceResult, invert_confidence
from .rational import rational_to_string
from .regions import INF, Interval, compute_region
from .segmentation import (
    Contrast,
    best_segmentation,
    build_contrast,
    parametric_cost_coefficients,
)
from .trend import TrendResult, run_trend_inference


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


def rational_output(value: Fraction) -> Any:
    """Render an exact rational in input order: integer or reduced ``p/q``."""
    if value.denominator == 1:
        return value.numerator
    return rational_to_string(value)


def serialize_trend_result(result: TrendResult) -> Dict[str, Any]:
    """Serialize a ``mode=trend`` result.

    An empty knot set returns only ``S`` and ``status``; otherwise the
    segmented-job keys are reused and the reduced rational ``fit`` array is
    appended in input order.
    """
    if not result.changed:
        return {"S": [], "status": "no_change"}
    return {
        "S": list(result.knots),
        "V": serialize_intervals(result.intervals),
        "p": result.confidence.p_value,
        "theta": result.confidence.theta_intervals,
        "fit": [rational_output(value) for value in result.fit],
    }


def run_job(job) -> Any:
    """Dispatch a validated job request to the matching inference pipeline."""
    if job.mode == "segment":
        return run_inference(
            job.y, job.k, job.s2, job.mu0, job.alpha
        )
    return run_trend_inference(
        job.y, job.penalty, job.s2, job.mu0, job.alpha
    )


def serialize_job_result(result: Any) -> Dict[str, Any]:
    """Serialize either pipeline's result to the JSON payload."""
    if isinstance(result, TrendResult):
        return serialize_trend_result(result)
    return serialize_result(result)
