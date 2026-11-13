"""Package entry point for the segmented-normal inference backend."""

from .engine import (
    InferenceResult,
    run_inference,
    run_job,
    serialize_job_result,
    serialize_result,
    serialize_trend_result,
)
from .errors import InputError
from .input_parsing import JobRequest, load_document, parse_input, parse_job
from .trend import TrendResult, fit_trend, run_trend_inference

__all__ = [
    "InferenceResult",
    "InputError",
    "JobRequest",
    "TrendResult",
    "fit_trend",
    "load_document",
    "parse_input",
    "parse_job",
    "run_inference",
    "run_job",
    "run_trend_inference",
    "serialize_job_result",
    "serialize_result",
    "serialize_trend_result",
]
