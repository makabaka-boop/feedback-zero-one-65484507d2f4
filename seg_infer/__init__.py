"""Package entry point for the segmented-normal inference backend."""

from .engine import InferenceResult, run_inference, serialize_result
from .errors import InputError
from .input_parsing import load_document, parse_input, parse_input_full

__all__ = [
    "InferenceResult",
    "InputError",
    "load_document",
    "parse_input",
    "parse_input_full",
    "run_inference",
    "serialize_result",
]
