"""Package entry point for the segmented-normal inference backend."""

from .engine import InferenceResult, run_inference, serialize_result
from .errors import InputError
from .input_parsing import load_document, parse_input

__all__ = [
    "InferenceResult",
    "InputError",
    "load_document",
    "parse_input",
    "run_inference",
    "serialize_result",
]
