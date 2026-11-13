"""Strict JSON input validation and parsing."""

import json
from fractions import Fraction
from typing import Any, Dict, List, Tuple

from .errors import InputError
from .rational import parse_rational

_REQUIRED = ("y", "k", "s2", "mu0", "alpha")
_OPTIONAL = ("mode",)
_VALID_MODES = ("ordered",)


class _RejectingObject(dict):
    """Dict that rejects repeated keys while JSON is being decoded."""

    def __init__(self, pairs):
        super().__init__()
        for key, value in pairs:
            if key in self:
                raise InputError(
                    "$",
                    f"duplicate member {key!r} in JSON object",
                )
            self[key] = value


def load_document(path: str) -> Dict[str, Any]:
    """Load a JSON document, flagging duplicate object members."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        document = json.loads(text, object_pairs_hook=_RejectingObject)
    except InputError:
        raise
    except json.JSONDecodeError as exc:
        raise InputError("$", f"invalid JSON: {exc.msg} at line {exc.lineno}")
    return document


def parse_input(document: Any) -> Tuple[List[Fraction], int, Fraction, Fraction, Fraction]:
    """Validate the top-level structure and every scalar field.

    Returns ``(y, k, s2, mu0, alpha)`` with all quantities exact rationals
    (``k`` is a genuine Python int, never a bool).  The optional ``mode``
    member is still validated; use :func:`parse_input_full` to obtain it.
    """
    y, k, s2, mu0, alpha, _mode = parse_input_full(document)
    return y, k, s2, mu0, alpha


def parse_input_full(
    document: Any,
) -> Tuple[List[Fraction], int, Fraction, Fraction, Fraction, str]:
    """Like :func:`parse_input`, additionally returning the validated mode."""
    if not isinstance(document, dict):
        raise InputError("$", "top-level value must be a JSON object")

    for key in _REQUIRED:
        if key not in document:
            raise InputError(f"$.{key}", "missing required member")
    unknown = sorted(set(document) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise InputError(f"$.{unknown[0]}", "unknown member")

    mode = "default"
    if "mode" in document:
        raw_mode = document["mode"]
        if not isinstance(raw_mode, str) or raw_mode not in _VALID_MODES:
            raise InputError(
                "$.mode",
                "must be the string 'ordered' when present",
            )
        mode = raw_mode

    raw_y = document["y"]
    if not isinstance(raw_y, list):
        raise InputError("$.y", "must be an array")
    if len(raw_y) < 2:
        raise InputError("$.y", "must contain at least n=2 observations")
    y: List[Fraction] = []
    for index, item in enumerate(raw_y):
        y.append(parse_rational(item, f"$.y[{index}]"))

    raw_k = document["k"]
    if isinstance(raw_k, bool) or not isinstance(raw_k, int):
        raise InputError("$.k", "must be an integer (booleans are not accepted)")
    k = raw_k
    n = len(y)
    if not 2 <= k <= n:
        raise InputError("$.k", f"must satisfy 2 <= k <= n with n={n}")

    s2 = parse_rational(document["s2"], "$.s2")
    if s2 <= 0:
        raise InputError("$.s2", "must be strictly positive")

    mu0 = parse_rational(document["mu0"], "$.mu0")

    alpha = parse_rational(document["alpha"], "$.alpha")
    if not 0 < alpha < 1:
        raise InputError("$.alpha", "must satisfy 0 < alpha < 1")

    return y, k, s2, mu0, alpha, mode
