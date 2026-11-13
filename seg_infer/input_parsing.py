"""Strict JSON input validation and parsing."""

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from .errors import InputError
from .rational import parse_rational

_REQUIRED = ("y", "k", "s2", "mu0", "alpha")
_VALID_MODES = ("segment", "trend")


@dataclass(frozen=True)
class JobRequest:
    """Validated job description.

    ``mode`` is ``"segment"`` (the original piecewise-constant job) or
    ``"trend"`` (L1 second-difference filtering).  Exactly one of ``k`` /
    ``lambda`` is populated, according to the mode.
    """

    y: List[Fraction]
    s2: Fraction
    mu0: Fraction
    alpha: Fraction
    mode: str
    k: Optional[int] = None
    penalty: Optional[Fraction] = None


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


def _parse_y(document: Dict[str, Any], minimum_n: int) -> List[Fraction]:
    raw_y = document["y"]
    if not isinstance(raw_y, list):
        raise InputError("$.y", "must be an array")
    if len(raw_y) < minimum_n:
        raise InputError(
            "$.y",
            f"must contain at least n={minimum_n} observations",
        )
    y: List[Fraction] = []
    for index, item in enumerate(raw_y):
        y.append(parse_rational(item, f"$.y[{index}]"))
    return y


def parse_input(document: Any) -> Tuple[List[Fraction], int, Fraction, Fraction, Fraction]:
    """Validate the original (segmentation) job, unchanged contract.

    Returns ``(y, k, s2, mu0, alpha)`` with all quantities exact rationals
    (``k`` is a genuine Python int, never a bool).  Documents using the new
    ``mode`` / ``lambda`` members must go through :func:`parse_job`.
    """
    job = parse_job(document)
    if job.mode != "segment" or job.k is None:  # pragma: no cover - defensive
        raise InputError("$", "parse_input only accepts segmentation jobs")
    return job.y, job.k, job.s2, job.mu0, job.alpha


def parse_job(document: Any) -> JobRequest:
    """Validate any supported job and return its normalized description.

    When ``mode`` is absent the document must be the legacy segmentation job
    (fixed members ``y, k, s2, mu0, alpha``).  ``mode="trend"`` replaces ``k``
    by a strictly positive rational ``lambda`` and requires ``n >= 3``; the
    remaining validation rules are inherited unchanged.
    """
    if not isinstance(document, dict):
        raise InputError("$", "top-level value must be a JSON object")

    if "mode" not in document:
        # Legacy form: exact fixed-member check, including unknown members.
        for key in _REQUIRED:
            if key not in document:
                raise InputError(f"$.{key}", "missing required member")
        unknown = sorted(set(document) - set(_REQUIRED))
        if unknown:
            raise InputError(f"$.{unknown[0]}", "unknown member")
        mode = "segment"
    else:
        raw_mode = document["mode"]
        if not isinstance(raw_mode, str) or raw_mode not in _VALID_MODES:
            raise InputError(
                "$.mode",
                "must be one of " + ", ".join(repr(v) for v in _VALID_MODES),
            )
        mode = raw_mode
        base_members = {"y", "mode", "s2", "mu0", "alpha"}
        mode_member = "k" if mode == "segment" else "lambda"
        allowed = base_members | {mode_member}
        for key in ("y", "s2", "mu0", "alpha"):
            if key not in document:
                raise InputError(f"$.{key}", "missing required member")
        if mode_member not in document:
            raise InputError(f"$.{mode_member}", "missing required member")
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise InputError(f"$.{unknown[0]}", "unknown member")

    minimum_n = 3 if mode == "trend" else 2
    y = _parse_y(document, minimum_n)
    n = len(y)

    s2 = parse_rational(document["s2"], "$.s2")
    if s2 <= 0:
        raise InputError("$.s2", "must be strictly positive")

    mu0 = parse_rational(document["mu0"], "$.mu0")

    alpha = parse_rational(document["alpha"], "$.alpha")
    if not 0 < alpha < 1:
        raise InputError("$.alpha", "must satisfy 0 < alpha < 1")

    if mode == "segment":
        raw_k = document["k"]
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise InputError(
                "$.k",
                "must be an integer (booleans are not accepted)",
            )
        k = raw_k
        if not 2 <= k <= n:
            raise InputError("$.k", f"must satisfy 2 <= k <= n with n={n}")
        return JobRequest(y, s2, mu0, alpha, "segment", k=k)

    penalty = parse_rational(document["lambda"], "$.lambda")
    if penalty <= 0:
        raise InputError("$.lambda", "must be strictly positive")
    return JobRequest(y, s2, mu0, alpha, "trend", penalty=penalty)
