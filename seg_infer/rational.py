"""Exact rational arithmetic helpers built on :class:`fractions.Fraction`."""

from fractions import Fraction

from .errors import InputError


def parse_rational(value, path: str) -> Fraction:
    """Parse ``value`` into an exact rational.

    Accepted forms are an explicit ``int`` (booleans are rejected as illegal
    types) or a ``"p/q"`` string where ``p`` is any integer and ``q`` is a
    positive integer.  A bare integer string (``"p"``) is also accepted with
    implicit denominator 1.  Floats, booleans and any other type are rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InputError(
            path,
            "must be an integer or a 'p/q' string with integer numerator and "
            "positive integer denominator (a bare integer string is also accepted)",
        )
    if isinstance(value, int):
        return Fraction(value)

    text = value.strip()
    parts = text.split("/")
    if len(parts) > 2:
        raise InputError(path, f"invalid rational {text!r}: expected 'p/q'")
    try:
        numerator = int(parts[0], 10)
    except ValueError:
        raise InputError(path, f"invalid numerator in rational {text!r}")
    if len(parts) == 1:
        denominator = 1
    else:
        denom_text = parts[1]
        if denom_text == "" or denom_text.startswith(("+", "-")):
            raise InputError(
                path, f"invalid denominator in rational {text!r}: must be positive integer"
            )
        try:
            denominator = int(denom_text, 10)
        except ValueError:
            raise InputError(path, f"invalid denominator in rational {text!r}")
    if denominator <= 0:
        raise InputError(path, f"denominator in {text!r} must be positive")
    return Fraction(numerator, denominator)


def rational_to_string(q: Fraction) -> str:
    """Render a rational as the canonical ``p/q`` string."""
    return f"{q.numerator}/{q.denominator}"


def frac_float(q: Fraction) -> float:
    """Convert a possibly large rational to float without raising."""
    try:
        return float(q)
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        if q > 0:
            return float("inf")
        return float("-inf")
