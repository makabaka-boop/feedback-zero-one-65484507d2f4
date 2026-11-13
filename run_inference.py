#!/usr/bin/env python3
"""Standalone entry point that works under ``python -I``.

The package directory is added to ``sys.path`` explicitly so that isolated
mode (which ignores PYTHONPATH and the current directory) still finds the
implementation next to this script.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from seg_infer.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
