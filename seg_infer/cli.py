"""Command-line entry point.

Usage::

    python -m seg_infer.cli INPUT.json OUTPUT.json

The output file is replaced only after every computation has succeeded; on any
validation or runtime failure the previous output file is left untouched.
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict

from .engine import run_job, serialize_job_result
from .errors import InputError
from .input_parsing import load_document, parse_job


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".seg_infer-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact segmented-normal inference backend "
        "(segmentation and trend modes)"
    )
    parser.add_argument("input", help="path to the input JSON document")
    parser.add_argument("output", help="path to the output JSON document")
    args = parser.parse_args(argv)

    try:
        document = load_document(args.input)
        job = parse_job(document)
        result = run_job(job)
        payload = serialize_job_result(result)
    except InputError as exc:
        print(f"input error {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        print(f"computation error: {exc}", file=sys.stderr)
        return 1

    _atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
