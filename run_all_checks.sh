#!/usr/bin/env bash
# Run every independent acceptance check in isolated Python mode.
set -euo pipefail
cd "$(dirname "$0")"
python3 -I tests/test_acceptance.py
python3 -I tests/test_exactness.py
python3 -I tests/test_stress_fast.py
