#!/usr/bin/env bash
set -euo pipefail
NERO_V2_PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${NERO_POLICY_PYTHON:-python3}" "$NERO_V2_PACKAGE_DIR/simulation/evaluate.py" "$@"
