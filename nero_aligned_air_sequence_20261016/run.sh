#!/usr/bin/env bash
set -eo pipefail
NERO_PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NERO_ROS_SETUP="${NERO_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
NERO_WS_SETUP="${NERO_WS_SETUP:-/home/user/projects/nero_ros2_ws/install/setup.bash}"
if [[ ! -f "$NERO_ROS_SETUP" || ! -f "$NERO_WS_SETUP" ]]; then
    echo 'ROS/workspace setup missing. Set NERO_ROS_SETUP and NERO_WS_SETUP to the installed setup.bash paths.' >&2
    exit 1
fi
source "$NERO_ROS_SETUP"
source "$NERO_WS_SETUP"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
cd "$NERO_PACKAGE_DIR"
NERO_RESPONSE_ARGS=("$@")
NERO_HAS_OUTPUT=false
for NERO_RESPONSE_ARG in "$@"; do
    if [[ "$NERO_RESPONSE_ARG" == '--output' || "$NERO_RESPONSE_ARG" == --output=* ]]; then
        NERO_HAS_OUTPUT=true
    fi
done
if [[ "$NERO_HAS_OUTPUT" == false ]]; then
    NERO_RESPONSE_ARGS+=(--output "logs/act_$(date +%Y%m%d_%H%M%S_%N)")
fi
exec /usr/bin/python3 full_runtime.py "${NERO_RESPONSE_ARGS[@]}"
