#!/usr/bin/env bash
# Reset only the previously identified USB-CAN; never enable or move the robot.
set -euo pipefail

if (( EUID != 0 )); then
  printf '%s\n' 'Run locally: sudo bash /home/user/wq/nero_feedback_diagnosis_20261015/recover_usb_can.sh'
  exit 2
fi
output="/home/user/wq/nero_feedback_diagnosis_20261015/adapter_recovery_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -m 755 "$output"
exec > >(tee "$output/terminal.log") 2>&1
printf 'Output: %s\n' "$output"
for tool in ip usbreset candump cansend timeout lsof grep udevadm; do
  if ! command -v "$tool" >/dev/null; then
    printf 'STOP: required system command not found: %s\n' "$tool"
    exit 2
  fi
done
holders=$(lsof -nP 2>/dev/null | grep -F 'protocol: CAN_RAW' || true)
if [[ -n "$holders" ]]; then
  printf 'STOP: CAN sockets are still open; no reset performed.\n%s\n' "$holders"
  exit 2
fi
device=$(readlink -f /sys/class/net/can0/device/..)
[[ $(<"$device/idVendor") == 1d50 ]]
[[ $(<"$device/idProduct") == 606f ]]
[[ $(<"$device/serial") == 0041001B4148570A20343133 ]]
read -r bus < "$device/busnum"
read -r number < "$device/devnum"
printf -v usb_address '%03d/%03d' "$bus" "$number"
printf 'USB-CAN: %s\n' "$usb_address"

restore_normal_can() {
  ip link set dev can0 down &&
    ip link set dev can0 type can bitrate 1000000 loopback off listen-only off &&
    ip link set dev can0 up
}
cleanup() {
  result=$?
  trap - EXIT
  if ! restore_normal_can; then
    printf '%s\n' 'ERROR: could not restore normal CAN mode. Do not start a robot controller.'
    exit 2
  fi
  exit "$result"
}
trap cleanup EXIT
ip link set dev can0 down
usbreset "$usb_address"
udevadm settle --timeout=5

# These are adapter test frames, never real robot observations.
ip link set dev can0 type can bitrate 1000000 loopback on listen-only off
ip link set dev can0 up
timeout 3 candump -L can0 > "$output/adapter_loopback_only.log" &
listener=$!
sleep 0.3
# This exact read-only firmware query was checked against the installed SDK.
cansend can0 4AF#01
if wait "$listener"; then :; else
  result=$?
  [[ $result == 124 ]] || exit "$result"
fi
if grep -qF '4AF#01' "$output/adapter_loopback_only.log"; then
  printf '%s\n' 'ADAPTER_LOOPBACK=PASS (self-test only, not robot feedback)'
else
  printf '%s\n' 'ADAPTER_LOOPBACK=FAIL'
fi

restore_normal_can
trap - EXIT
printf '%s\n' 'NORMAL_MODE_RESTORED: 1 Mbps, loopback off, listen-only off'
if timeout 5 candump -L can0 > "$output/real_can_passive.log"; then :; else
  result=$?
  [[ $result == 124 ]] || exit "$result"
fi
ip -json -details -statistics link show can0 > "$output/interface_after.json"
printf 'NORMAL_MODE_RECEIVED_LINES=%s\n' "$(wc -l < "$output/real_can_passive.log")"
printf '%s\n' 'No robot enable, disable, reset, motion, or gripper commands were sent.'
printf '%s\n' 'ROS remains stopped. Normal-mode feedback still requires independent verification.'
