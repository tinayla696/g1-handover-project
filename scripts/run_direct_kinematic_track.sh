#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="${1:-g1_handover_teacher}"
PLAYBACK_SPEED="${PLAYBACK_SPEED:-1.0}"
export PLAYBACK_SPEED
export PLAYBACK_MAX_EPISODES="${PLAYBACK_MAX_EPISODES:-1}"
export PLAYBACK_MAX_STEPS="${PLAYBACK_MAX_STEPS:--1}"

if [[ -z "${DISPLAY:-}" ]]; then
  DISPLAY=:1.0
fi
export DISPLAY

if [[ -z "${XAUTHORITY:-}" ]]; then
  XAUTHORITY="/run/user/$(id -u)/dcv/g1.xauth"
fi
export XAUTHORITY
export DCV_XAUTHORITY="${XAUTHORITY}"

if [[ ! -f "${XAUTHORITY}" ]]; then
  echo "Xauthority file not found: ${XAUTHORITY}"
  exit 1
fi

cd "${PROJECT_ROOT}"
exec ./scripts/run_playback_dcv.sh "${TASK_ID}" motion
