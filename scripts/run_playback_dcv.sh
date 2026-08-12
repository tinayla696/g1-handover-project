#!/bin/bash
set -e

S3_BUCKET="g1-gr00t-models-380421147972-us-east-1-an"
TASK_ID="${1:-g1_handover_teacher}"
PLAYBACK_MODE="${2:-policy}"

TASK_ID="${TASK_ID//-/_}"

if [[ "${PLAYBACK_MODE}" != "policy" && "${PLAYBACK_MODE}" != "motion" ]]; then
	echo "Invalid PLAYBACK_MODE: ${PLAYBACK_MODE}"
	echo "Usage: ./scripts/run_playback_dcv.sh [task_id] [policy|motion]"
	exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
	echo "DISPLAY is not set. Run this script from a terminal inside the DCV desktop session."
	exit 1
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
	DCV_XAUTHORITY="/run/user/$(id -u)/dcv/gui.xauth"
	if [[ -f "${DCV_XAUTHORITY}" ]]; then
		export XAUTHORITY="${DCV_XAUTHORITY}"
	fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
	if command -v xauth >/dev/null 2>&1; then
		XAUTHORITY=$(xauth info 2>/dev/null | sed -n 's/^Authority file: //p' | head -n 1)
		export XAUTHORITY
	fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
	if [[ -f "${HOME}/.Xauthority" ]]; then
		export XAUTHORITY="${HOME}/.Xauthority"
	else
		echo "XAUTHORITY is not set and ${HOME}/.Xauthority does not exist."
		echo "Run this script from the DCV desktop session, then check: echo \"$DISPLAY\""
		exit 1
	fi
fi

if [[ ! -f "${XAUTHORITY}" ]]; then
	echo "Xauthority file not found: ${XAUTHORITY}"
	exit 1
fi

export DCV_XAUTHORITY="${XAUTHORITY}"

echo "DCV X display: ${DISPLAY}"
echo "DCV Xauthority: ${XAUTHORITY}"
if command -v xdpyinfo >/dev/null 2>&1; then
	if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
		echo "Cannot access X display ${DISPLAY} from this DCV terminal."
		echo "Do not run from SSH; open a terminal inside the DCV desktop session."
		exit 1
	fi
fi

LOCAL_TASK_LOG_DIR="./logs/${TASK_ID}"
LOCAL_LATEST_DIR="${LOCAL_TASK_LOG_DIR}/latest"
S3_CHECKPOINT_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

find_latest_checkpoint() {
	find "$1" -type f -name 'model_*.pt' | sort -V | tail -n 1
}

export TASK_ID PLAYBACK_MODE
mkdir -p "${LOCAL_LATEST_DIR}"

LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "S3 から最新チェックポイントを同期します..."
	find "${LOCAL_LATEST_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
	aws s3 sync "${S3_CHECKPOINT_LATEST_DIR}" "${LOCAL_LATEST_DIR}/" --no-progress
	LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
fi

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "Checkpoint not found under ${LOCAL_TASK_LOG_DIR}"
	exit 1
fi

export PLAYBACK_CHECKPOINT="${LATEST_CHECKPOINT}"

echo "DCV display: ${DISPLAY}"
echo "Task ID: ${TASK_ID}"
echo "Checkpoint: ${PLAYBACK_CHECKPOINT}"
echo "Playback mode: ${PLAYBACK_MODE}"
echo "WebRTC: disabled; rendering through DCV X11 display"

./scripts/download_motions.sh

if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
	docker rm -f isaac-sim-groot >/dev/null
fi

# Permit the container's root user to connect to the DCV X server.
if command -v xhost >/dev/null 2>&1; then
	xhost +local:root >/dev/null 2>&1 || true
fi

echo "Starting Isaac Sim playback in DCV..."
echo "Checking container GPU/X11 prerequisites..."
docker run --rm --gpus all --network host \
	--entrypoint /bin/bash \
	--user "$(id -u):$(id -g)" \
	-e "DISPLAY=${DISPLAY}" \
	-e "XAUTHORITY=/tmp/dcv.gui.xauth" \
	-e "NVIDIA_DRIVER_CAPABILITIES=all" \
	-e "__GLX_VENDOR_LIBRARY_NAME=nvidia" \
	-v "${XAUTHORITY}:/tmp/dcv.gui.xauth:ro" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	nvcr.io/nvidia/isaac-sim:6.0.1 \
	bash -lc 'if command -v glxinfo >/dev/null 2>&1; then glxinfo -B; else echo "glxinfo is not available in the Isaac Sim image"; fi' || true
docker compose --profile playback-dcv up --force-recreate --no-build
