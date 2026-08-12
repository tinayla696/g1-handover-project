#!/bin/bash
set -e

S3_BUCKET="g1-gr00t-models-380421147972-us-east-1-an"
TASK_ID="${1:-g1_handover_teacher}"
PLAYBACK_MODE="${2:-policy}"

# Normalize TASK_ID: convert hyphens to underscores for directory compatibility
TASK_ID="${TASK_ID//-/_}"

if [[ -z "${TASK_ID}" ]]; then
	echo "❌ TASK_ID must not be empty"
	echo "Usage: ./scripts/run_playback_desktop.sh [task_id] [policy|motion]"
	exit 1
fi

if [[ "${PLAYBACK_MODE}" != "policy" && "${PLAYBACK_MODE}" != "motion" ]]; then
	echo "❌ Invalid PLAYBACK_MODE: ${PLAYBACK_MODE}"
	echo "Usage: ./scripts/run_playback_desktop.sh [task_id] [policy|motion]"
	exit 1
fi

LOCAL_TASK_LOG_DIR="./logs/${TASK_ID}"
LOCAL_LATEST_DIR="${LOCAL_TASK_LOG_DIR}/latest"
S3_CHECKPOINT_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

find_latest_checkpoint() {
	find "$1" -type f -name 'model_*.pt' | sort -V | tail -n 1
}

export TASK_ID
export PLAYBACK_MODE

if [[ ! -d "${LOCAL_LATEST_DIR}" ]]; then
	mkdir -p "${LOCAL_LATEST_DIR}"
fi

LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "☁️  S3 からタスク別最新チェックポイントをホスト側に同期します..."
	find "${LOCAL_LATEST_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
	aws s3 sync "${S3_CHECKPOINT_LATEST_DIR}" "${LOCAL_LATEST_DIR}/" --no-progress
	LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
fi

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "❌ Could not determine a checkpoint file under ${LOCAL_TASK_LOG_DIR}"
	exit 1
fi

export PLAYBACK_CHECKPOINT="${LATEST_CHECKPOINT}"

echo "🧩 Task ID: ${TASK_ID}"
echo "🎯 Playback checkpoint: ${PLAYBACK_CHECKPOINT}"
echo "🎞️ Playback mode: ${PLAYBACK_MODE}"
echo "🖥️ Desktop playback mode: requires active DCV/X11 session"

if [[ ! -x ./scripts/download_motions.sh ]]; then
	chmod +x ./scripts/download_motions.sh
fi

./scripts/download_motions.sh

if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
	echo "🧹 stale container 'isaac-sim-groot' を削除します..."
	docker rm -f isaac-sim-groot >/dev/null
fi

# Allow local root in container to open the host X display.
if command -v xhost >/dev/null 2>&1; then
	xhost +local:root >/dev/null 2>&1 || true
fi

echo "🐳 Docker Compose を使ってDesktop再生環境を起動します... (profile: playback-desktop)"
docker compose --profile playback-desktop up --force-recreate --no-build
