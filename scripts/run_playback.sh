#!/bin/bash
set -e

S3_BUCKET="g1-gr00t-models-380421147972-us-east-1-an"
TASK_ID="${1:-g1_handover_teacher}"
PLAYBACK_MODE="${2:-motion}"

# Normalize TASK_ID: convert hyphens to underscores for directory compatibility
TASK_ID="${TASK_ID//-/_}"

if [[ -z "${TASK_ID}" ]]; then
	echo "❌ TASK_ID must not be empty"
	echo "Usage: ./scripts/run_playback.sh [task_id] [policy|motion]"
	exit 1
fi

if [[ "${PLAYBACK_MODE}" != "policy" && "${PLAYBACK_MODE}" != "motion" ]]; then
	echo "❌ Invalid PLAYBACK_MODE: ${PLAYBACK_MODE}"
	echo "Usage: ./scripts/run_playback.sh [task_id] [policy|motion]"
	exit 1
fi

LOCAL_TASK_LOG_DIR="./logs/${TASK_ID}"
LOCAL_LATEST_DIR="${LOCAL_TASK_LOG_DIR}/latest"
S3_CHECKPOINT_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

find_latest_checkpoint() {
	find "$1" -type f -name 'model_*.pt' | sort -V | tail -n 1
}

export PUBLIC_IP=$(curl -s ifconfig.me)
export TASK_ID

if [[ ! -d "${LOCAL_LATEST_DIR}" ]]; then
	mkdir -p "${LOCAL_LATEST_DIR}"
fi

LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "☁️  S3 からタスク別最新チェックポイントをホスト側に同期します..."
	if [[ ! -d "${LOCAL_LATEST_DIR}" ]]; then
		mkdir -p "${LOCAL_LATEST_DIR}"
	fi
	find "${LOCAL_LATEST_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
	if aws s3 sync "${S3_CHECKPOINT_LATEST_DIR}" "${LOCAL_LATEST_DIR}/" --no-progress; then
		echo "✅ Checkpoint sync complete"
	else
		echo "❌ S3 sync failed for TASK_ID=${TASK_ID}"
		echo "   S3 path: ${S3_CHECKPOINT_LATEST_DIR}"
		echo "   Local path: ${LOCAL_LATEST_DIR}/"
		exit 1
	fi
	LATEST_CHECKPOINT=$(find_latest_checkpoint "${LOCAL_TASK_LOG_DIR}")
fi

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
	echo "❌ Could not determine a checkpoint file under ${LOCAL_TASK_LOG_DIR}"
	echo "   Available files:"
	ls -lh "${LOCAL_TASK_LOG_DIR}" 2>/dev/null | tail -10 || echo "   (directory not found or empty)"
	exit 1
fi

export PLAYBACK_CHECKPOINT="${LATEST_CHECKPOINT}"
export PLAYBACK_MODE

echo "🌍 Public IP Detected: ${PUBLIC_IP}"
echo "🧩 Task ID: ${TASK_ID}"
echo "🎯 Playback checkpoint: ${PLAYBACK_CHECKPOINT}"
echo "🎞️ Playback mode: ${PLAYBACK_MODE}"

echo "🌟 GitHub から最新のソースコードをプルします..."
git pull origin main || echo "Git pull skipped (not a tracking branch yet)"

if [[ ! -x ./scripts/download_motions.sh ]]; then
	chmod +x ./scripts/download_motions.sh
fi

echo "🎬 参照モーションを同期します..."
./scripts/download_motions.sh

if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
	echo "🧹 stale container 'isaac-sim-groot' を削除します..."
	docker rm -f isaac-sim-groot >/dev/null
fi

echo "🐳 Docker Compose を使って再生環境を起動します... (profile: playback)"
docker compose --profile playback up --force-recreate --no-build
