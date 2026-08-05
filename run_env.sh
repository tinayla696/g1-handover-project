#!/bin/bash
set -e

# S3管理設定
S3_BUCKET="g1-gr00t-models-380421147972-us-east-1-an"
RUN_PROFILE="${1:-train}"
TASK_ID="${2:-g1_handover_base}"

if [[ "${RUN_PROFILE}" != "train" && "${RUN_PROFILE}" != "visual-check" ]]; then
	echo "❌ Invalid profile: ${RUN_PROFILE}"
	echo "Usage: ./run_env.sh [train|visual-check] [task_id]"
	exit 1
fi

if [[ -z "${TASK_ID}" ]]; then
	echo "❌ TASK_ID must not be empty"
	echo "Usage: ./run_env.sh [train|visual-check] [task_id]"
	exit 1
fi

LOCAL_TASK_LOG_DIR="./logs/${TASK_ID}"
LOCAL_LATEST_DIR="${LOCAL_TASK_LOG_DIR}/latest"
S3_CHECKPOINT_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

# パブリックIPを動的に取得して環境変数にセット
export PUBLIC_IP=$(curl -s ifconfig.me)
export TASK_ID
export VISUAL_CHECK_STEPS="${VISUAL_CHECK_STEPS:-7200}"

echo "🌍 Public IP Detected: ${PUBLIC_IP}"
echo "🧩 Task ID: ${TASK_ID}"
if [[ "${RUN_PROFILE}" == "visual-check" ]]; then
	echo "🕒 Visual check steps: ${VISUAL_CHECK_STEPS}"
fi
echo "🌟 GitHub から最新のソースコードをプルします..."
git pull origin main || echo "Git pull skipped (not a tracking branch yet)"

if [[ ! -x ./scripts/download_motions.sh ]]; then
	chmod +x ./scripts/download_motions.sh
fi

echo "🎬 参照モーションを同期します..."
./scripts/download_motions.sh

echo "☁️  S3 からタスク別最新チェックポイントをホスト側に同期します..."
mkdir -p "${LOCAL_LATEST_DIR}"
if aws s3 sync "${S3_CHECKPOINT_LATEST_DIR}" "${LOCAL_LATEST_DIR}/"; then
	echo "✅ Checkpoint sync complete"
else
	echo "ℹ️ Existing checkpoints not found for TASK_ID=${TASK_ID}; starting fresh"
fi

# 以前の実行で残った同名コンテナを除去して、name conflictを防ぐ
if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
	echo "🧹 stale container 'isaac-sim-groot' を削除します..."
	docker rm -f isaac-sim-groot >/dev/null
fi

echo "🐳 Docker Compose を使ってポータブル環境を起動します... (profile: ${RUN_PROFILE})"
docker compose --profile "${RUN_PROFILE}" up --force-recreate --no-build

# 実行ログの保存先を実行ごとに切り分け
TIMESTAMP=$(date +"%Y%m%d_%H%M")
S3_LOG_DIR="s3://${S3_BUCKET}/logs/${TASK_ID}/${TIMESTAMP}/"
S3_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

echo "========================================================="
echo "☁️ 学習が完了しました。成果物を S3 へ退避します..."
echo "========================================================="

echo "📦 バックアップ作成中: ${S3_LOG_DIR}"
aws s3 sync "${LOCAL_TASK_LOG_DIR}/" "${S3_LOG_DIR}"

echo "🔄 最新チェックポイントを更新中: ${S3_LATEST_DIR}"
aws s3 sync "${LOCAL_TASK_LOG_DIR}/" "${S3_LATEST_DIR}" --delete

echo "🎉 S3 への同期がすべて完了しました！"