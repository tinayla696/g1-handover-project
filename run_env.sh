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
S3_LOG_ROOT="s3://${S3_BUCKET}/logs/${TASK_ID}"

export TASK_ID
export VISUAL_CHECK_STEPS="${VISUAL_CHECK_STEPS:-7200}"
export NUM_GPUS="${NUM_GPUS:-4}"

if [[ "${RUN_PROFILE}" == "train" ]]; then
	export NUM_ENVS="${NUM_ENVS:-512}"
	export TRAIN_TASK="${TRAIN_TASK:-G1-Handover-v0}"
	echo "🎛️  Distributed training: ${NUM_GPUS} GPUs × ${NUM_ENVS} envs/GPU"
fi

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
find "${LOCAL_LATEST_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
if aws s3 sync "${S3_CHECKPOINT_LATEST_DIR}" "${LOCAL_LATEST_DIR}/"; then
	echo "✅ Checkpoint sync complete"
else
	echo "ℹ️ Existing checkpoints not found for TASK_ID=${TASK_ID}; starting fresh"
fi

NESTED_LATEST_COUNT=$(find "${LOCAL_LATEST_DIR}" -mindepth 1 -type d -name latest | wc -l)
if [[ "${NESTED_LATEST_COUNT}" -gt 0 ]]; then
	echo "🧼 latest ディレクトリを正規化します..."
	while IFS= read -r nested_dir; do
		while IFS= read -r nested_file; do
			file_name=$(basename "${nested_file}")
			target_file="${LOCAL_LATEST_DIR}/${file_name}"
			if [[ ! -e "${target_file}" || "${nested_file}" -nt "${target_file}" ]]; then
				cp -f "${nested_file}" "${target_file}"
			fi
		done < <(find "${nested_dir}" -type f)
	done < <(find "${LOCAL_LATEST_DIR}" -mindepth 1 -type d -name latest)
	find "${LOCAL_LATEST_DIR}" -mindepth 1 -type d -name latest -exec rm -rf {} +
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
S3_LOG_DIR="${S3_LOG_ROOT}/${TIMESTAMP}/"
S3_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/${TASK_ID}/latest/"

echo "========================================================="
echo "☁️ 学習が完了しました。成果物を S3 へ退避します..."
echo "========================================================="

echo "📦 バックアップ作成中: ${S3_LOG_DIR}"
aws s3 sync "${LOCAL_TASK_LOG_DIR}/" "${S3_LOG_DIR}" \
	--exclude "latest/*" \
	--exclude "latest/latest/*" \
	--exclude "latest/latest/latest/*" \
	--exclude "latest/latest/latest/latest/*"

echo "🧹 古い日付ログを削除します (10日保持)"
EXPIRY_DATE=$(date -d '10 days ago' +%Y%m%d)
for run_dir in $(aws s3 ls "${S3_LOG_ROOT}/" | awk '{print $2}' | sed 's#/##' | grep -E '^[0-9]{8}_[0-9]{4}$'); do
	run_date=${run_dir%_*}
	if [[ "${run_date}" < "${EXPIRY_DATE}" ]]; then
		echo "   deleting ${S3_LOG_ROOT}/${run_dir}/"
		aws s3 rm "${S3_LOG_ROOT}/${run_dir}/" --recursive
	fi
done

echo "🔄 最新チェックポイントを更新中: ${S3_LATEST_DIR}"
aws s3 sync "${LOCAL_LATEST_DIR}/" "${S3_LATEST_DIR}" --delete

echo "🎉 S3 への同期がすべて完了しました！"