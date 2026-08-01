#!/bin/bash
set -e

# S3管理設定
S3_BUCKET="g1-gr00t-models-380421147972-us-east-1-an"
TASK_NAME="g1-handover-novelty"

# パブリックIPを動的に取得して環境変数にセット
export PUBLIC_IP=$(curl -s ifconfig.me)

echo "🌍 Public IP Detected: ${PUBLIC_IP}"
echo "🌟 GitHub から最新のソースコードをプルします..."
git pull origin main || echo "Git pull skipped (not a tracking branch yet)"

echo "☁️  S3 から最新の学習チェックポイントをホスト側に同期します..."
mkdir -p ./logs
# ホスト側のAWS権限を使って安全に事前ダウンロード
aws s3 sync "s3://${S3_BUCKET}/checkpoints/latest/" ./logs/

echo "🐳 Docker Compose を使ってポータブル環境を起動します..."
docker compose up --force-recreate --no-build

# 実行ログの保存先を実行ごとに切り分け
TIMESTAMP=$(date +"%Y%m%d_%H%M")
S3_LOG_DIR="s3://${S3_BUCKET}/logs/${TASK_NAME}/${TIMESTAMP}/"
S3_LATEST_DIR="s3://${S3_BUCKET}/checkpoints/latest/"

echo "========================================================="
echo "☁️ 学習が完了しました。成果物を S3 へ退避します..."
echo "========================================================="

echo "📦 バックアップ作成中: ${S3_LOG_DIR}"
aws s3 sync ./logs/ "${S3_LOG_DIR}"

echo "🔄 最新チェックポイントを更新中: ${S3_LATEST_DIR}"
aws s3 sync ./logs/ "${S3_LATEST_DIR}" --delete

echo "🎉 S3 への同期がすべて完了しました！"