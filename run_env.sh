#!/bin/bash
set -e

# パブリックIPを動的に取得して環境変数にセット
export PUBLIC_IP=$(curl -s ifconfig.me)

echo "🌍 Public IP Detected: ${PUBLIC_IP}"
echo "🌟 GitHub から最新のソースコードをプルします..."
git pull origin main || echo "Git pull skipped (not a tracking branch yet)"

echo "🐳 Docker Compose を使ってポータブル環境を起動します..."
docker compose up