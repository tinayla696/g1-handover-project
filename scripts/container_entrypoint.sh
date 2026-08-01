#!/bin/bash
set -e

echo "========================================================="
echo "🚀 1. Isaac Lab の自動インストールを開始します..."
echo "========================================================="
cd /workspace/IsaacLab
./isaaclab.sh --install

echo "========================================================="
echo "🤖 2. g1-handover-project を Python 拡張として登録します..."
echo "========================================================="
cd /workspace/g1-handover-project
# 自作の拡張タスク群をIsaac Lab環境のPythonに認識させる
python -m pip install -e .

echo "========================================================="
echo "☁️ 3. S3 から最新の学習チェックポイントをロードします..."
echo "========================================================="
if [ -d "/workspace/g1-handover-project/logs" ]; then
    # バケット名は確定したものを使用
    aws s3 sync s3://g1-gr00t-models-380421147972-us-east-1-an/latest_logs /workspace/g1-handover-project/logs/
else
    mkdir -p /workspace/g1-handover-project/logs
    aws s3 sync s3://g1-gr00t-models-380421147972-us-east-1-an/latest_logs /workspace/g1-handover-project/logs/
fi

echo "========================================================="
echo "🎉 4. 環境構築完了。Isaac Sim を起動します..."
echo "========================================================="
exec "$@"