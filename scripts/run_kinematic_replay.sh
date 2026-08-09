#!/bin/bash
set -e

export PUBLIC_IP=$(curl -s ifconfig.me)
export TASK_ID="${1:-g1_handover_teacher}"

echo "🌍 Public IP: ${PUBLIC_IP}"
echo "🎬 Kinematic replay (motion_replay.py / PD制御バイパス)"

./scripts/download_motions.sh

if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
    docker rm -f isaac-sim-groot >/dev/null
fi

git pull origin main || true

docker compose --profile kinematic-replay up --force-recreate --no-build
