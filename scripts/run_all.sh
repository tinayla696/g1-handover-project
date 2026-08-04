#!/bin/bash
set -e

# パブリックIPと最新リポジトリの確保
export PUBLIC_IP=$(curl -s ifconfig.me)
git pull origin main

echo "=== Isaac Sim 6.0.1 コンテナをポータブルモードで起動 ==="
docker run --name isaac-sim-groot --gpus all -it --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -v ~/IsaacLab:/workspace/IsaacLab \
  -v ~/g1-handover-project:/workspace/g1-handover-project \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  /bin/bash /workspace/g1-handover-project/scripts/container_entrypoint.sh \
  ./runheadless.sh \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp=$PUBLIC_IP \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998