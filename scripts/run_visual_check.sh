#!/bin/bash
set -e

export PUBLIC_IP=$(curl -s ifconfig.me)

echo "=== Isaac Sim 6.0.1 visual check launch ==="
docker compose --profile visual-check up --abort-on-container-exit --exit-code-from isaac-sim-visual-check isaac-sim-visual-check