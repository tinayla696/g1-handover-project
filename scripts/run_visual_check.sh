#!/bin/bash
set -e

echo "=== Isaac Sim 6.0.1 DCV visual check launch ==="
docker compose --profile visual-check up --abort-on-container-exit --exit-code-from isaac-sim-visual-check isaac-sim-visual-check