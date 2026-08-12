#!/bin/bash
set -e

echo "========================================================="
echo "🐍 0. Isaac Sim 内蔵 Python への透過ラッパーを生成します..."
echo "========================================================="
mkdir -p /tmp/bin

cat << 'EOF' > /tmp/bin/python3
#!/bin/bash
exec /isaac-sim/python.sh "$@"
EOF

# Headless container shim: some optional UI code paths try to invoke zenity.
# Return non-zero to indicate "dialog unavailable" without noisy "not found" logs.
cat << 'EOF' > /tmp/bin/zenity
#!/bin/bash
exit 1
EOF

chmod +x /tmp/bin/python3
chmod +x /tmp/bin/zenity

export PATH="/tmp/bin:$PATH"

echo "現在の環境変数 PATH: $PATH"
echo "python3 の実体パス: $(which python3)"

if ! command -v git >/dev/null 2>&1; then
	apt-get update
	apt-get install -y --no-install-recommends git
fi

export GIT_PYTHON_GIT_EXECUTABLE="$(command -v git)"

SETUP_CACHE_DIR="/tmp"
SETUP_STAMP_FILE="${SETUP_CACHE_DIR}/isaaclab_setup_v1.done"

if [[ -f "${SETUP_STAMP_FILE}" ]]; then
	echo "========================================================="
	echo "♻️  Cached setup detected. Skipping dependency installation."
	echo "   Stamp: ${SETUP_STAMP_FILE}"
	echo "========================================================="
	exec "$@"
fi

mkdir -p "${SETUP_CACHE_DIR}"

echo "========================================================="
echo "🚀 1. G1 学習に必要な最小限の Isaac Lab 拡張のみをインストールします..."
echo "========================================================="
cd /workspace/IsaacLab

# Update packaging tools once, then install only the modules required for headless RSL-RL training.
python3 -m pip install --upgrade pip "setuptools<82.0.0"

# Keep Isaac Lab aligned with Isaac Sim 6.0.1's tested CUDA/PyTorch stack and
# avoid reinstalling optional extras that fight each other.
python3 -m pip install \
	--extra-index-url https://download.pytorch.org/whl/cu128 \
	--extra-index-url https://pypi.nvidia.com \
	"torch==2.10.0+cu128" \
	"torchvision==0.25.0+cu128" \
	"triton==3.6.0" \
	"cuda-bindings==12.9.4" \
	"cuda-pathfinder==1.2.2" \
	"nvidia-cuda-nvrtc-cu12==12.8.93" \
	"nvidia-cuda-runtime-cu12==12.8.90" \
	"nvidia-cuda-cupti-cu12==12.8.90" \
	"nvidia-cudnn-cu12==9.10.2.21" \
	"nvidia-cublas-cu12==12.8.4.1" \
	"nvidia-cufft-cu12==11.3.3.83" \
	"nvidia-cufile-cu12==1.13.1.3" \
	"nvidia-curand-cu12==10.3.9.90" \
	"nvidia-cusolver-cu12==11.7.3.90" \
	"nvidia-cusparse-cu12==12.5.8.93" \
	"nvidia-cusparselt-cu12==0.7.1" \
	"nvidia-nccl-cu12==2.27.5" \
	"nvidia-nvjitlink-cu12==12.8.93" \
	"nvidia-nvshmem-cu12==3.4.5" \
	"nvidia-nvtx-cu12==12.8.90"

python3 -m pip install \
	"numpy>=2" \
	"onnx>=1.18.0" \
	"prettytable==3.3.0" \
	"toml" \
	"hidapi==0.14.0.post2" \
	"gymnasium==1.2.1" \
	"trimesh" \
	"pyglet>=2.1.6,<3" \
	"transformers==4.57.6" \
	"einops" \
	"warp-lang==1.13.0" \
	"matplotlib>=3.10.3" \
	"pillow>=12.1.1" \
	"botocore" \
	"starlette>=0.46.0,<0.50" \
	"omniverseclient==2.71.1.7015" \
	"coverage==7.6.1" \
	"debugpy>=1.8.20" \
	"flatdict>=4.1.0" \
	"flaky" \
	"packaging" \
	"psutil==5.9.8" \
	"filelock" \
	"typing_extensions==4.12.2" \
	"pydantic>=2.7,<2.12" \
	"lazy_loader>=0.4" \
	"pin-pink==3.3.0" \
	"daqp==0.8.5" \
	"usd-core>=25.11,<26.0" \
	"usd-exchange>=2.2" \
	"pytetwild==0.2.3" \
	"hf-xet>=1.4.1,<2.0.0"

python3 -m pip install \
	"hydra-core" \
	"h5py>=3.16.0" \
	"tensorboard" \
	"moviepy>=1.0.3,<2.0.0.dev0" \
	"tqdm==4.67.1" \
	"rsl-rl-lib==5.0.1" \
	"onnxscript>=0.5"

python3 -m pip install --no-deps -e source/isaaclab
python3 -m pip install --no-deps -e source/isaaclab_assets
python3 -m pip install --no-deps -e source/isaaclab_rl

echo "========================================================="
echo "🤖 2. g1-handover-project を Python 拡張として登録します..."
echo "========================================================="
cd /workspace/g1-handover-project
python3 -m pip install -e .

echo "========================================================="
echo "🎉 3. 環境構築完了。強化学習をキックします..."
echo "========================================================="
touch "${SETUP_STAMP_FILE}"
exec "$@"