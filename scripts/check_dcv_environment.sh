#!/bin/bash
set -e

if [[ -z "${DISPLAY:-}" ]]; then
	export DISPLAY=:0
	echo "DISPLAY was unset; using ${DISPLAY}"
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
	DCV_XAUTHORITY="/run/user/$(id -u)/dcv/gui.xauth"
	if [[ -f "${DCV_XAUTHORITY}" ]]; then
		export XAUTHORITY="${DCV_XAUTHORITY}"
	elif [[ -f "${HOME}/.Xauthority" ]]; then
		export XAUTHORITY="${HOME}/.Xauthority"
	else
		echo "Xauthority file not found."
		exit 1
	fi
fi

if [[ ! -f "${XAUTHORITY}" ]]; then
	echo "Xauthority file not found: ${XAUTHORITY}"
	exit 1
fi

export DCV_XAUTHORITY="${XAUTHORITY}"

echo "DISPLAY=${DISPLAY}"
echo "XAUTHORITY=${XAUTHORITY}"
xdpyinfo -display "${DISPLAY}" >/dev/null
echo "Host X11: OK"

glxinfo -B | grep -E 'direct rendering|OpenGL vendor|OpenGL renderer|OpenGL version'

echo "Host NVIDIA OpenGL: OK"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "Host NVIDIA driver: OK"

docker run --rm --gpus all --entrypoint /bin/bash \
	-e "DISPLAY=${DISPLAY}" \
	-e "XAUTHORITY=/root/.Xauthority" \
	-e "NVIDIA_DRIVER_CAPABILITIES=all" \
	-v "${XAUTHORITY}:/root/.Xauthority:ro" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	nvcr.io/nvidia/isaac-sim:6.0.1 \
	-lc 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; test -r /root/.Xauthority; test -S "/tmp/.X11-unix/X${DISPLAY#:}"'

echo "Container GPU/X11 mounts: OK"

echo "DCV environment is ready for: ./scripts/run_playback_dcv.sh g1_handover_teacher policy"
