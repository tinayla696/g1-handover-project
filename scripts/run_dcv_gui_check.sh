#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_DISPLAY="${DISPLAY:-}"
SESSION_XAUTHORITY="${XAUTHORITY:-}"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
	set -a
	# shellcheck disable=SC1091
	source "${PROJECT_ROOT}/.env"
	set +a
fi

if [[ -n "${SESSION_DISPLAY}" ]]; then
	export DISPLAY="${SESSION_DISPLAY}"
elif [[ -z "${DISPLAY:-}" ]]; then
	export DISPLAY=:0
fi

if [[ -n "${SESSION_XAUTHORITY}" ]]; then
	export XAUTHORITY="${SESSION_XAUTHORITY}"
elif [[ -z "${XAUTHORITY:-}" ]]; then
	DCV_XAUTHORITY="/run/user/$(id -u)/dcv/gui.xauth"
	if [[ -f "${DCV_XAUTHORITY}" ]]; then
		export XAUTHORITY="${DCV_XAUTHORITY}"
	elif [[ -f "${HOME}/.Xauthority" ]]; then
		export XAUTHORITY="${HOME}/.Xauthority"
	else
		echo "XAUTHORITY is not set and no DCV or home Xauthority file exists."
		exit 1
	fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
	if [[ -f "${HOME}/.Xauthority" ]]; then
		export XAUTHORITY="${HOME}/.Xauthority"
	else
		echo "XAUTHORITY is not set and ${HOME}/.Xauthority does not exist."
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
glxinfo -B | grep -E 'direct rendering|OpenGL vendor|OpenGL renderer|OpenGL version'

if docker ps -a --format '{{.Names}}' | grep -qx 'isaac-sim-groot'; then
	docker rm -f isaac-sim-groot >/dev/null
fi

xhost +local:root >/dev/null 2>&1 || true

echo "Starting Isaac Sim GUI popup for ${DCV_GUI_SECONDS:-30} seconds..."
docker compose --profile dcv-gui-check up --abort-on-container-exit --exit-code-from isaac-sim-dcv-gui-check isaac-sim-dcv-gui-check
