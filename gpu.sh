#!/bin/sh
set -eu
umask 077

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REMOTE_GPU_VENV=${REMOTE_GPU_VENV:-"$HOME/Library/Caches/remote-gpu/venv"}
export REMOTE_GPU_VENV

if [ "${1:-}" = setup ]; then
    python=python3
    if [ -x /opt/homebrew/bin/python3 ]; then
        python=/opt/homebrew/bin/python3
    fi
    "$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else "Python 3.13 이상이 필요하다.")'
    "$python" -m venv "$REMOTE_GPU_VENV"
    exec "$REMOTE_GPU_VENV/bin/python" -m pip --disable-pip-version-check install -q --no-cache-dir \
        -r "$root/requirements.txt" -r "$root/requirements-colab.txt"
fi

if [ ! -x "$REMOTE_GPU_VENV/bin/python" ]; then
    printf '%s\n' '먼저 gpu.sh setup으로 실행 환경을 설치할 것.' >&2
    exit 1
fi
exec "$REMOTE_GPU_VENV/bin/python" "$root/gpu_run.py" "$@"
