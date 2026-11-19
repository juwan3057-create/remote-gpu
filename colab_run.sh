#!/bin/sh
set -eu
umask 077

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="${REMOTE_GPU_VENV:-"$HOME/Library/Caches/remote-gpu/venv"}/bin/python"
if [ ! -x "$python" ]; then
    python="$root/.venv/bin/python"
fi
if [ ! -x "$python" ]; then
    printf '%s\n' 'Google Colab CLI가 필요하다. README의 Colab 설치 절차를 실행할 것.' >&2
    exit 1
fi

exec "$python" "$root/colab_run.py" "$@"
