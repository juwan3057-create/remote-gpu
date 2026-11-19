"""프로젝트의 설정으로 Colab GPU 작업 한 건을 실행한다."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from kaggle_run import provider_command
from private_config import load_config

ROOT = Path(__file__).resolve().parent
GPUS = ("T4", "L4", "G4", "A100", "H100")
PRESETS = {"small": ("T4", 300), "medium": ("L4", 900), "large": ("A100", 1800)}


def load_profile():
    config = load_config().get("profile", {})
    if unknown := config.keys() - {"kaggle_owner", "size", "gpu", "timeout_seconds"}:
        raise ValueError(f"지원하지 않는 설정이 있다. {', '.join(sorted(unknown))}")
    size = config.get("size", "small")
    if not isinstance(size, str) or size not in PRESETS:
        raise ValueError("size는 small, medium, large 중 하나여야 한다.")
    default_gpu, default_timeout = PRESETS[size]
    return config.get("kaggle_owner", ""), config.get("gpu", default_gpu), config.get("timeout_seconds", default_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", type=str.upper, choices=GPUS, help="이번 실행의 GPU")
    parser.add_argument("--timeout", type=int, help="이번 실행의 코드 제한 시간(초)")
    parser.add_argument("--allow-compute-units", action="store_true", help="이번 실행에 기존 컴퓨팅 유닛 사용 허용")
    parser.add_argument("script", type=Path, nargs="?", default=ROOT / "gpu_smoke.py")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        _, default_gpu, default_timeout = load_profile()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    gpu = args.gpu if args.gpu is not None else default_gpu
    timeout = args.timeout if args.timeout is not None else default_timeout
    if gpu not in GPUS:
        parser.error(f"지원 GPU는 {', '.join(GPUS)}이다.")
    if type(timeout) is not int or timeout <= 0:
        parser.error("제한 시간은 양의 정수여야 한다.")
    if not args.script.is_file():
        parser.error(f"실행할 Python 파일을 찾을 수 없다. {args.script}")
    if gpu != "T4" and not args.allow_compute_units:
        parser.error(f"{gpu} 요청에는 이번 실행의 컴퓨팅 유닛 사용 허용이 필요하다.")

    command = provider_command("colab")
    command += ["--auth", "oauth2"]
    if not args.allow_compute_units:
        usage = subprocess.run([*command, "usage"], stdout=subprocess.PIPE, text=True, timeout=45)
        if usage.returncode:
            return usage.returncode
        match = re.search(r"^Current balance: (\d+\.\d+) compute units$", usage.stdout, re.M)
        balance = match[1] if match else "확인 불가"
        if balance != "0.00":
            print(f"무료 검증은 표시 잔액이 0.00일 때 시작한다. 현재 잔액 {balance}", file=sys.stderr)
            return 1

    return subprocess.call([
        *command, "run", "--gpu", gpu, "--timeout", str(timeout),
        "--env", "GPU_SMOKE_PLATFORM=colab", "--", str(args.script), *args.script_args,
    ])


if __name__ == "__main__":
    sys.exit(main())
