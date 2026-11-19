"""Kaggle 무료 GPU를 먼저 사용하고, 가용하지 않으면 Colab으로 전환한다."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from colab_run import GPUS, load_profile
from kaggle_run import (
    IRIS, ROOT, SLUG, SMOKE, VERSIONED_REF, KaggleUnavailable, RunError,
    bounded_seconds, provider_command, excerpt, kaggle, submit, watch,
)


def remaining_gpu_hours():
    try:
        rows = json.loads(kaggle("quota", "--format", "json", timeout=30))
        remaining = next(row["remaining"] for row in rows if row["resource"] == "GPU")
        if not re.fullmatch(r"\d+(?:\.\d+)?h", remaining):
            raise ValueError
    except RunError as error:
        raise KaggleUnavailable(f"Kaggle 할당량 조회 실패. {error}") from error
    except (ValueError, TypeError, KeyError, StopIteration) as error:
        raise KaggleUnavailable("Kaggle GPU 잔여량을 확인할 수 없다.") from error
    return Decimal(remaining[:-1])


def run_colab(source, datasets, gpu, timeout, allow_compute_units):
    if source.suffix != ".py":
        raise RunError("Colab 전환에는 단일 Python 파일이 필요하다.")
    print(f"Colab {gpu} 실행 요청", flush=True)
    with tempfile.TemporaryDirectory(prefix="colab-gpu-") as directory:
        wrapped = Path(directory) / source.name
        code = source.read_text(encoding="utf-8-sig")
        prelude = ""
        if datasets:
            prelude = f"""\
import os
from pathlib import Path
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub>=1.0.2,<2"], check=True, timeout=120)
import kagglehub

input_dir = Path("/content/kaggle-input")
for dataset in {datasets!r}:
    dest = input_dir / "datasets" / dataset
    dest.parent.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(dataset, output_dir=str(dest))
os.environ["GPU_INPUT_DIR"] = str(input_dir)
"""
        wrapped.write_text(prelude + f"__file__ = {source.name!r}\nexec(compile({code!r}, __file__, 'exec'))\n", encoding="utf-8")
        command = [sys.executable, str(ROOT / "colab_run.py"), "--gpu", gpu, "--timeout", str(timeout)]
        if allow_compute_units:
            command.append("--allow-compute-units")
        result = subprocess.run(
            [*command, str(wrapped)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        )
        print(excerpt(result.stdout), flush=True)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="학습 제출과 완료 확인", allow_abbrev=False)
    run.add_argument("source", type=Path, help="단일 .py 또는 Kaggle용 .ipynb")
    run.add_argument("--provider", choices=("auto", "kaggle", "colab"), default="auto")
    run.add_argument("--dataset", action="append", default=[], metavar="OWNER/SLUG")
    run.add_argument("--timeout", type=bounded_seconds, help="원격 코드 실행 상한(초)")
    run.add_argument("--colab-gpu", type=str.upper, choices=GPUS)
    run.add_argument("--allow-compute-units", action="store_true", help="이번 Colab 실행에 기존 유닛 사용 허용")
    run.add_argument("--wait-seconds", type=bounded_seconds, help="Kaggle 상태 조회 상한(초)")
    for name, help_text in (("watch", "Kaggle 상태 조회 재개"), ("outputs", "Kaggle 결과 파일 목록")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("ref", metavar="OWNER/SLUG/VERSION")
        if name == "watch":
            command.add_argument("--wait-seconds", type=bounded_seconds, default=900)
    commands.add_parser("quota", help="Kaggle GPU 잔여량")
    commands.add_parser("sessions", help="Colab 활성 세션 조회")
    auth = commands.add_parser("auth", help="공급자 로그인")
    auth.add_argument("provider", choices=("kaggle", "colab"))
    args = parser.parse_args()

    if args.command == "quota":
        print(f"Kaggle GPU 잔여량 {remaining_gpu_hours():f}시간")
        return 0
    if args.command in {"watch", "outputs"}:
        if not re.fullmatch(VERSIONED_REF, args.ref):
            parser.error("실행 버전은 owner/slug/version 형식이어야 한다.")
        if args.command == "watch":
            return watch(args.ref, args.wait_seconds)
        print(kaggle("kernels", "files", args.ref), end="")
        return 0
    if args.command in {"auth", "sessions"}:
        provider = args.provider if args.command == "auth" else "colab"
        command = provider_command(provider)
        if provider == "kaggle":
            return subprocess.call([*command, "auth", "login"])
        env = {**os.environ, "OAUTHLIB_RELAX_TOKEN_SCOPE": "1"} if args.command == "auth" else os.environ
        return subprocess.call([*command, "--auth", "oauth2", "sessions"], env=env)

    owner, profile_gpu, profile_timeout = load_profile()
    timeout = args.timeout if args.timeout is not None else profile_timeout
    gpu = args.colab_gpu or profile_gpu
    if type(timeout) is not int or not 1 <= timeout <= 43200:
        parser.error("제한 시간은 1~43200초여야 한다.")
    if gpu not in GPUS:
        parser.error(f"지원 Colab GPU는 {', '.join(GPUS)}이다.")
    if args.source.suffix not in {".py", ".ipynb"} or not args.source.is_file():
        parser.error("기존 .py 또는 .ipynb 파일을 지정할 것.")
    if any(not re.fullmatch(rf"{SLUG}/{SLUG}", dataset) for dataset in args.dataset):
        parser.error("데이터셋은 owner/slug 형식이어야 한다.")
    datasets = list(dict.fromkeys(args.dataset))
    if args.source.resolve() == SMOKE and IRIS not in datasets:
        datasets.append(IRIS)

    if args.provider != "colab":
        if not isinstance(owner, str) or not re.fullmatch(SLUG, owner):
            parser.error("gpu.local.json의 profile에 kaggle_owner를 지정할 것.")
        try:
            remaining = remaining_gpu_hours()
            if remaining * 3600 < timeout:
                raise KaggleUnavailable(f"Kaggle 잔여량 {remaining:f}시간으로 요청한 실행 한도 {timeout}초를 충족할 수 없다.")
            ref = submit(owner, args.source, datasets, timeout)
        except KaggleUnavailable as error:
            if args.provider == "kaggle":
                raise
            print(error, file=sys.stderr, flush=True)
        else:
            wait_seconds = args.wait_seconds or min(43200, timeout + 600)
            return watch(ref, wait_seconds)

    return run_colab(args.source, datasets, gpu, timeout, args.allow_compute_units)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RunError, subprocess.SubprocessError) as error:
        print(excerpt(str(error)), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
