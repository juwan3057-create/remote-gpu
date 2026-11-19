#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / "gpu_smoke.py"
IRIS = "uciml/iris"
SLUG = r"[a-z0-9]+(?:[-_][a-z0-9]+)*"
VERSIONED_REF = rf"{SLUG}/{SLUG}/[1-9][0-9]*"
PUSHED = re.compile(r"^Kernel version ([1-9][0-9]*) successfully pushed\.", re.M)
STATUS = re.compile(r'^\S+ has status "(?:KernelWorkerStatus\.)?([a-z_]+)"', re.I | re.M)
ACTIVE = {"queued", "running", "newscript", "cancelrequested"}
TRANSIENT = re.compile(
    r"\b(?:429 Client Error|50[234] Server Error|ConnectionError|ReadTimeout|"
    r"ConnectTimeout|Temporary failure in name resolution|Failed to resolve)\b",
    re.I,
)
UNAVAILABLE = re.compile(
    r"^Kernel push error: (?:Maximum batch GPU session count\b|[^\n]*\bGPU quota\b)",
    re.I | re.M,
)


class RunError(Exception):
    pass


class TransientError(RunError):
    pass


class KaggleUnavailable(RunError):
    pass


def excerpt(output):
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    output = re.sub(r"\bKGAT_[A-Za-z0-9_-]+\b", "[가림]", output)
    output = re.sub(r"(https?://[^?\s\"']+)\?[^\s\"']+", r"\1?[가림]", output)
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        if secret := os.environ.get(name):
            output = output.replace(secret, "[가림]")
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-12:])[-1600:]


def log_excerpt(output):
    try:
        entries = json.loads(output)
    except ValueError:
        return excerpt(output)
    if isinstance(entries, list):
        return excerpt("\n".join(
            str(entry["data"]) for entry in entries if isinstance(entry, dict) and "data" in entry
        ))
    return excerpt(output)


def cli_executable(name):
    cache = Path(os.environ.get("REMOTE_GPU_VENV", Path.home() / "Library/Caches/remote-gpu/venv"))
    for venv in (cache, ROOT / ".venv"):
        executable = venv / "bin" / name
        if os.access(executable, os.X_OK):
            return str(executable)
    return shutil.which(name)


def provider_command(name):
    if not cli_executable(name):
        raise RunError("gpu.sh setup으로 CLI를 설치할 것.")
    return [sys.executable, str(ROOT / "provider_cli.py"), name]


def kaggle(*args, timeout=45):
    executable = cli_executable("kaggle")
    if not executable:
        raise RunError("Kaggle CLI가 필요하다. gpu.sh setup을 실행할 것.")
    try:
        result = subprocess.run(
            [*provider_command("kaggle"), *args], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientError("Kaggle CLI 요청 시간 초과") from exc
    if result.returncode:
        if args[:2] == ("kernels", "push") and UNAVAILABLE.search(result.stdout):
            raise KaggleUnavailable(excerpt(result.stdout))
        error_type = TransientError if TRANSIENT.search(result.stdout) else RunError
        raise error_type(excerpt(result.stdout) or f"CLI 종료 코드 {result.returncode}")
    return result.stdout


def submit(owner, source, datasets, gpu_seconds):
    # 실행마다 별도 노트북을 써서 동시 실행이나 이전 결과를 섞지 않는다.
    slug = f"codex-gpu-{uuid4().hex[:12]}"
    notebook = f"{owner}/{slug}"
    url = f"https://www.kaggle.com/code/{notebook}"
    print(f"제출 대상 {url}", flush=True)
    with tempfile.TemporaryDirectory(prefix="kaggle-gpu-") as directory:
        path = Path(directory)
        code = path / source.name
        shutil.copyfile(source, code)
        metadata = {
            "id": notebook,
            "title": slug,
            "code_file": code.name,
            "language": "python",
            "kernel_type": "notebook" if source.suffix == ".ipynb" else "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": datasets,
        }
        (path / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        try:
            output = kaggle(
                "kernels", "push", "-p", directory, "--accelerator", "NvidiaTeslaT4",
                "--timeout", str(gpu_seconds), timeout=90,
            )
        except TransientError as exc:
            raise RunError(f"제출 응답을 받지 못했다. 재제출 전 {url} 확인.\n{exc}") from exc
    version = PUSHED.search(output)
    if not version:
        if UNAVAILABLE.search(output):
            raise KaggleUnavailable(excerpt(output))
        raise RunError(f"제출 버전을 확인하지 못했다. {url}\n{excerpt(output)}")
    if "could not be added to the kernel" in output:
        print(excerpt(output), file=sys.stderr)
    return f"{notebook}/{version[1]}"


def watch(ref, wait_seconds):
    url = "https://www.kaggle.com/code/" + ref.rsplit("/", 1)[0]
    print(f"실행 {ref}\n상태 조회 재개: python3 kaggle_run.py watch {ref}", flush=True)
    deadline = time.monotonic() + wait_seconds
    delay = 5
    failures = 0
    previous = ""
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            output = kaggle("kernels", "status", ref, timeout=min(45, remaining))
            status = STATUS.search(output)
            if not status:
                raise RunError(f"상태 응답을 해석할 수 없다.\n{excerpt(output)}")
        except TransientError as exc:
            failures += 1
            if failures >= 3:
                raise RunError(f"상태 조회가 연속 3회 실패했다.\n{exc}") from exc
        else:
            failures = 0
            state = status[1].lower().replace("_", "")
            if state != previous:
                print(f"상태 {state}", flush=True)
                previous = state
            if state == "complete":
                print(f"원격 결과 {url}/output", flush=True)
            if state not in ACTIVE:
                if state != "complete":
                    print(excerpt(output), file=sys.stderr)
                if (remaining := deadline - time.monotonic()) > 0:
                    try:
                        logs = log_excerpt(kaggle("kernels", "logs", ref, timeout=min(30, remaining)))
                        if logs:
                            print(logs, flush=True)
                    except RunError as exc:
                        print(f"로그 조회 실패. {exc}", file=sys.stderr)
                return 0 if state == "complete" else 1
        time.sleep(max(0, min(delay, deadline - time.monotonic())))
        delay = min(delay * 2, 30)
    print(f"조회 시간 초과. 원격 상태 {url}", file=sys.stderr)
    return 124


def bounded_seconds(value):
    seconds = int(value)
    if not 1 <= seconds <= 43200:
        raise argparse.ArgumentTypeError("시간은 1~43200초여야 한다.")
    return seconds


def main():
    parser = argparse.ArgumentParser(description="Kaggle 비공개 GPU 작업 제출 및 조회")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="새 GPU 실행 제출 및 완료 대기")
    run.add_argument("owner", help="Kaggle 프로필 사용자명")
    run.add_argument("--source", type=Path, default=SMOKE, help="단일 .py 또는 .ipynb")
    run.add_argument("--dataset", action="append", default=[], help="Kaggle 데이터셋 owner/slug")
    run.add_argument("--gpu-seconds", type=bounded_seconds, default=300, help="원격 실행 상한")
    resume = commands.add_parser("watch", help="기존 실행 버전 상태 조회 재개")
    resume.add_argument("ref", help="owner/slug/version")
    for command in (run, resume):
        command.add_argument("--wait-seconds", type=bounded_seconds, default=900, help="로컬 조회 상한")
    args = parser.parse_args()

    if args.command == "watch":
        if not re.fullmatch(VERSIONED_REF, args.ref):
            parser.error("실행 버전은 owner/slug/version 형식이어야 한다.")
        return watch(args.ref, args.wait_seconds)

    if not re.fullmatch(SLUG, args.owner):
        parser.error("Kaggle 프로필의 사용자명을 지정할 것.")
    if args.source.suffix not in {".py", ".ipynb"} or not args.source.is_file():
        parser.error("기존 .py 또는 .ipynb 파일을 지정할 것.")
    if any(not re.fullmatch(rf"{SLUG}/{SLUG}", dataset) for dataset in args.dataset):
        parser.error("데이터셋은 owner/slug 형식이어야 한다.")
    datasets = list(args.dataset)
    if args.source.resolve() == SMOKE and IRIS not in datasets:
        datasets.append(IRIS)
    ref = submit(args.owner, args.source, datasets, args.gpu_seconds)
    return watch(ref, args.wait_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RunError, OSError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
