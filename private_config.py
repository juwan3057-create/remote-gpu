"""개인 설정과 OAuth 자격증명을 단일 비공개 파일로 관리한다."""
import fcntl
import json
import os
from pathlib import Path
import tempfile


def config_path():
    return Path(os.environ.get("REMOTE_GPU_CONFIG", Path.cwd() / "gpu.local.json")).expanduser().resolve()


def load_config():
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_credentials(provider, value):
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 경로 자체를 잠그면 원자적 교체 때 잠금이 풀린다. 별도 잠금 파일에는 비밀이 없다.
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = load_config()
        config.setdefault("auth", {})[provider] = value
        fd, temporary = tempfile.mkstemp(prefix=".gpu-private-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
