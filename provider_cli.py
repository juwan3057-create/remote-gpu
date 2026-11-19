"""CLI의 인증 파일을 임시로 구성하고 갱신된 토큰을 단일 설정에 회수한다."""
import json
import os
from pathlib import Path
import sys
import tempfile
from private_config import config_path, load_config, save_credentials


def main():
    os.umask(0o077)
    provider, *arguments = sys.argv[1:]
    if provider not in {"kaggle", "colab"}:
        raise ValueError("지원하지 않는 공급자")
    # subprocess에서 작업 디렉터리가 바뀌어도 동일한 개인 설정을 사용한다.
    os.environ["REMOTE_GPU_CONFIG"] = str(config_path())
    config = load_config()
    original = config.get("auth", {}).get(provider)
    with tempfile.TemporaryDirectory(prefix="remote-gpu-auth-") as directory:
        token = Path(directory) / "credentials.json"
        if original:
            token.write_text(json.dumps(original), encoding="utf-8")
        sys.argv = [provider, *arguments]
        try:
            if provider == "kaggle":
                os.environ["KAGGLE_CONFIG_DIR"] = directory
                from kagglesdk.kaggle_creds import KaggleCredentials
                KaggleCredentials.DEFAULT_CREDENTIALS_FILE = str(token)
                KaggleCredentials.save.__defaults__ = (str(token),)
                KaggleCredentials.delete.__defaults__ = (str(token),)
                from kaggle.cli import main as entrypoint
            else:
                import colab_cli.auth as auth
                auth.TOKEN_CONFIG_PATH = str(token)
                # Colab에 불필요한 Cloud Platform 권한은 요청하지 않는다.
                auth.PUBLIC_SCOPES = [scope for scope in auth.PUBLIC_SCOPES if not scope.endswith('/cloud-platform')]
                os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
                client_path = Path(directory) / "client.json"
                if client := config.get("auth", {}).get("colab_client"):
                    client_path.write_text(json.dumps(client), encoding="utf-8")
                sys.argv[1:1] = ["--client-oauth-config", str(client_path)]
                from colab_cli.cli import main as entrypoint
            return entrypoint()
        finally:
            updated = json.loads(token.read_text()) if token.exists() else None
            if updated != original:
                save_credentials(provider, updated)


if __name__ == "__main__":
    sys.exit(main())
