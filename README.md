# remote-gpu

Codex에서 원격 GPU 학습을 제출하고 상태·로그·결과를 조회하는 스킬이다. 반복 작업은 Python과 셸 스크립트가 처리한다. Kaggle 무료 GPU를 우선 사용하고, 제출 전 잔여량 부족·조회 실패 또는 명확한 GPU 한도 거절 시 Colab으로 전환한다.

## 환경과 설치

개발·실행 확인 환경은 macOS, POSIX 셸, Python 3.14이다. 재구현에는 Python 3.13 이상, `venv`, `pip`, 인터넷 연결이 필요하다. Windows는 WSL을 사용한다. Linux는 아래 캐시 경로를 지정해 사용할 수 있으나 실제 실행 검증은 macOS에서 수행했다.

```sh
git clone https://github.com/juwan3057-create/remote-gpu.git
cd remote-gpu
# Linux에서 권장. macOS 기본값은 ~/Library/Caches/remote-gpu/venv
# export REMOTE_GPU_VENV="$HOME/.cache/remote-gpu/venv"
./gpu.sh setup
cp gpu.example.json gpu.local.json
chmod 600 gpu.local.json
./gpu.sh auth kaggle
./gpu.sh auth colab
```

`gpu.local.json`의 `profile.kaggle_owner`를 본인의 Kaggle 사용자명으로 수정한다. Kaggle GPU는 계정의 전화번호 인증과 잔여 할당량이 필요하다. Colab 로그인은 표시된 주소에서 인증한 코드를 터미널에 입력한다. Python 의존성은 `requirements*.txt`에 고정했다.

Codex 스킬로 설치하려면 저장소 전체를 `~/.agents/skills/remote-gpu`에 복제한다. 프로젝트별 설정은 작업 프로젝트 루트에 둔다. 이 저장소 자체에서는 `.agents/skills/remote-gpu`로도 검색된다. 다른 프로젝트에서는 설치한 스킬 폴더의 `gpu.sh`를 절대경로로 호출한다.

## 설정과 사용

개인정보·계정명·OAuth 토큰은 **`gpu.local.json` 한 파일**에 저장하며 Git에서 제외한다. 공개본에는 빈 `gpu.example.json`만 포함한다. `REMOTE_GPU_CONFIG`로 설정 파일 경로를 지정할 수 있다. 로그인·토큰 갱신 시 CLI용 인증 파일은 권한이 제한된 임시 디렉터리에 생성하고 종료 시 제거한다. 기존 공식 CLI 인증 파일은 자동 삭제하지 않으며, 이 스킬은 이를 사용하지 않는다.

`profile.size` 기본값과 Colab 요청은 다음과 같다. 모델과 배치의 GPU 메모리 요구량에 맞춰 프로젝트별로 선택한다.

| size | Colab GPU | 코드 실행 한도 |
| --- | --- | --- |
| small | T4 | 300초 |
| medium | L4 | 900초 |
| large | A100 | 1800초 |

`profile.gpu`, `profile.timeout_seconds`로 기본값을 바꾸거나 실행 시 `--colab-gpu`, `--timeout`으로 덮어쓴다. Kaggle은 규모와 관계없이 T4를 요청한다. 실제 GPU 할당은 공급자 가용성에 달려 있다.

```sh
./gpu.sh run train.py --dataset owner/slug
./gpu.sh quota
./gpu.sh watch owner/notebook/1
./gpu.sh outputs owner/notebook/1
./gpu.sh sessions
```

`--provider kaggle` 또는 `--provider colab`으로 공급자를 고정한다. Colab의 기존 컴퓨팅 유닛을 허용할 때만 해당 실행에 `--allow-compute-units`를 붙인다. 플래그가 없으면 표시 잔액이 0.00이고 요청 GPU가 T4일 때만 시도한다. 무료 자원의 할당은 보장되지 않는다. `sessions`는 Colab 세션만 조회한다.

## 파이프라인

`Codex → gpu.sh → 개인 설정 로드 → Kaggle 잔여량 조회 → 비공개 노트북 제출 → 지정 버전 상태·로그 조회` 순서이다. 제출 전 Kaggle을 사용할 수 없으면 `Colab 잔액 확인 → 런타임 할당 → 코드·데이터 전달 → 실행 → 런타임 종료`로 전환한다. 제출 응답이 불확실하거나 학습이 실패하면 중복 실행을 막기 위해 자동 재제출하지 않는다.

입력은 단일 Python 파일이다. Kaggle에 한해 `.ipynb`도 지원한다. Kaggle 데이터는 `/kaggle/input`에 연결하며 인터넷은 비활성화한다. Colab에서는 `kagglehub`로 런타임에 내려받고 `GPU_INPUT_DIR`를 전달한다. 공통 학습 코드는 이 환경 변수가 없으면 `/kaggle/input`을 읽도록 작성한다. 비공개 데이터는 Colab 런타임에서 별도 인증이 필요하며 로컬 토큰을 자동 전송하지 않는다.

Kaggle 결과는 `/kaggle/working`에 저장한다. `outputs`는 목록만 조회한다. Colab 임시 결과는 런타임 종료 시 소실되므로 학습 코드에서 영구 저장소로 내보내야 한다. 학습용 PyTorch·CUDA와 데이터별 라이브러리는 원격 환경에서 제공하거나 학습 코드에서 준비한다. `gpu_smoke.py`는 Iris·CUDA 검증용이며 결과 파일을 생성하지 않는다.
