"""원격 Iris 데이터를 읽고 CUDA에서 분류 모델을 학습한다."""

import csv
import json
import os
from pathlib import Path

import torch


def load_kaggle_iris(input_dir):
    paths = sorted(Path(input_dir).rglob("Iris.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Iris.csv가 한 개 필요하다. 발견 {len(paths)}개")
    with paths[0].open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    features = ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
    species = sorted({row["Species"] for row in rows})
    samples = [[float(row[name]) for name in features] for row in rows]
    labels = [species.index(row["Species"]) for row in rows]
    return samples, labels, species


if not torch.cuda.is_available():
    devices = sorted(path.name for path in Path("/dev").glob("nvidia[0-9]*"))
    raise RuntimeError(f"CUDA를 사용할 수 없다. PyTorch CUDA 빌드 {torch.version.cuda}, GPU 장치 {devices}")

is_colab = os.environ.get("GPU_SMOKE_PLATFORM") == "colab"
input_dir = os.environ.get("GPU_INPUT_DIR")
if is_colab and not input_dir:
    from sklearn.datasets import load_iris

    iris = load_iris()
    samples, labels = iris.data.tolist(), iris.target.tolist()
    species = [f"Iris-{name}" for name in iris.target_names]
else:
    samples, labels, species = load_kaggle_iris(input_dir or "/kaggle/input")

x = torch.tensor(samples, dtype=torch.float32, device="cuda")
y = torch.tensor(labels, device="cuda")

torch.manual_seed(42)
order = torch.randperm(len(labels), device="cuda")
train, test = order[:120], order[120:]
x_train = x[train]
mean, scale = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
x = (x - mean) / scale

model = torch.nn.Linear(x.shape[1], len(species), device="cuda")
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
criterion = torch.nn.CrossEntropyLoss()
for step in range(120):
    optimizer.zero_grad()
    loss = criterion(model(x[train]), y[train])
    if step == 0:
        initial_loss = loss.item()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_loss = criterion(model(x[train]), y[train]).item()
    accuracy = (model(x[test]).argmax(1) == y[test]).float().mean().item()
if not (final_loss < initial_loss and accuracy >= 0.7):
    raise RuntimeError(f"학습 검증 실패. 손실 {final_loss:.4f}, 정확도 {accuracy:.4f}")

metrics = {
    "device": torch.cuda.get_device_name(0),
    "rows": len(labels),
    "initial_loss": round(initial_loss, 4),
    "final_loss": round(final_loss, 4),
    "accuracy": round(accuracy, 4),
}
print(json.dumps(metrics), flush=True)
