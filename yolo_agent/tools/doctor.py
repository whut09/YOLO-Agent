"""Environment doctor for training-ready YOLO Agent runs."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.batch_tuner import vram_batch_candidates


DoctorLevel = Literal["info", "warning", "error"]
DatasetKind = Literal["coco", "custom"]


class DoctorCheck(BaseModel):
    """One doctor/preflight check with a user-actionable fix."""

    name: str
    ok: bool
    level: DoctorLevel = "info"
    message: str = ""
    fix: str = ""


class BatchSizeEstimate(BaseModel):
    """Conservative batch-size estimate from visible GPU memory."""

    selected_batch: int | None = None
    candidate_batches: list[int] = Field(default_factory=list)
    imgsz: int = 640
    model: str = ""
    model_scale: str = "unknown"
    free_vram_gb: float | None = None
    total_vram_gb: float | None = None
    reserve_vram_gb: float = 2.0
    estimated_gb_per_sample: float | None = None
    confidence: Literal["none", "low", "medium"] = "none"
    limiting_reason: str = ""
    note: str = (
        "Preflight estimate only. Real optimize runs should keep batch=auto so BatchTuner can verify "
        "the largest safe batch with short probes before training."
    )


class DoctorReport(BaseModel):
    """Structured doctor result."""

    data_yaml: Path
    model: str
    run_root: Path
    kind: DatasetKind = "coco"
    checks: list[DoctorCheck] = Field(default_factory=list)
    batch_estimate: BatchSizeEstimate | None = None

    @property
    def ok(self) -> bool:
        """Return whether the environment passed all hard checks."""
        return not any(check.level == "error" and not check.ok for check in self.checks)

    @property
    def error_count(self) -> int:
        """Count failed hard checks."""
        return sum(1 for check in self.checks if check.level == "error" and not check.ok)

    @property
    def warning_count(self) -> int:
        """Count failed warning checks."""
        return sum(1 for check in self.checks if check.level == "warning" and not check.ok)


def run_doctor(
    data_yaml: Path | str,
    model: str = "yolo26n.pt",
    run_root: Path | str = "runs",
    kind: DatasetKind = "coco",
    min_disk_gb: float = 10.0,
    min_vram_gb: float = 4.0,
    imgsz: int = 640,
    candidate_batches: list[int] | None = None,
) -> DoctorReport:
    """Run environment and dataset checks for one-command training readiness."""
    data_path = Path(data_yaml)
    run_root_path = Path(run_root)
    checks: list[DoctorCheck] = [
        _python_check(),
        _package_check("ultralytics", 'py -3.12 -m pip install -e ".[train]"'),
        _model_check(model),
    ]

    dataset_root: Path | None = None
    raw_data: dict[str, object] = {}
    data_check = _data_yaml_check(data_path)
    checks.append(data_check)
    if data_check.ok:
        raw_data = _read_yaml(data_path)
        dataset_root = _dataset_root(data_path, raw_data)
        checks.append(_dataset_root_check(dataset_root))
        if dataset_root.exists():
            checks.extend(_split_checks(data_path, raw_data, dataset_root, kind))
            checks.extend(_annotation_checks(dataset_root, kind))
            checks.append(_disk_check(dataset_root, min_disk_gb, name="dataset_disk_free"))

    checks.append(_run_root_writable_check(run_root_path))
    checks.append(_disk_check(_existing_parent(run_root_path), min_disk_gb, name="run_root_disk_free"))
    gpu_status = _gpu_status()
    checks.append(_nvidia_smi_check(min_vram_gb, status=gpu_status))
    checks.append(_torch_cuda_check())
    batch_estimate = estimate_batch_size(
        model=model,
        free_vram_gb=_float_or_none(gpu_status.get("free_vram_gb")),
        total_vram_gb=_float_or_none(gpu_status.get("total_vram_gb")),
        imgsz=imgsz,
        candidate_batches=candidate_batches,
    )

    return DoctorReport(
        data_yaml=data_path,
        model=model,
        run_root=run_root_path,
        kind=kind,
        checks=checks,
        batch_estimate=batch_estimate,
    )


def estimate_batch_size(
    *,
    model: str,
    free_vram_gb: float | None,
    total_vram_gb: float | None = None,
    imgsz: int = 640,
    candidate_batches: list[int] | None = None,
) -> BatchSizeEstimate:
    """Estimate the largest candidate batch that fits visible free VRAM."""
    base_candidates = candidate_batches or [32, 48, 64, 96]
    expanded_candidates = [*base_candidates, *vram_batch_candidates(total_vram_gb, unit="gb")]
    candidates = sorted({int(value) for value in expanded_candidates if int(value) > 0})
    scale = _model_scale(model)
    estimate = BatchSizeEstimate(
        candidate_batches=candidates,
        imgsz=imgsz,
        model=model,
        model_scale=scale,
        free_vram_gb=free_vram_gb,
        total_vram_gb=total_vram_gb,
    )
    if free_vram_gb is None or free_vram_gb <= 0:
        estimate.limiting_reason = "No visible free VRAM; cannot estimate a training batch."
        return estimate
    if not candidates:
        estimate.limiting_reason = "No positive candidate batch sizes were provided."
        return estimate

    per_sample_gb = _gb_per_sample_at_640(scale) * (max(imgsz, 1) / 640) ** 2
    estimate.estimated_gb_per_sample = round(per_sample_gb, 4)
    usable_gb = max((free_vram_gb * 0.90) - estimate.reserve_vram_gb, 0.0)
    fitting = [batch for batch in candidates if batch * per_sample_gb <= usable_gb]
    if not fitting:
        estimate.confidence = "low"
        estimate.limiting_reason = (
            f"Even batch {candidates[0]} may exceed the conservative VRAM budget "
            f"({usable_gb:.1f} GB usable after reserve)."
        )
        return estimate

    estimate.selected_batch = max(fitting)
    estimate.confidence = "medium" if scale != "unknown" else "low"
    if estimate.selected_batch < max(candidates):
        estimate.limiting_reason = (
            f"Conservative VRAM budget allows up to batch {estimate.selected_batch}; "
            f"larger candidates may OOM."
        )
    else:
        estimate.limiting_reason = f"All requested candidates fit the conservative VRAM estimate."
    return estimate


def _python_check() -> DoctorCheck:
    ok = sys.version_info >= (3, 10)
    return DoctorCheck(
        name="python",
        ok=ok,
        level="error",
        message=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        fix="Install Python 3.10+ and rerun from that environment.",
    )


def _package_check(package: str, fix: str) -> DoctorCheck:
    spec = importlib.util.find_spec(package)
    if spec is None:
        return DoctorCheck(
            name=package,
            ok=False,
            level="error",
            message=f"{package} is not installed in this Python environment.",
            fix=fix,
        )
    version = "installed"
    try:
        module = importlib.import_module(package)
        version_value = getattr(module, "__version__", None)
        if version_value:
            version = f"version {version_value}"
    except Exception:
        version = "installed, but import raised an exception"
    return DoctorCheck(name=package, ok=True, level="info", message=version, fix=fix)


def _model_check(model: str) -> DoctorCheck:
    path = Path(model)
    if path.suffix == ".pt":
        ok = path.is_file()
        return DoctorCheck(
            name="model",
            ok=ok,
            level="error" if not ok else "info",
            message=f"{'found' if ok else 'missing'}: {path}",
            fix=f"Place the checkpoint at {path} or pass --model with a valid checkpoint/name.",
        )
    return DoctorCheck(
        name="model",
        ok=True,
        level="info",
        message=f"using model name/checkpoint reference: {model}",
        fix="Use --model E:\\path\\to\\weights.pt if automatic model resolution is not desired.",
    )


def _data_yaml_check(data_yaml: Path) -> DoctorCheck:
    if not data_yaml.is_file():
        return DoctorCheck(
            name="data_yaml",
            ok=False,
            level="error",
            message=f"missing: {data_yaml}",
            fix=f"Create or pass the correct data yaml, for example: yolo-agent doctor --data {data_yaml}",
        )
    try:
        raw = _read_yaml(data_yaml)
    except Exception as exc:
        return DoctorCheck(
            name="data_yaml",
            ok=False,
            level="error",
            message=f"YAML parse failed: {exc}",
            fix="Fix the YAML syntax and make sure it follows Ultralytics data.yaml format.",
        )
    ok = isinstance(raw, dict) and bool(raw)
    return DoctorCheck(
        name="data_yaml",
        ok=ok,
        level="error",
        message=f"found: {data_yaml}" if ok else f"empty or invalid YAML: {data_yaml}",
        fix="Add at least path/train/val/names entries to the YOLO data.yaml.",
    )


def _dataset_root_check(dataset_root: Path) -> DoctorCheck:
    ok = dataset_root.exists()
    return DoctorCheck(
        name="dataset_root",
        ok=ok,
        level="error",
        message=f"{'found' if ok else 'missing'}: {dataset_root}",
        fix=f"Create the dataset root or update the `path:` field in data.yaml to {dataset_root}.",
    )


def _split_checks(
    data_yaml: Path,
    raw_data: dict[str, object],
    dataset_root: Path,
    kind: DatasetKind,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    defaults = {
        "train": Path("images") / "train2017" if kind == "coco" else None,
        "val": Path("images") / "val2017" if kind == "coco" else None,
        "test": Path("images") / "test2017" if kind == "coco" else None,
    }
    for split in ("train", "val", "test"):
        resolved = _resolve_split_path(data_yaml, raw_data, dataset_root, split, defaults[split])
        if resolved is None:
            level: DoctorLevel = "error" if split in {"train", "val"} or kind == "coco" else "warning"
            checks.append(
                DoctorCheck(
                    name=f"{split}_split",
                    ok=False,
                    level=level,
                    message=f"`{split}` is missing from data.yaml.",
                    fix=f"Add `{split}: images/{split}2017` to data.yaml or pass a data.yaml with a valid {split} split.",
                )
            )
            continue
        ok = resolved.exists()
        checks.append(
            DoctorCheck(
                name=f"{split}_split",
                ok=ok,
                level="error" if not ok and (split in {"train", "val"} or kind == "coco") else "warning",
                message=f"{'found' if ok else 'missing'}: {resolved}",
                fix=f"Download/extract the {split} images to {resolved} or update `{split}:` in data.yaml.",
            )
        )
    return checks


def _annotation_checks(dataset_root: Path, kind: DatasetKind) -> list[DoctorCheck]:
    annotations = dataset_root / "annotations"
    checks = [
        DoctorCheck(
            name="annotations_dir",
            ok=annotations.is_dir(),
            level="error" if kind == "coco" else "warning",
            message=f"{'found' if annotations.is_dir() else 'missing'}: {annotations}",
            fix=f"Download/extract COCO annotations to {annotations}.",
        )
    ]
    if kind != "coco":
        return checks
    required_levels: dict[str, DoctorLevel] = {
        "instances_train2017.json": "warning",
        "instances_val2017.json": "error",
    }
    for filename, level in required_levels.items():
        path = annotations / filename
        checks.append(
            DoctorCheck(
                name=f"annotation_{filename}",
                ok=path.is_file(),
                level=level,
                message=f"{'found' if path.is_file() else 'missing'}: {path}",
                fix=(
                    f"Extract annotations_trainval2017.zip so {path} exists."
                    if level == "error"
                    else f"Extract annotations_trainval2017.zip so {path} exists if you need train split COCO JSON analysis."
                ),
            )
        )
    test_options = [
        annotations / "image_info_test2017.json",
        annotations / "image_info_test-dev2017.json",
    ]
    ok = any(path.is_file() for path in test_options)
    checks.append(
        DoctorCheck(
            name="annotation_test2017_image_info",
            ok=ok,
            level="warning",
            message="found test2017 image_info annotation" if ok else f"missing one of: {', '.join(str(path) for path in test_options)}",
            fix=f"Extract image_info_test2017.zip into {annotations} if you need test2017 bookkeeping.",
        )
    )
    return checks


def _run_root_writable_check(run_root: Path) -> DoctorCheck:
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        probe = run_root / ".yolo_agent_doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="run_root_writable",
            ok=False,
            level="error",
            message=f"cannot write to {run_root}: {exc}",
            fix=f"Create a writable run directory, for example: mkdir {run_root}",
        )
    return DoctorCheck(
        name="run_root_writable",
        ok=True,
        level="info",
        message=f"writable: {run_root}",
        fix=f"Create a writable run directory, for example: mkdir {run_root}",
    )


def _disk_check(path: Path, min_disk_gb: float, name: str) -> DoctorCheck:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return DoctorCheck(
            name=name,
            ok=False,
            level="error",
            message=f"cannot inspect disk at {path}: {exc}",
            fix="Use a valid local path with enough free space.",
        )
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_disk_gb
    return DoctorCheck(
        name=name,
        ok=ok,
        level="error" if not ok else "info",
        message=f"{free_gb:.1f} GB free at {path}",
        fix=f"Free disk space or pass --run-root to a drive with at least {min_disk_gb:.1f} GB free.",
    )


def _nvidia_smi_check(min_vram_gb: float, status: dict[str, object] | None = None) -> DoctorCheck:
    status = status or _gpu_status()
    if not status["ok"]:
        return DoctorCheck(
            name="cuda_driver",
            ok=False,
            level="error",
            message=status["message"],
            fix="Install/update the NVIDIA driver, reopen the terminal, then verify with: nvidia-smi",
        )
    free_gb = status.get("free_vram_gb")
    ok = free_gb is not None and free_gb >= min_vram_gb
    message = status["message"]
    if free_gb is not None:
        message = f"{message}; free_vram={free_gb:.1f} GB"
    total_gb = status.get("total_vram_gb")
    if isinstance(total_gb, int | float):
        message = f"{message}; total_vram={float(total_gb):.1f} GB"
    return DoctorCheck(
        name="cuda_driver",
        ok=ok,
        level="error" if not ok else "info",
        message=message,
        fix=f"Close other GPU jobs or choose a smaller profile until at least {min_vram_gb:.1f} GB VRAM is free.",
    )


def _torch_cuda_check() -> DoctorCheck:
    status = _torch_cuda_status()
    return DoctorCheck(
        name="torch_cuda",
        ok=status["ok"],
        level="error" if not status["ok"] else "info",
        message=status["message"],
        fix='Install a CUDA-enabled PyTorch build, for example: py -3.12 -m pip install -e ".[train]"',
    )


def _gpu_status() -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"ok": False, "message": "nvidia-smi unavailable"}
    if completed.returncode != 0:
        return {"ok": False, "message": "nvidia-smi returned an error"}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return {"ok": False, "message": "no visible NVIDIA GPU"}
    best_free_mb = 0.0
    best_total_mb = 0.0
    names: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 3:
            names.append(parts[0])
            try:
                total_mb = float(parts[1])
                free_mb = float(parts[2])
            except ValueError:
                continue
            if free_mb > best_free_mb:
                best_free_mb = free_mb
                best_total_mb = total_mb
        else:
            names.append(row)
    return {
        "ok": True,
        "message": ", ".join(names),
        "free_vram_gb": best_free_mb / 1024 if best_free_mb else None,
        "total_vram_gb": best_total_mb / 1024 if best_total_mb else None,
    }


def _torch_cuda_status() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {"ok": False, "message": "torch is not installed"}
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            return {"ok": False, "message": f"torch {getattr(torch, '__version__', 'unknown')} reports CUDA unavailable"}
        device_count = int(torch.cuda.device_count())
        device_name = torch.cuda.get_device_name(0) if device_count else "unknown"
        return {
            "ok": True,
            "message": f"torch {getattr(torch, '__version__', 'unknown')} CUDA ready; devices={device_count}; first={device_name}",
        }
    except Exception as exc:
        return {"ok": False, "message": f"torch CUDA probe failed: {exc}"}


def _resolve_split_path(
    data_yaml: Path,
    raw_data: dict[str, object],
    dataset_root: Path,
    split: str,
    default_relative: Path | None,
) -> Path | None:
    configured = raw_data.get(split)
    if isinstance(configured, list) and configured:
        configured = configured[0]
    if configured is None:
        return dataset_root / default_relative if default_relative is not None else None
    path = Path(str(configured))
    if path.is_absolute():
        return path
    root_relative = dataset_root / path
    if root_relative.exists() or raw_data.get("path") is not None:
        return root_relative
    return data_yaml.parent / path


def _dataset_root(data_yaml: Path, raw_data: dict[str, object]) -> Path:
    configured = raw_data.get("path")
    if configured is None:
        return data_yaml.parent
    root = Path(str(configured))
    return root if root.is_absolute() else data_yaml.parent / root


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _model_scale(model: str) -> str:
    stem = Path(model).stem.lower()
    for suffix in ("n", "s", "m", "l", "x"):
        if stem.endswith(suffix):
            return suffix
    return "unknown"


def _gb_per_sample_at_640(scale: str) -> float:
    return {
        "n": 0.08,
        "s": 0.12,
        "m": 0.20,
        "l": 0.32,
        "x": 0.45,
    }.get(scale, 0.12)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
