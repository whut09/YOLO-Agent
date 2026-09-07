"""Resolve real YOLO26 teacher checkpoints through verified sources.

The resolver is deliberately an asset preflight, not a training helper.  It
uses the installed Ultralytics model registry for official downloads, records
the resulting file digest, and binds the teacher to the execution dataset
protocol supplied by the caller.  Test-only files are retained as unavailable
evidence and can never become production-ready assets.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


OfficialTeacherName = Literal["yolo26s.pt", "yolo26m.pt"]
TeacherAssetDisposition = Literal["available", "unavailable"]
TeacherAssetSourceKind = Literal["local", "ultralytics_official", "unavailable"]
TeacherMetadataSource = Literal[
    "checkpoint",
    "sidecar",
    "official_model_registry",
    "filename",
    "missing",
]

OFFICIAL_YOLO26_TEACHERS: tuple[OfficialTeacherName, ...] = (
    "yolo26s.pt",
    "yolo26m.pt",
)


class TeacherAssetRecord(BaseModel, YAMLModelMixin):
    """One verified teacher file and the protocol it is bound to."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "teacher_asset_record.v2"
    requested_name: str
    architecture: str
    checkpoint_path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_source: str
    source_kind: TeacherAssetSourceKind
    dataset: str
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = "train"
    imgsz: int = 640
    frozen: Literal[True] = True
    metadata_verified: bool = False
    metadata_source: TeacherMetadataSource = "missing"
    checkpoint_loadable: bool = False
    protocol_bound: bool = False
    test_only: bool = False
    availability: TeacherAssetDisposition
    exact_blocker: str = ""
    recovery_action: str
    identity_hash: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_record(self) -> "TeacherAssetRecord":
        if self.requested_name not in OFFICIAL_YOLO26_TEACHERS:
            raise ValueError("teacher name is not an official YOLO26 teacher")
        if self.architecture not in {"yolo26s", "yolo26m"}:
            raise ValueError("teacher architecture must be yolo26s or yolo26m")
        if self.imgsz != 640:
            raise ValueError("teacher asset protocol requires imgsz=640")
        if self.split != "train":
            raise ValueError("teacher asset protocol requires train split")
        if not self.protocol_bound:
            raise ValueError("teacher asset must be bound to an execution protocol")
        if self.test_only and self.availability == "available":
            raise ValueError("test-only teacher cannot enter available assets")

        if self.checkpoint_path is not None:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.is_absolute() or not checkpoint.is_file():
                raise ValueError("teacher checkpoint path must be an existing absolute file")
            if not self.sha256:
                raise ValueError("teacher checkpoint requires SHA-256")
            if _sha256_file(checkpoint) != self.sha256:
                raise ValueError("teacher checkpoint SHA-256 does not match the file")
        elif self.sha256 is not None:
            raise ValueError("teacher SHA-256 requires a checkpoint path")

        if self.availability == "available":
            if not self.checkpoint_path or not self.sha256:
                raise ValueError("available teacher requires path and SHA-256")
            if not self.checkpoint_loadable:
                raise ValueError("available teacher requires a loadable checkpoint")
            if not self.metadata_verified:
                raise ValueError("available teacher requires verified metadata")
            if self.exact_blocker:
                raise ValueError("available teacher cannot retain a blocker")
        elif not self.exact_blocker:
            raise ValueError("unavailable teacher requires exact_blocker")

        expected = self.calculate_identity_hash()
        if self.identity_hash and self.identity_hash != expected:
            raise ValueError("teacher asset identity hash mismatch")
        return self

    def calculate_identity_hash(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"identity_hash", "generated_at"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "TeacherAssetRecord":
        return self.model_copy(update={"identity_hash": self.calculate_identity_hash()})


class TeacherAssetReport(BaseModel, YAMLModelMixin):
    """Persistent teacher-resolution report; it contains no accuracy result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "teacher_asset_report.v2"
    student_model: str
    student_architecture: Literal["yolo26n"] = "yolo26n"
    dataset: str
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = "train"
    imgsz: int = 640
    preferred_teacher: str | None = None
    records: list[TeacherAssetRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> "TeacherAssetReport":
        if self.imgsz != 640:
            raise ValueError("teacher report requires imgsz=640")
        if self.split != "train":
            raise ValueError("teacher report requires train split")
        if self.student_architecture != "yolo26n":
            raise ValueError("teacher report requires yolo26n student")
        names = [item.requested_name for item in self.records]
        if len(names) != len(set(names)):
            raise ValueError("teacher report contains duplicate requested names")
        for item in self.records:
            if item.dataset_manifest_hash != self.dataset_manifest_hash:
                raise ValueError("teacher record dataset hash does not match report")
            if item.split != self.split or item.imgsz != self.imgsz:
                raise ValueError("teacher record protocol does not match report")
        available = {
            item.requested_name
            for item in self.records
            if item.availability == "available"
        }
        if self.preferred_teacher is not None and self.preferred_teacher not in available:
            raise ValueError("preferred teacher must be available")
        if self.report_hash and self.report_hash != self.calculate_hash():
            raise ValueError("teacher report hash mismatch")
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"report_hash", "generated_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def with_hash(self) -> "TeacherAssetReport":
        return self.model_copy(update={"report_hash": self.calculate_hash()})

    @property
    def preferred_record(self) -> TeacherAssetRecord | None:
        if self.preferred_teacher is None:
            return None
        return next(
            (item for item in self.records if item.requested_name == self.preferred_teacher),
            None,
        )


class TeacherAssetResolver:
    """Resolve official YOLO26 teachers locally or via Ultralytics."""

    def resolve(
        self,
        *,
        student_model: str | Path = "yolo26n.pt",
        dataset: str | Path,
        output_path: str | Path | None = None,
        download_dir: str | Path | None = None,
        preferred_teachers: tuple[OfficialTeacherName, ...] = OFFICIAL_YOLO26_TEACHERS,
        allow_download: bool = True,
        test_only: bool = False,
    ) -> TeacherAssetReport:
        student_name = Path(student_model).name.lower()
        if student_name != "yolo26n.pt":
            raise ValueError("teacher resolution requires fixed student yolo26n.pt")
        dataset_path = Path(dataset).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"dataset manifest does not exist: {dataset_path}")
        dataset_hash = _sha256_file(dataset_path)
        target_dir = Path(download_dir or "runs/paper-readiness/teachers").expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        selected = tuple(dict.fromkeys(preferred_teachers))
        invalid = set(selected) - set(OFFICIAL_YOLO26_TEACHERS)
        if invalid:
            raise ValueError(f"unsupported official teacher names: {sorted(invalid)}")
        records = [
            self._resolve_one(
                name,
                dataset=dataset_path,
                dataset_hash=dataset_hash,
                download_dir=target_dir,
                allow_download=allow_download,
                test_only=test_only,
            )
            for name in selected
        ]
        available = next((item for item in records if item.availability == "available"), None)
        report = TeacherAssetReport(
            student_model=str(Path(student_model).expanduser().resolve(strict=False)),
            dataset=str(dataset_path),
            dataset_manifest_hash=dataset_hash,
            preferred_teacher=available.requested_name if available else None,
            records=records,
        ).with_hash()
        if output_path is not None:
            report.to_yaml(output_path, exclude_none=True, sort_keys=False)
        return report

    def _resolve_one(
        self,
        name: OfficialTeacherName,
        *,
        dataset: Path,
        dataset_hash: str,
        download_dir: Path,
        allow_download: bool,
        test_only: bool,
    ) -> TeacherAssetRecord:
        path: Path | None = None
        source_kind: TeacherAssetSourceKind = "unavailable"
        blocker = ""

        for candidate in (Path(name), download_dir / name):
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved.is_file():
                path = resolved
                source_kind = "local"
                break

        if path is None and allow_download:
            try:
                from ultralytics.utils.downloads import (
                    GITHUB_ASSETS_NAMES,
                    GITHUB_ASSETS_REPO,
                    attempt_download_asset,
                )

                if name not in GITHUB_ASSETS_NAMES:
                    blocker = "teacher_not_present_in_installed_ultralytics_registry"
                else:
                    downloaded = attempt_download_asset(
                        download_dir / name,
                        repo=GITHUB_ASSETS_REPO,
                        release="latest",
                    )
                    candidate = Path(downloaded).expanduser().resolve(strict=False)
                    if candidate.is_file() and candidate.name == name:
                        path = candidate
                        source_kind = "ultralytics_official"
                    else:
                        blocker = "official_teacher_download_did_not_return_expected_file"
            except (
                FileNotFoundError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                blocker = f"official_teacher_resolution_failed:{type(exc).__name__}"
        elif path is None:
            blocker = "official_teacher_download_disabled"

        if path is None:
            return TeacherAssetRecord(
                requested_name=name,
                architecture=Path(name).stem,
                checkpoint_source="ultralytics_official_model_registry",
                source_kind="unavailable",
                dataset=str(dataset),
                dataset_manifest_hash=dataset_hash,
                metadata_verified=False,
                metadata_source="missing",
                checkpoint_loadable=False,
                protocol_bound=True,
                availability="unavailable",
                exact_blocker=blocker or "teacher_checkpoint_unavailable",
                recovery_action=(
                    "install the official Ultralytics asset or provide a verified "
                    "frozen teacher checkpoint, then rerun resolve-teachers"
                ),
                test_only=test_only,
            ).with_hash()

        digest = _sha256_file(path)
        inspection = _inspect_checkpoint(path)
        architecture = inspection["architecture"] or Path(name).stem
        reasons: list[str] = []
        if not inspection["loadable"]:
            reasons.append("teacher_checkpoint_unreadable")
        if architecture != Path(name).stem:
            reasons.append("teacher_architecture_not_matching_requested_official_name")
        if source_kind == "local" and inspection["metadata_source"] in {"missing", "filename"}:
            reasons.append("teacher_checkpoint_metadata_missing")
        metadata_dataset = inspection["dataset_hash"]
        if metadata_dataset and metadata_dataset != dataset_hash:
            reasons.append("teacher_dataset_manifest_hash_mismatch")
        if inspection["split"] and inspection["split"] != "train":
            reasons.append("teacher_split_mismatch")
        if inspection["imgsz"] and inspection["imgsz"] != 640:
            reasons.append("teacher_imgsz_mismatch")
        if test_only:
            reasons.append("test_only_teacher_not_production_asset")
        available = not reasons
        metadata_verified = bool(
            inspection["loadable"]
            and architecture == Path(name).stem
            and (
                source_kind == "ultralytics_official"
                or inspection["metadata_source"] in {"checkpoint", "sidecar"}
            )
        )
        return TeacherAssetRecord(
            requested_name=name,
            architecture=architecture,
            checkpoint_path=str(path),
            sha256=digest,
            checkpoint_source=(
                "local_checkpoint"
                if source_kind == "local"
                else "ultralytics_official_model_registry"
            ),
            source_kind=source_kind,
            dataset=str(dataset),
            dataset_manifest_hash=dataset_hash,
            metadata_verified=metadata_verified,
            metadata_source=(
                "official_model_registry"
                if source_kind == "ultralytics_official"
                else inspection["metadata_source"]
            ),
            checkpoint_loadable=inspection["loadable"],
            protocol_bound=True,
            test_only=test_only,
            availability="available" if available else "unavailable",
            exact_blocker=";".join(dict.fromkeys(reasons)),
            recovery_action=(
                "none; frozen teacher checkpoint resolved and SHA-256 recorded"
                if available
                else "repair the teacher checkpoint identity or protocol evidence before production readiness"
            ),
        ).with_hash()


def verify_teacher_asset(
    record: TeacherAssetRecord,
    *,
    dataset_manifest_hash: str,
    expected_split: str = "train",
    expected_imgsz: int = 640,
) -> list[str]:
    """Revalidate a report record at readiness time, including file drift."""

    reasons: list[str] = []
    if record.availability != "available":
        reasons.append(record.exact_blocker or "teacher_checkpoint_unavailable")
    if record.test_only:
        reasons.append("test_only_teacher_not_production_asset")
    if not record.frozen:
        reasons.append("teacher_not_frozen")
    if record.dataset_manifest_hash != dataset_manifest_hash:
        reasons.append("teacher_dataset_manifest_hash_mismatch")
    if record.split != expected_split:
        reasons.append("teacher_split_mismatch")
    if record.imgsz != expected_imgsz:
        reasons.append("teacher_imgsz_mismatch")
    if not record.checkpoint_path or not record.sha256:
        reasons.append("teacher_checkpoint_sha256_missing")
    else:
        path = Path(record.checkpoint_path)
        if not path.is_absolute() or not path.is_file():
            reasons.append("teacher_checkpoint_missing")
        elif _sha256_file(path) != record.sha256:
            reasons.append("teacher_checkpoint_sha256_mismatch")
    return list(dict.fromkeys(reasons))


def resolve_teacher_assets(**kwargs: object) -> TeacherAssetReport:
    """Functional resolver used by the CLI and tests."""

    return TeacherAssetResolver().resolve(**kwargs)  # type: ignore[arg-type]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_checkpoint(path: Path) -> dict[str, object]:
    """Read lightweight checkpoint identity without constructing a detector."""

    metadata = _read_checkpoint_metadata(path)
    loadable = False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        loadable = isinstance(payload, dict) and any(
            key in payload for key in ("model", "ema", "state_dict", "architecture")
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        architecture = metadata["architecture"] or _architecture_from_payload(payload)
    else:
        architecture = metadata["architecture"]
    return {
        "architecture": architecture,
        "dataset_hash": metadata["dataset_hash"],
        "split": metadata["split"],
        "imgsz": metadata["imgsz"],
        "metadata_source": metadata["source"],
        "loadable": loadable,
    }


def _read_checkpoint_metadata(path: Path) -> dict[str, object]:
    for candidate in (
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(path.suffix + ".metadata.yaml"),
        path.with_suffix(".metadata.json"),
        path.with_suffix(".metadata.yaml"),
    ):
        if not candidate.is_file():
            continue
        try:
            import yaml

            raw = (
                json.loads(candidate.read_text(encoding="utf-8"))
                if candidate.suffix == ".json"
                else yaml.safe_load(candidate.read_text(encoding="utf-8"))
            )
            parsed = _metadata_from_mapping(raw, source="sidecar")
            if parsed["source"] != "missing":
                return parsed
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        raw = None
    parsed = _metadata_from_mapping(raw, source="checkpoint")
    if parsed["source"] != "missing":
        return parsed
    return {
        "architecture": _architecture_from_name(path),
        "dataset_hash": None,
        "split": None,
        "imgsz": None,
        "source": "filename",
    }


def _metadata_from_mapping(raw: object, *, source: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {
            "architecture": None,
            "dataset_hash": None,
            "split": None,
            "imgsz": None,
            "source": "missing",
        }
    candidates = [raw]
    for key in ("metadata", "meta", "args", "train_args"):
        value = raw.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    architecture = _first(candidates, "architecture", "arch", "model_name", "model_family")
    dataset_hash = _first(candidates, "dataset_hash", "data_hash", "dataset_manifest_hash")
    split = _first(candidates, "split", "dataset_split")
    imgsz = _first(candidates, "imgsz", "image_size", "image_size_train")
    if architecture is None and dataset_hash is None and split is None and imgsz is None:
        return {
            "architecture": None,
            "dataset_hash": None,
            "split": None,
            "imgsz": None,
            "source": "missing",
        }
    return {
        "architecture": str(architecture) if architecture is not None else None,
        "dataset_hash": str(dataset_hash) if dataset_hash is not None else None,
        "split": str(split) if split is not None else None,
        "imgsz": int(imgsz) if imgsz is not None else None,
        "source": source,
    }


def _first(mappings: list[dict[str, object]], *keys: str) -> object | None:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                return mapping[key]
    return None


def _architecture_from_payload(payload: dict[str, object]) -> str | None:
    for key in ("architecture", "model_name"):
        value = payload.get(key)
        if value:
            return str(value)
    model = payload.get("model")
    yaml_name = getattr(model, "yaml", None)
    if isinstance(yaml_name, dict):
        value = yaml_name.get("yaml_file") or yaml_name.get("model")
        if value:
            return Path(str(value)).stem
    return None


def _architecture_from_name(path: Path) -> str | None:
    stem = path.stem.lower()
    for scale in ("n", "s", "m", "l", "x"):
        if stem.endswith(scale) and "yolo26" in stem:
            return f"yolo26{scale}"
    return None


__all__ = [
    "OFFICIAL_YOLO26_TEACHERS",
    "OfficialTeacherName",
    "TeacherAssetRecord",
    "TeacherAssetReport",
    "TeacherAssetResolver",
    "verify_teacher_asset",
    "resolve_teacher_assets",
]
