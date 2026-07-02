"""Dataset split diagnostics and reconstruction planning."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_serializer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SplitName = Literal["train", "val", "test"]


class DatasetSample(BaseModel):
    """One image sample with split, label, and grouping metadata."""

    image_path: Path
    split: SplitName
    label_path: Path | None = None
    classes: list[int] = Field(default_factory=list)
    has_small_object: bool = False
    is_background: bool = False
    scene_group: str = "default"
    fingerprint: str

    @field_serializer("image_path", "label_path")
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize paths portably."""
        return value.as_posix() if value is not None else None


class DuplicateGroup(BaseModel):
    """Images that appear duplicated by lightweight fingerprint."""

    fingerprint: str
    image_paths: list[Path]
    splits: list[SplitName]

    @field_serializer("image_paths")
    def serialize_paths(self, value: list[Path]) -> list[str]:
        """Serialize duplicate paths portably."""
        return [path.as_posix() for path in value]


class LeakagePair(BaseModel):
    """Potential train/val/test leakage from the same image fingerprint."""

    fingerprint: str
    train_images: list[Path] = Field(default_factory=list)
    val_images: list[Path] = Field(default_factory=list)
    test_images: list[Path] = Field(default_factory=list)

    @field_serializer("train_images", "val_images", "test_images")
    def serialize_paths(self, value: list[Path]) -> list[str]:
        """Serialize leakage paths portably."""
        return [path.as_posix() for path in value]


class SplitAssignment(BaseModel):
    """Planned split assignment for one image."""

    image_path: Path
    split: SplitName
    reason: str

    @field_serializer("image_path")
    def serialize_image_path(self, value: Path) -> str:
        """Serialize paths portably."""
        return value.as_posix()


class DatasetSplitPlan(BaseModel):
    """Non-destructive split/reconstruction plan."""

    data_yaml: Path
    dataset_root: Path
    samples: list[DatasetSample]
    duplicates: list[DuplicateGroup] = Field(default_factory=list)
    leakage: list[LeakagePair] = Field(default_factory=list)
    scene_distribution: dict[str, dict[SplitName, int]] = Field(default_factory=dict)
    assignments: list[SplitAssignment] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        """Write split plan JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class DatasetSplitPlanner:
    """Analyze YOLO data.yaml and create non-destructive split plans."""

    def analyze(self, data_yaml: Path | str, train_ratio: float = 0.8, val_ratio: float = 0.2) -> DatasetSplitPlan:
        """Analyze duplicate/leakage risks and propose scene-balanced assignments."""
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
            raise ValueError("train_ratio and val_ratio must be positive and sum to <= 1.")
        data_path = Path(data_yaml)
        data = _read_yaml_mapping(data_path)
        root = _dataset_root(data_path, data)
        samples = _collect_samples(data_path, data, root)
        duplicates = _duplicate_groups(samples)
        leakage = _leakage_pairs(samples)
        assignments = _scene_balanced_assignments(samples, train_ratio, val_ratio)
        recommendations = _recommendations(duplicates, leakage, samples)
        return DatasetSplitPlan(
            data_yaml=data_path,
            dataset_root=root,
            samples=samples,
            duplicates=duplicates,
            leakage=leakage,
            scene_distribution=_scene_distribution(samples),
            assignments=assignments,
            recommendations=recommendations,
        )


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def _dataset_root(data_path: Path, data: dict[str, object]) -> Path:
    raw_root = data.get("path")
    if raw_root is None:
        return data_path.parent
    root = Path(str(raw_root))
    return root if root.is_absolute() else data_path.parent / root


def _collect_samples(data_path: Path, data: dict[str, object], root: Path) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for split in ("train", "val", "test"):
        for image_path in _collect_images(data_path, root, data.get(split)):
            label_path = _label_path_for_image(image_path)
            classes, has_small = _read_label_summary(label_path)
            samples.append(
                DatasetSample(
                    image_path=image_path,
                    split=split,  # type: ignore[arg-type]
                    label_path=label_path if label_path.exists() else None,
                    classes=classes,
                    has_small_object=has_small,
                    is_background=not classes,
                    scene_group=_scene_group(image_path, root),
                    fingerprint=_fingerprint(image_path),
                )
            )
    return sorted(samples, key=lambda sample: (sample.split, sample.image_path.as_posix()))


def _collect_images(data_path: Path, root: Path, raw_split: object) -> list[Path]:
    images: list[Path] = []
    raw_items = raw_split if isinstance(raw_split, list) else [raw_split]
    for raw_item in raw_items:
        if raw_item is None:
            continue
        split_path = Path(str(raw_item))
        if not split_path.is_absolute():
            split_path = root / split_path
        if split_path.is_dir():
            images.extend(path.resolve() for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        elif split_path.is_file() and split_path.suffix.lower() == ".txt":
            images.extend(_images_from_list(split_path, data_path, root))
        elif split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(split_path.resolve())
    return sorted(dict.fromkeys(images))


def _images_from_list(list_path: Path, data_path: Path, root: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        image_path = Path(stripped)
        if not image_path.is_absolute():
            image_path = root / image_path if (root / image_path).exists() else data_path.parent / image_path
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(image_path.resolve())
    return images


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_label_summary(label_path: Path) -> tuple[list[int], bool]:
    if not label_path.exists():
        return [], False
    classes: list[int] = []
    has_small = False
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        classes.append(class_id)
        has_small = has_small or width * height < 0.01
    return sorted(dict.fromkeys(classes)), has_small


def _fingerprint(path: Path) -> str:
    try:
        content_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        size = path.stat().st_size
    except OSError:
        content_hash = "missing"
        size = -1
    return f"{path.stem.lower()}:{size}:{content_hash}"


def _scene_group(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "images":
        return str(parts[2]) if len(parts) > 3 else str(parts[1])
    return path.parent.name


def _duplicate_groups(samples: list[DatasetSample]) -> list[DuplicateGroup]:
    by_fingerprint: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        by_fingerprint[sample.fingerprint].append(sample)
    return [
        DuplicateGroup(
            fingerprint=fingerprint,
            image_paths=[sample.image_path for sample in grouped],
            splits=sorted({sample.split for sample in grouped}),  # type: ignore[arg-type]
        )
        for fingerprint, grouped in sorted(by_fingerprint.items())
        if len(grouped) > 1
    ]


def _leakage_pairs(samples: list[DatasetSample]) -> list[LeakagePair]:
    by_fingerprint: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        by_fingerprint[sample.fingerprint].append(sample)
    pairs: list[LeakagePair] = []
    for fingerprint, grouped in sorted(by_fingerprint.items()):
        by_split = {split: [sample.image_path for sample in grouped if sample.split == split] for split in ("train", "val", "test")}
        if by_split["train"] and (by_split["val"] or by_split["test"]):
            pairs.append(
                LeakagePair(
                    fingerprint=fingerprint,
                    train_images=by_split["train"],
                    val_images=by_split["val"],
                    test_images=by_split["test"],
                )
            )
    return pairs


def _scene_distribution(samples: list[DatasetSample]) -> dict[str, dict[SplitName, int]]:
    distribution: dict[str, dict[SplitName, int]] = {}
    for sample in samples:
        distribution.setdefault(sample.scene_group, {"train": 0, "val": 0, "test": 0})
        distribution[sample.scene_group][sample.split] += 1
    return distribution


def _scene_balanced_assignments(
    samples: list[DatasetSample],
    train_ratio: float,
    val_ratio: float,
) -> list[SplitAssignment]:
    unique_by_fingerprint = {sample.fingerprint: sample for sample in samples}
    by_scene: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in unique_by_fingerprint.values():
        by_scene[sample.scene_group].append(sample)

    assignments: list[SplitAssignment] = []
    for scene, scene_samples in sorted(by_scene.items()):
        ordered = sorted(scene_samples, key=lambda sample: sample.image_path.as_posix())
        train_cutoff = round(len(ordered) * train_ratio)
        val_cutoff = train_cutoff + round(len(ordered) * val_ratio)
        for index, sample in enumerate(ordered):
            split: SplitName = "train" if index < train_cutoff else "val" if index < val_cutoff else "test"
            assignments.append(
                SplitAssignment(
                    image_path=sample.image_path,
                    split=split,
                    reason=f"scene_balanced:{scene}",
                )
            )
    return assignments


def _recommendations(
    duplicates: list[DuplicateGroup],
    leakage: list[LeakagePair],
    samples: list[DatasetSample],
) -> list[str]:
    recommendations: list[str] = []
    if duplicates:
        recommendations.append("Filter duplicate frames before training or keep duplicates within a single split.")
    if leakage:
        recommendations.append("Fix train/val/test leakage by assigning duplicate fingerprints to only one split.")
    if samples and not any(sample.is_background for sample in samples):
        recommendations.append("Inject background-only images or mine hard negatives for precision stress tests.")
    if any(sample.has_small_object for sample in samples):
        recommendations.append("Oversample images containing small objects when recall is the priority.")
    recommendations.append("Use scene-balanced split planning before comparing model variants.")
    return recommendations
