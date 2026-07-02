"""YOLO dataset profiling utilities."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from pydantic import BaseModel, Field


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_AREA_THRESHOLD = 0.01
MEDIUM_AREA_THRESHOLD = 0.05


class BBoxStats(BaseModel):
    """Summary statistics for normalized YOLO bounding boxes."""

    width_mean: float = 0.0
    height_mean: float = 0.0
    area_mean: float = 0.0
    area_min: float | None = None
    area_max: float | None = None


class DatasetHealth(BaseModel):
    """Interpretable dataset quality score."""

    score: int = Field(default=0, ge=0, le=100)
    class_balance_score: float = 0.0
    box_size_distribution_score: float = 0.0
    annotation_noise_score: float = 0.0
    scene_diversity_score: float = 0.0
    duplication_penalty: float = 0.0
    train_val_leak_penalty: float = 0.0
    problems: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DatasetReport(BaseModel):
    """Serializable dataset profiling report."""

    data_yaml: Path
    dataset_root: Path
    scene: str = "generic"
    image_count: int = 0
    label_count: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    boxes_per_image: dict[str, float] = Field(default_factory=dict)
    bbox: BBoxStats = Field(default_factory=BBoxStats)
    object_size_ratio: dict[str, float] = Field(default_factory=dict)
    empty_label_images: int = 0
    missing_label_files: int = 0
    dataset_health: DatasetHealth = Field(default_factory=DatasetHealth)
    potential_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        """Write report JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def to_markdown(self, path: Path | str) -> None:
        """Write report Markdown."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown_text(), encoding="utf-8")

    def to_markdown_text(self) -> str:
        """Render a concise Markdown report."""
        lines = [
            "# Dataset Report",
            "",
            f"- Data YAML: `{self.data_yaml}`",
            f"- Dataset root: `{self.dataset_root}`",
            f"- Scene: `{self.scene}`",
            f"- Images: {self.image_count}",
            f"- Labels: {self.label_count}",
            f"- Empty label images: {self.empty_label_images}",
            f"- Missing label files: {self.missing_label_files}",
            f"- Dataset health: {self.dataset_health.score}/100",
            "",
            "## Dataset Health",
            "",
            f"- Class balance score: {self.dataset_health.class_balance_score:.1f}/25",
            f"- Box size distribution score: {self.dataset_health.box_size_distribution_score:.1f}/25",
            f"- Annotation noise score: {self.dataset_health.annotation_noise_score:.1f}/25",
            f"- Scene diversity score: {self.dataset_health.scene_diversity_score:.1f}/25",
            f"- Duplication penalty: {self.dataset_health.duplication_penalty:.1f}",
            f"- Train/val leak penalty: {self.dataset_health.train_val_leak_penalty:.1f}",
            "",
            "## Class Distribution",
            "",
        ]
        if self.class_distribution:
            lines.extend(f"- {name}: {count}" for name, count in self.class_distribution.items())
        else:
            lines.append("- No labeled objects found.")

        lines.extend(
            [
                "",
                "## Box Statistics",
                "",
                f"- Mean width: {self.bbox.width_mean:.4f}",
                f"- Mean height: {self.bbox.height_mean:.4f}",
                f"- Mean area: {self.bbox.area_mean:.4f}",
                f"- Small ratio: {self.object_size_ratio.get('small', 0.0):.4f}",
                f"- Medium ratio: {self.object_size_ratio.get('medium', 0.0):.4f}",
                f"- Large ratio: {self.object_size_ratio.get('large', 0.0):.4f}",
                "",
                "## Potential Issues",
                "",
            ]
        )
        lines.extend(f"- {issue}" for issue in self.potential_issues) if self.potential_issues else lines.append("- None.")
        lines.extend(["", "## Dataset Health Problems", ""])
        lines.extend(f"- {problem}" for problem in self.dataset_health.problems) if self.dataset_health.problems else lines.append("- None.")
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in self.recommendations) if self.recommendations else lines.append("- None.")
        lines.append("")
        return "\n".join(lines)


class DatasetProfiler:
    """Profile a YOLO-format dataset without image decoding dependencies."""

    def profile(self, data_yaml: Path | str) -> DatasetReport:
        """Read YOLO data.yaml and compute dataset statistics."""
        data_path = Path(data_yaml)
        data = _read_yaml_mapping(data_path)
        dataset_root = _dataset_root(data_path, data)
        scene = str(data.get("scene", data.get("scenario", "generic")))
        names = _class_names(data)
        images_by_split = _collect_images_by_split(data_path, data, dataset_root)
        image_paths = sorted(dict.fromkeys(path for paths in images_by_split.values() for path in paths))

        class_counts = {name: 0 for name in names}
        boxes_per_image: list[int] = []
        widths: list[float] = []
        heights: list[float] = []
        areas: list[float] = []
        missing_label_files = 0
        empty_label_images = 0
        potential_issues: list[str] = []

        for image_path in image_paths:
            label_path = _label_path_for_image(image_path)
            if not label_path.exists():
                missing_label_files += 1
                boxes_per_image.append(0)
                continue

            boxes = _read_label_file(label_path, len(names), potential_issues)
            if not boxes:
                empty_label_images += 1
            boxes_per_image.append(len(boxes))
            for class_id, width, height in boxes:
                class_name = names[class_id] if class_id < len(names) else str(class_id)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                widths.append(width)
                heights.append(height)
                areas.append(width * height)

        report = DatasetReport(
            data_yaml=data_path,
            dataset_root=dataset_root,
            scene=scene,
            image_count=len(image_paths),
            label_count=len(areas),
            class_distribution=class_counts,
            boxes_per_image=_boxes_per_image_stats(boxes_per_image),
            bbox=_bbox_stats(widths, heights, areas),
            object_size_ratio=_object_size_ratio(areas),
            empty_label_images=empty_label_images,
            missing_label_files=missing_label_files,
            potential_issues=potential_issues,
        )
        report.dataset_health = _dataset_health(report, images_by_split)
        report.potential_issues.extend(_dataset_issues(report))
        report.recommendations = _recommendations(report)
        return report


def profile_dataset(data_yaml: Path | str, out_prefix: Path | str) -> DatasetReport:
    """Profile a YOLO dataset and write JSON plus Markdown reports."""
    report = DatasetProfiler().profile(data_yaml)
    json_path, markdown_path = _output_paths(out_prefix)
    report.to_json(json_path)
    report.to_markdown(markdown_path)
    return report


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def _dataset_root(data_path: Path, data: dict[str, Any]) -> Path:
    raw_root = data.get("path")
    if raw_root is None:
        return data_path.parent
    root = Path(str(raw_root))
    return root if root.is_absolute() else data_path.parent / root


def _class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    nc = data.get("nc")
    if isinstance(nc, int):
        return [str(index) for index in range(nc)]
    return []


def _collect_images(data_path: Path, data: dict[str, Any], dataset_root: Path) -> list[Path]:
    images_by_split = _collect_images_by_split(data_path, data, dataset_root)
    return sorted(dict.fromkeys(path for paths in images_by_split.values() for path in paths))


def _collect_images_by_split(data_path: Path, data: dict[str, Any], dataset_root: Path) -> dict[str, list[Path]]:
    images_by_split: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        images: list[Path] = []
        raw_split = data.get(split)
        for item in _as_list(raw_split):
            images.extend(_images_from_split_item(data_path, dataset_root, item))
        images_by_split[split] = sorted(dict.fromkeys(path.resolve() for path in images))
    return images_by_split


def _images_from_split_item(data_path: Path, dataset_root: Path, item: object) -> list[Path]:
    split_path = Path(str(item))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    if split_path.is_dir():
        return [path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        return _images_from_list_file(split_path, data_path, dataset_root)
    if split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [split_path]
    return []


def _images_from_list_file(list_path: Path, data_path: Path, dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        image_path = Path(stripped)
        if not image_path.is_absolute():
            root = dataset_root if (dataset_root / image_path).exists() else data_path.parent
            image_path = root / image_path
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(image_path)
    return images


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def _read_label_file(
    label_path: Path,
    class_count: int,
    potential_issues: list[str],
) -> list[tuple[int, float, float]]:
    boxes: list[tuple[int, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            potential_issues.append(f"Malformed label row in {label_path}:{line_number}")
            continue
        try:
            class_id = int(float(parts[0]))
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            potential_issues.append(f"Non-numeric label row in {label_path}:{line_number}")
            continue
        if class_id < 0 or (class_count and class_id >= class_count):
            potential_issues.append(f"Class id out of range in {label_path}:{line_number}")
        if not 0 <= width <= 1 or not 0 <= height <= 1:
            potential_issues.append(f"Normalized bbox size out of range in {label_path}:{line_number}")
        boxes.append((class_id, max(width, 0.0), max(height, 0.0)))
    return boxes


def _boxes_per_image_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(mean(values)), "min": float(min(values)), "max": float(max(values))}


def _bbox_stats(widths: list[float], heights: list[float], areas: list[float]) -> BBoxStats:
    if not areas:
        return BBoxStats()
    return BBoxStats(
        width_mean=float(mean(widths)),
        height_mean=float(mean(heights)),
        area_mean=float(mean(areas)),
        area_min=float(min(areas)),
        area_max=float(max(areas)),
    )


def _object_size_ratio(areas: list[float]) -> dict[str, float]:
    if not areas:
        return {"small": 0.0, "medium": 0.0, "large": 0.0}
    small = sum(area < SMALL_AREA_THRESHOLD for area in areas)
    medium = sum(SMALL_AREA_THRESHOLD <= area < MEDIUM_AREA_THRESHOLD for area in areas)
    large = len(areas) - small - medium
    total = len(areas)
    return {"small": small / total, "medium": medium / total, "large": large / total}


def _dataset_issues(report: DatasetReport) -> list[str]:
    issues: list[str] = []
    if report.image_count == 0:
        issues.append("No images found from train/val/test entries.")
    if report.missing_label_files:
        issues.append(f"{report.missing_label_files} images are missing label files.")
    if report.image_count and report.empty_label_images / report.image_count > 0.3:
        issues.append("High empty-image ratio detected.")
    if report.label_count == 0:
        issues.append("No labeled objects found.")
    return issues


def _recommendations(report: DatasetReport) -> list[str]:
    recommendations: list[str] = []
    small_ratio = report.object_size_ratio.get("small", 0.0)
    if report.scene == "infrared_small_target" and small_ratio > 0.5:
        recommendations.append("Enable the small-object recipe for infrared small target detection.")
    if report.scene == "infrared_small_target" and report.image_count:
        empty_ratio = report.empty_label_images / report.image_count
        if empty_ratio > 0.3:
            recommendations.append("Many empty images found; check hard negative mining strategy.")
    if report.missing_label_files:
        recommendations.append("Create empty label files for intentional negative images or fix missing annotations.")
    recommendations.extend(report.dataset_health.recommendations)
    recommendations = list(dict.fromkeys(recommendations))
    return recommendations


def _dataset_health(report: DatasetReport, images_by_split: dict[str, list[Path]]) -> DatasetHealth:
    class_balance = _class_balance_score(report.class_distribution)
    box_size = _box_size_distribution_score(report.object_size_ratio, report.label_count)
    annotation_noise = _annotation_noise_score(report)
    scene_diversity = _scene_diversity_score(images_by_split)
    duplication_penalty = _duplication_penalty(images_by_split)
    train_val_leak_penalty = _train_val_leak_penalty(images_by_split)
    score = round(
        class_balance
        + box_size
        + annotation_noise
        + scene_diversity
        - duplication_penalty
        - train_val_leak_penalty
    )
    score = max(0, min(100, score))
    problems = _health_problems(report, duplication_penalty, train_val_leak_penalty)
    recommendations = _health_recommendations(problems)
    return DatasetHealth(
        score=score,
        class_balance_score=class_balance,
        box_size_distribution_score=box_size,
        annotation_noise_score=annotation_noise,
        scene_diversity_score=scene_diversity,
        duplication_penalty=duplication_penalty,
        train_val_leak_penalty=train_val_leak_penalty,
        problems=problems,
        recommendations=recommendations,
    )


def _class_balance_score(class_distribution: dict[str, int]) -> float:
    if not class_distribution:
        return 0.0
    counts = list(class_distribution.values())
    total = sum(counts)
    if total == 0:
        return 0.0
    nonzero = [count for count in counts if count > 0]
    if len(nonzero) <= 1 and len(counts) <= 1:
        return 25.0
    if len(nonzero) < len(counts):
        return 10.0
    imbalance = min(nonzero) / max(nonzero)
    return 25.0 * imbalance


def _box_size_distribution_score(object_size_ratio: dict[str, float], label_count: int) -> float:
    if label_count == 0:
        return 0.0
    small_ratio = object_size_ratio.get("small", 0.0)
    large_ratio = object_size_ratio.get("large", 0.0)
    if small_ratio > 0.8 or large_ratio > 0.95:
        return 10.0
    if small_ratio > 0.5 or large_ratio > 0.8:
        return 15.0
    return 25.0


def _annotation_noise_score(report: DatasetReport) -> float:
    if report.image_count == 0:
        return 0.0
    malformed_issue_count = sum(
        "Malformed label row" in issue
        or "Non-numeric label row" in issue
        or "Class id out of range" in issue
        or "Normalized bbox size out of range" in issue
        for issue in report.potential_issues
    )
    missing_ratio = report.missing_label_files / report.image_count
    noise_penalty = min(25.0, malformed_issue_count * 4.0 + missing_ratio * 25.0)
    return max(0.0, 25.0 - noise_penalty)


def _scene_diversity_score(images_by_split: dict[str, list[Path]]) -> float:
    image_paths = [path for paths in images_by_split.values() for path in paths]
    if not image_paths:
        return 0.0
    parent_count = len({path.parent for path in image_paths})
    split_count = sum(bool(paths) for paths in images_by_split.values())
    if parent_count >= 3 or split_count >= 3:
        return 25.0
    if parent_count == 2 or split_count == 2:
        return 18.0
    return 12.0


def _duplication_penalty(images_by_split: dict[str, list[Path]]) -> float:
    image_paths = [path for paths in images_by_split.values() for path in paths]
    if not image_paths:
        return 0.0
    fingerprints = [_image_fingerprint(path) for path in image_paths]
    duplicate_count = len(fingerprints) - len(set(fingerprints))
    duplicate_ratio = duplicate_count / len(fingerprints)
    return min(15.0, duplicate_ratio * 30.0)


def _train_val_leak_penalty(images_by_split: dict[str, list[Path]]) -> float:
    train = {_image_fingerprint(path) for path in images_by_split.get("train", [])}
    val = {_image_fingerprint(path) for path in images_by_split.get("val", [])}
    if not train or not val:
        return 0.0
    leak_count = len(train & val)
    leak_ratio = leak_count / max(1, len(val))
    return min(15.0, leak_ratio * 30.0)


def _image_fingerprint(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    return f"{path.stem.lower()}:{size}"


def _health_problems(
    report: DatasetReport,
    duplication_penalty: float,
    train_val_leak_penalty: float,
) -> list[str]:
    problems: list[str] = []
    small_ratio = report.object_size_ratio.get("small", 0.0)
    if small_ratio > 0.5:
        problems.append("severe_small_object_bias")
    if duplication_penalty >= 5.0:
        problems.append("high_duplicate_frames")
    if report.image_count and report.empty_label_images / report.image_count < 0.05:
        problems.append("missing_hard_backgrounds")
    if report.missing_label_files or any("label row" in issue for issue in report.potential_issues):
        problems.append("annotation_noise")
    if _has_long_tail(report.class_distribution):
        problems.append("class_imbalance_long_tail")
    if train_val_leak_penalty > 0:
        problems.append("train_val_leakage")
    return problems


def _health_recommendations(problems: list[str]) -> list[str]:
    recommendations: list[str] = []
    if "missing_hard_backgrounds" in problems:
        recommendations.append("add_background_only_images")
    if "class_imbalance_long_tail" in problems:
        recommendations.append("re_sample_long_tail_classes")
    if "annotation_noise" in problems:
        recommendations.append("relabel_3_percent_suspicious_boxes")
    if "severe_small_object_bias" in problems:
        recommendations.append("enable_small_object_recipe")
    if "high_duplicate_frames" in problems:
        recommendations.append("deduplicate_near_duplicate_frames")
    if "train_val_leakage" in problems:
        recommendations.append("fix_train_val_split_leakage")
    return recommendations


def _has_long_tail(class_distribution: dict[str, int]) -> bool:
    nonzero = [count for count in class_distribution.values() if count > 0]
    return len(nonzero) > 1 and min(nonzero) / max(nonzero) < 0.2


def _output_paths(out_prefix: Path | str) -> tuple[Path, Path]:
    prefix = Path(out_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".json"), prefix.with_suffix(".md")


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
