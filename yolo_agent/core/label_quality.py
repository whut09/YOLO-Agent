"""Label quality analysis for YOLO-format datasets and prediction files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IssueType = Literal[
    "suspected_missing_label",
    "suspicious_box_geometry",
    "class_confusion",
    "low_class_coverage",
]


class AnnotationRuleThresholds(BaseModel):
    """Configurable thresholds for label quality checks."""

    high_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    match_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    near_iou: float = Field(default=0.2, ge=0.0, le=1.0)
    min_box_width: float = Field(default=0.002, ge=0.0, le=1.0)
    min_box_height: float = Field(default=0.002, ge=0.0, le=1.0)
    max_box_width: float = Field(default=0.98, ge=0.0, le=1.0)
    max_box_height: float = Field(default=0.98, ge=0.0, le=1.0)
    min_box_area: float = Field(default=0.000004, ge=0.0, le=1.0)
    max_box_area: float = Field(default=0.95, ge=0.0, le=1.0)
    max_aspect_ratio: float = Field(default=12.0, gt=0.0)
    min_class_instances: int = Field(default=5, ge=0)


class AnnotationReviewConfig(BaseModel):
    """Limits for review-oriented outputs."""

    max_samples_per_reason: int = Field(default=50, ge=1)
    max_boxes_to_redraw: int = Field(default=100, ge=1)


class AnnotationRules(BaseModel):
    """Rules and messages used by label quality analysis."""

    thresholds: AnnotationRuleThresholds = Field(default_factory=AnnotationRuleThresholds)
    review: AnnotationReviewConfig = Field(default_factory=AnnotationReviewConfig)
    recommendations: dict[IssueType, str] = Field(default_factory=dict)


class YoloBox(BaseModel):
    """Normalized YOLO bounding box."""

    class_id: int = Field(ge=0)
    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)
    source: str = ""

    @property
    def area(self) -> float:
        """Return normalized box area."""
        return self.width * self.height


class PredictionBox(YoloBox):
    """Normalized prediction box with confidence."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ImagePredictions(BaseModel):
    """Prediction boxes for one image."""

    image: str
    boxes: list[PredictionBox] = Field(default_factory=list)


class LabelQualityIssue(BaseModel):
    """One actionable label quality issue."""

    issue_type: IssueType
    image: str | None = None
    class_name: str | None = None
    predicted_class: str | None = None
    target_class: str | None = None
    confidence: float | None = None
    iou: float | None = None
    box: YoloBox | None = None
    message: str
    severity: Literal["low", "medium", "high"] = "medium"


class LabelQualityReport(BaseModel):
    """Serializable label quality report."""

    data_yaml: Path
    dataset_root: Path
    class_names: list[str] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    issues: list[LabelQualityIssue] = Field(default_factory=list)
    suspicious_missing_labels: list[LabelQualityIssue] = Field(default_factory=list)
    suspicious_boxes: list[LabelQualityIssue] = Field(default_factory=list)
    class_confusions: dict[str, int] = Field(default_factory=dict)
    low_coverage_classes: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        """Write report JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def default_annotation_rules_path() -> Path:
    """Return bundled annotation quality rules."""
    return Path(__file__).resolve().parents[2] / "configs" / "annotation_rules.yaml"


def load_annotation_rules(path: Path | str | None = None) -> AnnotationRules:
    """Load annotation quality rules from YAML."""
    rule_path = Path(path) if path is not None else default_annotation_rules_path()
    with rule_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Annotation rules YAML must contain a mapping: {rule_path}")
    return AnnotationRules.model_validate(data)


def analyze_label_quality(
    data_yaml: Path | str,
    predictions_path: Path | str | None = None,
    rules_path: Path | str | None = None,
) -> LabelQualityReport:
    """Analyze YOLO labels and optional predictions for annotation advice signals."""
    rules = load_annotation_rules(rules_path)
    data_path = Path(data_yaml)
    data = _read_yaml_mapping(data_path)
    root = _dataset_root(data_path, data)
    class_names = _class_names(data)
    image_paths = _collect_images(data_path, data, root)
    labels_by_image = {image_path: _read_label_file(_label_path_for_image(image_path), len(class_names)) for image_path in image_paths}
    predictions = _read_predictions(predictions_path) if predictions_path is not None else {}

    class_counts: Counter[str] = Counter()
    issues: list[LabelQualityIssue] = []
    suspicious_boxes: list[LabelQualityIssue] = []
    suspicious_missing: list[LabelQualityIssue] = []
    confusion_counts: Counter[str] = Counter()

    for image_path, boxes in labels_by_image.items():
        image_key = _relative_key(image_path, root)
        for box in boxes:
            class_name = _class_name(class_names, box.class_id)
            class_counts[class_name] += 1
            issue = _geometry_issue(image_key, class_name, box, rules)
            if issue is not None:
                suspicious_boxes.append(issue)
                issues.append(issue)

        for pred in _predictions_for_image(predictions, image_path, root):
            if pred.confidence < rules.thresholds.high_confidence:
                continue
            best_box, best_iou = _best_match(pred, boxes)
            pred_class = _class_name(class_names, pred.class_id)
            if best_box is None or best_iou < rules.thresholds.near_iou:
                issue = LabelQualityIssue(
                    issue_type="suspected_missing_label",
                    image=image_key,
                    predicted_class=pred_class,
                    confidence=pred.confidence,
                    box=pred,
                    message=f"High-confidence {pred_class} prediction has no nearby GT box.",
                    severity="high",
                )
                suspicious_missing.append(issue)
                issues.append(issue)
                continue
            if best_iou >= rules.thresholds.match_iou and best_box.class_id != pred.class_id:
                target_class = _class_name(class_names, best_box.class_id)
                pair = f"{target_class}->{pred_class}"
                confusion_counts[pair] += 1
                issue = LabelQualityIssue(
                    issue_type="class_confusion",
                    image=image_key,
                    predicted_class=pred_class,
                    target_class=target_class,
                    confidence=pred.confidence,
                    iou=best_iou,
                    box=best_box,
                    message=f"Prediction overlaps {target_class} GT but predicts {pred_class}.",
                    severity="medium",
                )
                issues.append(issue)

    low_coverage = [
        class_name
        for class_name in class_names
        if class_counts.get(class_name, 0) < rules.thresholds.min_class_instances
    ]
    for class_name in low_coverage:
        issues.append(
            LabelQualityIssue(
                issue_type="low_class_coverage",
                class_name=class_name,
                message=f"Class {class_name} has fewer than {rules.thresholds.min_class_instances} labeled instances.",
                severity="medium",
            )
        )

    return LabelQualityReport(
        data_yaml=data_path,
        dataset_root=root,
        class_names=class_names,
        class_distribution={class_name: class_counts.get(class_name, 0) for class_name in class_names},
        issues=issues,
        suspicious_missing_labels=suspicious_missing[: rules.review.max_samples_per_reason],
        suspicious_boxes=suspicious_boxes[: rules.review.max_boxes_to_redraw],
        class_confusions=dict(confusion_counts),
        low_coverage_classes=low_coverage,
        recommendations=_recommendations(issues, rules),
    )


def _geometry_issue(
    image_key: str,
    class_name: str,
    box: YoloBox,
    rules: AnnotationRules,
) -> LabelQualityIssue | None:
    thresholds = rules.thresholds
    reasons: list[str] = []
    if box.width <= thresholds.min_box_width or box.height <= thresholds.min_box_height:
        reasons.append("box is extremely small")
    if box.width >= thresholds.max_box_width or box.height >= thresholds.max_box_height:
        reasons.append("box covers almost the full image")
    if box.area <= thresholds.min_box_area or box.area >= thresholds.max_box_area:
        reasons.append("box area is outside configured bounds")
    aspect_ratio = max(box.width / max(box.height, 1e-12), box.height / max(box.width, 1e-12))
    if aspect_ratio >= thresholds.max_aspect_ratio:
        reasons.append("box aspect ratio is extreme")
    if not reasons:
        return None
    return LabelQualityIssue(
        issue_type="suspicious_box_geometry",
        image=image_key,
        class_name=class_name,
        box=box,
        message=f"Suspicious {class_name} label: {', '.join(reasons)}.",
        severity="medium",
    )


def _recommendations(issues: list[LabelQualityIssue], rules: AnnotationRules) -> list[str]:
    issue_types = {issue.issue_type for issue in issues}
    return [
        message
        for issue_type, message in rules.recommendations.items()
        if issue_type in issue_types
    ]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
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
        return [str(names[index]) for index in sorted(names)]
    nc = int(data.get("nc", 0) or 0)
    return [str(index) for index in range(nc)]


def _collect_images(data_path: Path, data: dict[str, Any], root: Path) -> list[Path]:
    images: list[Path] = []
    for split in ("train", "val", "test"):
        raw = data.get(split)
        if raw is None:
            continue
        raw_paths = raw if isinstance(raw, list) else [raw]
        for raw_path in raw_paths:
            path = Path(str(raw_path))
            path = path if path.is_absolute() else root / path
            if path.is_file() and path.suffix == ".txt":
                images.extend(_read_image_list(path, data_path.parent, root))
            elif path.is_dir():
                images.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS))
            elif path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return sorted(dict.fromkeys(images))


def _read_image_list(path: Path, data_dir: Path, root: Path) -> list[Path]:
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = root / image_path if (root / image_path).exists() else data_dir / image_path
        images.append(image_path)
    return images


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_label_file(path: Path, class_count: int) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            if class_count and class_id >= class_count:
                continue
            boxes.append(
                YoloBox(
                    class_id=class_id,
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    source=f"{path}:{line_number}",
                )
            )
        except ValueError:
            continue
    return boxes


def _read_predictions(path: Path | str) -> dict[str, list[PredictionBox]]:
    prediction_path = Path(path)
    if prediction_path.suffix.lower() == ".json":
        data = json.loads(prediction_path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(prediction_path.read_text(encoding="utf-8")) or {}
    raw_predictions = data.get("predictions", data) if isinstance(data, dict) else data
    if not isinstance(raw_predictions, list):
        raise ValueError("Prediction file must contain a list or a 'predictions' list.")
    predictions: dict[str, list[PredictionBox]] = defaultdict(list)
    for item in raw_predictions:
        image_predictions = ImagePredictions.model_validate(item)
        predictions[_normalize_key(image_predictions.image)].extend(image_predictions.boxes)
    return dict(predictions)


def _predictions_for_image(
    predictions: dict[str, list[PredictionBox]],
    image_path: Path,
    root: Path,
) -> list[PredictionBox]:
    keys = {
        _normalize_key(str(image_path)),
        _normalize_key(_relative_key(image_path, root)),
        _normalize_key(image_path.name),
        _normalize_key(image_path.stem),
    }
    boxes: list[PredictionBox] = []
    for key in keys:
        boxes.extend(predictions.get(key, []))
    return boxes


def _best_match(prediction: PredictionBox, boxes: list[YoloBox]) -> tuple[YoloBox | None, float]:
    best_box: YoloBox | None = None
    best_iou = 0.0
    for box in boxes:
        iou = _iou(prediction, box)
        if iou > best_iou:
            best_iou = iou
            best_box = box
    return best_box, best_iou


def _iou(left: YoloBox, right: YoloBox) -> float:
    left_x1, left_y1, left_x2, left_y2 = _xyxy(left)
    right_x1, right_y1, right_x2, right_y2 = _xyxy(right)
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = left.area + right.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2,
        box.y_center - box.height / 2,
        box.x_center + box.width / 2,
        box.y_center + box.height / 2,
    )


def _relative_key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_key(value: str) -> str:
    return Path(value).as_posix().replace("\\", "/")


def _class_name(class_names: list[str], class_id: int) -> str:
    return class_names[class_id] if class_id < len(class_names) else str(class_id)
