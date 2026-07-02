"""Annotation advice from label quality signals."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from yolo_agent.core.label_quality import LabelQualityIssue, LabelQualityReport, analyze_label_quality


class AnnotationAdviceReport(BaseModel):
    """Actionable annotation worklist for dataset improvement."""

    label_quality: LabelQualityReport
    classes_to_collect: list[str] = Field(default_factory=list)
    scenes_to_annotate: list[str] = Field(default_factory=list)
    samples_for_review: list[str] = Field(default_factory=list)
    boxes_to_redraw: list[LabelQualityIssue] = Field(default_factory=list)
    labeling_tool_targets: list[str] = Field(default_factory=list)
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
        """Render an annotation advice report."""
        lines = [
            "# Annotation Advice",
            "",
            f"- Dataset: `{self.label_quality.data_yaml}`",
            f"- Issues: {len(self.label_quality.issues)}",
            f"- Suspected missing labels: {len(self.label_quality.suspicious_missing_labels)}",
            f"- Suspicious boxes: {len(self.label_quality.suspicious_boxes)}",
            "",
            "## Classes To Collect",
            "",
        ]
        lines.extend(f"- {name}" for name in self.classes_to_collect) if self.classes_to_collect else lines.append("- None.")
        lines.extend(["", "## Scenes To Annotate", ""])
        lines.extend(f"- {item}" for item in self.scenes_to_annotate) if self.scenes_to_annotate else lines.append("- None.")
        lines.extend(["", "## Samples For Review", ""])
        lines.extend(f"- `{sample}`" for sample in self.samples_for_review) if self.samples_for_review else lines.append("- None.")
        lines.extend(["", "## Boxes To Redraw", ""])
        if self.boxes_to_redraw:
            lines.extend(f"- `{issue.image}`: {issue.message}" for issue in self.boxes_to_redraw)
        else:
            lines.append("- None.")
        lines.extend(["", "## Class Confusions", ""])
        if self.label_quality.class_confusions:
            lines.extend(f"- {pair}: {count}" for pair, count in self.label_quality.class_confusions.items())
        else:
            lines.append("- None.")
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in self.recommendations) if self.recommendations else lines.append("- None.")
        lines.append("")
        return "\n".join(lines)


class AnnotationAdvisor:
    """Build annotation worklists from label quality analysis."""

    def advise(
        self,
        data_yaml: Path | str,
        predictions_path: Path | str | None = None,
        rules_path: Path | str | None = None,
    ) -> AnnotationAdviceReport:
        """Analyze labels and return actionable annotation advice."""
        quality = analyze_label_quality(data_yaml, predictions_path, rules_path)
        samples = _review_samples(quality)
        boxes = quality.suspicious_boxes
        recommendations = list(dict.fromkeys([*quality.recommendations, *_advisor_recommendations(quality)]))
        return AnnotationAdviceReport(
            label_quality=quality,
            classes_to_collect=quality.low_coverage_classes,
            scenes_to_annotate=_scene_advice(quality),
            samples_for_review=samples,
            boxes_to_redraw=boxes,
            labeling_tool_targets=samples,
            recommendations=recommendations,
        )


def advise_annotations(
    data_yaml: Path | str,
    out_prefix: Path | str,
    predictions_path: Path | str | None = None,
    rules_path: Path | str | None = None,
) -> AnnotationAdviceReport:
    """Create annotation advice and write JSON plus Markdown reports."""
    report = AnnotationAdvisor().advise(data_yaml, predictions_path, rules_path)
    json_path, markdown_path = _output_paths(out_prefix)
    report.to_json(json_path)
    report.to_markdown(markdown_path)
    return report


def _review_samples(quality: LabelQualityReport) -> list[str]:
    samples = [
        issue.image
        for issue in [*quality.suspicious_missing_labels, *quality.suspicious_boxes]
        if issue.image is not None
    ]
    return list(dict.fromkeys(samples))


def _scene_advice(quality: LabelQualityReport) -> list[str]:
    advice: list[str] = []
    if quality.suspicious_missing_labels:
        advice.append("Annotate scenes represented by high-confidence unmatched predictions.")
    if quality.class_confusions:
        advice.append("Add clarified examples for scenes where confused classes co-occur.")
    if quality.low_coverage_classes:
        advice.append("Collect scenes containing underrepresented classes.")
    return advice


def _advisor_recommendations(quality: LabelQualityReport) -> list[str]:
    recommendations: list[str] = []
    if quality.suspicious_missing_labels:
        recommendations.append("Queue suspected missed-object images for Label Studio or CVAT review.")
    if quality.suspicious_boxes:
        recommendations.append("Redraw suspicious boxes before running ablations that depend on localization metrics.")
    if quality.class_confusions:
        recommendations.append("Review class naming guidelines and add side-by-side examples for confused classes.")
    if quality.low_coverage_classes:
        recommendations.append("Prioritize data collection for low-coverage classes before increasing model complexity.")
    return recommendations


def _output_paths(out_prefix: Path | str) -> tuple[Path, Path]:
    prefix = Path(out_prefix)
    if prefix.suffix:
        return prefix.with_suffix(".json"), prefix.with_suffix(".md")
    return Path(f"{prefix}.json"), Path(f"{prefix}.md")
