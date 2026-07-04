"""Active-learning mining loop for unlabeled inference results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


MiningStrategy = Literal["low_confidence", "high_entropy", "model_disagreement"]
LabelingTarget = Literal["cvat", "label_studio", "generic"]


class PredictionSummary(BaseModel):
    """Compact prediction summary for one unlabeled sample."""

    image_path: Path
    max_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    class_probabilities: list[float] = Field(default_factory=list)
    model_predictions: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MiningConfig(BaseModel):
    """Thresholds and limits for active-learning sample mining."""

    low_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    high_entropy_threshold: float = Field(default=0.7, ge=0.0)
    disagreement_threshold: float = Field(default=0.34, ge=0.0, le=1.0)
    max_samples: int = Field(default=100, gt=0)
    strategies: list[MiningStrategy] = Field(
        default_factory=lambda: ["low_confidence", "high_entropy", "model_disagreement"]
    )


class MinedSample(BaseModel):
    """A selected sample and the active-learning reasons behind it."""

    image_path: Path
    score: float
    reasons: list[MiningStrategy]
    details: dict[str, float] = Field(default_factory=dict)

    @field_serializer("image_path")
    def serialize_image_path(self, value: Path) -> str:
        """Serialize manifests with portable POSIX-style paths."""
        return value.as_posix()


class LabelingManifest(BaseModel):
    """Portable manifest for CVAT, Label Studio, or a generic labeling queue."""

    target: LabelingTarget = "generic"
    dataset_version: str
    next_dataset_version: str
    samples: list[MinedSample]

    def to_json(self, path: Path | str) -> None:
        """Write manifest JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class ActiveLearningPlan(BaseModel):
    """Inference-to-mining-to-relabel plan."""

    dataset_version: str
    next_dataset_version: str
    strategy_counts: dict[MiningStrategy, int] = Field(default_factory=dict)
    mined_samples: list[MinedSample] = Field(default_factory=list)
    labeling_manifest: LabelingManifest
    recommendations: list[str] = Field(default_factory=list)


class LabelHandoffResult(BaseModel):
    """Result of handing mined samples to a labeling queue."""

    target: LabelingTarget = "generic"
    dataset_version: str
    next_dataset_version: str
    sample_count: int
    labeling_manifest_path: Path
    status: Literal["ready_for_labeling"] = "ready_for_labeling"

    @field_serializer("labeling_manifest_path")
    def serialize_labeling_manifest_path(self, value: Path) -> str:
        """Serialize handoff paths portably."""
        return value.as_posix()


class DatasetPromotionPlan(BaseModel):
    """A controlled plan for promoting reviewed labels into the next dataset version."""

    dataset_version: str
    next_dataset_version: str
    labeling_manifest_path: Path
    status: Literal["pending_label_review", "ready_to_promote"] = "pending_label_review"
    promoted: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_serializer("labeling_manifest_path")
    def serialize_labeling_manifest_path(self, value: Path) -> str:
        """Serialize promotion paths portably."""
        return value.as_posix()


class ActiveLearningMiner:
    """Mine unlabeled samples from prediction summaries."""

    def __init__(self, config: MiningConfig | None = None) -> None:
        self.config = config or MiningConfig()

    def mine(
        self,
        predictions: list[PredictionSummary],
        dataset_version: str,
        labeling_target: LabelingTarget = "generic",
    ) -> ActiveLearningPlan:
        """Create an active-learning relabeling plan."""
        mined: list[MinedSample] = []
        for prediction in predictions:
            sample = self._score_prediction(prediction)
            if sample is not None:
                mined.append(sample)

        mined.sort(key=lambda item: item.score, reverse=True)
        mined = mined[: self.config.max_samples]
        next_version = increment_dataset_version(dataset_version)
        strategy_counts = {
            strategy: sum(strategy in sample.reasons for sample in mined)
            for strategy in self.config.strategies
        }
        manifest = LabelingManifest(
            target=labeling_target,
            dataset_version=dataset_version,
            next_dataset_version=next_version,
            samples=mined,
        )
        return ActiveLearningPlan(
            dataset_version=dataset_version,
            next_dataset_version=next_version,
            strategy_counts=strategy_counts,
            mined_samples=mined,
            labeling_manifest=manifest,
            recommendations=_recommendations(strategy_counts),
        )

    def _score_prediction(self, prediction: PredictionSummary) -> MinedSample | None:
        reasons: list[MiningStrategy] = []
        details: dict[str, float] = {}
        score = 0.0

        if "low_confidence" in self.config.strategies and prediction.max_confidence is not None:
            low_confidence_score = 1.0 - prediction.max_confidence
            details["low_confidence"] = low_confidence_score
            if prediction.max_confidence <= self.config.low_confidence_threshold:
                reasons.append("low_confidence")
                score += low_confidence_score

        if "high_entropy" in self.config.strategies and prediction.class_probabilities:
            entropy = normalized_entropy(prediction.class_probabilities)
            details["entropy"] = entropy
            if entropy >= self.config.high_entropy_threshold:
                reasons.append("high_entropy")
                score += entropy

        if "model_disagreement" in self.config.strategies and prediction.model_predictions:
            disagreement = disagreement_rate(prediction.model_predictions)
            details["disagreement"] = disagreement
            if disagreement >= self.config.disagreement_threshold:
                reasons.append("model_disagreement")
                score += disagreement

        if not reasons:
            return None
        return MinedSample(
            image_path=prediction.image_path,
            score=score,
            reasons=reasons,
            details=details,
        )


def load_prediction_summaries(path: Path | str) -> list[PredictionSummary]:
    """Load prediction summaries from a JSON file."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8-sig") as file:
        raw = json.load(file)
    items = raw.get("predictions", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Active-learning predictions must be a list or contain a 'predictions' list.")
    return [PredictionSummary.model_validate(item) for item in items]


def normalized_entropy(probabilities: list[float]) -> float:
    """Compute normalized entropy in [0, 1] for class probabilities."""
    cleaned = [max(probability, 0.0) for probability in probabilities]
    total = sum(cleaned)
    if total <= 0 or len(cleaned) <= 1:
        return 0.0
    normalized = [probability / total for probability in cleaned if probability > 0]
    entropy = -sum(probability * math.log(probability) for probability in normalized)
    return entropy / math.log(len(cleaned))


def disagreement_rate(model_predictions: list[str]) -> float:
    """Estimate ensemble disagreement from predicted labels."""
    if len(model_predictions) <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for prediction in model_predictions:
        counts[prediction] = counts.get(prediction, 0) + 1
    majority = max(counts.values())
    return 1.0 - majority / len(model_predictions)


def increment_dataset_version(version: str) -> str:
    """Increment simple dataset versions such as v1, dataset_v2, or 3."""
    digits = ""
    for character in reversed(version):
        if not character.isdigit():
            break
        digits = character + digits
    if not digits:
        return f"{version}_v2"
    prefix = version[: -len(digits)]
    return f"{prefix}{int(digits) + 1}"


def _recommendations(strategy_counts: dict[MiningStrategy, int]) -> list[str]:
    recommendations: list[str] = []
    if strategy_counts.get("low_confidence", 0):
        recommendations.append("Send low-confidence samples to labeling queue for relabel review.")
    if strategy_counts.get("high_entropy", 0):
        recommendations.append("Prioritize high-entropy samples to clarify ambiguous class boundaries.")
    if strategy_counts.get("model_disagreement", 0):
        recommendations.append("Use ensemble-disagreement samples for hard-case adjudication.")
    if not recommendations:
        recommendations.append("No active-learning samples selected with current thresholds.")
    return recommendations
