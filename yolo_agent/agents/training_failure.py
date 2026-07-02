"""Training failure mode diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


TrainingFailureMode = Literal[
    "overfitting",
    "underfitting",
    "unstable_loss",
    "low_recall",
    "low_precision",
]


class TrainingRunSignals(BaseModel):
    """Metrics and context used to diagnose training failures."""

    train_loss: float | None = None
    val_loss: float | None = None
    loss_history: list[float] = Field(default_factory=list)
    recall: float | None = None
    precision: float | None = None
    map50: float | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    notes: list[str] = Field(default_factory=list)


class FailureModePolicy(BaseModel):
    """Configured diagnosis and action policy for one failure mode."""

    diagnosis: str
    actions: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


class FailureDiagnosis(BaseModel):
    """One diagnosed training failure."""

    mode: TrainingFailureMode
    severity: Literal["low", "medium", "high"]
    diagnosis: str
    actions: list[str]
    evidence: list[str] = Field(default_factory=list)


class TrainingFailureReport(BaseModel):
    """Training failure diagnosis report."""

    diagnoses: list[FailureDiagnosis] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether no failure mode was detected."""
        return not self.diagnoses


class TrainingFailureDiagnoser:
    """Diagnose common YOLO training failure modes from metrics."""

    def __init__(self, policies: dict[TrainingFailureMode, FailureModePolicy], thresholds: dict[str, float]) -> None:
        self.policies = policies
        self.thresholds = thresholds

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "TrainingFailureDiagnoser":
        """Load failure mode policies from YAML."""
        policy_path = Path(path) if path is not None else default_training_failure_path()
        with policy_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Training failure YAML must contain a mapping: {policy_path}")
        raw_modes = data.get("modes", {})
        if not isinstance(raw_modes, dict):
            raise ValueError("Training failure YAML requires a 'modes' mapping.")
        policies = {
            mode: FailureModePolicy.model_validate(policy)
            for mode, policy in raw_modes.items()
            if isinstance(policy, dict)
        }
        thresholds = data.get("thresholds", {})
        return cls(
            policies=policies,  # type: ignore[arg-type]
            thresholds={str(key): float(value) for key, value in thresholds.items()} if isinstance(thresholds, dict) else {},
        )

    def diagnose(self, signals: TrainingRunSignals) -> TrainingFailureReport:
        """Diagnose training failure modes."""
        diagnoses: list[FailureDiagnosis] = []
        for mode, evidence in (
            ("overfitting", self._overfitting(signals)),
            ("underfitting", self._underfitting(signals)),
            ("unstable_loss", self._unstable_loss(signals)),
            ("low_recall", self._low_recall(signals)),
            ("low_precision", self._low_precision(signals)),
        ):
            if evidence:
                diagnoses.append(self._diagnosis(mode, evidence))  # type: ignore[arg-type]
        return TrainingFailureReport(diagnoses=diagnoses)

    def _diagnosis(self, mode: TrainingFailureMode, evidence: list[str]) -> FailureDiagnosis:
        policy = self.policies[mode]
        return FailureDiagnosis(
            mode=mode,
            severity=policy.severity,
            diagnosis=policy.diagnosis,
            actions=policy.actions,
            evidence=evidence,
        )

    def _overfitting(self, signals: TrainingRunSignals) -> list[str]:
        if signals.train_loss is None or signals.val_loss is None:
            return []
        gap = signals.val_loss - signals.train_loss
        threshold = self.thresholds.get("overfit_loss_gap", 0.25)
        if gap > threshold:
            return [f"val_loss - train_loss = {gap:.4f} > {threshold:.4f}"]
        return []

    def _underfitting(self, signals: TrainingRunSignals) -> list[str]:
        if signals.train_loss is None or signals.val_loss is None:
            return []
        threshold = self.thresholds.get("high_loss", 1.0)
        if signals.train_loss > threshold and signals.val_loss > threshold:
            return [f"train_loss={signals.train_loss:.4f} and val_loss={signals.val_loss:.4f} are both high"]
        return []

    def _unstable_loss(self, signals: TrainingRunSignals) -> list[str]:
        if len(signals.loss_history) < 3:
            return []
        minimum = min(signals.loss_history)
        maximum = max(signals.loss_history)
        if minimum <= 0:
            return []
        ratio = (maximum - minimum) / minimum
        threshold = self.thresholds.get("unstable_loss_delta_ratio", 0.35)
        if ratio > threshold:
            return [f"loss oscillation ratio={ratio:.4f} > {threshold:.4f}"]
        return []

    def _low_recall(self, signals: TrainingRunSignals) -> list[str]:
        if signals.recall is None:
            return []
        threshold = self.thresholds.get("low_recall", 0.5)
        if signals.recall < threshold:
            return [f"recall={signals.recall:.4f} < {threshold:.4f}"]
        return []

    def _low_precision(self, signals: TrainingRunSignals) -> list[str]:
        if signals.precision is None:
            return []
        threshold = self.thresholds.get("low_precision", 0.5)
        if signals.precision < threshold:
            return [f"precision={signals.precision:.4f} < {threshold:.4f}"]
        return []


def default_training_failure_path() -> Path:
    """Return bundled training failure mode library."""
    return Path(__file__).resolve().parents[2] / "configs" / "training_failure_modes.yaml"

