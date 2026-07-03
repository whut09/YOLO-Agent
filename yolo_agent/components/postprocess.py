"""Post-processing strategy registry and recommender."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.utils import dedupe_list


PostProcessFamily = Literal[
    "nms",
    "fusion",
    "threshold",
    "calibration",
    "scale",
    "tta",
    "slicing",
]
LatencyCost = Literal["low", "medium", "high"]
AccuracyRisk = Literal["low", "medium", "high"]


class PostProcessStrategy(BaseModel):
    """Metadata for an inference-time post-processing strategy."""

    id: str
    name: str
    family: PostProcessFamily
    description: str = ""
    target_scenarios: list[str] = Field(default_factory=list)
    target_problems: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    companion_actions: list[str] = Field(default_factory=list)
    latency_cost: LatencyCost = "low"
    accuracy_risk: AccuracyRisk = "low"
    deployment_notes: list[str] = Field(default_factory=list)


class PostProcessRecommendation(BaseModel):
    """Recommended post-processing bundle for a scenario."""

    scenario: str
    recommended_postprocess: list[PostProcessStrategy]
    rationale: list[str] = Field(default_factory=list)
    companion_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        """Return recommended strategy ids."""
        return [strategy.id for strategy in self.recommended_postprocess]


class PostProcessRegistry:
    """In-memory registry of post-processing strategies."""

    def __init__(
        self,
        strategies: list[PostProcessStrategy],
        recommendations: dict[str, list[str]],
        error_recommendations: dict[str, list[str]] | None = None,
    ) -> None:
        self.strategies = strategies
        self.recommendations = recommendations
        self.error_recommendations = error_recommendations or {}

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "PostProcessRegistry":
        """Load strategies and scenario recommendations from YAML."""
        registry_path = Path(path) if path is not None else default_postprocess_registry_path()
        with registry_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Post-process registry YAML must contain a mapping: {registry_path}")

        raw_strategies = data.get("strategies", {})
        if not isinstance(raw_strategies, dict):
            raise ValueError("Post-process registry requires a 'strategies' mapping.")
        strategies = [
            PostProcessStrategy.model_validate({"id": strategy_id, **strategy_data})
            for strategy_id, strategy_data in raw_strategies.items()
            if isinstance(strategy_data, dict)
        ]

        raw_recommendations = data.get("recommendations", {})
        recommendations = {
            str(scenario): [str(item) for item in strategy_ids]
            for scenario, strategy_ids in raw_recommendations.items()
            if isinstance(strategy_ids, list)
        } if isinstance(raw_recommendations, dict) else {}

        raw_error_recommendations = data.get("error_recommendations", {})
        error_recommendations = {
            str(error_type): [str(item) for item in strategy_ids]
            for error_type, strategy_ids in raw_error_recommendations.items()
            if isinstance(strategy_ids, list)
        } if isinstance(raw_error_recommendations, dict) else {}
        return cls(
            strategies=strategies,
            recommendations=recommendations,
            error_recommendations=error_recommendations,
        )

    def get(self, strategy_id: str) -> PostProcessStrategy:
        """Return one strategy by id."""
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return strategy
        raise KeyError(f"Unknown post-process strategy: {strategy_id}")

    def get_by_family(self, family: PostProcessFamily) -> list[PostProcessStrategy]:
        """Return strategies in a family."""
        return [strategy for strategy in self.strategies if strategy.family == family]

    def get_by_problem(self, problem: str) -> list[PostProcessStrategy]:
        """Return strategies targeting a problem tag."""
        normalized = _normalize(problem)
        return [
            strategy
            for strategy in self.strategies
            if normalized in {_normalize(item) for item in strategy.target_problems}
        ]

    def recommend(self, task_or_scenario: TaskSpec | str) -> PostProcessRecommendation:
        """Recommend strategies for a TaskSpec or scenario string."""
        scenario = task_or_scenario.scene if isinstance(task_or_scenario, TaskSpec) else task_or_scenario
        strategy_ids = self.recommendations.get(scenario, self.recommendations.get("generic", []))
        strategies = [self.get(strategy_id) for strategy_id in strategy_ids]
        rationale = [
            f"{strategy.id} targets {', '.join(strategy.target_problems) or 'default inference'}"
            for strategy in strategies
        ]
        return PostProcessRecommendation(
            scenario=scenario,
            recommended_postprocess=strategies,
            rationale=rationale,
            companion_actions=_companion_actions(strategies),
            warnings=_warnings(strategies),
        )

    def recommend_for_errors(
        self,
        observations: list[DetectionErrorObservation],
        task_or_scenario: TaskSpec | str | None = None,
    ) -> PostProcessRecommendation:
        """Recommend inference policy from detection errors, optionally seeded by scenario."""
        scenario = "generic"
        strategy_ids: list[str] = []
        rationale: list[str] = []
        if task_or_scenario is not None:
            scenario = task_or_scenario.scene if isinstance(task_or_scenario, TaskSpec) else task_or_scenario
            strategy_ids.extend(self.recommendations.get(scenario, []))

        for observation in observations:
            mapped_ids = self.error_recommendations.get(observation.error_type, [])
            strategy_ids.extend(mapped_ids)
            if mapped_ids:
                rationale.append(
                    f"{observation.error_type} observed {observation.count} times; "
                    f"adding {', '.join(mapped_ids)}."
                )

        strategies = [self.get(strategy_id) for strategy_id in dedupe_list(strategy_ids)]
        return PostProcessRecommendation(
            scenario=scenario,
            recommended_postprocess=strategies,
            rationale=rationale or [
                f"{strategy.id} targets {', '.join(strategy.target_problems) or 'default inference'}"
                for strategy in strategies
            ],
            companion_actions=_companion_actions(strategies),
            warnings=_warnings(strategies),
        )


def default_postprocess_registry_path() -> Path:
    """Return bundled post-processing registry config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "postprocess_strategies.yaml"


def _normalize(value: str) -> str:
    return value.lower().replace("-", "_")


def _companion_actions(strategies: list[PostProcessStrategy]) -> list[str]:
    actions: list[str] = []
    for strategy in strategies:
        actions.extend(strategy.companion_actions)
    return dedupe_list(actions)


def _warnings(strategies: list[PostProcessStrategy]) -> list[str]:
    warnings: list[str] = []
    if any(strategy.latency_cost == "high" for strategy in strategies):
        warnings.append("High-latency post-processing requires validation against deployment FPS and latency budgets.")
    if any(strategy.accuracy_risk == "medium" for strategy in strategies):
        warnings.append("Threshold/NMS changes must be calibrated on validation evidence before deployment.")
    return warnings
