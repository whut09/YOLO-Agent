"""Generate candidate experiment configurations from task and component metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker, RiskLevel
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.components.schema import ComponentCard
from yolo_agent.core.task_spec import TaskSpec


class CandidateConfig(BaseModel):
    """A single candidate experiment configuration."""

    candidate_id: str
    base_model: str
    scale: str
    framework: str
    components: list[str] = Field(default_factory=list)
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    expected_effect: list[str] = Field(default_factory=list)
    risk: RiskLevel = "low"


class CandidatePlan(BaseModel):
    """Serializable candidate plan written to runs/plan.yaml."""

    task_scene: str
    candidates: list[CandidateConfig]
    skipped: list[dict[str, Any]] = Field(default_factory=list)

    def to_yaml(self, path: Path | str) -> None:
        """Write the plan to YAML."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CandidatePlan":
        """Load a generated candidate plan."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Candidate plan YAML must contain a mapping: {input_path}")
        return cls.model_validate(data)


class CandidateGenerator:
    """Generate compatible candidate model/component combinations."""

    def __init__(
        self,
        registry: ComponentRegistry,
        search_space: dict[str, Any],
        checker: CompatibilityChecker | None = None,
    ) -> None:
        self.registry = registry
        self.search_space = search_space
        self.checker = checker or CompatibilityChecker()

    @classmethod
    def from_yaml(cls, registry: ComponentRegistry, search_space_path: Path | str) -> "CandidateGenerator":
        """Create a generator from a search-space YAML file."""
        return cls(registry=registry, search_space=load_search_space(search_space_path))

    def generate(self, task_spec: TaskSpec) -> CandidatePlan:
        """Generate a plan of compatible candidates for a task."""
        defaults = _mapping(self.search_space.get("defaults"))
        candidates: list[CandidateConfig] = []
        skipped: list[dict[str, Any]] = []

        for baseline in _baseline_models(defaults):
            candidate = self._build_candidate(
                task_spec=task_spec,
                raw_candidate={
                    "suffix": f"baseline_{baseline['scale']}",
                    "base_model": baseline["name"],
                    "scale": baseline["scale"],
                    "expected_effect": ["Baseline reference experiment."],
                    "estimated_latency_ms": baseline.get("estimated_latency_ms"),
                    "estimated_model_size_mb": baseline.get("estimated_model_size_mb"),
                },
                defaults=defaults,
            )
            if candidate is not None:
                candidates.append(candidate)

        strategy_names = _strategy_names(task_spec)
        for strategy_name in strategy_names:
            strategy = _mapping(_mapping(self.search_space.get("scenario_strategies")).get(strategy_name))
            for raw_candidate in _list_of_mappings(strategy.get("candidates")):
                candidate = self._build_candidate(task_spec, raw_candidate, defaults)
                if candidate is None:
                    skipped.append(
                        {
                            "candidate_id": _candidate_id(raw_candidate, defaults),
                            "reason": "missing component card",
                        }
                    )
                    continue

                components = self._resolve_components(candidate.components)
                base_model = self._base_model_spec(raw_candidate, defaults)
                result = self.checker.check(task_spec, base_model, components)
                if result.ok:
                    candidate.risk = result.estimated_risk
                    candidates.append(candidate)
                else:
                    skipped.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "reason": "compatibility_failed",
                            "errors": result.errors,
                            "warnings": result.warnings,
                        }
                    )

        return CandidatePlan(task_scene=task_spec.scene, candidates=_dedupe(candidates), skipped=skipped)

    def _build_candidate(
        self,
        task_spec: TaskSpec,
        raw_candidate: dict[str, Any],
        defaults: dict[str, Any],
    ) -> CandidateConfig | None:
        component_ids = [str(component_id) for component_id in raw_candidate.get("components", [])]
        if any(self._find_component(component_id) is None for component_id in component_ids):
            return None

        base_model = self._base_model_spec(raw_candidate, defaults)
        components = self._resolve_components(component_ids)
        result = self.checker.check(task_spec, base_model, components)
        if not result.ok:
            return CandidateConfig(
                candidate_id=_candidate_id(raw_candidate, defaults),
                base_model=base_model.name,
                scale=str(raw_candidate.get("scale", _scale_from_name(base_model.name))),
                framework=str(base_model.framework),
                components=component_ids,
                train_overrides=_mapping(raw_candidate.get("train_overrides")),
                expected_effect=[str(effect) for effect in raw_candidate.get("expected_effect", [])],
                risk=result.estimated_risk,
            )

        return CandidateConfig(
            candidate_id=_candidate_id(raw_candidate, defaults),
            base_model=base_model.name,
            scale=str(raw_candidate.get("scale", _scale_from_name(base_model.name))),
            framework=str(base_model.framework),
            components=component_ids,
            train_overrides=_mapping(raw_candidate.get("train_overrides")),
            expected_effect=[str(effect) for effect in raw_candidate.get("expected_effect", [])],
            risk=result.estimated_risk,
        )

    def _base_model_spec(self, raw_candidate: dict[str, Any], defaults: dict[str, Any]) -> BaseModelSpec:
        return BaseModelSpec(
            name=str(raw_candidate.get("base_model", defaults.get("base_model", "yolo11n"))),
            framework=str(raw_candidate.get("framework", defaults.get("framework", "ultralytics"))),
            model_family=str(raw_candidate.get("model_family", defaults.get("model_family", "yolov11"))),
            export_format=str(raw_candidate.get("export_format", defaults.get("export_format", "none"))),
            estimated_latency_ms=raw_candidate.get("estimated_latency_ms"),
            estimated_model_size_mb=raw_candidate.get("estimated_model_size_mb"),
        )

    def _resolve_components(self, component_ids: list[str]) -> list[ComponentCard]:
        return [
            component
            for component_id in component_ids
            if (component := self._find_component(component_id)) is not None
        ]

    def _find_component(self, component_id: str) -> ComponentCard | None:
        return next((card for card in self.registry.cards if card.id == component_id), None)


def generate_plan(
    task_path: Path | str,
    component_path: Path | str,
    search_space_path: Path | str,
    out_path: Path | str,
) -> CandidatePlan:
    """Generate and write a candidate plan from YAML inputs."""
    task_spec = TaskSpec.from_yaml(task_path)
    registry = ComponentRegistry.from_path(component_path)
    generator = CandidateGenerator.from_yaml(registry, search_space_path)
    plan = generator.generate(task_spec)
    plan.to_yaml(out_path)
    return plan


def default_search_space_path() -> Path:
    """Return the bundled search-space file."""
    return Path(__file__).resolve().parents[2] / "configs" / "search_space.yaml"


def load_search_space(path: Path | str) -> dict[str, Any]:
    """Load search-space YAML."""
    search_path = Path(path)
    with search_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Search-space YAML must contain a mapping: {search_path}")
    return data


def _baseline_models(defaults: dict[str, Any]) -> list[dict[str, Any]]:
    raw_baselines = defaults.get("baseline_models")
    baselines = _list_of_mappings(raw_baselines)
    if baselines:
        return baselines
    return [{"name": "yolo11n", "scale": "n"}, {"name": "yolo11s", "scale": "s"}]


def _strategy_names(task_spec: TaskSpec) -> list[str]:
    names = [task_spec.scene]
    problem_metrics = {task_spec.primary_metric.name}
    problem_metrics.update(metric.name for metric in task_spec.secondary_metrics)
    if "map50_95" in problem_metrics or "map50" in problem_metrics:
        names.append("localization_error")
    return list(dict.fromkeys(names))


def _candidate_id(raw_candidate: dict[str, Any], defaults: dict[str, Any]) -> str:
    base_model = str(raw_candidate.get("base_model", defaults.get("base_model", "yolo11n")))
    suffix = str(raw_candidate.get("suffix", "candidate"))
    return f"{base_model}_{suffix}".replace(".", "_").replace("-", "_")


def _scale_from_name(model_name: str) -> str:
    return model_name[-1] if model_name and model_name[-1] in {"n", "s", "m", "l", "x"} else "custom"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe(candidates: list[CandidateConfig]) -> list[CandidateConfig]:
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped

