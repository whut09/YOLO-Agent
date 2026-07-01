"""YAML-driven compatibility checks for component combinations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.components.schema import ComponentCard, FrameworkName, ModelFamily
from yolo_agent.core.task_spec import TaskSpec


RiskLevel = Literal["low", "medium", "high"]
ExportFormat = Literal["none", "onnx", "tensorrt", "openvino", "torchscript"]


class BaseModelSpec(BaseModel):
    """Minimal model identity needed before assembling candidate components."""

    name: str = "unknown"
    framework: FrameworkName | str = "generic"
    model_family: ModelFamily | str = "generic"
    export_format: ExportFormat | str = "none"
    estimated_latency_ms: float | None = Field(default=None, gt=0.0)
    estimated_model_size_mb: float | None = Field(default=None, gt=0.0)


class CompatibilityResult(BaseModel):
    """Compatibility checker output."""

    ok: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    estimated_risk: RiskLevel = "low"


class CompatibilityChecker:
    """Check task, model, framework, deployment, and component compatibility."""

    def __init__(self, rules_path: Path | str | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path is not None else default_rules_path()
        self.rules = _load_rules(self.rules_path)

    def check(
        self,
        task_spec: TaskSpec,
        base_model: BaseModelSpec | dict[str, Any],
        components: list[ComponentCard],
    ) -> CompatibilityResult:
        """Validate a candidate component set before model generation."""
        model_spec = _normalize_base_model(base_model)
        warnings: list[str] = []
        errors: list[str] = []
        risk: RiskLevel = "low"

        for component in components:
            if task_spec.task_type not in component.compatible_tasks:
                errors.append(
                    f"{component.id} is not compatible with task_type={task_spec.task_type}."
                )
                risk = _max_risk(risk, "high", self.rules)

            if not _matches_value(model_spec.framework, component.compatible_frameworks):
                errors.append(
                    f"{component.id} is not compatible with framework={model_spec.framework}."
                )
                risk = _max_risk(risk, "high", self.rules)

            if not _matches_value(model_spec.model_family, component.compatible_model_families):
                errors.append(
                    f"{component.id} is not compatible with model_family={model_spec.model_family}."
                )
                risk = _max_risk(risk, "high", self.rules)

        for message, severity, rule_risk in _evaluate_combination_rules(components, self.rules):
            if severity == "error":
                errors.append(message)
            else:
                warnings.append(message)
            risk = _max_risk(risk, rule_risk, self.rules)

        for message, rule_risk in _evaluate_budget_rules(task_spec, model_spec, components, self.rules):
            warnings.append(message)
            risk = _max_risk(risk, rule_risk, self.rules)

        for message, severity, rule_risk in _evaluate_export_rules(model_spec, components, self.rules):
            if severity == "error":
                errors.append(message)
            else:
                warnings.append(message)
            risk = _max_risk(risk, rule_risk, self.rules)

        return CompatibilityResult(
            ok=not errors,
            warnings=warnings,
            errors=errors,
            estimated_risk=risk,
        )


def default_rules_path() -> Path:
    """Return the bundled compatibility rule file."""
    return Path(__file__).resolve().parents[2] / "configs" / "compatibility_rules.yaml"


def _load_rules(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Compatibility rules YAML must contain a mapping: {path}")
    return data


def _normalize_base_model(base_model: BaseModelSpec | dict[str, Any]) -> BaseModelSpec:
    if isinstance(base_model, BaseModelSpec):
        return base_model
    return BaseModelSpec.model_validate(base_model)


def _matches_value(value: str, allowed: list[str]) -> bool:
    return value in allowed or "generic" in allowed or value == "generic"


def _evaluate_combination_rules(
    components: list[ComponentCard],
    rules: dict[str, Any],
) -> list[tuple[str, str, RiskLevel]]:
    component_ids = {component.id for component in components}
    findings: list[tuple[str, str, RiskLevel]] = []

    for rule in rules.get("combination_rules", []):
        if not isinstance(rule, dict):
            continue
        exact_ids = set(rule.get("component_ids", []))
        prefixes = tuple(rule.get("component_id_prefixes", []))
        exact_match = exact_ids and exact_ids.issubset(component_ids)
        prefix_match = bool(prefixes) and any(
            component.id.startswith(prefixes) for component in components
        )
        if not exact_match and not prefix_match:
            continue

        required = rule.get("requires_constraint")
        if isinstance(required, dict) and _required_constraint_satisfied(components, required):
            continue

        findings.append(
            (
                str(rule.get("message", "Component combination requires review.")),
                str(rule.get("severity", "warning")),
                _coerce_risk(rule.get("risk", "medium")),
            )
        )

    return findings


def _required_constraint_satisfied(components: list[ComponentCard], required: dict[str, Any]) -> bool:
    key = required.get("key")
    expected = required.get("value")
    return any(component.constraints.get(key) == expected for component in components)


def _evaluate_budget_rules(
    task_spec: TaskSpec,
    model_spec: BaseModelSpec,
    components: list[ComponentCard],
    rules: dict[str, Any],
) -> list[tuple[str, RiskLevel]]:
    budget_rules = rules.get("budget_rules", {})
    if not isinstance(budget_rules, dict):
        return []

    findings: list[tuple[str, RiskLevel]] = []
    if task_spec.max_latency_ms is not None:
        if model_spec.estimated_latency_ms is not None and model_spec.estimated_latency_ms > task_spec.max_latency_ms:
            findings.append(
                (
                    f"Base model latency {model_spec.estimated_latency_ms} ms exceeds budget {task_spec.max_latency_ms} ms.",
                    "high",
                )
            )
        for component in _components_with_constraints(
            components,
            budget_rules.get("latency_warning_constraints", []),
        ):
            findings.append(
                (
                    f"{component.id} may affect latency; verify against max_latency_ms={task_spec.max_latency_ms}.",
                    "medium",
                )
            )

    if task_spec.max_model_size_mb is not None:
        if (
            model_spec.estimated_model_size_mb is not None
            and model_spec.estimated_model_size_mb > task_spec.max_model_size_mb
        ):
            findings.append(
                (
                    f"Base model size {model_spec.estimated_model_size_mb} MB exceeds budget {task_spec.max_model_size_mb} MB.",
                    "high",
                )
            )
        for component in _components_with_constraints(
            components,
            budget_rules.get("model_size_warning_constraints", []),
        ):
            findings.append(
                (
                    f"{component.id} may affect model size; verify against max_model_size_mb={task_spec.max_model_size_mb}.",
                    "medium",
                )
            )

    return findings


def _evaluate_export_rules(
    model_spec: BaseModelSpec,
    components: list[ComponentCard],
    rules: dict[str, Any],
) -> list[tuple[str, str, RiskLevel]]:
    export_format = str(model_spec.export_format).lower()
    if export_format in {"none", ""}:
        return []

    export_rules = rules.get("export_rules", {})
    if not isinstance(export_rules, dict):
        return []
    selected = export_rules.get(export_format, {})
    if not isinstance(selected, dict):
        return []

    findings: list[tuple[str, str, RiskLevel]] = []
    for component in components:
        export_safe = component.constraints.get("export_safe")
        if export_safe is False:
            findings.append((f"{component.id} is marked export_safe=false for {export_format}.", "error", "high"))
        elif str(export_safe).lower() == "unknown":
            findings.append((f"{component.id} has unknown export safety for {export_format}.", "warning", "medium"))

        for constraint in selected.get("unsafe_constraints", []):
            if component.constraints.get(constraint) is True:
                message = str(selected.get("unsafe_message", "{component_id} violates {constraint}.")).format(
                    component_id=component.id,
                    constraint=constraint,
                )
                findings.append((message, "error", "high"))

        for constraint in selected.get("warning_constraints", []):
            if component.constraints.get(constraint) is True:
                message = str(selected.get("warning_message", "{component_id} uses {constraint}.")).format(
                    component_id=component.id,
                    constraint=constraint,
                )
                findings.append((message, "warning", "medium"))

    return findings


def _components_with_constraints(
    components: list[ComponentCard],
    constraint_names: list[str],
) -> list[ComponentCard]:
    return [
        component
        for component in components
        if any(component.constraints.get(name) not in {None, False, 0} for name in constraint_names)
    ]


def _max_risk(current: RiskLevel, candidate: RiskLevel, rules: dict[str, Any]) -> RiskLevel:
    order = rules.get("risk_order", ["low", "medium", "high"])
    if not isinstance(order, list):
        order = ["low", "medium", "high"]
    current_index = order.index(current) if current in order else 0
    candidate_index = order.index(candidate) if candidate in order else 0
    return candidate if candidate_index > current_index else current


def _coerce_risk(value: object) -> RiskLevel:
    if value in {"low", "medium", "high"}:
        return value  # type: ignore[return-value]
    return "medium"

