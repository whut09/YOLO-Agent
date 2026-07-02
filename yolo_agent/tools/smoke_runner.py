"""Smoke-test guard for generated candidate plans."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.yaml_generator import UltralyticsYamlGenerator
from yolo_agent.agents.candidate_generator import CandidatePlan
from yolo_agent.core.evidence_store import EvidenceStore


SmokeStatus = Literal["passed", "failed", "skipped"]


class SmokeCandidateResult(BaseModel):
    """Smoke result for a single candidate."""

    candidate_id: str
    status: SmokeStatus
    generated_yaml: Path | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SmokeRunResult(BaseModel):
    """Aggregate smoke-run result."""

    run_id: str
    status: SmokeStatus
    plan_path: Path
    data_path: Path
    ultralytics_available: bool
    try_forward: bool = False
    candidates: list[SmokeCandidateResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SmokeRunner:
    """Guard that validates candidate plans before any real training."""

    def __init__(
        self,
        evidence_store: EvidenceStore | None = None,
        yaml_generator: UltralyticsYamlGenerator | None = None,
    ) -> None:
        self.evidence_store = evidence_store or EvidenceStore()
        self.yaml_generator = yaml_generator or UltralyticsYamlGenerator()

    def run(
        self,
        plan_path: Path | str,
        data_path: Path | str,
        run_id: str = "smoke",
        base_template: Path | str | None = None,
        generated_dir: Path | str | None = None,
        try_forward: bool = False,
    ) -> SmokeRunResult:
        """Run smoke checks and record evidence."""
        plan_file = Path(plan_path)
        data_file = Path(data_path)
        run_dir = self.evidence_store.create_run(run_id)
        template_path = Path(base_template) if base_template is not None else default_ultralytics_template_path()
        output_dir = Path(generated_dir) if generated_dir is not None else run_dir / "generated_models"

        errors: list[str] = []
        warnings: list[str] = []
        if not plan_file.is_file():
            errors.append(f"Plan file does not exist: {plan_file}")
        if not data_file.is_file():
            errors.append(f"Dataset YAML does not exist: {data_file}")
        if not template_path.is_file():
            errors.append(f"Base model YAML template does not exist: {template_path}")

        if errors:
            result = SmokeRunResult(
                run_id=run_id,
                status="failed",
                plan_path=plan_file,
                data_path=data_file,
                ultralytics_available=False,
                try_forward=try_forward,
                errors=errors,
            )
            self._write_evidence(result, template_path)
            return result

        plan = CandidatePlan.from_yaml(plan_file)
        nc = _class_count_from_data_yaml(data_file)
        ultralytics_module = _import_ultralytics()
        ultralytics_available = ultralytics_module is not None
        if not ultralytics_available:
            warnings.append("ultralytics is not installed; import and forward checks skipped.")

        candidate_results: list[SmokeCandidateResult] = []
        for candidate in plan.candidates:
            candidate_result = self._check_candidate(
                candidate_id=candidate.candidate_id,
                plan=plan,
                template_path=template_path,
                output_dir=output_dir,
                nc=nc,
                ultralytics_module=ultralytics_module,
                try_forward=try_forward,
            )
            candidate_results.append(candidate_result)
            if candidate_result.generated_yaml is not None and candidate_result.generated_yaml.exists():
                self.evidence_store.log_artifact(run_id, candidate_result.generated_yaml)

        status = _aggregate_status(candidate_results)
        result = SmokeRunResult(
            run_id=run_id,
            status=status,
            plan_path=plan_file,
            data_path=data_file,
            ultralytics_available=ultralytics_available,
            try_forward=try_forward,
            candidates=candidate_results,
            warnings=warnings,
        )
        self._write_evidence(result, template_path)
        return result

    def _check_candidate(
        self,
        candidate_id: str,
        plan: CandidatePlan,
        template_path: Path,
        output_dir: Path,
        nc: int | None,
        ultralytics_module: object | None,
        try_forward: bool,
    ) -> SmokeCandidateResult:
        candidate = next(candidate for candidate in plan.candidates if candidate.candidate_id == candidate_id)
        warnings: list[str] = []
        errors: list[str] = []
        try:
            generation = self.yaml_generator.generate(
                candidate=candidate,
                base_template=template_path,
                output_dir=output_dir,
                nc=nc,
            )
            warnings.extend(generation.warnings)
        except Exception as exc:  # pragma: no cover - defensive guard path
            return SmokeCandidateResult(
                candidate_id=candidate.candidate_id,
                status="failed",
                warnings=warnings,
                errors=[f"Candidate YAML generation failed: {exc}"],
            )

        if ultralytics_module is None:
            return SmokeCandidateResult(
                candidate_id=candidate.candidate_id,
                status="skipped",
                generated_yaml=generation.output_path,
                warnings=warnings + ["ultralytics is not installed; YOLO import skipped."],
            )

        if try_forward:
            try:
                yolo_cls = getattr(ultralytics_module, "YOLO")
                model = yolo_cls(str(generation.output_path))
                if hasattr(model, "info"):
                    model.info()
            except Exception as exc:  # pragma: no cover - exercised with mocks in tests
                errors.append(f"Ultralytics forward/info smoke failed: {exc}")

        return SmokeCandidateResult(
            candidate_id=candidate.candidate_id,
            status="failed" if errors else "passed",
            generated_yaml=generation.output_path,
            warnings=warnings,
            errors=errors,
        )

    def _write_evidence(self, result: SmokeRunResult, base_template: Path) -> None:
        config = {
            "run_id": result.run_id,
            "plan_path": str(result.plan_path),
            "data_path": str(result.data_path),
            "base_template": str(base_template),
            "try_forward": result.try_forward,
            "ultralytics_available": result.ultralytics_available,
            "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
            "warnings": result.warnings,
            "errors": result.errors,
        }
        metrics = {
            "candidate_count": len(result.candidates),
            "passed": sum(candidate.status == "passed" for candidate in result.candidates),
            "failed": sum(candidate.status == "failed" for candidate in result.candidates),
            "skipped": sum(candidate.status == "skipped" for candidate in result.candidates),
        }
        self.evidence_store.log_config(result.run_id, config)
        self.evidence_store.log_metrics(result.run_id, metrics)


def default_ultralytics_template_path() -> Path:
    """Return the bundled minimal Ultralytics YAML template."""
    return Path(__file__).resolve().parents[2] / "configs" / "templates" / "ultralytics_base.yaml"


def _class_count_from_data_yaml(data_path: Path) -> int | None:
    with data_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    names = data.get("names")
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    nc = data.get("nc")
    return int(nc) if isinstance(nc, int) else None


def _import_ultralytics() -> object | None:
    try:
        return importlib.import_module("ultralytics")
    except ImportError:
        return None


def _aggregate_status(candidate_results: list[SmokeCandidateResult]) -> SmokeStatus:
    if any(candidate.status == "failed" for candidate in candidate_results):
        return "failed"
    if candidate_results and all(candidate.status == "skipped" for candidate in candidate_results):
        return "skipped"
    return "passed"

