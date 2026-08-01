"""Opt-in real GPU acceptance driver for the training evidence loop."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.coco_post_eval import write_coco_eval_report
from yolo_agent.adapters.ultralytics.plugin_context import PluginRuntimeEvidence
from yolo_agent.adapters.ultralytics.training import discover_coco_predictions_artifact
from yolo_agent.agents.asha_scheduler import (
    ASHAObservation,
    ASHAScheduler,
    ASHAStudy,
    default_asha_rungs,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.certification.fixture import create_mini_coco_fixture
from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.component_runner import ComponentCertificationRunner
from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationObjectiveResult,
    CertificationPromotionResult,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.certification.paper_auto_optimization_schemas import PaperProtocolIdentity
from yolo_agent.certification.paper_auto_optimization_protocol import (
    build_paper_protocol_identity,
)
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingAdapter,
    SmallObjectSamplingManifest,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import PairedExperimentResult, build_paired_experiment_result
from yolo_agent.core.paired_bootstrap import (
    PairedBootstrapConfig,
    PairedBootstrapReport,
    paired_bootstrap_coco_predictions,
)
from yolo_agent.core.pilot_evidence import validate_coco_evidence_artifacts
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics
from yolo_agent.tools.coco_error_mining import mine_coco_errors


class BackendRun(BaseModel):
    candidate_id: str
    node_id: str
    run_dir: Path
    checkpoint: Path
    command: list[str] = Field(default_factory=list)
    runtime_artifacts: dict[str, Path] = Field(default_factory=dict)
    protocol_identity: PaperProtocolIdentity | None = None


class BackendEvaluation(BaseModel):
    eval_path: Path
    predictions_path: Path
    error_report_path: Path
    latency_ms: float = Field(gt=0)
    model_size_mb: float = Field(gt=0)
    command: list[str] = Field(default_factory=list)


class CertificationRecipe(BaseModel):
    """Small, deterministic recipe cohort used by the GPU acceptance suite."""

    recipe_id: str
    changed_variable: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    execution_class: str = "native_atomic"
    target_class_names: list[str] = Field(default_factory=list)
    primary_metric: str = "map50_95"
    max_map_regression: float = Field(default=0.01, ge=0)
    max_latency_regression: float = Field(default=0.05, ge=0)
    max_model_size_regression: float = Field(default=0.05, ge=0)


CERTIFICATION_RECIPES = (
    CertificationRecipe(recipe_id="reduce_mosaic", changed_variable="mosaic", overrides={"mosaic": 0.0}),
    CertificationRecipe(recipe_id="close_mosaic_early", changed_variable="close_mosaic", overrides={"close_mosaic": 5}),
    CertificationRecipe(recipe_id="light_mixup", changed_variable="mixup", overrides={"mixup": 0.05}),
    CertificationRecipe(
        recipe_id="small_object_sampling",
        changed_variable="data.sampling_policy",
        overrides={
            "data.sampling_policy": {
                "small_object_boost": 2.0,
                "class_balance": True,
                "rare_class_boost": 1.5,
                "fn_heavy_class_ids": [0],
                "target_class_ids": [0],
                "max_oversampling_ratio": 3.0,
            }
        },
        execution_class="component_adapter",
        target_class_names=["object"],
        primary_metric="ap_small",
    ),
)

NATIVE_CERTIFICATION_RECIPE_IDS = {
    "reduce_mosaic",
    "close_mosaic_early",
    "light_mixup",
}

CERTIFICATION_DEPENDENCIES = ("torch", "ultralytics", "pycocotools")
CERTIFICATION_INSTALL_COMMAND = 'python -m pip install -e ".[certification]"'


class GpuAcceptanceBackend(Protocol):
    def environment(self) -> dict[str, Any]: ...
    def certify_component(self, *, component_id: str, workdir: Path, device: str) -> CertificationStage: ...
    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]: ...
    def train(self, *, candidate_id: str, node_id: str, data_yaml: Path, model: str, workdir: Path, device: str, epochs: int, seed: int, protocol_hash: str, overrides: dict[str, Any]) -> BackendRun: ...
    def evaluate(self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str) -> BackendEvaluation: ...


class PaperRecipeCertificationBackend(Protocol):
    def prepare(self, root: Path) -> tuple[list[CertificationStage], dict[str, str]]: ...
    def finalize(self, root: Path, *, recipe_id: str, paired_result: PairedExperimentResult) -> CertificationStage: ...


class OfflinePaperRecipeCertificationBackend:
    """Deterministic offline paper path used by tests and real-GPU acceptance."""

    def prepare(self, root: Path) -> tuple[list[CertificationStage], dict[str, str]]:
        catalog = root / "paper_certification" / "catalog.json"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(json.dumps({"papers": [{"paper_id": "mock-paper", "component_ids": ["mock_adapter"]}]}), encoding="utf-8")
        snapshot_hash = _hash_files(catalog.parent)
        snapshot = catalog.parent / "snapshot.yaml"
        snapshot.write_text(f"snapshot_hash: {snapshot_hash}\n", encoding="utf-8")
        stages = [
            _passed_stage("catalog_import", artifacts={"catalog": catalog.as_posix()}),
            _passed_stage("snapshot_creation", artifacts={"snapshot": snapshot.as_posix()}, metrics={"snapshot_hash": snapshot_hash}),
            _passed_stage("diagnosis_linked_paper_prior", metrics={"paper_id": "mock-paper", "diagnosis": "small_object_false_negative"}),
            _passed_stage("eligibility_gate", metrics={"eligible": True, "imgsz": 640}),
            _passed_stage("executable_recipe", metrics={"recipe_id": "mock-paper-recipe", "adapter": "mock_adapter", "maturity": "smoke_passed"}),
        ]
        return stages, {"recipe_id": "mock-paper-recipe", "snapshot_hash": snapshot_hash}

    def finalize(self, root: Path, *, recipe_id: str, paired_result: PairedExperimentResult) -> CertificationStage:
        memory = root / "paper_certification" / "policy_memory.jsonl"
        memory.write_text(json.dumps({"recipe_id": recipe_id, "paired_result_hash": paired_result.result_hash}) + "\n", encoding="utf-8")
        return _passed_stage("policy_memory_update", artifacts={"policy_memory": memory.as_posix()}, metrics={"recipe_id": recipe_id})


class RealGpuAcceptanceSuite:
    """Run the mini COCO certification only when explicitly requested."""

    def __init__(self, backend: GpuAcceptanceBackend | None = None, paper_backend: PaperRecipeCertificationBackend | None = None) -> None:
        self.backend = backend or UltralyticsGpuBackend()
        self.paper_backend = paper_backend or OfflinePaperRecipeCertificationBackend()

    def run(
        self,
        *,
        workdir: Path | str,
        model: str = "yolo26n.pt",
        device: str = "0",
        execute_real_gpu: bool = False,
        recipe_id: str = "reduce_mosaic",
    ) -> CertificationReport:
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        data_yaml = create_mini_coco_fixture(root / "mini_coco")
        code_hash = certification_code_hash()
        protocol_hash = _hash_payload(
            {"suite": "mini_gpu_pilot.v1", "model": model, "imgsz": 640, "code_hash": code_hash}
        )
        stages: list[CertificationStage] = []
        failures: list[str] = []
        promotion_results: list[CertificationPromotionResult] = []
        recipe = next((item for item in CERTIFICATION_RECIPES if item.recipe_id == recipe_id), None)
        if recipe is None:
            raise ValueError(
                f"unknown certification recipe {recipe_id!r}; choose from "
                + ", ".join(item.recipe_id for item in CERTIFICATION_RECIPES)
            )
        if not execute_real_gpu:
            report = CertificationReport(
                certification_id=root.name or "mini-gpu-certification",
                level="mini_gpu_pilot",
                status="skipped",
                model=model,
                data_yaml=data_yaml.as_posix(),
                device=device,
                protocol_hash=protocol_hash,
                certified_code_hash=code_hash,
                executed_recipe_id=recipe.recipe_id,
                executed_changed_variable=recipe.changed_variable,
                stages=[CertificationStage(stage_id="environment", status="skipped", message="Pass --execute-real-gpu to opt in.")],
                failures=["real_gpu_execution_not_confirmed"],
            )
            report.to_yaml(root / "certification_report.yaml", exclude_none=True, sort_keys=False)
            return report
        try:
            environment = self.backend.environment()
            stages.append(_passed_stage("environment", metrics=environment))
            if recipe.execution_class == "component_adapter":
                certify_component = getattr(self.backend, "certify_component", None)
                if not callable(certify_component):
                    raise RuntimeError(
                        "GPU backend does not implement component runtime certification"
                    )
                stages.append(
                    certify_component(
                        component_id="sampling.small_object",
                        workdir=root,
                        device=device,
                    )
                )
            paper_stages, paper_identity = self.paper_backend.prepare(root)
            stages.extend(paper_stages)
            entry_command = self.backend.train_entrypoint(data_yaml=data_yaml, model=model, workdir=root, device=device)
            stages.append(_passed_stage("train_entrypoint", command=entry_command))
            stages.append(
                _passed_stage(
                    "recipe_execution_contract",
                    metrics={
                        "recipe_id": recipe.recipe_id,
                        "changed_variable": recipe.changed_variable,
                        "fixed_imgsz": 640,
                        "execution_class": recipe.execution_class,
                    },
                )
            )

            debug = self.backend.train(
                candidate_id="debug",
                node_id="debug",
                data_yaml=data_yaml,
                model=model,
                workdir=root,
                device=device,
                epochs=1,
                seed=1,
                protocol_hash=_fidelity_hash(protocol_hash, "debug"),
                overrides={},
            )
            stages.append(_passed_stage("debug", command=debug.command, artifacts={"checkpoint": debug.checkpoint.as_posix()}))

            pilot_3_hash = _fidelity_hash(protocol_hash, "pilot_3")
            control_3 = self.backend.train(
                candidate_id="baseline_pilot_3",
                node_id="baseline_pilot_3",
                data_yaml=data_yaml,
                model=model,
                workdir=root,
                device=device,
                epochs=3,
                seed=1,
                protocol_hash=pilot_3_hash,
                overrides={},
            )
            stages.append(_passed_stage("pilot_3_control", command=control_3.command, artifacts={"checkpoint": control_3.checkpoint.as_posix()}))
            cohort = (
                [recipe]
                if recipe.execution_class == "component_adapter"
                else [
                    item
                    for item in CERTIFICATION_RECIPES
                    if item.recipe_id in NATIVE_CERTIFICATION_RECIPE_IDS
                ]
            )
            candidates = [(item.recipe_id, dict(item.overrides)) for item in cohort]
            candidate_runs = [
                self.backend.train(
                    candidate_id=candidate_id,
                    node_id=f"{candidate_id}_pilot_3",
                    data_yaml=data_yaml,
                    model=model,
                    workdir=root,
                    device=device,
                    epochs=3,
                    seed=1,
                    protocol_hash=pilot_3_hash,
                    overrides=overrides,
                )
                for candidate_id, overrides in candidates
            ]
            stages.append(_passed_stage("pilot_3_candidates", metrics={"candidate_count": len(candidate_runs)}))
            if recipe.execution_class == "component_adapter":
                runtime_artifacts, runtime_checks = _validate_runtime_artifacts(
                    candidate_runs[0],
                    protocol_hash=pilot_3_hash,
                )
                stages.append(
                    _passed_stage(
                        "runtime_adapter",
                        artifacts={key: value.as_posix() for key, value in runtime_artifacts.items()},
                        metrics={
                            "recipe_id": recipe.recipe_id,
                            "changed_variable": recipe.changed_variable,
                            "runtime_execution_ready": True,
                            **runtime_checks,
                        },
                    )
                )

            store = EvidenceStore(root / "evidence")
            error_store = ErrorFactStore(root / "evidence")
            run_id = "mini_gpu_certification"
            eval_control_3 = self.backend.evaluate(run=control_3, data_yaml=data_yaml, workdir=root, device=device)
            identity_3 = _matched_identity(data_yaml, environment, protocol_hash=pilot_3_hash, epochs=3, fidelity="pilot_3", seed=1)
            _import_observation(store, run_id, control_3, eval_control_3, identity_3, "baseline_reference")
            evaluations: dict[str, BackendEvaluation] = {}
            paired_results: dict[str, PairedExperimentResult] = {}
            bootstrap_reports: dict[str, PairedBootstrapReport] = {}
            for candidate_run in candidate_runs:
                evaluation = self.backend.evaluate(run=candidate_run, data_yaml=data_yaml, workdir=root, device=device)
                evaluations[candidate_run.candidate_id] = evaluation
                _import_observation(store, run_id, candidate_run, evaluation, identity_3, "current_observation")
                bootstrap = _run_paired_bootstrap(
                    data_yaml=data_yaml,
                    control=eval_control_3,
                    candidate=evaluation,
                    output=root / "paired_bootstrap" / f"{candidate_run.node_id}.json",
                    seed=1,
                )
                bootstrap_reports[candidate_run.candidate_id] = bootstrap
                _import_bootstrap_metrics(
                    store,
                    run_id=run_id,
                    run=candidate_run,
                    identity=identity_3,
                    report=bootstrap,
                )
                paired_results[candidate_run.candidate_id] = build_paired_experiment_result(
                    run_id=run_id,
                    candidate_id=candidate_run.candidate_id,
                    candidate_node_id=candidate_run.node_id,
                    metric_records=store.load_run(run_id).metric_records,
                    error_facts=error_store.read(run_id),
                    primary_metric=recipe.primary_metric,
                    target_error_facts=_target_error_facts(recipe),
                    additional_metrics=_additional_metrics(recipe),
                )
            stages.append(_passed_stage("post_eval", metrics={"evaluated_nodes": 1 + len(evaluations)}))
            fact_count = len(error_store.read(run_id))
            if fact_count == 0:
                raise RuntimeError("COCO post-eval produced no error facts")
            stages.append(_passed_stage("error_facts", metrics={"error_fact_count": fact_count}))
            if not all(result.verified for result in paired_results.values()):
                raise RuntimeError("one or more pilot_3 paired results are not verified")
            if not all(item.status == "completed" for item in bootstrap_reports.values()):
                raise RuntimeError("one or more pilot_3 paired bootstrap reports are incomplete")
            stages.append(
                _passed_stage(
                    "paired_bootstrap",
                    artifacts={
                        key: (root / "paired_bootstrap" / f"{key}_pilot_3.json").as_posix()
                        for key in bootstrap_reports
                    },
                    metrics={
                        key: item.overall.observed_delta if item.overall is not None else None
                        for key, item in bootstrap_reports.items()
                    },
                )
            )
            stages.append(
                _passed_stage(
                    "paired_delta",
                    metrics={
                        key: value.metric_deltas[recipe.primary_metric].paired_delta
                        for key, value in paired_results.items()
                    },
                )
            )

            scheduler = _certification_scheduler(
                run_id,
                cohort_size=len(candidates),
                target_error_required=recipe.execution_class == "component_adapter",
            )
            baseline_node = _node(control_3.candidate_id, control_3.node_id, {})
            for candidate_id, overrides in candidates:
                candidate_run = next(item for item in candidate_runs if item.candidate_id == candidate_id)
                scheduler.register_trial(
                    trial_id=candidate_id,
                    candidate_id=candidate_id,
                    source_run_id=run_id,
                    source_node=_node(candidate_id, candidate_run.node_id, overrides),
                    baseline_control_node=baseline_node,
                    target_error_facts=_target_error_facts(recipe),
                )
            for candidate_id, _ in candidates:
                paired = paired_results[candidate_id]
                promotion = _promotion_result(
                    "pilot_3",
                    recipe,
                    paired,
                    eval_control_3,
                    evaluations[candidate_id],
                    bootstrap_reports[candidate_id],
                )
                if recipe.execution_class == "component_adapter":
                    promotion_results.append(promotion)
                    stages.append(_promotion_stage(promotion))
                scheduler.report(
                    candidate_id,
                    _asha_observation(
                        "pilot_3",
                        paired,
                        seed=1,
                        primary_metric=recipe.primary_metric,
                        promotion=promotion,
                    ),
                )
            assignment = scheduler.next_assignment()
            if assignment is None or assignment.stage_id != "pilot_10":
                reasons = [
                    reason
                    for item in promotion_results
                    for reason in item.rejection_reasons
                ]
                raise RuntimeError(
                    "ASHA did not produce a pilot_10 survivor"
                    + (f": {';'.join(reasons)}" if reasons else "")
                )
            survivor = assignment.candidate_id
            stages.append(_passed_stage("asha_decision", metrics={"survivor": survivor, "assignment_id": assignment.assignment_id}))

            winner_overrides = dict(next(overrides for candidate_id, overrides in candidates if candidate_id == survivor))
            pilot_10_hash = _fidelity_hash(protocol_hash, "pilot_10")
            control_10 = self.backend.train(candidate_id="baseline_pilot_10", node_id="baseline_pilot_10", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=10, seed=1, protocol_hash=pilot_10_hash, overrides={})
            winner_10 = self.backend.train(candidate_id=survivor, node_id=f"{survivor}_pilot_10", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=10, seed=1, protocol_hash=pilot_10_hash, overrides=winner_overrides)
            if recipe.execution_class == "component_adapter":
                _validate_runtime_artifacts(
                    winner_10,
                    protocol_hash=pilot_10_hash,
                )
            eval_control_10 = self.backend.evaluate(run=control_10, data_yaml=data_yaml, workdir=root, device=device)
            eval_winner_10 = self.backend.evaluate(run=winner_10, data_yaml=data_yaml, workdir=root, device=device)
            identity_10 = _matched_identity(data_yaml, environment, protocol_hash=pilot_10_hash, epochs=10, fidelity="pilot_10", seed=1)
            _import_observation(store, run_id, control_10, eval_control_10, identity_10, "baseline_reference")
            _import_observation(store, run_id, winner_10, eval_winner_10, identity_10, "current_observation")
            bootstrap_10 = _run_paired_bootstrap(
                data_yaml=data_yaml,
                control=eval_control_10,
                candidate=eval_winner_10,
                output=root / "paired_bootstrap" / f"{winner_10.node_id}.json",
                seed=10,
            )
            _import_bootstrap_metrics(
                store,
                run_id=run_id,
                run=winner_10,
                identity=identity_10,
                report=bootstrap_10,
            )
            paired_10 = build_paired_experiment_result(
                run_id=run_id,
                candidate_id=survivor,
                candidate_node_id=winner_10.node_id,
                metric_records=store.load_run(run_id).metric_records,
                error_facts=error_store.read(run_id),
                primary_metric=recipe.primary_metric,
                target_error_facts=_target_error_facts(recipe),
                additional_metrics=_additional_metrics(recipe),
            )
            if not paired_10.verified:
                raise RuntimeError("pilot_10 paired result is not verified")
            promotion_10 = _promotion_result(
                "pilot_10",
                recipe,
                paired_10,
                eval_control_10,
                eval_winner_10,
                bootstrap_10,
            )
            if recipe.execution_class == "component_adapter":
                promotion_results.append(promotion_10)
                stages.append(_promotion_stage(promotion_10))
            trial = scheduler.report(
                survivor,
                _asha_observation(
                    "pilot_10",
                    paired_10,
                    seed=1,
                    primary_metric=recipe.primary_metric,
                    promotion=promotion_10,
                ),
            )
            if trial.status != "full_pending_confirmation":
                raise RuntimeError(
                    "pilot_10 promotion gate rejected survivor: "
                    + (trial.eliminated_reason or ";".join(promotion_10.rejection_reasons))
                )
            stages.append(_passed_stage("pilot_10", metrics={"candidate": survivor, "paired_delta": paired_10.metric_deltas[recipe.primary_metric].paired_delta}))
            stages.append(self.paper_backend.finalize(root, recipe_id=paper_identity["recipe_id"], paired_result=paired_10))

            objective = (
                _certification_objective(recipe, paired_10, promotion_10, identity_10)
                if recipe.execution_class == "component_adapter"
                else None
            )
            capability_ids = [
                "candidate_coco_error_facts",
                "error_delta_next_round",
                "asha_queue_control",
            ]
            if recipe.recipe_id == "small_object_sampling":
                capability_ids.append("small_object_sampling_runtime")

            report = CertificationReport(
                certification_id=root.name or "mini-gpu-certification",
                level="mini_gpu_pilot",
                status="passed",
                model=model,
                data_yaml=data_yaml.as_posix(),
                device=device,
                environment=environment,
                protocol_hash=protocol_hash,
                certified_code_hash=code_hash,
                stages=stages,
                executed_recipe_id=survivor,
                executed_changed_variable=next(
                    item.changed_variable for item in CERTIFICATION_RECIPES if item.recipe_id == survivor
                ),
                paired_result_hashes=[*[result.result_hash for result in paired_results.values()], paired_10.result_hash],
                asha_survivor=survivor,
                objective=objective,
                promotion_results=promotion_results,
                capability_claims=[
                    CertificationCapabilityClaim(
                        capability_id=capability_id,
                        local_reproduction="locally_pilot_reproduced",
                        certification_level="mini_gpu_pilot",
                        recipe_id=(
                            recipe.recipe_id
                            if capability_id == "small_object_sampling_runtime"
                            else paper_identity["recipe_id"]
                        ),
                        snapshot_hash=paper_identity["snapshot_hash"],
                        evidence_hash=paired_10.result_hash,
                    )
                    for capability_id in capability_ids
                ],
            )
        except Exception as exc:
            failures.append(str(exc))
            stages.append(CertificationStage(stage_id="failure", status="failed", message=str(exc)))
            report = CertificationReport(
                certification_id=root.name or "mini-gpu-certification",
                level="mini_gpu_pilot",
                status="failed",
                model=model,
                data_yaml=data_yaml.as_posix(),
                device=device,
                protocol_hash=protocol_hash,
                certified_code_hash=code_hash,
                executed_recipe_id=recipe.recipe_id,
                executed_changed_variable=recipe.changed_variable,
                stages=stages,
                promotion_results=promotion_results,
                failures=failures,
            )
        report.to_yaml(root / "certification_report.yaml", exclude_none=True, sort_keys=False)
        return report


def missing_certification_dependencies() -> list[str]:
    """Return certification packages unavailable to the active Python interpreter."""
    missing: list[str] = []
    for package in CERTIFICATION_DEPENDENCIES:
        try:
            available = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return missing


class UltralyticsGpuBackend:
    """Subprocess backend used only by the explicit real-GPU command/test."""

    def environment(self) -> dict[str, Any]:
        missing = missing_certification_dependencies()
        if missing:
            raise RuntimeError(
                "Missing GPU certification dependencies: "
                f"{', '.join(missing)}. Install with: {CERTIFICATION_INSTALL_COMMAND}"
            )
        import torch
        import ultralytics

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; real GPU certification cannot run")
        executable = shutil.which("yolo")
        if executable is None:
            raise RuntimeError("Ultralytics yolo executable is not installed")
        device = torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "gpu_name": device.name,
            "gpu_memory_mb": round(device.total_memory / 1024 / 1024),
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "yolo_executable": executable,
        }

    def certify_component(
        self,
        *,
        component_id: str,
        workdir: Path,
        device: str,
    ) -> CertificationStage:
        root = workdir / "component_runtime_certification" / component_id
        registry_path = workdir / "component_maturity_registry.yaml"
        runner = ComponentCertificationRunner()
        cpu = runner.run(
            component_id=component_id,
            mode="cpu",
            workdir=root,
            registry_path=registry_path,
        )
        if cpu.status != "passed":
            raise RuntimeError(
                "component CPU certification failed: " + "; ".join(cpu.errors)
            )
        gpu = runner.run(
            component_id=component_id,
            mode="gpu",
            workdir=root,
            registry_path=registry_path,
            device=device,
            execute_gpu=True,
        )
        if gpu.status != "passed":
            raise RuntimeError(
                "component GPU certification failed: " + "; ".join(gpu.errors)
            )
        return _passed_stage(
            "component_runtime_certification",
            artifacts={
                "cpu_report": (
                    root / "component_certification.cpu.yaml"
                ).as_posix(),
                "gpu_report": (
                    root / "component_certification.gpu.yaml"
                ).as_posix(),
                "maturity_registry": registry_path.as_posix(),
            },
            metrics={
                "component_id": component_id,
                "cpu_final_maturity": cpu.final_maturity,
                "gpu_final_maturity": gpu.final_maturity,
                "cpu_report_hash": cpu.report_hash,
                "gpu_report_hash": gpu.report_hash,
            },
        )
    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]:
        command = [sys.executable, "-m", "yolo_agent.cli", "train", "--model", model, "--data", str(data_yaml), "--run-id", "certification-entrypoint", "--run-root", str(workdir / "entrypoint_runs"), "--profile", "debug", "--dry-run", "--auto-rounds", "0", "--no-auto-advance"]
        _run_command(command, workdir / "logs" / "train_entrypoint.log")
        return command

    def train(self, *, candidate_id: str, node_id: str, data_yaml: Path, model: str, workdir: Path, device: str, epochs: int, seed: int, protocol_hash: str, overrides: dict[str, Any]) -> BackendRun:
        project = workdir / "ultralytics"
        base_command = [
            str(shutil.which("yolo") or "yolo"), "detect", "train",
            f"model={model}", f"data={data_yaml}", f"project={project}", f"name={node_id}", "exist_ok=True",
            f"epochs={epochs}", "imgsz=640", "batch=4", f"device={device}", "workers=0", f"seed={seed}",
            "cache=False", "plots=False", "save=True", "val=True",
        ]
        runtime_artifacts: dict[str, Path] = {}
        if candidate_id in {"small_object_sampling", "sampling.small_object"}:
            payload_dir = workdir / "runtime_payloads" / node_id
            options = dict(overrides.get("data.sampling_policy", {}))
            options.update(
                {
                    "imgsz": 640,
                    "seed": seed,
                    "dataset_manifest": _hash_files(data_yaml.parent),
                }
            )
            contract = ComponentContract(
                component_id="sampling.small_object",
                display_name="Small Object Sampling",
                category="sampling",
                implementation_path=(
                    "yolo_agent.components.adapters.sampling.small_object_sampling"
                ),
                adapter_class="SmallObjectSamplingAdapter",
                insertion_point="train_dataloader_sampler",
                supported_detector_families=["yolo26"],
                fixed_imgsz_compatible=True,
                supports_amp=True,
                supports_ddp=True,
                maturity="smoke_passed",
            )
            context = AdapterContext(
                contract=contract,
                detector_family="yolo26",
                head="one_to_one",
                imgsz=640,
                workspace=payload_dir,
                options=options,
            )
            adapter = SmallObjectSamplingAdapter()
            preview = adapter.prepare_patch({}, {}, context, dry_run=False)
            payload = adapter.build_runtime_payload(
                context,
                protocol_hash=protocol_hash,
                base_command=base_command,
                generated_config={
                    "model_config": preview.patched_model_config,
                    "training_config": preview.patched_training_config,
                },
            )
            payload_path = payload.write(payload_dir / "adapter_runtime_payload.yaml")
            command = [
                sys.executable,
                "-m",
                payload.runtime_entrypoint,
                "--payload",
                str(payload_path),
                "--",
                *base_command,
            ]
            runtime_artifacts = {
                "runtime_payload": payload_path,
                "sampler_manifest": payload_dir / "sampler_manifest.json",
                "plugin_runtime_evidence": payload_dir / "plugin_runtime_evidence.json",
            }
        else:
            command = [
                *base_command,
                *[f"{key}={value}" for key, value in sorted(overrides.items())],
            ]
        _run_command(command, workdir / "logs" / f"{node_id}_train.log")
        run_dir = project / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not produce {checkpoint}")
        return BackendRun(
            candidate_id=candidate_id,
            node_id=node_id,
            run_dir=run_dir,
            checkpoint=checkpoint,
            command=command,
            runtime_artifacts=runtime_artifacts,
            protocol_identity=_backend_protocol_identity(
                data_yaml=data_yaml,
                protocol_hash=protocol_hash,
                epochs=epochs,
                seed=seed,
            ),
        )

    def evaluate(self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str) -> BackendEvaluation:
        output = workdir / "post_eval" / run.node_id
        log_path = workdir / "logs" / f"{run.node_id}_eval.log"
        command = [
            str(shutil.which("yolo") or "yolo"), "detect", "val",
            f"model={run.checkpoint}", f"data={data_yaml}", f"project={output.parent}", f"name={output.name}",
            "exist_ok=True", "imgsz=640", "split=val", f"device={device}", "workers=0", "save_json=True", "plots=False", "conf=0.001", "iou=0.7",
        ]
        _run_command(command, log_path)
        latency_ms = _parse_ultralytics_inference_latency(log_path)
        predictions = discover_coco_predictions_artifact(output)
        if predictions is None:
            raise RuntimeError(f"post-eval did not produce predictions.json for {run.node_id}")
        annotations = data_yaml.parent / "annotations" / "instances_val2017.json"
        eval_path = output / "coco_eval.json"
        write_coco_eval_report(annotations_path=annotations, predictions_path=predictions, output_path=eval_path)
        error_report = mine_coco_errors(annotations, predictions, out_prefix=output / "coco_error_report")
        error_path = output / "coco_error_report.json"
        if not error_path.is_file():
            error_path.write_text(error_report.model_dump_json(indent=2), encoding="utf-8")
        return BackendEvaluation(
            eval_path=eval_path,
            predictions_path=predictions,
            error_report_path=error_path,
            latency_ms=latency_ms,
            model_size_mb=run.checkpoint.stat().st_size / (1024 * 1024),
            command=command,
        )


def _import_observation(store: EvidenceStore, run_id: str, run: BackendRun, evaluation: BackendEvaluation, identity: dict[str, Any], role: str) -> None:
    contract = validate_coco_evidence_artifacts(
        predictions_path=evaluation.predictions_path,
        eval_path=evaluation.eval_path,
        error_report_path=evaluation.error_report_path,
    )
    if not contract.valid:
        raise RuntimeError(
            f"COCO evidence artifact contract failed for {run.node_id}: "
            f"{json.dumps(contract.invalid_artifacts, sort_keys=True)}"
        )
    _append_guard_metrics(evaluation)
    store.log_artifact_manifest(
        run_id=run_id,
        name=f"{run.node_id}_coco_predictions",
        artifact_path=evaluation.predictions_path,
        producer_stage="real_gpu_certification",
        candidate_id=run.candidate_id,
        node_id=run.node_id,
        protocol_hash=str(identity.get("protocol_hash") or "") or None,
    )
    import_coco_eval_metrics(
        evaluation.eval_path,
        store,
        run_id,
        run.candidate_id,
        run.node_id,
        dataset_version="mini-coco-v1",
        split="val2017",
        source="real_gpu_certification",
        matched_identity=identity,
        evidence_role=role,
        error_report_path=evaluation.error_report_path,
    )


def _append_guard_metrics(evaluation: BackendEvaluation) -> None:
    """Persist measured guard metrics beside COCO metrics before evidence import."""
    payload = json.loads(evaluation.eval_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO eval must be a mapping: {evaluation.eval_path}")
    payload["latency_ms"] = evaluation.latency_ms
    payload["model_size_mb"] = evaluation.model_size_mb
    evaluation.eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _certification_scheduler(
    run_id: str,
    *,
    cohort_size: int,
    target_error_required: bool,
) -> ASHAScheduler:
    rungs = []
    for rung in default_asha_rungs():
        if rung.stage_id == "pilot_3":
            rungs.append(
                rung.model_copy(
                    update={
                        "require_positive_paired_delta": target_error_required,
                        "minimum_completed": cohort_size,
                        "minimum_promotions": 1 if cohort_size == 1 else 0,
                    }
                )
            )
        elif rung.stage_id == "pilot_10":
            rungs.append(
                rung.model_copy(
                    update={
                        "require_positive_paired_delta": target_error_required,
                        "require_target_error_improvement": target_error_required,
                    }
                )
            )
        else:
            rungs.append(rung)
    return ASHAScheduler(ASHAStudy(study_id=f"{run_id}_certification", base_run_id=run_id, rungs=rungs))


def _asha_observation(
    stage_id: str,
    paired: PairedExperimentResult,
    *,
    seed: int,
    primary_metric: str,
    promotion: CertificationPromotionResult,
) -> ASHAObservation:
    primary = paired.metric_deltas[primary_metric]
    return ASHAObservation(
        stage_id=stage_id,  # type: ignore[arg-type]
        node_id=paired.candidate_node_id,
        seed=seed,
        paired_delta=primary.paired_delta,
        paired_result_verified=paired.verified,
        paired_result_hash=paired.result_hash,
        protocol_match_status=paired.protocol_match_status,
        paired_experiment_result=paired,
        target_error_improved_count=sum(
            1 for value in promotion.error_fact_deltas.values() if value > 0
        ),
        latency_regression=promotion.guard_regressions.get("latency"),
        model_size_regression=promotion.guard_regressions.get("model_size"),
        diagnosis_gate_passed=promotion.passed,
        diagnosis_checks=[
            {"check": key, "passed": value}
            for key, value in sorted(promotion.checks.items())
        ],
        promotion_rejection_reasons=list(promotion.rejection_reasons),
        evidence_complete=True,
    )


def _target_error_facts(recipe: CertificationRecipe) -> list[dict[str, Any]]:
    if recipe.recipe_id != "small_object_sampling":
        return []
    target = recipe.target_class_names[0]
    return [
        {
            "fact_type": "area_metric",
            "subject": "small",
            "area": "small",
            "metric_name": "ap_small",
        },
        {
            "fact_type": "per_class_metric",
            "subject": target,
            "class_name": target,
            "metric_name": "per_class_ar",
        },
    ]


def _additional_metrics(recipe: CertificationRecipe) -> list[str]:
    metrics = ["map50_95"]
    metrics.extend(f"per_class_ar/{name}" for name in recipe.target_class_names)
    return metrics


def _run_paired_bootstrap(
    *,
    data_yaml: Path,
    control: BackendEvaluation,
    candidate: BackendEvaluation,
    output: Path,
    seed: int,
) -> PairedBootstrapReport:
    report = paired_bootstrap_coco_predictions(
        data_yaml.parent / "annotations" / "instances_val2017.json",
        control.predictions_path,
        candidate.predictions_path,
        config=PairedBootstrapConfig(
            iterations=200,
            minimum_images=4,
            random_seed=20260728 + seed,
        ),
    )
    report.to_json(output)
    return report


def _import_bootstrap_metrics(
    store: EvidenceStore,
    *,
    run_id: str,
    run: BackendRun,
    identity: dict[str, Any],
    report: PairedBootstrapReport,
) -> None:
    if report.status != "completed" or report.overall is None:
        return
    metrics: dict[str, Any] = {
        "bootstrap/diagnostic_map50_ci_low": report.overall.confidence_interval_low,
        "bootstrap/diagnostic_map50_ci_high": report.overall.confidence_interval_high,
        "bootstrap/diagnostic_map50_probability_improvement": report.overall.probability_improvement,
        "bootstrap/diagnostic_map50_direction": report.overall.direction,
        "bootstrap/matched_image_count": report.matched_image_count,
    }
    store.upsert_candidate_metrics(
        run_id=run_id,
        candidate_id=run.candidate_id,
        node_id=run.node_id,
        metrics=metrics,
        dataset_version="mini-coco-v1",
        split="val2017",
        source="real_gpu_certification_paired_bootstrap",
        verified=True,
        validator="paired_bootstrap",
        source_artifact=report.candidate_predictions,
        evidence_role="current_observation",
        **identity,
    )


def _promotion_result(
    stage_id: str,
    recipe: CertificationRecipe,
    paired: PairedExperimentResult,
    control: BackendEvaluation,
    candidate: BackendEvaluation,
    bootstrap: PairedBootstrapReport,
) -> CertificationPromotionResult:
    primary_delta = paired.metric_deltas[recipe.primary_metric].paired_delta
    if recipe.execution_class != "component_adapter":
        return CertificationPromotionResult(
            stage_id=stage_id,  # type: ignore[arg-type]
            passed=True,
            primary_metric=recipe.primary_metric,
            metric_deltas={recipe.primary_metric: primary_delta},
            checks={"native_certification_policy": True},
        )

    map_delta = paired.metric_deltas["map50_95"].paired_delta
    target_recall = {
        name: paired.metric_deltas[f"per_class_ar/{name}"].paired_delta
        for name in recipe.target_class_names
    }
    fn_deltas = {
        name: _false_negative_effect(control.error_report_path, candidate.error_report_path, name)
        for name in recipe.target_class_names
    }
    latency_regression = _relative_regression(control.latency_ms, candidate.latency_ms)
    size_regression = _relative_regression(control.model_size_mb, candidate.model_size_mb)
    bootstrap_not_regressed = bool(
        bootstrap.status == "completed"
        and bootstrap.overall is not None
        and bootstrap.overall.direction != "stable_regression"
        and not set(recipe.target_class_names).intersection(bootstrap.stable_regressed_classes)
    )
    checks = {
        "protocol_matched": paired.protocol_match_status == "matched" and paired.verified,
        "ap_small_improved": primary_delta > 0,
        "target_class_recall_improved": all(value > 0 for value in target_recall.values()),
        "false_negative_reduced": all(value > 0 for value in fn_deltas.values()),
        "overall_map_guard": map_delta >= -recipe.max_map_regression,
        "latency_guard": latency_regression <= recipe.max_latency_regression,
        "model_size_guard": size_regression <= recipe.max_model_size_regression,
        "paired_bootstrap_not_regressed": bootstrap_not_regressed,
    }
    reasons = [key for key, passed in checks.items() if not passed]
    return CertificationPromotionResult(
        stage_id=stage_id,  # type: ignore[arg-type]
        passed=all(checks.values()),
        primary_metric=recipe.primary_metric,
        metric_deltas={
            "ap_small": primary_delta,
            "map50_95": map_delta,
            **{f"per_class_ar/{key}": value for key, value in target_recall.items()},
        },
        error_fact_deltas={f"false_negative/{key}": value for key, value in fn_deltas.items()},
        guard_regressions={"latency": latency_regression, "model_size": size_regression},
        checks=checks,
        rejection_reasons=reasons,
    )


def _promotion_stage(result: CertificationPromotionResult) -> CertificationStage:
    return CertificationStage(
        stage_id="promotion_gate",
        status="passed" if result.passed else "failed",
        message="diagnosis-bound promotion gate",
        metrics=result.model_dump(mode="json"),
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


def _false_negative_effect(control_path: Path, candidate_path: Path, class_name: str) -> float:
    control = _class_error_count(control_path, "false_negative_top_classes", class_name, "false_negative")
    candidate = _class_error_count(candidate_path, "false_negative_top_classes", class_name, "false_negative")
    return float(control - candidate)


def _class_error_count(path: Path, group: str, class_name: str, field: str) -> int:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = payload.get(group, []) if isinstance(payload, dict) else []
    for item in values if isinstance(values, list) else []:
        if isinstance(item, dict) and str(item.get("name")) == class_name:
            return int(item.get(field, 0) or 0)
    return 0


def _relative_regression(control: float, candidate: float) -> float:
    return (candidate - control) / control if control > 0 else float("inf")


def _validate_runtime_artifacts(
    run: BackendRun,
    *,
    protocol_hash: str,
) -> tuple[dict[str, Path], dict[str, bool | int | str]]:
    required = {"runtime_payload", "sampler_manifest", "plugin_runtime_evidence"}
    missing = sorted(
        key
        for key in required
        if key not in run.runtime_artifacts or not run.runtime_artifacts[key].is_file()
    )
    if missing:
        raise RuntimeError("small-object runtime artifacts missing: " + ", ".join(missing))
    payload = AdapterRuntimePayload.read(
        run.runtime_artifacts["runtime_payload"],
        verify_imports=True,
    )
    manifest = SmallObjectSamplingManifest.model_validate_json(
        run.runtime_artifacts["sampler_manifest"].read_text(encoding="utf-8-sig")
    )
    evidence = PluginRuntimeEvidence.model_validate_json(
        run.runtime_artifacts["plugin_runtime_evidence"].read_text(
            encoding="utf-8-sig"
        )
    )
    hook_calls = sum(
        hooks.get("build_train_dataloader", 0)
        for hooks in evidence.hook_call_counts.values()
    )
    checks: dict[str, bool | int | str] = {
        "payload_component_matched": payload.component_ids == ["sampling.small_object"],
        "payload_protocol_matched": payload.protocol_hash == protocol_hash,
        "manifest_protocol_matched": manifest.protocol_hash == protocol_hash,
        "manifest_payload_matched": manifest.runtime_payload_hash == payload.payload_hash,
        "manifest_train_split": manifest.split == "train",
        "manifest_val_unchanged": manifest.val_unchanged,
        "manifest_weights_complete": bool(
            manifest.image_count > 0
            and len(manifest.raw_weights)
            == len(manifest.final_weights)
            == manifest.image_count
        ),
        "runtime_evidence_matched": bool(
            evidence.payload_hash == payload.payload_hash
            and evidence.protocol_hash == protocol_hash
        ),
        "runtime_compatible": evidence.compatible,
        "runtime_failures_empty": not evidence.failures,
        "train_dataloader_hook_calls": hook_calls,
        "train_dataloader_hook_called": hook_calls > 0,
    }
    failed = sorted(
        key
        for key, value in checks.items()
        if key != "train_dataloader_hook_calls" and value is not True
    )
    if failed:
        raise RuntimeError(
            "small-object runtime artifact contract failed: " + ", ".join(failed)
        )
    return run.runtime_artifacts, checks


def _certification_objective(
    recipe: CertificationRecipe,
    paired: PairedExperimentResult,
    promotion: CertificationPromotionResult,
    identity: dict[str, Any],
) -> CertificationObjectiveResult:
    bootstrap = paired.paired_bootstrap_ci
    return CertificationObjectiveResult(
        objective_hash=_hash_payload(
            {
                "recipe_id": recipe.recipe_id,
                "primary_metric": recipe.primary_metric,
                "max_map_regression": recipe.max_map_regression,
                "max_latency_regression": recipe.max_latency_regression,
                "max_model_size_regression": recipe.max_model_size_regression,
            }
        ),
        primary_metric=recipe.primary_metric,
        required_delta=0.0,
        observed_delta=paired.metric_deltas[recipe.primary_metric].paired_delta,
        baseline_seeds=[int(identity["seed"])],
        candidate_seeds=[int(identity["seed"])],
        latency_regression=promotion.guard_regressions.get("latency"),
        model_size_regression=promotion.guard_regressions.get("model_size"),
        passed=promotion.passed,
        dataset_manifest_hash=str(identity["dataset_manifest_sha256"]),
        subset_manifest_hash=str(identity["subset_manifest_sha256"]),
        seed_policy_hash=_hash_payload({"seed": identity["seed"]}),
        batch_policy_hash=str(identity["batch_policy_hash"]),
        ultralytics_version=str(identity["ultralytics_version"]),
        eval_protocol_hash=str(identity["eval_protocol_hash"]),
        paired_bootstrap_ci=(
            (bootstrap.confidence_interval_low, bootstrap.confidence_interval_high)
            if bootstrap is not None
            else None
        ),
        latency_guard_passed=promotion.checks.get("latency_guard", False),
        model_size_guard_passed=promotion.checks.get("model_size_guard", False),
        target_metric_deltas=dict(promotion.metric_deltas),
        target_error_fact_deltas=dict(promotion.error_fact_deltas),
    )


def _node(candidate_id: str, node_id: str, overrides: dict[str, Any]) -> ExperimentNode:
    return ExperimentNode(
        node_id=node_id,
        candidate_config=CandidateConfig(candidate_id=candidate_id, base_model="yolo26n.pt", scale="n", framework="ultralytics", train_overrides=overrides),
        data_version="mini-coco-v1",
        changed_variables=overrides or {"baseline": True},
        command_spec=CommandSpec(command_type="custom", argv=["real-gpu-certification"]),
    )


def _matched_identity(data_yaml: Path, environment: dict[str, Any], *, protocol_hash: str, epochs: int, fidelity: str, seed: int) -> dict[str, Any]:
    dataset_hash = _hash_files(data_yaml.parent)
    return {
        "protocol_hash": protocol_hash,
        "dataset_manifest_sha256": dataset_hash,
        "subset_manifest_sha256": dataset_hash,
        "eval_protocol_hash": _hash_payload({"protocol": "mini-coco-post-eval", "imgsz": 640}),
        "seed": seed,
        "fidelity": fidelity,
        "epochs": epochs,
        "batch_policy_hash": _hash_payload({"batch": 4, "device": "single_gpu"}),
        "ultralytics_version": str(environment.get("ultralytics_version") or importlib.metadata.version("ultralytics")),
        "imgsz": 640,
    }


def _backend_protocol_identity(
    *,
    data_yaml: Path,
    protocol_hash: str,
    epochs: int,
    seed: int,
) -> PaperProtocolIdentity:
    return build_paper_protocol_identity(
        data_yaml=data_yaml,
        protocol_hash=protocol_hash,
        epochs=epochs,
        seed=seed,
        objective_hash=_hash_payload(
            {
                "primary_metric": "ap_small",
                "target_metrics": ["per_class_ar/object"],
                "target_error_facts": ["false_negative/object"],
            }
        ),
    )


def _passed_stage(stage_id: str, *, command: list[str] | None = None, artifacts: dict[str, str] | None = None, metrics: dict[str, Any] | None = None) -> CertificationStage:
    now = datetime.now(timezone.utc)
    return CertificationStage(stage_id=stage_id, status="passed", command=command or [], artifacts=artifacts or {}, metrics=metrics or {}, started_at=now, completed_at=now)


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONUTF8", "1")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log_path}")


def _parse_ultralytics_inference_latency(log_path: Path) -> float:
    """Read Ultralytics' measured per-image inference latency from a val log."""
    text = log_path.read_text(encoding="utf-8-sig", errors="replace")
    matches = re.findall(
        r"Speed:.*?([0-9]+(?:\.[0-9]+)?)ms inference.*?per image",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise RuntimeError(
            "Ultralytics validation did not report per-image inference latency; "
            f"inspect {log_path}"
        )
    latency_ms = float(matches[-1])
    if latency_ms <= 0:
        raise RuntimeError(f"invalid inference latency {latency_ms}ms in {log_path}")
    return latency_ms


def _fidelity_hash(protocol_hash: str, fidelity: str) -> str:
    return _hash_payload({"protocol_hash": protocol_hash, "fidelity": fidelity})


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _hash_files(root: Path) -> str:
    values = [(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob("*")) if path.is_file()]
    return _hash_payload({"files": values})


__all__ = ["BackendEvaluation", "BackendRun", "GpuAcceptanceBackend", "RealGpuAcceptanceSuite", "UltralyticsGpuBackend"]
