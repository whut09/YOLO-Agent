"""Opt-in real GPU acceptance driver for the training evidence loop."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.coco_post_eval import write_coco_eval_report
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
from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.paired_experiment import PairedExperimentResult, build_paired_experiment_result
from yolo_agent.core.pilot_evidence import validate_coco_evidence_artifacts
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics
from yolo_agent.tools.coco_error_mining import mine_coco_errors


class BackendRun(BaseModel):
    candidate_id: str
    node_id: str
    run_dir: Path
    checkpoint: Path
    command: list[str] = Field(default_factory=list)


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


CERTIFICATION_RECIPES = (
    CertificationRecipe(recipe_id="reduce_mosaic", changed_variable="mosaic", overrides={"mosaic": 0.0}),
    CertificationRecipe(recipe_id="close_mosaic_early", changed_variable="close_mosaic", overrides={"close_mosaic": 5}),
    CertificationRecipe(recipe_id="light_mixup", changed_variable="mixup", overrides={"mixup": 0.05}),
)


class GpuAcceptanceBackend(Protocol):
    def environment(self) -> dict[str, Any]: ...
    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]: ...
    def train(self, *, candidate_id: str, node_id: str, data_yaml: Path, model: str, workdir: Path, device: str, epochs: int, seed: int, overrides: dict[str, Any]) -> BackendRun: ...
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
        root = Path(workdir)
        root.mkdir(parents=True, exist_ok=True)
        data_yaml = create_mini_coco_fixture(root / "mini_coco")
        code_hash = certification_code_hash()
        protocol_hash = _hash_payload(
            {"suite": "mini_gpu_pilot.v1", "model": model, "imgsz": 640, "code_hash": code_hash}
        )
        stages: list[CertificationStage] = []
        failures: list[str] = []
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
                stages=[CertificationStage(stage_id="environment", status="skipped", message="Pass --execute-real-gpu to opt in.")],
                failures=["real_gpu_execution_not_confirmed"],
            )
            report.to_yaml(root / "certification_report.yaml", exclude_none=True, sort_keys=False)
            return report
        try:
            environment = self.backend.environment()
            stages.append(_passed_stage("environment", metrics=environment))
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

            debug = self.backend.train(candidate_id="debug", node_id="debug", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=1, seed=1, overrides={})
            stages.append(_passed_stage("debug", command=debug.command, artifacts={"checkpoint": debug.checkpoint.as_posix()}))

            control_3 = self.backend.train(candidate_id="baseline_pilot_3", node_id="baseline_pilot_3", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=3, seed=1, overrides={})
            stages.append(_passed_stage("pilot_3_control", command=control_3.command, artifacts={"checkpoint": control_3.checkpoint.as_posix()}))
            candidates = [(item.recipe_id, dict(item.overrides)) for item in CERTIFICATION_RECIPES]
            candidate_runs = [
                self.backend.train(candidate_id=candidate_id, node_id=f"{candidate_id}_pilot_3", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=3, seed=1, overrides=overrides)
                for candidate_id, overrides in candidates
            ]
            stages.append(_passed_stage("pilot_3_candidates", metrics={"candidate_count": len(candidate_runs)}))

            store = EvidenceStore(root / "evidence")
            error_store = ErrorFactStore(root / "evidence")
            run_id = "mini_gpu_certification"
            eval_control_3 = self.backend.evaluate(run=control_3, data_yaml=data_yaml, workdir=root, device=device)
            identity_3 = _matched_identity(data_yaml, environment, protocol_hash=_fidelity_hash(protocol_hash, "pilot_3"), epochs=3, fidelity="pilot_3", seed=1)
            _import_observation(store, run_id, control_3, eval_control_3, identity_3, "baseline_reference")
            evaluations: dict[str, BackendEvaluation] = {}
            paired_results: dict[str, PairedExperimentResult] = {}
            for candidate_run in candidate_runs:
                evaluation = self.backend.evaluate(run=candidate_run, data_yaml=data_yaml, workdir=root, device=device)
                evaluations[candidate_run.candidate_id] = evaluation
                _import_observation(store, run_id, candidate_run, evaluation, identity_3, "current_observation")
                paired_results[candidate_run.candidate_id] = build_paired_experiment_result(
                    run_id=run_id,
                    candidate_id=candidate_run.candidate_id,
                    candidate_node_id=candidate_run.node_id,
                    metric_records=store.load_run(run_id).metric_records,
                    error_facts=error_store.read(run_id),
                    primary_metric="map50_95",
                    target_error_facts=[],
                )
            stages.append(_passed_stage("post_eval", metrics={"evaluated_nodes": 1 + len(evaluations)}))
            fact_count = len(error_store.read(run_id))
            if fact_count == 0:
                raise RuntimeError("COCO post-eval produced no error facts")
            stages.append(_passed_stage("error_facts", metrics={"error_fact_count": fact_count}))
            if not all(result.verified for result in paired_results.values()):
                raise RuntimeError("one or more pilot_3 paired results are not verified")
            stages.append(_passed_stage("paired_delta", metrics={key: value.metric_deltas["map50_95"].paired_delta for key, value in paired_results.items()}))

            scheduler = _certification_scheduler(run_id)
            baseline_node = _node(control_3.candidate_id, control_3.node_id, {})
            for candidate_id, overrides in candidates:
                candidate_run = next(item for item in candidate_runs if item.candidate_id == candidate_id)
                scheduler.register_trial(
                    trial_id=candidate_id,
                    candidate_id=candidate_id,
                    source_run_id=run_id,
                    source_node=_node(candidate_id, candidate_run.node_id, overrides),
                    baseline_control_node=baseline_node,
                )
            for candidate_id, _ in candidates:
                paired = paired_results[candidate_id]
                scheduler.report(candidate_id, _asha_observation("pilot_3", paired, seed=1))
            assignment = scheduler.next_assignment()
            if assignment is None or assignment.stage_id != "pilot_10":
                raise RuntimeError("ASHA did not produce a pilot_10 survivor")
            survivor = assignment.candidate_id
            stages.append(_passed_stage("asha_decision", metrics={"survivor": survivor, "assignment_id": assignment.assignment_id}))

            winner_overrides = dict(next(overrides for candidate_id, overrides in candidates if candidate_id == survivor))
            control_10 = self.backend.train(candidate_id="baseline_pilot_10", node_id="baseline_pilot_10", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=10, seed=1, overrides={})
            winner_10 = self.backend.train(candidate_id=survivor, node_id=f"{survivor}_pilot_10", data_yaml=data_yaml, model=model, workdir=root, device=device, epochs=10, seed=1, overrides=winner_overrides)
            eval_control_10 = self.backend.evaluate(run=control_10, data_yaml=data_yaml, workdir=root, device=device)
            eval_winner_10 = self.backend.evaluate(run=winner_10, data_yaml=data_yaml, workdir=root, device=device)
            identity_10 = _matched_identity(data_yaml, environment, protocol_hash=_fidelity_hash(protocol_hash, "pilot_10"), epochs=10, fidelity="pilot_10", seed=1)
            _import_observation(store, run_id, control_10, eval_control_10, identity_10, "baseline_reference")
            _import_observation(store, run_id, winner_10, eval_winner_10, identity_10, "current_observation")
            paired_10 = build_paired_experiment_result(run_id=run_id, candidate_id=survivor, candidate_node_id=winner_10.node_id, metric_records=store.load_run(run_id).metric_records, error_facts=error_store.read(run_id), primary_metric="map50_95", target_error_facts=[])
            if not paired_10.verified:
                raise RuntimeError("pilot_10 paired result is not verified")
            scheduler.report(survivor, _asha_observation("pilot_10", paired_10, seed=1))
            stages.append(_passed_stage("pilot_10", metrics={"candidate": survivor, "paired_delta": paired_10.metric_deltas["map50_95"].paired_delta}))
            stages.append(self.paper_backend.finalize(root, recipe_id=paper_identity["recipe_id"], paired_result=paired_10))

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
                capability_claims=[
                    CertificationCapabilityClaim(capability_id=capability_id, local_reproduction="locally_pilot_reproduced", certification_level="mini_gpu_pilot", recipe_id=paper_identity["recipe_id"], snapshot_hash=paper_identity["snapshot_hash"], evidence_hash=paired_10.result_hash)
                    for capability_id in ("candidate_coco_error_facts", "error_delta_next_round", "asha_queue_control")
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
                stages=stages,
                failures=failures,
            )
        report.to_yaml(root / "certification_report.yaml", exclude_none=True, sort_keys=False)
        return report


class UltralyticsGpuBackend:
    """Subprocess backend used only by the explicit real-GPU command/test."""

    def environment(self) -> dict[str, Any]:
        import torch
        import ultralytics
        import pycocotools  # noqa: F401

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

    def train_entrypoint(self, *, data_yaml: Path, model: str, workdir: Path, device: str) -> list[str]:
        command = [sys.executable, "-m", "yolo_agent.cli", "train", "--model", model, "--data", str(data_yaml), "--run-id", "certification-entrypoint", "--run-root", str(workdir / "entrypoint_runs"), "--profile", "debug", "--dry-run", "--auto-rounds", "0", "--no-auto-advance"]
        _run_command(command, workdir / "logs" / "train_entrypoint.log")
        return command

    def train(self, *, candidate_id: str, node_id: str, data_yaml: Path, model: str, workdir: Path, device: str, epochs: int, seed: int, overrides: dict[str, Any]) -> BackendRun:
        project = workdir / "ultralytics"
        command = [
            str(shutil.which("yolo") or "yolo"), "detect", "train",
            f"model={model}", f"data={data_yaml}", f"project={project}", f"name={node_id}", "exist_ok=True",
            f"epochs={epochs}", "imgsz=640", "batch=4", f"device={device}", "workers=0", f"seed={seed}",
            "cache=False", "plots=False", "save=True", "val=True",
            *[f"{key}={value}" for key, value in sorted(overrides.items())],
        ]
        _run_command(command, workdir / "logs" / f"{node_id}_train.log")
        run_dir = project / node_id
        checkpoint = run_dir / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not produce {checkpoint}")
        return BackendRun(candidate_id=candidate_id, node_id=node_id, run_dir=run_dir, checkpoint=checkpoint, command=command)

    def evaluate(self, *, run: BackendRun, data_yaml: Path, workdir: Path, device: str) -> BackendEvaluation:
        output = workdir / "post_eval" / run.node_id
        command = [
            str(shutil.which("yolo") or "yolo"), "detect", "val",
            f"model={run.checkpoint}", f"data={data_yaml}", f"project={output.parent}", f"name={output.name}",
            "exist_ok=True", "imgsz=640", "split=val", f"device={device}", "workers=0", "save_json=True", "plots=False", "conf=0.001", "iou=0.7",
        ]
        started = time.perf_counter()
        _run_command(command, workdir / "logs" / f"{run.node_id}_eval.log")
        duration_ms = (time.perf_counter() - started) * 1000.0
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
            latency_ms=duration_ms,
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


def _certification_scheduler(run_id: str) -> ASHAScheduler:
    rungs = []
    for rung in default_asha_rungs():
        if rung.stage_id == "pilot_3":
            rungs.append(rung.model_copy(update={"require_positive_paired_delta": False, "minimum_completed": 3}))
        elif rung.stage_id == "pilot_10":
            rungs.append(rung.model_copy(update={"require_positive_paired_delta": False, "require_target_error_improvement": False}))
        else:
            rungs.append(rung)
    return ASHAScheduler(ASHAStudy(study_id=f"{run_id}_certification", base_run_id=run_id, rungs=rungs))


def _asha_observation(stage_id: str, paired: PairedExperimentResult, *, seed: int) -> ASHAObservation:
    primary = paired.metric_deltas["map50_95"]
    return ASHAObservation(
        stage_id=stage_id,  # type: ignore[arg-type]
        node_id=paired.candidate_node_id,
        seed=seed,
        paired_delta=primary.paired_delta,
        paired_result_verified=paired.verified,
        paired_result_hash=paired.result_hash,
        protocol_match_status=paired.protocol_match_status,
        paired_experiment_result=paired,
        target_error_improved_count=sum(1 for item in paired.target_error_fact_deltas if item.improved),
        diagnosis_gate_passed=True,
        evidence_complete=True,
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


def _fidelity_hash(protocol_hash: str, fidelity: str) -> str:
    return _hash_payload({"protocol_hash": protocol_hash, "fidelity": fidelity})


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _hash_files(root: Path) -> str:
    values = [(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob("*")) if path.is_file()]
    return _hash_payload({"files": values})


__all__ = ["BackendEvaluation", "BackendRun", "GpuAcceptanceBackend", "RealGpuAcceptanceSuite", "UltralyticsGpuBackend"]
