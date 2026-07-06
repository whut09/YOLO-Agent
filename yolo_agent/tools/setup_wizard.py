"""First-run setup wizard for friendly YOLO Agent onboarding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.loop_io import write_yaml
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.doctor import DoctorCheck, DoctorReport, run_doctor


SetupKind = Literal["coco"]


class SetupResult(BaseModel):
    """Artifacts and next commands produced by the setup wizard."""

    kind: SetupKind
    ok: bool
    data_yaml: Path
    model: str
    run_id: str
    run_root: Path
    env_file: Path
    llm_config_path: Path
    setup_report_path: Path
    doctor_errors: int = 0
    doctor_warnings: int = 0
    failed_checks: list[DoctorCheck] = Field(default_factory=list)
    warning_checks: list[DoctorCheck] = Field(default_factory=list)
    openai_key_detected: bool = False
    next_command: str
    status_command: str
    notes: list[str] = Field(default_factory=list)


def run_setup_wizard(
    *,
    kind: SetupKind,
    data_yaml: Path | str,
    model: str = "yolo26n.pt",
    run_id: str | None = None,
    run_root: Path | str = "runs",
    env_file: Path | str = ".env.local",
    llm_config_path: Path | str = ResourcePaths.LLM_DECISION_LOCAL,
    setup_report_path: Path | str | None = None,
    overwrite: bool = False,
) -> SetupResult:
    """Generate local onboarding files and a COCO path check report."""
    data_path = Path(data_yaml)
    run_root_path = Path(run_root)
    resolved_run_id = run_id or _default_run_id(kind, model)
    env_path = Path(env_file)
    llm_path = Path(llm_config_path)
    report_path = Path(setup_report_path) if setup_report_path is not None else run_root_path / resolved_run_id / "setup_report.yaml"
    doctor = run_doctor(data_yaml=data_path, model=model, run_root=run_root_path, kind=kind)
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    _write_env_file(env_path, openai_key=openai_key, overwrite=overwrite)
    _write_llm_config(llm_path, overwrite=overwrite)
    llm_config = _load_local_llm_config(llm_path)
    llm_key_detected = bool(llm_config.resolved_api_key()) if llm_config is not None else bool(openai_key)
    _write_setup_report(
        report_path,
        kind=kind,
        data_yaml=data_path,
        model=model,
        run_id=resolved_run_id,
        run_root=run_root_path,
        env_file=env_path,
        llm_config_path=llm_path,
        doctor=doctor,
        openai_key_detected=llm_key_detected,
        llm_config=llm_config,
    )
    return SetupResult(
        kind=kind,
        ok=doctor.ok,
        data_yaml=data_path,
        model=model,
        run_id=resolved_run_id,
        run_root=run_root_path,
        env_file=env_path,
        llm_config_path=llm_path,
        setup_report_path=report_path,
        doctor_errors=doctor.error_count,
        doctor_warnings=doctor.warning_count,
        failed_checks=_failed_checks(doctor),
        warning_checks=_warning_checks(doctor),
        openai_key_detected=llm_key_detected,
        next_command=_optimize_command(kind, data_path, model, resolved_run_id, run_root_path),
        status_command=f"yolo-agent loop status --run {(run_root_path / resolved_run_id).as_posix()}",
        notes=_notes(doctor, openai_key_detected=llm_key_detected),
    )


def _write_env_file(path: Path, *, openai_key: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    value = openai_key if openai_key else "PUT_YOUR_OPENAI_API_KEY_HERE"
    lines = [
        "# Local environment for YOLO Agent. This file is git-ignored.",
        f"OPENAI_API_KEY={value}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_llm_config(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    data = {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-5.5",
        "model_alias": "codex-gpt5.5",
        "api_key": None,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "base_url_env": "OPENAI_BASE_URL",
        "timeout_seconds": 60,
        "temperature": 0.1,
        "max_output_tokens": 4096,
        "decision_role": "proposal_generator_only",
        "use_by_default": True,
        "require_api_key": True,
        "allowed_outputs": [
            "diagnosis_summary",
            "policy_proposals",
            "evidence_requests",
            "doctor_report_draft",
        ],
        "blocked_outputs": [
            "direct_experiment_approval",
            "direct_training_execution",
            "best_model_claim_without_evidence",
        ],
        "safety_contract": [
            "The LLM may generate policy proposals and analysis text only.",
            "Harness gates make executable decisions.",
            "If required evidence is missing, propose evidence actions before training actions.",
        ],
        "prompt_contract": {
            "system_summary": (
                "You are the YOLO Agent decision-analysis model. Produce structured diagnosis, "
                "evidence requests, and policy proposals. Do not approve experiments directly."
            ),
            "required_inputs": [
                "task_spec",
                "dataset_report",
                "error_facts",
                "evidence_status",
                "deployment_constraints",
                "loop_policy",
            ],
            "required_output_schema": {
                "primary_problem": "string",
                "likely_causes": "list[string]",
                "evidence": "list[string]",
                "rejected_actions": "list[object]",
                "selected_actions": "list[object]",
                "expected_improvement": "object",
                "stop_condition": "list[string]",
            },
        },
    }
    write_yaml(path, data)


def _write_setup_report(
    path: Path,
    *,
    kind: SetupKind,
    data_yaml: Path,
    model: str,
    run_id: str,
    run_root: Path,
    env_file: Path,
    llm_config_path: Path,
    doctor: DoctorReport,
    openai_key_detected: bool,
    llm_config: LLMDecisionConfig | None = None,
) -> None:
    data = {
        "kind": kind,
        "data_yaml": data_yaml.as_posix(),
        "model": model,
        "run_id": run_id,
        "run_root": run_root.as_posix(),
        "env_file": env_file.as_posix(),
        "llm_config_path": llm_config_path.as_posix(),
        "openai_key_detected": openai_key_detected,
        "llm_key_source": llm_config.api_key_source() if llm_config is not None else "env:OPENAI_API_KEY",
        "llm_base_url_source": llm_config.base_url_source() if llm_config is not None else "default",
        "doctor": doctor.model_dump(mode="json"),
        "next_command": _optimize_command(kind, data_yaml, model, run_id, run_root),
        "status_command": f"yolo-agent loop status --run {(run_root / run_id).as_posix()}",
    }
    write_yaml(path, data)


def _default_run_id(kind: SetupKind, model: str) -> str:
    stem = Path(model).stem.replace(".", "-").replace("_", "-")
    return f"{kind}-{stem}"


def _optimize_command(kind: SetupKind, data_yaml: Path, model: str, run_id: str, run_root: Path) -> str:
    return (
        f"yolo-agent optimize {kind} --model {model} --data {data_yaml.as_posix()} "
        f"--goal +2map --run-id {run_id} --run-root {run_root.as_posix()} --profile debug --execute"
    )


def _notes(doctor: DoctorReport, *, openai_key_detected: bool) -> list[str]:
    notes: list[str] = []
    if not openai_key_detected:
        notes.append(
            "LLM API key was not detected; edit .env.local, set OPENAI_API_KEY, or set api_key in "
            "configs/local/llm_decision.local.yaml before LLM proposals are used."
        )
    if doctor.error_count:
        notes.append("Doctor found hard errors; fix them before running optimize.")
    if doctor.warning_count:
        notes.append("Doctor found warnings; review setup_report.yaml before full COCO runs.")
    if not notes:
        notes.append("Setup looks ready for debug optimize.")
    return notes


def _failed_checks(doctor: DoctorReport) -> list[DoctorCheck]:
    return [check for check in doctor.checks if check.level == "error" and not check.ok]


def _warning_checks(doctor: DoctorReport) -> list[DoctorCheck]:
    return [check for check in doctor.checks if check.level == "warning" and not check.ok]


def _load_local_llm_config(path: Path) -> LLMDecisionConfig | None:
    try:
        return load_llm_decision_config(path)
    except (OSError, ValueError):
        return None


def setup_result_to_text(result: SetupResult) -> str:
    """Render setup output for CLI users."""
    lines = [
        f"setup status={'ok' if result.ok else 'needs_fix'} errors={result.doctor_errors} warnings={result.doctor_warnings}",
        f"run_id={result.run_id}",
        f"env_file={result.env_file}",
        f"llm_config={result.llm_config_path}",
        f"setup_report={result.setup_report_path}",
        f"openai_key_detected={str(result.openai_key_detected).lower()}",
    ]
    for note in result.notes:
        lines.append(f"note: {note}")
    for index, check in enumerate(result.failed_checks, start=1):
        lines.append(f"error[{index}].{check.name}: {check.message}")
        if check.fix:
            lines.append(f"  fix: {check.fix}")
    for index, check in enumerate(result.warning_checks, start=1):
        lines.append(f"warning[{index}].{check.name}: {check.message}")
        if check.fix:
            lines.append(f"  fix: {check.fix}")
    lines.extend(
        [
            f"next: {result.next_command}",
            f"status: {result.status_command}",
        ]
    )
    return "\n".join(lines)


__all__ = ["SetupResult", "run_setup_wizard", "setup_result_to_text"]
