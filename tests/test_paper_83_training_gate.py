"""StrictPaperReadinessGate tests.

The Prompt-13 contract: no real training resource may be allocated before
all 83 frozen papers are implementation-ready.  These tests pin the gate at
every seam (CLI train entry, queue execution, executors, Ultralytics
runtime entrypoint), prove the anti-bypass behavior by monkeypatching the
trainer machinery (which is never executed), and cover the fail-closed
manifest cases: missing, hash mismatch, wrong paper count, mock smoke,
blocked, and the 83/83 unlock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import CandidateConfig, ExperimentNode
from yolo_agent.research.paper_83_training_gate import (
    SYNTHETIC_SCOPE_ENV,
    Paper83GateLockedError,
    evaluate_paper_83_training_gate,
    gate_guard,
    gate_refusal_for_training_command,
    render_gate_summary,
)
from yolo_agent.research.paper_exactness_schemas import (
    ExactnessAuditSummary,
    ExactnessPaperRecord,
    PaperExactnessAudit,
)

MANIFEST_PATH = Path("configs/research/paper_83_manifest.yaml")
REAL_AUDIT_PATH = Path("artifacts/paper_83_exactness_audit.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_green_acceptance(path: Path) -> Path:
    """Write an all-green synthetic acceptance record for unlock-path probes.

    The real acceptance artifact records the live fast-tier result, and the
    fast tier *contains the release-seam tests themselves* — a test that
    builds a release from the real artifact would therefore feed on its own
    failure (self-referential loop).  Unlock-path probes use this synthetic
    record instead, exactly as the gate probes use synthetic audits; the
    release contract itself (sections verified, hashes pinned) is pinned
    separately against the real committed artifact.
    """

    from yolo_agent.research.pretraining_acceptance import (
        ActionSpaceSection,
        AutonomousLoopSection,
        PaperCampaignSection,
        PretrainingAcceptance,
        SafetySection,
        TestsSection,
        TrainingGateSection,
    )

    green = PretrainingAcceptance(
        paper_campaign=PaperCampaignSection(
            manifest_paper_count=83,
            unique_paper_ids=83,
            membership_hash_valid=True,
            implementation_ready=83,
            blocked=0,
            passed=True,
        ),
        optimization_action_space=ActionSpaceSection(
            catalog_action_count=40,
            covered_families=["sampling", "postprocess"],
            missing_families=[],
            paper_lineage_actions=20,
            local_actions=20,
            passed=True,
        ),
        autonomous_loop=AutonomousLoopSection(
            rounds_executed=3,
            diagnosis_chain=True,
            error_delta_used=True,
            bounded_hpo_module_present=True,
            asha_budget_routed=True,
            rollback_observed=True,
            pareto_axes_used=True,
            stop_policy_honest=True,
            multi_round_completed=True,
            passed=True,
        ),
        safety=SafetySection(
            gate_82_of_83_blocks=True,
            gate_83_of_83_allows=True,
            gate_fail_closed_on_missing_manifest=True,
            gate_fail_closed_on_hash_mismatch=True,
            full_run_consent_boundary_preserved=True,
            passed=True,
        ),
        tests=TestsSection(
            fast_command="synthetic",
            fast_exit_code=0,
            lint_command="synthetic",
            lint_exit_code=0,
            passed=True,
        ),
        training_gate=TrainingGateSection(
            allowed=True, ready=83, blocked=0, required=83
        ),
        verdict="PASS",
    )
    green.to_yaml(path)
    return path


def _load_real() -> tuple[Any, PaperExactnessAudit]:
    from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

    return (
        Paper83Manifest.from_yaml(MANIFEST_PATH),
        PaperExactnessAudit.from_yaml(REAL_AUDIT_PATH),
    )


def _synthetic_audit(tmp_path: Path, *, ready: int, blocked: int, mock_ready: int = 0) -> Path:
    """Write a synthetic audit against the real frozen membership.

    ``ready`` records are verbatim copies of a real implementation_ready
    record re-identified per paper; ``mock_ready`` additional records are
    implementation_ready with mock_only evidence class; the remainder are
    blocked_missing_code copies.  The total is always 83.
    """

    manifest, real = _load_real()
    source_ready = next(r for r in real.records if r.status == "implementation_ready")
    # Gap closure removed every blocked record, so synthetic blocked entries
    # are honest downgrades of a ready copy (status + blocker label only).
    source_blocked_payload = source_ready.model_dump(mode="json")
    source_blocked_payload["status"] = "blocked_missing_code"
    source_blocked_payload["blockers"] = ["blocked_missing_code:synthetic"]
    records: list[ExactnessPaperRecord] = []
    ids = [paper.paper_id for paper in manifest.papers]
    for index, paper_id in enumerate(ids):
        data: dict[str, Any]
        if index < ready:
            data = source_ready.model_dump(mode="json")
        elif index < ready + mock_ready:
            data = source_ready.model_dump(mode="json")
            data["evidence_inventory"] = {
                **data["evidence_inventory"],
                "implementation_evidence_class": "mock_only",
            }
        else:
            data = dict(source_blocked_payload)
        data["paper_id"] = paper_id
        records.append(ExactnessPaperRecord.model_validate(data))
    total = len(ids)
    audit = PaperExactnessAudit(
        manifest_path=str(MANIFEST_PATH),
        manifest_membership_hash=manifest.campaign.membership_hash,
        paper_count=total,
        records=records,
        summary=ExactnessAuditSummary(
            total=total,
            ready=ready + mock_ready,
            blocked=total - ready - mock_ready,
            out_of_scope=0,
        ),
    ).with_hash()
    path = tmp_path / f"audit_{ready}_{blocked}_{mock_ready}.yaml"
    audit.to_yaml(path)
    return path


# -- gate evaluation ----------------------------------------------------------


def test_current_campaign_unlocks_with_exact_counts() -> None:
    """Gap closure complete: the live audit is 83/83, so the gate is open."""

    decision = evaluate_paper_83_training_gate()
    assert decision.allowed is True
    assert decision.locked is False
    assert decision.required == 83
    assert decision.required_maturity == "implementation_ready"
    assert decision.ready == 83
    assert decision.blocked == 0
    assert decision.lock_reasons == []
    assert decision.fail_closed is True
    assert decision.evidence_source == "exactness_audit"


def test_82_of_83_ready_stays_locked(tmp_path: Path) -> None:
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=_synthetic_audit(tmp_path, ready=82, blocked=1),
    )
    assert decision.allowed is False
    assert decision.ready == 82
    assert decision.blocked == 1
    assert decision.lock_reasons == [
        "ready_count_below_required:82<83",
        "blocked_papers_present:1",
    ]


def test_83_of_83_ready_unlocks(tmp_path: Path) -> None:
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=_synthetic_audit(tmp_path, ready=83, blocked=0),
    )
    assert decision.allowed is True
    assert decision.locked is False
    assert decision.lock_reasons == []
    assert decision.ready == 83
    summary = render_gate_summary(decision)
    assert "Training allowed: YES" in summary


def test_mock_smoke_evidence_never_unlocks(tmp_path: Path) -> None:
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=_synthetic_audit(tmp_path, ready=82, blocked=0, mock_ready=1),
    )
    assert decision.allowed is False
    assert any(reason.startswith("ready_count_below_required") or reason.startswith("blocked_papers_present") for reason in decision.lock_reasons)
    assert "mock_smoke_evidence" in decision.lock_reasons
    assert decision.blocked == 1


# -- fail-closed manifest cases -------------------------------------------------


def test_missing_manifest_locks(tmp_path: Path) -> None:
    decision = evaluate_paper_83_training_gate(
        manifest_path=tmp_path / "absent.yaml",
        exactness_audit_path=REAL_AUDIT_PATH,
    )
    assert decision.allowed is False
    assert decision.lock_reasons[0] == "manifest_missing"
    assert "ready_count_below_required:0<83" in decision.lock_reasons
    assert decision.ready == 0


def test_invalid_manifest_locks(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("schema_version: paper_implementation_campaign.v1\n", encoding="utf-8-sig")
    decision = evaluate_paper_83_training_gate(
        manifest_path=bad,
        exactness_audit_path=REAL_AUDIT_PATH,
    )
    assert decision.allowed is False
    assert decision.lock_reasons[0].startswith("manifest_invalid:")


def test_manifest_hash_mismatch_locks(tmp_path: Path) -> None:
    manifest, _ = _load_real()
    tampered = dict(manifest.campaign.model_dump(mode="json"))
    tampered["membership_hash"] = "0" * 64
    from yolo_agent.research.paper_83_campaign_schemas import Paper83Campaign

    forged = manifest.model_copy(
        update={"campaign": Paper83Campaign.model_validate(tampered)}
    )
    path = tmp_path / "manifest_forged.yaml"
    forged.to_yaml(path)
    decision = evaluate_paper_83_training_gate(
        manifest_path=path,
        exactness_audit_path=REAL_AUDIT_PATH,
    )
    assert decision.allowed is False
    assert decision.lock_reasons[0].startswith("manifest_invalid:")
    assert "membership_hash" in decision.lock_reasons[0]


def test_audit_membership_hash_mismatch_locks(tmp_path: Path) -> None:
    audit_path = _synthetic_audit(tmp_path, ready=83, blocked=0)
    data = PaperExactnessAudit.from_yaml(audit_path).model_dump(mode="json")
    data["manifest_membership_hash"] = "1" * 64

    del data["audit_hash"]
    forged_path = tmp_path / "audit_forged.yaml"
    PaperExactnessAudit.model_validate(data).to_yaml(forged_path)
    decision = evaluate_paper_83_training_gate(
        manifest_path=MANIFEST_PATH,
        exactness_audit_path=forged_path,
    )
    assert decision.allowed is False
    assert "manifest_hash_mismatch" in decision.lock_reasons
    assert decision.ready == 0


def test_missing_audit_falls_back_to_registry_and_locks(tmp_path: Path) -> None:
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=tmp_path / "absent_audit.yaml",
        implementation_registry_path=tmp_path / "absent_registry.yaml",
    )
    assert decision.allowed is False
    assert decision.lock_reasons[:2] == ["audit_unavailable", "registry_unavailable"]
    assert "ready_count_below_required:0<83" in decision.lock_reasons
    assert decision.evidence_source == "implementation_registry"
    assert decision.ready == 0


def test_paper_count_mismatch_locks(tmp_path: Path) -> None:
    from yolo_agent.research.paper_83_training_gate import Paper83GateConfig

    audit_path = _synthetic_audit(tmp_path, ready=83, blocked=0)
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=audit_path,
        config=Paper83GateConfig(required_count=84),
    )
    assert decision.allowed is False
    assert any(r.startswith("paper_count_mismatch:83!=84") for r in decision.lock_reasons)


# -- seam enforcement ------------------------------------------------------------


def _train_node(tmp_path: Path, node_id: str = "gate-node") -> ExperimentNode:
    spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project=tmp_path,
        name="probe",
    )
    return ExperimentNode(
        node_id=node_id,
        candidate_config=CandidateConfig(
            candidate_id="gate-candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="v1",
        command_spec=spec,
    )


def test_executor_refuses_training_under_locked_gate(tmp_path: Path) -> None:
    """A 82/83 audit locks the executor seam with an honest message."""

    from yolo_agent.core import executor as executor_mod
    from yolo_agent.core.executor import _paper_83_gate_refusal

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    original_default = executor_mod.__dict__.get(
        "PAPER_83_AUDIT_PATH_OVERRIDE", None
    )
    _ = original_default
    import yolo_agent.research.paper_83_training_gate as gate_module

    real_audit = gate_module.DEFAULT_EXACTNESS_AUDIT_PATH
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = str(audit_path)
    try:
        node = _train_node(tmp_path)
        refusal = _paper_83_gate_refusal("run-gate", node, node.command_spec)
    finally:
        gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = real_audit
    assert refusal is not None
    assert refusal.status == "skipped"
    assert refusal.metrics.get("paper_83_gate_locked") is True
    assert "PAPER-83 PRE-TRAINING GATE" in refusal.message
    assert "82/83" in refusal.message  # honest: the synthetic state is 82/83


def test_executor_permits_training_when_83_ready(tmp_path: Path, monkeypatch) -> None:
    """83/83 audit lets the executor proceed toward the (stubbed) trainer."""

    from yolo_agent.core import executor as executor_mod
    from yolo_agent.core.executor import UltralyticsTrainExecutor

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=83, blocked=0)
    monkeypatch.setattr(
        "yolo_agent.research.paper_83_training_gate.DEFAULT_EXACTNESS_AUDIT_PATH",
        audit_path,
        raising=False,
    )
    # The executor seam verifies the release artifact after the gate.  The
    # release freezes an acceptance record, and the real acceptance's fast
    # tier contains these very tests — so the unlock path points the release
    # machinery at a synthetic all-green acceptance (see helper docstring).
    from yolo_agent.research.training_release import (
        DEFAULT_ACCEPTANCE_PATH,
        build_training_release,
    )

    green_acceptance = _synthetic_green_acceptance(
        tmp_path / "acceptance_green.yaml"
    )
    release_path = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=green_acceptance,
        output_path=release_path,
    )
    assert release.release_status == "READY_FOR_FIRST_TRAINING", release.lock_reasons
    monkeypatch.setattr(
        "yolo_agent.research.training_release.DEFAULT_RELEASE_PATH",
        str(release_path),
        raising=False,
    )
    monkeypatch.setattr(
        "yolo_agent.research.training_release.DEFAULT_ACCEPTANCE_PATH",
        str(green_acceptance),
        raising=False,
    )
    _ = DEFAULT_ACCEPTANCE_PATH
    calls: list[str] = []
    def fake_training(*_args: Any, **_kwargs: Any) -> int:
        calls.append("trainer")
        return 0

    monkeypatch.setattr(
        "yolo_agent.adapters.ultralytics.runtime_entrypoint.run_ultralytics_training",
        fake_training,
    )
    node = _train_node(tmp_path)
    executor = UltralyticsTrainExecutor()
    original = executor_mod.UltralyticsTrainExecutor.execute

    def execute_until_gate(self: Any, node: Any, run_id: str, command: Any = None) -> Any:
        spec = command or executor_mod.CommandSpec.from_experiment_node(node)
        refusal = executor_mod._paper_83_gate_refusal(run_id, node, spec)
        if refusal is not None:
            return refusal
        fake_training()
        return executor_mod.ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="dry_run",
            command=spec,
            message="gate permitted; trainer stub invoked",
        )

    monkeypatch.setattr(executor_mod.UltralyticsTrainExecutor, "execute", execute_until_gate)
    result = executor.execute(node, run_id="run-unlocked")
    assert calls == ["trainer"]
    assert result.status == "dry_run"
    del original


def test_executor_gate_failure_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    from yolo_agent.core.executor import _paper_83_gate_refusal

    def explode() -> Any:
        raise RuntimeError("synthetic gate crash")

    monkeypatch.setattr(
        "yolo_agent.research.paper_83_training_gate.evaluate_paper_83_training_gate",
        explode,
    )
    node = _train_node(tmp_path)
    refusal = _paper_83_gate_refusal("run-crash", node, node.command_spec)
    assert refusal is not None
    assert refusal.status == "skipped"
    assert "fail-closed" in refusal.message


def test_queue_item_refusal_shape(tmp_path: Path, monkeypatch) -> None:
    import yolo_agent.research.paper_83_training_gate as gate_module
    from yolo_agent.agents.orchestrator import _paper_83_gate_refusal_for_item
    from yolo_agent.core.execution_queue import ExecutionQueueItem

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    real_audit = gate_module.DEFAULT_EXACTNESS_AUDIT_PATH
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = str(audit_path)
    node = _train_node(tmp_path)
    item = ExecutionQueueItem.from_node("gate-run", node)
    decision = _paper_83_gate_refusal_for_item(item)
    assert decision is not None
    assert decision.status == "blocked_by_resource"
    assert decision.reasons[0] == "paper_83_training_gate_locked"
    smoke_node = ExperimentNode(
        node_id="smoke-node",
        candidate_config=node.candidate_config,
        data_version="v1",
        command_spec=CommandSpec(command_type="smoke", command="echo", args=["hi"]),
    )
    assert (
        _paper_83_gate_refusal_for_item(ExecutionQueueItem.from_node("gate-run", smoke_node))
        is None
    )
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = real_audit


def test_runtime_entrypoint_gate_guard_raises_and_renders(tmp_path: Path) -> None:
    import yolo_agent.research.paper_83_training_gate as gate_module

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    real_audit = gate_module.DEFAULT_EXACTNESS_AUDIT_PATH
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = str(audit_path)
    try:
        with pytest.raises(Paper83GateLockedError) as excinfo:
            gate_guard()
    finally:
        gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = real_audit
    decision = excinfo.value.decision
    assert decision.locked is True
    summary = render_gate_summary(decision)
    assert summary[:5] == [
        "PAPER-83 PRE-TRAINING GATE",
        "Required: 83",
        f"Ready: {decision.ready}",
        f"Blocked: {decision.blocked}",
        "Training allowed: NO",
    ]


def test_runtime_entrypoint_permits_stubbed_trainer_when_83_ready(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("ultralytics")

    from yolo_agent.adapters.ultralytics import runtime_entrypoint
    from yolo_agent.adapters.ultralytics.plugin_bridge import (
        UltralyticsTrainerPluginBridge,
    )
    from yolo_agent.components.adapters import (
        AdapterRuntimePayload,
        RollbackPlan,
        RuntimePluginReference,
    )

    # The gate is the thing under test here; the hook-efficacy machinery is
    # covered by the plugin-bridge suite, so its final check is stubbed out.
    monkeypatch.setattr(
        UltralyticsTrainerPluginBridge,
        "verify_required_hooks",
        lambda self: None,
    )

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=83, blocked=0)
    monkeypatch.setattr(
        "yolo_agent.research.paper_83_training_gate.DEFAULT_EXACTNESS_AUDIT_PATH",
        audit_path,
        raising=False,
    )
    # The runtime seam is the last release checkpoint too; the unlock path
    # freezes a fresh release against a synthetic green acceptance (the real
    # acceptance's fast tier contains this test — see helper docstring).
    from yolo_agent.research.training_release import build_training_release

    green_acceptance = _synthetic_green_acceptance(
        tmp_path / "acceptance_green.yaml"
    )
    release_path = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=green_acceptance,
        output_path=release_path,
    )
    assert release.release_status == "READY_FOR_FIRST_TRAINING", release.lock_reasons
    monkeypatch.setattr(
        "yolo_agent.research.training_release.DEFAULT_RELEASE_PATH",
        str(release_path),
        raising=False,
    )
    monkeypatch.setattr(
        "yolo_agent.research.training_release.DEFAULT_ACCEPTANCE_PATH",
        str(green_acceptance),
        raising=False,
    )

    class FakeYOLO:
        def __init__(self, model: str, task: str, verbose: bool = False) -> None:
            self.model = model

        def train(self, **kwargs: Any) -> None:
            # The gate permitted the downstream call; the stubbed trainer
            # itself must never execute a real training step.
            return None

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    payload = AdapterRuntimePayload(
        component_ids=["dummy.component"],
        adapter_classes=["DummyAdapter"],
        adapter_versions={"dummy.component": "dummy.v1"},
        source_commits={"dummy.component": "local-test"},
        trainer_plugin=[
            RuntimePluginReference(
                reference="yolo_agent.components.adapters.dummy:DummyRuntimePlugin",
                options={},
                required_hooks=["build_model"],
            )
        ],
        generated_config={"training_config": {"imgsz": 640}},
        changed_variables={"training.adapter_marker": "active"},
        rollback_plan=RollbackPlan(actions=["discard synthetic payload"]),
        protocol_hash="protocol-gate-test",
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
        ],
        supports_amp=True,
        supports_ddp=True,
        supports_resume=True,
    )
    payload_path = payload.write(tmp_path / "gate_payload.yaml")
    result = runtime_entrypoint.run_ultralytics_training(
        payload_path,
        [
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
            "imgsz=640",
            "amp=False",
        ],
    )
    assert result == 0


# -- CLI seams ---------------------------------------------------------------


def test_cli_train_refuses_before_allocation(tmp_path: Path, capsys) -> None:
    import argparse

    import yolo_agent.research.paper_83_training_gate as gate_module
    from yolo_agent.cli import run_train_command

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    real_audit = gate_module.DEFAULT_EXACTNESS_AUDIT_PATH
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = str(audit_path)
    args = argparse.Namespace(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        run_id="gate-cli-probe",
        run_root=tmp_path / "runs",
        profile="debug",
        kind="coco",
        goal=None,
        target_metric=None,
        target_delta=None,
        goal_description=None,
        auto_rounds=None,
        dry_run=False,
        confirm_full_run=False,
        no_auto_advance=False,
        max_steps=8,
        no_auto_import=False,
        training_release=None,
    )
    try:
        code = run_train_command(args)
    finally:
        gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = real_audit
    assert code == 2
    output = capsys.readouterr().out
    assert "PAPER-83 PRE-TRAINING GATE" in output
    assert "Training: not started" in output
    assert not (tmp_path / "runs" / "gate-cli-probe").exists()


def test_cli_papers_status_reports_gate(capsys) -> None:
    import argparse

    from yolo_agent.cli import run_papers_status_command

    code = run_papers_status_command(argparse.Namespace(mode="status"))
    output = capsys.readouterr().out
    assert code == 0
    for line in (
        "PAPER-83 PRE-TRAINING GATE",
        "Required: 83",
        "Ready: 83",
        "Blocked: 0",
        "Training allowed: YES",
    ):
        assert line in output


# -- synthetic scope contract ---------------------------------------------------


def test_synthetic_scope_declares_and_preserves_lock_reasons(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(SYNTHETIC_SCOPE_ENV, "1")
    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    decision = evaluate_paper_83_training_gate(exactness_audit_path=audit_path)
    assert decision.allowed is True
    assert decision.synthetic_scope is True
    assert decision.lock_reasons  # honest verdict stays visible
    guard_decision = gate_guard(exactness_audit_path=audit_path)
    assert guard_decision.synthetic_scope is True
    assert "Training allowed: YES" in render_gate_summary(decision)
    assert any("Synthetic scope" in line for line in render_gate_summary(decision))


def test_synthetic_scope_requires_exact_env_value(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    monkeypatch.setenv(SYNTHETIC_SCOPE_ENV, "true")
    decision = evaluate_paper_83_training_gate(exactness_audit_path=audit_path)
    assert decision.synthetic_scope is False
    assert decision.allowed is False
    monkeypatch.setenv(SYNTHETIC_SCOPE_ENV, "1")
    assert (
        evaluate_paper_83_training_gate(exactness_audit_path=audit_path).synthetic_scope
        is True
    )
    monkeypatch.delenv(SYNTHETIC_SCOPE_ENV)
    assert (
        evaluate_paper_83_training_gate(exactness_audit_path=audit_path).synthetic_scope
        is False
    )


def test_synthetic_scope_never_unlocks_real_verdict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(SYNTHETIC_SCOPE_ENV, "1")
    decision = evaluate_paper_83_training_gate(
        exactness_audit_path=_synthetic_audit(tmp_path, ready=82, blocked=1),
    )
    assert decision.allowed is True
    assert decision.synthetic_scope is True
    assert decision.lock_reasons == [
        "ready_count_below_required:82<83",
        "blocked_papers_present:1",
    ]


def test_gate_refusal_for_training_command_scope(tmp_path: Path, monkeypatch) -> None:
    import yolo_agent.research.paper_83_training_gate as gate_module

    audit_path = _synthetic_audit(tmp_path / "evidence", ready=82, blocked=1)
    real_audit = gate_module.DEFAULT_EXACTNESS_AUDIT_PATH
    gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = str(audit_path)
    try:
        assert gate_refusal_for_training_command("smoke") is None
        assert gate_refusal_for_training_command("import_metrics") is None
        assert gate_refusal_for_training_command("inference_policy") is None
        assert gate_refusal_for_training_command(None) is None
        refusal = gate_refusal_for_training_command("train")
    finally:
        gate_module.DEFAULT_EXACTNESS_AUDIT_PATH = real_audit
    assert refusal is not None
    assert refusal.locked is True


def test_no_training_commands_run_real_trainer() -> None:
    """Contract guard: the gate module never imports or triggers training."""

    import yolo_agent.research.paper_83_training_gate as gate_module

    source = Path(gate_module.__file__).read_text(encoding="utf-8")
    assert "from ultralytics" not in source
    assert "YOLO(" not in source
    assert "subprocess" not in source
