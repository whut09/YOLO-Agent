"""CPU-only tests for the final paper training authorization gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.certification.paper_readiness import (
    PaperReadinessRecord,
    PaperReadinessReport,
    ReadinessCheck,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.paper_training_readiness import (
    PaperTrainingReadinessReport,
    _required_asset_blocker,
    build_paper_training_readiness,
)
from yolo_agent.cli import main
from yolo_agent.research.paper_asset_schemas import (
    PaperAssetRecord,
    PaperAssetRegistry,
    _aggregate_hash,
    _sha256,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


def _node(tmp_path: Path, *, baseline: bool = False) -> ExperimentNode:
    candidate_id = "matched-control" if baseline else "paper-candidate"
    metadata: dict[str, object] = {
        "run_protocol_hash": "d" * 64,
        "baseline_protocol_hash": "d" * 64,
        "dataset_manifest_hash": "e" * 64,
        "fidelity": "pilot_3",
        "split": "val2017",
    }
    if not baseline:
        metadata.update(
            {
                "adapter_runtime_entrypoint": "mock.paper.runtime",
                "paper_readiness_state": "asha_eligible",
                "paper_readiness_blockers": "[]",
            }
        )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=2,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node-{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=[] if baseline else ["loss.quality.correlation"],
            action_domain="paper",
            action_id="quality-correlation",
            target_error_facts=[] if baseline else [{"fact_type": "localization_error"}],
        ),
        data_version="dataset",
        command=command.display(),
        command_spec=command,
        changed_variables={} if baseline else {"loss.correlation.weight": 0.1},
    )


def _inventory(node: ExperimentNode) -> PaperExecutionInventory:
    fingerprint = execution_fingerprint(node)
    record = PaperExecutionSpec(
        paper_id="paper:001",
        profile_id="profile:001",
        title="Fixture paper",
        source_locations=["fixture"],
        canonical_component_ids=["loss.quality.correlation"],
        paper_specific_mechanism_ids=["quality_correlation"],
        required_evidence=["localization_error"],
        recipe_ids=["quality-correlation"],
        execution_fingerprint=fingerprint,
        current_disposition="runtime_ready",
        disposition_reason="fixture is ready",
        readiness_state="asha_eligible",
        runtime_ready_adapters=["loss.quality.correlation"],
    )
    return PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[record],
    ).with_hash()


def _requirement(inventory: PaperExecutionInventory) -> PaperExecutionRequirementsMatrix:
    return PaperExecutionRequirementsMatrix(
        source_inventory_path="inventory.yaml",
        source_inventory_hash=inventory.inventory_hash,
        compatible_paper_count=1,
        requirements=[
            PaperExecutionRequirement(
                paper_id="paper:001",
                paper_specific_mechanism="quality_correlation",
                paper_specific_mechanism_ids=["quality_correlation"],
                execution_route="training",
                required_adapter="loss.quality.correlation",
                required_changed_variables=["loss.correlation.weight"],
                required_runtime_payload={"loss_mode": "correlation"},
                required_evidence=["localization_error"],
                required_dataset_protocol={"imgsz": 640},
                compatible_with_yolo26=True,
                training_candidate_allowed=True,
                recovery_action="none",
                recipe_ids=["quality-correlation"],
                current_disposition="runtime_ready",
                protocol_hash="d" * 64,
                execution_fingerprint=inventory.records[0].execution_fingerprint,
            )
        ],
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _checks(*, passed: bool = True, blocker: str | None = None) -> ReadinessCheck:
    return ReadinessCheck(passed=passed, blocker=blocker)


def _readiness(inventory: PaperExecutionInventory, *, blocked: str | None = None) -> PaperReadinessReport:
    passed = blocked is None
    record = PaperReadinessRecord(
        paper_id="paper:001",
        mechanism_id="quality_correlation",
        recipe_id="quality-correlation",
        adapter_hash="b" * 64,
        protocol_hash="d" * 64,
        dataset_manifest_hash="e" * 64,
        runtime_payload_hash="f" * 64,
        cache_key="1" * 64,
        cpu_contract_result=_checks(passed=passed, blocker=blocked),
        shape_result=_checks(passed=passed, blocker=blocked),
        forward_result=_checks(passed=passed, blocker=blocked),
        backward_result=_checks(passed=passed, blocker=blocked),
        payload_result=_checks(passed=passed, blocker=blocked),
        dataset_evidence_result=_checks(passed=passed, blocker=blocked),
        teacher_evidence_result=_checks(),
        graph_evidence_result=_checks(),
        matched_control_readiness=_checks(passed=passed, blocker=blocked),
        asha_eligibility=passed,
        readiness_state="asha_eligible" if passed else "blocked",
        pre_registered=True,
        cpu_checks_passed=passed,
        runtime_checks_passed=passed,
        final_disposition="runtime_ready" if passed else "blocked_runtime",
        exact_blocker=blocked,
        source_inventory_hash=inventory.inventory_hash,
    )
    return PaperReadinessReport(
        status="passed" if passed else "partial",
        inventory_hash=inventory.inventory_hash,
        paper_count=1,
        registry_hash="registry",
        model="yolo26n.pt",
        data="coco.yaml",
        records=[record],
        disposition_counts=(
            {"runtime_ready": 1}
            if passed
            else {"blocked_runtime": 1}
        ),
        readiness_state_counts=(
            {"asha_eligible": 1}
            if passed
            else {"blocked": 1}
        ),
    ).with_hash()


def _assets(
    tmp_path: Path,
    inventory: PaperExecutionInventory,
    requirements_path: Path,
    *,
    available: bool = True,
) -> PaperAssetRegistry:
    source = tmp_path / "source.manifest"
    baseline = tmp_path / "baseline.yaml"
    source.write_text("train: train", encoding="utf-8")
    baseline.write_text("protocol: d", encoding="utf-8")
    fields = {
        "source_dataset_manifest": str(source.resolve()),
        "matched_baseline_artifact": str(baseline.resolve()),
    }
    hashes = {key: _sha256(Path(value)) for key, value in fields.items()}
    asset_hash = _aggregate_hash(hashes)
    record = PaperAssetRecord(
        paper_id="paper:001",
        mechanism_id="quality_correlation",
        **fields,
        asset_sha256=asset_hash,
        protocol_hash="d" * 64,
        availability="available" if available else "unavailable",
        exact_blocker="" if available else "asset_missing",
        recovery_action="none" if available else "provide_assets",
        current_disposition="runtime_ready" if available else "blocked_runtime",
        asset_hashes=hashes,
        validated_assets=list(fields),
    )
    return PaperAssetRegistry(
        source_inventory_path=str((tmp_path / "inventory.yaml").resolve()),
        source_inventory_hash=inventory.inventory_hash,
        source_requirements_path=str(requirements_path.resolve()),
        source_requirements_hash=_file_hash(requirements_path),
        compatible_paper_count=1,
        records=[record],
    ).with_hash()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path, *, with_trial: bool = True):  # type: ignore[no-untyped-def]
    node = _node(tmp_path)
    baseline = _node(tmp_path, baseline=True)
    inventory = _inventory(node)
    inventory_path = tmp_path / "inventory.yaml"
    inventory.to_yaml(inventory_path, sort_keys=False)
    requirements = _requirement(inventory)
    requirements.source_inventory_path = str(inventory_path.resolve())
    requirements_path = tmp_path / "requirements.yaml"
    requirements.to_yaml(requirements_path, sort_keys=False)
    assets = _assets(tmp_path, inventory, requirements_path)
    assets_path = tmp_path / "assets.yaml"
    assets.to_yaml(assets_path, sort_keys=False)
    readiness = _readiness(inventory)
    readiness_path = tmp_path / "readiness.yaml"
    readiness.to_yaml(readiness_path, sort_keys=False)
    scheduler = ASHAScheduler.create("training-readiness")
    if with_trial:
        scheduler.register_trial(
            trial_id="trial:001",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="training-readiness",
            source_node=node,
            baseline_control_node=baseline,
            target_error_facts=node.candidate_config.target_error_facts,
            paper_ids=["paper:001"],
        )
    asha_path = tmp_path / "asha.yaml"
    scheduler.study.to_yaml(asha_path, sort_keys=False)
    return inventory_path, requirements_path, assets_path, readiness_path, asha_path


def test_final_gate_authorizes_only_a_registered_paired_candidate(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    report = build_paper_training_readiness(
        inventory_path=inputs[0],
        requirements_path=inputs[1],
        assets_path=inputs[2],
        readiness_path=inputs[3],
        asha_path=inputs[4],
        output_path=tmp_path / "training-readiness.yaml",
        expected_paper_count=1,
    )
    assert report.training_allowed is True
    assert report.asha_eligible_count == 1
    assert report.asha_registered_count == 1
    assert report.inventory_count == 1
    assert report.implementation_complete_count == 1
    assert report.cpu_ready_count == 1
    assert report.runtime_ready_count == 1
    assert report.matched_control_ready_count == 1
    assert report.pre_registered_count == 1
    assert report.inference_only_count == 0
    assert report.actual_trained_count == 0
    assert report.exact_reproduction_count == 0
    assert report.training_started is False
    assert report.gpu_probe == "not_run"


def test_eligible_paper_without_asha_trial_is_blocked(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path, with_trial=False)
    report = build_paper_training_readiness(
        inventory_path=inputs[0],
        requirements_path=inputs[1],
        assets_path=inputs[2],
        readiness_path=inputs[3],
        asha_path=inputs[4],
        output_path=tmp_path / "training-readiness.yaml",
        expected_paper_count=1,
    )
    assert report.training_allowed is False
    assert report.asha_eligible_count == 0
    assert report.records[0].blocker == "asha_eligible_paper_missing_runnable_trial"


def test_stale_requirements_are_rejected_before_authorization(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs[1].write_text(inputs[1].read_text(encoding="utf-8") + "\n# stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="asset registry source requirements hash is stale"):
        build_paper_training_readiness(
            inventory_path=inputs[0],
            requirements_path=inputs[1],
            assets_path=inputs[2],
            readiness_path=inputs[3],
            asha_path=inputs[4],
            output_path=tmp_path / "training-readiness.yaml",
            expected_paper_count=1,
        )


def test_report_schema_forbids_claiming_gpu_training() -> None:
    with pytest.raises(ValueError, match="training_started"):
        PaperTrainingReadinessReport(
            status="ready",
            training_allowed=True,
            training_started=True,
            inventory_path="inventory",
            requirements_path="requirements",
            assets_path="assets",
            readiness_path="readiness",
            asha_path="asha",
            inventory_hash="a" * 64,
            requirements_file_hash="b" * 64,
            asset_registry_hash="c" * 64,
            readiness_report_hash="d" * 64,
            paper_count=0,
        )


def test_required_real_assets_block_final_authorization(tmp_path: Path) -> None:
    requirement = SimpleNamespace(
        required_teacher_assets=["teacher_checkpoint"],
        required_domain_assets=[],
        required_manifest_assets=[],
        paper_specific_mechanism_ids=["logits_distillation"],
    )
    item = SimpleNamespace(paper_specific_mechanism_ids=["logits_distillation"])
    asset = SimpleNamespace(
        teacher_checkpoint=None,
        teacher_sha256=None,
        source_dataset_manifest=None,
        target_dataset_manifest=None,
        hard_negative_manifest=None,
    )
    assert _required_asset_blocker(item=item, requirement=requirement, asset=asset) == (
        "teacher_checkpoint_missing"
    )


def test_missing_teacher_is_reported_before_generic_readiness_blocker(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    inventory = PaperExecutionInventory.from_yaml(inputs[0])
    requirements = PaperExecutionRequirementsMatrix.from_yaml(inputs[1])
    requirement = requirements.requirements[0].model_copy(
        update={
            "paper_specific_mechanism": "logits_distillation",
            "paper_specific_mechanism_ids": ["logits_distillation"],
            "required_teacher_assets": ["teacher_checkpoint"],
        }
    )
    requirements.requirements = [requirement]
    requirements.to_yaml(inputs[1], sort_keys=False)
    assets = _assets(tmp_path, inventory, inputs[1])
    assets.to_yaml(inputs[2], sort_keys=False)
    report = build_paper_training_readiness(
        inventory_path=inputs[0],
        requirements_path=inputs[1],
        assets_path=inputs[2],
        readiness_path=inputs[3],
        asha_path=inputs[4],
        output_path=tmp_path / "training-readiness.yaml",
        expected_paper_count=1,
    )
    assert report.asha_eligible_count == 0
    assert report.records[0].blocker == "teacher_checkpoint_missing"


def test_domain_and_replay_assets_are_checked_for_real_split_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    target = tmp_path / "target.yaml"
    replay = tmp_path / "replay.yaml"
    source.write_text("source", encoding="utf-8")
    target.write_text("target", encoding="utf-8")
    replay.write_text("source_split: val\nrecords: []\n", encoding="utf-8")
    requirement = SimpleNamespace(
        required_teacher_assets=[],
        required_domain_assets=["source", "target"],
        required_manifest_assets=[],
        paper_specific_mechanism_ids=["domain_adaptation.feature_alignment"],
    )
    item = SimpleNamespace(
        paper_specific_mechanism_ids=["domain_adaptation.feature_alignment"]
    )
    asset = SimpleNamespace(
        teacher_checkpoint=None,
        teacher_sha256=None,
        source_dataset_manifest=str(source),
        target_dataset_manifest=str(source),
        hard_negative_manifest=None,
    )
    assert _required_asset_blocker(item=item, requirement=requirement, asset=asset) == (
        "domain_source_target_must_differ"
    )

    replay_requirement = SimpleNamespace(
        required_teacher_assets=[],
        required_domain_assets=[],
        required_manifest_assets=["hard_negative_manifest"],
        paper_specific_mechanism_ids=["sampling.hard_negative_replay"],
    )
    replay_item = SimpleNamespace(
        paper_specific_mechanism_ids=["sampling.hard_negative_replay"]
    )
    replay_asset = SimpleNamespace(
        teacher_checkpoint=None,
        teacher_sha256=None,
        source_dataset_manifest=None,
        target_dataset_manifest=None,
        hard_negative_manifest=str(replay),
    )
    assert _required_asset_blocker(
        item=replay_item,
        requirement=replay_requirement,
        asset=replay_asset,
    ) == "hard_negative_manifest_not_train_split"


def test_terminal_mock_trial_is_not_actual_training_evidence(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    study = ASHAScheduler.create("training-readiness")
    node = _node(tmp_path)
    baseline = _node(tmp_path, baseline=True)
    study.register_trial(
        trial_id="trial:001",
        candidate_id=node.candidate_config.candidate_id,
        source_run_id="mock-backend",
        source_node=node,
        baseline_control_node=baseline,
        target_error_facts=node.candidate_config.target_error_facts,
        paper_ids=["paper:001"],
    )
    trial = study.study.trials[0]
    trial.status = "confirmed"
    trial.observations.append(
        ASHAObservation(
            stage_id="pilot_3",
            node_id=node.node_id,
            seed=42,
        )
    )
    study.study.to_yaml(inputs[4], sort_keys=False)
    report = build_paper_training_readiness(
        inventory_path=inputs[0],
        requirements_path=inputs[1],
        assets_path=inputs[2],
        readiness_path=inputs[3],
        asha_path=inputs[4],
        output_path=tmp_path / "training-readiness.yaml",
        expected_paper_count=1,
    )
    assert report.actual_trained_count == 0


def test_cli_reports_blocked_gate_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _write_inputs(tmp_path, with_trial=False)
    expected = build_paper_training_readiness(
        inventory_path=inputs[0],
        requirements_path=inputs[1],
        assets_path=inputs[2],
        readiness_path=inputs[3],
        asha_path=inputs[4],
        output_path=tmp_path / "expected.yaml",
        expected_paper_count=1,
    )
    monkeypatch.setattr(
        "yolo_agent.cli.run_paper_training_readiness",
        lambda **kwargs: expected,
    )
    assert main(
        [
            "research",
            "paper-training-readiness",
            "--inventory",
            str(inputs[0]),
            "--requirements",
            str(inputs[1]),
            "--assets",
            str(inputs[2]),
            "--readiness",
            str(inputs[3]),
            "--asha",
            str(inputs[4]),
            "--output",
            str(tmp_path / "cli.yaml"),
            "--expected-compatible-count",
            "1",
        ]
    ) == 0
    output = capsys.readouterr().out.strip()
    assert output == "当前没有可训练论文候选，代码或真实资产仍需补齐"
