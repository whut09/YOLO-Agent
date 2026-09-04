"""CPU tests for the current-data paper training cohort."""

from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.certification.paper_readiness import (
    PaperReadinessRecord,
    PaperReadinessReport,
    ReadinessCheck,
)
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
from yolo_agent.research.paper_training_cohort import build_paper_training_cohort
from yolo_agent.cli import main


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _check(passed: bool, blocker: str | None = None) -> ReadinessCheck:
    return ReadinessCheck(passed=passed, blocker=blocker)


def _bundle(
    tmp_path: Path, *, baseline_result_missing: bool = False
) -> tuple[Path, Path, Path, Path, Path]:
    source = tmp_path / "coco-train.manifest"
    source.write_text("train: true\n", encoding="utf-8")
    inventory_path = tmp_path / "inventory.yaml"
    requirements_path = tmp_path / "requirements.yaml"
    assets_path = tmp_path / "assets.yaml"
    readiness_path = tmp_path / "readiness.yaml"
    asha_path = tmp_path / "asha.yaml"

    ready_fingerprint = "a" * 64
    specs = [
        PaperExecutionSpec(
            paper_id="paper:001",
            profile_id="profile:001",
            title="Quality A",
            source_locations=["paper"],
            canonical_component_ids=["loss.quality.correlation"],
            paper_specific_mechanism_ids=["quality_correlation"],
            recipe_ids=["quality-correlation"],
            execution_fingerprint=ready_fingerprint,
            current_disposition="runtime_ready",
            disposition_reason="ready",
            readiness_state="asha_eligible",
            runtime_ready_adapters=["loss.quality.correlation"],
        ),
        PaperExecutionSpec(
            paper_id="paper:002",
            profile_id="profile:002",
            title="Quality B",
            source_locations=["paper"],
            canonical_component_ids=["loss.quality.correlation"],
            paper_specific_mechanism_ids=["quality_correlation"],
            recipe_ids=["quality-correlation"],
            execution_fingerprint=ready_fingerprint,
            current_disposition="runtime_ready",
            disposition_reason="same executable identity",
            readiness_state="asha_eligible",
            runtime_ready_adapters=["loss.quality.correlation"],
        ),
        PaperExecutionSpec(
            paper_id="paper:003",
            profile_id="profile:003",
            title="Domain paper",
            source_locations=["paper"],
            canonical_component_ids=["domain_adaptation.feature_alignment"],
            paper_specific_mechanism_ids=["domain_adaptation.feature_alignment"],
            recipe_ids=["domain-feature"],
            execution_fingerprint="b" * 64,
            current_disposition="evidence_recovery",
            disposition_reason="target domain is required",
            readiness_state="blocked",
            readiness_blocker="domain_source_target_missing",
        ),
        PaperExecutionSpec(
            paper_id="paper:004",
            profile_id="profile:004",
            title="Inference paper",
            source_locations=["paper"],
            canonical_component_ids=["inference.sahi_slicing"],
            paper_specific_mechanism_ids=["inference.sahi_slicing"],
            recipe_ids=["sahi"],
            execution_fingerprint="c" * 64,
            current_disposition="incompatible",
            disposition_reason="inference only",
            readiness_state="incompatible",
            readiness_blocker="inference_only_not_training_candidate",
        ),
        PaperExecutionSpec(
            paper_id="paper:005",
            profile_id="profile:005",
            title="Replay paper",
            source_locations=["paper"],
            canonical_component_ids=["sampling.hard_negative_replay"],
            paper_specific_mechanism_ids=["sampling.hard_negative_replay"],
            recipe_ids=["hard-negative"],
            execution_fingerprint="d" * 64,
            current_disposition="evidence_recovery",
            disposition_reason="train replay evidence is missing",
            readiness_state="blocked",
            readiness_blocker="hard_negative_train_manifest_missing",
        ),
        PaperExecutionSpec(
            paper_id="paper:006",
            profile_id="profile:006",
            title="Unresolved paper",
            source_locations=["paper"],
            paper_specific_mechanism_ids=["paper.unresolved.006"],
            recipe_ids=[],
            execution_fingerprint="e" * 64,
            current_disposition="implementation_request",
            disposition_reason="adapter identity is unresolved",
            readiness_state="blocked",
            readiness_blocker="paper_adapter_unresolved",
        ),
    ]
    inventory = PaperExecutionInventory(
        source_method_coverage_hash=_hash("coverage"),
        all_paper_count=6,
        compatible_paper_count=6,
        exact_reproduction_candidates=0,
        records=specs,
    ).with_hash()
    inventory.to_yaml(inventory_path, sort_keys=False)

    def requirement(spec: PaperExecutionSpec) -> PaperExecutionRequirement:
        mechanism = spec.paper_specific_mechanism_ids[0]
        if spec.paper_id in {"paper:001", "paper:002"}:
            return PaperExecutionRequirement(
                paper_id=spec.paper_id,
                paper_specific_mechanism=mechanism,
                paper_specific_mechanism_ids=[mechanism],
                execution_route="training",
                required_adapter="loss.quality.correlation",
                required_changed_variables=["loss.correlation.weight"],
                required_runtime_payload={"loss_mode": "correlation"},
                required_evidence=["localization_error"],
                required_dataset_protocol={"imgsz": 640},
                compatible_with_yolo26=True,
                training_candidate_allowed=True,
                recovery_action="none",
                recipe_ids=spec.recipe_ids,
                current_disposition="runtime_ready",
                protocol_hash="f" * 64,
                execution_fingerprint=spec.execution_fingerprint,
            )
        route = "inference" if spec.paper_id == "paper:004" else (
            "implementation_request" if spec.paper_id == "paper:006" else (
                "evidence_recovery" if spec.paper_id in {"paper:003", "paper:005"}
                else "blocked_runtime"
            )
        )
        domain = spec.paper_id == "paper:003"
        replay = spec.paper_id == "paper:005"
        return PaperExecutionRequirement(
            paper_id=spec.paper_id,
            paper_specific_mechanism=mechanism,
            paper_specific_mechanism_ids=[mechanism],
            execution_route=route,
            required_adapter=None if spec.paper_id == "paper:006" else mechanism,
            required_changed_variables=[] if spec.paper_id == "paper:006" else ["enabled"],
            required_runtime_payload={} if spec.paper_id == "paper:006" else {"mode": "paper"},
            required_evidence=[],
            required_dataset_protocol={"imgsz": 640},
            required_domain_assets=["target"] if domain else [],
            required_manifest_assets=["train_replay"] if replay else [],
            compatible_with_yolo26=not spec.paper_id == "paper:004",
            training_candidate_allowed=False,
            exact_blocker=(
                "inference_only_not_training_candidate"
                if spec.paper_id == "paper:004"
                else spec.readiness_blocker or "paper_adapter_unresolved"
            ),
            recovery_action="recover",
            recipe_ids=spec.recipe_ids,
            current_disposition=(
                "incompatible" if spec.paper_id == "paper:004" else spec.current_disposition
            ),
            protocol_hash="f" * 64,
            execution_fingerprint=spec.execution_fingerprint,
        )

    requirements = PaperExecutionRequirementsMatrix(
        source_inventory_path=str(inventory_path.resolve()),
        source_inventory_hash=inventory.inventory_hash,
        compatible_paper_count=6,
        requirements=[requirement(spec) for spec in specs],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    requirements.to_yaml(requirements_path, sort_keys=False)

    def asset(spec: PaperExecutionSpec) -> PaperAssetRecord:
        available = spec.paper_id in {"paper:001", "paper:002"}
        fields = {"source_dataset_manifest": str(source.resolve())} if available else {}
        hashes = {key: _sha256(Path(value)) for key, value in fields.items()}
        return PaperAssetRecord(
            paper_id=spec.paper_id,
            mechanism_id=spec.paper_specific_mechanism_ids[0],
            **fields,
            asset_sha256=_aggregate_hash(hashes) if available else None,
            protocol_hash="f" * 64,
            availability="available" if available else "unavailable",
            exact_blocker="" if available else "assets_missing",
            recovery_action="recover" if not available else "none",
            current_disposition="runtime_ready" if available else "blocked_runtime",
            asset_hashes=hashes,
            validated_assets=list(fields),
        )

    assets = PaperAssetRegistry(
        source_inventory_path=str(inventory_path.resolve()),
        source_inventory_hash=inventory.inventory_hash,
        source_requirements_path=str(requirements_path.resolve()),
        source_requirements_hash=_sha256(requirements_path),
        compatible_paper_count=6,
        records=[asset(spec) for spec in specs],
    ).with_hash()
    assets.to_yaml(assets_path, sort_keys=False)

    def readiness(spec: PaperExecutionSpec) -> PaperReadinessRecord:
        ready = spec.paper_id in {"paper:001", "paper:002"}
        baseline_only_failure = baseline_result_missing and ready
        effective_ready = ready and not baseline_only_failure
        blocker = (
            "matched_baseline_artifact_missing"
            if baseline_only_failure
            else None if ready else spec.readiness_blocker or "blocked"
        )

        def check(passed: bool = ready) -> ReadinessCheck:
            return _check(passed, None if passed else blocker)

        inference = spec.paper_id == "paper:004"
        return PaperReadinessRecord(
            paper_id=spec.paper_id,
            mechanism_id=spec.paper_specific_mechanism_ids[0],
            recipe_id=spec.recipe_ids[0] if spec.recipe_ids else None,
            adapter_hash="1" * 64,
            protocol_hash="f" * 64,
            dataset_manifest_hash=_hash("dataset"),
            runtime_payload_hash="2" * 64,
            cache_key="3" * 64,
            cpu_contract_result=check(),
            shape_result=check(),
            forward_result=check(),
            backward_result=check(),
            payload_result=check(),
            dataset_evidence_result=check(),
            teacher_evidence_result=_check(True),
            graph_evidence_result=check(),
            matched_control_readiness=(
                _check(True) if baseline_only_failure else check()
            ),
            matched_control_plan_readiness=(
                _check(True) if baseline_only_failure else check()
            ),
            asha_eligibility=effective_ready,
            readiness_state="asha_eligible" if effective_ready else (
                "incompatible" if inference else "blocked"
            ),
            pre_registered=effective_ready,
            inference_only=inference,
            final_disposition="runtime_ready" if effective_ready else (
                "incompatible" if inference else spec.current_disposition
            ),
            exact_blocker=blocker,
            source_inventory_hash=inventory.inventory_hash,
        )

    readiness_records = [readiness(spec) for spec in specs]
    readiness_report = PaperReadinessReport(
        status="partial",
        inventory_hash=inventory.inventory_hash,
        paper_count=6,
        registry_hash="registry",
        model="yolo26n.pt",
        data="coco.yaml",
        records=readiness_records,
        disposition_counts={
            disposition: sum(
                item.final_disposition == disposition for item in readiness_records
            )
            for disposition in {
                item.final_disposition for item in readiness_records
            }
        },
        readiness_state_counts={
            state: sum(item.readiness_state == state for item in readiness_records)
            for state in {item.readiness_state for item in readiness_records}
        },
    ).with_hash()
    readiness_report.to_yaml(readiness_path, sort_keys=False)

    ASHAScheduler.create("cohort-fixture").study.to_yaml(asha_path, sort_keys=False)
    return inventory_path, requirements_path, assets_path, readiness_path, asha_path


def test_cohort_keeps_blocked_domain_papers_from_blocking_coco_candidates(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)
    cohort = build_paper_training_cohort(
        inventory_path=paths[0],
        requirements_path=paths[1],
        assets_path=paths[2],
        readiness_path=paths[3],
        asha_path=paths[4],
        output_path=tmp_path / "cohort.yaml",
        expected_paper_count=6,
    )
    assert cohort.training_allowed is True
    assert cohort.executable_fingerprint_count == 1
    assert cohort.category_counts["coco_supervised_ready"] == 2
    assert cohort.category_counts["requires_external_domain"] == 1
    assert cohort.category_counts["inference_only"] == 1
    assert cohort.category_counts["evidence_bootstrap_ready"] == 1
    assert cohort.category_counts["implementation_blocked"] == 1
    assert {item.paper_id for item in cohort.records} == {
        f"paper:{index:03d}" for index in range(1, 7)
    }


def test_cohort_preserves_provenance_for_shared_execution_identity(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    cohort = build_paper_training_cohort(
        inventory_path=paths[0],
        requirements_path=paths[1],
        assets_path=paths[2],
        readiness_path=paths[3],
        asha_path=paths[4],
        output_path=tmp_path / "cohort.yaml",
        expected_paper_count=6,
    )
    ready = [item for item in cohort.records if item.asha_eligible]
    assert {item.execution_fingerprint for item in ready} == {"a" * 64}
    assert all(item.paper_ids == ["paper:001", "paper:002"] for item in ready)
    assert cohort.cohort_hash


def test_adapter_failure_is_isolated_and_output_round_trips(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    output = tmp_path / "cohort.yaml"
    cohort = build_paper_training_cohort(
        inventory_path=paths[0],
        requirements_path=paths[1],
        assets_path=paths[2],
        readiness_path=paths[3],
        asha_path=paths[4],
        output_path=output,
        expected_paper_count=6,
    )
    restored = type(cohort).from_yaml(output)
    assert restored.paper_count == 6
    assert restored.records[-1].category == "implementation_blocked"
    assert restored.records[0].training_candidate is True


def test_missing_baseline_result_does_not_block_first_cohort_schedule(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path, baseline_result_missing=True)
    cohort = build_paper_training_cohort(
        inventory_path=paths[0],
        requirements_path=paths[1],
        assets_path=paths[2],
        readiness_path=paths[3],
        asha_path=paths[4],
        output_path=tmp_path / "cohort.yaml",
        expected_paper_count=6,
    )
    assert cohort.training_allowed is True
    assert cohort.executable_fingerprint_count == 1


def test_cli_builds_cohort_without_starting_training(tmp_path: Path, capsys) -> None:
    paths = _bundle(tmp_path)
    output = tmp_path / "cli-cohort.yaml"
    assert main(
        [
            "research",
            "paper-training-cohort",
            "--inventory",
            str(paths[0]),
            "--requirements",
            str(paths[1]),
            "--assets",
            str(paths[2]),
            "--readiness",
            str(paths[3]),
            "--asha",
            str(paths[4]),
            "--output",
            str(output),
            "--expected-compatible-count",
            "6",
        ]
    ) == 0
    assert "Training: not started" in capsys.readouterr().out
    assert output.is_file()
