"""CPU-only tests for the paper readiness preflight and CLI contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
)
from yolo_agent.certification.paper_adapter_discovery import (
    ReusableAdapterDescriptor,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
    PaperAdapterCertificationReport,
    PaperAdapterCertificationResult,
)
from yolo_agent.certification.paper_readiness import (
    PaperReadinessPreflight,
    PaperReadinessReport,
)
from yolo_agent.cli import main
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


def _inventory(*, component: str = "loss.quality.correlation", evidence: list[str] | None = None) -> PaperExecutionInventory:
    records = [
        PaperExecutionSpec(
            paper_id=f"paper:{index:03d}",
            profile_id=f"profile:{index:03d}",
            title=f"Fixture {index}",
            source_locations=["fixture"],
            canonical_component_ids=[component],
            paper_specific_mechanism_ids=["quality_correlation"],
            required_evidence=list(evidence or []),
            recipe_ids=["recipe-quality"],
            execution_fingerprint=f"{index + 1:064x}",
            current_disposition="implementation_request",
            disposition_reason="fixture",
        )
        for index in range(83)
    ]
    return PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=728,
        compatible_paper_count=83,
        exact_reproduction_candidates=0,
        records=records,
    ).with_hash()


class _EmptyDiscovery:
    def discover(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(adapters=[], errors={})


def _descriptor(tmp_path: Path) -> ReusableAdapterDescriptor:
    contract = ComponentContract(
        component_id="loss.quality.correlation",
        display_name="Quality correlation",
        category="loss",
        implementation_path="fixture.adapter",
        adapter_class="FixtureAdapter",
        tensor_input_contract={"prediction": "tensor"},
        tensor_output_contract={"loss": "scalar"},
        runtime_payload_schema={"weight": "float"},
        fixed_imgsz_compatible=True,
        maturity="smoke_passed",
    )
    identity = AdapterCertificationIdentity(
        component_id=contract.component_id,
        adapter_hash="b" * 64,
        code_commit="fixture",
        ultralytics_version="fixture",
        protocol_hash="c" * 64,
    )
    return ReusableAdapterDescriptor(
        component_id=contract.component_id,
        contract=contract,
        contract_path=tmp_path / "contract.yaml",
        adapter_qualified_name="fixture.adapter:FixtureAdapter",
        identity=identity,
    )


class _Discovery:
    def __init__(self, descriptor: ReusableAdapterDescriptor) -> None:
        self.descriptor = descriptor

    def discover(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(adapters=[self.descriptor], errors={})


class _PassingFactory:
    def __init__(self, descriptor: ReusableAdapterDescriptor) -> None:
        self.descriptor = descriptor

    def run(self, **kwargs: object) -> PaperAdapterCertificationReport:
        assert kwargs["device"] == "cpu"
        assert kwargs["execute_real_gpu"] is False
        workdir = Path(str(kwargs["workdir"]))
        workdir.mkdir(parents=True, exist_ok=True)
        cpu_path = workdir / "component_certification.cpu.yaml"
        stages = [
            ComponentCertificationStage(stage_id=name, status="passed")
            for name in (
                "adapter_import",
                "runtime_payload",
                "hook_signature",
                "unit_tests",
                "isolated_smoke",
            )
        ]
        ComponentCertificationReport(
            component_id=self.descriptor.component_id,
            mode="cpu",
            status="passed",
            initial_maturity="smoke_passed",
            final_maturity="smoke_passed",
            protocol_hash=self.descriptor.identity.protocol_hash,
            adapter_hash=self.descriptor.identity.adapter_hash,
            code_commit=self.descriptor.identity.code_commit,
            ultralytics_version=self.descriptor.identity.ultralytics_version,
            registry_path=Path(str(kwargs["registry_path"])),
            workdir=workdir,
            stages=stages,
        ).to_yaml(cpu_path, sort_keys=False)
        result = PaperAdapterCertificationResult(
            component_id=self.descriptor.component_id,
            identity=self.descriptor.identity,
            status="passed",
            initial_maturity="smoke_passed",
            final_maturity="smoke_passed",
            selection_reason="fixture",
            cpu_report=cpu_path,
        )
        return PaperAdapterCertificationReport(
            status="passed",
            mode="cpu",
            registry_path=Path(str(kwargs["registry_path"])),
            selected_component_ids=[self.descriptor.component_id],
            results=[result],
        )


def test_preflight_emits_all_83_papers_without_training(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("path: .\ntrain: train\nval: val\n", encoding="utf-8")
    report = PaperReadinessPreflight(discovery=_EmptyDiscovery()).run(
        inventory=_inventory(),
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "paper_readiness_report.yaml",
        run_cpu_certification=False,
    )

    assert report.paper_count == 83
    assert len(report.records) == 83
    assert {item.paper_id for item in report.records} == {
        f"paper:{index:03d}" for index in range(83)
    }
    assert report.cpu_only is True
    assert report.training_started is False
    assert report.resource_policy == "cpu_only_no_gpu_probe"
    assert report.accuracy_claim == "none"
    assert report.gpu_probe == "not_run"
    assert all(item.asha_eligibility is False for item in report.records)
    assert all(item.exact_blocker == "adapter_missing:loss.quality.correlation" for item in report.records)
    fields = report.records[0].model_dump(mode="json")
    assert {
        "paper_id",
        "mechanism_id",
        "recipe_id",
        "adapter_hash",
        "protocol_hash",
        "cpu_contract_result",
        "shape_result",
        "forward_result",
        "backward_result",
        "payload_result",
        "dataset_evidence_result",
        "teacher_evidence_result",
        "graph_evidence_result",
        "matched_control_readiness",
        "asha_eligibility",
        "final_disposition",
        "exact_blocker",
    } <= set(fields)
    assert (tmp_path / "paper_readiness_report.yaml").is_file()


def test_passing_cpu_identity_makes_all_shared_papers_asha_eligible(
    tmp_path: Path,
) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    descriptor = _descriptor(tmp_path)
    report = PaperReadinessPreflight(
        discovery=_Discovery(descriptor),
        certification_factory=_PassingFactory(descriptor),
    ).run(
        inventory=_inventory(),
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "report.yaml",
        run_cpu_certification=True,
    )

    assert report.status == "passed"
    assert report.disposition_counts == {"runtime_ready": 83}
    assert all(item.asha_eligibility for item in report.records)
    assert all(item.adapter_hash == "b" * 64 for item in report.records)
    assert all(item.exact_blocker is None for item in report.records)


def test_missing_adapter_is_isolated_from_other_papers(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    records = _inventory().records
    records[0] = records[0].model_copy(
        update={
            "canonical_component_ids": ["component.unknown"],
            "paper_specific_mechanism_ids": ["unknown_mechanism"],
        }
    )
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=728,
        compatible_paper_count=83,
        exact_reproduction_candidates=0,
        records=sorted(records, key=lambda item: item.paper_id),
    ).with_hash()
    report = PaperReadinessPreflight(discovery=_EmptyDiscovery()).run(
        inventory=inventory,
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "report.yaml",
        run_cpu_certification=False,
    )
    assert report.records[0].exact_blocker == "adapter_missing:component.unknown"
    assert all(item.paper_id != report.records[0].paper_id or not item.asha_eligibility for item in report.records)


def test_preflight_reuses_identity_bound_cache_and_invalidates_registry_change(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: test\n", encoding="utf-8")
    preflight = PaperReadinessPreflight(discovery=_EmptyDiscovery())
    kwargs = {
        "inventory": _inventory(),
        "registry_path": registry,
        "model": "yolo26n.pt",
        "data": data,
        "output_path": tmp_path / "report.yaml",
        "run_cpu_certification": False,
    }
    first = preflight.run(**kwargs)
    second = preflight.run(**kwargs)
    assert first.cache_hits == 0
    assert second.cache_hits == 83
    registry.write_text("schema_version: changed\n", encoding="utf-8")
    third = preflight.run(**kwargs)
    assert third.cache_hits == 0


def test_preflight_reports_specific_protocol_blockers(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    preflight = PaperReadinessPreflight(discovery=_EmptyDiscovery())
    cases = [
        ("distillation.yolo26_teacher_student", ["teacher_checkpoint"], "teacher_checkpoint_missing"),
        ("domain_adaptation.general", [], "target_domain_dataset_missing"),
        ("sampling.hard_negative_replay", ["hard_negative_manifest"], "hard_negative_manifest_missing"),
        ("inference.sahi_slicing", [], "inference_only_not_training_candidate"),
    ]
    for index, (component, evidence, blocker) in enumerate(cases):
        inventory = _inventory(component=component, evidence=evidence)
        report = preflight.run(
            inventory=inventory,
            registry_path=tmp_path / f"registry-{index}.yaml",
            model="yolo26n.pt",
            data=data,
            output_path=tmp_path / f"report-{index}.yaml",
            run_cpu_certification=False,
        )
        assert report.records[0].exact_blocker == blocker


def test_report_rejects_missing_paper_and_disposition_accounting(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    report = PaperReadinessPreflight(discovery=_EmptyDiscovery()).run(
        inventory=_inventory(),
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "report.yaml",
        run_cpu_certification=False,
    )
    payload = report.model_dump(mode="json")
    payload["records"] = payload["records"][:-1]
    payload["paper_count"] = 82
    try:
        PaperReadinessReport.model_validate(payload)
    except ValueError as exc:
        assert "disposition counts" in str(exc) or "cache_hits" in str(exc)
    else:  # pragma: no cover - assertion documents the invariant
        raise AssertionError("truncated paper readiness report was accepted")


def test_report_loader_rejects_stale_report_hash(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    path = tmp_path / "report.yaml"
    report = PaperReadinessPreflight(discovery=_EmptyDiscovery()).run(
        inventory=_inventory(),
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=path,
        run_cpu_certification=False,
    )
    payload = report.model_dump(mode="json")
    payload["records"][0]["exact_blocker"] = "tampered"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    try:
        PaperReadinessPreflight.load_report(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale readiness report was accepted")


def test_cli_paper_readiness_snapshot_is_decision_oriented(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    source = _inventory()
    base = PaperReadinessPreflight(discovery=_EmptyDiscovery()).run(
        inventory=source,
        registry_path=tmp_path / "registry.yaml",
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        output_path=tmp_path / "base.yaml",
        run_cpu_certification=False,
    )
    monkeypatch.setattr("yolo_agent.cli.run_paper_readiness", lambda **kwargs: base)

    assert main(
        [
            "research",
            "paper-readiness",
            "--root",
            str(tmp_path),
            "--registry",
            str(tmp_path / "registry.yaml"),
            "--model",
            "yolo26n.pt",
            "--data",
            str(tmp_path / "coco.yaml"),
            "--no-cpu-certification",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Training: not started (CPU-only readiness evidence)" in output
    assert "Papers:   83/83" in output
    assert output.count("\tblocker=") == 83
    assert "mAP" not in output


def test_cli_paper_readiness_failure_has_no_traceback(
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "yolo_agent.cli.run_paper_readiness",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("inventory stale")),
    )
    assert main(["research", "paper-readiness", "--no-cpu-certification"]) == 1
    output = capsys.readouterr().out
    assert "FAILED - paper readiness audit could not complete" in output
    assert "inventory stale" in output
    assert "Training: not started" in output
    assert "Traceback" not in output
