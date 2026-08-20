"""CPU-only tests for the paper readiness preflight and CLI contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from yolo_agent.certification.paper_readiness import (
    PaperReadinessPreflight,
)
from yolo_agent.cli import main
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
