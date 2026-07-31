from __future__ import annotations

from pathlib import Path

from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.tools.paper_method_coverage import build_report, main


def _populate(root: Path) -> None:
    registry = PaperRegistry(root)
    registry.add(
        PaperRecord(
            paper_id="sampling-paper",
            title="Sampling paper",
            year=2025,
            component_ids=["small_object_sampling"],
            applicability="direct_adapter_candidate",
        )
    )
    registry.add(
        PaperRecord(
            paper_id="unknown-paper",
            title="Unknown paper",
            year=2025,
            component_ids=["unknown_new_mechanism"],
        )
    )


def test_build_report_maps_papers_and_explains_unimplemented_components(
    tmp_path: Path,
) -> None:
    root = tmp_path / "research"
    _populate(root)

    report = build_report(root=root)

    assert report.paper_count == 2
    assert report.decision_counts["reuse_existing_adapter"] == 1
    assert report.decision_counts["insufficient_information"] == 1
    assert report.adapter_to_papers == {
        "sampling.small_object": ["sampling-paper"]
    }
    assert report.unimplemented_reasons["unknown_new_mechanism"] == [
        "canonical_component_mapping_required"
    ]


def test_cli_writes_machine_readable_coverage_without_mutating_registry(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "research"
    report_path = tmp_path / "paper-to-adapter.yaml"
    _populate(root)
    before = (root / "papers.jsonl").read_bytes()

    result = main(["--root", str(root), "--report", str(report_path)])

    assert result == 0
    written = PaperMethodCoverageReport.from_yaml(report_path)
    assert written.paper_count == 2
    assert [item.paper_id for item in written.decisions] == [
        "sampling-paper",
        "unknown-paper",
    ]
    assert (root / "papers.jsonl").read_bytes() == before
    output = capsys.readouterr().out
    assert "Mechanisms: referenced=" in output
    assert "Coverage: adapter=" in output
