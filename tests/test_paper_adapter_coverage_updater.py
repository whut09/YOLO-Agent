from pathlib import Path

from yolo_agent.certification.paper_adapter_coverage_updater import (
    PaperAdapterCoverageUpdater,
)
from yolo_agent.tools.paper_adapter_coverage import LocalPaperAdapterCoverageReport


def test_coverage_updater_writes_atomic_local_report(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    output = tmp_path / "coverage" / "paper_adapter_coverage.yaml"

    report = PaperAdapterCoverageUpdater().refresh(
        registry_path=registry,
        output_path=output,
    )

    restored = LocalPaperAdapterCoverageReport.from_yaml(output)
    assert restored == report
    assert restored.registry_path == registry.resolve()
    assert not list(output.parent.glob("*.tmp"))
