"""The capability maturity matrix must be regenerable from its manifest."""

from __future__ import annotations

from tests.docs_inventory import REPO_ROOT, read_doc
from yolo_agent.tools import capability_matrix


def test_capability_matrix_is_fresh() -> None:
    exit_code = capability_matrix.main(
        [
            "--check",
            "--config",
            str(REPO_ROOT / "configs/capability_maturity.yaml"),
            "--document",
            str(REPO_ROOT / "docs/capability-maturity.md"),
            "--readme",
            str(REPO_ROOT / "README.md"),
            "--readme-zh",
            str(REPO_ROOT / "README.zh-CN.md"),
            "--coverage-report",
            str(REPO_ROOT / "docs/paper-adapter-coverage.yaml"),
        ]
    )
    assert exit_code == 0


def test_capability_maturity_document_pins_manifest_metadata() -> None:
    text = read_doc(REPO_ROOT / "docs" / "capability-maturity.md")
    assert "2026-08-03" in text, "manifest reviewed_at missing from generated document"
    assert "Schema：`v1`" in text, "manifest schema_version missing from generated document"
