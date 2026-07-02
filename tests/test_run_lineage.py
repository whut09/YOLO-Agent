"""Run lineage graph tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.run_lineage import RunLineageStore, build_lineage_record


def test_run_lineage_store_answers_parent_delta_and_best(tmp_path: Path) -> None:
    """Lineage graph should answer parent, evidence delta, and best trusted run."""
    store = RunLineageStore(tmp_path / "runs")
    store.append(
        build_lineage_record(
            run_id="exp001",
            run_dir=tmp_path / "runs" / "exp001",
            dataset_manifest_sha256="sha-parent",
            current_missing_evidence=["map50", "recall"],
            trusted=False,
        )
    )
    store.append(
        build_lineage_record(
            run_id="exp002",
            run_dir=tmp_path / "runs" / "exp002",
            parent_run_id="exp001",
            dataset_manifest_sha256="sha-parent",
            inherited_missing_evidence=["map50", "recall"],
            current_missing_evidence=[],
            trusted=True,
            metrics={"map50": 0.72, "recall": 0.81},
        )
    )
    store.append(
        build_lineage_record(
            run_id="exp003",
            run_dir=tmp_path / "runs" / "exp003",
            parent_run_id="exp001",
            dataset_manifest_sha256="sha-parent",
            trusted=True,
            metrics={"map50": 0.68},
        )
    )

    graph = store.graph()

    assert graph.parent_of("exp002") == "exp001"
    assert graph.children_of("exp001") == ["exp002", "exp003"]
    assert graph.inherited_dataset_manifest_sha("exp002") == "sha-parent"
    assert graph.evidence_delta("exp002") == {
        "inherited_missing": ["map50", "recall"],
        "current_missing": [],
        "resolved": ["map50", "recall"],
    }
    best = graph.best_trusted_run()
    assert best is not None
    assert best.run_id == "exp002"
    assert best.best_metric_name == "map50"
    assert best.best_metric_value == 0.72
