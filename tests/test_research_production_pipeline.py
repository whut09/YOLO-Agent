"""Offline research production and frozen snapshot tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.agents.decision_bundle import LLMDecisionBundle
from yolo_agent.agents.auto_optimization_loop import AutoOptimizationLoopDriver
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.cli import main
from yolo_agent.research.component_extractor import (
    ComponentExtractionBundle,
    ComponentExtractionResult,
    ExtractedClaim,
    ExtractedComponent,
    SourceLocation,
)
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.production_pipeline import ResearchProductionPipeline
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import ResearchSnapshot, load_research_snapshot
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore


class FakeAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, *, paper, taxonomy):  # type: ignore[no-untyped-def]
        self.calls += 1
        component = ExtractedComponent(
            component_id="sampling.paper_small_object",
            name="Paper Small Object Sampler",
            component_category="sampling",
            insertion_point="train_dataloader",
            required_inputs=["bbox_area"],
            produced_outputs=["sample_weight"],
            claimed_effects=[
                ExtractedClaim(
                    claim="Improves AP_small under the paper protocol.",
                    paper_id=paper.paper_id,
                    source_location="abstract",
                    evidence_level="paper_claim",
                )
            ],
            target_error_types=["area_metric"],
            coupling_dependencies=["none"],
            incompatible_components=["unknown"],
            training_only=True,
            inference_only=False,
            implementation_notes=["Adapter is not implemented."],
            evidence_level="paper_claim",
            uncertainties=["Local reproduction is missing."],
            source_locations=[SourceLocation(paper_id=paper.paper_id, location="abstract")],
        )
        return ComponentExtractionResult(
            status="used",
            paper_id=paper.paper_id,
            provider="test",
            model="test-model",
            bundle=ComponentExtractionBundle(extracted_components=[component]),
        )


def _paper() -> PaperRecord:
    return PaperRecord(
        paper_id="paper-small-object",
        title="Small Object Sampling for Real-Time Detection",
        abstract="A sampling method improves AP_small for real-time object detection.",
        year=2025,
        task_families=["object_detection", "small_object_detection"],
        detector_family="yolo",
        datasets=["COCO"],
    )


def _dataset(root: Path) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"image")
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    path = root / "data.yaml"
    path.write_text("path: .\ntrain: images/train\nnames: [object]\n", encoding="utf-8")
    return path


def test_pipeline_builds_replayable_snapshot_and_reuses_extractions(tmp_path: Path) -> None:
    root = tmp_path / "research"
    PaperRegistry(root).add(_paper())
    analyzer = FakeAnalyzer()
    pipeline = ResearchProductionPipeline(root, analyzer=analyzer)

    first = pipeline.run()
    second = pipeline.run()

    assert first.status == "completed"
    assert first.snapshot_hash == second.snapshot_hash
    assert analyzer.calls == 1
    loaded = load_research_snapshot(root)
    assert loaded is not None
    snapshot, snapshot_dir = loaded
    assert snapshot.snapshot_hash == first.snapshot_hash
    assert snapshot.verify(snapshot_dir) == []
    assert snapshot.paper_count == 1
    assert snapshot.component_count == 1
    assert snapshot.recipe_count == 1
    queue = yaml.safe_load((snapshot_dir / "reproduction_queue.yaml").read_text(encoding="utf-8-sig"))
    assert queue["items"][0]["status"] == "adapter_required"
    assert queue["items"][0]["queued_for_training"] is False


def test_frozen_snapshot_does_not_change_when_live_registry_changes(tmp_path: Path) -> None:
    root = tmp_path / "research"
    registry = PaperRegistry(root)
    registry.add(_paper())
    result = ResearchProductionPipeline(root, analyzer=FakeAnalyzer()).run()
    snapshot_dir = Path(result.snapshot_path or "")
    snapshot = ResearchSnapshot.from_snapshot_dir(snapshot_dir)
    frozen_papers_hash = snapshot.artifacts["papers"].sha256

    registry.add(PaperRecord(paper_id="new-paper", title="New paper", year=2026))

    assert snapshot.verify(snapshot_dir) == []
    assert snapshot.artifacts["papers"].sha256 == frozen_papers_hash
    assert len(PaperRegistry(snapshot_dir).list()) == 1


def test_decision_bundle_references_frozen_snapshot(tmp_path: Path, monkeypatch) -> None:
    research_root = tmp_path / "research"
    PaperRegistry(research_root).add(_paper())
    built = ResearchProductionPipeline(research_root, analyzer=FakeAnalyzer()).run()
    data_yaml = _dataset(tmp_path / "dataset")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_type: detect\nscene: generic\nclass_names: [object]\nprimary_metric: {name: map50_95}\n",
        encoding="utf-8",
    )
    errors_path = tmp_path / "errors.yaml"
    errors_path.write_text(
        "errors:\n  - error_type: small_object_miss\n    count: 1\n    severity: high\n",
        encoding="utf-8",
    )
    orchestrator = LoopOrchestrator.initialize(
        run_id="snapshot-decision",
        task_path=task,
        data_yaml=data_yaml,
        run_root=tmp_path / "runs",
        detection_errors_path=errors_path,
    )
    orchestrator.context.metadata.update(
        {
            "research_snapshot_hash": built.snapshot_hash,
            "research_snapshot_path": built.snapshot_path,
            "research_snapshot_verified": True,
        }
    )
    orchestrator.context.to_yaml()

    assert orchestrator.run_stage("profile_data").status == "completed"
    assert orchestrator.run_stage("advise_labels").status == "completed"
    assert orchestrator.run_stage("diagnose_errors").status == "completed"
    assert orchestrator.run_stage("generate_loop_plan").status == "completed"

    bundle = LLMDecisionBundle.from_yaml(orchestrator.context.artifact_path("llm_decision_bundle.yaml"))
    assert bundle.context.research_snapshot_hash == built.snapshot_hash
    assert bundle.context.research_snapshot_path == built.snapshot_path
    assert bundle.context.research_snapshot_verified is True


def test_research_snapshot_cli_is_offline_by_default(tmp_path: Path, capsys) -> None:
    PaperRegistry(tmp_path / "research").add(_paper())

    exit_code = main(["research", "build-snapshot", "--root", str(tmp_path / "research")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Status:     completed" in output
    assert "Snapshot:" in output


def test_auto_round_loads_only_the_bound_snapshot(tmp_path: Path) -> None:
    research_root = tmp_path / "research"
    PaperRegistry(research_root).add(_paper())
    built = ResearchProductionPipeline(research_root, analyzer=FakeAnalyzer()).run()
    data_yaml = _dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="snapshot-auto",
        run_root=run_root,
        profile="pilot",
        execute=False,
    )
    PaperRegistry(research_root).add(PaperRecord(paper_id="later-paper", title="Later paper", year=2026))
    later = ResearchProductionPipeline(research_root, analyzer=FakeAnalyzer()).run()
    assert later.snapshot_hash != built.snapshot_hash
    ErrorFactStore(run_root).append(
        base.run_id,
        [
            ErrorFact(
                run_id=base.run_id,
                candidate_id="baseline",
                node_id="node_baseline",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
                action_candidates=["small_object_recipe"],
            )
        ],
    )

    result = AutoOptimizationLoopDriver().run(
        base_run_dir=base.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )

    child = run_root / result.rounds[0].run_id
    plan = yaml.safe_load((child / "artifacts" / "paper_recipe_plan.yaml").read_text(encoding="utf-8-sig"))
    context = yaml.safe_load((child / "run_context.yaml").read_text(encoding="utf-8-sig"))
    assert plan["research_snapshot_hash"] == built.snapshot_hash
    assert plan["research_snapshot_verified"] is True
    assert context["metadata"]["research_snapshot_hash"] == built.snapshot_hash
