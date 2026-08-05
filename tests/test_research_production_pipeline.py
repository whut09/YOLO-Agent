"""Offline research production and frozen snapshot tests."""

from __future__ import annotations

from pathlib import Path
import shutil

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
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.mechanism_clusters import (
    PaperMechanismClusterReport,
    load_frozen_mechanism_cluster_report,
)
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
)
from yolo_agent.research.production_pipeline import (
    ResearchProductionPipeline,
    _observable_target_error_facts,
)
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.snapshot import ResearchSnapshot, load_research_snapshot
from yolo_agent.resources import ResourcePaths
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


class FakeRegistry:
    def __init__(self, root: Path, papers: list[PaperRecord]) -> None:
        self._registry = PaperRegistry(root)
        for paper in papers:
            self._registry.upsert(paper)
        self.papers_path = self._registry.papers_path
        self.deduplicate_calls = 0

    def deduplicate(self):  # type: ignore[no-untyped-def]
        self.deduplicate_calls += 1
        return self._registry.deduplicate()

    def list(self):  # type: ignore[no-untyped-def]
        return self._registry.list()


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


def test_scale_paper_recipe_binds_to_observable_small_object_fact() -> None:
    result = FakeAnalyzer().analyze(paper=_paper(), taxonomy=None)
    assert result.bundle is not None
    component = result.bundle.extracted_components[0].model_copy(
        update={
            "component_id": "neck.multi_scale_fusion",
            "component_category": "neck",
            "target_error_types": ["scale_variation", "small_object_false_negative"],
        }
    )

    facts = _observable_target_error_facts(component)

    assert {"fact_type": "scale_variation"} in facts
    assert {"fact_type": "small_object_false_negative"} in facts
    assert {
        "fact_type": "area_metric",
        "area": "small",
        "metric_name": "ap_small",
    } in facts


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
    assert snapshot.snapshot_status == "current"
    assert snapshot.paper_method_coverage_version != "not_available"
    assert snapshot.effective_maturity_version != "not_available"
    assert snapshot.verify(snapshot_dir) == []
    assert snapshot.paper_count == 1
    assert snapshot.component_count == 1
    assert snapshot.recipe_count == 1
    assert snapshot.paper_intelligence == "available"
    assert snapshot.maturity_summary.metadata_only == 1
    assert snapshot.maturity_summary.recipe_idea_only == 0
    assert snapshot.maturity_summary.adapter_implemented == 0
    assert snapshot.maturity_summary.runtime_integrated == 0
    assert snapshot.maturity_summary.unit_tested == 0
    assert snapshot.maturity_summary.smoke_passed == 0
    assert snapshot.maturity_summary.gpu_certified == 0
    assert snapshot.maturity_summary.pilot_reproduced == 0
    assert snapshot.maturity_summary.full_reproduced == 0
    assert snapshot.maturity_summary.confirmed_multi_seed == 0
    queue = yaml.safe_load((snapshot_dir / "reproduction_queue.yaml").read_text(encoding="utf-8-sig"))
    assert queue["items"][0]["status"] == "adapter_required"
    assert queue["items"][0]["queued_for_training"] is False
    method_coverage = PaperMethodCoverageReport.from_yaml(
        snapshot_dir / "paper_method_coverage.yaml"
    )
    assert method_coverage.paper_count == 1
    assert method_coverage.profile_count == 1
    assert method_coverage.decisions[0].paper_id == "paper-small-object"
    mechanism_report = PaperMechanismClusterReport.from_yaml(
        root / "production" / "paper_mechanism_clusters.yaml"
    )
    assert mechanism_report.paper_count == 1
    assert mechanism_report.matches[0].paper_id == "paper-small-object"
    assert mechanism_report.report_hash
    assert (root / "production" / "paper_mechanism_clusters.md").is_file()
    assert snapshot.artifacts["paper_method_coverage"].sha256
    assert snapshot.artifacts["paper_mechanism_clusters"].sha256
    assert snapshot.artifacts["paper_mechanism_cluster_taxonomy"].sha256
    frozen_mechanisms = PaperMechanismClusterReport.from_yaml(
        snapshot_dir / snapshot.artifacts["paper_mechanism_clusters"].path
    )
    assert frozen_mechanisms.report_hash == mechanism_report.report_hash
    assert (
        load_frozen_mechanism_cluster_report(snapshot_dir).report_hash
        == mechanism_report.report_hash
    )
    assert snapshot.artifacts["paper_method_evidence"].sha256
    assert snapshot.artifacts["cached_code_metadata"].sha256
    assert snapshot.artifacts["paper_method_evidence_coverage"].sha256
    assert "paper_method_evidence_report" not in snapshot.artifacts
    executable_artifact = snapshot.artifacts["executable_coverage_baseline"]
    markdown_artifact = snapshot.artifacts["executable_coverage_report"]
    executable = ExecutablePaperCoverageBaseline.from_yaml(
        snapshot_dir / executable_artifact.path
    )
    assert executable.denominators["all_papers"].paper_count == snapshot.paper_count
    assert executable_artifact.sha256
    assert markdown_artifact.sha256
    assert (snapshot_dir / markdown_artifact.path).is_file()


def test_mechanism_taxonomy_change_creates_new_snapshot_hash(tmp_path: Path) -> None:
    root = tmp_path / "research"
    PaperRegistry(root).add(_paper())
    taxonomy = tmp_path / "paper_mechanism_clusters.yaml"
    shutil.copy2(ResourcePaths.PAPER_MECHANISM_CLUSTERS, taxonomy)

    first = ResearchProductionPipeline(
        root,
        mechanism_cluster_path=taxonomy,
    ).run()
    original = taxonomy.read_text(encoding="utf-8")
    taxonomy.write_text(
        original.replace(
            "weighted_training_data_exposure",
            "weighted_training_data_exposure_v2",
            1,
        ),
        encoding="utf-8",
    )
    second = ResearchProductionPipeline(
        root,
        mechanism_cluster_path=taxonomy,
    ).run()

    assert first.snapshot_hash != second.snapshot_hash


def test_pipeline_accepts_mock_registry_and_mock_llm(tmp_path: Path) -> None:
    analyzer = FakeAnalyzer()
    registries: list[FakeRegistry] = []

    def registry_factory(root: Path) -> FakeRegistry:
        registry = FakeRegistry(root, [_paper()])
        registries.append(registry)
        return registry

    result = ResearchProductionPipeline(
        tmp_path / "research",
        analyzer=analyzer,
        registry_factory=registry_factory,  # type: ignore[arg-type]
    ).run()

    assert result.status == "completed"
    assert result.paper_intelligence == "available"
    assert registries[0].deduplicate_calls == 1
    assert analyzer.calls == 1


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


def test_empty_registry_freezes_explicit_unavailable_snapshot(tmp_path: Path) -> None:
    research_root = tmp_path / "research"

    result = ResearchProductionPipeline(research_root, analyzer=FakeAnalyzer()).run()

    assert result.status == "completed"
    assert result.paper_intelligence == "unavailable"
    assert result.unavailable_reason == "empty_registry"
    assert result.snapshot_hash
    assert result.maturity_summary.model_dump() == {
        "metadata_only": 0,
        "recipe_idea_only": 0,
        "adapter_implemented": 0,
        "runtime_integrated": 0,
        "unit_tested": 0,
        "smoke_passed": 0,
        "gpu_certified": 0,
        "pilot_reproduced": 0,
        "full_reproduced": 0,
        "confirmed_multi_seed": 0,
    }
    loaded = load_research_snapshot(research_root)
    assert loaded is not None
    snapshot, _ = loaded
    assert snapshot.paper_intelligence == "unavailable"
    assert snapshot.unavailable_reason == "empty_registry"


def test_training_binds_unavailable_snapshot_without_research_network(tmp_path: Path) -> None:
    research_root = tmp_path / "research"
    built = ResearchProductionPipeline(research_root, analyzer=FakeAnalyzer()).run()
    data_yaml = _dataset(tmp_path / "dataset")

    result = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=data_yaml,
        run_id="empty-research",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )

    context = yaml.safe_load((result.run_dir / "run_context.yaml").read_text(encoding="utf-8-sig"))
    metadata = context["metadata"]
    assert metadata["research_snapshot_hash"] == built.snapshot_hash
    assert metadata["snapshot_status"] == "current"
    assert metadata["paper_method_coverage_version"] != "not_available"
    assert metadata["effective_maturity_version"] != "not_available"
    assert metadata["paper_intelligence"] == "unavailable"
    assert metadata["unavailable_reason"] == "empty_registry"
    assert metadata["research_network_allowed"] is False

    ErrorFactStore(tmp_path / "runs").append(
        result.run_id,
        [
            ErrorFact(
                run_id=result.run_id,
                candidate_id="baseline",
                node_id="node_baseline",
                dataset_version="coco2017",
                fact_type="area_metric",
                subject="small",
                metric_name="ap_small",
                value=0.1,
                severity="high",
            )
        ],
    )
    loop = AutoOptimizationLoopDriver().run(
        base_run_dir=result.run_dir,
        auto_rounds=1,
        execute=False,
        executor="dry-run",
        max_steps=4,
    )
    child = tmp_path / "runs" / loop.rounds[0].run_id
    plan = yaml.safe_load((child / "artifacts" / "paper_recipe_plan.yaml").read_text(encoding="utf-8-sig"))
    child_context = yaml.safe_load((child / "run_context.yaml").read_text(encoding="utf-8-sig"))
    assert child_context["metadata"]["research_snapshot_hash"] == built.snapshot_hash
    assert child_context["metadata"]["snapshot_status"] == "current"
    assert child_context["metadata"]["paper_method_coverage_version"] == (
        metadata["paper_method_coverage_version"]
    )
    assert child_context["metadata"]["effective_maturity_version"] == (
        metadata["effective_maturity_version"]
    )
    assert plan["research_snapshot_hash"] == built.snapshot_hash
    assert plan["paper_intelligence"] == "unavailable"
    assert plan["research_network_allowed"] is False
    assert plan["decision_context_inputs"]["paper_candidates"] == []


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
