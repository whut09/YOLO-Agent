"""Offline tests for paper query and benchmark preference rules."""
from yolo_agent.research.paper_index import PaperIndex
from yolo_agent.research.schemas import PaperBenchmark, PaperComponentClaim, PaperRecord

def make_index_paper(paper_id: str, year: int, *, local: bool = False) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id, title=f"Method {paper_id}", year=year,
        task_families=["object_detection"], detector_family="yolo26", framework="pytorch",
        datasets=["COCO2017"], component_ids=["assigner.stal"],
        official_code_url="https://example.invalid/code" if local else None,
        applicability="direct_adapter_candidate" if local else "recipe_idea_only",
        claimed_effects=[PaperComponentClaim(component_id="assigner.stal", component_category="assigner", claimed_effect="Assignment", evidence_level="paper_claim")],
        benchmarks=[PaperBenchmark(dataset="COCO2017", model="yolo26n", metric_name="map50_95", value=0.45 if local else 0.4, evidence_level="locally_pilot_reproduced" if local else "paper_claim", verified=local)],
    )

def test_index_filters_requested_dimensions() -> None:
    index = PaperIndex([make_index_paper("p1", 2024), make_index_paper("p2", 2025, local=True)])
    assert [p.paper_id for p in index.list(year_from=2025)] == ["p2"]
    assert [p.paper_id for p in index.list(task_family="object_detection")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(detector_family="yolo26")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(component="assigner.stal")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(component_category="assigner")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(dataset="COCO2017")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(metric="map50_95")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(framework="pytorch")] == ["p2", "p1"]
    assert [p.paper_id for p in index.list(official_code=True)] == ["p2"]
    assert [p.paper_id for p in index.list(official_code=False)] == ["p1"]
    assert [p.paper_id for p in index.list(applicability="direct_adapter_candidate")] == ["p2"]

def test_index_selects_local_verified_benchmark() -> None:
    base = make_index_paper("p1", 2024)
    local = PaperBenchmark(dataset="COCO2017", model="yolo26n", metric_name="map50_95", value=0.43, evidence_level="locally_pilot_reproduced", verified=True, schema_version="research.v2")
    index = PaperIndex([base.model_copy(update={"benchmarks": [base.benchmarks[0], local]})])
    preferred = index.get_preferred_benchmark(paper_id="p1", dataset="COCO2017", model="yolo26n", metric_name="map50_95")
    all_records = index.get_all_benchmarks(paper_id="p1", dataset="COCO2017", model="yolo26n", metric_name="map50_95")
    assert preferred is not None and preferred.value == 0.43
    assert len(all_records) == 2 and all_records[0].value == 0.43

def test_index_preserves_same_rank_duplicates() -> None:
    base = make_index_paper("p1", 2024)
    duplicate = base.benchmarks[0].model_copy(update={"value": 0.41})
    records = PaperIndex([base.model_copy(update={"benchmarks": [base.benchmarks[0], duplicate]})]).get_all_benchmarks(paper_id="p1", dataset="COCO2017", model="yolo26n", metric_name="map50_95")
    assert {record.value for record in records} == {0.4, 0.41}
