"""Offline tests for the local paper registry."""
from pathlib import Path
import pytest
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperBenchmark, PaperComponentClaim, PaperRecord

def make_paper(
    paper_id: str,
    *,
    title: str = "A Detection Method",
    doi: str | None = None,
    local: bool = False,
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id, doi=doi, title=title, year=2024,
        official_code_url="https://example.invalid/code" if local else None,
        component_ids=["assigner.example"], datasets=["COCO2017"], framework="pytorch",
        applicability="direct_adapter_candidate" if local else "recipe_idea_only",
        claimed_effects=[PaperComponentClaim(component_id="assigner.example", component_category="assigner", claimed_effect="Improves assignment.", evidence_level="paper_claim")],
        benchmarks=[PaperBenchmark(dataset="COCO2017", model="detector-n", metric_name="map50_95", value=0.42 if local else 0.4, evidence_level="locally_pilot_reproduced" if local else "paper_claim", verified=local)],
    )

def test_registry_crud_and_atomic_files(tmp_path: Path) -> None:
    registry = PaperRegistry(tmp_path / "research")
    first = make_paper("paper-1")
    registry.add(first)
    assert registry.get("paper-1") == first
    assert len(registry.list()) == 1
    assert (tmp_path / "research" / "papers.jsonl").is_file()
    assert (tmp_path / "research" / "paper_components.jsonl").is_file()
    assert (tmp_path / "research" / "paper_index.json").is_file()
    registry.update(first.model_copy(update={"title": "Updated Detection Method"}))
    assert registry.get("paper-1").title == "Updated Detection Method"  # type: ignore[union-attr]
    registry.upsert(make_paper("paper-2", title="Second Method"))
    assert {item.paper_id for item in registry.list()} == {"paper-1", "paper-2"}
    assert registry.remove("paper-2").paper_id == "paper-2"

def test_registry_rejects_duplicate_identity_on_add(tmp_path: Path) -> None:
    registry = PaperRegistry(tmp_path / "research")
    registry.add(make_paper("paper-1", doi="10.1000/example"))
    with pytest.raises(ValueError, match="already exists"):
        registry.add(make_paper("paper-2", doi="https://doi.org/10.1000/example"))

def test_registry_upsert_deduplicates_arxiv_versions(tmp_path: Path) -> None:
    registry = PaperRegistry(tmp_path / "research")
    registry.upsert(make_paper("arxiv:2401.12345v1"))
    registry.upsert(make_paper("https://arxiv.org/abs/2401.12345v2", local=True))
    papers = registry.list()
    assert len(papers) == 1
    assert papers[0].paper_id.endswith("v2")
    assert papers[0].official_code_url is not None

def test_registry_deduplicate_prefers_local_record(tmp_path: Path) -> None:
    registry = PaperRegistry(tmp_path / "research")
    registry.upsert(make_paper("p1", doi="10.1000/x"))
    registry.upsert(make_paper("p2", doi="10.1000/x", local=True))
    assert len(registry.list()) == 1
    assert registry.list()[0].paper_id == "p2"

def test_registry_rejects_corrupt_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "research"
    root.mkdir()
    (root / "papers.jsonl").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSONL"):
        PaperRegistry(root)


def test_registry_accepts_utf8_bom(tmp_path: Path) -> None:
    import json

    root = tmp_path / "research"
    root.mkdir()
    root.joinpath("papers.jsonl").write_text(
        json.dumps(make_paper("bom-paper").model_dump(mode="json")),
        encoding="utf-8-sig",
    )

    assert PaperRegistry(root).get("bom-paper") is not None
