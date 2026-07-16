"""Mock-LLM tests for strict paper component extraction."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from yolo_agent.core.llm_config import LLMDecisionConfig
from yolo_agent.research.component_extractor import ComponentExtractionBundle
from yolo_agent.research.llm_paper_analyzer import LLMPaperAnalyzer
from yolo_agent.research.schemas import ComponentTaxonomy, PaperBenchmark, PaperRecord

def paper() -> PaperRecord:
    return PaperRecord(paper_id="paper-1", title="Assignment for Object Detection", abstract="We add a training-only matcher at the target assignment stage.", year=2025, benchmarks=[PaperBenchmark(dataset="COCO2017", model="detector", metric_name="map50_95", value=0.4, evidence_level="paper_claim")])

def taxonomy() -> ComponentTaxonomy:
    return ComponentTaxonomy(categories={"matching": [], "assigner": [], "bbox_regression_loss": []})

def config() -> LLMDecisionConfig:
    return LLMDecisionConfig(enabled=True, provider="openai", model="gpt-5.5", api_key="test-key", api_key_env="TEST_KEY", require_api_key=True, retry_backoff_seconds=0)

def valid_payload() -> dict[str, object]:
    return {"schema_version": "component_extraction.v1", "extracted_components": [{"component_id": "matching.paper1", "name": "Paper Matcher", "component_category": "matching", "insertion_point": "target assignment", "required_inputs": ["predictions", "ground_truth"], "produced_outputs": ["matched_pairs"], "claimed_effects": [{"claim": "Improves assignment quality.", "paper_id": "paper-1", "source_location": "abstract", "evidence_level": "paper_claim"}], "target_error_types": ["false_negative"], "coupling_dependencies": ["unknown"], "incompatible_components": ["unknown"], "training_only": True, "inference_only": False, "implementation_notes": ["unknown"], "evidence_level": "paper_claim", "uncertainties": ["exact tensor contract unknown"], "source_locations": [{"paper_id": "paper-1", "location": "abstract"}]}]}

def test_analyzer_parses_strict_grounded_components_and_writes_ledger(tmp_path: Path) -> None:
    captured = {}
    def transport(cfg, messages):
        captured.update(json.loads(messages[1]["content"]))
        return json.dumps(valid_payload())
    ledger = tmp_path / "research_decisions.jsonl"
    result = LLMPaperAnalyzer(config=config(), transport=transport, ledger_path=ledger).analyze(paper=paper(), taxonomy=taxonomy(), body_summary="Section 3 describes matching.", yolo26_compatibility_context={"dfl_free": True})
    assert result.status == "used"
    assert result.extracted_components[0].component_category == "matching"
    assert result.extracted_components[0].evidence_level == "paper_claim"
    assert captured["paper"]["paper_id"] == "paper-1"
    assert "CandidateConfig" in " ".join(captured["hard_rules"])
    record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert record["input_summary"]["paper_id"] == "paper-1"
    assert record["prompt_sha256"] == result.prompt_sha256
    assert record["model_metadata"]["model"] == "gpt-5.5"

def test_analyzer_skips_without_key_and_does_not_touch_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_RESEARCH_KEY", raising=False)
    cfg = config().model_copy(update={"api_key": None, "api_key_env": "MISSING_RESEARCH_KEY"})
    registry_file = tmp_path / "papers.jsonl"
    registry_file.write_text("sentinel", encoding="utf-8")
    result = LLMPaperAnalyzer(config=cfg, ledger_path=tmp_path / "ledger.jsonl").analyze(paper=paper(), taxonomy=taxonomy())
    assert result.status == "skipped"
    assert result.extracted_components == []
    assert registry_file.read_text(encoding="utf-8") == "sentinel"

def test_extractor_rejects_wrong_paper_claim_reference(tmp_path: Path) -> None:
    payload = valid_payload()
    payload["extracted_components"][0]["claimed_effects"][0]["paper_id"] = "other-paper"  # type: ignore[index]
    result = LLMPaperAnalyzer(config=config(), transport=lambda cfg, messages: json.dumps(payload), ledger_path=tmp_path / "ledger.jsonl").analyze(paper=paper(), taxonomy=taxonomy())
    assert result.status == "failed"
    assert any("claim_wrong_paper_id" in warning for warning in result.warnings)

def test_extractor_rejects_benchmark_and_candidate_fields(tmp_path: Path) -> None:
    for extra in ({"benchmarks": [{"value": 99}]}, {"candidate_config": {"model": "yolo26n.pt"}}, {"train_overrides": {"imgsz": 1280}}):
        payload = valid_payload()
        payload["extracted_components"][0].update(extra)  # type: ignore[index]
        result = LLMPaperAnalyzer(config=config(), transport=lambda cfg, messages, value=payload: json.dumps(value), ledger_path=tmp_path / "ledger.jsonl").analyze(paper=paper(), taxonomy=taxonomy())
        assert result.status == "failed"
        assert any("component_schema_validation_failed" in warning for warning in result.warnings)

def test_unknown_fields_are_explicit_and_empty_lists_normalize() -> None:
    payload = valid_payload()
    component = payload["extracted_components"][0]  # type: ignore[index]
    component["required_inputs"] = []
    component["implementation_notes"] = []
    bundle = ComponentExtractionBundle.model_validate(payload)
    assert bundle.extracted_components[0].required_inputs == ["unknown"]
    assert bundle.extracted_components[0].implementation_notes == ["unknown"]

def test_claim_requires_source_location() -> None:
    payload = valid_payload()
    del payload["extracted_components"][0]["claimed_effects"][0]["source_location"]  # type: ignore[index]
    with pytest.raises(ValidationError):
        ComponentExtractionBundle.model_validate(payload)

def test_analyzer_failure_is_audited(tmp_path: Path) -> None:
    result = LLMPaperAnalyzer(config=config(), transport=lambda cfg, messages: (_ for _ in ()).throw(TimeoutError("timeout")), ledger_path=tmp_path / "ledger.jsonl").analyze(paper=paper(), taxonomy=taxonomy())
    assert result.status == "failed"
    assert "timeout" in result.warnings[0]
    assert (tmp_path / "ledger.jsonl").is_file()
