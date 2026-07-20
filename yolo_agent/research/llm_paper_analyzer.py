"""LLM-backed paper component analysis with strict local guards."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.research.component_extractor import ComponentExtractionResult, ComponentExtractor
from yolo_agent.research.schemas import ComponentTaxonomy, PaperRecord

PaperLLMTransport = Callable[[LLMDecisionConfig, list[dict[str, str]]], str]

class ResearchDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "research_decision.v1"
    paper_id: str
    decision_type: str = "component_extraction"
    status: str
    prompt_sha256: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LLMPaperAnalyzer:
    def __init__(self, *, config: LLMDecisionConfig | None = None, transport: PaperLLMTransport | None = None, ledger_path: Path | str = Path("research") / "decision_ledger.jsonl") -> None:
        self.config = config or load_llm_decision_config()
        self.transport = transport or _unavailable_transport
        self.ledger_path = Path(ledger_path)
        self.extractor = ComponentExtractor()

    def analyze(self, *, paper: PaperRecord, taxonomy: ComponentTaxonomy, body_summary: str | None = None, yolo26_compatibility_context: dict[str, Any] | None = None) -> ComponentExtractionResult:
        compatibility = yolo26_compatibility_context or {}
        summary = _input_summary(paper, taxonomy, body_summary, compatibility)
        messages = _messages(paper, taxonomy, body_summary, compatibility)
        prompt_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        config = self.config
        if not config.can_generate_proposals:
            return self._finish(_result(config, paper.paper_id, prompt_hash, "skipped", "llm_config_disabled"), summary)
        if config.provider == "XX" or config.model == "XX":
            return self._finish(_result(config, paper.paper_id, prompt_hash, "skipped", "llm_config_redacted"), summary)
        if config.require_api_key and not config.resolved_api_key():
            return self._finish(_result(config, paper.paper_id, prompt_hash, "skipped", f"missing_api_key:{config.api_key_source()}"), summary)
        try:
            raw_text = self.transport(config, messages)
        except Exception as exc:
            return self._finish(_result(config, paper.paper_id, prompt_hash, "failed", f"llm_call_failed:{exc}"), summary)
        bundle, warnings = self.extractor.parse(raw_text, paper=paper, taxonomy=taxonomy)
        result = ComponentExtractionResult(status="used" if bundle else "failed", paper_id=paper.paper_id, provider=config.provider, model=config.model, prompt_sha256=prompt_hash, bundle=bundle, warnings=warnings, raw_text=raw_text)
        return self._finish(result, summary)

    def _finish(self, result: ComponentExtractionResult, summary: dict[str, Any]) -> ComponentExtractionResult:
        record = ResearchDecisionRecord(paper_id=result.paper_id, status=result.status, prompt_sha256=result.prompt_sha256 or "", input_summary=summary, model_metadata={"provider": result.provider, "model": result.model}, output=result.bundle.model_dump(mode="json") if result.bundle else None, parse_warnings=result.warnings)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + chr(10))
        return result

def _messages(paper: PaperRecord, taxonomy: ComponentTaxonomy, body_summary: str | None, compatibility: dict[str, Any]) -> list[dict[str, str]]:
    payload = {"paper": paper.model_dump(mode="json"), "optional_body_summary": body_summary or "unknown", "component_taxonomy": list(taxonomy.categories), "yolo26_compatibility_context": compatibility, "hard_rules": ["Return JSON only.", "Never output CandidateConfig, train_overrides, commands, executable decisions, or benchmark records.", "Do not invent benchmark values.", "Every claim must use the supplied paper_id and a source location.", "Use unknown when evidence is absent.", "All evidence_level fields must equal paper_claim."]}
    return [{"role": "system", "content": "Extract non-executable research component metadata."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]

def _input_summary(paper: PaperRecord, taxonomy: ComponentTaxonomy, body_summary: str | None, compatibility: dict[str, Any]) -> dict[str, Any]:
    return {"paper_id": paper.paper_id, "title": paper.title, "abstract_present": bool(paper.abstract), "benchmark_count": len(paper.benchmarks), "body_summary_present": bool(body_summary), "taxonomy_categories": sorted(taxonomy.categories), "compatibility_context_keys": sorted(compatibility)}

def _result(config: LLMDecisionConfig, paper_id: str, prompt_hash: str, status: str, warning: str) -> ComponentExtractionResult:
    return ComponentExtractionResult(status=status, paper_id=paper_id, provider=config.provider, model=config.model, prompt_sha256=prompt_hash, warnings=[warning])

def _unavailable_transport(config: LLMDecisionConfig, messages: list[dict[str, str]]) -> str:
    raise RuntimeError("paper LLM transport was not configured")

__all__ = ["LLMPaperAnalyzer", "PaperLLMTransport", "ResearchDecisionRecord"]
