"""Offline parsing of non-executable diagnostic hints from paper catalogs."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.component_aliases import normalize_component_id
from yolo_agent.research.schemas import ComponentTaxonomy, PaperRecord


class PaperDiagnosticHint(BaseModel):
    """A paper-level diagnostic prior, never an executable training action."""

    model_config = ConfigDict(extra="forbid")

    symptom: str = "unknown"
    likely_cause: str = "unknown"
    evidence_needed: list[str] = Field(default_factory=lambda: ["unknown"])
    candidate_component_ids: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    target_error_facts: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_location: str
    evidence_level: Literal["paper_claim"] = "paper_claim"


class HarnessHintParseResult(BaseModel):
    """Rule-parser output with warnings instead of executable actions."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    hints: list[PaperDiagnosticHint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HarnessHintParser:
    """Extract explicit symptoms, causes, metrics, and evidence requests."""

    def parse(
        self,
        paper: PaperRecord,
        taxonomy: ComponentTaxonomy,
        harness_hints: list[str] | None = None,
    ) -> HarnessHintParseResult:
        del taxonomy  # The schema constrains categories; rules only use explicit paper text.
        raw_hints = harness_hints
        if raw_hints is None and paper.provenance is not None:
            raw_hints = paper.provenance.original_harness_hints
        warnings: list[str] = []
        parsed: list[PaperDiagnosticHint] = []
        for index, raw in enumerate(raw_hints or []):
            text, text_warnings = sanitize_research_text(raw)
            warnings.extend(f"harness_hints[{index}]:{warning}" for warning in text_warnings)
            if not text:
                warnings.append(f"harness_hints[{index}]:empty_hint")
                continue
            metrics = extract_target_metrics(text)
            error_facts = extract_error_facts(text)
            evidence = _extract_evidence_requests(text, metrics, error_facts)
            parsed.append(PaperDiagnosticHint(
                symptom=_extract_symptom(text),
                likely_cause=_extract_cause(text),
                evidence_needed=evidence or ["unknown"],
                candidate_component_ids=_explicit_component_ids(text, paper.component_ids),
                target_metrics=metrics,
                target_error_facts=error_facts,
                confidence=_confidence(text, metrics, error_facts),
                source_location=f"harness_hints[{index}]",
            ))
        return HarnessHintParseResult(paper_id=paper.paper_id, hints=parsed, warnings=warnings)


def sanitize_research_text(value: str) -> tuple[str, list[str]]:
    """Remove control noise while retaining mixed Chinese/English content."""
    warnings: list[str] = []
    text = str(value or "")
    if "\ufffd" in text:
        warnings.append("encoding_replacement_character")
    if "\x00" in text:
        warnings.append("nul_character_removed")
    text = text.replace("\x00", " ")
    text = "".join(character if character in "\n\t" or ord(character) >= 32 else " " for character in text)
    return re.sub(r"[ \t]+", " ", text).strip(), warnings


def extract_target_metrics(text: str) -> list[str]:
    patterns = (
        (r"\b(?:mAP50[-_:]?95|mAP@[.]?5[:：.]95|AP50[-_:]?95)\b", "map50_95"),
        (r"\b(?:AP[_ -]?small|APs)\b|小目标\s*AP", "ap_small"),
        (r"\b(?:AP[_ -]?medium|APm)\b", "ap_medium"),
        (r"\b(?:AP[_ -]?large|APl)\b", "ap_large"),
        (r"\brecall\b|召回率", "recall"),
        (r"\bprecision\b|精确率", "precision"),
        (r"\blatency(?:_ms)?\b|延迟", "latency_ms"),
        (r"\bthroughput\b|吞吐", "throughput"),
        (r"\bmodel[_ -]?size\b|模型大小", "model_size"),
    )
    return [name for pattern, name in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def extract_error_facts(text: str) -> list[str]:
    patterns = (
        (r"false[- ]?negative|\bFN\b|漏检", "false_negative"),
        (r"false[- ]?positive|\bFP\b|误检", "false_positive"),
        (r"small[- ]?object|小目标", "small_object"),
        (r"locali[sz]ation|定位误差", "localization_error"),
        (r"class[- ]?imbalance|类别不平衡|long[- ]?tail|长尾", "class_imbalance"),
        (r"domain[- ]?shift|域偏移", "domain_shift"),
        (r"duplicate prediction|重复预测", "duplicate_prediction"),
    )
    return [name for pattern, name in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def _extract_symptom(text: str) -> str:
    patterns = (
        r"\b(?:when|if)\s+(.+?)(?:[,;。；]|\b(?:then|consider|use|check|prioritize)\b)",
        r"(?:当|如果)(.+?)(?:时|，|。|则)",
        r"((?:AP|mAP|recall|precision|召回率|精确率|小目标).+?(?:low|poor|drops?|低|较差|下降))",
    )
    return _first_capture(text, patterns)


def _extract_cause(text: str) -> str:
    patterns = (
        r"\b(?:because|due to|caused by|likely cause[:：]?)\s+(.+?)(?:[.;。；]|$)",
        r"(?:由于|原因(?:是|为|[:：])?)(.+?)(?:[。；;]|$)",
    )
    return _first_capture(text, patterns)


def _extract_evidence_requests(text: str, metrics: list[str], error_facts: list[str]) -> list[str]:
    requests: list[str] = []
    patterns = (
        r"\b(?:need|requires?|check|measure|inspect|evidence[:：]?)\s+(.+?)(?:[.;。；]|$)",
        r"(?:需要|检查|测量|分析)(.+?)(?:[。；;]|$)",
    )
    captured = _first_capture(text, patterns)
    if captured != "unknown":
        requests.append(captured)
    requests.extend(metric for metric in metrics if metric not in requests)
    requests.extend(f"error_fact:{item}" for item in error_facts if f"error_fact:{item}" not in requests)
    return requests


def _explicit_component_ids(text: str, component_ids: list[str]) -> list[str]:
    normalized_text = normalize_component_id(text)
    matched = []
    for component_id in component_ids:
        normalized_id = normalize_component_id(component_id)
        if normalized_id and re.search(rf"(?:^|_){re.escape(normalized_id)}(?:_|$)", normalized_text):
            matched.append(component_id)
    return sorted(set(matched))


def _confidence(text: str, metrics: list[str], error_facts: list[str]) -> float:
    score = 0.45
    if metrics:
        score += 0.15
    if error_facts:
        score += 0.15
    if re.search(r"because|due to|由于|原因", text, flags=re.IGNORECASE):
        score += 0.1
    return min(score, 0.85)


def _first_capture(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ,;，；。")
            if value:
                return value
    return "unknown"


__all__ = [
    "HarnessHintParseResult",
    "HarnessHintParser",
    "PaperDiagnosticHint",
    "extract_error_facts",
    "extract_target_metrics",
    "sanitize_research_text",
]
