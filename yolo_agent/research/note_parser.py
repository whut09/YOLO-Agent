"""Offline, evidence-grounded parsing of paper summaries and Markdown notes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.component_aliases import normalize_component_id
from yolo_agent.research.harness_hint_parser import (
    HarnessHintParser,
    PaperDiagnosticHint,
    sanitize_research_text,
)
from yolo_agent.research.llm_paper_analyzer import ResearchDecisionRecord
from yolo_agent.research.schemas import ComponentTaxonomy, PaperRecord


UnknownValue = str | dict[str, Any] | list[Any]


class PaperMethodClaim(BaseModel):
    """A method claim copied only from explicit paper/catalog text."""

    model_config = ConfigDict(extra="forbid")

    method_name: str = "unknown"
    component_ids: list[str] = Field(default_factory=list)
    insertion_point: str = "unknown"
    changed_variables: list[str] = Field(default_factory=list)
    baseline_description: str = "unknown"
    reported_delta: dict[str, float | str] = Field(default_factory=dict)
    dataset: str = "unknown"
    model_family: str = "unknown"
    training_cost: UnknownValue = "unknown"
    inference_cost: UnknownValue = "unknown"
    limitation: str = "unknown"
    formulae: list[str] = Field(default_factory=list)
    source_location: str
    evidence_level: Literal["paper_claim"] = "paper_claim"


class PaperLimitation(BaseModel):
    """An explicitly stated limitation, not a local risk assessment."""

    model_config = ConfigDict(extra="forbid")

    limitation: str
    affected_component_ids: list[str] = Field(default_factory=list)
    source_location: str
    evidence_level: Literal["paper_claim"] = "paper_claim"


class PaperAblationHint(BaseModel):
    """A paper-reported comparison extracted from prose or a Markdown table."""

    model_config = ConfigDict(extra="forbid")

    comparison: str = "unknown"
    variants: list[str] = Field(default_factory=list)
    changed_variables: list[str] = Field(default_factory=list)
    reported_results: dict[str, str] = Field(default_factory=dict)
    dataset: str = "unknown"
    source_location: str
    evidence_level: Literal["paper_claim"] = "paper_claim"


class PaperEvidenceClaim(BaseModel):
    """Formula, metric, or quoted result retained as a paper claim."""

    model_config = ConfigDict(extra="forbid")

    claim_type: Literal["metric", "formula", "result"]
    content: str
    source_location: str
    evidence_level: Literal["paper_claim"] = "paper_claim"


class PaperEvidenceSummary(BaseModel):
    """Structured paper evidence used as prior context, never local evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_evidence_summary.v1"
    paper_id: str
    status: Literal["parsed", "partial", "skipped", "failed"] = "parsed"
    method_claims: list[PaperMethodClaim] = Field(default_factory=list)
    diagnostic_hints: list[PaperDiagnosticHint] = Field(default_factory=list)
    limitations: list[PaperLimitation] = Field(default_factory=list)
    ablation_hints: list[PaperAblationHint] = Field(default_factory=list)
    explicit_claims: list[PaperEvidenceClaim] = Field(default_factory=list)
    source_locations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_level: Literal["paper_claim"] = "paper_claim"


PaperEvidenceEnricher = Callable[[PaperRecord, PaperEvidenceSummary], PaperEvidenceSummary]


class PaperNoteParser:
    """Parse summary, harness hints, and an optional local Markdown note."""

    def __init__(
        self,
        *,
        ledger_path: Path | str | None = None,
        llm_enricher: PaperEvidenceEnricher | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.llm_enricher = llm_enricher
        self.harness_parser = HarnessHintParser()

    def parse(
        self,
        *,
        paper: PaperRecord,
        taxonomy: ComponentTaxonomy,
        note_path: Path | str | None = None,
        note_text: str | None = None,
    ) -> PaperEvidenceSummary:
        warnings: list[str] = []
        source_locations: list[str] = []
        try:
            summary_text, summary_warnings = sanitize_research_text(paper.abstract)
            warnings.extend(f"summary:{warning}" for warning in summary_warnings)
            if summary_text:
                source_locations.append("summary")

            note_content = ""
            if note_text is not None:
                note_content, note_warnings = sanitize_research_text(note_text)
                warnings.extend(f"note:{warning}" for warning in note_warnings)
                if note_content:
                    source_locations.append("note")
            elif note_path is not None:
                path = Path(note_path)
                try:
                    raw = path.read_bytes()
                    note_content, note_warnings = sanitize_research_text(raw.decode("utf-8-sig", errors="replace"))
                    warnings.extend(f"note:{warning}" for warning in note_warnings)
                    if note_content:
                        source_locations.append("note")
                except FileNotFoundError:
                    warnings.append("note_read_failed:not_found")
                except OSError as exc:
                    warnings.append(f"note_read_failed:{type(exc).__name__}")
            elif paper.provenance and paper.provenance.original_note_path:
                warnings.append("note_path_requires_catalog_root")

            method_claims, explicit_claims = _parse_method_and_claims(
                paper,
                summary_text,
                note_content,
                source_locations,
            )
            limitations = _parse_limitations(paper, summary_text, note_content)
            ablations = _parse_ablation_tables(paper, note_content)
            hint_result = self.harness_parser.parse(paper, taxonomy)
            warnings.extend(hint_result.warnings)
            source_locations.extend(item.source_location for item in hint_result.hints)
            result = PaperEvidenceSummary(
                paper_id=paper.paper_id,
                status="parsed" if source_locations else "skipped",
                method_claims=method_claims,
                diagnostic_hints=hint_result.hints,
                limitations=limitations,
                ablation_hints=ablations,
                explicit_claims=explicit_claims,
                source_locations=sorted(set(source_locations)),
                warnings=warnings,
            )
            if warnings and result.status == "parsed":
                result.status = "partial"
            if self.llm_enricher is not None:
                try:
                    enriched = self.llm_enricher(paper, result)
                    result = _accept_enrichment(result, enriched, summary_text + "\n" + note_content)
                except Exception as exc:
                    result.warnings.append(f"llm_enrichment_failed:{exc}")
                    result.status = "partial"
            source_sha256 = hashlib.sha256(f"{summary_text}\n{note_content}".encode("utf-8")).hexdigest()
            return self._finish(paper, result, source_sha256=source_sha256)
        except Exception as exc:
            result = PaperEvidenceSummary(
                paper_id=paper.paper_id,
                status="failed",
                source_locations=source_locations,
                warnings=[*warnings, f"parser_failed:{exc}"],
            )
            return self._finish(paper, result, source_sha256="unavailable")

    def _finish(
        self,
        paper: PaperRecord,
        result: PaperEvidenceSummary,
        *,
        source_sha256: str,
    ) -> PaperEvidenceSummary:
        if self.ledger_path is None:
            return result
        payload = {
            "paper": paper.model_dump(mode="json"),
            "output_schema": result.schema_version,
            "parser": "rules",
            "source_sha256": source_sha256,
        }
        record = ResearchDecisionRecord(
            paper_id=paper.paper_id,
            decision_type="paper_evidence_extraction",
            status=result.status,
            prompt_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            input_summary={
                "title": paper.title,
                "summary_present": bool(paper.abstract),
                "harness_hint_count": len(paper.provenance.original_harness_hints) if paper.provenance else 0,
                "note_path": paper.provenance.original_note_path if paper.provenance else None,
                "source_sha256": source_sha256,
            },
            model_metadata={"provider": "rules", "model": "none"},
            output=result.model_dump(mode="json"),
            parse_warnings=result.warnings,
        )
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as exc:
            result.warnings.append(f"decision_ledger_write_failed:{type(exc).__name__}")
            if result.status == "parsed":
                result.status = "partial"
        return result


def _parse_method_and_claims(
    paper: PaperRecord,
    summary: str,
    note: str,
    source_locations: list[str],
) -> tuple[list[PaperMethodClaim], list[PaperEvidenceClaim]]:
    claims: list[PaperEvidenceClaim] = []
    methods: list[PaperMethodClaim] = []
    for location, text in (("summary", summary), ("note", note)):
        if not text:
            continue
        metrics = _extract_metric_claims(text, location)
        formulae = _extract_formulae(text, location)
        claims.extend(metrics)
        claims.extend(formulae)
        component_ids = _explicit_component_ids(text, paper.component_ids)
        method_name = _method_name(text)
        if method_name != "unknown" or component_ids or metrics or formulae:
            methods.append(PaperMethodClaim(
                method_name=method_name,
                component_ids=component_ids,
                insertion_point=_insertion_point(text),
                changed_variables=_changed_variables(text),
                baseline_description=_extract_field(text, (r"baseline\s*[:：]\s*(.+?)(?:[.;。；]|$)", r"基线(?:为|是|[:：])(.+?)(?:[。；;]|$)")),
                reported_delta=_reported_delta(text),
                dataset=_explicit_dataset(text, paper),
                model_family=_explicit_model_family(text, paper),
                training_cost=_explicit_cost(text, "training"),
                inference_cost=_explicit_cost(text, "inference"),
                limitation=_extract_limitation_text(text),
                formulae=[item.content for item in formulae],
                source_location=location,
            ))
    return methods, claims


def _parse_limitations(paper: PaperRecord, summary: str, note: str) -> list[PaperLimitation]:
    del paper
    result: list[PaperLimitation] = []
    for location, text in (("summary", summary), ("note", note)):
        for sentence in _sentences(text):
            if re.search(r"limitation|limited|cannot|fails?|however|但|局限|限制|不足|缺点", sentence, flags=re.IGNORECASE):
                result.append(PaperLimitation(limitation=sentence, source_location=location))
    return result


def _parse_ablation_tables(paper: PaperRecord, note: str) -> list[PaperAblationHint]:
    result: list[PaperAblationHint] = []
    lines = note.splitlines()
    table_index = 0
    index = 0
    while index < len(lines) - 1:
        if "|" not in lines[index] or "|" not in lines[index + 1] or not re.search(r"\|?\s*:?-{3,}", lines[index + 1]):
            index += 1
            continue
        headers = _table_cells(lines[index])
        table_index += 1
        row_index = index + 2
        while row_index < len(lines) and "|" in lines[row_index]:
            cells = _table_cells(lines[row_index])
            if cells:
                values = {headers[pos]: cells[pos] for pos in range(min(len(headers), len(cells)))}
                variants = [value for key, value in values.items() if re.search(r"variant|method|setting|方案|配置|模型", key, flags=re.IGNORECASE)]
                results = {key: value for key, value in values.items() if _contains_number(value)}
                result.append(PaperAblationHint(
                    comparison="; ".join(f"{key}={value}" for key, value in values.items()) or "unknown",
                    variants=variants,
                    changed_variables=_changed_variables(" ".join(cells)),
                    reported_results=results,
                    dataset=_explicit_dataset(" ".join(cells), paper),
                    source_location=f"note:table:{table_index}:row:{row_index - index - 1}",
                ))
            row_index += 1
        index = row_index
    return result


def _extract_metric_claims(text: str, location: str) -> list[PaperEvidenceClaim]:
    result: list[PaperEvidenceClaim] = []
    pattern = r"\b(?:mAP(?:50[-_:]?95)?|AP(?:[_ -]?(?:small|medium|large|s|m|l))?|precision|recall|latency(?:_ms)?)\b\s*(?:[:=]|is|为)?\s*([+-]?\d+(?:\.\d+)?%?)"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        result.append(PaperEvidenceClaim(
            claim_type="metric",
            content=f"{match.group(0)}",
            source_location=location,
        ))
    return result


def _extract_formulae(text: str, location: str) -> list[PaperEvidenceClaim]:
    formulae: list[str] = []
    formulae.extend(re.findall(r"\$([^$]+)\$", text))
    formulae.extend(re.findall(r"\\\((.+?)\\\)", text))
    return [PaperEvidenceClaim(claim_type="formula", content=item.strip(), source_location=location) for item in formulae if item.strip()]


def _reported_delta(text: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    pattern = r"\b(mAP(?:50[-_:]?95)?|AP(?:[_ -]?(?:small|medium|large|s|m|l))?|precision|recall)\b[^\n]{0,20}?([+-]\d+(?:\.\d+)?%?)"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        key = _metric_name(match.group(1))
        value = match.group(2)
        result[key] = float(value.rstrip("%")) if value.rstrip("%").replace(".", "", 1).lstrip("+").isdigit() and "%" not in value else value
    return result


def _method_name(text: str) -> str:
    return _extract_field(text, (
        r"(?:we\s+)?propose\s+(?:a|an|the)?\s*(.+?)(?:\s+(?:method|module|framework)\b|[,.;:]|$)",
        r"(?:提出|提出了)(.+?)(?:方法|模块|框架|，|。|；|;|$)",
    ))


def _insertion_point(text: str) -> str:
    patterns = (
        (r"\b(?:in|at|into)\s+the\s+(backbone|neck|head|loss|assigner|sampler|inference)\b", lambda m: m.group(1)),
        (r"(?:在|插入|应用于)(?:模型的)?(骨干网络|颈部|检测头|损失|分配器|采样器|推理阶段)", lambda m: m.group(1)),
    )
    for pattern, convert in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return convert(match)
    return "unknown"


def _changed_variables(text: str) -> list[str]:
    terms = (
        (r"\b(?:backbone|neck|head|loss|assigner|matching|sampler|sampling|augmentation|optimizer|scheduler|threshold|nms|inference|imgsz|batch)\b", None),
        (r"(?:骨干|颈部|检测头|损失|分配|匹配|采样|增强|优化器|调度器|阈值|推理|输入尺寸|批大小)", None),
    )
    result: list[str] = []
    for pattern, _ in terms:
        result.extend(match.group(0).lower() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return sorted(set(result))


def _explicit_dataset(text: str, paper: PaperRecord) -> str:
    for dataset in paper.datasets:
        if dataset and re.search(re.escape(dataset), text, flags=re.IGNORECASE):
            return dataset
    match = re.search(r"\b(?:on|using|dataset[:：])\s*([A-Z][A-Za-z0-9_-]{2,})", text)
    return match.group(1) if match else "unknown"


def _explicit_model_family(text: str, paper: PaperRecord) -> str:
    if paper.detector_family and re.search(re.escape(paper.detector_family), text, flags=re.IGNORECASE):
        return paper.detector_family
    match = re.search(r"\b(YOLO\d*|DETR|Faster R-CNN|RetinaNet|FCOS)\b", text, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _explicit_cost(text: str, kind: str) -> UnknownValue:
    match = re.search(rf"\b{kind}\s+cost\s*[:：=]\s*([^.;。；]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "unknown"


def _extract_limitation_text(text: str) -> str:
    for sentence in _sentences(text):
        if re.search(r"limitation|limited|cannot|fails?|however|但|局限|限制|不足|缺点", sentence, flags=re.IGNORECASE):
            return sentence
    return "unknown"


def _explicit_component_ids(text: str, component_ids: list[str]) -> list[str]:
    normalized_text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    text_tokens = {token for token in normalized_text.split("_") if token}
    matched: set[str] = set()
    for item in component_ids:
        normalized = normalize_component_id(item)
        if normalized and normalized in normalized_text:
            matched.add(item)
            continue
        component_tokens = [token for token in normalized.split("_") if token]
        if component_tokens and all(_token_explicitly_present(token, text_tokens) for token in component_tokens):
            matched.add(item)
    return sorted(matched)


def _token_explicitly_present(token: str, text_tokens: set[str]) -> bool:
    if token in text_tokens:
        return True
    if len(token) < 6:
        return False
    stem = token[:5]
    return any(candidate.startswith(stem) for candidate in text_tokens)


def _metric_name(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"map", "map50_95", "map50_95"}:
        return "map50_95" if "95" in normalized else "map"
    return {"aps": "ap_small", "apm": "ap_medium", "apl": "ap_large"}.get(normalized, normalized)


def _extract_field(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ,;，；。")
            if value:
                return value
    return "unknown"


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?。！？；;])\s+|\n+", text) if item.strip()]


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _contains_number(value: str) -> bool:
    return bool(re.search(r"[+-]?\d+(?:\.\d+)?%?", value))


def _accept_enrichment(
    original: PaperEvidenceSummary,
    enriched: PaperEvidenceSummary,
    source_text: str,
) -> PaperEvidenceSummary:
    """Accept optional LLM additions only when they remain source-grounded."""
    if enriched.paper_id != original.paper_id:
        raise ValueError("llm enrichment changed paper_id")
    for location in enriched.source_locations:
        if not location.strip():
            raise ValueError("llm enrichment contains empty source location")
    if not source_text.strip() and enriched.method_claims:
        raise ValueError("llm enrichment produced claims without source text")
    for field in ("method_claims", "diagnostic_hints", "limitations", "ablation_hints", "explicit_claims"):
        original_items = {
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for item in getattr(original, field)
        }
        enriched_items = {
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for item in getattr(enriched, field)
        }
        if not enriched_items.issubset(original_items):
            raise ValueError(f"llm enrichment added ungrounded {field}")
    original.warnings = list(dict.fromkeys([*original.warnings, *enriched.warnings]))
    return original


__all__ = [
    "PaperAblationHint",
    "PaperEvidenceClaim",
    "PaperEvidenceSummary",
    "PaperLimitation",
    "PaperMethodClaim",
    "PaperNoteParser",
]
