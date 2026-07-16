"""Explainable, offline classification of research papers for YOLO26 work."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.schemas import Applicability, ComponentCategory, PaperRecord
from yolo_agent.resources import ResourcePaths


Complexity = Literal["low", "medium", "high", "unknown"]
Relevance = Literal["high", "medium", "low", "unknown"]


class PaperClassification(BaseModel):
    """Rule-derived classification and priority for one paper."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "paper_classification.v1"
    paper_id: str
    task_families: list[str] = Field(default_factory=list)
    detector_family: str | None = None
    component_categories: list[ComponentCategory] = Field(default_factory=list)
    target_error_types: list[str] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    likely_yolo26_relevance: Relevance = "unknown"
    implementation_complexity: Complexity = "unknown"
    training_cost: Complexity = "unknown"
    inference_cost: Complexity = "unknown"
    reproduction_risk: Complexity = "unknown"
    priority_score: float = Field(ge=0.0, le=100.0)
    priority_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    applicability: Applicability = "insufficient_information"
    classifier: str = "rules.v1"
    llm_status: Literal["not_used", "available", "used", "failed"] = "not_used"


class PaperClassifierLLM(Protocol):
    """Optional future LLM classifier interface."""

    def classify(self, paper: PaperRecord) -> dict[str, Any]:
        ...


class ResearchPriorityConfig(BaseModel):
    """Weights and keyword rules for deterministic paper prioritization."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "research_priority.v1"
    weights: dict[str, float] = Field(default_factory=dict)
    category_keywords: dict[str, list[str]] = Field(default_factory=dict)
    error_keywords: dict[str, list[str]] = Field(default_factory=dict)
    metric_keywords: dict[str, list[str]] = Field(default_factory=dict)
    detector_family_keywords: dict[str, list[str]] = Field(default_factory=dict)
    incompatibility_keywords: list[str] = Field(default_factory=list)
    separate_family_keywords: list[str] = Field(default_factory=list)
    direct_adapter_keywords: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "ResearchPriorityConfig":
        config_path = ResourcePaths.RESEARCH_PRIORITY if path is None else path
        data = yaml.safe_load(open(config_path, encoding="utf-8-sig")) or {}
        return cls.model_validate(data)


class PaperClassifier:
    """Classify papers using only local metadata and configurable rules."""

    def __init__(self, config: ResearchPriorityConfig | None = None) -> None:
        self.config = config or ResearchPriorityConfig.from_yaml()

    def classify(self, paper: PaperRecord, *, llm: PaperClassifierLLM | None = None) -> PaperClassification:
        text = _paper_text(paper)
        categories = _categories(text, self.config.category_keywords)
        errors = _matches(text, self.config.error_keywords)
        metrics = _matches(text, self.config.metric_keywords)
        detector_family = _detector_family(paper, text, self.config.detector_family_keywords)
        breakdown = self._score(paper, text, categories, detector_family)
        applicability, applicability_reason = _applicability(paper, text, categories, self.config)
        relevance_score = 0.0 if applicability == "incompatible" else breakdown["yolo26_compatibility"]
        reasons = [applicability_reason]
        reasons.extend(_classification_reasons(paper, text, categories, detector_family, breakdown))
        result = PaperClassification(
            paper_id=paper.paper_id,
            task_families=list(dict.fromkeys([*paper.task_families, *_task_families(text)])),
            detector_family=detector_family,
            component_categories=categories,
            target_error_types=errors,
            target_metrics=metrics,
            likely_yolo26_relevance=_relevance(relevance_score),
            implementation_complexity=_complexity(breakdown["implementation_cost"]),
            training_cost=_complexity(breakdown["training_cost"]),
            inference_cost=_complexity(breakdown["inference_cost"]),
            reproduction_risk=_complexity(breakdown["reproduction_risk"]),
            priority_score=round(sum(breakdown.values()), 2),
            priority_breakdown=breakdown,
            reasons=reasons,
            applicability=applicability,
        )
        if llm is not None:
            result.llm_status = "available"
        return result

    def classify_many(self, papers: Iterable[PaperRecord]) -> list[PaperClassification]:
        return sorted((self.classify(paper) for paper in papers), key=lambda item: (-item.priority_score, item.paper_id))

    def _score(
        self,
        paper: PaperRecord,
        text: str,
        categories: list[ComponentCategory],
        detector_family: str | None,
    ) -> dict[str, float]:
        weights = self.config.weights
        has_coco = _has_any(text, ["coco", "coco2017", "mscoco"])
        real_time = _has_any(text, ["real-time", "realtime", "real time", "efficient", "latency"])
        ablation = _has_any(text, ["ablation", "ablation study", "消融"])
        latency = any(item.latency_ms is not None for item in paper.benchmarks) or _has_any(text, ["latency", "fps", "throughput"])
        size = _has_any(text, ["model size", "parameters", "params", "flops"])
        official_code = bool(paper.official_code_url)
        permissive_license = _has_any((paper.code_license or "").lower(), ["mit", "apache", "bsd", "cc-by"])
        direct = bool(set(categories) & {"assigner", "matching", "augmentation", "sampling", "distillation", "bbox_regression_loss"})
        separate = detector_family in {"detr", "grounding_dino", "open_vocabulary"} or bool(set(categories) & {"pretraining", "domain_adaptation"})
        return {
            "coco_relevance": weights.get("coco_relevance", 0.0) * (1.0 if has_coco else 0.0),
            "real_time_relevance": weights.get("real_time_relevance", 0.0) * (1.0 if real_time else 0.0),
            "yolo26_compatibility": weights.get("yolo26_compatibility", 0.0) * (1.0 if direct and not separate else 0.45 if real_time else 0.0),
            "official_code": weights.get("official_code", 0.0) * (1.0 if official_code else 0.0),
            "license": weights.get("license", 0.0) * (1.0 if permissive_license else 0.0),
            "ablation_evidence": weights.get("ablation_evidence", 0.0) * (1.0 if ablation else 0.0),
            "latency_evidence": weights.get("latency_evidence", 0.0) * (1.0 if latency else 0.0),
            "size_evidence": weights.get("size_evidence", 0.0) * (1.0 if size else 0.0),
            "single_variable": weights.get("single_variable", 0.0) * (1.0 if direct and len(categories) <= 2 else 0.0),
            "implementation_cost": weights.get("implementation_cost", 0.0) * (0.0 if direct and not separate else 0.6 if categories else 1.0),
            "training_cost": weights.get("training_cost", 0.0) * (0.0 if _has_any(text, ["few-shot", "lightweight", "fine-tuning"]) else 0.5 if direct else 1.0),
            "inference_cost": weights.get("inference_cost", 0.0) * (0.0 if _has_any(text, ["training-only", "training only", "distillation"]) else 0.5 if "attention" in categories else 1.0 if separate else 0.0),
            "reproduction_risk": weights.get("reproduction_risk", 0.0) * (0.0 if official_code and permissive_license else 0.5 if official_code else 1.0),
        }


def _paper_text(paper: PaperRecord) -> str:
    claims = " ".join(
        f"{claim.component_id} {claim.claimed_effect} {' '.join(claim.target_error_types)} {' '.join(claim.target_metrics)}"
        for claim in paper.claimed_effects
    )
    return " ".join([paper.title, paper.abstract, paper.detector_family or "", " ".join(paper.task_families), " ".join(paper.datasets), claims]).casefold()


def _categories(text: str, rules: dict[str, list[str]]) -> list[ComponentCategory]:
    result = [category for category, keywords in rules.items() if _has_any(text, keywords)]
    return [item for item in result if item in ComponentCategory.__args__]  # type: ignore[attr-defined]


def _matches(text: str, rules: dict[str, list[str]]) -> list[str]:
    return [key for key, keywords in rules.items() if _has_any(text, keywords)]


def _detector_family(paper: PaperRecord, text: str, rules: dict[str, list[str]]) -> str | None:
    if paper.detector_family:
        return paper.detector_family
    return next((family for family, keywords in rules.items() if _has_any(text, keywords)), None)


def _task_families(text: str) -> list[str]:
    result = []
    if _has_any(text, ["object detection", "detector", "detection"]):
        result.append("object_detection")
    if _has_any(text, ["open vocabulary", "open-vocabulary", "grounding"]):
        result.append("open_vocabulary_detection")
    return result


def _applicability(paper: PaperRecord, text: str, categories: list[ComponentCategory], config: ResearchPriorityConfig) -> tuple[Applicability, str]:
    if _has_any(text, config.incompatibility_keywords):
        return "incompatible", "Rules detected an incompatible detector or training assumption."
    if _has_any(text, config.separate_family_keywords) or paper.detector_family in {"detr", "grounding_dino", "open_vocabulary"}:
        return "separate_detector_family", "The paper likely requires a separate detector family or graph."
    if paper.official_code_url and categories and _has_any(text, config.direct_adapter_keywords):
        return "direct_adapter_candidate", "The paper exposes a concrete, code-backed component that may be adapted."
    if categories:
        return "recipe_idea_only", "The paper contains a relevant idea but lacks enough adapter evidence."
    return "insufficient_information", "No reliable component category or compatibility signal was found."


def _classification_reasons(paper: PaperRecord, text: str, categories: list[str], family: str | None, breakdown: dict[str, float]) -> list[str]:
    reasons = []
    if breakdown["coco_relevance"] > 0:
        reasons.append("Mentions COCO-like object detection evidence.")
    if breakdown["real_time_relevance"] > 0:
        reasons.append("Contains real-time, efficiency, latency, FPS, or throughput signals.")
    if paper.official_code_url:
        reasons.append("Official code URL is available.")
    if not categories:
        reasons.append("No configured component category matched the local metadata.")
    if family:
        reasons.append(f"Detector family classified as {family}.")
    if _has_any(text, ["nms-free", "nms free", "end-to-end"]):
        reasons.append("End-to-end/NMS-free terminology requires YOLO26 head compatibility checks.")
    return reasons


def _complexity(value: float) -> Complexity:
    return "low" if value <= 0.25 else "medium" if value <= 0.65 else "high" if value <= 1.0 else "unknown"


def _relevance(value: float) -> Relevance:
    return "high" if value >= 5.0 else "medium" if value >= 2.0 else "low" if value > 0 else "unknown"


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords if keyword)


__all__ = ["PaperClassification", "PaperClassifier", "PaperClassifierLLM", "ResearchPriorityConfig"]
