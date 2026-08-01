"""Deterministic extraction of paper method boundaries from offline inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from yolo_agent.research.component_aliases import (
    ComponentAliasResolver,
    normalize_component_id,
)
from yolo_agent.research.paper_method_evidence import (
    MethodEvidenceConfidence,
    MethodEvidenceField,
    MethodEvidenceSource,
    PaperMethodEvidenceObservation,
    PaperMethodEvidenceProfile,
)
from yolo_agent.research.schemas import PaperRecord


@dataclass(frozen=True)
class _Rule:
    field_name: MethodEvidenceField
    value: str | bool
    patterns: tuple[str, ...]
    confidence: MethodEvidenceConfidence = "high"


_RULES: tuple[_Rule, ...] = (
    _Rule("canonical_mechanism", "assigner.optimal_transport", (r"\bOTA\b",)),
    _Rule("canonical_mechanism", "assigner.dynamic_smooth_label", (r"\bDSLA\b",)),
    _Rule("canonical_mechanism", "inference.sahi_slicing", (r"\bSAHI\b",)),
    _Rule("method_family", "sampling", (r"\b(?:over|re)?sampl(?:e|ing)\b", r"\bimage weights?\b")),
    _Rule("method_family", "distillation", (r"\b(?:knowledge|feature|logits?|localization) distillation\b", r"\bteacher[- ]student\b")),
    _Rule("method_family", "quality_alignment", (r"\bclassification[- /]localization (?:alignment|correlation)\b", r"\bquality[- ]aware (?:loss|score|target)\b")),
    _Rule("method_family", "multi_scale_fusion", (r"\bmulti[- ]scale (?:feature )?fusion\b", r"\bcross[- ]scale fusion\b")),
    _Rule("method_family", "assignment", (r"\b(?:label|task[- ]aligned|optimal transport) assign(?:er|ment)\b", r"\bOTA\b", r"\bDSLA\b")),
    _Rule("method_family", "sliced_inference", (r"\bsliced? inference\b", r"\btiled? inference\b", r"\bSAHI\b")),
    _Rule("insertion_point", "train_dataloader_sampler", (r"\b(?:training |train )?(?:data ?loader|sampler|image sampling)\b",)),
    _Rule("insertion_point", "trainer_loss", (r"\b(?:auxiliary|additional|training) loss\b", r"\bloss function\b")),
    _Rule("insertion_point", "detection_head", (r"\bdetection head\b", r"\bprediction head\b")),
    _Rule("insertion_point", "neck_feature_pyramid", (r"\b(?:feature pyramid|neck|cross[- ]scale fusion)\b",)),
    _Rule("insertion_point", "one_to_many_assignment", (r"\b(?:positive sample|label|target) assign(?:er|ment)\b",)),
    _Rule("insertion_point", "inference_policy", (r"\b(?:sliced?|tiled?) inference\b", r"\btest[- ]time slicing\b")),
    _Rule("changed_variable", "data.sampling_policy", (r"\b(?:sampling policy|sampling probability|image weights?|oversampling ratio)\b",)),
    _Rule("changed_variable", "loss.auxiliary.weight", (r"\b(?:auxiliary|additional) loss(?: weight)?\b",)),
    _Rule("changed_variable", "model.head", (r"\b(?:adds?|replaces?|modifies?|introduces?) (?:a |the )?(?:detection|prediction|P2) head\b",)),
    _Rule("changed_variable", "model.neck", (r"\b(?:adds?|replaces?|modifies?|introduces?) (?:a |the )?(?:neck|feature pyramid|multi[- ]scale fusion)\b",)),
    _Rule("changed_variable", "train.assigner", (r"\b(?:replaces?|modifies?|uses?|introduces?) (?:a |the )?(?:label|target|task[- ]aligned|optimal transport) assign(?:er|ment)\b",)),
    _Rule("changed_variable", "inference.slicing_policy", (r"\b(?:slice size|tile size|slice overlap|merge policy)\b",)),
    _Rule("detector_family", "yolo", (r"\bYOLO(?:v?\d+|X|26)?\b",)),
    _Rule("detector_family", "transformer_detector", (r"\b(?:DETR|transformer detector|object queries?)\b",)),
    _Rule("detector_family", "two_stage", (r"\b(?:Faster R-CNN|Cascade R-CNN|two[- ]stage detector)\b",)),
    _Rule("detector_family", "one_stage", (r"\b(?:FCOS|RetinaNet|one[- ]stage detector)\b",)),
    _Rule("component_type", "loss", (r"\b(?:auxiliary |additional )?loss\b",)),
    _Rule("component_type", "head", (r"\b(?:detection|prediction|P2) head\b",)),
    _Rule("component_type", "neck", (r"\b(?:neck|feature pyramid|multi[- ]scale fusion)\b",)),
    _Rule("component_type", "data", (r"\b(?:sampler|oversampling|image weights?|data augmentation)\b",)),
    _Rule("component_type", "assigner", (r"\bassign(?:er|ment)\b", r"\bOTA\b", r"\bDSLA\b")),
    _Rule("component_type", "inference", (r"\b(?:sliced?|tiled?) inference\b", r"\bSAHI\b")),
    _Rule("component_type", "distillation", (r"\bdistillation\b", r"\bteacher[- ]student\b")),
    _Rule("training_only", True, (r"\btraining[- ]only\b", r"\bused only during training\b")),
    _Rule("inference_changed", True, (r"\b(?:changes?|modifies?) (?:the )?inference\b", r"\b(?:sliced?|tiled?) inference\b")),
    _Rule("inference_changed", False, (r"\bno (?:change|overhead) (?:at|during|to) inference\b", r"\binference architecture remains unchanged\b")),
    _Rule("compatibility_constraint", "requires_one_to_many_assignment", (r"\brequires? (?:the )?one[- ]to[- ]many assignment\b",)),
    _Rule("compatibility_constraint", "changes_inference_protocol", (r"\b(?:sliced?|tiled?) inference\b",)),
    _Rule("compatibility_constraint", "requires_teacher_checkpoint", (r"\bpre[- ]trained teacher\b", r"\bteacher checkpoint\b")),
    _Rule("method_family", "sampling", (r"(?:小目标|长尾|类别不均衡).{0,12}(?:采样|过采样)", r"图像采样权重")),
    _Rule("method_family", "distillation", (r"知识蒸馏|特征蒸馏|定位蒸馏|师生(?:网络|模型)",)),
    _Rule("method_family", "quality_alignment", (r"分类.{0,6}定位.{0,6}(?:对齐|相关)", r"质量感知(?:损失|分数|目标)")),
    _Rule("method_family", "multi_scale_fusion", (r"多尺度(?:特征)?融合|跨尺度融合",)),
    _Rule("method_family", "assignment", (r"标签分配|任务对齐分配|最优传输分配",)),
    _Rule("method_family", "sliced_inference", (r"切片推理|分块推理",)),
    _Rule("insertion_point", "train_dataloader_sampler", (r"训练(?:数据加载器|采样器)|图像采样",)),
    _Rule("insertion_point", "trainer_loss", (r"辅助损失|附加损失|损失函数",)),
    _Rule("insertion_point", "detection_head", (r"检测头|预测头|P2 ?头",)),
    _Rule("insertion_point", "neck_feature_pyramid", (r"颈部网络|特征金字塔|多尺度(?:特征)?融合",)),
    _Rule("insertion_point", "one_to_many_assignment", (r"(?:正样本|标签|目标)分配",)),
    _Rule("insertion_point", "inference_policy", (r"切片推理|分块推理|测试时切片",)),
    _Rule("changed_variable", "data.sampling_policy", (r"采样策略|采样概率|图像权重|过采样比例",)),
    _Rule("changed_variable", "loss.auxiliary.weight", (r"辅助损失(?:权重)?|附加损失(?:权重)?",)),
    _Rule("changed_variable", "model.head", (r"(?:增加|替换|修改|引入).{0,4}(?:检测头|预测头|P2 ?头)",)),
    _Rule("changed_variable", "model.neck", (r"(?:增加|替换|修改|引入).{0,4}(?:颈部网络|特征金字塔|多尺度融合)",)),
    _Rule("changed_variable", "train.assigner", (r"(?:替换|修改|使用|引入).{0,4}(?:标签|目标|任务对齐|最优传输)分配",)),
    _Rule("changed_variable", "inference.slicing_policy", (r"切片大小|分块大小|切片重叠|合并策略",)),
    _Rule("component_type", "loss", (r"损失(?:函数|项)?",)),
    _Rule("component_type", "head", (r"检测头|预测头|P2 ?头",)),
    _Rule("component_type", "neck", (r"颈部网络|特征金字塔|多尺度(?:特征)?融合",)),
    _Rule("component_type", "data", (r"采样器|过采样|图像权重|数据增强",)),
    _Rule("component_type", "assigner", (r"分配器|标签分配|目标分配",)),
    _Rule("component_type", "inference", (r"切片推理|分块推理",)),
    _Rule("component_type", "distillation", (r"蒸馏|师生(?:网络|模型)",)),
    _Rule("training_only", True, (r"仅用于训练|只在训练(?:阶段)?使用",)),
    _Rule("inference_changed", True, (r"改变推理(?:流程|协议)|切片推理|分块推理",)),
    _Rule("inference_changed", False, (r"推理(?:结构|流程)不变|不增加推理开销",)),
)


_HOOKS_BY_INSERTION_POINT: dict[str, tuple[str, ...]] = {
    "train_dataloader_sampler": ("build_train_dataset", "build_train_dataloader"),
    "trainer_loss": ("build_criterion", "compute_loss"),
    "detection_head": ("build_model",),
    "neck_feature_pyramid": ("build_model",),
    "one_to_many_assignment": ("build_criterion", "compute_loss"),
    "inference_policy": ("build_validator",),
}


class PaperMethodEvidenceExtractor:
    """Extract auditable method evidence without network access or benchmark inference."""

    def __init__(self, resolver: ComponentAliasResolver) -> None:
        self.resolver = resolver
        self._mechanism_terms = _mechanism_terms(resolver)

    def extract(
        self,
        paper: PaperRecord,
        *,
        evidence_summary: Any | None = None,
        cached_metadata: Iterable[tuple[MethodEvidenceSource, str, str]] = (),
    ) -> PaperMethodEvidenceProfile:
        observations: list[PaperMethodEvidenceObservation] = []
        sources = _paper_sources(paper, evidence_summary, cached_metadata)
        for source, location, text in sources:
            if not text.strip():
                continue
            observations.extend(self._extract_mechanisms(source, location, text))
            observations.extend(_extract_rules(source, location, text))
        observations.extend(_derived_runtime_hooks(observations))
        observations = _deduplicate(observations)
        authorizing = [item for item in observations if item.authorizes_method_profile]
        mechanisms = _values(observations, "canonical_mechanism")
        boundary_fields = {
            item.field_name
            for item in authorizing
            if item.field_name in {
                "insertion_point",
                "changed_variable",
                "component_type",
                "required_runtime_hook",
            }
        }
        authorized = bool(
            any(
                item.field_name == "canonical_mechanism"
                and item.authorizes_method_profile
                for item in observations
            )
            and boundary_fields
        )
        reasons: list[str] = []
        if not mechanisms:
            reasons.append("canonical_mechanism_not_explicit")
        if not boundary_fields:
            reasons.append("method_boundary_not_explicit")
        if mechanisms and not any(
            item.field_name == "canonical_mechanism"
            and item.authorizes_method_profile
            for item in observations
        ):
            reasons.append("mechanism_is_prior_only")
        if authorized:
            reasons = ["explicit_local_mechanism_and_method_boundary"]
        profile = PaperMethodEvidenceProfile(
            paper_id=paper.paper_id,
            method_families=_values(observations, "method_family"),
            canonical_mechanisms=mechanisms,
            insertion_points=_values(observations, "insertion_point"),
            changed_variables=_values(observations, "changed_variable"),
            detector_families=_values(observations, "detector_family"),
            component_types=_values(observations, "component_type"),  # type: ignore[arg-type]
            training_only=_optional_bool(observations, "training_only"),
            inference_changed=_optional_bool(observations, "inference_changed"),
            compatibility_constraints=_values(
                observations, "compatibility_constraint"
            ),
            required_runtime_hooks=_values(observations, "required_runtime_hook"),
            observations=observations,
            authorizes_method_profile=authorized,
            authorization_reasons=reasons,
        )
        return profile.with_hash()

    def _extract_mechanisms(
        self,
        source: MethodEvidenceSource,
        location: str,
        text: str,
    ) -> list[PaperMethodEvidenceObservation]:
        normalized = normalize_component_id(text)
        result: list[PaperMethodEvidenceObservation] = []
        for term in self._mechanism_terms:
            if not re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", normalized):
                continue
            resolution = self.resolver.resolve(term)
            for mapping in resolution.mappings:
                confidence: MethodEvidenceConfidence = (
                    "low" if source == "title" else "medium"
                    if source in {"harness_hint", "category", "official_code_metadata"}
                    else "high"
                )
                result.append(_observation(
                    "canonical_mechanism",
                    mapping.canonical_component_id,
                    source,
                    location,
                    confidence,
                ))
        return result


def _paper_sources(
    paper: PaperRecord,
    evidence_summary: Any | None,
    cached_metadata: Iterable[tuple[MethodEvidenceSource, str, str]],
) -> list[tuple[MethodEvidenceSource, str, str]]:
    sources: list[tuple[MethodEvidenceSource, str, str]] = [
        ("title", "paper_record.title", paper.title),
        ("summary", "summary", paper.abstract),
    ]
    provenance = paper.provenance
    if provenance is not None:
        if provenance.original_category:
            sources.append(("category", "paper_record.provenance.original_category", provenance.original_category))
        for index, hint in enumerate(provenance.original_harness_hints):
            sources.append(("harness_hint", f"harness_hints[{index}]", hint))
    if paper.detector_family:
        sources.append(("category", "paper_record.detector_family", paper.detector_family))
    if paper.framework:
        sources.append(("official_code_metadata", "paper_record.framework", paper.framework))
    if evidence_summary is not None:
        for claim in getattr(evidence_summary, "method_claims", []) or []:
            source: MethodEvidenceSource = (
                "note" if str(claim.source_location).startswith("note") else "summary"
            )
            text = " ".join([
                claim.method_name,
                *claim.component_ids,
                claim.insertion_point,
                *claim.changed_variables,
                claim.model_family,
                claim.limitation,
            ])
            sources.append((source, str(claim.source_location), text))
    sources.extend(cached_metadata)
    return sources


def _extract_rules(
    source: MethodEvidenceSource,
    location: str,
    text: str,
) -> list[PaperMethodEvidenceObservation]:
    result: list[PaperMethodEvidenceObservation] = []
    for rule in _RULES:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule.patterns):
            confidence = "low" if source == "title" else rule.confidence
            result.append(_observation(
                rule.field_name,
                rule.value,
                source,
                location,
                confidence,
            ))
    return result


def _observation(
    field_name: MethodEvidenceField,
    value: str | bool,
    source: MethodEvidenceSource,
    source_location: str,
    confidence: MethodEvidenceConfidence,
) -> PaperMethodEvidenceObservation:
    return PaperMethodEvidenceObservation(
        field_name=field_name,
        value=value,
        source=source,
        source_location=source_location,
        confidence=confidence,
        authorizes_method_profile=(
            source not in {"title", "harness_hint", "category"}
            and confidence != "low"
        ),
    )


def _derived_runtime_hooks(
    observations: list[PaperMethodEvidenceObservation],
) -> list[PaperMethodEvidenceObservation]:
    result: list[PaperMethodEvidenceObservation] = []
    for item in observations:
        if item.field_name != "insertion_point" or not isinstance(item.value, str):
            continue
        for hook in _HOOKS_BY_INSERTION_POINT.get(item.value, ()):
            result.append(PaperMethodEvidenceObservation(
                field_name="required_runtime_hook",
                value=hook,
                source=item.source,
                source_location=item.source_location,
                confidence=item.confidence,
                authorizes_method_profile=item.authorizes_method_profile,
            ))
    return result


def _mechanism_terms(resolver: ComponentAliasResolver) -> list[str]:
    terms: set[str] = set()
    for definition in resolver.config.canonical_components:
        terms.update(definition.aliases)
        terms.update(definition.semantic_aliases)
    for definition in resolver.config.compound_aliases:
        terms.update(definition.aliases)
        terms.update(definition.semantic_aliases)
    return sorted(
        {normalize_component_id(term) for term in terms if term.strip()},
        key=lambda value: (-len(value), value),
    )


def _deduplicate(
    observations: list[PaperMethodEvidenceObservation],
) -> list[PaperMethodEvidenceObservation]:
    unique = {
        (
            item.field_name,
            str(item.value),
            item.source,
            item.source_location,
            item.confidence,
        ): item
        for item in observations
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            item.field_name,
            str(item.value),
            item.source,
            item.source_location,
        ),
    )


def _values(
    observations: list[PaperMethodEvidenceObservation],
    field_name: MethodEvidenceField,
) -> list[str]:
    return sorted({
        str(item.value)
        for item in observations
        if item.field_name == field_name and isinstance(item.value, str)
    })


def _optional_bool(
    observations: list[PaperMethodEvidenceObservation],
    field_name: MethodEvidenceField,
) -> bool | None:
    values = {
        item.value
        for item in observations
        if item.field_name == field_name and isinstance(item.value, bool)
    }
    if len(values) == 1:
        return next(iter(values))
    return None


__all__ = ["PaperMethodEvidenceExtractor"]
