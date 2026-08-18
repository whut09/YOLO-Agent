from __future__ import annotations

from yolo_agent.research.component_aliases import (
    CanonicalComponentDefinition,
    ComponentAliasConfig,
    ComponentAliasResolver,
)
from yolo_agent.research.method_profiles import (
    CanonicalMechanismCoverage,
    CompatibleMechanismCoverage,
    PaperAdaptationGap,
    PaperEvidenceInventory,
    PaperMechanismMapping,
    PaperMethodProfileBuilder,
)
from yolo_agent.research.note_parser import PaperEvidenceSummary, PaperMethodClaim
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.schemas import PaperProvenance


def _resolver() -> ComponentAliasResolver:
    return ComponentAliasResolver.from_yaml()


def _paper(
    paper_id: str,
    component_ids: list[str],
    *,
    applicability: str = "direct_adapter_candidate",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=paper_id,
        year=2025,
        component_ids=component_ids,
        applicability=applicability,  # type: ignore[arg-type]
    )


def test_reuses_one_existing_adapter_and_keeps_profile_paper_only() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("sampling-paper", ["small_object_sampling"])],
    )

    decision = report.decisions[0]
    profile = report.profiles[0]
    assert decision.decision == "reuse_existing_adapter"
    assert decision.reusable_adapter_ids == ["sampling.small_object"]
    assert report.adapter_to_papers == {"sampling.small_object": ["sampling-paper"]}
    assert profile.evidence_level == "paper_prior"
    assert profile.exact_reproduction_claim is False
    assert profile.component_adaptation is True
    assert decision.adaptation_mode == "component_adaptation"
    assert profile.adaptation_mode == "component_adaptation"


def test_new_component_adapter_is_required_when_method_details_are_explicit() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build(
        [
            _paper(
                "deformable-paper",
                ["deformable_attention"],
            )
        ],
        evidence_summaries={
            "deformable-paper": PaperEvidenceSummary(
                paper_id="deformable-paper",
                method_claims=[
                    PaperMethodClaim(
                        method_name="deformable attention",
                        component_ids=["deformable_attention"],
                        insertion_point="neck",
                        changed_variables=["neck.attention"],
                        source_location="note.md:4",
                    )
                ],
            )
        },
    )

    decision = report.decisions[0]
    assert decision.decision == "new_component_adapter"
    assert decision.required_adapter_ids == ["attention.deformable"]
    assert decision.unimplemented_reasons["attention.deformable"]


def test_descriptive_known_component_becomes_method_profile_not_fake_adapter() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("attention-paper", ["deformable_attention"])],
    )

    decision = report.decisions[0]
    assert decision.decision == "new_method_profile"
    assert decision.required_adapter_ids == ["attention.deformable"]
    assert "method_profile_requires_explicit_runtime_contract" in (
        decision.unimplemented_reasons["attention.deformable"]
    )


def test_multiple_canonical_mechanisms_require_coupled_recipe() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build(
        [
            _paper(
                "coupled-paper",
                ["small_object_sampling", "p2_head"],
            )
        ],
    )

    decision = report.decisions[0]
    assert decision.decision == "coupled_recipe"
    assert decision.canonical_component_ids == [
        "head.p2_small_object",
        "sampling.small_object",
    ]


def test_separate_detector_family_and_insufficient_information_are_explicit() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build(
        [
            _paper(
                "detr-paper",
                ["open_vocabulary_detection"],
                applicability="separate_detector_family",
            ),
            _paper("unknown-paper", ["unmapped_method"]),
        ],
    )

    decisions = {item.paper_id: item for item in report.decisions}
    assert decisions["detr-paper"].decision == "separate_detector_family"
    assert decisions["detr-paper"].adaptation_mode == "separate_detector_family"
    assert decisions["unknown-paper"].decision == "insufficient_information"
    assert decisions["unknown-paper"].adaptation_mode == "insufficient_information"
    assert "unresolved_paper_component_alias" in decisions["unknown-paper"].reasons


def test_decision_hash_is_stable_and_alias_config_still_rejects_conflicts() -> None:
    first = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("stable-paper", ["small_object_sampling"])]
    ).decisions[0]
    second = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("stable-paper", ["small_object_sampling"])]
    ).decisions[0]
    assert first.decision_hash == second.decision_hash

    try:
        ComponentAliasConfig(
            canonical_components=[
                CanonicalComponentDefinition(
                    canonical_component_id="a",
                    category="sampling",
                    aliases=["same"],
                    mapping_reason="test",
                ),
                CanonicalComponentDefinition(
                    canonical_component_id="b",
                    category="sampling",
                    aliases=["same"],
                    mapping_reason="test",
                ),
            ]
        )
    except ValueError as exc:
        assert "conflicting component alias" in str(exc)
    else:  # pragma: no cover - pydantic must reject the conflict
        raise AssertionError("conflicting aliases must be rejected")


def test_field_level_adaptation_gap_roundtrips() -> None:
    gap = PaperAdaptationGap(
        field_name="canonical_component_ids",
        reason_code="canonical_component_mapping_required",
        severity="blocking",
        observed_value="unknown_component",
        paper_component_id="unknown_component",
        source_locations=["summary"],
        required_evidence=["curated canonical mechanism mapping"],
    )

    assert gap.severity == "blocking"
    assert gap.model_dump(mode="json")["paper_component_id"] == "unknown_component"


def test_mechanism_coverage_schema_uses_unique_mechanism_denominator() -> None:
    coverage = CompatibleMechanismCoverage(
        referenced_mechanism_count=1,
        compatible_mechanism_count=1,
        potentially_adaptable_mechanism_count=1,
        reusable_adapter_mechanism_count=1,
        runtime_ready_mechanism_count=1,
        compatible_adapter_coverage_ratio=1.0,
        runtime_ready_coverage_ratio=1.0,
        mechanisms=[
            CanonicalMechanismCoverage(
                canonical_component_id="sampling.small_object",
                paper_ids=["paper-a", "paper-b"],
                reference_count=2,
                yolo26_compatibility="compatible",
                implementation_status="smoke_passed",
                reusable_adapter=True,
                runtime_execution_ready=True,
            )
        ],
    )

    assert coverage.referenced_mechanism_count == 1
    assert coverage.mechanisms[0].reference_count == 2


def test_builder_computes_coverage_by_unique_compatible_mechanism() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build([
        _paper("sampling-a", ["small_object_sampling"]),
        _paper("sampling-b", ["small_object_sampling"]),
        _paper("open-vocabulary", ["open_vocabulary_detection"]),
        _paper("unknown", ["broad_detection_task"]),
    ])

    coverage = report.compatible_mechanism_coverage
    assert coverage.referenced_mechanism_count == 2
    assert coverage.potentially_adaptable_mechanism_count == 1
    assert coverage.reusable_adapter_mechanism_count == 1
    assert coverage.compatible_adapter_coverage_ratio == 1.0
    sampling = next(
        item
        for item in coverage.mechanisms
        if item.canonical_component_id == "sampling.small_object"
    )
    assert sampling.paper_ids == ["sampling-a", "sampling-b"]
    assert sampling.reference_count == 2
    assert sampling.priority_family == "small_object"
    assert coverage.priority_family_mechanism_counts["small_object"] == 1


def test_offline_evidence_inventory_is_explicit() -> None:
    inventory = PaperEvidenceInventory(
        summary_available=True,
        summary_source="summary",
        note_available=True,
        note_path="notes/paper.md",
        harness_hint_count=2,
        official_code_available=True,
        code_license_known=False,
        framework_known=True,
        source_locations=["summary", "note", "harness_hints[0]"],
    )

    assert inventory.summary_available is True
    assert inventory.harness_hint_count == 2
    assert inventory.code_license_known is False


def test_profile_freezes_local_evidence_inventory_and_code_metadata() -> None:
    paper = PaperRecord(
        paper_id="evidence-paper",
        title="Evidence paper",
        year=2025,
        abstract="Uses small object sampling.",
        component_ids=["small_object_sampling"],
        official_code_url="https://github.com/owner/sampler",
        code_license="MIT",
        framework="pytorch",
        provenance=PaperProvenance(
            source_repository="local",
            source_path="papers.json",
            source_record_hash="record-hash",
            importer_version="test",
            original_harness_hints=["Measure AP_small."],
            original_note_path="notes/sampler.md",
            abstract_source="summary",
        ),
    )
    summary = PaperEvidenceSummary(
        paper_id=paper.paper_id,
        source_locations=["summary", "note", "harness_hints[0]"],
    )

    profile = PaperMethodProfileBuilder(_resolver()).build(
        [paper], evidence_summaries={paper.paper_id: summary}
    ).profiles[0]

    assert profile.evidence_inventory.summary_source == "summary"
    assert profile.evidence_inventory.note_available is True
    assert profile.evidence_inventory.harness_hint_count == 1
    assert profile.official_code_metadata.repository_slug == "owner/sampler"


def test_profile_adds_only_explicit_summary_mechanisms() -> None:
    paper = PaperRecord(
        paper_id="summary-mechanism",
        title="Summary mechanism",
        year=2025,
        abstract="Uses teacher student distillation for the detector.",
        component_ids=["object_detection"],
    )

    profile = PaperMethodProfileBuilder(_resolver()).build([paper]).profiles[0]

    assert "distillation.yolo26_teacher_student" in profile.canonical_component_ids
    assert any(
        item.source == "summary"
        and item.canonical_component_id == "distillation.yolo26_teacher_student"
        for item in profile.mechanism_evidence
    )


def test_mechanism_mapping_records_full_adapter_chain() -> None:
    mapping = PaperMechanismMapping(
        paper_id="paper",
        profile_id="profile",
        source_term="small_object_sampling",
        source="catalog_component_id",
        source_location="paper_record.component_ids",
        canonical_component_id="sampling.small_object",
        alias_match_type="exact_match",
        yolo26_compatibility="compatible",
        implementation_status="smoke_passed",
        reusable_adapter_id="sampling.small_object",
        adapter_verified=True,
        runtime_execution_ready=True,
    )

    assert mapping.reusable_adapter_id == "sampling.small_object"
    assert mapping.runtime_execution_ready is True


def test_partial_alias_resolution_keeps_proven_mechanism() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build([
        _paper("partial-paper", ["pseudo_iou", "label_assignment"])
    ])

    decision = report.decisions[0]
    assert decision.decision == "reuse_existing_adapter"
    assert decision.reusable_adapter_ids == ["loss.quality.pseudo_iou"]
    assert decision.mechanism_mappings[0].source_term == "pseudo_iou"
    unresolved_gap = next(
        item
        for item in decision.adaptation_gaps
        if item.paper_component_id == "label_assignment"
    )
    assert unresolved_gap.severity == "non_blocking"
    assert decision.unimplemented_reasons["label_assignment"] == [
        "canonical_component_mapping_required"
    ]


def test_generic_distillation_summary_requires_specific_mechanism() -> None:
    paper = PaperRecord(
        paper_id="summary-rescue",
        title="Summary rescue",
        year=2025,
        abstract="Uses teacher student distillation.",
        component_ids=["object_detection"],
    )

    report = PaperMethodProfileBuilder(_resolver()).build([paper])
    decision = report.decisions[0]

    assert decision.decision == "new_method_profile"
    assert decision.canonical_component_ids == [
        "distillation.yolo26_teacher_student"
    ]
    assert decision.reusable_adapter_ids == []
    assert decision.paper_mechanism_resolutions[0].resolved is False
    assert any(item.source == "summary" for item in decision.mechanism_mappings)
    assert decision.exact_reproduction_claim is False
    assert decision.adaptation_mode == "component_adaptation"


def test_specific_distillation_mechanism_selects_its_own_adapter() -> None:
    paper = PaperRecord(
        paper_id="relation-distillation",
        title="Relational transfer",
        year=2025,
        component_ids=["knowledge_distillation"],
    )
    summary = PaperEvidenceSummary(
        paper_id=paper.paper_id,
        method_claims=[PaperMethodClaim(
            method_name="relation distillation",
            component_ids=["relation_distillation"],
            changed_variables=["loss.distillation.relation.weight"],
            insertion_point="intermediate_feature_relations",
            source_location="summary:method",
        )],
    )

    decision = PaperMethodProfileBuilder(_resolver()).build(
        [paper],
        evidence_summaries={paper.paper_id: summary},
    ).decisions[0]

    assert decision.canonical_component_ids == ["distillation.relation"]
    assert decision.decision == "new_component_adapter"
    assert decision.required_adapter_ids == ["distillation.relation"]
    assert decision.paper_mechanism_resolutions[0].paper_specific_mechanism_id == (
        "relation_distillation"
    )


def test_title_only_mechanism_does_not_rescue_generic_catalog_label() -> None:
    paper = PaperRecord(
        paper_id="title-prior-only",
        title="Teacher Student Distillation for Detection",
        year=2025,
        abstract="A general object detection study.",
        component_ids=["object_detection"],
    )

    report = PaperMethodProfileBuilder(_resolver()).build([paper])
    decision = report.decisions[0]

    assert decision.decision == "insufficient_information"
    prior = next(
        item
        for item in decision.mechanism_mappings
        if item.canonical_component_id == "distillation.yolo26_teacher_student"
    )
    assert prior.source == "title"
    assert prior.confidence == "low"
    assert prior.authorizes_method_profile is False


def test_title_only_incompatible_family_is_conservatively_separated() -> None:
    paper = PaperRecord(
        paper_id="open-vocabulary-title",
        title="Open Vocabulary Object Detection with Captions",
        year=2025,
        abstract="A general detection study.",
        component_ids=["object_detection"],
    )

    decision = PaperMethodProfileBuilder(_resolver()).build([paper]).decisions[0]

    assert decision.decision == "separate_detector_family"
    prior = next(
        item
        for item in decision.mechanism_mappings
        if item.canonical_component_id == "detection_head.open_vocabulary"
    )
    assert prior.authorizes_method_profile is False
    assert prior.yolo26_compatibility == "incompatible"


def test_explicit_summary_boundary_promotes_generic_catalog_label() -> None:
    paper = PaperRecord(
        paper_id="summary-boundary",
        title="Small object detector",
        year=2025,
        abstract=(
            "Small object sampling modifies image weights in the train dataloader."
        ),
        component_ids=["object_detection"],
    )

    report = PaperMethodProfileBuilder(_resolver()).build([paper])
    profile = report.profiles[0]
    decision = report.decisions[0]

    assert decision.decision == "reuse_existing_adapter"
    assert decision.canonical_component_ids == ["sampling.small_object"]
    assert profile.structured_method_evidence is not None
    assert profile.structured_method_evidence.authorizes_method_profile is True
    assert profile.paper_parameters["changed_variables"] == [
        "data.sampling_policy"
    ]
    assert profile.protocol_constraints["required_runtime_hooks"] == [
        "build_train_dataloader",
        "build_train_dataset",
    ]


def test_harness_hint_cannot_authorize_generic_catalog_label() -> None:
    paper = PaperRecord(
        paper_id="harness-prior-only",
        title="Detector",
        year=2025,
        abstract="Object detection study.",
        component_ids=["object_detection"],
        provenance=PaperProvenance(
            source_repository="local",
            source_path="papers.json",
            source_record_hash="hash",
            importer_version="test",
            original_harness_hints=[
                "Try small object sampling in the train dataloader with image weights."
            ],
        ),
    )

    decision = PaperMethodProfileBuilder(_resolver()).build([paper]).decisions[0]

    assert decision.decision == "insufficient_information"
    assert any(
        item.source == "harness_hint"
        and item.authorizes_method_profile is False
        for item in decision.mechanism_mappings
    )


def test_explicit_noncanonical_method_becomes_profile_not_adapter() -> None:
    paper = PaperRecord(
        paper_id="augmentation-profile",
        title="Detection augmentation",
        year=2025,
        abstract=(
            "The method applies scale-aware augmentation to training data "
            "and changes the augmentation policy."
        ),
        component_ids=["object_detection"],
    )

    report = PaperMethodProfileBuilder(_resolver()).build([paper])
    decision = report.decisions[0]

    assert decision.decision == "new_method_profile"
    assert decision.canonical_component_ids == []
    assert decision.reusable_adapter_ids == []
    assert decision.reasons == [
        "explicit_method_boundary_requires_canonical_mechanism_mapping"
    ]


def test_multiple_sahi_terms_reuse_one_inference_adapter() -> None:
    decision = PaperMethodProfileBuilder(_resolver()).build([
        _paper(
            "sahi-paper",
            ["high_resolution_tiling", "overlap_merge", "sliced_inference"],
        )
    ]).decisions[0]

    assert decision.decision == "reuse_existing_adapter"
    assert decision.canonical_component_ids == ["inference.sahi_slicing"]
    assert decision.reusable_adapter_ids == ["inference.sahi_slicing"]


def test_generic_quality_family_does_not_claim_exact_loss_implementation() -> None:
    decision = PaperMethodProfileBuilder(_resolver()).build([
        _paper(
            "mutual-supervision",
            ["mutual_supervision", "classification_localization"],
        )
    ]).decisions[0]

    assert decision.decision == "new_method_profile"
    assert decision.canonical_component_ids == ["quality_alignment.general"]
    assert decision.reusable_adapter_ids == []
    assert decision.paper_mechanism_resolutions[0].resolved is False
    assert decision.exact_reproduction_claim is False
    assert decision.component_adaptation is True


def test_vision_language_distillation_routes_whole_method_to_separate_track() -> None:
    paper = PaperRecord(
        paper_id="vl-distillation",
        title="Visual linguistic distillation",
        year=2024,
        abstract="Uses visual linguistic knowledge distillation.",
        component_ids=["knowledge_distillation"],
    )

    report = PaperMethodProfileBuilder(_resolver()).build([paper])
    decision = report.decisions[0]

    assert decision.decision == "separate_detector_family"
    assert decision.adaptation_mode == "separate_detector_family"
    assert decision.reusable_adapter_ids == []
    assert decision.component_adaptation is False
    assert report.profiles[0].adaptation_mode == "separate_detector_family"
    assert {
        "distillation.vision_language",
        "distillation.yolo26_teacher_student",
    }.issubset(decision.canonical_component_ids)


def test_cross_scale_fusion_is_component_adaptation_not_detector_reproduction() -> None:
    decision = PaperMethodProfileBuilder(_resolver()).build([
        _paper("cross-scale-paper", ["cross_scale_fusion"])
    ]).decisions[0]

    assert decision.decision == "reuse_existing_adapter"
    assert decision.reusable_adapter_ids == ["neck.multi_scale_fusion"]
    assert decision.adaptation_mode == "component_adaptation"
    assert decision.exact_reproduction_claim is False


def test_insufficient_decision_reports_field_level_blockers() -> None:
    decision = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("unknown-fields", ["generic_detection_task"])]
    ).decisions[0]

    assert decision.decision == "insufficient_information"
    gaps = {item.reason_code: item for item in decision.adaptation_gaps}
    assert gaps["canonical_component_mapping_required"].severity == "blocking"
    assert gaps["method_name_not_explicit"].field_name == "method_name"
    assert gaps["official_code_metadata_missing"].severity == "non_blocking"


def test_task_and_detector_labels_are_not_reported_as_missing_adapters() -> None:
    report = PaperMethodProfileBuilder(_resolver()).build([
        _paper("small-task", ["small_object"]),
        PaperRecord(
            paper_id="detr-family",
            title="DETR family method",
            year=2025,
            detector_family="detr",
            component_ids=["detr"],
        ),
    ])
    decisions = {item.paper_id: item for item in report.decisions}

    small_gap = decisions["small-task"].adaptation_gaps[0]
    assert small_gap.reason_code == "task_scope_not_canonical_mechanism"
    assert decisions["small-task"].decision == "insufficient_information"
    detr = decisions["detr-family"]
    assert detr.decision == "separate_detector_family"
    assert detr.reusable_adapter_ids == []
    assert any(
        item.reason_code == "detector_family_label_not_component"
        for item in detr.adaptation_gaps
    )


def test_missing_adapter_reports_runtime_artifact_requirements() -> None:
    decision = PaperMethodProfileBuilder(_resolver()).build(
        [_paper("adapter-gap", ["deformable_attention"])]
    ).decisions[0]

    adapter_gap = next(
        item for item in decision.adaptation_gaps if item.reason_code == "adapter_not_verified"
    )
    assert adapter_gap.severity == "blocking"
    assert "runtime and smoke artifacts bound to adapter hash" in (
        adapter_gap.required_evidence
    )
