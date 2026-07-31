from __future__ import annotations

from yolo_agent.research.component_aliases import (
    CanonicalComponentDefinition,
    ComponentAliasConfig,
    ComponentAliasResolver,
)
from yolo_agent.research.method_profiles import (
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
    assert decisions["unknown-paper"].decision == "insufficient_information"
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


def test_summary_mechanism_can_rescue_generic_catalog_label() -> None:
    paper = PaperRecord(
        paper_id="summary-rescue",
        title="Summary rescue",
        year=2025,
        abstract="Uses teacher student distillation.",
        component_ids=["object_detection"],
    )

    decision = PaperMethodProfileBuilder(_resolver()).build([paper]).decisions[0]

    assert decision.decision == "reuse_existing_adapter"
    assert decision.canonical_component_ids == [
        "distillation.yolo26_teacher_student"
    ]
    assert any(item.source == "summary" for item in decision.mechanism_mappings)
