"""fix 24: contract-target classes must produce facts even outside top-N lists.

A target class named by an evaluation contract (e.g. ``dining table`` for
``localization_heavy_class``) that falls out of the top-N mined list still has
a real measured count in ``class_summaries``.  Without a materialized fact on
BOTH sides of a paired experiment the target fact pair can never be built and
verification blocks forever.  These tests pin the materialization semantics:

- required class present in class_summaries -> fact is generated;
- no requirement -> no extra facts (previous behavior preserved);
- class absent from the report entirely -> NOT fabricated (a zero from a
  broken evaluation must not silently satisfy a contract);
- class already in the top-N list -> no duplicate fact;
- fact types outside the count-fact family are ignored.
"""

from __future__ import annotations

from yolo_agent.core.error_facts import (
    build_error_facts_from_coco_error_report,
    required_fact_classes_from_targets,
)


def _report_with_dining_table_only_in_summaries() -> dict:
    return {
        "localization_error_top_classes": [
            {"name": "person", "localization_error": 500},
            {"name": "car", "localization_error": 320},
        ],
        "false_negative_top_classes": [],
        "background_false_positive_top_classes": [],
        "class_summaries": [
            {"name": "person", "localization_error": 500},
            {"name": "car", "localization_error": 320},
            {"name": "dining table", "localization_error": 2140},
        ],
        "class_confusion_pairs": {},
    }


def test_required_class_outside_top_n_is_materialized_from_class_summaries() -> None:
    report = _report_with_dining_table_only_in_summaries()

    facts = build_error_facts_from_coco_error_report(
        report,
        "run-1",
        "candidate-1",
        "node-1",
        required_fact_classes={"localization_heavy_class": {"dining table"}},
    )

    dining = [item for item in facts if item.class_name == "dining table"]
    assert len(dining) == 1
    fact = dining[0]
    assert fact.fact_type == "localization_heavy_class"
    assert fact.subject == "dining table"
    assert fact.count == 2140
    assert fact.rank is None  # top-N facts carry ranks; required facts do not
    assert fact.evidence.get("localization_error") == 2140
    assert "bbox_loss_recipe" in fact.action_candidates


def test_no_required_classes_preserves_previous_behavior() -> None:
    report = _report_with_dining_table_only_in_summaries()

    facts = build_error_facts_from_coco_error_report(
        report,
        "run-1",
        "candidate-1",
        "node-1",
    )

    assert not [item for item in facts if item.class_name == "dining table"]


def test_required_class_absent_from_report_is_not_fabricated() -> None:
    report = _report_with_dining_table_only_in_summaries()
    report["class_summaries"] = [
        {"name": "person", "localization_error": 500},
        {"name": "car", "localization_error": 320},
    ]

    facts = build_error_facts_from_coco_error_report(
        report,
        "run-1",
        "candidate-1",
        "node-1",
        required_fact_classes={"localization_heavy_class": {"dining table"}},
    )

    assert not [item for item in facts if item.class_name == "dining table"]


def test_required_class_already_in_top_list_is_not_duplicated() -> None:
    report = _report_with_dining_table_only_in_summaries()

    facts = build_error_facts_from_coco_error_report(
        report,
        "run-1",
        "candidate-1",
        "node-1",
        required_fact_classes={"localization_heavy_class": {"person"}},
    )

    person_facts = [item for item in facts if item.class_name == "person"]
    assert len(person_facts) == 1
    assert person_facts[0].rank == 1  # came from the top-N list, not the required path


def test_unknown_fact_type_in_requirements_is_ignored() -> None:
    report = _report_with_dining_table_only_in_summaries()

    facts = build_error_facts_from_coco_error_report(
        report,
        "run-1",
        "candidate-1",
        "node-1",
        required_fact_classes={"class_low_ap": {"dining table"}},
    )

    assert not [item for item in facts if item.class_name == "dining table"]


def test_required_fact_classes_from_targets_extracts_fact_type_and_class() -> None:
    targets = [
        {"fact_type": "localization_heavy_class", "class_name": "dining table"},
        {"fact_type": "localization_heavy_class", "subject": "dining table"},
        {"fact_type": "false_negative_heavy_class", "class_name": "cat"},
        "not-a-dict",
        {"class_name": "no-fact-type"},
        {"fact_type": "class_low_ap"},
    ]

    required = required_fact_classes_from_targets(targets)  # type: ignore[arg-type]

    assert required == {
        "localization_heavy_class": {"dining table"},
        "false_negative_heavy_class": {"cat"},
    }


def test_required_fact_classes_from_targets_handles_none_and_empty() -> None:
    assert required_fact_classes_from_targets(None) == {}
    assert required_fact_classes_from_targets([]) == {}
