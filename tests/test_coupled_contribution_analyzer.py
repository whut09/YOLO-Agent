import pytest

from yolo_agent.agents.coupled_contribution import (
    CoupledArmObservation,
    CoupledContributionAnalyzer,
)


def _observation(
    arm: str,
    seed: int,
    delta: float,
    **updates: object,
) -> CoupledArmObservation:
    values = {
        "arm": arm,
        "node_id": f"node-{arm}-{seed}",
        "matched_control_node_id": f"baseline-{seed}",
        "seed": seed,
        "protocol_hash": "protocol-1",
        "metric_deltas": {"ap_small": delta},
        "paired_result_verified": True,
    }
    values.update(updates)
    return CoupledArmObservation.model_validate(values)


def _analyze(observations: list[CoupledArmObservation]):
    return CoupledContributionAnalyzer().analyze(
        recipe_id="coupled-one",
        component_a="head.p2_small_object",
        component_b="sampling.small_object",
        observations=observations,
    )


def test_single_seed_reports_atomic_total_and_interaction_as_possible() -> None:
    report = _analyze(
        [
            _observation("A", 1, 0.02),
            _observation("B", 1, 0.01),
            _observation("A+B", 1, 0.04),
        ]
    )

    effects = {item.effect_id.split(":")[1]: item for item in report.effects}
    assert report.complete_seeds == [1]
    assert effects["A"].mean_delta == pytest.approx(0.02)
    assert effects["B"].mean_delta == pytest.approx(0.01)
    assert effects["A+B"].mean_delta == pytest.approx(0.04)
    assert effects["interaction"].mean_delta == pytest.approx(0.01)
    assert {item.confidence for item in report.effects} == {"possible"}


def test_three_consistent_seeds_confirm_positive_interaction() -> None:
    observations = []
    for seed, offset in [(1, 0.0000), (2, 0.0002), (3, -0.0002)]:
        observations.extend(
            [
                _observation("A", seed, 0.01 + offset),
                _observation("B", seed, 0.01 + offset),
                _observation("A+B", seed, 0.04 + offset),
            ]
        )

    report = _analyze(observations)
    interaction = next(
        item for item in report.effects if item.effect_kind == "interaction"
    )

    assert interaction.seed_count == 3
    assert interaction.confidence == "confirmed"
    assert interaction.direction == "positive"
    assert interaction.confidence_interval_low is not None
    assert interaction.confidence_interval_low > 0


def test_interval_crossing_zero_remains_possible() -> None:
    observations = []
    for seed, combined in [(1, 0.05), (2, 0.01), (3, 0.03)]:
        observations.extend(
            [
                _observation("A", seed, 0.01),
                _observation("B", seed, 0.01),
                _observation("A+B", seed, combined),
            ]
        )
    interaction = next(
        item for item in _analyze(observations).effects
        if item.effect_kind == "interaction"
    )

    assert interaction.confidence == "possible"
    assert interaction.direction == "uncertain"
    assert interaction.confidence_interval_low < 0 < interaction.confidence_interval_high


def test_missing_arm_and_identity_mismatch_block_seed_attribution() -> None:
    report = _analyze(
        [
            _observation("A", 1, 0.01),
            _observation("B", 1, 0.01),
            _observation(
                "A+B", 1, 0.03, matched_control_node_id="other-baseline"
            ),
            _observation("A", 2, 0.01),
            _observation("B", 2, 0.01),
        ]
    )

    assert report.effects == []
    assert report.incomplete_seeds[1] == ["matched_control_mismatch"]
    assert report.incomplete_seeds[2] == ["missing_arms:A+B"]


def test_unverified_and_inherited_observations_are_rejected() -> None:
    report = _analyze(
        [
            _observation("A", 1, 0.01, paired_result_verified=False),
            _observation("B", 1, 0.01, inheritance_depth=1),
            _observation("A+B", 1, 0.03, evidence_role="inherited_context"),
        ]
    )

    assert report.effects == []
    assert sorted(report.rejected_observations.values()) == [
        "current_observation_required",
        "inherited_evidence_forbidden",
        "paired_result_not_verified",
    ]


def test_arm_exclusions_prevent_invalid_single_and_interaction_attribution() -> None:
    report = _analyze(
        [
            _observation(
                "A",
                1,
                0.02,
                metric_deltas={"ap_small": 0.02, "latency_ms": 2.0},
            ),
            _observation(
                "B",
                1,
                0.01,
                metric_deltas={"ap_small": 0.01, "latency_ms": 0.1},
                attribution_excluded_metrics=["latency_ms"],
            ),
            _observation(
                "A+B",
                1,
                0.04,
                metric_deltas={"ap_small": 0.04, "latency_ms": 2.1},
            ),
        ]
    )

    effect_metrics = {
        (item.effect_kind, tuple(item.component_ids), item.metric_name)
        for item in report.effects
    }
    assert ("single", ("head.p2_small_object",), "latency_ms") in effect_metrics
    assert ("single", ("sampling.small_object",), "latency_ms") not in effect_metrics
    assert (
        "interaction",
        ("head.p2_small_object", "sampling.small_object"),
        "latency_ms",
    ) not in effect_metrics


def test_arm_specific_guard_metric_does_not_require_other_arms() -> None:
    report = _analyze(
        [
            _observation(
                "A",
                1,
                0.02,
                metric_deltas={"ap_small": 0.02, "peak_vram_mb": 128.0},
            ),
            _observation("B", 1, 0.01),
            _observation("A+B", 1, 0.04),
        ]
    )

    vram = [item for item in report.effects if item.metric_name == "peak_vram_mb"]
    assert len(vram) == 1
    assert vram[0].effect_kind == "single"
    assert vram[0].component_ids == ["head.p2_small_object"]
    assert vram[0].mean_delta == pytest.approx(128.0)
