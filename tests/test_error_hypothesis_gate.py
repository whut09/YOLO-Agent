"""Prompt-18I-v2 §3: hypothesis-to-training gate tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.detection_error_profile import ErrorProfileSource  # noqa: E402
from yolo_agent.core.detection_error_profile_builder import (  # noqa: E402
    build_detection_error_profile,
)
from yolo_agent.core.error_fact_identity import error_fact_identities  # noqa: E402
from yolo_agent.core.error_hypothesis_gate import (  # noqa: E402
    admit_hypotheses,
    admit_hypothesis,
    training_candidate_allowed,
)
from yolo_agent.core.error_root_cause import derive_root_cause_hypotheses  # noqa: E402


@pytest.fixture()
def facts(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    profile = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json",
            predictions_json=data / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )
    return profile, error_fact_identities(profile)


def test_real_hypotheses_all_admitted(facts) -> None:
    profile, extracted = facts
    hypotheses = derive_root_cause_hypotheses(profile)
    assert hypotheses
    admissions = admit_hypotheses(hypotheses, extracted)
    assert all(item.admitted for item in admissions)
    assert training_candidate_allowed(admissions)


def test_hypothesis_without_evidence_rejected(facts) -> None:
    _, extracted = facts

    class Naked:
        hypothesis_id = "guessy"

    admission = admit_hypothesis(Naked(), extracted)
    assert not admission.admitted
    assert "cites no evidence" in admission.reason
    assert not training_candidate_allowed([admission])


def test_hypothesis_citing_unmeasured_facts_rejected(facts) -> None:
    profile, extracted = facts

    class WithFakeLink:
        hypothesis_id = "half-real"

        class _Link:
            fact_ids = [
                f"{profile.profile_id}:global:map50",
                f"{profile.profile_id}:global:unicorn_ap",
            ]

        evidence = [_Link()]

    admission = admit_hypothesis(WithFakeLink(), extracted)
    assert not admission.admitted
    assert f"{profile.profile_id}:global:unicorn_ap" in admission.missing_fact_ids
    assert f"{profile.profile_id}:global:map50" not in admission.missing_fact_ids
    assert not training_candidate_allowed([admission])


def test_one_bad_hypothesis_blocks_the_whole_candidate(facts) -> None:
    profile, extracted = facts
    good = derive_root_cause_hypotheses(profile)

    class Naked:
        hypothesis_id = "naked"

    admissions = admit_hypotheses([*good, Naked()], extracted)
    assert not training_candidate_allowed(admissions)
