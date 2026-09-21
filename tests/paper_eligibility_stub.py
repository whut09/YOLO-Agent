"""Shared Prompt-18E eligibility stubs for synthetic-world test modules.

Several suites exercise the materialization/ASHA state machines with
synthetic paper ids that are deliberately outside the frozen 83 manifest.
The Prompt-18E eligibility gate correctly rejects such ids at registration;
those modules pin the state machines themselves, not the eligibility gate
(which has its own dedicated suite).  ``synthetic_eligibility_fixture``
returns a ready-made autouse pytest fixture that monkeypatches the
eligibility evaluator to a permissive synthetic-world verdict.  No
training runs anywhere.
"""

from __future__ import annotations

from typing import Callable

import pytest


def synthetic_eligibility_fixture() -> Callable[[pytest.MonkeyPatch], None]:
    """Build an autouse fixture stubbing the Prompt-18E eligibility gate."""

    @pytest.fixture(autouse=True)
    def _fixture(monkeypatch: pytest.MonkeyPatch) -> None:
        from yolo_agent.research.paper_candidate_eligibility import (
            CandidateEligibilityReport,
        )

        monkeypatch.setattr(
            "yolo_agent.research.paper_candidate_eligibility."
            "evaluate_candidate_eligibility",
            lambda **kwargs: CandidateEligibilityReport(
                candidate_id="synthetic",
                paper_ids=sorted(kwargs.get("paper_ids") or []),
                eligible=True,
                blockers=[],
            ),
        )

    return _fixture
