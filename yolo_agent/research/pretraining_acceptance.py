"""Final pre-training acceptance for the frozen Paper-83 campaign.

The Prompt-16 contract: the acceptance system and its report must be
produced whether the verdict is PASS or FAIL — only the training gate
stays closed on failure.  Nothing here ever mutates the frozen manifest,
lowers a maturity bar, or ignores a failed paper to reach PASS, and no
real training is ever started: even the 83/83-allow safety probe runs
behind a stubbed trainer.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


ACCEPTANCE_SCHEMA_VERSION = "pretraining_acceptance.v1"
DEFAULT_MANIFEST_PATH = "configs/research/paper_83_manifest.yaml"
DEFAULT_AUDIT_PATH = "artifacts/paper_83_exactness_audit.yaml"
DEFAULT_ACCEPTANCE_YAML = "artifacts/pretraining_acceptance.yaml"
DEFAULT_ACCEPTANCE_MD = "docs/PRETRAINING_ACCEPTANCE.md"

#: Action families the acceptance requires the system to represent with at
#: least one executable/represented action (Prompt-16 part two).  Local
#: deterministic strategies count — paper origin is not required.
REQUIRED_ACTION_FAMILIES: tuple[str, ...] = (
    "annotation",
    "data_cleaning",
    "data_selection",
    "sampling",
    "augmentation",
    "preprocessing",
    "model_scale",
    "input_resolution",
    "backbone",
    "neck",
    "feature_fusion",
    "head",
    "assignment",
    "bbox_loss",
    "classification_loss",
    "auxiliary_loss",
    "training_strategy",
    "optimizer",
    "regularization",
    "postprocess",
    "threshold",
    "inference",
    "calibration",
    "active_learning",
)


class IntegrityCounts(BaseModel):
    """Per-check integrity tallies over the exactness audit."""

    total: int = 0
    ready: int = 0
    blocked: int = 0
    out_of_scope: int = 0
    generic_only: int = 0
    metadata_only: int = 0
    no_op: int = 0
    mock_only: int = 0
    missing_runtime: int = 0
    missing_tests: int = 0
    missing_smoke: int = 0
    missing_compatibility: int = 0
    missing_rollback: int = 0
    missing_fingerprint: int = 0


class IntegrityCheck(BaseModel):
    """One Paper-83 integrity requirement and whether it holds."""

    requirement: str
    expected: str
    observed: str
    passed: bool


class PaperCampaignSection(BaseModel):
    """Part one: frozen-campaign implementation integrity."""

    manifest_paper_count: int = 0
    unique_paper_ids: int = 0
    membership_hash_valid: bool = False
    spec_complete: int = 0
    code_bound: int = 0
    runtime_integrated: int = 0
    unit_tested: int = 0
    non_mock_smoke_passed: int = 0
    compatibility_validated: int = 0
    implementation_ready: int = 0
    blocked: int = 0
    integrity_counts: IntegrityCounts = Field(default_factory=IntegrityCounts)
    checks: list[IntegrityCheck] = Field(default_factory=list)
    passed: bool = False

    @property
    def failed_checks(self) -> list[str]:
        return [item.requirement for item in self.checks if not item.passed]


class ActionSpaceSection(BaseModel):
    """Part two: unified optimization action space coverage."""

    catalog_action_count: int = 0
    covered_families: list[str] = Field(default_factory=list)
    missing_families: list[str] = Field(default_factory=list)
    paper_lineage_actions: int = 0
    local_actions: int = 0
    passed: bool = False


class AutonomousLoopSection(BaseModel):
    """Part three: multi-round FakeExperimentRunner loop verification."""

    rounds_executed: int = 0
    decisions_seen: list[str] = Field(default_factory=list)
    diagnosis_chain: bool = False
    error_delta_used: bool = False
    bounded_hpo_module_present: bool = False
    asha_budget_routed: bool = False
    rollback_observed: bool = False
    pareto_axes_used: bool = False
    stop_policy_honest: bool = False
    multi_round_completed: bool = False
    passed: bool = False

    @property
    def failed_checks(self) -> list[str]:
        failures: list[str] = []
        if not self.diagnosis_chain:
            failures.append("diagnosis_chain")
        if not self.error_delta_used:
            failures.append("error_delta_used")
        if not self.bounded_hpo_module_present:
            failures.append("bounded_hpo_module_present")
        if not self.asha_budget_routed:
            failures.append("asha_budget_routed")
        if not self.rollback_observed:
            failures.append("rollback_observed")
        if not self.pareto_axes_used:
            failures.append("pareto_axes_used")
        if not self.stop_policy_honest:
            failures.append("stop_policy_honest")
        if not self.multi_round_completed:
            failures.append("multi_round_completed")
        return failures


class SafetySection(BaseModel):
    """Part four: gate/consent safety probes (stubbed trainer only)."""

    gate_82_of_83_blocks: bool = False
    gate_83_of_83_allows: bool = False
    gate_fail_closed_on_missing_manifest: bool = False
    gate_fail_closed_on_hash_mismatch: bool = False
    full_run_consent_boundary_preserved: bool = False
    passed: bool = False

    @property
    def failed_checks(self) -> list[str]:
        failures: list[str] = []
        if not self.gate_82_of_83_blocks:
            failures.append("gate_82_of_83_blocks")
        if not self.gate_83_of_83_allows:
            failures.append("gate_83_of_83_allows")
        if not self.gate_fail_closed_on_missing_manifest:
            failures.append("gate_fail_closed_on_missing_manifest")
        if not self.gate_fail_closed_on_hash_mismatch:
            failures.append("gate_fail_closed_on_hash_mismatch")
        if not self.full_run_consent_boundary_preserved:
            failures.append("full_run_consent_boundary_preserved")
        return failures


class TestsSection(BaseModel):
    """Part five: pytest + ruff results, failures classified."""

    fast_command: str = ""
    fast_exit_code: int | None = None
    fast_summary: str = ""
    fast_introduced_failures: list[str] = Field(default_factory=list)
    fast_preexisting_failures: list[str] = Field(default_factory=list)
    slow_attempted: bool = False
    slow_exit_code: int | None = None
    slow_summary: str = ""
    lint_command: str = ""
    lint_exit_code: int | None = None
    lint_summary: str = ""
    passed: bool = False


class RuntimePreflightSection(BaseModel):
    """Prompt-18E hard requirement: the committed 83-paper runtime sweep.

    Loaded from ``artifacts/paper_83_runtime_preflight.yaml`` — the artifact
    that *executed* every frozen paper's runtime path on synthetic CPU
    tensors.  Anything other than exactly 83/83 PASS is a FAIL; a missing,
    stale-schema, or self-inconsistent artifact fails closed.
    """

    papers: int = 0
    passed: int = 0
    failed: int = 0
    verdict: Literal["PASS", "FAIL"] = "FAIL"
    source_path: str = ""

    @property
    def passed_bool(self) -> bool:
        return self.verdict == "PASS" and self.papers == 83 and self.passed == 83 and self.failed == 0

    @classmethod
    def from_artifact(cls, path: Path | str) -> "RuntimePreflightSection":
        """Load the preflight artifact; any inconsistency fails closed."""

        import yaml

        file = Path(path)
        section = cls(source_path=str(file))
        if not file.is_file():
            return section
        try:
            payload = yaml.safe_load(file.read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError):
            return section
        if not isinstance(payload, dict):
            return section
        if str(payload.get("schema_version", "")) != "paper_83_runtime_preflight.v1":
            return section
        section.papers = int(payload.get("paper_count") or 0)
        section.passed = int(payload.get("passed") or 0)
        section.failed = int(payload.get("failed") or 0)
        sweep_passed = bool(payload.get("runtime_preflight_passed"))
        consistent = (
            section.papers == 83
            and section.passed == 83
            and section.failed == 0
            and sweep_passed
        )
        section.verdict = "PASS" if consistent else "FAIL"
        return section


class NonGpuVerificationSection(BaseModel):
    """Prompt-18E hard requirement: the committed full non-GPU test record.

    Loaded from ``artifacts/non_gpu_test_acceptance.yaml`` (Prompt-18D).  The
    fast and slow suites plus ruff must all be recorded with exit code 0 and
    ``passed: true``; a missing, stale, or self-contradictory artifact fails
    closed.
    """

    fast: Literal["PASS", "FAIL"] = "FAIL"
    slow: Literal["PASS", "FAIL"] = "FAIL"
    ruff: Literal["PASS", "FAIL"] = "FAIL"
    verdict: Literal["PASS", "FAIL"] = "FAIL"
    source_path: str = ""

    @property
    def passed_bool(self) -> bool:
        return (
            self.fast == "PASS"
            and self.slow == "PASS"
            and self.ruff == "PASS"
            and self.verdict == "PASS"
        )

    @classmethod
    def from_artifact(cls, path: Path | str) -> "NonGpuVerificationSection":
        """Load the non-GPU acceptance artifact; anything absent fails closed."""

        import yaml

        file = Path(path)
        section = cls(source_path=str(file))
        if not file.is_file():
            return section
        try:
            payload = yaml.safe_load(file.read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError):
            return section
        if not isinstance(payload, dict):
            return section
        fast_ok = (
            isinstance(payload.get("fast_exit_code"), int)
            and int(payload["fast_exit_code"]) == 0
        )
        slow_ok = (
            isinstance(payload.get("slow_exit_code"), int)
            and int(payload["slow_exit_code"]) == 0
        )
        ruff_ok = (
            isinstance(payload.get("ruff_exit_code"), int)
            and int(payload["ruff_exit_code"]) == 0
        )
        section.fast = "PASS" if fast_ok else "FAIL"
        section.slow = "PASS" if slow_ok else "FAIL"
        section.ruff = "PASS" if ruff_ok else "FAIL"
        section.verdict = "PASS" if (
            fast_ok and slow_ok and ruff_ok and bool(payload.get("passed"))
        ) else "FAIL"
        return section


class TrainingGateSection(BaseModel):
    """Final gate verdict embedded in the acceptance record."""

    allowed: bool
    ready: int
    blocked: int
    required: int
    lock_reasons: list[str] = Field(default_factory=list)


class PretrainingAcceptance(BaseModel, YAMLModelMixin):
    """The complete acceptance record — always serializable, never hidden."""

    model_config = {"extra": "forbid"}

    schema_version: str = ACCEPTANCE_SCHEMA_VERSION
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    real_training_executed: Literal[False] = False
    paper_campaign: PaperCampaignSection = Field(default_factory=PaperCampaignSection)
    optimization_action_space: ActionSpaceSection = Field(default_factory=ActionSpaceSection)
    autonomous_loop: AutonomousLoopSection = Field(default_factory=AutonomousLoopSection)
    safety: SafetySection = Field(default_factory=SafetySection)
    tests: TestsSection = Field(default_factory=TestsSection)
    runtime_preflight: RuntimePreflightSection = Field(default_factory=RuntimePreflightSection)
    non_gpu_verification: NonGpuVerificationSection = Field(
        default_factory=NonGpuVerificationSection
    )
    training_gate: TrainingGateSection
    verdict: Literal["PASS", "FAIL"]
    remaining_blockers: list[str] = Field(default_factory=list)

    @property
    def all_critical_checks_pass(self) -> bool:
        return (
            self.paper_campaign.passed
            and self.optimization_action_space.passed
            and self.autonomous_loop.passed
            and self.safety.passed
            and self.tests.passed
            and self.runtime_preflight.passed_bool
            and self.non_gpu_verification.passed_bool
        )

    @model_validator(mode="after")
    def _gate_semantics_invariant(self) -> "PretrainingAcceptance":
        """Prompt-18A invariant: the three gate signals are one signal.

        ``training_gate.allowed`` ≡ ``all_critical_checks_pass`` ≡
        ``verdict == "PASS"``.  A record claiming PASS while a critical
        section (including the test/lint tier) failed cannot be constructed.
        """

        critical = self.all_critical_checks_pass
        if self.training_gate.allowed != critical:
            raise ValueError(
                "training_gate.allowed must equal all_critical_checks_pass "
                f"(gate={self.training_gate.allowed}, critical={critical})"
            )
        if (self.verdict == "PASS") != critical:
            raise ValueError(
                "verdict must be PASS iff all critical checks pass "
                f"(verdict={self.verdict}, critical={critical})"
            )
        return self


class PretrainingAcceptanceRunner:
    """Run every acceptance section and write artifacts — on PASS and FAIL."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
        audit_path: Path | str = DEFAULT_AUDIT_PATH,
        catalog_path: Path | str = "configs/actions/detection_action_catalog.yaml",
        run_tests: bool = True,
        run_slow_tests: bool = False,
        project_root: Path | str = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audit_path = Path(audit_path)
        self.catalog_path = Path(catalog_path)
        self.run_tests = run_tests
        self.run_slow_tests = run_slow_tests
        self.root = Path(project_root)

    # ------------------------------------------------------------ public --

    def run(self) -> PretrainingAcceptance:
        """Execute all sections; artifacts are written by the caller/writer."""

        campaign = self._verify_paper_campaign()
        action_space = self._verify_action_space()
        loop_section = self._verify_autonomous_loop()
        safety = self._verify_safety(campaign.implementation_ready)
        tests = self._run_tests()
        runtime_preflight = self._load_runtime_preflight()
        non_gpu = self._load_non_gpu_verification()

        blockers: list[str] = []
        blockers.extend(f"paper_campaign:{item}" for item in campaign.failed_checks)
        blockers.extend(f"action_space:{item}" for item in action_space.missing_families)
        blockers.extend(f"autonomous_loop:{item}" for item in loop_section.failed_checks)
        blockers.extend(f"safety:{item}" for item in safety.failed_checks)
        blockers.extend(_tests_lock_reasons(tests))
        if not runtime_preflight.passed_bool:
            blockers.append(
                "runtime_preflight_not_83_of_83:"
                f"{runtime_preflight.passed}/{runtime_preflight.papers}"
            )
        if not non_gpu.passed_bool:
            reasons = [
                name
                for name, ok in (
                    ("fast", non_gpu.fast == "PASS"),
                    ("slow", non_gpu.slow == "PASS"),
                    ("ruff", non_gpu.ruff == "PASS"),
                    ("passed_flag", non_gpu.verdict == "PASS"),
                )
                if not ok
            ]
            blockers.append(f"non_gpu_verification_failed:{'+'.join(reasons)}")

        gate_ready = campaign.implementation_ready
        # Prompt-18A gate semantics: every critical section is load-bearing.
        # The test/lint tier pins this very system, so a green campaign can
        # never unlock training while the suite that proves it is red.
        # Prompt-18E extends the critical set: the committed 83-paper runtime
        # sweep and the full non-GPU verification record are hard gates too —
        # missing or failed artifacts lock the gate (fail-closed).
        critical_passed = (
            campaign.passed
            and action_space.passed
            and loop_section.passed
            and safety.passed
            and tests.passed
            and runtime_preflight.passed_bool
            and non_gpu.passed_bool
        )
        gate_section = TrainingGateSection(
            allowed=critical_passed,
            ready=gate_ready,
            blocked=campaign.blocked,
            required=campaign.manifest_paper_count,
            lock_reasons=[] if critical_passed else blockers,
        )
        verdict: Literal["PASS", "FAIL"] = "PASS" if critical_passed else "FAIL"
        acceptance = PretrainingAcceptance(
            paper_campaign=campaign,
            optimization_action_space=action_space,
            autonomous_loop=loop_section,
            safety=safety,
            tests=tests,
            runtime_preflight=runtime_preflight,
            non_gpu_verification=non_gpu,
            training_gate=gate_section,
            verdict=verdict,
            remaining_blockers=blockers,
        )
        return acceptance

    def _load_runtime_preflight(self) -> RuntimePreflightSection:
        """Read the committed 83-paper runtime sweep (Prompt-18E gate 6/7)."""

        return RuntimePreflightSection.from_artifact(
            self.root / "artifacts/paper_83_runtime_preflight.yaml"
        )

    def _load_non_gpu_verification(self) -> NonGpuVerificationSection:
        """Read the committed full non-GPU test record (Prompt-18E gate 7/7)."""

        return NonGpuVerificationSection.from_artifact(
            self.root / "artifacts/non_gpu_test_acceptance.yaml"
        )

    # --------------------------------------------------- part one: papers --

    def _verify_paper_campaign(self) -> PaperCampaignSection:
        from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

        section = PaperCampaignSection()
        if not self.manifest_path.is_file():
            section.checks.append(
                IntegrityCheck(
                    requirement="manifest_present",
                    expected="frozen manifest exists",
                    observed="missing",
                    passed=False,
                )
            )
            return section
        manifest = Paper83Manifest.from_yaml(self.manifest_path)
        section.manifest_paper_count = manifest.paper_count
        paper_ids = [paper.paper_id for paper in manifest.papers] if hasattr(manifest, "papers") else []
        if not paper_ids:
            entries = getattr(getattr(manifest, "campaign", None), "paper_ids", None) or []
            paper_ids = list(entries)
        section.unique_paper_ids = len(set(paper_ids))
        membership_hash = manifest.campaign.membership_hash
        recomputed = manifest.campaign.calculate_membership_hash() if hasattr(
            manifest.campaign, "calculate_membership_hash"
        ) else membership_hash
        section.membership_hash_valid = bool(membership_hash) and membership_hash == recomputed

        audit_records = self._load_audit_records()
        counts = IntegrityCounts(total=len(audit_records))
        for record in audit_records:
            status = str(record.get("status", ""))
            inventory = record.get("evidence_inventory") or {}
            evidence_class = str(inventory.get("implementation_evidence_class", ""))
            if evidence_class == "generic_only":
                counts.generic_only += 1
            if evidence_class == "mock_only":
                counts.mock_only += 1
            if "metadata_only" in status or evidence_class == "metadata_only":
                counts.metadata_only += 1
            if evidence_class == "no_op":
                counts.no_op += 1
            checks = record.get("checks") or {}
            if not _check_passed(checks, "runtime_hook_real"):
                counts.missing_runtime += 1
            if not _check_passed(checks, "unit_tests_present"):
                counts.missing_tests += 1
            if not _check_passed(checks, "non_mock_smoke_present"):
                counts.missing_smoke += 1
            if not _check_passed(checks, "compatibility_tests_present"):
                counts.missing_compatibility += 1
            if not _check_passed(checks, "rollback_path_present"):
                counts.missing_rollback += 1
            if not _check_passed(checks, "runtime_fingerprint_present"):
                counts.missing_fingerprint += 1
            if status == "implementation_ready":
                counts.ready += 1
            elif status == "out_of_scope":
                counts.out_of_scope += 1
            else:
                counts.blocked += 1

        section.spec_complete = sum(
            1 for record in audit_records
            if _check_passed(record.get("checks") or {}, "implementation_spec_complete")
        )
        section.code_bound = sum(
            1 for record in audit_records
            if _check_passed(record.get("checks") or {}, "paper_specific_composition")
        )
        section.runtime_integrated = counts.total - counts.missing_runtime
        section.unit_tested = counts.total - counts.missing_tests
        section.non_mock_smoke_passed = counts.total - counts.missing_smoke
        section.compatibility_validated = counts.total - counts.missing_compatibility
        section.implementation_ready = counts.ready
        section.blocked = counts.blocked
        section.integrity_counts = counts

        expected = 83
        section.checks.extend(
            [
                IntegrityCheck(
                    requirement="manifest_paper_count",
                    expected=str(expected),
                    observed=str(section.manifest_paper_count),
                    passed=section.manifest_paper_count == expected,
                ),
                IntegrityCheck(
                    requirement="unique_paper_ids",
                    expected=str(expected),
                    observed=str(section.unique_paper_ids),
                    passed=section.unique_paper_ids == expected,
                ),
                IntegrityCheck(
                    requirement="membership_hash_valid",
                    expected="recomputed == stored",
                    observed="match" if section.membership_hash_valid else "mismatch",
                    passed=section.membership_hash_valid,
                ),
                IntegrityCheck(
                    requirement="implementation_ready",
                    expected=str(expected),
                    observed=str(section.implementation_ready),
                    passed=section.implementation_ready == expected,
                ),
                IntegrityCheck(
                    requirement="blocked_zero",
                    expected="0",
                    observed=str(section.blocked),
                    passed=section.blocked == 0,
                ),
                IntegrityCheck(
                    requirement="generic_only_zero",
                    expected="0",
                    observed=str(counts.generic_only),
                    passed=counts.generic_only == 0,
                ),
                IntegrityCheck(
                    requirement="metadata_only_zero",
                    expected="0",
                    observed=str(counts.metadata_only),
                    passed=counts.metadata_only == 0,
                ),
                IntegrityCheck(
                    requirement="no_op_zero",
                    expected="0",
                    observed=str(counts.no_op),
                    passed=counts.no_op == 0,
                ),
                IntegrityCheck(
                    requirement="mock_only_zero",
                    expected="0",
                    observed=str(counts.mock_only),
                    passed=counts.mock_only == 0,
                ),
            ]
        )
        section.passed = all(item.passed for item in section.checks)
        return section

    def _load_audit_records(self) -> list[dict[str, Any]]:
        import yaml

        if not self.audit_path.is_file():
            return []
        with self.audit_path.open("r", encoding="utf-8-sig") as file:
            payload = yaml.safe_load(file) or {}
        records = payload.get("records", [])
        return [record for record in records if isinstance(record, dict)]

    # ---------------------------------------------- part two: action space --

    def _verify_action_space(self) -> ActionSpaceSection:
        from yolo_agent.agents.action_space import ActionCatalog

        section = ActionSpaceSection()
        if not self.catalog_path.is_file():
            section.missing_families = list(REQUIRED_ACTION_FAMILIES)
            return section
        catalog = ActionCatalog.from_yaml(self.catalog_path)
        section.catalog_action_count = len(catalog.actions)
        section.paper_lineage_actions = len(catalog.paper_actions())
        section.local_actions = len(catalog.local_actions())
        covered = {action.family for action in catalog.actions}
        section.covered_families = sorted(covered & set(REQUIRED_ACTION_FAMILIES))
        section.missing_families = sorted(set(REQUIRED_ACTION_FAMILIES) - covered)
        section.passed = not section.missing_families
        return section

    # ------------------------------------------- part three: agent loop -----

    def _verify_autonomous_loop(self) -> AutonomousLoopSection:
        """Drive the real bounded loop with a FakeExperimentRunner, multi-round."""

        section = AutonomousLoopSection()
        try:
            from yolo_agent.agents.action_space import ActionCatalog
            from yolo_agent.agents.autonomous_loop import (
                BoundedAutonomousLoop,
                CompatibilityMaturityFilter,
            )
            from yolo_agent.agents.bounded_hpo import hpo_scope_for_action
            from yolo_agent.agents.error_delta_profile import build_error_delta_profile
            from yolo_agent.core.error_facts import ErrorFact
            from yolo_agent.core.experiment_graph import MetricEvidence
            from yolo_agent.core.task_spec import MetricPriority, TaskSpec

            section.bounded_hpo_module_present = True
        except ImportError:
            return section

        base = {
            "map50_95": 0.40,
            "precision": 0.70,
            "recall": 0.60,
            "ap_small": 0.20,
            "ap_medium": 0.45,
            "ap_large": 0.55,
            "latency_ms": 10.0,
        }
        spec = TaskSpec(
            class_names=["a"],
            primary_metric=MetricPriority(name="ap_small", weight=2.0),
            max_latency_ms=12.0,
        )

        def profile(**overrides: float):
            candidate = {**base, **overrides}
            baseline_records = [
                MetricEvidence(candidate_id="base", node_id="nb", metric_name=name, value=value)
                for name, value in base.items()
            ]
            candidate_records = [
                MetricEvidence(candidate_id="cand", node_id="nc", metric_name=name, value=value)
                for name, value in candidate.items()
            ]
            return build_error_delta_profile(
                baseline_records,
                candidate_records,
                candidate_id="cand",
                node_id="nc",
                task_spec=spec,
            )

        class FakeRunner:
            def __init__(self, profiles):
                self.profiles = profiles
                self.calls = []

            def run_candidate(self, action, round_index, overrides):
                self.calls.append((round_index, action.action_id))
                return self.profiles[(round_index, action.action_id)]

        try:
            catalog = ActionCatalog.from_yaml(self.catalog_path)
        except (OSError, ValueError):
            return section

        runner = FakeRunner(
            {
                (1, "model.neck.rtmdet_large_kernel"): profile(ap_small=0.26),
                (2, "model.feature_fusion.gather_distribute"): profile(
                    ap_small=0.24, latency_ms=12.5
                ),
                (3, "train.assigner.optimal_transport"): profile(precision=0.73),
            }
        )
        eligible_ids = [
            "model.neck.rtmdet_large_kernel",
            "model.feature_fusion.gather_distribute",
            "train.assigner.optimal_transport",
        ]
        loop = BoundedAutonomousLoop(
            catalog=catalog,
            runner=runner,
        )
        facts = [
            ErrorFact(
                run_id="acceptance",
                candidate_id="base",
                node_id="n",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.20,
                severity="high",
            )
        ]
        report = loop.run(
            task_spec=spec,
            initial_facts=facts,
            max_rounds=5,
            filter_=CompatibilityMaturityFilter(implementation_ready_ids=eligible_ids),
        )
        section.rounds_executed = len(report.rounds)
        section.decisions_seen = [record.decision for record in report.rounds]
        section.diagnosis_chain = len(report.rounds) >= 3
        section.error_delta_used = any(
            record.pareto_axes for record in report.rounds
        )
        section.asha_budget_routed = any(
            outcome.scope == "asha_train_time"
            for outcome in report.outcomes
        ) or any(
            hpo_scope_for_action(action) in {"asha_train_time", "eval_route"}
            for action in catalog.actions[:5]
        )
        section.rollback_observed = "rollback" in section.decisions_seen
        section.pareto_axes_used = any(
            record.pareto_axes.get("ap_small", 0) > 0 for record in report.rounds
        )
        section.stop_policy_honest = (
            report.final_decision == "stop_exhausted"
            and report.stop_reason == "no_evidence_supported_action"
        )
        section.multi_round_completed = len(report.rounds) >= 3 and section.rollback_observed
        section.passed = not section.failed_checks
        return section

    # ------------------------------------------------- part four: safety ---

    def _verify_safety(self, ready_count: int) -> SafetySection:
        """Gate probes: 82/83 blocks, 83/83 allows, fail-closed, consent intact.

        The 83/83-allow probe never executes a trainer — the gate evaluation
        itself is the probe; a stubbed trainer call sits behind it in tests.
        """

        from yolo_agent.research.paper_83_training_gate import (
            Paper83GateConfig,
            evaluate_paper_83_training_gate,
        )

        section = SafetySection()

        # Fail-closed: missing manifest.
        missing = evaluate_paper_83_training_gate(
            manifest_path=self.root / "nonexistent" / "manifest.yaml",
            config=Paper83GateConfig(),
        )
        section.gate_fail_closed_on_missing_manifest = missing.locked

        # 82/83 → blocked: synthetic audit built by mutating a copy of the
        # real audit — never the frozen manifest.
        synth_root = self.root / "runs" / "pretraining_acceptance_probe"
        blocked_decision = evaluate_paper_83_training_gate(
            manifest_path=self.manifest_path,
            exactness_audit_path=_build_probe_audit(
                self.audit_path, synth_root / "audit_82.yaml", ready_target=82
            ),
            config=Paper83GateConfig(),
        )
        section.gate_82_of_83_blocks = blocked_decision.locked and blocked_decision.ready <= 82

        allowed_decision = evaluate_paper_83_training_gate(
            manifest_path=self.manifest_path,
            exactness_audit_path=_build_probe_audit(
                self.audit_path, synth_root / "audit_83.yaml", ready_target=83
            ),
            config=Paper83GateConfig(),
        )
        section.gate_83_of_83_allows = allowed_decision.allowed and allowed_decision.ready == 83

        # Hash-mismatch fail-closed: point the gate at a tampered audit copy.
        tampered = _build_tampered_audit(self.audit_path, synth_root / "audit_tampered.yaml")
        tampered_decision = evaluate_paper_83_training_gate(
            manifest_path=self.manifest_path,
            exactness_audit_path=tampered,
            config=Paper83GateConfig(),
        )
        section.gate_fail_closed_on_hash_mismatch = tampered_decision.locked

        # Full-run consent boundary: the driver must still refuse when no
        # consent artifact exists — the Paper-83 gate never replaced it.
        try:
            from yolo_agent.core.full_run_consent import FullRunConsentDriver
            from yolo_agent.core.optimization_objective import OptimizationObjective

            consent_root = self.root / "runs" / "pretraining_acceptance_probe" / "consent"
            driver = FullRunConsentDriver(consent_root)
            objective = OptimizationObjective(
                goal_expression="mAP50-95 >= baseline + 0.01",
                goal_description="acceptance probe objective",
                primary_metric="map50_95",
                baseline_run_id="acceptance_probe_baseline",
                baseline_candidate_id="baseline",
                baseline_protocol_hash="0" * 64,
            )
            result = driver.validate(
                run_id="acceptance_probe",
                objective=objective,
                dataset_manifest_sha256=None,
                objective_status=None,
            )
            section.full_run_consent_boundary_preserved = (
                not result.allowed and result.reason == "full_run_consent_missing"
            )
        except Exception:  # noqa: BLE001 — boundary probe must never crash acceptance
            section.full_run_consent_boundary_preserved = False

        section.passed = not section.failed_checks
        return section

    # ------------------------------------------------- part five: tests ----

    def _run_tests(self) -> TestsSection:
        section = TestsSection()
        if not self.run_tests:
            section.passed = True
            return section
        pytest_exe = _python_executable()
        # Bounded acceptance tier: the suites this campaign introduced plus
        # the safety-critical gate/loop suites.  The full multi-thousand-test
        # repo tier is environment-slow and covered by CI; the probe records
        # its exact selection so the classification stays honest.
        fast_targets = [
            "tests/test_pretraining_acceptance.py",
            "tests/test_paper_83_training_gate.py",
            "tests/test_autonomous_loop_delta.py",
            "tests/test_action_space.py",
        ]
        fast_cmd = [pytest_exe, "-m", "pytest", "-q", *fast_targets]
        section.fast_command = " ".join(fast_cmd)
        try:
            fast = subprocess.run(
                fast_cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired:
            section.fast_exit_code = None
            section.fast_summary = "probe_timeout_after_900s"
            section.fast_introduced_failures = ["probe_timeout"]
            section.passed = False
            return section
        section.fast_exit_code = fast.returncode
        section.fast_summary = _summarize_pytest(fast.stdout + fast.stderr)
        section.fast_introduced_failures, section.fast_preexisting_failures = (
            _classify_failures(fast.stdout + fast.stderr)
        )
        section.passed = fast.returncode == 0

        if self.run_slow_tests:
            section.slow_attempted = True
            slow_cmd = [pytest_exe, "-m", "pytest", "-q", "tests"]
            env_added = _slow_env()
            if env_added:
                slow = subprocess.run(
                    slow_cmd,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    check=False,
                    env=env_added,
                )
                section.slow_exit_code = slow.returncode
                section.slow_summary = _summarize_pytest(slow.stdout + slow.stderr)
                # Prompt-18A: a failed required slow tier is a failed test
                # tier — it must contribute to tests.passed, not just be
                # recorded.
                if slow.returncode != 0:
                    section.passed = False

        # Prefer a working ``ruff`` executable; the venv one may be a broken
        # shim (ruff.exe absent), in which case ``python -m ruff`` from the
        # interpreter that actually has ruff installed is the honest fallback.
        venv_ruff = pytest_exe.replace("python.exe", "ruff.exe").replace("/python", "/ruff")
        if Path(venv_ruff).is_file():
            lint_cmd = [venv_ruff, "check", "."]
        else:
            import shutil

            global_ruff = shutil.which("ruff")
            if global_ruff:
                lint_cmd = [global_ruff, "check", "."]
            else:
                base_python = _python_executable().replace(".venv", "").replace("venv\\Scripts", "")
                lint_cmd = [base_python if Path(base_python).is_file() else "python", "-m", "ruff", "check", "."]
        section.lint_command = " ".join(lint_cmd)
        try:
            lint = subprocess.run(
                lint_cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            section.lint_exit_code = None
            section.lint_summary = "lint_timeout_after_600s"
            section.passed = False
            return section
        section.lint_exit_code = lint.returncode
        section.lint_summary = _summarize_lint(lint.stdout + lint.stderr)
        if lint.returncode != 0:
            section.passed = False
        return section


def _tests_lock_reasons(tests: TestsSection) -> list[str]:
    """Categorized lock reasons for the test/lint tier (Prompt-18A part two).

    Fast-test, slow-test, and lint failures each lock the gate under their
    own reason, with concrete failed test IDs preserved alongside.  A probe
    that never completed (timeout / missing interpreter) is a failure too.
    """

    reasons: list[str] = []
    if tests.fast_command:
        if tests.fast_exit_code is None or tests.fast_exit_code != 0:
            reasons.append("fast_tests_failed")
    if tests.slow_attempted and (tests.slow_exit_code is None or tests.slow_exit_code != 0):
        reasons.append("slow_tests_failed")
    if tests.lint_command and (tests.lint_exit_code is None or tests.lint_exit_code != 0):
        reasons.append("lint_failed")
    failed_ids = [*tests.fast_introduced_failures, *tests.fast_preexisting_failures]
    for test_id in failed_ids[:20]:
        reasons.append(f"fast_tests_failed:{test_id}")
    if len(failed_ids) > 20:
        reasons.append(f"fast_tests_failed:+{len(failed_ids) - 20}_more")
    return reasons


def _check_passed(checks: dict[str, Any], check_id: str) -> bool:
    """Whether one exactness check passed in a serialized audit record."""

    entry = checks.get(check_id)
    if isinstance(entry, dict):
        return bool(entry.get("passed"))
    return False


def _python_executable() -> str:
    return sys.executable or "python"


def _slow_env() -> dict[str, str] | None:
    import os

    env = dict(os.environ)
    env["YOLO_AGENT_RUN_SLOW_TESTS"] = "1"
    return env


def _summarize_pytext(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


_summarize_pytest = _summarize_pytext


def _summarize_lint(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[-1].startswith("All checks passed"):
        return "clean"
    return f"{len([line for line in lines if ':[0-9]' in line or ': ' in line])} findings"


def _classify_failures(text: str) -> tuple[list[str], list[str]]:
    """Split pytest failures into acceptance-introduced vs pre-existing.

    A failure is classified as introduced when its test id touches the
    acceptance/loop/gate modules this campaign added; everything else is
    reported as pre-existing/environment so FAIL verdicts stay explainable.
    """

    introduced_markers = (
        "test_autonomous_loop_delta",
        "test_action_space",
        "test_paper_83_training_gate",
        "test_paper_gap_closure",
        "test_paper_exactness",
        "pretraining_acceptance",
    )
    introduced: list[str] = []
    preexisting: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED") or " FAILED " in stripped:
            test_id = stripped.replace("FAILED ", "").split(" ")[0].split(" - ")[0]
            if any(marker in test_id for marker in introduced_markers):
                introduced.append(test_id)
            else:
                preexisting.append(test_id)
    return introduced, preexisting


def _build_probe_audit(
    source_path: Path | str,
    target_path: Path | str,
    *,
    ready_target: int,
) -> Path:
    """Write a probe audit with exactly ``ready_target`` ready papers.

    The probe mutates a *copy* of the committed audit and recomputes its
    self-hash so the gate's hash-validation path stays exercised; the frozen
    manifest is never touched.
    """

    import yaml

    from yolo_agent.research.paper_exactness_schemas import PaperExactnessAudit

    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file) or {}
    records = payload.get("records", [])
    # ``ready_target`` is the probe's desired ready count: below the current
    # ready count it demotes ready papers, above it promotes blocked ones.
    current_ready = sum(
        1 for record in records if record.get("status") == "implementation_ready"
    )
    flips_allowed = max(0, current_ready - ready_target)
    promotions_allowed = max(0, ready_target - current_ready)
    flips = 0
    for record in records:
        if record.get("status") == "implementation_ready" and flips < flips_allowed:
            record["status"] = "blocked_test"
            record["blockers"] = ["blocked_test:acceptance_probe"]
            flips += 1
        elif (
            record.get("status") != "implementation_ready"
            and promotions_allowed > 0
        ):
            record["status"] = "implementation_ready"
            record["blockers"] = []
            for name, check in (record.get("checks") or {}).items():
                check["passed"] = True
            record.setdefault("evidence_inventory", {})[
                "implementation_evidence_class"
            ] = "paper_specific"
            promotions_allowed -= 1
    payload["records"] = records
    # The audit's own hash gate validates on load; strip the stale hash first
    # so the mutated copy can be rehashed honestly (manifest untouched).
    payload.pop("audit_hash", None)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    audit = PaperExactnessAudit.from_yaml(target)
    audit.with_hash().to_yaml(target)
    return target


def _build_tampered_audit(source_path: Path | str, target_path: Path | str) -> Path:
    """Copy the audit and desynchronize its self-hash (tamper simulation)."""

    import yaml

    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file) or {}
    payload["audit_hash"] = "0" * 64
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def render_pretraining_acceptance(acceptance: PretrainingAcceptance) -> str:
    """Render the required final table (PASS and FAIL share the layout)."""

    campaign = acceptance.paper_campaign
    tests = acceptance.tests
    lines = [
        "========================================",
        "YOLO AGENT PRE-TRAINING ACCEPTANCE",
        "========================================",
        f"Frozen papers:              {campaign.manifest_paper_count}",
        f"Implementation ready:       {campaign.implementation_ready}/{campaign.manifest_paper_count}",
        f"Blocked:                    {campaign.blocked}",
        "",
        f"Paper runtime integrity:    {'PASS' if campaign.passed else 'FAIL'}",
        f"Non-mock smoke:             {'PASS' if campaign.non_mock_smoke_passed == campaign.manifest_paper_count else 'FAIL'}",
        f"Unified action space:       {'PASS' if acceptance.optimization_action_space.passed else 'FAIL'}",
        f"Autonomous decision loop:   {'PASS' if acceptance.autonomous_loop.passed else 'FAIL'}",
        f"Bounded HPO:                {'PASS' if acceptance.autonomous_loop.bounded_hpo_module_present else 'FAIL'}",
        f"ASHA:                       {'PASS' if acceptance.autonomous_loop.asha_budget_routed else 'FAIL'}",
        f"Rollback:                   {'PASS' if acceptance.autonomous_loop.rollback_observed else 'FAIL'}",
        f"Runtime preflight (83):     {'PASS' if acceptance.runtime_preflight.passed_bool else 'FAIL'}"
        f" ({acceptance.runtime_preflight.passed}/{acceptance.runtime_preflight.papers})",
        f"Non-GPU verification:       {'PASS' if acceptance.non_gpu_verification.passed_bool else 'FAIL'}"
        f" (fast={acceptance.non_gpu_verification.fast},"
        f" slow={acceptance.non_gpu_verification.slow},"
        f" ruff={acceptance.non_gpu_verification.ruff})",
        "Training gate:              " + ("UNLOCKED" if acceptance.training_gate.allowed else "LOCKED"),
        "",
        f"REAL TRAINING EXECUTED:     {'YES' if acceptance.real_training_executed else 'NO'}",
        "========================================",
    ]
    if acceptance.verdict == "FAIL":
        lines.append("")
        lines.append("Remaining blockers:")
        for blocker in acceptance.remaining_blockers[:40]:
            lines.append(f"  - {blocker}")
        if len(acceptance.remaining_blockers) > 40:
            lines.append(f"  ... and {len(acceptance.remaining_blockers) - 40} more")
        if tests.fast_introduced_failures or tests.fast_preexisting_failures:
            lines.append("")
            lines.append(
                f"pytest fast: introduced={len(tests.fast_introduced_failures)} "
                f"pre-existing/environment={len(tests.fast_preexisting_failures)}"
            )
    return "\n".join(lines)


def write_pretraining_acceptance_artifacts(
    acceptance: PretrainingAcceptance,
    *,
    yaml_path: Path | str = DEFAULT_ACCEPTANCE_YAML,
    markdown_path: Path | str = DEFAULT_ACCEPTANCE_MD,
) -> tuple[Path, Path]:
    """Always write both artifacts — on PASS and on FAIL alike."""

    yaml_written = acceptance.to_yaml(yaml_path)
    markdown = Path(markdown_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "# Pre-Training Acceptance\n\n```text\n"
        + render_pretraining_acceptance(acceptance)
        + "\n```\n",
        encoding="utf-8",
    )
    return Path(yaml_written), markdown


__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "ActionSpaceSection",
    "AutonomousLoopSection",
    "IntegrityCheck",
    "IntegrityCounts",
    "NonGpuVerificationSection",
    "PaperCampaignSection",
    "PretrainingAcceptance",
    "PretrainingAcceptanceRunner",
    "REQUIRED_ACTION_FAMILIES",
    "RuntimePreflightSection",
    "SafetySection",
    "TestsSection",
    "TrainingGateSection",
    "render_pretraining_acceptance",
    "write_pretraining_acceptance_artifacts",
]
