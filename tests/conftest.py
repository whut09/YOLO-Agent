"""Shared pytest policy for fast regression and explicitly gated test tiers."""

from __future__ import annotations

import os

import pytest


# These modules exercise multi-stage runners, large paper cohorts, or CPU model
# graphs. They remain part of the suite but are opt-in so ordinary edits get a
# useful result quickly.
SLOW_TEST_MODULES = frozenset(
    {
        "test_all_83_paper_execution_acceptance.py",
        "test_all_83_readiness_and_asha_acceptance.py",
        "test_all_83_real_training_readiness.py",
        "test_assignment_pilot_gate.py",
        "test_auto_optimization_loop.py",
        "test_awesome_snapshot_builder.py",
        "test_complete_paper_asha_cohort.py",
        "test_component_certification_runner.py",
        "test_component_execution_bridge.py",
        "test_data_pipeline_snapshot.py",
        "test_domain_paper_routes.py",
        "test_distillation_mechanisms.py",
        "test_distillation_paper_routes.py",
        "test_execution_failure_cli.py",
        "test_execution_failure_recovery.py",
        "test_gpu_certification.py",
        "test_hard_negative_bootstrap.py",
        "test_independent_component_certification.py",
        "test_independent_component_router.py",
        "test_independent_runtime_adapters.py",
        "test_loop_policy_evaluator.py",
        "test_loop_status.py",
        "test_mechanism_cluster_report.py",
        "test_paper_execution_inventory.py",
        "test_paper_execution_requirements.py",
        "test_optimize_runner.py",
        "test_orchestrator.py",
        "test_paper_candidate_routing_acceptance.py",
        "test_paper_candidate_orchestrator.py",
        "test_paper_component_gate.py",
        "test_paper_method_evidence_catalog.py",
        "test_paper_method_evidence.py",
        "test_paper_method_profiles.py",
        "test_paper_mechanism_clusterer.py",
        "test_paper_readiness_pipeline.py",
        "test_paper_readiness.py",
        "test_paper_recipe_auto_training_state_machine.py",
        "test_paper_training_readiness.py",
        "test_paper_training_cohort.py",
        "test_paper_adapter_certification_factory.py",
        "test_paper_auto_optimization_acceptance.py",
        "test_quality_alignment_auxiliary_losses.py",
        "test_quality_loss_certification.py",
        "test_quality_candidate_contract.py",
        "test_recipe_ablation_planner.py",
        "test_research_production_pipeline.py",
        "test_research_maturity_overlay_snapshot.py",
        "test_note_parser.py",
        "test_persistent_paper_coverage_ledger.py",
        "test_ultralytics_plugin_bridge.py",
        "test_ultralytics_training.py",
        "test_yolo26_distillation.py",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    """Keep tests deterministic even when a developer has local LLM credentials."""
    os.environ.setdefault("YOLO_AGENT_DISABLE_LOCAL_LLM", "1")
    # Most tests use small CPU tensors. Avoid paying a thread-pool startup cost
    # for each tiny operation; callers can raise this for local benchmarks.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("YOLO_AGENT_TEST_TORCH_THREADS", "1")))
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError, ValueError):
        # Torch is optional for metadata-only test environments. A runtime
        # error here means another plugin initialized its pools first.
        pass
    config.addinivalue_line(
        "markers",
        "slow: multi-stage CPU/mock or large-cohort test (opt in with --run-slow)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-gpu",
        action="store_true",
        default=False,
        help="run tests marked real_gpu (may train models on CUDA)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow CPU/mock integration tests (does not enable real GPU tests)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption("--run-slow") or os.getenv("YOLO_AGENT_RUN_SLOW_TESTS") == "1"
    if not run_slow:
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            if item.path.name in SLOW_TEST_MODULES or "slow" in item.keywords:
                item.add_marker(pytest.mark.slow)
                deselected.append(item)
            else:
                selected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    enabled = config.getoption("--run-real-gpu") or os.getenv("YOLO_AGENT_RUN_REAL_GPU") == "1"
    if enabled:
        return
    marker = pytest.mark.skip(reason="real GPU acceptance is opt-in; pass --run-real-gpu")
    for item in items:
        if "real_gpu" in item.keywords:
            item.add_marker(marker)
