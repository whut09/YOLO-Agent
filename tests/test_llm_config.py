"""LLM decision config tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.resources import ResourcePaths


def test_redacted_llm_decision_example_is_committable() -> None:
    """Committed LLM config should not expose local model credentials."""
    config = LLMDecisionConfig.from_yaml(ResourcePaths.LLM_DECISION_EXAMPLE)

    assert config.enabled is False
    assert config.provider == "XX"
    assert config.model == "XX"
    assert config.api_key_env == "XX"
    assert config.executable_decisions_allowed is False
    assert "direct_training_execution" in config.blocked_outputs


def test_local_llm_decision_config_can_enable_codex_model(tmp_path: Path) -> None:
    """Local ignored config should be able to name the decision-analysis model."""
    path = tmp_path / "llm_decision.local.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-5.5",
                "model_alias": "codex-gpt5.5",
                "api_key_env": "OPENAI_API_KEY",
                "decision_role": "proposal_generator_only",
                "allowed_outputs": ["policy_proposals", "doctor_report_draft"],
                "blocked_outputs": ["direct_experiment_approval"],
                "prompt_contract": {
                    "system_summary": "Draft proposals only.",
                    "required_inputs": ["error_facts"],
                    "required_output_schema": {"selected_actions": "list[object]"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_llm_decision_config(path)

    assert config.can_generate_proposals is True
    assert config.executable_decisions_allowed is False
    assert config.model == "gpt-5.5"
    assert config.model_alias == "codex-gpt5.5"
