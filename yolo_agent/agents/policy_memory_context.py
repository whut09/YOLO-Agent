"""Compact policy-memory context for LLM proposal generation."""

from __future__ import annotations

from typing import Any

from yolo_agent.core.policy_memory import PolicyMemoryStore, PolicyMemorySummary


def build_policy_memory_context(
    store: PolicyMemoryStore,
    *,
    dataset_version: str | None = None,
    dataset_signature: str | None = None,
    scenario: str | None = None,
    model_family: str | None = None,
    target_metrics: list[str] | None = None,
    target_actions: list[str] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Return a compact, prompt-safe summary of historical action effects."""
    summaries = store.summarize(
        dataset_signature=dataset_signature or dataset_version,
        scenario=scenario,
        model_family=model_family,
    )
    if not summaries and dataset_version is not None:
        summaries = store.summarize()
    filtered = _filter_summaries(summaries, target_metrics or [], target_actions or [])
    ranked = sorted(filtered, key=_summary_rank)[:limit]
    return {
        "source": store.path.as_posix(),
        "dataset_version": dataset_version,
        "dataset_signature": dataset_signature,
        "scenario": scenario,
        "model_family": model_family,
        "target_metrics": list(dict.fromkeys(target_metrics or [])),
        "target_actions": list(dict.fromkeys(target_actions or [])),
        "summary_count": len(ranked),
        "historical_effects": [_summary_payload(summary) for summary in ranked],
        "usage_rules": [
            "Policy memory is prior experience, not executable approval.",
            "Prefer actions with positive mean_effect_delta and acceptable latency/model-size cost.",
            "Use posterior confidence intervals and seed counts; one failed pilot only lowers confidence.",
            "Reject an action family only after repeated comparable evidence has a non-positive upper confidence bound.",
            "Use pilot-to-full correlation when estimating whether pilot gains are likely to transfer.",
            "Compatibility, evidence, budget, and single-variable gates override policy memory.",
        ],
    }


def _filter_summaries(
    summaries: list[PolicyMemorySummary],
    target_metrics: list[str],
    target_actions: list[str],
) -> list[PolicyMemorySummary]:
    metric_tokens = {_normalize_metric(metric) for metric in target_metrics if metric}
    action_tokens = {str(action) for action in target_actions if str(action)}
    if not metric_tokens and not action_tokens:
        return summaries
    filtered: list[PolicyMemorySummary] = []
    for summary in summaries:
        metric = _normalize_metric(summary.metric_name or "")
        target = _normalize_metric(summary.target or "")
        action = summary.action
        if metric_tokens and (metric in metric_tokens or any(token in target for token in metric_tokens)):
            filtered.append(summary)
            continue
        if action_tokens and action in action_tokens:
            filtered.append(summary)
    return filtered


def _summary_rank(summary: PolicyMemorySummary) -> tuple[int, float, float, float]:
    posterior_rank = {"high": 3, "medium": 2, "low": 1}[summary.posterior_confidence]
    effect = summary.mean_effect_delta if summary.mean_effect_delta is not None else -999.0
    latency_penalty = abs(summary.mean_latency_delta_pct or 0.0)
    return (-posterior_rank, -summary.effective_sample_size, -effect, latency_penalty)


def _summary_payload(summary: PolicyMemorySummary) -> dict[str, Any]:
    return {
        "action": summary.action,
        "action_fingerprint_sha256": summary.action_fingerprint_sha256,
        "posterior_key_sha256": summary.posterior_key_sha256,
        "action_fingerprint": (
            summary.action_fingerprint.model_dump(mode="json")
            if summary.action_fingerprint is not None
            else None
        ),
        "target": summary.target,
        "metric_name": summary.metric_name,
        "record_count": summary.record_count,
        "mean_delta": summary.mean_delta,
        "mean_effect_delta": summary.mean_effect_delta,
        "mean_latency_delta_pct": summary.mean_latency_delta_pct,
        "mean_model_size_delta_pct": summary.mean_model_size_delta_pct,
        "mean_target_metric_gain": summary.mean_target_metric_gain,
        "mean_error_fact_gain": summary.mean_error_fact_gain,
        "effect_variance": summary.effect_variance,
        "confidence_interval_95": summary.confidence_interval_95,
        "posterior_confidence": summary.posterior_confidence,
        "seed_count": summary.seed_count,
        "success_count": summary.success_count,
        "failure_count": summary.failure_count,
        "pilot_to_full_correlation": summary.pilot_to_full_correlation,
        "pilot_to_full_gain_ratio": summary.pilot_to_full_gain_ratio,
        "latency_cost_distribution": summary.latency_cost_distribution.model_dump(mode="json"),
        "model_size_cost_distribution": summary.model_size_cost_distribution.model_dump(mode="json"),
        "mean_dataset_similarity_weight": summary.mean_dataset_similarity_weight,
        "effective_sample_size": summary.effective_sample_size,
        "confidence_counts": summary.confidence_counts,
        "latest_record_ids": summary.latest_record_ids,
        "interpretation": _interpretation(summary),
    }


def _interpretation(summary: PolicyMemorySummary) -> str:
    effect = summary.mean_effect_delta
    latency = summary.mean_latency_delta_pct
    if effect is None:
        return "unknown_effect"
    interval = summary.confidence_interval_95
    if summary.posterior_confidence == "low":
        return "insufficient_repeated_evidence"
    if interval is not None and interval[1] <= 0:
        return "repeated_negative_effect"
    if interval is not None and interval[0] <= 0 <= interval[1]:
        return "uncertain_effect"
    if effect > 0 and latency is not None and latency > 15:
        return "positive_effect_high_latency_cost"
    if effect > 0:
        return "positive_effect"
    if effect < 0:
        return "negative_effect"
    return "neutral_effect"


def _normalize_metric(value: str) -> str:
    lowered = value.lower().replace("map_small", "ap_small").replace("mAP_small", "ap_small")
    return lowered.replace(" ", "_")


__all__ = ["build_policy_memory_context"]
