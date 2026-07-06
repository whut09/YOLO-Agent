"""Compact policy-memory context for LLM proposal generation."""

from __future__ import annotations

from typing import Any

from yolo_agent.core.policy_memory import PolicyMemoryStore, PolicyMemorySummary


def build_policy_memory_context(
    store: PolicyMemoryStore,
    *,
    dataset_version: str | None = None,
    target_metrics: list[str] | None = None,
    target_actions: list[str] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Return a compact, prompt-safe summary of historical action effects."""
    summaries = store.summarize(dataset_version=dataset_version)
    if not summaries and dataset_version is not None:
        summaries = store.summarize()
    filtered = _filter_summaries(summaries, target_metrics or [], target_actions or [])
    ranked = sorted(filtered, key=_summary_rank)[:limit]
    return {
        "source": store.path.as_posix(),
        "dataset_version": dataset_version,
        "target_metrics": list(dict.fromkeys(target_metrics or [])),
        "target_actions": list(dict.fromkeys(target_actions or [])),
        "summary_count": len(ranked),
        "historical_effects": [_summary_payload(summary) for summary in ranked],
        "usage_rules": [
            "Policy memory is prior experience, not executable approval.",
            "Prefer actions with positive mean_effect_delta and acceptable latency/model-size cost.",
            "Defer or request evidence for actions with low confidence, negative effect, or high cost.",
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


def _summary_rank(summary: PolicyMemorySummary) -> tuple[int, float, float]:
    confidence_rank = {
        "high": summary.confidence_counts.get("high", 0),
        "medium": summary.confidence_counts.get("medium", 0),
        "low": summary.confidence_counts.get("low", 0),
    }
    effect = summary.mean_effect_delta if summary.mean_effect_delta is not None else -999.0
    latency_penalty = abs(summary.mean_latency_delta_pct or 0.0)
    return (-(confidence_rank["high"] * 3 + confidence_rank["medium"] * 2 + confidence_rank["low"]), -effect, latency_penalty)


def _summary_payload(summary: PolicyMemorySummary) -> dict[str, Any]:
    return {
        "action": summary.action,
        "target": summary.target,
        "metric_name": summary.metric_name,
        "record_count": summary.record_count,
        "mean_delta": summary.mean_delta,
        "mean_effect_delta": summary.mean_effect_delta,
        "mean_latency_delta_pct": summary.mean_latency_delta_pct,
        "mean_model_size_delta_pct": summary.mean_model_size_delta_pct,
        "confidence_counts": summary.confidence_counts,
        "latest_record_ids": summary.latest_record_ids,
        "interpretation": _interpretation(summary),
    }


def _interpretation(summary: PolicyMemorySummary) -> str:
    effect = summary.mean_effect_delta
    latency = summary.mean_latency_delta_pct
    if effect is None:
        return "unknown_effect"
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
