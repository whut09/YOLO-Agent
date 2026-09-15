"""Pseudo-label confidence filtering for domain-adaptive and semi-supervised routes.

The pseudo-label branch declares a ``confidence_threshold`` but consumed the
labels unfiltered, which let low-quality teacher guesses into the consistency
loss.  This module owns the real filter: a sigmoid/softmax-agnostic confidence
score vector is thresholded, at least one pseudo-label must survive per batch,
and the returned keep-mask is what the runtime loss must index with.  Masking
is the only output — no silent score renormalization, no per-class heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class PseudoLabelFilterError(ValueError):
    """Raised when the pseudo-label contract is violated."""


@dataclass(frozen=True)
class PseudoLabelFilterResult:
    """The keep-mask and the scores it selected, for loss weighting."""

    keep_mask: torch.Tensor
    kept_scores: torch.Tensor
    kept_labels: torch.Tensor
    threshold: float

    @property
    def kept_count(self) -> int:
        return int(self.keep_mask.sum().item())


def filter_pseudo_labels(
    pseudo_labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    confidence_threshold: float,
) -> PseudoLabelFilterResult:
    """Threshold pseudo-labels by teacher confidence.

    ``scores`` must be one confidence value per pseudo-label in ``[0, 1]``
    (already sigmoided or max-softmaxed by the caller).  Labels whose score is
    at or above ``confidence_threshold`` survive.  An empty batch raises
    instead of silently returning zero positives.
    """

    if confidence_threshold <= 0.0 or confidence_threshold >= 1.0:
        raise PseudoLabelFilterError(
            f"confidence threshold must be in (0, 1), got {confidence_threshold}"
        )
    labels = pseudo_labels.detach()
    values = scores.detach().float()
    if values.ndim != 1:
        raise PseudoLabelFilterError(
            f"pseudo-label scores must be one per sample, got shape {tuple(values.shape)}"
        )
    flat_labels = labels.reshape(values.shape[0], -1)
    if flat_labels.shape[0] != values.shape[0]:
        raise PseudoLabelFilterError(
            "pseudo-label count must match score count: "
            f"{flat_labels.shape[0]} vs {values.shape[0]}"
        )
    keep_mask = values >= confidence_threshold
    if bool(keep_mask.sum().item()) == 0:
        raise PseudoLabelFilterError(
            f"confidence threshold {confidence_threshold} kept no pseudo-labels"
        )
    return PseudoLabelFilterResult(
        keep_mask=keep_mask,
        kept_scores=values[keep_mask],
        kept_labels=flat_labels[keep_mask],
        threshold=float(confidence_threshold),
    )


__all__ = ["PseudoLabelFilterError", "PseudoLabelFilterResult", "filter_pseudo_labels"]
