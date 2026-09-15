"""Deterministic graph fingerprints and diffs for model-graph plugins.

A paper graph mechanism is only real if it changes the executed module graph.
This module gives every :class:`~yolo_agent.components.model_graph.ModelGraphPlugin`
a deterministic fingerprint (module tree, parameter shapes, and shape-derived
multiply-accumulate estimate) plus an explicit diff between the base graph and
the mechanism-enabled graph, so an audit can prove the forward path changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

BASE_GRAPH_PLUGIN = "direct_detect_passthrough"


class _PassthroughModule:
    """Base-graph stand-in: the Detect node receives the backbone features directly."""

    def forward(self, features: list[Any]) -> list[Any]:
        return list(features)

    def named_children(self):
        return []

    def named_parameters(self, *_, **__):
        return []


def safe_paper_id(paper_id: str) -> str:
    """Filesystem-safe artifact name for a paper ID such as ``arxiv:2212.07784``."""

    return "".join(
        character if character.isalnum() else "_"
        for character in paper_id.replace(":", "_")
    ).strip("_")


@dataclass(frozen=True)
class GraphFingerprint:
    """Deterministic identity of one graph composition."""

    plugin_name: str
    module_tree: str
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_count: int
    estimated_macs: int
    graph_hash: str

    @property
    def module_names(self) -> tuple[str, ...]:
        """Named leaf modules, used as the inserted-module record."""

        return tuple(
            line.split(": ", 1)[1]
            for line in self.module_tree.splitlines()
            if line.startswith("  ") and ": " in line
        )


def _module_tree(module: Any) -> str:
    lines: list[str] = []

    def walk(current: Any, prefix: str) -> None:
        name = type(current).__name__
        lines.append(f"{prefix}{name}")
        children = current.named_children() if hasattr(current, "named_children") else []
        for child_name, child in children:
            walk(child, f"{prefix}  {child_name}: ")

    walk(module, "")
    return "\n".join(lines)


def _parameter_shapes(module: Any) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not hasattr(module, "named_parameters"):
        return ()
    return tuple(
        (name, tuple(int(value) for value in parameter.shape))
        for name, parameter in module.named_parameters()
    )


def estimate_macs(module: Any, features: list[Any]) -> int:
    """Estimate multiply-accumulates for one forward over ``features``.

    Convolutional and linear MACs are accumulated with forward hooks, so the
    number is derived from the real executed shapes rather than a guessed
    formula.  Fallback tools such as ``thop`` are deliberately not required.
    """

    import torch.nn as nn

    totals = {"macs": 0}
    handles: list[Any] = []

    def register(current: Any) -> None:

        if isinstance(current, nn.Conv2d):
            def conv_forward_hook(
                layer: nn.Conv2d, inputs: tuple[Any, ...], output: Any
            ) -> None:
                input_shape = tuple(int(value) for value in inputs[0].shape)
                output_shape = tuple(int(value) for value in output.shape)
                kh, kw = layer.kernel_size
                cin = input_shape[1] // layer.groups
                spatial = output_shape[-2] * output_shape[-1]
                totals["macs"] += int(spatial * cin * layer.out_channels * kh * kw)

            handles.append(current.register_forward_hook(conv_forward_hook))
        elif isinstance(current, nn.Linear):
            def linear_forward_hook(
                layer: nn.Linear, inputs: tuple[Any, ...], output: Any
            ) -> None:
                features_in = int(inputs[0].shape[-1])
                totals["macs"] += int(output.numel() * features_in)

            handles.append(current.register_forward_hook(linear_forward_hook))

        children = current.children() if hasattr(current, "children") else []
        for child in children:
            register(child)

    register(module)
    try:
        module.forward(list(features))
    finally:
        for handle in handles:
            handle.remove()
    return int(totals["macs"])


def graph_fingerprint(
    module: Any,
    *,
    features: list[Any],
    plugin_name: str,
) -> GraphFingerprint:
    """Fingerprint one composition, including executed-shape MAC estimation."""

    import math

    tree = _module_tree(module)
    parameter_shapes = _parameter_shapes(module)
    parameter_count = sum(
        int(math.prod(shape)) if shape else 0 for _, shape in parameter_shapes
    )
    estimated_macs = estimate_macs(module, features)
    payload = json.dumps(
        {
            "plugin": plugin_name,
            "tree": tree,
            "parameters": [[name, list(shape)] for name, shape in parameter_shapes],
            "macs": estimated_macs,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return GraphFingerprint(
        plugin_name=plugin_name,
        module_tree=tree,
        parameter_shapes=parameter_shapes,
        parameter_count=parameter_count,
        estimated_macs=estimated_macs,
        graph_hash=digest,
    )


@dataclass(frozen=True)
class GraphDiff:
    """Explicit base-versus-mechanism graph difference for one paper."""

    paper_id: str
    mechanism_id: str
    component_id: str
    base_graph_hash: str
    modified_graph_hash: str
    graph_hash_changed: bool
    inserted_modules: tuple[str, ...]
    removed_replaced_modules: tuple[str, ...]
    changed_edges: tuple[str, ...]
    feature_levels: dict[str, Any]
    params_base: int
    params_modified: int
    macs_base: int
    macs_modified: int

    def to_payload(self, *, mapping: dict[str, Any], adaptation: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "paper_graph_diff.v1",
            "paper_id": self.paper_id,
            "mechanism_id": self.mechanism_id,
            "component_id": self.component_id,
            "base_graph_hash": self.base_graph_hash,
            "modified_graph_hash": self.modified_graph_hash,
            "graph_hash_changed": self.graph_hash_changed,
            "inserted_modules": list(self.inserted_modules),
            "removed_replaced_modules": list(self.removed_replaced_modules),
            "changed_edges": list(self.changed_edges),
            "feature_levels": dict(self.feature_levels),
            "params": {"base": self.params_base, "modified": self.params_modified},
            "estimated_macs": {"base": self.macs_base, "modified": self.macs_modified},
            "paper_mechanism_mapping": mapping,
            "adaptation_record": adaptation,
        }


def compute_graph_diff(
    *,
    paper_id: str,
    mechanism_id: str,
    component_id: str,
    base: GraphFingerprint,
    modified: GraphFingerprint,
    feature_levels: dict[str, Any],
    inserted_modules: tuple[str, ...] | None = None,
    removed_replaced_modules: tuple[str, ...] = (),
) -> GraphDiff:
    """Diff the base and mechanism-enabled compositions explicitly."""

    if base.graph_hash == modified.graph_hash:
        raise ValueError(
            "mechanism-enabled graph hash equals the base graph hash; "
            "the mechanism does not enter the forward path"
        )
    base_edges = tuple(base.module_tree.splitlines())
    modified_edges = tuple(modified.module_tree.splitlines())
    changed_edges = tuple(
        edge for edge in modified_edges if edge not in base_edges
    ) or ("graph composition changed",)
    names = inserted_modules or modified.module_names
    return GraphDiff(
        paper_id=paper_id,
        mechanism_id=mechanism_id,
        component_id=component_id,
        base_graph_hash=base.graph_hash,
        modified_graph_hash=modified.graph_hash,
        graph_hash_changed=True,
        inserted_modules=tuple(names),
        removed_replaced_modules=removed_replaced_modules,
        changed_edges=changed_edges,
        feature_levels=dict(feature_levels),
        params_base=base.parameter_count,
        params_modified=modified.parameter_count,
        macs_base=base.estimated_macs,
        macs_modified=modified.estimated_macs,
    )


def config_fingerprint(payload: Any) -> str:
    """Stable fingerprint that separates two papers sharing one plugin class."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "BASE_GRAPH_PLUGIN",
    "GraphDiff",
    "GraphFingerprint",
    "compute_graph_diff",
    "config_fingerprint",
    "estimate_macs",
    "graph_fingerprint",
    "safe_paper_id",
]
