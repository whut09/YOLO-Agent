"""Paper model-graph audit tests.

Covers the Prompt-7 graph contract: a paper mechanism must be a real module on
the forward path (graph hash changes), forward/backward must run on synthetic
CPU tensors, contracts must validate channels/strides/levels, each paper must
emit a graph-diff artifact with hashes/modules/edges/levels/mapping, and two
papers sharing one plugin class must still be distinguishable by composition
fingerprint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.model_graph import FeaturePyramidContract  # noqa: E402
from yolo_agent.components.model_graph_diff import (  # noqa: E402
    _PassthroughModule,
    config_fingerprint,
    graph_fingerprint,
    safe_paper_id,
)
from yolo_agent.research.paper_graph_side import (  # noqa: E402
    GRAPH_PLUGIN_CHANNELS,
    PaperGraphSideAuditBuilder,
    _resolve_neck_plugin,
    resolve_graph_route,
)

IN_SCOPE = {
    "arxiv:2108.07755": "detection_head.task_aligned",
    "arxiv:2212.07784": "neck.rtmdet_large_kernel",
    "arxiv:2309.11331": "feature_pyramid.multi_scale",
}


def test_audit_builds_and_reports_per_paper_status() -> None:
    audit = PaperGraphSideAuditBuilder().build(write_artifacts=False)
    assert audit.paper_count == 83
    assert audit.graph_side_paper_count == 3
    assert audit.summary["ready"] == 3
    assert audit.summary["blocked_missing_evidence"] == 0
    assert audit.summary["out_of_scope"] == 80
    assert audit.audit_hash

    by_id = {item.paper_id: item for item in audit.records}
    assert set(by_id) >= set(IN_SCOPE)
    for paper_id in IN_SCOPE:
        record = by_id[paper_id]
        assert record.status == "ready"
        assert not record.blockers
        assert record.behavior.passed
        assert record.graph_diff["graph_hash_changed"] is True
    # TOOD is domain head; RTMDet neck; Gold-YOLO feature_fusion.
    assert by_id["arxiv:2108.07755"].primary_domain == "head"
    assert by_id["arxiv:2212.07784"].primary_domain == "neck"
    assert by_id["arxiv:2309.11331"].primary_domain == "feature_fusion"


def test_no_backbone_domain_paper_exists_in_frozen_83() -> None:
    audit = PaperGraphSideAuditBuilder().build(write_artifacts=False)
    assert not any(
        record.primary_domain == "backbone" and record.in_scope
        for record in audit.records
    )


def test_every_in_scope_record_has_real_behavior_evidence() -> None:
    audit = PaperGraphSideAuditBuilder().build(write_artifacts=False)
    for record in audit.records:
        if not record.in_scope:
            continue
        checks = record.behavior.checks
        assert checks["finite_output"] is True
        gradient_keys = [
            key for key in checks if key.startswith("gradient_flow_")
        ]
        assert gradient_keys, record.paper_id
        assert any(checks[key] is True for key in gradient_keys), record.paper_id
        assert checks["graph_hash_changed"] is True
        assert record.graph_diff["params"]["modified"] > record.graph_diff["params"]["base"]


def test_graph_hash_changes_when_mechanism_enabled() -> None:
    plugin = _resolve_neck_plugin("neck.rtmdet_large_kernel", GRAPH_PLUGIN_CHANNELS)
    features = [
        torch.zeros(1, value, 64 // stride, 64 // stride)
        for value, stride in zip(
            GRAPH_PLUGIN_CHANNELS, (8, 16, 32), strict=True
        )
    ]
    base = graph_fingerprint(
        _PassthroughModule(), features=features, plugin_name="base"
    )
    modified = graph_fingerprint(
        plugin, features=features, plugin_name=plugin.plugin_id
    )
    assert base.graph_hash != modified.graph_hash
    assert modified.parameter_count > base.parameter_count
    assert modified.estimated_macs > 0


def test_identity_placeholder_module_is_rejected_by_diff_guard() -> None:
    from yolo_agent.components.model_graph_diff import compute_graph_diff

    features = [torch.zeros(1, 8, 8, 8)]
    passthrough = graph_fingerprint(
        _PassthroughModule(), features=features, plugin_name="base"
    )
    with pytest.raises(ValueError, match="does not enter the forward path"):
        compute_graph_diff(
            paper_id="x",
            mechanism_id="m",
            component_id="c",
            base=passthrough,
            modified=passthrough,
            feature_levels={"strides": [8]},
        )


def test_feature_contract_rejects_wrong_channels_and_strides() -> None:
    contract = FeaturePyramidContract(strides=[8, 16, 32], channels=[64, 128, 256])
    features = [
        torch.zeros(1, 64, 8, 8),
        torch.zeros(1, 128, 4, 4),
        torch.zeros(1, 256, 2, 2),
    ]
    contract.validate_features(features, 64)
    with pytest.raises(ValueError, match="channels"):
        contract.validate_features(
            [torch.zeros(1, 32, 8, 8), *features[1:]], 64
        )
    with pytest.raises(ValueError, match="stride"):
        contract.validate_features(
            [torch.zeros(1, 64, 7, 7), *features[1:]], 64
        )


def test_neck_plugins_preserve_feature_levels_and_support_backward() -> None:
    for mechanism_id in ("neck.rtmdet_large_kernel", "feature_pyramid.multi_scale"):
        plugin = _resolve_neck_plugin(mechanism_id, GRAPH_PLUGIN_CHANNELS)
        features = [
            torch.zeros(1, value, 64 // stride, 64 // stride, requires_grad=True)
            for value, stride in zip(
                GRAPH_PLUGIN_CHANNELS, (8, 16, 32), strict=True
            )
        ]
        outputs = plugin.forward(features)
        assert [item.shape for item in outputs] == [item.shape for item in features]
        loss = sum(item.float().sum() for item in outputs)
        loss.backward()
        assert all(item.grad is not None for item in features)


def test_task_aligned_head_wrapper_preserves_native_training_dict() -> None:
    from yolo_agent.research.paper_graph_side import _resolve_head_wrapper

    wrapper = _resolve_head_wrapper("detection_head.task_aligned")
    assert wrapper is not None
    features = [torch.zeros(1, 64, 8, 8, requires_grad=True)]
    output = wrapper.forward(features)
    assert set(output) == {"one2many", "one2one"}
    scale_before = wrapper.quality_scale.detach().item()
    output["one2one"]["scores"].float().sum().backward()
    assert wrapper.quality_scale.grad is not None
    assert wrapper.quality_scale.detach().item() == pytest.approx(scale_before)


def test_graph_diff_artifacts_are_written_per_paper(tmp_path) -> None:
    builder = PaperGraphSideAuditBuilder(artifacts_dir=tmp_path / "diffs")
    audit = builder.build()
    paths = sorted((tmp_path / "diffs").glob("*.yaml"))
    assert len(paths) == 3
    for record in audit.records:
        if not record.in_scope:
            continue
        expected = tmp_path / "diffs" / f"{safe_paper_id(record.paper_id)}.yaml"
        assert expected.is_file(), expected
        text = expected.read_text(encoding="utf-8")
        for key in (
            "base_graph_hash",
            "modified_graph_hash",
            "graph_hash_changed",
            "inserted_modules",
            "removed_replaced_modules",
            "changed_edges",
            "feature_levels",
            "paper_mechanism_mapping",
            "adaptation_record",
            "composition_fingerprint",
        ):
            assert key in text, f"{expected}: missing {key}"
        diff = record.graph_diff
        assert diff["base_graph_hash"] != diff["modified_graph_hash"]
        assert diff["feature_levels"]["strides"] == [8, 16, 32]


def test_shared_plugin_class_compositions_are_distinguishable() -> None:
    base = {
        "component_id": "neck.rtmdet_large_kernel",
        "graph_hash": "same-hash-if-same-class",
    }
    left = config_fingerprint(
        {
            **base,
            "paper_id": "arxiv:2212.07784",
            "config": {"kernel_size": 5},
        }
    )
    right = config_fingerprint(
        {
            **base,
            "paper_id": "arxiv:9999.99999",
            "config": {"kernel_size": 7},
        }
    )
    assert left != right


def test_route_resolution_rejects_unknown_mechanisms() -> None:
    assert resolve_graph_route("loss.bbox.wiou") is None
    assert resolve_graph_route("totally_unknown_mechanism") is None
    assert resolve_graph_route("detection_head.identity_placeholder") is None


def test_status_renderer_lists_all_papers_and_method_notes() -> None:
    from yolo_agent.research.paper_graph_side import (
        render_paper_83_graph_side_status,
    )

    audit = PaperGraphSideAuditBuilder().build(write_artifacts=False)
    report = render_paper_83_graph_side_status(audit)
    assert "# Paper-83 Model-graph Side Status" in report
    assert "arxiv:2212.07784" in report
    assert "out-of-scope" in report
    assert "does not train a model" in report
    assert "no `backbone`-domain paper" in report
