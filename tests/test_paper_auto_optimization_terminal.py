from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
    PaperAutoOptimizationStage,
    PaperPairedDelta,
)
from yolo_agent.certification.paper_auto_optimization_terminal import (
    render_paper_auto_optimization_report,
)


def test_terminal_renders_component_identity_and_recipe_specific_delta() -> None:
    report = PaperAutoOptimizationReport(
        acceptance_id="acceptance",
        status="failed",
        execute_real_gpu=True,
        model="yolo26n.pt",
        device="0",
        paper_ids=["paper-loss"],
        component_ids=["loss.quality.correlation"],
        component_families=["auxiliary_loss"],
        stages=[
            PaperAutoOptimizationStage(
                stage_id="certified_adapter",
                status="passed",
                metrics={
                    "tracks": {
                        "loss.quality.correlation": {
                            "component_family": "auxiliary_loss",
                            "adapter_hash": "adapter-hash",
                            "maturity": "gpu_certified",
                        }
                    }
                },
            )
        ],
        paired_deltas=[
            PaperPairedDelta(
                stage_id="pilot_3",
                track_id="auxiliary_loss",
                recipe_id="loss.quality.correlation",
                component_id="loss.quality.correlation",
                component_family="auxiliary_loss",
                primary_metric="map50_95",
                baseline_id="baseline",
                candidate_id="loss.quality.correlation",
                verified=True,
                protocol_match=True,
                metric_deltas={"map50_95": 0.02},
                target_error_fact_deltas={"localization/object": 2.0},
            )
        ],
        failures=["test stop"],
    )

    output = "\n".join(render_paper_auto_optimization_report(report))

    assert "component=loss.quality.correlation" in output
    assert "family=auxiliary_loss" in output
    assert "adapter=adapter-hash" in output
    assert "map50_95=+0.020000" in output
    assert "localization/object=+2.000000" in output
