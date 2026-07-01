# yolo-agent

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.

The project is intended to help choose YOLO model sizes, network components, losses, training strategies, and reproducible experiment plans from task, dataset, and deployment constraints.

This repository currently provides the maintainable project skeleton only. Real training, benchmarking, adapter integration, and search logic will be added behind stable module boundaries.

## CLI

```bash
yolo-agent --help
yolo-agent init
yolo-agent init --scenario infrared_small_target --output task.yaml
yolo-agent profile-data
yolo-agent plan
yolo-agent check
yolo-agent smoke
yolo-agent search
yolo-agent ablate
yolo-agent benchmark
yolo-agent report
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Bundled scenario templates live under `configs/scenarios/` and validate against `yolo_agent.core.task_spec.TaskSpec`.
