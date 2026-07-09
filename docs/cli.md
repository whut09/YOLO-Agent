# CLI 参考

## 初次 setup

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

生成 `.env.local`、`configs/local/llm_decision.local.yaml`、默认 run-id、COCO 路径检查报告和推荐启动命令。

## 环境检查

```powershell
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
yolo-agent doctor --llm
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt --llm
```

LLM 配置说明见：[llm-setup.md](llm-setup.md)。没有可解析的 API key 时不会失败，会回退到规则策略。

## 一键优化

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
yolo-agent train --kind custom --model yolo26n.pt --data data.yaml --run-id custom-yolo26n
```

默认 `train` 会在 debug 成功后自动进入 pilot。只想预演时加 `--dry-run`；需要停在当前 profile 时，加 `--no-auto-advance`。

运行模式说明见：[training-modes.md](training-modes.md)。

full profile 需要：

```powershell
--confirm-full-run
```

## 状态和停止

```powershell
yolo-agent status --run runs/coco-yolo26n
yolo-agent stop --run runs/coco-yolo26n
```

## 高级兼容命令

普通用户只需要 `train/status/stop/doctor/setup`。下面这些命令是内部 harness 零件，默认帮助页已隐藏；只有调试或开发时才需要。

```powershell
yolo-agent loop init --run-id exp001 --task task.yaml --data data.yaml
yolo-agent loop diagnose --run runs/exp001 --errors errors.yaml
yolo-agent loop plan --run runs/exp001
yolo-agent loop enqueue --run runs/exp001
yolo-agent loop execute --run runs/exp001 --executor dry-run
yolo-agent loop next --run runs/exp001
```

## 数据工具

```powershell
yolo-agent profile-data --data data.yaml --out runs/dataset_report
yolo-agent advise-labels --data data.yaml --predictions predictions.yaml --out runs/annotation_advice
yolo-agent smoke --plan runs/plan.yaml --data data.yaml
yolo-agent ablate-plan --plan runs/plan.yaml --out runs/ablation_plan.yaml
yolo-agent report --run runs/coco-yolo26n --out report.md
```
