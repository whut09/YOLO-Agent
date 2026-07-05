# CLI 参考

## 环境检查

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

## 一键优化

```powershell
yolo-agent optimize coco --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n --profile debug --execute
yolo-agent optimize custom --model yolo26n.pt --data data.yaml --run-id custom-yolo26n --profile debug --execute
yolo-agent optimize advance --run runs/coco-yolo26n --to-profile pilot --execute
```

full profile 需要：

```powershell
--confirm-full-run
```

## 状态和报告

```powershell
yolo-agent loop status --run runs/coco-yolo26n
yolo-agent report --run runs/coco-yolo26n --out report.md
```

## 显式 loop 阶段

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
```

