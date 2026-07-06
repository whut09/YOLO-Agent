# COCO + YOLO26 Runbook

这个 runbook 面向目标：在标准 COCO 上，以 YOLO26 baseline 为起点，使用 evidence-driven loop 逐步寻找可验证的提升。

## 数据要求

建议目录：

```text
E:\dataset\
  coco.yaml
  images\
    train2017\
    val2017\
    test2017\
  labels\
    train2017\
    val2017\
  annotations\
    instances_train2017.json
    instances_val2017.json
```

先运行：

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

## Budget Profiles

- `dry-run`: 不训练，只生成计划和队列
- `debug`: 小比例 COCO，`epochs=1`，验证链路
- `pilot`: 约 10% COCO，`epochs=10`，筛选候选
- `baseline_full`: 完整 COCO，单 seed，建立可信 baseline
- `baseline_confirm`: 完整 COCO，3 seeds，确认 baseline 稳定性
- `candidate_full`: 完整 COCO，3 seeds，只给通过 pilot promotion 的候选

详细区别见：[运行模式说明](training-modes.md)。

## 启动自动优化

```powershell
yolo-agent optimize coco `
  --model yolo26n.pt `
  --data E:\dataset\coco.yaml `
  --run-id coco-yolo26n `
  --profile debug `
  --execute
```

默认会先跑 `debug`。如果 debug 成功，会自动推进到 `pilot`。如果只想停在当前 profile，可以加 `--no-auto-advance`。

## 查看状态

```powershell
yolo-agent loop status --run runs/coco-yolo26n
```

## full profile

```powershell
yolo-agent optimize advance --run runs/coco-yolo26n --to-profile baseline_full --execute --confirm-full-run
```

## 优化纪律

- 固定 `imgsz=640`，不要通过增大输入尺寸制造不可比提升
- baseline evidence 不完整时，不允许进入 candidate full
- proposal 必须绑定 COCO error facts 和 Diagnosis Graph 原因诊断，例如 AP_small、per-class AP、false negative heavy classes
- proposal 排序使用 Utility Model，而不是盲目遍历组件或只看 priority hint
- candidate full 必须由 pilot promotion gate 放行
- 贡献结论必须来自单变量消融和 repeated seeds，否则只能写 possible contribution
- 每轮 parent/current COCO error delta 会写入 `runs/policy_memory.jsonl`，记录 action 对目标错误的实际收益、latency/model size 成本和置信度
- 单 seed 只能形成低置信 policy memory；至少 3 seeds 后，报告才应把贡献从 possible 升级为 confirmed
