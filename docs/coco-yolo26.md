# COCO + YOLO26 Runbook

这个 runbook 面向目标：在标准 COCO 上，以 YOLO26 baseline 为起点，使用 evidence-driven loop 逐步寻找可验证的提升。

## 数据要求

建议目录：

```text
E:\datatset\
  coco.yaml
  coco\
    train2017.txt
    val2017.txt
    test2017.txt
    images\
      train2017\
      val2017\
      test2017\
    labels\
      train2017\
      val2017\
    annotations\
      instances_val2017.json
      instances_train2017.json  # optional for train-split COCO JSON analysis
```

先运行：

```powershell
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
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
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

默认会先跑 `debug`。如果 debug 成功，会自动推进到 `pilot`。如果只想停在当前 profile，可以加 `--no-auto-advance`。

## 查看状态

```powershell
yolo-agent status --run runs/coco-yolo26n
```

## full profile

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

## 优化纪律

- 固定 `imgsz=640`，不要通过增大输入尺寸制造不可比提升
- baseline evidence 不完整时，不允许进入 candidate full
- proposal 必须绑定 COCO error facts 和 Diagnosis Graph 原因诊断，例如 AP_small、per-class AP、false negative heavy classes
- proposal 排序使用 Utility Model，而不是盲目遍历组件或只看 priority hint
- candidate full 必须由 pilot promotion gate 放行
- 贡献结论必须来自单变量消融和 repeated seeds，否则只能写 possible contribution
- 每轮 parent/current COCO error delta 会写入 `runs/policy_memory.jsonl`，记录 action 对目标错误的实际收益、latency/model size 成本和置信度
- 单 seed 只能形成低置信 policy memory；即使 image-level paired bootstrap 显示稳定改善，也仍是 possible。至少 3 个 matched seeds 且跨 seed 收益置信区间下界大于 0，报告才把贡献升级为 confirmed
- Bandit / BO 只在 evaluator 已接受的候选里分配预算，不直接搜索组件空间
- 持久 ASHA 默认按 `pilot_3 -> pilot_10 -> candidate_full seed 1 -> seeds 2/3` 跨自动轮次收窄候选；3 epoch cohort 使用 `eta=3`，10 epoch 必须改善绑定的 error fact，full 仍需要 baseline acceptance、pilot promotion 和显式 full-run 确认
- Promotion 是 diagnosis-bound：目标 `AP_small` 时必须检查 AP_small、绑定类别 AP、对应 FN、overall mAP、latency、model size，并且目标改善要超过同协议 baseline 的测量噪声；任意非目标指标略涨不能晋级
- 每个 pilot fidelity 都带 matched baseline control；排序和晋级使用 paired delta。subset、seed、epoch、batch policy、Ultralytics 版本、`imgsz=640` 或 eval protocol 不匹配时不会比较
- matched control/candidate 都有 COCO predictions 时自动生成 paired image bootstrap：显示可能的样本波动、稳定改善类别和稳定退化类别；该诊断 AP@0.5 不覆盖官方 COCO AP50-95
- 每个 COCO error 会同时考虑 model/data/augmentation/postprocess/label/training 动作；例如 background false positives 会把 hard negative mining、background-only images、reduce mosaic、per-class threshold、missing-label check 和 focal gamma 放进同一 utility 排序
- 如果缺 AP_small、per-class AP/AR、confusion matrix、false-positive samples、label quality report 或 latency，系统会优先生成 evidence action，而不是继续训练
- `next_round.yaml` 会输出 `doctor_report`：主问题、可能原因、证据、拒绝动作、选中动作、预期改善方向和停止条件。固定 COCO baseline 时，`increase_imgsz` 会被明确记录为 rejected action。
