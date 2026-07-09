# 运行模式说明

YOLO Agent 里最容易混淆的是 `dry-run`、`debug`、`pilot` 和 `full COCO`。它们不是四种模型，而是四种训练预算和可信等级。

## 一句话区别

```text
dry-run = 只预演，不训练
debug = 真训练一下，检查能不能跑通
pilot = 小规模训练，看方向有没有希望
full COCO = 正式完整训练，用来形成可信结论
```

## 对比表

| 模式 | 会不会训练 | 默认规模 | 主要目的 | 结果能不能当最终结论 |
| --- | --- | --- | --- | --- |
| `dry-run` | 不会 | 0 epoch | 生成计划，确认 Agent 准备跑什么 | 不能 |
| `debug` | 会 | COCO `fraction=0.01`，`epochs=1`，通常不做完整验证 | 检查数据、GPU、Ultralytics、日志、产物导入是否跑通 | 不能 |
| `pilot` | 会 | COCO `fraction=0.1`，`epochs=10` | 用较低成本判断 baseline 或候选策略有没有希望 | 只能作为初步证据 |
| `baseline_full` | 会 | 完整 COCO，`epochs=100`，单 seed | 建立正式 baseline | 可以作为 baseline 证据 |
| `baseline_confirm` | 会 | 完整 COCO，`epochs=100`，3 seeds | 确认 baseline 稳定性 | 可以作为更可信 baseline |
| `candidate_full` | 会 | 完整 COCO，`epochs=100`，3 seeds | 验证候选策略是否真的提升 | 可以，但必须有消融和 repeated seeds |

## dry-run 是什么

`train` 默认会启动真实训练；只有显式加 `--dry-run` 时才是 dry-run：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --dry-run
```

dry-run 会生成：

- `task.yaml`
- `run_context.yaml`
- `experiment_plan.yaml`
- `execution_queue.yaml`
- 初始 report

dry-run 不会占 GPU，也不会启动 Ultralytics 训练。它适合第一次运行前检查“Agent 到底准备做什么”。

## debug 是什么

不加 `--dry-run` 时，`debug` 会真实启动训练：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

debug 的目标不是提高精度，而是确认链路跑通：

- 数据路径能读
- label 格式没炸
- Ultralytics 能启动
- GPU 能用
- 日志能采集
- `results.csv` / `best.pt` / `args.yaml` 等产物能进入 evidence store

debug 成功只代表“可以训练”，不代表“模型效果好”。

## pilot 是什么

`pilot` 是小规模试跑。它比 debug 更接近真实训练，但仍然不是最终结论。

当前一键优化默认行为是：

```text
debug 成功 -> 自动进入 pilot -> 到 full COCO 前停住
```

如果命令里加了 `--auto-rounds N`，pilot 完成后会继续做 N 轮 pilot-only 自动优化：

```text
baseline/pilot evidence
  -> LLM / 规则诊断
  -> policy proposal
  -> evidence gate / compatibility / utility guard
  -> 只执行当前 adapter 真支持的 pilot 候选
  -> 导入 evidence
  -> error delta 分析
  -> 下一轮
```

这不是简单遍历组件参数。LLM 和规则只能生成 proposal；真正能不能跑，由 guard 和候选可执行性分类决定：

- `executable`：当前 Ultralytics executor 能真实执行的 pilot 训练。
- `recommendation_only`：数据、标注、后处理或补证据建议，不会伪装成训练。
- `adapter_required`：组件 metadata 已有，但还缺真实 adapter，例如自定义 loss/head/assigner，不能假跑。

pilot 的作用：

- 判断当前 baseline 是否正常
- 判断某个候选策略有没有希望
- 发现明显的数据、速度、显存或训练异常
- 给下一轮 error diagnosis 提供初步 evidence

如果 pilot 都没有改善目标问题，就不应该升级到 full COCO。

## full COCO 是什么

full COCO 是正式训练预算，会消耗大量时间和 GPU。它包括：

- `baseline_full`
- `baseline_confirm`
- `candidate_full`

这些 profile 必须显式加 `--confirm-full-run`：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

这个二次确认是故意设计的，避免用户误跑 100 epoch COCO。

## 为什么不直接 full COCO

因为 full COCO 成本高，而且如果数据路径、标签、batch、cache、GPU 或日志采集有问题，直接 full run 会浪费很多时间。

推荐顺序：

```text
dry-run -> debug -> pilot -> baseline_full -> baseline_confirm -> candidate_full
```

## 如何停止自动进入 pilot

如果你只想跑 debug，不想自动进入 pilot：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --no-auto-advance
```

## 新人应该怎么选

- 第一次使用：先跑 dry-run
- 环境刚装好：跑 debug
- debug 成功：让它自动进入 pilot
- pilot 结果正常且你准备投入算力：再手动确认 full COCO
- 想比较候选是否真的有效：必须用 full profile + evidence + repeated seeds
