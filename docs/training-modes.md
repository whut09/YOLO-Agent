# 训练生命周期与运行模式

> 本文所有命令示例均为单行 PowerShell 写法，可直接复制到 PowerShell 执行。PowerShell **不支持** Bash 的 `\` 续行符；如需 Bash 多行示例，见 [CLI 与高级命令](cli.md)，并注意它们单独标注为 `bash`。

本文是训练生命周期的**唯一权威描述**。[快速开始](quickstart.md)只保留最短路径，[CLI 与高级命令](cli.md)只保留参数清单；两者与本文冲突时，以本文和源码为准。

## 1. 唯一的生命周期流程

`yolo-agent train` 是唯一训练入口。从第一次安装到最终结论，只有这一条流程：

```text
setup（一次性环境准备）
  -> train --dry-run          # 可选：只预演，不训练
  -> train                    # 默认 debug profile，真实训练 1 epoch sanity
  -> debug 通过 -> 自动进入 pilot（10 epoch / 10% COCO）
  -> pilot 完成 -> budget=auto 的自动优化轮次（子 run：{run_id}-r1、{run_id}-r2 ...）
  -> ASHA 阶梯：pilot_3 -> pilot_10 -> candidate_full_seed_1 -> candidate_full_confirmation
  -> full 系 profile 始终停在显式确认边界
  -> baseline_full -> baseline_confirm ->（候选）candidate_full
```

这条流程有三个硬边界，全部来自代码，不存在"高级模式"绕过它们：

1. **所有 profile 都需要 GPU。** debug/pilot/full 没有任何一个提供 CPU 训练档位；CUDA 不可用时 run 不会开始（见第 7 节错误表）。
2. **full 系必须二次确认。** `baseline_full`、`baseline_confirm`、`candidate_full` 都在 `FULL_RUN_CONFIRMATION_PROFILES` 里（`yolo_agent/agents/optimize_runner.py`），没有 `--confirm-full-run` 就只会停在确认边界。
3. **`candidate_full` 不在自动推进链上。** 自动链最多走到 `baseline_confirm`；`candidate_full` 只能通过 ASHA 阶梯晋级或 prepared cohort 进入，Agent 永远不会替你决定跑候选 full。

## 2. 运行 profile 一览（数值来自真实配置）

下表数值来自 `configs/training/yolo26_coco_goal.yaml` 的 `training.budget_profiles` 块，并与 `yolo_agent/adapters/ultralytics/training.py` 中 `default_training_budget_profiles()` 的代码默认值逐项一致。**两处将来不一致时以代码为准**，不要把本文当默认值来源。

| profile | fraction | epochs | seeds | 验证方式 | 需要 GPU | 审批要求 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `debug` | 0.01 | 1 | [1] | 仅 quick_val（`val: false`） | 是 | 无 | 链路 sanity；禁止用于任何模型结论 |
| `pilot` | 0.1 | 10 | [1] | quick_val + 固定 COCO post-eval | 是 | 无 | 低成本筛选 baseline 与候选 |
| `baseline_full` | 1.0 | 100 | [1] | 完整 val + 固定 COCO post-eval | 是 | `--confirm-full-run` | 单 seed 正式 baseline |
| `baseline_confirm` | 1.0 | 100 | [1, 2, 3] | 完整 val + 固定 COCO post-eval | 是 | `--confirm-full-run` | 3 seed 确认 baseline 稳定性 |
| `candidate_full` | 1.0 | 100 | [1, 2, 3] | 完整 val + 固定 COCO post-eval | 是 | `--confirm-full-run` + `requires_pilot_pass` | 只允许通过 pilot 的候选进入 |

固定 COCO post-eval（配置 `coco_post_eval` 块）覆盖 `pilot`、`baseline_full`、`baseline_confirm`、`candidate_full` 四个 profile：`split=val`、`imgsz=640`、`conf=0.001`、`iou=0.7`。固定协议延迟测量（`inference_latency` 块）覆盖同样四个 profile。

自动预算边界来自同一配置文件的 `goal` 块：`max_gpu_hours=24`、`max_pilot_rounds=12`、`no_improvement_patience=4`、`max_concurrent_pilots=1`、`max_auto_rounds_safety=60`、`confidence_level=0.95`、`minimum_seeds=3`。任一边界先达到即停止；60 round cap 只是状态机防死循环保险，不等于要训练 60 个候选。

### ASHA 阶梯

自动优化轮次内部使用固定 `imgsz=640` 的 ASHA 阶梯（`yolo_agent/agents/asha_scheduler.py` 的 `default_asha_rungs()`）：

| rung | epochs | fraction | 晋级规则 | 真实淘汰原因字面量 |
| --- | --- | --- | --- | --- |
| `pilot_3` | 3 | 0.1 | `reduction_factor=3`：每 3 个完成的 trial 晋级 1 个；paired delta 低于 noise floor `-0.0015` 直接淘汰 | `pilot_3_delta_below_noise_floor` / `pilot_3_non_positive_paired_delta` / `pilot_3_diagnosis_promotion_gate_failed` |
| `pilot_10` | 10 | 0.1 | paired delta 必须为正，且目标 error fact 必须改善 | `pilot_10_non_positive_paired_delta` / `pilot_10_target_error_fact_not_improved` |
| `candidate_full_seed_1` | 100 | 1.0 | seed 1 通过诊断门 | `candidate_full_seed_1_non_positive_paired_delta` |
| `candidate_full_confirmation` | 100 | 1.0 | 确认 seeds `[42, 43, 44]`，paired seed 置信区间下界必须大于 0 | `candidate_full_confirmation_confidence_interval_not_positive` |

## 3. 一句话区别

```text
dry-run = 只预演，不训练
debug = 真训练一下，检查能不能跑通
pilot = 小规模训练，看方向有没有希望
full COCO = 正式完整训练，用来形成可信结论
```

### dry-run 是什么

`train` 默认启动真实训练；只有显式加 `--dry-run` 才是预演：

```powershell
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n --dry-run
```

dry-run 生成 `task.yaml`、`run_context.yaml`、`experiment_plan.yaml`、`execution_queue.yaml` 和初始 report，不占 GPU、不启动 Ultralytics。它会跳过真实训练所需的安全门（包括 paper-83 门与 release verify），所以 dry-run 通过**不代表**真实训练会被放行——那些门只在实际启动时检查。

### debug 是什么

不加 `--dry-run` 时，默认 `debug` profile 真实启动训练：

```powershell
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n
```

debug 的目标不是提高精度，而是确认链路跑通：数据路径能读、label 没炸、Ultralytics 能启动、GPU 能用、日志能采集、`results.csv` / `best.pt` / `args.yaml` 能进入 evidence store。debug 成功只代表"可以训练"，不代表"模型效果好"。

### pilot 是什么

pilot 是 10 epoch / 10% COCO 的低成本筛选，带固定协议 COCO post-eval，用来判断 baseline 或候选策略有没有希望。自动优化轮次每轮执行：

```text
pilot evidence -> LLM/规则分析 -> policy proposal -> guard 过滤 -> 可执行候选 pilot -> error delta
```

LLM 和规则只能生成 proposal；真正能不能跑由候选可执行性分类决定：`executable`（当前 Ultralytics executor 能真实执行）、`recommendation_only`（数据/标注/后处理/补证据建议，不会伪装成训练）、`adapter_required`（缺真实 adapter，不能假跑）。如果 pilot 都没有改善目标问题，就不应该升级到 full COCO。

### full COCO 是什么

full COCO 是正式训练预算（`baseline_full` / `baseline_confirm` / `candidate_full`），会消耗大量时间和 GPU，必须显式 `--confirm-full-run`：

```powershell
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

这个二次确认是故意设计的，避免误跑 100 epoch COCO。

## 4. 自动推进（auto-advance）与停止

自动推进链（`optimize_runner.py` 的 `_next_auto_profile`）：

- `debug -> pilot`：**无条件自动**（需要 execute、auto_advance 开启、run ok、queue 干净）。
- `pilot -> baseline_full -> baseline_confirm`：**只在命令带 `--confirm-full-run` 时**逐级推进。
- `candidate_full`：**不在链上**，只经 ASHA 阶梯或 prepared cohort。

控制开关：

| 参数 | 作用 |
| --- | --- |
| （默认） | debug 成功自动进入 pilot，之后在 `budget=auto` 边界内继续自动轮次，full 前停住 |
| `--no-auto-advance` | 停在 debug，不自动进入 pilot |
| `--auto-rounds 0` | pilot 完成后停止，不做自动优化轮次 |
| `--auto-rounds N` | 仅作为高级安全上限覆盖，普通用户不需要设置 |
| `--confirm-full-run` | 解锁 full 系 profile；没有它自动链最多走到 pilot 轮次 |

pilot 之后每轮自动 fork 子 run（如 `coco-yolo26n-r1`、`coco-yolo26n-r2`）。自动轮次不会启动 full COCO；full 候选会写进 `runs/<run_id>/artifacts/full_candidate_recommendations.yaml`，等确认预算后再手动 full run。

## 5. 恢复（resume）语义

**`train` 没有 `--resume` 参数。** 恢复方式是重跑同一条 train 命令：

```powershell
yolo-agent train --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n
```

系统会自动完成以下事情，不需要额外 flag：

- **profile 推断**：从 run 目录的 `run_context.yaml` 读取该 run 的 profile，命令行不需要（也不应该）另写。
- **active queue 检测**：存在未完成的执行队列时，继续执行而不是新建。
- **needs_resume 标记**：Ctrl+C 中断、外部 GPU 冲突或 `yolo-agent stop --run runs/coco-yolo26n` 都会把 run 标记为 `needs_resume`；外部冲突还会自动 requeue。
- **确认矩阵保留已完成 seed**：`baseline_confirm`/`candidate_full` 重跑时只补缺失的 seed，已完成的 seed 不会重复训练。

两个例外要注意：

- prepared cohort（由 loop 写入执行队列的 run）恢复时必须保持 `--profile pilot` 并加 `--no-auto-advance`，profile 不一致会被拦下。
- `yolo-agent loop --resume` 是 stage 级内部恢复命令，不属于新人工位；普通训练永远用重跑 `train` 的方式。

## 6. 训练发布（TrainingRelease）语义

`--training-release` 的默认值是 `artifacts/training_release_v1.yaml`。三种用法：

1. **自动使用（默认）**：不传参时使用默认路径。真实训练会在两个关口验证它：run-id 分配前（L0.7）和启动执行前（L0.8），发布物缺失、过期或与当前代码漂移都会被拦下。
2. **显式指定**：`yolo-agent train ... --training-release artifacts/training_release_v1.yaml`。
3. **漂移失败**：验证不过时进程 exit 2，终端打印 `Next: yolo-agent papers release, then re-run with --training-release artifacts/training_release_v1.yaml`——按提示先重新构建发布物，再用显式参数重跑。

dry-run 会跳过 release verify，所以 dry-run 通过不能作为发布物有效的证据。发布物的字段、构建与 fail-closed 规则见 [训练发布](training-release.md)。

## 7. 常见错误与下一步

每一行都说明**会停在哪里**和**下一步执行什么**：

| 现象 | 会停在哪里 | 下一步执行什么 |
| --- | --- | --- |
| 数据集路径不存在 | preflight 直接报错退出（exit 1），不创建 run 目录 | 修正 `--data` 指向真实 coco.yaml，重跑同一条 `train` 命令 |
| 模型文件缺失 | preflight 不拦截，在 Ultralytics 加载阶段失败 | 给 `--model` 一个真实存在的本地 checkpoint 路径，重跑 |
| CUDA 不可用 | preflight 报错，或调度器把 trial 标记 `blocked_by_resource: gpu_unavailable` | 安装 CUDA 版 PyTorch；确认 `nvidia-smi` 可见 GPU 后重跑同一条命令 |
| 其他进程占用 GPU | run 标记 `needs_resume` 并自动 requeue | 等待自动恢复；或释放 GPU 后重跑同一条 `train` 命令 |
| batch 过大触发 OOM（BatchTuner 试跑阶段） | BatchTuner 自动按最大优先顺序回退候选 batch（32–96，显存允许时自动扩展到 256） | 无需操作，这是设计内行为 |
| 训练中途 OOM | 自动恢复：干净 GPU 时重试一次，之后 batch 减半最多两次 | 反复失败时先跑 `yolo-agent setup coco` 检查显存与 batch 估算 |
| TrainingRelease 过期或漂移 | exit 2，打印 `Next: yolo-agent papers release, then re-run with --training-release artifacts/training_release_v1.yaml` | 按提示先 `yolo-agent papers release`，再带 `--training-release` 重跑 |
| paper-83 门未通过 | CLI 阶段 exit 2；运行时安全缝 exit 86（"86 = refused at a safety seam"） | 按 `Next:` 提示补齐论文侧产物；exit 86 表示被安全缝拒绝，不代表训练本身失败 |
| runtime adapter 执行失败 | 写出 `adapter_runtime_failure.json`，exit 86，该候选不进入比较 | 打开失败 artifact 查看原因，修复后重跑同一条命令 |
| pilot 证据不完整 | 下一轮 `next_round.yaml` 中 `proposal_mode: evidence_only`，并列出 `pilot_evidence_actions` | 执行列出的动作：`run_coco_post_eval` / `import_coco_eval` / `mine_coco_errors` / `repair_coco_evidence_artifacts` / `import_current_node_error_facts` |

## 8. 新人应该怎么选

- 第一次使用：先跑 `--dry-run`，看 Agent 准备做什么。
- 环境刚装好：跑默认 `debug`。
- debug 成功：让它自动进入 pilot 和自动轮次。
- pilot 结果正常且准备投入算力：再手动确认 full COCO。
- 想比较候选是否真的有效：必须走 full profile + evidence + repeated seeds（3 seed 置信区间），pilot 结果永远只是初步证据。

为什么不直接 full COCO：full COCO 成本高，如果数据路径、标签、batch、cache、GPU 或日志采集有问题，直接 full run 会浪费大量时间；先让 debug 和 pilot 把链路跑硬，后续优化才有可信证据基础。

## 相关文档

- [快速开始](quickstart.md) — 最短路径与 goal 写法。
- [CLI 与高级命令](cli.md) — `train` 全部参数与恢复语义的参数清单。
- [训练发布](training-release.md) — TrainingRelease 的字段、构建与 fail-closed 验证。
- [训练就绪流水线](training-readiness.md) — 第一次训练前的六道关卡。
- [证据架构](evidence.md) — pilot 证据与 full 结论的声明边界。
- [GPU 认证](gpu-certification.md) — 组件认证 / mini-GPU 验收 / 真实训练 / 完整复现四层区别。
- [系统架构](architecture.md) — 训练流程在整体架构中的位置。
