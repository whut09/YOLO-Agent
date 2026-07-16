# yolo-agent

中文 | [English](README.en.md)

YOLO Agent 是一个证据驱动的 YOLO 自动优化训练 harness。

它不是自由形式的代码生成 Agent，也不会盲目改模型代码。它把目标检测优化固定成一个可恢复、可审计的闭环：

```text
任务 + 数据 + 错误事实 + 约束
        -> LLM / 规则生成策略 proposal
        -> evidence gate / compatibility / utility 过滤
        -> debug / pilot / full 实验
        -> evidence / report / next round
```

## 这是什么

- 给 COCO 或自定义 YOLO 数据集做自动化训练、诊断和下一轮优化建议。
- 用状态机、queue、EvidenceStore、doctor report 和 LLM proposal 管住实验过程。
- LLM 默认参与“建议和策略生成”，但不能绕过 evidence gate 直接启动训练或声称最佳模型。

## 一条命令开始训练

第一次使用必须先安装项目，否则系统里不会有 `yolo-agent` 命令。真实训练建议安装 train 依赖：

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

安装完成后，直接运行：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

这条命令会从 `debug` 开始，链路健康后自动进入 `pilot`，然后按 `budget=auto` 继续“分析 -> 生成候选 -> 再跑 pilot -> 对比 delta”的自动优化闭环：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

自动预算会在 GPU 小时、pilot 数、无改善耐心或搜索停止条件任一先到达时结束，并发默认为 1。内部最大轮数只用于防死循环，不代表一定执行这些实验。自动闭环只跑 debug/pilot 级别实验；到 full COCO 前会停住，并输出 `auto_optimization_summary.md` 和 `full_candidate_recommendations.yaml`。metadata-only 组件会被标记为需要 adapter，不会被伪装成真实训练。

查看状态或停止训练只需要：

```powershell
yolo-agent status --run runs/coco-yolo26n
yolo-agent stop --run runs/coco-yolo26n
```

首次使用先运行 `setup`。它会生成本地 LLM 配置、`.env.local`、run-id 和 COCO 路径检查报告，并完成环境与 batch 预估：

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

真正训练时，`batch=auto` 会由 BatchTuner 试跑验证后再自动替换成实测可用 batch。

## 智能优化是怎么做的

自动 loop 不会把所有论文组件排列组合。每一轮会按以下顺序工作：

baseline/pilot evidence -> COCO error facts -> 论文和组件查询 -> compatibility/maturity 过滤 -> LLM 医生式 Recipe Proposal -> RecipeCritic -> utility/budget/ablation gate -> pilot candidate -> evidence 和 reproduction status

论文中的指标只记录为 paper claim 或 paper prior，不能直接变成本地证据。metadata-only 组件只能进入 implementation request，必须有 adapter、单元测试和 smoke evidence 后才可能进入训练队列。Coupled Recipe 会先生成 baseline、单组件和组合消融，避免把多变量提升误归因给某一个组件。

详细记录在每轮的 paper_recipe_plan.yaml、component_compatibility.yaml、reproduction_state_*.yaml 和 decision_ledger.jsonl 中；终端只显示当前轮次、阶段、recipe、训练进度和最终结论。

高级研究流程可以预先冻结论文、组件 contract、YOLO26 compatibility review、recipes 和 reproduction queue，并让所有后续轮次引用同一个 `snapshot_hash`。具体命令下沉到 [Paper Intelligence 文档](docs/paper-intelligence.md)，不属于新人训练入口。

## 当前能力成熟度

代码存在、能够自动执行、已经本地复现是三件不同的事。下表由 `configs/capability_maturity.yaml` 自动生成；完整状态定义和源码依据见 [能力成熟度矩阵](docs/capability-maturity.md)。

<!-- capability-maturity:start -->
| 能力 | 当前状态 | 代码存在 | 自动执行 | 本地复现 | 现实边界 |
| --- | --- | --- | --- | --- | --- |
| Pilot 自动训练 | `executable` | 是 | 是 | 取决于本地 run | 默认训练入口可执行 debug/pilot；是否成功取决于本机环境和数据。 |
| 自动导入基础指标 | `executable` | 是 | 是 | 取决于本地 run | 可导入 results.csv、训练 artifacts 和基础 runtime evidence；缺失产物仍会形成 evidence gap。 |
| Candidate COCO error facts | `incomplete` | 是 | 部分 | 部分 | 已有 post-eval、导入和 completeness gate，但每个候选都稳定产出 predictions.json 与完整 per-class/FN/FP/localization facts 的闭环尚未完全保证。 |
| Error-delta 下一轮决策 | `partial` | 是 | 部分 | 部分 | 能比较 parent/current error facts 并约束 proposal；候选 error facts 不完整时会退回补证据或规则路径。 |
| ASHA / successive halving 队列控制 | `executable` | 是 | 有门禁 | 未声明 | ASHA assignment 已进入权威 RoundExecutionPlan 和队列；full rung 仍必须显式确认，不能理解为默认自动跑完整 COCO。 |
| 论文组件 Adapter | `mixed` | 是 | 混合 | 混合 | registry 同时包含 metadata-only、已实现 adapter 和可执行组件；必须逐组件查看 maturity，不能把论文条目等同于可训练实现。 |
| 3-seed confirmation | `supported, not automatic end-to-end` | 是 | 需显式确认 | 未声明 | 调度器和 confidence gate 支持 3 seeds；candidate_full 需要显式 full 确认，默认 pilot loop 不会自动完成全部 seeds。 |
| 稳定提升 +2 mAP | `not guaranteed` | 否 | 否 | 未声明 | +2 mAP 是优化目标和验收条件，不是项目保证；必须由 matched baseline、full COCO、3 seeds 和置信区间证明。 |
<!-- capability-maturity:end -->

## 运行模式一句话

```text
dry-run = 只预演，不训练；train 默认真训练，只有加 --dry-run 才预演
debug = 真训练一下，检查链路能不能跑通
pilot = 小规模训练，看方向有没有希望
full = 完整预算训练，用来形成可信结论
```

默认从 `debug` 开始；debug 成功后可以自动进入 `pilot`。进入 full COCO 前必须显式确认，避免误跑大任务。

## 下一步读哪个文档

- 第一次安装：[安装指南](docs/install.md)
- 跟着跑一遍：[快速开始](docs/quickstart.md)
- 不理解 dry-run/debug/pilot/full：[运行模式说明](docs/training-modes.md)
- 跑 COCO + YOLO26：[COCO + YOLO26 Runbook](docs/coco-yolo26.md)
- 跑自己的数据集：[自定义数据集](docs/custom-dataset.md)
- 配置 LLM proposal：[LLM 配置](docs/llm-setup.md)
- 理解决策逻辑：[核心概念](docs/concepts.md)
- 看状态机和 evidence：[Loop Engineering](docs/loop-engineering.md)
- 查命令参数：[CLI 参考](docs/cli.md)
- 出问题了：[故障排查](docs/troubleshooting.md)

## 项目定位

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.
