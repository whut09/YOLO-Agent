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

这条命令会从 `debug` 开始，链路健康后自动进入 `pilot`，然后默认继续做 30 轮“分析 -> 生成候选 -> 再跑 pilot -> 对比 delta”的自动优化闭环：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n
```

自动闭环只跑 debug/pilot 级别实验；到 full COCO 前会停住，并输出 `auto_optimization_summary.md` 和 `full_candidate_recommendations.yaml`。metadata-only 组件会被标记为需要 adapter，不会被伪装成真实训练。只想跑到 pilot 就停住时，加 `--auto-rounds 0`。

查看状态或停止训练只需要：

```powershell
yolo-agent status --run runs/coco-yolo26n
yolo-agent stop --run runs/coco-yolo26n
```

可选：`setup` 会生成本地 LLM 配置、`.env.local`、run-id 和 COCO 路径检查报告。需要单独体检环境时也可以运行：

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
yolo-agent doctor --data E:\datatset\coco.yaml --model yolo26n.pt
```

`doctor` 会预估一个保守 batch 上限；真正训练时，`batch=auto` 会由 BatchTuner 试跑验证后再自动替换成实测可用 batch。

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
