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

## 3 条命令跑起来

```powershell
yolo-agent setup coco --data E:\dataset\coco.yaml --model yolo26n.pt
yolo-agent optimize coco --model yolo26n.pt --data E:\dataset\coco.yaml --run-id coco-yolo26n --profile debug --execute
yolo-agent loop status --run runs/coco-yolo26n
```

`setup` 会生成本地 LLM 配置、`.env.local`、run-id 和 COCO 路径检查报告。需要单独体检环境时也可以运行：

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

## 运行模式一句话

```text
dry-run = 只预演，不训练
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
- 理解决策逻辑：[核心概念](docs/concepts.md)
- 看状态机和 evidence：[Loop Engineering](docs/loop-engineering.md)
- 查命令参数：[CLI 参考](docs/cli.md)
- 出问题了：[故障排查](docs/troubleshooting.md)

## 项目定位

YOLO Agent is a componentized object-detection optimization harness, not a free-form code-generation agent.
