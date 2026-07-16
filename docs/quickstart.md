# 快速开始

最快路径是启动一个自动优化 run。它会先跑安全的 debug；debug 成功后自动进入 pilot；pilot 完成后默认继续做有边界的 pilot-only 自动优化轮次。debug 只验证最小训练链路，不代表最终模型效果。

## 0. 先安装一次

还没安装时，`yolo-agent` 命令不会存在。真实训练建议安装 train 依赖：

```powershell
cd E:\codex\YOLO-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

只做开发、文档和 dry-run 时可以安装 `".[dev]"`，但要启动 Ultralytics 训练请用 `".[train]"`。

## 1. 运行 setup 向导

```powershell
yolo-agent setup coco --data E:\datatset\coco.yaml --model yolo26n.pt
```

setup 会生成 `.env.local`、`configs/local/llm_decision.local.yaml`、默认 run-id、COCO 路径检查报告和下一条 `optimize` 命令。

setup 内部会跑一次 `doctor`。它会根据当前可用显存、模型 scale、`imgsz=640` 和 batch 候选预估一个保守 batch；训练前的 BatchTuner 会按显存自动扩展并优先试大 batch，例如 24GB 显存会从 `256` 开始，再回退到 `224,192,160,128,96,64,48,32`。这只是检查阶段的估算，不会替代训练前的 BatchTuner 实测。

如果输出里有 `note:` 或报告里有 doctor error，先按提示修复。没有可解析的 LLM API key 时，setup 会创建占位 `.env.local`；在 `.env.local`、环境变量或 `configs/local/llm_decision.local.yaml` 里设置好 key 后，默认 LLM proposal 才会参与策略生成。

本文示例使用当前本机已准备好的 `E:\datatset\coco.yaml`。如果你的 COCO 放在其他目录，把命令里的 `--data` 改成自己的真实路径。

## 2. 启动 COCO + YOLO26 自动优化

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --goal +2map --run-id coco-yolo26n
```

`train` 默认真训练，默认流程是 `debug -> pilot -> 自动分析 -> budget=auto pilot 候选搜索`。如果你只想预演、不启动训练，加 `--dry-run`。

不理解 `dry-run`、`debug`、`pilot` 和 `full COCO` 的区别时，先看：[运行模式说明](training-modes.md)。

默认预算不是固定轮数。启动前会显示预计范围和以下边界，任一先达到就停止：

- 最大 24 GPU 小时
- 最多 12 个实际 pilot
- 连续 4 个 pilot 无改善
- 最大并发 1
- 60 个状态机轮次作为最后的防死循环保险
- full COCO 必须显式 `--confirm-full-run`

`--auto-rounds` 仅保留为高级安全上限覆盖；普通用户不需要设置。

这会在 pilot 后自动 fork 子 run，例如 `coco-yolo26n-r1`、`coco-yolo26n-r2`，每轮执行：

```text
pilot evidence -> LLM/规则分析 -> policy proposal -> guard 过滤 -> 可执行候选 pilot -> error delta
```

自动轮次不会启动 full COCO。它会把 full 候选写到 `runs/coco-yolo26n/artifacts/full_candidate_recommendations.yaml`，等你确认预算后再手动 full run。

## 3. 查看状态

```powershell
yolo-agent status --run runs/coco-yolo26n
```

状态面板会显示当前 stage、queue counts、训练心跳、已有 evidence、blocked reason 和下一条建议命令。

输出顶部会先给人话摘要，例如当前是否正在训练、epoch/GPU/ETA、当前结论是否可信，以及下一步该等训练完成还是执行某条命令；后面保留机器可读字段。

## 4. full COCO 训练

full profile 会跑完整 COCO 预算，需要二次确认：

```powershell
yolo-agent train --model yolo26n.pt --data E:\datatset\coco.yaml --run-id coco-yolo26n --profile baseline_full --confirm-full-run
```

## 推荐节奏

```text
setup -> train debug -> auto pilot -> auto pilot rounds -> status -> baseline_full -> baseline_confirm -> candidate_full
```

不要一上来直接跑 full COCO。先把 debug 和 pilot 跑硬，才能让后续优化有可信证据。
