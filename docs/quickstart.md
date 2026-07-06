# 快速开始

最快路径是启动一个自动优化 run。它会先跑安全的 debug；debug 成功后自动进入 pilot。debug 只验证最小训练链路，不代表最终模型效果。

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
yolo-agent setup coco --data E:\dataset\coco.yaml --model yolo26n.pt
```

setup 会生成 `.env.local`、`configs/local/llm_decision.local.yaml`、默认 run-id、COCO 路径检查报告和下一条 `optimize` 命令。

setup 内部会跑一次 `doctor`。它会根据当前可用显存、模型 scale、`imgsz=640` 和候选 `[32,48,64,96]` 预估一个保守 batch；这只是检查阶段的估算，不会替代训练前的 BatchTuner 实测。

如果输出里有 `note:` 或报告里有 doctor error，先按提示修复。没有可解析的 LLM API key 时，setup 会创建占位 `.env.local`；在 `.env.local`、环境变量或 `configs/local/llm_decision.local.yaml` 里设置好 key 后，默认 LLM proposal 才会参与策略生成。

## 2. 启动 COCO + YOLO26 自动优化

```powershell
yolo-agent optimize coco `
  --model yolo26n.pt `
  --data E:\dataset\coco.yaml `
  --goal +2map `
  --run-id coco-yolo26n `
  --profile debug `
  --execute
```

不加 `--execute` 时只做 dry-run，不会启动真实训练。
加了 `--execute` 后，默认流程是 `debug -> pilot`。如果你只想停在 debug，可以加 `--no-auto-advance`。

不理解 `dry-run`、`debug`、`pilot` 和 `full COCO` 的区别时，先看：[运行模式说明](training-modes.md)。

## 3. 查看状态

```powershell
yolo-agent loop status --run runs/coco-yolo26n
```

状态面板会显示当前 stage、queue counts、训练心跳、已有 evidence、blocked reason 和下一条建议命令。

输出顶部会先给人话摘要，例如当前是否正在训练、epoch/GPU/ETA、当前结论是否可信，以及下一步该等训练完成还是执行某条命令；后面保留机器可读字段。

## 4. full COCO 训练

full profile 会跑完整 COCO 预算，需要二次确认：

```powershell
yolo-agent optimize advance `
  --run runs/coco-yolo26n `
  --to-profile baseline_full `
  --execute `
  --confirm-full-run
```

## 推荐节奏

```text
setup -> optimize debug --execute -> auto pilot -> status -> baseline_full -> baseline_confirm -> candidate_full
```

不要一上来直接跑 full COCO。先把 debug 和 pilot 跑硬，才能让后续优化有可信证据。
