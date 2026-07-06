# 快速开始

最快路径是启动一个自动优化 run。它会先跑安全的 debug；debug 成功后自动进入 pilot。debug 只验证最小训练链路，不代表最终模型效果。

## 1. 运行 setup 向导

```powershell
yolo-agent setup coco --data E:\dataset\coco.yaml --model yolo26n.pt
```

setup 会生成 `.env.local`、`configs/local/llm_decision.local.yaml`、默认 run-id、COCO 路径检查报告和下一条 `optimize` 命令。

如果输出里有 `note:` 或报告里有 doctor error，先按提示修复。没有 `OPENAI_API_KEY` 时，setup 会创建占位 `.env.local`；设置好 key 后，默认 LLM proposal 才会参与策略生成。

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
