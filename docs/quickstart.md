# 快速开始

最快路径是先跑一个安全的 debug run。debug 只验证最小训练链路，不代表最终模型效果。

## 1. 检查环境

```powershell
yolo-agent doctor --data E:\dataset\coco.yaml --model yolo26n.pt
```

如果输出里有 `fix:`，先按提示修复。

## 2. 启动 COCO + YOLO26 debug

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

## 3. 查看状态

```powershell
yolo-agent loop status --run runs/coco-yolo26n
```

状态面板会显示当前 stage、queue counts、训练心跳、已有 evidence、blocked reason 和下一条建议命令。

## 4. 进入 pilot

debug 通过后，再进入 pilot：

```powershell
yolo-agent optimize advance `
  --run runs/coco-yolo26n `
  --to-profile pilot `
  --execute
```

## 5. full COCO 训练

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
doctor -> debug -> status -> pilot -> import evidence -> baseline_full -> baseline_confirm -> candidate_full
```

不要一上来直接跑 full COCO。先把 debug 和 pilot 跑硬，才能让后续优化有可信证据。

