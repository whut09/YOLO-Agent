# 自定义 YOLO 数据集

> 本文所有命令示例均为单行 PowerShell 写法，可直接复制到 PowerShell 执行。PowerShell **不支持** Bash 的 `\` 续行符。

自定义数据集使用标准 YOLO `data.yaml`，通过 `train --kind custom` 进入与 COCO 相同的训练生命周期（profile、自动推进、恢复语义见[训练生命周期与运行模式](training-modes.md)）。本文说明四件事：真实的数据检查清单、自动画像（`DatasetReport`）统计了什么、标注能力的边界（自动 vs 仅建议）、哪些评估能力只有 COCO 格式才有。

## 1. data.yaml 与类名

```yaml
path: E:\dataset\my_dataset
train: images/train
val: images/val
names:
  0: defect
  1: scratch
```

- `names` 支持 list 与 dict 两种写法；`profile-data` 额外支持只写 `nc`（此时用序号当类名）。
- 类名读取有三处独立实现（`yolo_agent/agents/optimize_runner.py:1651-1660`、`yolo_agent/tools/dataset_stats.py:381-390`、`yolo_agent/core/label_quality.py:289-296`）。
- custom 数据集没有 `names` 时回退为 `["object"]`（`optimize_runner.py:1272-1273`）。
- 目录布局：`images/` 递归、`.txt` 图像列表文件、单图三种 split 写法都支持；label 路径按 `images → labels` 同名 `.txt` 推导（`dataset_stats.py:422-477`）。
- data.yaml 里的可选 `scene:` / `scenario:` 键只是透传为画像的描述性字符串字段（`dataset_stats.py:231`），不参与任何评估切分。

## 2. 三步上手

```powershell
yolo-agent doctor --kind custom --data E:\dataset\my_dataset\data.yaml --model yolo26n.pt
```

```powershell
yolo-agent setup custom --data E:\dataset\my_dataset\data.yaml --model yolo26n.pt
```

```powershell
yolo-agent train --kind custom --model yolo26n.pt --data E:\dataset\my_dataset\data.yaml --run-id my-yolo26n
```

- `doctor` 检查 data.yaml 可解析、`path:` 根目录与各 split 路径存在（custom 模式下 `test` 与 `annotations/` 缺失只是 warning）、run 目录可写、磁盘、`nvidia-smi`、torch CUDA 与 batch 估算（`yolo_agent/tools/doctor.py:79-131, 238-329`）。
- `setup custom` 只生成 `.env.local`、LLM 本地配置、默认 run-id 和 doctor 报告，**不生成数据画像**（`tools/setup_wizard.py:61-80`）。
- `train --kind custom` 复用 COCO preset（`cli.py:2412`），没有 custom 专用 preset 文件。

## 3. 数据画像何时发生

真实训练启动后，loop 驱动器的第一步就是 `profile_data` stage，第二步是 `advise_labels`（`agents/orchestrator.py:296-302`），都发生在训练开始前。画像写入：

- `runs/<run_id>/artifacts/dataset_report.json` 与 `dataset_report.md`
- 进度心跳 `artifacts/dataset_profile_progress.json`（`agents/data_stage_runner.py:47, 64-77`）

也可以独立运行：

```powershell
yolo-agent profile-data --data E:\dataset\my_dataset\data.yaml --out runs/dataset_report
```

画像只读 label 文本文件与文件元数据，**不解码图像**（`tools/dataset_stats.py:209`）。COCO 规模（约 16 万 label 文件）的扫描需要数分钟（`cli.py:2841-2843` 注释），custom 小库通常秒级。

**命名澄清**：自动统计的是 `DatasetReport`（`dataset_stats.py:119-135`）。代码里另有一个手填的 `DatasetProfile`（`core/schemas.py:11-18`：`name/path/image_count/class_count/average_objects_per_image`），它只是用户/TaskSpec 提供的 metadata，没有任何代码自动填充它——两者不要混淆。

## 4. Custom Dataset Checklist

下表是当前代码真实具备的检查项。**"发生位置"列是权威的**：不在表里的检查就不存在。

| 检查项 | 状态 | 发生位置 | 说明 |
| --- | --- | --- | --- |
| data.yaml 存在且可解析 | 有 | `doctor` + train preflight | preflight 中为 error 级，直接退出不建 run（`optimize_runner.py:1062-1069`） |
| train/val 路径存在 | 有 | `doctor` | custom 模式 `test`/`annotations/` 缺失仅 warning（`doctor.py:278-329`） |
| class id 越界 | 有 | 画像 | `class_id < 0 or >= nc` → "Class id out of range"（`dataset_stats.py:501-503`） |
| label 行格式错误 | 有 | 画像 | 少于 5 列或非数值 → "Malformed/Non-numeric label row"（`dataset_stats.py:490-499`） |
| 缺失 label 文件 | 有 | 画像 | `missing_label_files` 计数（`dataset_stats.py:267-278`）；Ultralytics 运行时也会打自己的 missing-labels 警告 |
| 空 label（纯背景图） | 有 | 画像 | `empty_label_images` 计数；占比 >30% 报 "High empty-image ratio detected"（`dataset_stats.py:282-283, 572-573`） |
| box 尺寸越界 | 部分 | 画像 | 只检查归一化 `w/h ∈ [0,1]`（`dataset_stats.py:503-504`）；**不检查 x_center/y_center 越界** |
| 可疑框几何（过小/近全图/极端长宽比） | 有 | `advise-labels` | 需要 `--predictions` 之外的静态规则即可运行；长宽比阈值等见 `label_quality.py:26-39, 235-261` |
| 类别不平衡 | 有 | 画像 | `class_balance_score` + 长尾判定（非零类 min/max < 0.2 → `class_imbalance_long_tail`，`dataset_stats.py:626-639, 725-726, 749-751`） |
| 目标尺寸分布 | 有 | 画像 | `object_size_ratio` small/medium/large（归一化面积阈值 0.01/0.05）+ `severe_small_object_bias`（`dataset_stats.py:546-563, 716-718`） |
| 重复图像 | 弱指纹 | 画像 | 指纹 = `文件名:文件字节大小`（`dataset_stats.py:702-707`）；**不是图像内容 hash**，改个名字或重压缩即可绕过 |
| 近重复视频帧 | 弱指纹 | 画像 | 同一弱指纹机制 → `high_duplicate_frames` / 建议 `deduplicate_near_duplicate_frames`（`dataset_stats.py:719-720, 742-743`） |
| train/val 泄漏 | 弱指纹 | 画像 | train∩val 指纹交集占 val 比例 → `train_val_leak_penalty` / `train_val_leakage`（`dataset_stats.py:692-699, 727-728`） |
| 场景/域分布 | 仅代理 | 画像 | `scene_diversity_score` 基于"不同父目录数 + split 数"打分（`dataset_stats.py:669-679`），**不是图像内容级场景统计**；评估侧按场景切分见 [数据优化](data-optimization.md) 的 scene slices |

画像还会给出整体健康分 `dataset_health`（0-100）：`class_balance_score`、`box_size_distribution_score`、`annotation_noise_score`、`scene_diversity_score`、`duplication_penalty`、`train_val_leak_penalty`，以及 `problems` 与 `recommendations` 列表（`dataset_stats.py:105-116, 595-623`）。

## 5. train preflight 检查什么（以及不检查什么）

train 的 preflight（`optimize_runner.py:1051-1110`）对 custom 数据集只检查：Python ≥3.10、`--data` 文件存在、`dataset_root` 存在、磁盘 ≥10GB（warning）、ultralytics 可导入、GPU 可用；full profile 另需 `--confirm-full-run`。

**preflight 不读取 label 内容。** 坏标签不会在训练前被 YOLO-Agent 拦下——要么由 Ultralytics 运行时报错，要么由画像/advise 阶段事后报告。所以第一次跑 custom 数据集的推荐顺序不变：`doctor` → `--dry-run` → `debug`（画像在 debug 训练前就会产出，先看 `dataset_report.md` 再继续）。

## 6. 标注质检与建议（advise-labels）

```powershell
yolo-agent advise-labels --data E:\dataset\my_dataset\data.yaml --predictions runs/my-yolo26n/artifacts/predictions.json --out runs/annotation_advice
```

`--predictions` 可选（YAML/JSON 归一化框）。四类真实 issue 类型（`core/label_quality.py:18-23`）：

| Issue | 含义 | 是否需要 predictions |
| --- | --- | --- |
| `suspected_missing_label` | 高置信预测周围没有 GT → 疑似漏标 | 需要 |
| `suspicious_box_geometry` | 极小框 / 近全图框 / 面积越界 / 长宽比 ≥12 | 不需要 |
| `class_confusion` | GT 与预测类别交叉混淆 | 需要 |
| `low_class_coverage` | 类实例数 <5 | 不需要 |

上游的 `AnnotationAdvisor`（`agents/annotation_advisor.py:73-95`）把画像 + issue 汇总成建议报告：`classes_to_collect`、`scenes_to_annotate`、`samples_for_review`、`boxes_to_redraw`、`labeling_tool_targets`、`recommendations`，写 JSON + Markdown。

**边界（必须清楚）**：

- **Agent 不修改任何标注文件。** 全链路没有任何写回用户 label 的代码；`active_learning_stage_runner.py:89` 的 `dataset_promote` docstring 明示 "without mutating data"。
- 唯一"自动"的标注相关行为是训练时的**内存过滤**（`AnnotationQualityFilterAdapter` / `AnnotationFilterDataset`，`components/adapters/data_pipeline/annotation.py:51-110`），只影响当次训练读取，不触磁盘。
- 跨轮持久化的 "hard sample queue" **不存在**；与 hard sample 最接近的机制是 active learning 挖掘（见[数据优化](data-optimization.md)）。
- "类定义不一致审计"只有动作名字符串（`error_facts.py:402` 的 `class_definition_audit`），**没有实现**。

## 7. COCO-only 与通用能力对照

custom 数据集能获得完整的训练链路，但**评估与误差分析能力大部分绑定 COCO 格式 evaluator**：

| 能力 | custom YOLO 数据集 | 说明 |
| --- | --- | --- |
| 训练全链路（profile/预算/ASHA/恢复/manifest） | 可用 | 与 COCO 同一套生命周期 |
| Ultralytics 指标（mAP50/50-95/precision/recall，results.csv） | 可用 | `parse_ultralytics_run`（`training.py:940-1013`） |
| 数据画像 / 标注建议 / active learning 挖掘 | 可用 | 本文第 3、6 节 |
| 延迟测量 / BatchTuner / baseline 导入 | 可用 | 数据集无关 |
| 固定 COCO post-eval（per-class AP、AP_small/medium/large） | 不可用 | `coco_post_eval.py` 依赖 pycocotools 或 faster-coco-eval + COCO 格式 GT；缺失即 RuntimeError |
| pilot evidence gate（要求 ap_small 等 COCO artifacts） | 不可用 | `core/pilot_evidence.py:34-166` |
| error facts 挖掘（`mine-coco-errors`） | 不可用 | 输入必须是 COCO GT json + 预测 json |
| scene slices（按场景切分的 recall） | 仅当自备 COCO 格式 GT + image metadata | 见[数据优化](data-optimization.md) |
| image-level paired bootstrap / matched error delta | 不可用 | 配对键要求 COCO 评估 artifacts 与 manifest hash 匹配 |

如果你能为 custom 数据集额外提供 COCO 格式的 GT json 和预测 json（并安装 `pycocotools`），`mine-coco-errors` 等工具可以独立运行，但这不是 `train --kind custom` 自动链路的一部分。

## 8. 数据集版本与 manifest

每次 run 初始化（`loop init` / `loop auto` / train）会自动给数据根建 manifest：`runs/<run_id>/dataset_versions/<version>/manifest.json`（`agents/run_initializer.py:96-122`）。两种模式：`sha256`（逐文件内容 hash）与 `metadata`（指纹 = `文件名:size:mtime_ns`，适合大库快速建版）。默认模式按入口区分：train / optimize 走 `coco_yolo26_auto` preset 默认 `metadata`，`loop init` / `loop auto` 的 CLI 默认是 `sha256`（`cli.py:1348, 1560`）。manifest 的用途是**跨 run/跨候选的哈希一致性**：baseline 验收（`require_dataset_manifest_match`）、候选提升（`same_dataset_manifest`）、hard-negative 协议、数据集晋升决策。

没有手动"构建/校验 manifest"的 CLI；`diff_manifests`（`core/dataset_versioning.py:122-143`）提供 added/removed/modified 差异。版本晋升与数据闭环见[数据优化](data-optimization.md)。

## 9. 注意事项

- 先跑 `debug`，不要直接 full profile；画像在训练前产出，先看 `dataset_report.md`。
- 类别名、label 路径、图片路径先过 `doctor --kind custom`。
- 小目标比例高时，先看画像的 `severe_small_object_bias` 与建议，再决定是否尝试 small-object 相关动作（映射见[数据优化](data-optimization.md)）。
- 候选提升要求通过 debug/pilot 门（`candidate_promotion.required_debug_metric` / `required_pilot_metric`）；没有 verified metrics 时，报告不会把候选当最佳模型推荐。

## 相关文档

- [数据优化与数据闭环](data-optimization.md) — error-driven augmentation、scene slices schema、active learning 数据闭环。
- [训练生命周期与运行模式](training-modes.md) — profile 表、auto-advance 与恢复语义。
- [证据架构](evidence.md) — 画像/建议报告与证据等级的声明边界。
- [系统架构](architecture.md) — 数据平面在整体架构中的位置。
