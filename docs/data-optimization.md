# 数据优化与数据闭环

> 本文所有命令示例均为单行 PowerShell 写法。PowerShell **不支持** Bash 的 `\` 续行符。

本文描述 Agent 在**数据侧**（标注、增广、采样、数据集版本）能做什么、不能做什么。所有结论以当前代码为准；未实现的能力明确标注"不存在"或"仅规划"，不假装存在。数据集本身的检查清单与画像见[自定义 YOLO 数据集](custom-dataset.md)。

## 1. 真实边界：数据动作默认 recommendation-only

自动优化循环的动作空间里，`data`、`label`、`postprocess`、`evidence` 四个域的 proposal 一律被标为 `execution_class="recommendation_only"`（`agents/auto_optimization_loop.py:911, 5043-5045`）。属于训练预算的动作只有两个：

```text
BUDGET_ACTIONS = {"promote", "refine"}   # core/error_round_decision.py:44-46
```

其余一切数据侧决策（收集数据、补标注、调增广）都只是"记录与建议"，不会创建训练 run，也不会修改你的数据集。

## 2. Augmentation 是 error-driven 动作，不是越多越好

增广调整的唯一合法入口是**误差诊断**：先有 error facts / error profile 指出具体错误类型，再有针对性的动作提案。`configs/error_action_policies.yaml` 是错误类型 → 动作的真实映射，例如：

| 错误类型 | 真实动作条目（id） | 说明 |
| --- | --- | --- |
| `small_object_miss` | `increase_imgsz`、`add_p2_head`、`enable_tiling_inference`（SAHI）、`switch_to_nwd_loss`、`increase_positive_assigner_weight` | 小目标漏检优先考虑分辨率与 loss/分配器，而不是盲目加增广 |
| `occlusion_miss` | `add_occlusion_augmentation`（组件 `augmentation.occlusion`） | 受控 cutout/copy-paste；过度会伤干净样本精度 |
| `low_contrast_miss` | `add_contrast_augmentation`（组件 `augmentation.low_contrast`）、`increase_imgsz` | 对比度/亮度增广策略 |
| `background_confusion` | `add_hard_negative_mining`、`reduce_mosaic_strength`（`mosaic: lower`）、`add_background_only_sampling`、`increase_focal_loss_gamma` | 背景 FP 先挖 hard negative、降 mosaic，而不是加增广 |
| `out_of_distribution_miss` | `collect_ood_samples` | 红外、夜间等 domain 偏移没有专属策略条目，走 OOD 收集建议 + scene slices 的 `domain` tag（见第 4 节） |

没有对应错误类型的增广调整不会被提案——**"默认把 mosaic/mixup 拉满"不在任何策略里**。YOLO-Agent 也不设置默认增广超参：`UltralyticsTrainingConfig` 没有任何 mosaic/mixup 字段（`adapters/ultralytics/training.py:149-179`），未显式 override 时用的就是 Ultralytics 自身默认值。

### 双轨执行模型

数据/增广提案只有两条路径能碰到真实训练：

1. **轨道 A — SAFE Ultralytics 超参 override（真实执行）**。`SAFE_ULTRALYTICS_OVERRIDE_KEYS`（`agents/auto_optimization_loop.py:876-909`）允许 `mosaic/mixup/copy_paste/close_mosaic/hsv_h/hsv_s/hsv_v/degrees/translate/scale/shear/perspective/flipud/fliplr/erasing/crop_fraction` 等键透传给 Ultralytics CLI。策略阶段会把诊断动作物化为具体超参，例如 `light_mixup → mixup: 0.05`、`close_mosaic_early → close_mosaic: 5`、`mosaic: lower → mosaic: 0.2`（`agents/policy_stage_runner.py:1323-1383`、`agents/error_driven_loop.py:521-526`）。
2. **轨道 B — data pipeline adapters（可执行，但成熟度受限）**。`RareClassCopyPasteAdapter`、scale-aware crop、multi-image sampling 等 9 个 paper-data 契约组件挂在 Ultralytics trainer hooks 上（契约清单 `configs/components/data_pipeline/paper_data_adapters.yaml` 恰 9 条；hook 集合 `adapters/ultralytics/plugin_bridge.py:29-42`）。`components/adapters/data_pipeline/adapters.py:421-491` 实际定义 12 个具体 adapter 类——上述 9 个契约类之外还有 `AnnotationQualityFilterAdapter`、`NormalizationPreprocessingAdapter`、`ActiveLearningAcquisitionAdapter` 三个不在 paper-data 契约清单中的类。它们有真实生效测试（`tests/test_data_pipeline_ultralytics_bridge.py:54, 83`），但 paper-data 契约的 maturity 全部是 `adapter_implemented`（`configs/components/data_pipeline/paper_data_adapters.yaml`），**不是 certified**，启用需要 paper route 证据（`research/paper_data_side.py:196-303`）。

两个反面例子（防止误解）：

- `augmentation_policy` 作为 bundle 键**不在** SAFE 集合内 → 提案会被判为 `adapter_required`，不会直接执行（`auto_optimization_loop.py:5174-5177`）。
- `data.augmentation.small_copy_paste` 在动作目录里**没有 `component_ids`** → 只产生推荐，不产生训练（`configs/actions/detection_action_catalog.yaml:125-144`）。

## 3. request_annotation 与 collect_data

这两个是轮次决策的枚举动作（`core/error_round_decision.py:31-42`），不是可执行命令：

- 触发规则：primary delta < 0 且数据不平衡为 medium/high 时发 `collect_data`，否则发 `request_annotation`（`error_round_decision.py:253-270`）。
- 它们不属于 `BUDGET_ACTIONS`，只把问题写进下一轮的规划信号（如 `missing_labels` problem tag，`agents/autonomous_loop.py:256-261`），不会自己创建任何 run 或数据。
- 注意：`next_round.yaml` 里的 `pilot_evidence_actions` 是另一回事——那是 COCO 证据恢复动作（`run_coco_post_eval` / `import_coco_eval` / `mine_coco_errors` / `repair_coco_evidence_artifacts` / `import_current_node_error_facts`），与 request_annotation 无关（`core/pilot_evidence.py:134-144`）。

## 4. Scene slices 与自定义 metadata

`DetectionErrorProfile.scene_slices` 支持按场景切分的 recall 统计（`core/detection_error_profile.py:149-158, 187`）：

- **固定 6 个 tag**：`day_night`、`indoor_outdoor`、`weather`、`distance`、`camera`、`domain`（`detection_error_profile_builder.py:51`）。不能扩展任意自定义 slice 名。
- 每个 slice 记录 `slice_name / gt_count / true_positives / recall`；`ap50` 字段存在但恒为 `None`（`detection_error_profile_builder.py:447-485`）。
- metadata 覆盖不到的 tag 会全部列进 `unavailable_scene_slices`——缺失可审计，而不是被静默吞掉（`detection_error_profile_builder.py:394-401`）。

可选的图片 metadata JSON 支持两种 schema（`detection_error_profile_builder.py:100-122`）。两种 schema 的顶层键都必须是**整数 image_id**（代码用 `int(key)` 解析，文件名键会直接解析失败）：

```json
{
  "1": {"day_night": "night", "weather": "rain", "domain": "infrared"},
  "2": {"day_night": "day", "weather": "clear", "domain": "visible"}
}
```

或 COCO 风格：

```json
{
  "images": [
    {"id": 1, "file_name": "000001.jpg", "day_night": "night", "domain": "infrared"}
  ]
}
```

限制：GT 必须是 **COCO 格式**（builder 走 `_load_coco_ground_truth`），纯 YOLO txt 数据集不能直接生成 scene slices；也没有任何从图像内容自动推断 day_night/weather 的代码——metadata 全靠用户提供。custom 数据集想用这个能力，需要自备 COCO 格式 GT json + metadata json 并安装 `pycocotools` 或 `faster-coco-eval`。

## 5. 数据闭环（active learning）

目标闭环是：`inference → hard samples → review → dataset version → retrain`。当前代码的状态是**可执行的部分闭环**：

| 环节 | 状态 | 真实入口 |
| --- | --- | --- |
| ① 挖掘 hard samples | 有 | `loop mine --run runs/<run_id> --predictions <unlabeled.json> --target generic|cvat|label_studio`；三策略：`low_confidence` / `high_entropy` / `model_disagreement`（`agents/active_learning.py:114-153`） |
| ② 生成标注交接 | 有 | 上一步同时产出 `labeling_manifest.json` + `active_learning_plan.json`（`agents/active_learning_stage_runner.py:61-86`），target 支持 cvat / label_studio / generic |
| ③ 人工 review | 外部输入 | 没有 CVAT/Label Studio API 集成执行；review 结果由用户以 `ReviewedLabels` YAML/JSON 提供（`status: accepted/rejected/needs_fix`） |
| ④ 晋升决策 | 有 | `loop dataset-promote --run runs/<run_id> --reviewed-labels <file>`；门槛：`min_reviewed_ratio=1.0`、`max_rejected_ratio=0.2`（`core/dataset_promotion.py:69-205`） |
| ⑤ 数据集版本 | 有 | manifest 自动建于 run 初始化（sha256/metadata 两种指纹模式，`core/dataset_versioning.py:75-102`）；`diff_manifests` 给 added/removed/modified |
| ⑥ 用新版本重训 | 部分 | promote 只产出**决策**，不建数据、不自动触发重训；重训入口仍然是手动 `train --data <new version>` |

诚实结论：**全自动主动学习闭环不存在。** 标注完全外部化（只有 manifest 交接），review 是外部文件，promote 明确 "without mutating data"。数据闭环的自动化程度止步于：挖掘、交接清单、版本 manifest、晋升判断。

## 6. 相关文档

- [自定义 YOLO 数据集](custom-dataset.md) — 数据检查清单、`DatasetReport` 画像、标注质检边界。
- [自动优化架构](automatic-optimization-architecture.md) — 数据动作在完整决策循环中的位置（数据动作 ≠ 训练 run）。
- [训练生命周期与运行模式](training-modes.md) — 数据侧动作产出后的训练入口。
- [证据架构](evidence.md) — error facts / error profile 如何支撑 error-driven 提案。
