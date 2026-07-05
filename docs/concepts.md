# 核心概念

YOLO Agent 把检测效果视为完整系统问题，而不仅是模型结构问题。

## 受控闭环

```text
任务 + 数据 + 错误样本 + 部署约束
        -> 诊断
        -> 策略提案
        -> 受保护的候选实验
        -> 证据
        -> 下一轮
```

核心规则：LLM、人类和规则引擎只能提出策略；只有 evaluator 和 evidence gate 才能把策略变成实验候选。

## 优化对象

- 模型尺寸和 YOLO family
- backbone、neck、head、loss、assigner、optimizer 元数据
- 标注质量和复标 worklist
- 数据健康度、采样、划分泄漏、重复帧
- 数据增强策略
- 后处理策略，例如 NMS、threshold、TTA、SAHI
- 部署限制，例如 latency、FPS、导出格式和模型大小
- 实验可复现性、消融纪律和证据质量

## 自动化成熟度

当前成熟度：Level 4，具备 Level 5 的基础模块。

- Level 1: schema + metadata
- Level 2: guarded candidate generation
- Level 3: evidence-driven loop
- Level 4: queued execution + cross-run learning
- Level 5: active learning + dataset version evolution

## 非目标

- 默认启动真实训练
- 复制未经验证的第三方 loss 实现
- 让 LLM 输出直接决定实验
- 在没有 evidence 时推荐最佳模型
- 用编造指标隐藏缺失 evidence

