# 自定义 YOLO 数据集

自定义数据集使用标准 YOLO `data.yaml`。

## 示例 data.yaml

```yaml
path: E:\dataset\my_dataset
train: images/train
val: images/val
names:
  0: defect
  1: scratch
```

## 体检

```powershell
yolo-agent doctor --data E:\dataset\my_dataset\data.yaml --model yolo26n.pt
```

## 启动 debug

```powershell
yolo-agent optimize custom `
  --model yolo26n.pt `
  --data E:\dataset\my_dataset\data.yaml `
  --run-id my-yolo26n `
  --profile debug `
  --execute
```

## 数据画像

```powershell
yolo-agent profile-data --data E:\dataset\my_dataset\data.yaml --out runs/my-yolo26n/dataset_report
```

它会统计类别分布、框尺寸、小目标比例、空标签图片、缺失 label 文件，并输出 JSON 和 Markdown。

## 注意事项

- 先跑 `debug`，不要直接 full profile
- 类别名、label 文件路径和图片路径必须先通过 `doctor`
- 小目标比例高时，应优先看数据画像和 error facts，再决定是否尝试 small-object recipe
- 没有 verified metrics 时，报告不会推荐最佳模型

