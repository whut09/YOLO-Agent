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

## 初始化和体检

```powershell
yolo-agent setup custom --data E:\dataset\my_dataset\data.yaml --model yolo26n.pt
```

## 启动 debug

```powershell
yolo-agent train --kind custom --model yolo26n.pt --data E:\dataset\my_dataset\data.yaml --run-id my-yolo26n
```

## 数据画像

`train` 会自动统计类别分布、框尺寸、小目标比例、空标签图片和缺失 label 文件，并把 JSON/Markdown 画像写入 run artifacts，不需要新人再运行单独命令。

## 注意事项

- 先跑 `debug`，不要直接 full profile
- 类别名、label 文件路径和图片路径必须先通过 `doctor`
- 小目标比例高时，应优先看数据画像和 error facts，再决定是否尝试 small-object recipe
- 没有 verified metrics 时，报告不会推荐最佳模型
