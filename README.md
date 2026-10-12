# AlexNet 迁移学习图像分类

基于 **AlexNet** 的迁移学习图像分类实验，在 4 类小规模数据集上对比**特征提取**与**全网络微调**两种策略，并配有 **Grad-CAM** 可解释性分析。机器智能课程实验。

## 项目结构

```
alexnet_model.py   ← 模型定义（手动复现 AlexNet 结构 + 加载 ImageNet 预训练权重）
       ↓
   train.py        ← 训练（数据增强 → 冻结/微调 → 保存 checkpoint）
       ↓
  evaluate.py      ← 评估（加载 checkpoint → 测试集推理 → 分类报告）
  predict.py       ← 单图推理 + 中间特征图可视化
  gradcam.py       ← Grad-CAM 热力图，观察网络"看哪里"
```

各模块通过 `checkpoints/best_alexnet.pth` 解耦：训练一次，其余模块独立加载。

## 环境

```bash
pip install -r requirements.txt
```

## 数据

使用 `ImageFolder` 结构，根目录下分 `train/` 与 `test/`，每类一个子文件夹：

```
dataset/
├── train/{bottle, car, dog, sombrero}/
└── test/{bottle, car, dog, sombrero}/
```

> 数据集为 ImageNet 子集（4 类），因体积与版权原因**未包含在本仓库中**，请自行准备同类结构的数据。
> 类别索引见 [`checkpoints/class_to_idx.json`](checkpoints/class_to_idx.json)。

## 用法

```bash
# 方案一：特征提取 —— 冻结全部卷积层，只训练分类头
python train.py --epochs 5 --batch-size 32 --lr 1e-3 --freeze-conv-layers 5

# 方案二：全网络微调 —— 解冻全部卷积层
python train.py --epochs 10 --batch-size 16 --lr 1e-4 --unfreeze-features

# 冻结前 N 个卷积层（N = 0..5）
python train.py --freeze-conv-layers 2

# 评估
python evaluate.py --checkpoint checkpoints/best_alexnet.pth

# 单图预测（--visualize 可导出中间特征图）
python predict.py path/to/image.jpg --topk 4 --visualize

# Grad-CAM 可解释性分析
python gradcam.py path/to/image.jpg --checkpoint checkpoints/best_alexnet.pth
```

完整参数说明见 [`commands.txt`](commands.txt)，设计原理见 [`代码设计思路及原理.md`](代码设计思路及原理.md)。

## 输出

| 文件 | 内容 |
|---|---|
| `outputs/predict_visualization.png` | 单图预测 + 中间层特征图 |
| `outputs/gradcam.png` | Grad-CAM 热力图叠加 |
