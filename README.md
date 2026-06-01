# GANormaly

基于生成对抗网络（GAN）的无缺陷样本异常检测系统。仅使用正常（非缺陷）样本进行训练，通过编码空间距离实现缺陷检测。

## 目录

- [方法原理](#方法原理)
  - [创新点：LPIPS感知损失驱动的高分辨率异常检测](#创新点lpips感知损失驱动的高分辨率异常检测)
- [网络结构](#网络结构)
- [损失函数](#损失函数)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [使用方法](#使用方法)
- [超参数说明](#超参数说明)
- [训练细节](#训练细节)
- [评估指标](#评估指标)
- [实验结果](#实验结果)
- [自定义网络块说明](#自定义网络块说明)
- [参考文献](#参考文献)

## 方法原理

### 核心思想

GANormaly 基于 GANomaly 框架，核心思想是：**正常样本可以被很好地重建，而异常样本的重建会出现显著偏差**。

训练阶段仅使用正常样本，模型学习正常数据的分布特征。测试阶段，异常样本由于偏离正常分布，其编码距离会显著高于正常样本。

### 创新点：LPIPS感知损失驱动的高分辨率异常检测

原始 GANomaly（2018）仅使用 MSE/L1 等像素级损失进行重建，这导致在高分辨率图像上重建结果模糊，细节丢失严重，因此只能在低分辨率（如 32×32、64×64）场景下使用——这也是 GAN 类重建网络的固有缺陷。

本项目引入 **LPIPS（Learned Perceptual Image Patch Similarity）感知损失**，基于预训练 AlexNet 的深度特征计算感知距离，而非像素级距离。这使得模型能够：

1. **保留高频细节**：纹理、边缘、缺陷特征不会被模糊掉
2. **高分辨率重建**：在 512×512 乃至更高分辨率下仍能保持良好的重建质量
3. **缺陷检测精度提升**：重建质量越高，正常/异常样本的编码距离分离度越大

```
原始 GANomaly (2018):
  L = w_adv·L_adv + w_con·L_con + w_enc·L_enc
  → 低分辨率可用，高分辨率重建模糊

本项目 (GANormaly):
  L = w_adv·L_adv + w_con·L_con + w_lpips·L_lpips + w_enc·L_enc
  → 高分辨率清晰重建，缺陷检测能力显著提升
```

### 训练流程

```
输入图像 x (正常样本)
    │
    ├──→ [G_Encoder] ──→ z (编码向量) + feature_map (特征图)
    │                                        │
    │                                        ▼
    │                               [G_Decoder] ──→ x' (重建图像)
    │                                        │
    │                                        ▼
    ├──→ [D_MLP] ──→ 真/假概率 + z_D (判别器编码)
    │         ▲
    │         │
    └─────── x' 送入判别器
```

### 异常检测流程

```
测试图像 x_test
    │
    ├──→ [G_Encoder] ──→ z, feature_map
    │                         │
    │                         ▼
    │                    [G_Decoder] ──→ x'_test (重建图像)
    │                                        │
    │                                        ▼
    │                               [E (额外编码器)] ──→ z_e
    │
    └──→ L_enc = ||z - z_e||₂  ← 编码距离（异常分数）

    正常样本：L_enc 小（重建质量好，编码一致）
    异常样本：L_enc 大（重建质量差，编码不一致）
```

### 判别标准

- **正常样本**：重建误差小，编码距离低 → 判定为正常
- **异常样本**：重建误差大，编码距离高 → 判定为异常

## 网络结构

### 整体架构

| 模块 | 类名 | 说明 |
|------|------|------|
| **生成器编码器** | `G_E` | 输入图像 → 潜在向量 z + 特征图，4级下采样，带 Dropout2d |
| **生成器解码器** | `G_D` | 特征图 → 重建图像，4级上采样（深度可分离转置卷积） |
| **判别器** | `D_MLP` | 输入图像 → 真/假概率 + 编码向量 z，用于对抗训练 |
| **额外编码器** | `E` | 重建图像 → 编码向量 z_e，用于计算编码距离 |

### G_E 编码器

```
输入: [batch, 3, 512, 512]  (RGB图像，经 Resolution_Multiplier 缩放)

第一阶段 (512通道):
  DWConv2d(3→512, firstBlock) → LeakyReLU
  DWConv2d(512→512) → LeakyReLU
  DWConv2d(512→512) → LeakyReLU
  DWConv2d(512→512, stride=2)  ← 下采样

第二阶段 (256通道):
  DWConv2d(512→256) → LeakyReLU
  DWConv2d(256→256) → LeakyReLU
  DWConv2d(256→256) → LeakyReLU
  DWConv2d(256→256, stride=2)  ← 下采样

第三阶段 (128通道):
  DWConv2d(256→128) → LeakyReLU
  DWConv2d(128→128) → LeakyReLU
  DWConv2d(128→128) → LeakyReLU
  DWConv2d(128→128, stride=2)  ← 下采样

第四阶段 (64→latent_size通道):
  DWConv2d(128→64) → LeakyReLU
  DWConv2d(64→64) → LeakyReLU
  DWConv2d(64→64) → LeakyReLU
  DWConv2d(64→100, stride=2, endBlock)  ← 下采样

Dropout2d(0.3) → AdaptiveAvgPool2d(1) → MLP(100→100)

输出: z [batch, 100], feature_map [batch, 100, H/16, W/16]
```

### G_D 解码器

```
输入: feature_map [batch, 100, H/16, W/16]

第一阶段:
  DWConvTranspose2d(100→64, stride=2, firstBlock) ← 上采样
  DWConv2d(64→64) → LeakyReLU × 2
  DWConv2d(64→128) → LeakyReLU

第二阶段:
  DWConvTranspose2d(128→128, stride=2) ← 上采样
  DWConv2d(128→128) → LeakyReLU × 2
  DWConv2d(128→256) → LeakyReLU

第三阶段:
  DWConvTranspose2d(256→256, stride=2) ← 上采样
  DWConv2d(256→256) → LeakyReLU × 2
  DWConv2d(256→512) → LeakyReLU

第四阶段:
  DWConvTranspose2d(512→512, stride=2) ← 上采样
  DWConv2d(512→512) → LeakyReLU × 2
  DWConv2d(512→3, endBlock) → Tanh

输出: 重建图像 [batch, 3, 512, 512]
```

### D_MLP 判别器

```
输入: [batch, 3, 512, 512]

DWConv2d(3→512, firstBlock) → LeakyReLU
DWConv2d(512→512) → Dropout2d(0.4)
AdaptiveAvgPool2d(1) → 展平 [batch, 128]  (512 × Width_Multiplier=0.25)

MLP:
  Linear(128→100) → LeakyReLU
  Linear(100→100) → LeakyReLU

Linear(100→1) → Sigmoid

输出: 真/假概率 [batch, 1], 编码向量 z [batch, 100]
```

### E 额外编码器

结构与 G_E 相同，独立参数。用于对重建图像进行编码，计算与原始编码的一致性。

### 关键设计

- **深度可分离卷积**：将标准卷积分解为 Depthwise + Pointwise，计算量降为原来的 `(1/N + 1/DK²)`
- **宽度乘子 (Width_Multiplier)**：统一缩放所有中间层通道数，控制模型规模
- **分辨率乘子 (Resolution_Multiplier)**：缩放输入图像尺寸，平衡精度与效率
- **残差连接**：DWConv2d 中，当输入输出通道数相同且 stride=1 时自动启用

## 损失函数

总损失：

```
L_total = w_adv × L_adv + w_con × L_con + w_lpips × L_lpips + w_enc × L_enc + w_anti × L_anti
```

| 损失 | 公式 | 默认权重 | 作用 |
|------|------|----------|------|
| **L_adv** (对抗损失) | `‖z_D(x) - z_D(x')‖₂` | w_adv=1 | 使重建图像的判别器编码逼近真实图像 |
| **L_con** (重建损失) | `‖x - x'‖₁` | w_con=10 | 像素级重建一致性（L1距离） |
| **L_lpips** (感知损失) | `LPIPS(x, x')` | w_lpips=20 | 基于深度特征的感知相似度，保留纹理细节 |
| **L_enc** (编码损失) | `‖z - z_e‖₂` | w_enc=1 | 编码一致性：G_E编码 vs E编码 |
| **L_anti** (多样性损失) | 编码相似度 + 标准差惩罚 | w_anti=0 | 防止编码空间坍缩（默认关闭） |

### 损失详解

- **L_adv**：不是用判别器的真假概率，而是比较判别器对真实图和重建图的编码向量距离
- **L_con**：L1损失比MSE更不容易模糊，但仍需LPIPS补充感知质量
- **L_lpips**：使用AlexNet预训练特征，衡量高层语义相似度。这是本项目的核心创新——弥补了原始GANomaly在高分辨率下重建模糊的缺陷
- **L_enc**：训练额外编码器E，使其对重建图的编码与G_E对原图的编码一致
- **L_anti**：计算batch内编码向量的余弦相似度，惩罚过高的相似度和过低的标准差

### 防崩溃策略

- **标签平滑**：real=0.9, fake=0.1，防止判别器过度自信
- **Dropout**：判别器使用 Dropout2d(0.4)，编码器使用 Dropout2d(0.3)
- **LeakyReLU**：判别器使用 LeakyReLU(0.2)，避免梯度消失

## 项目结构

```
GANormaly/
├── GANormaly.py          # 主训练脚本（模型定义 + 训练循环）
├── addBlock.py           # 自定义网络块（DWConv2d, DWConvTranspose2d, ResidualBlock等）
├── test.py               # 异常检测测试与评估脚本
├── ckpt/                 # 模型权重保存目录
│   ├── D_MLP_{epoch}.ckpt
│   ├── G_Encoder_{epoch}.ckpt
│   ├── G_Decoder_{epoch}.ckpt
│   └── add_Encoder_{epoch}.ckpt
├── pstimg(no LPIPS)/     # 无LPIPS基线实验结果（网络结构、重建图、评估指标）
├── samples1/             # 训练过程中生成的重建图像
│   ├── real_image-{epoch}.png
│   └── fake_image-{epoch}.png
├── D_MLP.ckpt            # 训练结束后保存的最终判别器权重（脚本目录）
├── G_Encoder.ckpt        # 训练结束后保存的最终编码器权重
├── G_Decoder.ckpt        # 训练结束后保存的最终解码器权重
├── add_Encoder.ckpt      # 训练结束后保存的最终额外编码器权重
└── data/data_root/       # 数据集目录
    ├── train/
    │   └── normal/       # 训练用正常样本
    └── test/
        ├── normal/       # 测试用正常样本
        └── anormaly/     # 测试用异常样本
```

## 环境依赖

```
torch >= 1.9
torchvision
lpips
matplotlib
numpy
pandas
tqdm
torchsummary
torchinfo
torchviz
scikit-learn
```

安装：

```bash
pip install torch torchvision lpips matplotlib numpy pandas tqdm torchsummary torchinfo torchviz scikit-learn
```

## 使用方法

### 1. 准备数据

按上述目录结构放置正常/异常样本图片。支持的格式：jpg, jpeg, png, bmp。

```
data/data_root/
├── train/
│   └── normal/          # 训练集：仅正常样本
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── test/
    ├── normal/          # 测试集：正常样本
    │   ├── img001.jpg
    │   └── ...
    └── anormaly/        # 测试集：异常样本
        ├── img001.jpg
        └── ...
```

### 2. 训练

```bash
python GANormaly.py
```

训练过程中：
- 每个 epoch 在 `samples1/` 保存重建图像（real/fake对比）
- 每 2 个 epoch 在 `ckpt/` 保存模型权重（带epoch编号）
- 训练结束后在脚本目录保存最终模型权重
- 训练结束后显示 D_loss / G_loss 曲线图

### 3. 测试评估

```bash
python test.py
```

输出内容：
- 正常/异常样本编码距离分布直方图
- 编码距离箱线图对比
- ROC 曲线及 AUC 分数
- PR 曲线及最佳阈值
- 性能指标随阈值变化曲线
- 完整评估报告（检测率、F1分数、精确率、召回率等）

评估结果图片保存为：
- `encoding_distance_histogram.png`
- `roc_curve.png`
- `threshold_performance.png`

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.002 | Adam优化器学习率 |
| `beta1` | 0.5 | Adam优化器一阶矩衰减率 |
| `beta2` | 0.999 | Adam优化器二阶矩衰减率 |
| `batch_size` | 2 | 批次大小 |
| `num_epoch` | 50 | 训练轮数 |
| `latent_size` | 100 | 潜在空间维度 |
| `input_channel` | 3 | 输入图像通道数（RGB=3） |
| `image_width` | 1024 | 原始图像宽度 |
| `image_height` | 1024 | 原始图像高度 |
| `Width_Multiplier` | 0.25 | 网络宽度乘子，控制中间层通道数 |
| `Resolution_Multiplier` | 0.5 | 图像分辨率乘子，实际输入 = 1024×0.5 = 512 |
| `w_adv` | 1 | 对抗损失权重 |
| `w_con` | 10 | L1重建损失权重 |
| `w_lpips` | 20 | LPIPS感知损失权重 |
| `w_enc` | 1 | 编码一致性损失权重 |
| `w_anti` | 0 | 多样性损失权重（默认关闭） |
| `w_D` | 1 | 判别器损失权重 |

### Width_Multiplier 说明

Width_Multiplier 会缩放所有中间层的通道数。例如：
- `Width_Multiplier=0.25`：512通道 → 128通道，256通道 → 64通道
- 降低此值可大幅减少参数量和计算量，但可能影响模型表达能力

### Resolution_Multiplier 说明

Resolution_Multiplier 缩放输入图像尺寸。例如：
- `Resolution_Multiplier=0.5`：1024×1024 → 512×512
- 降低此值可加快训练速度，但可能丢失细节信息

## 训练细节

### 优化器配置

| 模块 | 优化器 | 权重衰减 |
|------|--------|----------|
| D_MLP (判别器) | Adam | 1e-5 |
| G_Encoder (编码器) | Adam | 无 |
| G_Decoder (解码器) | Adam | 无 |
| add_Encoder (额外编码器) | Adam | 无 |

### 训练策略

- **判别器弱化**：使用 Dropout2d(0.4)、标签平滑、精简层数（2 conv + 2 MLP）
- **梯度清零**：每个步骤前调用 `reset_grad()` 清零所有优化器
- **训练分离**：训练D时G不更新（`torch.no_grad()`），训练G时D不更新

### 模型保存

训练过程中保存两种模型：
1. **中间检查点**（`ckpt/` 目录）：每2个epoch保存，文件名带epoch编号
2. **最终模型**（脚本目录）：训练结束后保存，文件名不带epoch编号

## 评估指标

测试脚本输出以下评估指标：

| 指标 | 说明 |
|------|------|
| **ROC-AUC** | ROC曲线下面积，衡量整体区分能力 |
| **PR-AUC** | 精确率-召回率曲线下面积 |
| **最佳阈值 (Youden)** | 基于Youden指数 (TPR-FPR) 的最优阈值 |
| **最佳F1分数** | 精确率和召回率的调和平均最大值 |
| **检测率** | 不同阈值下的异常检出率 |
| **假阳性率** | 正常样本被误判为异常的比例 |

AUC评估等级：
- AUC >= 0.9：优秀 (Excellent)
- AUC >= 0.8：良好 (Good)
- AUC >= 0.7：一般 (Fair)
- AUC < 0.7：较差 (Poor)

## 实验结果

### 网络结构

![网络结构](pstimg(no%20LPIPS)/network.png)

### 无 LPIPS 损失的基线结果

以下为不使用 LPIPS 感知损失时的实验结果（仅使用 L1 重建损失）：

**原图 vs 重建图：**

| 原图 | 重建图（无LPIPS） |
|:----:|:----:|
| ![原图](pstimg(no%20LPIPS)/ori.png) | ![重建图](pstimg(no%20LPIPS)/fake.png) |

**评估指标：**

| 指标 | 结果图 |
|------|--------|
| 编码距离分布 | ![编码距离分布](pstimg(no%20LPIPS)/encoding_distance_histogram.png) |
| ROC 曲线 | ![ROC曲线](pstimg(no%20LPIPS)/roc_curve.png) |
| 阈值性能分析 | ![阈值性能](pstimg(no%20LPIPS)/threshold_performance.png) |

> 无 LPIPS 时，重建图像存在明显模糊，高频细节（纹理、边缘）丢失严重，导致异常检测精度受限。加入 LPIPS 感知损失后，重建质量显著提升，正常/异常样本的编码距离分离度更大。

## 自定义网络块说明

### DWConv2d（深度可分离卷积）

```
标准卷积: Conv2d(in, out, 3×3) → 参数量 = in × out × 9
深度可分离: Depthwise(in, 3×3) + Pointwise(in→out, 1×1) → 参数量 = in × 9 + in × out
```

计算复杂度约为标准卷积的 `(1/N + 1/DK²)`，其中 N 为输出通道数，DK 为卷积核尺寸。

支持 `firstBlock`（输入不缩放）和 `endBlock`（输出不缩放）标志，用于网络首尾层。

### DWConvTranspose2d（深度可分离转置卷积）

上采样版本的深度可分离卷积，结构类似：
1. Depthwise Transpose Conv：空间上采样
2. Pointwise Conv：通道混合

### ResidualBlock（残差块）

支持两种模式：
- **标准模式**：Conv → BN → ReLU → Conv → BN → Add → ReLU
- **预激活模式**：BN → ReLU → Conv → BN → ReLU → Conv → Add

### ResidualBottleneckBlock（瓶颈残差块）

1×1降维 → 3×3特征提取 → 1×1升维，减少计算量。支持 `expansion=4` 扩展系数。

## 参考文献

- [GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training](https://arxiv.org/abs/1805.06725)
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) (LPIPS)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (深度可分离卷积)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (残差连接)
