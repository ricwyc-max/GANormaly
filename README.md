# GANormaly

基于生成对抗网络（GAN）的无缺陷样本异常检测。仅使用正常样本训练，通过编码空间距离实现缺陷检测。

## 方法原理

```
输入图像 x → [G_Encoder] → z, feature_map → [G_Decoder] → 重建图像 x'

异常判定：
L_enc = ||G_E(x) - E(G_D(G_E(x)))||₂
```

- 正常样本：重建误差小，编码距离低
- 异常样本：重建误差大，编码距离高

## 网络结构

| 模块 | 说明 |
|------|------|
| **G_E (编码器)** | 输入图像 → 潜在向量 z + 特征图，4级下采样 + Dropout2d |
| **G_D (解码器)** | 特征图 → 重建图像，4级上采样（深度可分离转置卷积） |
| **D_MLP (判别器)** | 输入图像 → 真/假概率 + 编码向量 z，用于对抗训练 |
| **E (额外编码器)** | 重建图像 → 编码向量 z_e，用于计算编码距离 |

核心网络块基于 **深度可分离卷积**（DWConv2d），大幅减少参数量。

## 损失函数

| 损失 | 公式 | 权重 | 作用 |
|------|------|------|------|
| L_adv | \|\|z_D(x) - z_D(x')\|\|₂ | w_adv=1 | 对抗损失，欺骗判别器 |
| L_con | \|x - x'\|₁ | w_con=10 | L1像素重建损失 |
| L_lpips | LPIPS(x, x') | w_lpips=20 | 感知损失，保留细节纹理 |
| L_enc | \|\|z - z_e\|\|₂ | w_enc=1 | 编码一致性损失 |
| L_anti | 多样性惩罚 | w_anti=0 | 防止编码空间坍缩 |

## 项目结构

```
GANormaly/
├── GANormaly.py          # 主训练脚本
├── addBlock.py           # 自定义网络块（DWConv2d, DWConvTranspose2d, ResidualBlock等）
├── test.py               # 异常检测测试与评估脚本
├── ckpt/                 # 模型权重保存目录
├── samples1/             # 训练过程中生成的图像
└── data/data_root/       # 数据集目录
    ├── train/
    │   └── normal/       # 训练用正常样本
    └── test/
        ├── normal/       # 测试用正常样本
        └── anormaly/     # 测试用异常样本
```

## 环境依赖

```
torch
torchvision
lpips
matplotlib
numpy
torchsummary
torchinfo
torchviz
scikit-learn
```

安装：

```bash
pip install torch torchvision lpips matplotlib numpy torchsummary torchinfo torchviz scikit-learn
```

## 使用方法

### 1. 准备数据

按上述目录结构放置正常/异常样本图片。

### 2. 训练

```bash
python GANormaly.py
```

训练过程中每个 epoch 会在 `samples1/` 保存重建图像，每 2 个 epoch 在 `ckpt/` 保存模型权重。

### 3. 测试评估

```bash
python test.py
```

输出内容：
- 正常/异常样本编码距离分布直方图
- ROC 曲线及 AUC 分数
- PR 曲线及最佳阈值
- 完整评估报告（检测率、F1分数等）

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 0.002 | 学习率 |
| batch_size | 8 | 批次大小 |
| num_epoch | 50 | 训练轮数 |
| latent_size | 100 | 潜在空间维度 |
| Width_Multiplier | 0.25 | 网络宽度乘子，控制参数量 |
| Resolution_Multiplier | 0.25 | 图像分辨率乘子 |

## 参考

- [GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training](https://arxiv.org/abs/1805.06725)
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) (LPIPS)
