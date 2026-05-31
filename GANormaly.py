__author__ = 'Eric'

"""
GANormaly基于生成对抗网络的无缺陷样本的缺陷（异常）检测！
"""


#=================================导包=======================================
import numpy as np
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import netron
from collections import OrderedDict
import os
from torchvision.utils import save_image
import addBlock
from torchsummary import summary
from torchinfo import summary as sumy
from torchviz import make_dot
import netron
import torch.onnx
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import gc
from torchvision.datasets import ImageFolder
import lpips

# =================================设置超参数=================================
# 清除显存缓存
gc.collect()
torch.cuda.empty_cache()

# 查看当前环境是否有GPU，有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 定义超参数
learning_rate = 0.002
beta1 = 0.5
beta2 = 0.999
num_epoch = 50

batch_size = 8
latent_size = 100

image_width = 1024
image_height = 1024

input_channel = 3#RGB3通道图还是1通道灰度图

w_adv =1    #对抗损失权重
w_con =10   #语义损失权重
w_enc =1    #编码损失权重
w_anti = 0 #多样性编码损失（防止编码空间坍缩）
w_lpips = 20 #LPIPS感知损失权重

w_D = 1 #判别器损失权重

# 创建保存目录
sample_dir = './samples1'
ckpt_dir = './ckpt'
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

Width_Multiplier = 0.5 #宽度乘子（）
Resolution_Multiplier = 0.25 #分辨率乘子（加载数据用）


#============================================================================
#==================================00加载数据=================================
#============================================================================
image_width = int(image_width*Resolution_Multiplier)#应用分辨率乘子
image_height = int(image_height*Resolution_Multiplier)#应用分辨率乘子

# #先用minist数据集删去一个类别跑一遍先
# # =================================数据预处理=================================
# transform = transforms.Compose([
#     transforms.Resize((image_height, image_height)),  # 统一缩放
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # 可选：归一化到[-1,1]
# ])

# # =================================加载全部数据=================================
# print("正在加载MNIST数据集...")
# full_train_dataset = datasets.MNIST(
#     root='./data',
#     train=True,
#     transform=transform,
#     download=True
# )
#
# # =================================按类别划分=================================
# def split_by_label(dataset, target_label=2):
#     """根据标签划分数据，返回(目标类别数据, 其他类别数据)"""
#     target_indices = []
#     other_indices = []
#
#     for idx, (_, label) in enumerate(dataset):
#         if label == target_label:
#             target_indices.append(idx)      # 类别2的索引
#         else:
#             other_indices.append(idx)       # 非类别2的索引
#
#     target_data = Subset(dataset, target_indices)   # 类别2的数据
#     other_data = Subset(dataset, other_indices)     # 非类别2的数据
#
#     return target_data, other_data
#
# # 划分数据
# print("按类别划分数据...")
# # 修正：other_data 是非2的数据（训练集用）
# # target_data 是2的数据（测试集用）
# test_data_raw, train_data_raw = split_by_label(full_train_dataset, target_label=2)
#
# print(f"训练数据（除2外）: {len(train_data_raw)} 张")
# print(f"测试数据（类别2）: {len(test_data_raw)} 张")
#
# # =================================划分验证集=================================
# # 从训练数据（不含2）中取20%作为验证集
# val_size = int(0.2 * len(train_data_raw))
# train_size = len(train_data_raw) - val_size
#
# train_dataset, val_dataset = random_split(
#     train_data_raw,
#     [train_size, val_size],
#     generator=torch.Generator().manual_seed(42)
# )
#
# print(f"训练集: {len(train_dataset)} 张 (不含2)")
# print(f"验证集: {len(val_dataset)} 张 (不含2)")
# print(f"测试集: {len(test_data_raw)} 张 (全是2)")
#
# # =================================创建DataLoader=================================
# batch_size = 64
#
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True
# )
#
# valid_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )
#
# test_loader = DataLoader(
#     test_data_raw,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )

# # =================================验证数据分布=================================
# def verify_distribution(loader, name, should_contain_2=False):
#     """验证数据分布"""
#     labels = []
#     for _, lbl in loader:
#         labels.extend(lbl.numpy().tolist())
#
#     unique_labels = np.unique(labels)
#     print(f"\n{name}:")
#     print(f"  样本数: {len(labels)}")
#     print(f"  包含的类别: {unique_labels}")
#
#     has_2 = 2 in unique_labels
#     if should_contain_2:
#         print(f"  包含类别2: {has_2} " if has_2 else "  包含类别2: False ")
#     else:
#         print(f"  包含类别2: {has_2} " if has_2 else "  包含类别2: False ")
#
#     return labels
#
# print("\n" + "="*50)
# print("数据分布验证")
# print("="*50)
#
# # 验证各个数据集
# train_labels = verify_distribution(train_loader, "训练集", should_contain_2=False)
# valid_labels = verify_distribution(valid_loader, "验证集", should_contain_2=False)
# test_labels = verify_distribution(test_loader, "测试集", should_contain_2=True)
#
# # 详细标签分布
# print("\n" + "="*50)
# print("详细标签分布")
# print("="*50)
#
# for name, loader in [("训练集", train_loader), ("验证集", valid_loader)]:
#     labels = []
#     for _, lbl in loader:
#         labels.extend(lbl.numpy().tolist())
#
#     unique, counts = np.unique(labels, return_counts=True)
#     print(f"\n{name} (不含2):")
#     for label, count in zip(unique, counts):
#         print(f"  类别 {label}: {count} 张 ({count/len(labels)*100:.1f}%)")
#
# # 测试集详细分布
# labels = []
# for _, lbl in test_loader:
#     labels.extend(lbl.numpy().tolist())
# unique, counts = np.unique(labels, return_counts=True)
# print(f"\n测试集 (全是2):")
# for label, count in zip(unique, counts):
#     print(f"  类别 {label}: {count} 张 ({count/len(labels)*100:.1f}%)")
#
# print("\n数据加载完成！")


#==================用fashionMINIST数据集跑一遍=============================
# =================================加载全部数据=================================
# print("正在加载Fashion-MNIST数据集...")
# full_train_dataset = datasets.FashionMNIST(
#     root='./data',
#     train=True,
#     transform=transform,
#     download=True
# )

# =================================按类别划分=================================
# def split_by_label(dataset, target_label=2,normal_label = 3):
#     """根据标签划分数据，返回(目标类别数据, 其他类别数据)"""
#     target_indices = []
#     other_indices = []
#
#     for idx, (_, label) in enumerate(dataset):
#         if label == target_label:
#             target_indices.append(idx)      # 目标类别的索引
#         elif label == normal_label:
#             other_indices.append(idx)       # 其他类别的索引
#
#     target_data = Subset(dataset, target_indices)   # 目标类别数据
#     other_data = Subset(dataset, other_indices)     # 其他类别数据
#
#     return target_data, other_data
#
# # Fashion-MNIST类别映射（方便查看）
# fashion_classes = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',      # 默认用作异常类
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }
#
# print("\nFashion-MNIST类别:")
# for idx, name in fashion_classes.items():
#     print(f"  {idx}: {name}")
#
# # 划分数据
# print("\n按类别划分数据...")
# # 选择作为异常的类别（可以根据需要修改）
# anomaly_label = 2  # Pullover 作为异常类
# # 也可以尝试其他类别，比如：
# # anomaly_label = 6  # Shirt（和正常类更相似，更难检测）
# # anomaly_label = 9  # Ankle boot（和鞋子类容易混淆）
# nomaly_label = 3
#
# test_data_raw, train_data_raw = split_by_label(full_train_dataset, target_label=anomaly_label,normal_label=nomaly_label)
#
# print(f"\n训练数据（类别 {nomaly_label}:{fashion_classes[nomaly_label]} ）: {len(train_data_raw)} 张")
# print(f"测试数据（类别 {anomaly_label}: {fashion_classes[anomaly_label]}）: {len(test_data_raw)} 张")
#
# # =================================划分验证集=================================
# # 从训练数据（不含异常类）中取20%作为验证集
# val_size = int(0.2 * len(train_data_raw))
# train_size = len(train_data_raw) - val_size
#
# train_dataset, val_dataset = random_split(
#     train_data_raw,
#     [train_size, val_size],
#     generator=torch.Generator().manual_seed(42)
# )
#
# print(f"\n训练集: {len(train_dataset)} 张 ")
# print(f"验证集: {len(val_dataset)} 张 ")
# print(f"测试集: {len(test_data_raw)} 张 ")

#==================用工业检测数据集跑一遍=============================
# =================================加载自定义文件夹数据=================================
def load_custom_dataset(data_root, normal_class_name='normal', anomaly_class_name='anomaly'):
    """
    加载自定义文件夹数据
    文件夹结构：
    data_root/
    ├── train/
    │   └── normal/          # 训练用正常样本
    │       ├── img1.jpg
    │       ├── img2.jpg
    │       └── ...
    └── test/
        ├── normal/          # 测试用正常样本
        │   ├── img1.jpg
        │   └── ...
        └── anomaly/         # 测试用异常样本
            ├── img1.jpg
            └── ...
    """

    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练数据（只有正常样本）
    train_path = os.path.join(data_root, 'train')
    train_dataset = ImageFolder(
        root=train_path,
        transform=transform
    )

    return train_dataset

# 使用
data_root = './data/data_root'
train_dataset = load_custom_dataset(data_root)

# =================================创建DataLoader=================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# valid_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )
#
# test_loader = DataLoader(
#     test_data_raw,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True
# )

# =================================验证数据分布=================================
def check_data_distribution(loader, name):
    """检查数据加载器的类别分布"""
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.numpy())

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} 类别分布:")
    for label, count in zip(unique, counts):
        print(f"  {label} ({fashion_classes[label]}): {count} 张")

# 注意：验证集和训练集都是 Subset，需要重新包装才能查看标签分布
# 这里简单打印一下数据集大小
print(f"\n{'='*50}")
print("数据准备完成！")
print(f"{'='*50}")
print(f"训练集大小: {len(train_dataset)}")


#============================================================================
#==================================01构建判别器=================================
#============================================================================
class D(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+下采样
        self.conv_first = addBlock.DWConv2d(in_channels=input_channel,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
        self.conv_1a = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1b = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_1 = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第二部分卷积+下采样
        self.conv_2a = addBlock.DWConv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2b = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2c = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_2 = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第三部分卷积+下采样
        self.conv_3a = addBlock.DWConv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3b = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3c = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_3 = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第四部分卷积+下采样
        self.conv_4a = addBlock.DWConv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4b = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4c = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_4 = addBlock.DWConv2d(in_channels=64,out_channels=latent_size,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier,endBlock=True)
        #全局平均池化
        self.GAP = nn.AdaptiveAvgPool2d(1)
        #MLP线性运算
        self.MLP1 = nn.Linear(in_features=latent_size,out_features=latent_size)
        self.MLP2 = nn.Linear(in_features=latent_size,out_features=latent_size)
        self.MLP3 = nn.Linear(in_features=latent_size,out_features=latent_size)
        self.MLP4 = nn.Linear(in_features=latent_size,out_features=latent_size)
        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.Linear = nn.Linear(latent_size,1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_first(x)
        x = self.leaky_relu(x)
        x = self.conv_1a(x)
        x = self.leaky_relu(x)
        x = self.conv_1b(x)
        x = self.leaky_relu(x)
        x = self.downSample_1(x)
        x = self.leaky_relu(x)

        x = self.conv_2a(x)
        x = self.leaky_relu(x)
        x = self.conv_2b(x)
        x = self.leaky_relu(x)
        x = self.conv_2c(x)
        x = self.leaky_relu(x)
        x = self.downSample_2(x)
        x = self.leaky_relu(x)

        x = self.conv_3a(x)
        x = self.leaky_relu(x)
        x = self.conv_3b(x)
        x = self.leaky_relu(x)
        x = self.conv_3c(x)
        x = self.leaky_relu(x)
        x = self.downSample_3(x)
        x = self.leaky_relu(x)

        x = self.conv_4a(x)
        x = self.leaky_relu(x)
        x = self.conv_4b(x)
        x = self.leaky_relu(x)
        x = self.conv_4c(x)
        x = self.leaky_relu(x)
        x = self.downSample_4(x)
        x = self.leaky_relu(x)

        #全局池化
        x = self.GAP(x)# [batch, latent_size, 1, 1]

        # 展平
        x = x.view(x.size(0), -1)  # [batch, latent_size]

        # MLP 处理（用于做z与z^的对比）
        x = self.MLP1(x)  # [batch, latent_size]
        x = self.leaky_relu(x)
        x = self.MLP2(x)
        x = self.leaky_relu(x)
        x = self.MLP3(x)
        x = self.leaky_relu(x)
        z = self.MLP4(x)
        x = self.leaky_relu(x)

        x = self.Linear(z)#映射到[batch, 1]供sigmoid输出唯一概率

        result = self.activate(x)#二分类SIGMOID判断真假图片

        return result,z

class D_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = addBlock.DWConv2d(in_channels=input_channel,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv_1a = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1b = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #MLP线性运算
        self.in_channel = int(512*Width_Multiplier)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channel, latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(0.2),
        )
        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.Linear = nn.Linear(latent_size,1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_1a(x)
        x = self.conv_1b(x)

        x = self.GAP(x)# [batch, 512, 1, 1]

        # 展平
        x = x.view(x.size(0), -1)  # [batch, 512]


        # MLP 处理（用于做z与z^的对比）
        z = self.mlp(x)  # [batch, latent_size]

        x = self.Linear(x)#映射到[batch, 1]供sigmoid输出唯一概率

        result = self.activate(x)#二分类SIGMOID判断真假图片

        return result,z

#============================================================================
#==================================02构建生成器=================================
#============================================================================
'''
        :param in_channels:输入通道数
        :param out_channels:输出通道数
        :param kernel_size:卷积核尺寸
        :param stride:步长
        :param padding:边缘填充
        :param bias:偏置
        :param Width_Multiplier:宽度乘子
        :param firstBlock:是否为第一层
'''
#====================================================
#                   1、先构建编码器
#====================================================
class G_E(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+下采样
        self.conv_first = addBlock.DWConv2d(in_channels=input_channel,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
        self.conv_1a = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1b = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_1 = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第二部分卷积+下采样
        self.conv_2a = addBlock.DWConv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2b = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2c = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_2 = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第三部分卷积+下采样
        self.conv_3a = addBlock.DWConv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3b = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3c = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_3 = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第四部分卷积+下采样
        self.conv_4a = addBlock.DWConv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4b = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4c = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_4 = addBlock.DWConv2d(in_channels=64,out_channels=latent_size,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier,endBlock=True)
        #全局平均池化
        self.GAP = nn.AdaptiveAvgPool2d(1)
        #MLP线性运算
        self.MLP = nn.Linear(in_features=latent_size,out_features=latent_size)
        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_fc = nn.Dropout(0.3)
        self.dropout_feat = nn.Dropout2d(0.3)  # 2D Dropout适合卷积特征

    def forward(self, x):
        x = self.conv_first(x)
        x = self.leaky_relu(x)
        x = self.conv_1a(x)
        x = self.leaky_relu(x)
        x = self.conv_1b(x)
        x = self.leaky_relu(x)
        x = self.downSample_1(x)

        x = self.conv_2a(x)
        x = self.leaky_relu(x)
        x = self.conv_2b(x)
        x = self.leaky_relu(x)
        x = self.conv_2c(x)
        x = self.leaky_relu(x)
        x = self.downSample_2(x)

        x = self.conv_3a(x)
        x = self.leaky_relu(x)
        x = self.conv_3b(x)
        x = self.leaky_relu(x)
        x = self.conv_3c(x)
        x = self.leaky_relu(x)
        x = self.downSample_3(x)

        x = self.conv_4a(x)
        x = self.leaky_relu(x)
        x = self.conv_4b(x)
        x = self.leaky_relu(x)
        x = self.conv_4c(x)
        x = self.leaky_relu(x)
        x = self.downSample_4(x)

        #用于反卷积恢复
        map = x# [batch, latent_size, H, W]

        # 在特征图上应用Dropout
        # 在训练时使用dropout
        if self.training:
            x = self.dropout_feat(x)

        #全局池化
        x = self.GAP(x)# [batch, latent_size, 1, 1]

        # 展平
        x = x.view(x.size(0), -1)  # [batch, latent_size]

        # 全连接前应用Dropout
         # 在训练时使用dropout
        # if self.training:
        #     x = self.dropout_fc(x)

        # MLP 处理（用于做z与z^的对比）
        z = self.MLP(x)  # [batch, latent_size]

        return z,map


# G_E = nn.Sequential(
#     #================第一部分卷积==============
#     addBlock.DWConv2d(in_channels=3,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     #================第一部分下采样==============
#     addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#下采样
#     nn.LeakyReLU(0.2),
#     #================第二部分卷积==============
#     addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     #================第二部分下采样==============
#     addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#下采样
#     nn.LeakyReLU(0.2),
#     #================第三部分卷积==============
#     addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     #================第三部分下采样==============
#     addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#下采样
#     nn.LeakyReLU(0.2),
#     #================第四部分卷积==============
#     addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#卷积
#     nn.LeakyReLU(0.2),
#     #================第四部分下采样=============
#     addBlock.DWConv2d(in_channels=64,out_channels=latent_size,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier),#下采样
#     #================转换为隐空间向量=============
#     nn.AdaptiveAvgPool2d(1),#全局平均池化，torch.Size([batchsize,outchannels,1,1]),可以使用y = y.view(8, -1)去掉最后两个维度
#     #================MLP处理下=============
#     nn.Linear(in_features=latent_size,out_features=latent_size),
#     nn.LeakyReLU(0.2)
# )

#====================================================
#                   2、再构建解码器
#====================================================
class G_D(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+上采样
        self.up1 = addBlock.DWConvTranspose2d(
            in_channels=latent_size,      # 输入通道数
            out_channels=64,     # 输出通道数
            kernel_size=3,      # 卷积核大小
            stride=2,         # 步长（控制上采样倍数）
            padding=1,        # 填充
            output_padding=1, # 输出填充（处理奇数尺寸）
            bias=True,         # 是否使用偏置
            firstBlock=True,
            Width_Multiplier=Width_Multiplier
        )
        self.conv_1a = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1b = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1c = addBlock.DWConv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第二部分卷积+上采样
        self.up2 = addBlock.DWConvTranspose2d(
            in_channels=128,      # 输入通道数
            out_channels=128,     # 输出通道数
            kernel_size=3,      # 卷积核大小
            stride=2,         # 步长（控制上采样倍数）
            padding=1,        # 填充
            output_padding=1, # 输出填充（处理奇数尺寸）
            bias=True ,        # 是否使用偏置
            Width_Multiplier=Width_Multiplier
        )
        self.conv_2a = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2b = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2c = addBlock.DWConv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第三部分卷积+上采样
        self.up3 = addBlock.DWConvTranspose2d(
            in_channels=256,      # 输入通道数
            out_channels=256,     # 输出通道数
            kernel_size=3,      # 卷积核大小
            stride=2,         # 步长（控制上采样倍数）
            padding=1,        # 填充
            output_padding=1, # 输出填充（处理奇数尺寸）
            bias=True ,        # 是否使用偏置
            Width_Multiplier=Width_Multiplier
        )
        self.conv_3a = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3b = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3c = addBlock.DWConv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第四部分卷积+上采样
        self.up4 = addBlock.DWConvTranspose2d(
            in_channels=512,      # 输入通道数
            out_channels=512,     # 输出通道数
            kernel_size=3,      # 卷积核大小
            stride=2,         # 步长（控制上采样倍数）
            padding=1,        # 填充
            output_padding=1, # 输出填充（处理奇数尺寸）
            bias=True,         # 是否使用偏置
            Width_Multiplier=Width_Multiplier
        )
        self.conv_4b = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4a = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_end = addBlock.DWConv2d(in_channels=512,out_channels=input_channel,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,endBlock=True)


        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.dropout_feat = nn.Dropout2d(0.3)  # 2D Dropout适合卷积特征

    def forward(self,x):
        x = self.up1(x)
        x = self.leaky_relu(x)
        x = self.conv_1a(x)
        x = self.leaky_relu(x)
        x = self.conv_1b(x)
        x = self.leaky_relu(x)
        x = self.conv_1c(x)
        x = self.leaky_relu(x)


        x = self.up2(x)
        x = self.leaky_relu(x)
        x = self.conv_2a(x)
        x = self.leaky_relu(x)
        x = self.conv_2b(x)
        x = self.leaky_relu(x)
        x = self.conv_2c(x)
        x = self.leaky_relu(x)


        x = self.up3(x)
        x = self.leaky_relu(x)
        x = self.conv_3a(x)
        x = self.leaky_relu(x)
        x = self.conv_3b(x)
        x = self.leaky_relu(x)
        x = self.conv_3c(x)
        x = self.leaky_relu(x)

        # # 在特征图上应用Dropout
        # # 在训练时使用dropout
        # if self.training:
        #     x = self.dropout_feat(x)

        x = self.up4(x)
        x = self.leaky_relu(x)
        x = self.conv_4a(x)
        x = self.leaky_relu(x)
        x = self.conv_4b(x)
        x = self.leaky_relu(x)
        x = self.conv_end(x)
        x = self.tanh(x)


        return x

#=========================测试生成器效果==================================

# class Generator(nn.Module):
#     """完整的生成器：编码器 + 解码器"""
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, x):
#         # 编码
#         z, feature_map = self.encoder(x)
#         # 解码
#         reconstructed = self.decoder(feature_map)
#         return reconstructed
#
#     def encode(self, x):
#         """仅编码"""
#         return self.encoder(x)
#
#     def decode(self, z, feature_map=None):
#         """仅解码"""
#         return self.decoder(z, feature_map)
#
# G_Encoder = G_E().to(device)
# G_Decoder = G_D().to(device)
# #
# # # 使用
# full_generator = Generator(G_Encoder, G_Decoder).to(device)
# #
# # # torchinfo 能更好地处理多输出
# sumy(full_generator, input_size=(1, 3, image_height, image_width),
#         col_names=["input_size", "output_size", "num_params"],
#         device="cuda")
# # # summary(G_Decoder, input_size=(100, 15, 20), device="cuda")#如果有多个返回值，它没法处理



#============================================================================
#==================================03构建额外编码器=================================
#============================================================================
class E(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+下采样
        self.conv_first = addBlock.DWConv2d(in_channels=input_channel,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
        self.conv_1a = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_1b = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_1 = addBlock.DWConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第二部分卷积+下采样
        self.conv_2a = addBlock.DWConv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2b = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_2c = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_2 = addBlock.DWConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第三部分卷积+下采样
        self.conv_3a = addBlock.DWConv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3b = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_3c = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_3 = addBlock.DWConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        #第四部分卷积+下采样
        self.conv_4a = addBlock.DWConv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4b = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.conv_4c = addBlock.DWConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier)
        self.downSample_4 = addBlock.DWConv2d(in_channels=64,out_channels=latent_size,kernel_size=3,stride=2,padding=1,bias=True,Width_Multiplier=Width_Multiplier,endBlock=True)
        #全局平均池化
        self.GAP = nn.AdaptiveAvgPool2d(1)
        #MLP线性运算
        self.MLP = nn.Linear(in_features=latent_size,out_features=latent_size)
        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_fc = nn.Dropout(0.3)
        self.dropout_feat = nn.Dropout2d(0.3)  # 2D Dropout适合卷积特征

    def forward(self, x):
        x = self.conv_first(x)
        x = self.leaky_relu(x)
        x = self.conv_1a(x)
        x = self.leaky_relu(x)
        x = self.conv_1b(x)
        x = self.leaky_relu(x)
        x = self.downSample_1(x)

        x = self.conv_2a(x)
        x = self.leaky_relu(x)
        x = self.conv_2b(x)
        x = self.leaky_relu(x)
        x = self.conv_2c(x)
        x = self.leaky_relu(x)
        x = self.downSample_2(x)

        x = self.conv_3a(x)
        x = self.leaky_relu(x)
        x = self.conv_3b(x)
        x = self.leaky_relu(x)
        x = self.conv_3c(x)
        x = self.leaky_relu(x)
        x = self.downSample_3(x)

        x = self.conv_4a(x)
        x = self.leaky_relu(x)
        x = self.conv_4b(x)
        x = self.leaky_relu(x)
        x = self.conv_4c(x)
        x = self.leaky_relu(x)
        x = self.downSample_4(x)

        # 在特征图上应用Dropout
        # 在训练时使用dropout
        if self.training:
            x = self.dropout_feat(x)

        #全局池化
        x = self.GAP(x)# [batch, latent_size, 1, 1]

        # 展平
        x= x.view(x.size(0), -1)  # [batch, latent_size]

        # 全连接前应用Dropout
         # 在训练时使用dropout
        # if self.training:
        #     x = self.dropout_fc(x)

        # MLP 处理（用于做z与z^的对比）
        z = self.MLP(x)  # [batch, latent_size]

        return z

#============================================================================
#==================================04定义损失函数，优化器等=================================
#============================================================================
#通常在深度学习中，图像数据可能经过了标准化处理，取值范围从[0,1]或者[-1,1]
#denorm函数的作用就是将输入数据范围从[-1,1]转回到[0,1]，即逆标准化的过程
#clamp(0,1)是限制（clamp）操作，确保返回的数据不会超出0到1的范围
#把判别器和生成器等迁移到GPU上
# D = D().to(device)
D_MLP = D_MLP().to(device)
G_Encoder = G_E().to(device)
G_Decoder = G_D().to(device)
add_Encoder = E().to(device)

#初始化LPIPS感知损失模型
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()  # LPIPS网络本身不需要训练

#定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
D_MLP_optimizer= torch.optim.Adam(D_MLP.parameters(),lr=learning_rate,betas=(beta1, beta2),weight_decay=1e-5 )
# D_optimizer = torch.optim.Adam(D.parameters(),lr=learning_rate,betas=(beta1, beta2),weight_decay=1e-5 )
# G_E_optimizer = torch.optim.Adam(G_Encoder.parameters(),lr=learning_rate,betas=(beta1, beta2),weight_decay=1e-5 )
# G_D_optimizer = torch.optim.Adam(G_Decoder.parameters(),lr=learning_rate,betas=(beta1, beta2),weight_decay=1e-5 )
# add_E_optimizer = torch.optim.Adam(add_Encoder.parameters(),lr=learning_rate,betas=(beta1, beta2),weight_decay=1e-5 )
G_E_optimizer = torch.optim.Adam(G_Encoder.parameters(),lr=learning_rate,betas=(beta1, beta2) )
G_D_optimizer = torch.optim.Adam(G_Decoder.parameters(),lr=learning_rate,betas=(beta1, beta2) )
add_E_optimizer = torch.optim.Adam(add_Encoder.parameters(),lr=learning_rate,betas=(beta1, beta2) )

#Clamp函数x限制在区间[min,max]内
def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def reset_grad():
    D_MLP_optimizer.zero_grad()
    # D_optimizer.zero_grad()
    G_E_optimizer.zero_grad()
    G_D_optimizer.zero_grad()
    add_E_optimizer.zero_grad()

#开始训练
total_step = len(train_loader)

#===============添加编码多样性损失==================
class AntiCollapseLoss(nn.Module):
    def __init__(self, target_std=1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, z):
        batch_size = z.size(0)

        # 1. 标准化编码
        z_norm = F.normalize(z, dim=1)

        # 2. 计算编码之间的相似度
        similarity = torch.mm(z_norm, z_norm.t())

        # 3. 惩罚过高的相似度（鼓励编码分散）
        # 对角线是自己，排除
        off_diag = similarity * (1 - torch.eye(batch_size, device=z.device))
        diversity_loss = off_diag.mean()  # 希望相似度低

        # 4. 惩罚编码标准差太小
        z_std = z.std(dim=0).mean()
        std_loss = torch.relu(self.target_std - z_std)

        return diversity_loss + std_loss
#============================================================================
#==================================05训练模型=================================
#============================================================================
# print("="*50)
# print("模型诊断")
# print("="*50)
#
# # 1. 检查模型类型
# print(f"D 类型: {type(D)}")
# print(f"G_Encoder 类型: {type(G_Encoder)}")
# print(f"G_Decoder 类型: {type(G_Decoder)}")
# print(f"add_Encoder 类型: {type(add_Encoder)}")
#
# # 2. 检查是否为 nn.Module 实例
# print(f"D 是 nn.Module: {isinstance(D, nn.Module)}")
# print(f"G_Encoder 是 nn.Module: {isinstance(G_Encoder, nn.Module)}")
#
# # 3. 检查参数
# def count_params(model, name):
#     if isinstance(model, nn.Module):
#         total = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"{name}: {total:,} 参数")
#     else:
#         print(f"{name}: 不是 nn.Module，类型为 {type(model)}")
#
# count_params(D, "判别器")
# count_params(G_Encoder, "编码器")
# count_params(G_Decoder, "解码器")
# count_params(add_Encoder, "额外编码器")
#
# # 4. 检查是否有梯度（简单前向测试）
# print("\n测试前向传播...")
# test_input = torch.randn(2, 1, 32, 32).to(device)
#
# try:
#     out, feat = D(test_input)
#     print(f"判别器前向成功: 输出形状 {out.shape}")
# except Exception as e:
#     print(f"判别器前向失败: {e}")
#
# try:
#     z, fm = G_Encoder(test_input)
#     print(f"编码器前向成功: z形状 {z.shape}, fm形状 {fm.shape}")
# except Exception as e:
#     print(f"编码器前向失败: {e}")
#============================防止出现模式崩溃策略========================
# 添加标签平滑
def get_labels(batch_size, smooth=True):
    if smooth:
        real_labels = torch.full((batch_size, 1), 0.9, device=device)  # 标签平滑 1->0.9
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # 标签平滑 0->0.1
    else:
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
    return real_labels, fake_labels


# 在每个 epoch 结束时记录平均损失
D_loss_epoch = []
G_loss_epoch = []
#=================================================================
for epoch in range(num_epoch):
    epoch_d_loss = 0
    epoch_g_loss = 0
    for i,(images,_) in enumerate(train_loader):
        images = images.to(device)
        #定义图像是真或假的标签
        batch = images.size(0)  # 实际批次大小
        # 获取平滑标签
        real_labels, fake_labels = get_labels(batch, smooth=False)
        #====================================================
        #                   训练判别器
        #====================================================
        #1、定义判别器对真图片的损失函数
        #outputs,_ = D(images)
        outputs,_ = D_MLP(images)
        d_loss_real = criterion(outputs,real_labels)
        real_score = outputs
        #2、定义判别器对假图片（即由潜在空间点生成的图片）的损失函数
        with torch.no_grad():  # 训练D时G不更新
            _,fake_maps = G_Encoder(images)
            fake_images = G_Decoder(fake_maps)
        outputs,_ = D_MLP(fake_images)
        d_loss_fake = criterion(outputs,fake_labels)
        fake_score = outputs

        #得到判别器总的损失函数
        d_loss = (d_loss_real+d_loss_fake)*w_D
        epoch_d_loss += d_loss.item()

        #对生成器、判别器的梯度清零
        reset_grad()#梯度清零
        d_loss.backward()#反向传播

        #梯度消失了!
        #修改方案：1、加残差连接（有效，已修改）2、破案了，D_MLP的多层感知机部分没加RELU激活
        #不激活（或者说不使用合适的激活函数）会导致网络退化为线性模型，而线性模型在多层叠加后，其“表达能力”有限，并且在反向传播中会产生梯度消失问题。
        # 在 d_loss.backward() 之后添加
        # print("=== 梯度检查 ===")
        # for name, param in D_MLP.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}: grad_norm = {grad_norm:.6f}")
        #     else:
        #         print(f"{name}: 梯度为 None！")

        D_MLP_optimizer.step()#参数更新


        #====================================================
        #                   训练生成器
        #====================================================
        #===================1、获得对抗损失（原图和假图在D上编码向量的L2距离）==========================
        #1）、获得原图的编码向量
        _,fz = D_MLP(images)
        #1）、获得假图的编码向量
        z,fake_maps = G_Encoder(images)
        fake_images = G_Decoder(fake_maps)
        with torch.no_grad():  # 训练G时D不更新
            outputs,fz_g = D_MLP(fake_images)
        # L_adv = criterion(outputs,real_labels)#1、或者使用标签损失
        L_adv = torch.mean(torch.norm(fz - fz_g, p=2))#2、计算欧式距离
        #===================2、获得语义损失（原图和假图的L1距离）==========================
        #计算两张假图之间的L1距离
        L_con = torch.mean(torch.abs(images - fake_images))
        #===================2.1、获得LPIPS感知损失（基于深度特征的感知距离）==========================
        L_lpips = lpips_model(images, fake_images).mean()
        #===================3、获得编码损失（原图在G中编码和假图在E中编码向量的L2距离）==========================
        z_e = add_Encoder(fake_images)
        L_enc = torch.norm(z - z_e, p=2)#计算欧式距离


        #计算总损失
        #在总损失中加入
        anti_collapse = AntiCollapseLoss(target_std=1.0)#编码多样性损失
        L_anti = anti_collapse(z)
        g_loss = w_adv*L_adv+w_con*L_con+w_lpips*L_lpips+w_enc*L_enc+ w_anti*L_anti
        epoch_g_loss += g_loss.item()
        #对生成器、判别器的梯度清零
        reset_grad()#梯度清零
        g_loss.backward()#反向传播
        #参数更新
        # print("=== 梯度检查 ===")
        # for name, param in G_Decoder.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}: grad_norm = {grad_norm:.6f}")
        #     else:
        #         print(f"{name}: 梯度为 None！")
        # for name, param in G_Encoder.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}: grad_norm = {grad_norm:.6f}")
        #     else:
        #         print(f"{name}: 梯度为 None！")
        # for name, param in add_Encoder.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"{name}: grad_norm = {grad_norm:.6f}")
        #     else:
        #         print(f"{name}: 梯度为 None！")
        G_E_optimizer.step()
        G_D_optimizer.step()
        add_E_optimizer.step()

        #打印训练信息
        if (i+1)%2 == 0:
            print('Epoch[{}/{}],step[{}/{}],d_loss:{:.4f},g_loss:{:.4f},D(x):{:.2f},D(G(z)):{:.2f},L_adv:{:2f},L_con:{:2f},L_lpips:{:2f},L_enc:{:2f},L_anti:{:2f}'
                  .format(epoch+1,num_epoch,i+1,total_step,d_loss.item(),g_loss.item(),
                          real_score.mean().item(),fake_score.mean().item(),
                          L_adv.item(),L_con.item(),L_lpips.item(),L_enc.item(),L_anti.item()))

    # #保存图片
    # if (epoch+1)%10 == 0:
    #     save_image(denorm(fake_images),os.path.join(sample_dir,'fake_image-{}.png'.format(epoch+1)))
    #     save_image(denorm(images),os.path.join(sample_dir,'real_image-{}.png'.format(epoch+1)))

    save_image(denorm(fake_images),os.path.join(sample_dir,'fake_image-{}.png'.format(epoch+1)))
    save_image(denorm(images),os.path.join(sample_dir,'real_image-{}.png'.format(epoch+1)))

    if (epoch+1)%2==0:
        torch.save(D_MLP.state_dict(),'./ckpt/D_MLP_{}.ckpt'.format(epoch+1))
        torch.save(G_Encoder.state_dict(),'./ckpt/G_Encoder_{}.ckpt'.format(epoch+1))
        torch.save(G_Decoder.state_dict(),'./ckpt/G_Decoder_{}.ckpt'.format(epoch+1))
        torch.save(add_Encoder.state_dict(),'./ckpt/add_Encoder_{}.ckpt'.format(epoch+1))

    # 记录每个 epoch 的平均损失
    D_loss_epoch.append(epoch_d_loss / len(train_loader))
    G_loss_epoch.append(epoch_g_loss / len(train_loader))

#============================================================================
#==================================06保存模型=================================
#============================================================================
torch.save(D_MLP.state_dict(),'D_MLP.ckpt')
torch.save(G_Encoder.state_dict(),'G_Encoder.ckpt')
torch.save(G_Decoder.state_dict(),'G_Decoder.ckpt')
torch.save(add_Encoder.state_dict(),'add_Encoder.ckpt')


# 绘制曲线（x轴为epoch）
plt.plot(range(1, num_epoch+1), D_loss_epoch, label='D_loss')
plt.plot(range(1, num_epoch+1), G_loss_epoch, label='G_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()





