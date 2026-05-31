"""
test_anomaly.py - 测试异常检测：对比正常/异常样本的编码距离分布
比较 G_E(原图) 和 E(重建图) 的隐向量距离
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import gc
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset  # 添加 ConcatDataset


# =================================设置参数=================================
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数（必须与训练时一致）
latent_size = 100
input_channel = 3
image_width = 1024
image_height = 1024
Width_Multiplier = 0.5 #宽度乘子（）
Resolution_Multiplier = 0.125 #分辨率乘子（加载数据用）
batch_size = 16
# =================================导入网络结构=================================
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

        x = self.Linear(z)#映射到[batch, 1]供sigmoid输出唯一概率

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
        self.dropout = nn.Dropout(0.3)
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
        if self.training:
            x = self.dropout_fc(x)

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

        # 在特征图上应用Dropout
        # 在训练时使用dropout
        if self.training:
            x = self.dropout_feat(x)

        x = self.up2(x)
        x = self.leaky_relu(x)
        x = self.conv_2a(x)
        x = self.leaky_relu(x)
        x = self.conv_2b(x)
        x = self.leaky_relu(x)
        x = self.conv_2c(x)
        x = self.leaky_relu(x)

        # 在特征图上应用Dropout
        # 在训练时使用dropout
        if self.training:
            x = self.dropout_feat(x)

        x = self.up3(x)
        x = self.leaky_relu(x)
        x = self.conv_3a(x)
        x = self.leaky_relu(x)
        x = self.conv_3b(x)
        x = self.leaky_relu(x)
        x = self.conv_3c(x)
        x = self.leaky_relu(x)

        # 在特征图上应用Dropout
        # 在训练时使用dropout
        if self.training:
            x = self.dropout_feat(x)

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
        self.dropout = nn.Dropout(0.3)
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
        x = x.view(x.size(0), -1)  # [batch, latent_size]

        # 全连接前应用Dropout
         # 在训练时使用dropout
        if self.training:
            x = self.dropout_fc(x)

        # MLP 处理（用于做z与z^的对比）
        z = self.MLP(x)  # [batch, latent_size]

        return z

# # =================================加载数据(MINIST)=================================
# transform = transforms.Compose([
#     transforms.Resize((image_height*Resolution_Multiplier, image_width*Resolution_Multiplier)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# print("加载MNIST测试集...")
# full_dataset = datasets.MNIST(
#     root='./data',
#     train=False,
#     transform=transform,
#     download=True
# )
#
# def split_by_label(dataset, abnormal_label=2):
#     """划分正常和异常样本"""
#     normal_idx, abnormal_idx = [], []
#     for idx, (_, label) in enumerate(dataset):
#         if label != abnormal_label:
#             normal_idx.append(idx)      # 正常样本（不是2的类别）
#         else:
#             abnormal_idx.append(idx)    # 异常样本（类别2）
#     return Subset(dataset, normal_idx), Subset(dataset, abnormal_idx)
#
# normal_data, abnormal_data = split_by_label(full_dataset, abnormal_label=2)
# print(f"正常样本（其他类别 0,1,3,4,5,6,7,8,9）: {len(normal_data)} 张")
# print(f"异常样本（类别2）: {len(abnormal_data)} 张")
#
# batch_size = 64
# normal_loader = DataLoader(normal_data, batch_size=batch_size, shuffle=False)
# abnormal_loader = DataLoader(abnormal_data, batch_size=batch_size, shuffle=False)

# =================================加载数据(FASHIOMINIST)=================================
# transform = transforms.Compose([
#     transforms.Resize((image_height*Resolution_Multiplier, image_width*Resolution_Multiplier)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# print("加载Fashion-MNIST测试集...")
# full_dataset = datasets.FashionMNIST(
#     root='./data',
#     train=False,  # 使用测试集
#     transform=transform,
#     download=True
# )
#
# # Fashion-MNIST类别映射
# fashion_classes = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }
#
# def split_by_label(dataset, abnormal_label=2,normal_label=3):
#     """划分正常和异常样本"""
#     normal_idx, abnormal_idx = [], []
#     for idx, (_, label) in enumerate(dataset):
#         if label == abnormal_label:
#             abnormal_idx.append(idx)      # 正常样本（不是异常类别的）
#         elif label == normal_label:
#             normal_idx.append(idx)    # 异常样本（指定类别）
#     return Subset(dataset, normal_idx), Subset(dataset, abnormal_idx)
#
# # 选择作为异常的类别（可以根据需要修改）
# abnormal_label = 2  # Pullover 作为异常
# # abnormal_label = 6  # 或者用 Shirt
# # abnormal_label = 9  # 或者用 Ankle boot
# normal_label=3
# normal_data, abnormal_data = split_by_label(full_dataset, abnormal_label=abnormal_label,normal_label=normal_label)
#
# # 打印正常类别列表
# normal_classes = [name for idx, name in fashion_classes.items() if idx != abnormal_label]
# print(f"正常样本: {len(normal_data)} 张")
# print(f"正常类别: {fashion_classes[normal_label]}")
# print(f"异常样本（类别 {abnormal_label}: {fashion_classes[abnormal_label]}）: {len(abnormal_data)} 张")
#
# batch_size = 64
# normal_loader = DataLoader(normal_data, batch_size=batch_size, shuffle=False)
# abnormal_loader = DataLoader(abnormal_data, batch_size=batch_size, shuffle=False)
#
transform = transforms.Compose([
    transforms.Resize((int(image_height * Resolution_Multiplier),
                       int(image_width * Resolution_Multiplier))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ================== 自定义 Dataset（用于直接存放图片的文件夹）==================
class FlatImageDataset(Dataset):
    """用于文件夹内直接放图片的情况（没有子文件夹）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        print(f"从 {root_dir} 加载了 {len(self.image_paths)} 张图片（平铺结构）")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 返回图片和虚拟标签


# ================== 智能加载函数 ==================
def load_dataset(root_path, transform):
    """
    自动判断文件夹结构并加载数据
    - 如果有子文件夹，使用 ImageFolder（保留类别信息）
    - 如果直接放图片，使用 FlatImageDataset
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"路径不存在: {root_path}")

    # 检查是否有子文件夹
    items = os.listdir(root_path)
    has_subdirs = any(os.path.isdir(os.path.join(root_path, item)) for item in items)

    if has_subdirs:
        # 有子文件夹：使用 ImageFolder
        dataset = ImageFolder(root=root_path, transform=transform)
        print(f"从 {root_path} 加载了 {len(dataset)} 张图片（子文件夹结构）")
        print(f"  类别: {dataset.classes}")
    else:
        # 没有子文件夹：使用 FlatImageDataset
        dataset = FlatImageDataset(root_path, transform=transform)

    return dataset


# ================== 加载所有数据 ==================
# 训练数据
train_path = './data/data_root/train'
train_dataset = load_dataset(train_path, transform)

# 测试正常数据
test_normal_path = './data/data_root/test/normal'
test_normal_dataset = load_dataset(test_normal_path, transform)

# 测试异常数据（可能有多个子文件夹，需要合并）
test_anomaly_path = './data/data_root/test/anormaly'

if os.path.exists(test_anomaly_path):
    # 检查异常文件夹下是否有子文件夹
    items = os.listdir(test_anomaly_path)
    subdirs = [os.path.join(test_anomaly_path, item) for item in items
               if os.path.isdir(os.path.join(test_anomaly_path, item))]

    if len(subdirs) > 0:
        # 有多个子文件夹：分别加载每个子文件夹，然后合并
        anomaly_datasets = []
        for subdir in subdirs:
            ds = load_dataset(subdir, transform)
            if len(ds) > 0:
                anomaly_datasets.append(ds)
        if len(anomaly_datasets) > 0:
            test_anomaly_dataset = ConcatDataset(anomaly_datasets)
            print(f"合并后异常样本总数: {len(test_anomaly_dataset)} 张")
        else:
            test_anomaly_dataset = FlatImageDataset(test_anomaly_path, transform)
    else:
        # 没有子文件夹，直接加载
        test_anomaly_dataset = load_dataset(test_anomaly_path, transform)
else:
    print(f"警告: 异常路径不存在 {test_anomaly_path}")
    test_anomaly_dataset = FlatImageDataset(test_anomaly_path, transform)


# ================== 创建 DataLoader ==================
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

normal_loader = DataLoader(
    test_normal_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

abnormal_loader = DataLoader(
    test_anomaly_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ================== 打印信息 ==================
print("\n" + "="*50)
print("数据加载完成:")
print(f"训练数据: {len(train_dataset)} 张")
print(f"测试正常样本: {len(test_normal_dataset)} 张")
print(f"测试异常样本: {len(test_anomaly_dataset)} 张")
print("="*50)



# # =================================加载模型=================================
def load_model(model_class, weight_path):
    model = model_class().to(device)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        print(f"加载 {weight_path}")
        return model
    except FileNotFoundError:
        print(f"未找到 {weight_path}")
        return None

# 加载生成器编码器、解码器、额外编码器
G_Encoder = load_model(G_E, './ckpt/G_Encoder_20.ckpt')
G_Decoder = load_model(G_D, './ckpt/G_Decoder_20.ckpt')
add_Encoder = load_model(E, './ckpt/add_Encoder_20.ckpt')
G_Encoder.eval()    # 固定BatchNorm，关闭Dropout
G_Decoder.eval()    # 固定BatchNorm，关闭Dropout
add_Encoder.eval()  # 固定BatchNorm，关闭Dropout

if None in [G_Encoder, G_Decoder, add_Encoder]:
    print("请确保训练完成并生成了所有 .ckpt 文件")
    exit()


#Clamp函数x限制在区间[min,max]内
def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


# =================================计算编码距离 (L_enc)=================================
@torch.no_grad()
def compute_enc_distances(loader, G_Encoder, G_Decoder, add_Encoder):
    """
    计算 L_enc = ||z - z_e||_2
    z = G_E(原图)
    z_e = E(重建图)
    """
    distances = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        
        # 1. 生成器编码器提取原图的隐向量 z
        z, feature_map = G_Encoder(images)  # z: [batch, latent_size]
        
        # 2. 解码器重建图像
        fake_images = G_Decoder(feature_map)
        
        # 3. 额外编码器提取重建图的隐向量 z_e
        z_e = add_Encoder(fake_images)  # z_e: [batch, latent_size]

        # 3.1. 放到原来编码器提取重建图的隐向量 z_e_1
        # z_e_1,_ = G_Encoder(fake_images)
        
        # 4. 计算 L2 距离
        # distance_1 = torch.norm(z - z_e, p=2, dim=1)  # [batch]
        # 4.1. 计算 L2 距离
        # distance_2 = torch.norm(z - z_e_1, p=2, dim=1)  # [batch]

        distance = torch.norm(z-z_e,p=2,dim=1)

        # distance = distance_1+distance_2
        
        distances.extend(distance.cpu().numpy())
        all_labels.extend(labels.numpy())
    #生成重建后的图像，看下是编码器过拟合还是解码器过拟合
    #问题：编码空间坍缩了，不同图像编码根本没有差别，解决方案：添加多样性损失
    save_image(denorm(fake_images),os.path.join('./','fake_image.png'))
    save_image(denorm(images),os.path.join('./','real_image.png'))
    return np.array(distances), np.array(all_labels)

print("\n计算编码距离 L_enc = ||G_E(原图) - E(重建图)||...")
normal_distances, normal_labels = compute_enc_distances(normal_loader, G_Encoder, G_Decoder, add_Encoder)
abnormal_distances, abnormal_labels = compute_enc_distances(abnormal_loader, G_Encoder, G_Decoder, add_Encoder)

print(f"正常样本距离 - 均值: {np.mean(normal_distances):.4f}, 标准差: {np.std(normal_distances):.4f}")
print(f"异常样本距离 - 均值: {np.mean(abnormal_distances):.4f}, 标准差: {np.std(abnormal_distances):.4f}")

# =================================绘制直方图=================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 直方图
ax1 = axes[0]
max_dist = max(normal_distances.max(), abnormal_distances.max())
bins = np.linspace(0, max_dist, 50)
ax1.hist(normal_distances, bins=bins, alpha=0.7, label='正常样本 (类别≠2)', 
         color='blue', density=True, edgecolor='black')
ax1.hist(abnormal_distances, bins=bins, alpha=0.7, label='异常样本 (类别=2)', 
         color='red', density=True, edgecolor='black')
ax1.set_xlabel('编码距离 L_enc = ||G_E(x) - E(G_D(G_E(x)))||')
ax1.set_ylabel('密度')
ax1.set_title('正常 vs 异常样本编码距离分布')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 箱线图
ax2 = axes[1]
bp = ax2.boxplot([normal_distances, abnormal_distances], 
                  labels=['正常样本\n(类别≠2)', '异常样本\n(类别=2)'], 
                  patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('编码距离 L_enc')
ax2.set_title('编码距离箱线图对比')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('encoding_distance_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

# =================================打印统计信息=================================
print("\n" + "="*50)
print("统计信息")
print("="*50)

print(f"\n正常样本 (类别≠2):")
print(f"  样本数: {len(normal_distances)}")
print(f"  编码距离均值: {np.mean(normal_distances):.4f}")
print(f"  编码距离标准差: {np.std(normal_distances):.4f}")
print(f"  编码距离中位数: {np.median(normal_distances):.4f}")
print(f"  最小值: {np.min(normal_distances):.4f}")
print(f"  最大值: {np.max(normal_distances):.4f}")
print(f"  95%分位数: {np.percentile(normal_distances, 95):.4f}")
print(f"  99%分位数: {np.percentile(normal_distances, 99):.4f}")

print(f"\n异常样本 (类别=2):")
print(f"  样本数: {len(abnormal_distances)}")
print(f"  编码距离均值: {np.mean(abnormal_distances):.4f}")
print(f"  编码距离标准差: {np.std(abnormal_distances):.4f}")
print(f"  编码距离中位数: {np.median(abnormal_distances):.4f}")
print(f"  最小值: {np.min(abnormal_distances):.4f}")
print(f"  最大值: {np.max(abnormal_distances):.4f}")
print(f"  5%分位数: {np.percentile(abnormal_distances, 5):.4f}")

# 分离度
separation = abs(np.mean(normal_distances) - np.mean(abnormal_distances))
print(f"\n分离度 (均值差): {separation:.4f}")

# 异常检测性能
threshold_95 = np.percentile(normal_distances, 95)
threshold_99 = np.percentile(normal_distances, 99)

detection_rate_95 = (abnormal_distances > threshold_95).mean()
detection_rate_99 = (abnormal_distances > threshold_99).mean()

print(f"\n使用95%分位数作为阈值 ({threshold_95:.4f}):")
print(f"  异常检测率 (召回率): {detection_rate_95*100:.2f}%")
print(f"  假阳性率: 5% (设定)")

print(f"\n使用99%分位数作为阈值 ({threshold_99:.4f}):")
print(f"  异常检测率 (召回率): {detection_rate_99*100:.2f}%")
print(f"  假阳性率: 1% (设定)")


# =================================计算AUC和绘制ROC曲线=================================
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve

print("\n" + "="*50)
print("AUC (ROC曲线) 分析")
print("="*50)

# 合并所有距离和标签
all_distances = np.concatenate([normal_distances, abnormal_distances])
all_labels = np.concatenate([np.zeros(len(normal_distances)), np.ones(len(abnormal_distances))])

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(all_labels, all_distances)
roc_auc = auc(fpr, tpr)

print(f"ROC-AUC 分数: {roc_auc:.4f}")
print(f"解释: AUC={roc_auc:.4f}，模型区分正常/异常样本的能力为 {roc_auc*100:.2f}%")

# 找到最佳阈值（使用Youden指数）
youden_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[youden_idx]
best_tpr = tpr[youden_idx]
best_fpr = fpr[youden_idx]

print(f"\n最佳阈值 (Youden指数): {best_threshold:.4f}")
print(f"  对应的真正例率 (TPR/召回率): {best_tpr:.4f}")
print(f"  对应的假正例率 (FPR): {best_fpr:.4f}")
print(f"  准确率: {(best_tpr + (1 - best_fpr)) / 2:.4f}")

# 绘制ROC曲线
fig, ax = plt.subplots(figsize=(8, 6))

# ROC曲线
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC曲线 (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
        label='随机分类器 (AUC = 0.5)')

# 标记最佳阈值点
ax.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5,
           label=f'最佳阈值 (Youden)\n阈值={best_threshold:.3f}\nTPR={best_tpr:.3f}, FPR={best_fpr:.3f}')

# 标记95%和99%分位数对应的点
threshold_95_point = (fpr[np.argmin(np.abs(thresholds - threshold_95))]
                      if len(thresholds) > 0 else 0)
tpr_95_point = (tpr[np.argmin(np.abs(thresholds - threshold_95))]
                if len(thresholds) > 0 else 0)
ax.scatter(threshold_95_point, tpr_95_point, color='green', s=100, marker='s', zorder=5,
           label=f'95%分位数阈值\n阈值={threshold_95:.3f}\nTPR={tpr_95_point:.3f}, FPR={threshold_95_point:.3f}')

threshold_99_point = (fpr[np.argmin(np.abs(thresholds - threshold_99))]
                      if len(thresholds) > 0 else 0)
tpr_99_point = (tpr[np.argmin(np.abs(thresholds - threshold_99))]
                if len(thresholds) > 0 else 0)
ax.scatter(threshold_99_point, tpr_99_point, color='blue', s=100, marker='^', zorder=5,
           label=f'99%分位数阈值\n阈值={threshold_99:.3f}\nTPR={tpr_99_point:.3f}, FPR={threshold_99_point:.3f}')

# 设置坐标轴
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正例率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真正例率 (True Positive Rate)', fontsize=12)
ax.set_title('异常检测ROC曲线', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# =================================额外：不同阈值下的性能对比图=================================
print("\n" + "="*50)
print("阈值性能分析")
print("="*50)

# 计算不同阈值下的性能指标
threshold_range = np.linspace(all_distances.min(), all_distances.max(), 100)
precision_list = []
recall_list = []
f1_list = []
specificity_list = []

for thresh in threshold_range:
    # 预测为正（异常）的样本
    pred_positive = all_distances > thresh

    # 真实为正（异常）的样本
    true_positive = all_labels == 1

    # 计算TP, FP, TN, FN
    TP = np.sum((pred_positive == 1) & (true_positive == 1))
    FP = np.sum((pred_positive == 1) & (true_positive == 0))
    TN = np.sum((pred_positive == 0) & (true_positive == 0))
    FN = np.sum((pred_positive == 0) & (true_positive == 1))

    # 计算指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)

# 绘制性能随阈值变化曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：性能指标曲线
ax1 = axes[0]
ax1.plot(threshold_range, precision_list, 'b-', linewidth=2, label='精确率 (Precision)')
ax1.plot(threshold_range, recall_list, 'r-', linewidth=2, label='召回率/检测率 (Recall)')
ax1.plot(threshold_range, f1_list, 'g-', linewidth=2, label='F1分数')
ax1.plot(threshold_range, specificity_list, 'orange', linewidth=2, label='特异性 (Specificity)')

# 标记最佳F1分数对应的阈值
best_f1_idx = np.argmax(f1_list)
best_f1_thresh = threshold_range[best_f1_idx]
best_f1 = f1_list[best_f1_idx]
ax1.axvline(x=best_f1_thresh, color='gray', linestyle='--', alpha=0.7)
ax1.scatter(best_f1_thresh, best_f1, color='green', s=100, zorder=5,
           label=f'最佳F1分数\n阈值={best_f1_thresh:.3f}\nF1={best_f1:.3f}')

# 标记95%和99%分位数阈值
ax1.axvline(x=threshold_95, color='blue', linestyle='--', alpha=0.5, label=f'95%分位数 ({threshold_95:.3f})')
ax1.axvline(x=threshold_99, color='purple', linestyle='--', alpha=0.5, label=f'99%分位数 ({threshold_99:.3f})')

ax1.set_xlabel('阈值 (编码距离)', fontsize=12)
ax1.set_ylabel('分数', fontsize=12)
ax1.set_title('性能指标随阈值变化曲线', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# 右图：PR曲线 (Precision-Recall Curve)
ax2 = axes[1]
# 计算PR曲线
precision_pr, recall_pr, _ = precision_recall_curve(all_labels, all_distances)
pr_auc = auc(recall_pr, precision_pr)

ax2.plot(recall_pr, precision_pr, 'b-', linewidth=2,
         label=f'PR曲线 (AUC-PR = {pr_auc:.4f})')
ax2.set_xlabel('召回率 (Recall)', fontsize=12)
ax2.set_ylabel('精确率 (Precision)', fontsize=12)
ax2.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# 标记最佳F1点（在PR曲线上）
best_f1_recall = recall_list[best_f1_idx]
best_f1_precision = precision_list[best_f1_idx]
ax2.scatter(best_f1_recall, best_f1_precision, color='red', s=100, zorder=5,
           label=f'最佳F1点\nRecall={best_f1_recall:.3f}, Precision={best_f1_precision:.3f}')

plt.tight_layout()
plt.savefig('threshold_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# =================================输出完整的评估报告=================================
print("\n" + "="*60)
print("异常检测完整评估报告")
print("="*60)

print(f"\n数据集信息:")
print(f"  - 正常样本数: {len(normal_distances)}")
print(f"  - 异常样本数: {len(abnormal_distances)}")
print(f"  - 总样本数: {len(all_distances)}")

print(f"\n距离统计:")
print(f"  - 正常样本距离均值: {np.mean(normal_distances):.4f} ± {np.std(normal_distances):.4f}")
print(f"  - 异常样本距离均值: {np.mean(abnormal_distances):.4f} ± {np.std(abnormal_distances):.4f}")
print(f"  - 分离度 (均值差): {separation:.4f}")

print(f"\nAUC评估:")
print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"  - PR-AUC: {pr_auc:.4f}")
print(f"  - 评估: ", end="")
if roc_auc >= 0.9:
    print("优秀 (Excellent)")
elif roc_auc >= 0.8:
    print("良好 (Good)")
elif roc_auc >= 0.7:
    print("一般 (Fair)")
else:
    print("较差 (Poor)")

print(f"\n最佳阈值 (基于Youden指数):")
print(f"  - 阈值: {best_threshold:.4f}")
print(f"  - 真正例率 (TPR/召回率): {best_tpr:.4f}")
print(f"  - 假正例率 (FPR): {best_fpr:.4f}")
print(f"  - 精确率: {best_f1_precision:.4f}")
print(f"  - F1分数: {best_f1:.4f}")

print(f"\n最佳阈值 (基于F1分数):")
print(f"  - 阈值: {best_f1_thresh:.4f}")
print(f"  - F1分数: {best_f1:.4f}")
print(f"  - 精确率: {best_f1_precision:.4f}")
print(f"  - 召回率: {best_f1_recall:.4f}")

print(f"\n固定分位数阈值性能:")
print(f"  - 95%分位数 ({threshold_95:.4f}): 检测率 = {detection_rate_95*100:.2f}%, 假阳性率 = 5%")
print(f"  - 99%分位数 ({threshold_99:.4f}): 检测率 = {detection_rate_99*100:.2f}%, 假阳性率 = 1%")

print("\n" + "="*60)