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
from torchinfo import summary as sum
from torchviz import make_dot
import netron
import torch.onnx


# =================================设置超参数=================================

# 查看当前环境是否有GPU，有则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 定义超参数
learning_rate = 0.001
num_epoch = 500
batch_size = 128
latent_size = 100  # 使用2维潜空间便于可视化
image_widht = 640
image_height = 480

w_adv =1    #对抗损失权重
w_con =50   #语义损失权重
w_enc =1    #编码损失权重

# 创建保存目录
sample_dir = './samples1'
os.makedirs(sample_dir, exist_ok=True)

Width_Multiplier = 1 #宽度乘子（）
Resolution_Multiplier = 1 #分辨率乘子（加载数据用）

#==================================00加载数据=================================
image_width = int(image_widht*Resolution_Multiplier)#应用分辨率乘子
image_height = int(image_height*Resolution_Multiplier)#应用分辨率乘子

#先用minist数据集删去一个类别跑一遍先

#==================================01构建判别器=================================
class D(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+下采样
        self.conv_first = addBlock.DWConv2d(in_channels=3,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
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
        z = self.MLP(x)  # [batch, latent_size]

        result = self.activate(z)#二分类SIGMOID判断真假图片

        return result,z


#==================================02构建生成器=================================
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
        self.conv_first = addBlock.DWConv2d(in_channels=3,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
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

        #全局池化
        x = self.GAP(x)# [batch, latent_size, 1, 1]

        # 展平
        z = x.view(x.size(0), -1)  # [batch, latent_size]

        # MLP 处理（用于做z与z^的对比）
        # z = self.MLP(x)  # [batch, latent_size]

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
        self.conv_end = addBlock.DWConv2d(in_channels=512,out_channels=3,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,endBlock=True)


        #激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

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
#
# # 使用
# generator = Generator(G_Encoder, G_Decoder).to(device)
#
# # torchinfo 能更好地处理多输出
# sum(generator, input_size=(1, 3, image_height, image_width),
#         col_names=["input_size", "output_size", "num_params"],
#         device="cuda")
# # summary(G_Decoder, input_size=(100, 15, 20), device="cuda")#如果有多个返回值，它没法处理




#==================================03构建额外编码器=================================
class E(nn.Module):
    def __init__(self):
        super().__init__()
        #第一部分卷积+下采样
        self.conv_first = addBlock.DWConv2d(in_channels=3,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=Width_Multiplier,firstBlock=True)
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
        z = x.view(x.size(0), -1)  # [batch, latent_size]

        # MLP 处理（用于做z与z^的对比）
        # z = self.MLP(x)  # [batch, latent_size]

        return z


#==================================04定义损失函数，优化器等=================================
#通常在深度学习中，图像数据可能经过了标准化处理，取值范围从[0,1]或者[-1,1]
#denorm函数的作用就是将输入数据范围从[-1,1]转回到[0,1]，即逆标准化的过程
#clamp(0,1)是限制（clamp）操作，确保返回的数据不会超出0到1的范围
#把判别器和生成器等迁移到GPU上
D = D.to(device)
G_Encoder = G_E.to(device)
G_Decoder = G_D.to(device)
add_Encoder = E.to(device)


#定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()
D_optimizer = torch.optim.Adam(D.parameters(),lr=0.0001)
G_E_optimizer = torch.optim.Adam(G_Encoder.parameters(),lr=0.0001)
G_D_optimizer = torch.optim.Adam(G_Decoder.parameters(),lr=0.0001)
add_E_optimizer = torch.optim.Adam(add_Encoder.parameters(),lr=0.0001)


#Clamp函数x限制在区间[min,max]内
def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

def reset_grad():
    D_optimizer.zero_grad()
    G_E_optimizer.zero_grad()
    G_D_optimizer.zero_grad()
    add_E_optimizer.zero_grad()

#开始训练
total_step = len(train_loader)

#==================================05训练模型=================================
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
        real_labels, fake_labels = get_labels(batch, smooth=True)
        #====================================================
        #                   训练判别器
        #====================================================
        #1、定义判别器对真图片的损失函数
        outputs,_ = D(images)
        d_loss_real = criterion(outputs,real_labels)
        real_score = outputs
        #2、定义判别器对假图片（即由潜在空间点生成的图片）的损失函数
        with torch.no_grad():  # 训练D时G不更新
            _,fake_maps = G_Encoder(images)
            fake_images = G_Decoder(fake_maps)
        outputs,_ = D(fake_images)
        d_loss_fake = criterion(outputs,fake_labels)
        fake_score = outputs

        #得到判别器总的损失函数
        d_loss = d_loss_real+d_loss_fake
        epoch_d_loss += d_loss.item()

        #对生成器、判别器的梯度清零
        reset_grad()#梯度清零
        d_loss.backward()#反向传播
        D_optimizer.step()#参数更新


        #====================================================
        #                   训练生成器
        #====================================================
        #===================1、获得对抗损失（原图和假图在D上编码向量的L2距离）==========================
        #1）、获得原图的编码向量
        _,fz = D(images)
        #1）、获得假图的编码向量
        z,fake_maps = G_Encoder(images)
        fake_images = G_Decoder(fake_maps)
        _,fz_g = D(fake_images)
        L_adv = torch.norm(fz - fz_g, p=2)#计算欧式距离
        #===================2、获得语义损失（原图和假图的L1距离）==========================
        #计算两张假图之间的L1距离
        L_con = torch.sum(torch.abs(images - fake_images))
        #===================3、获得编码损失（原图在G中编码和假图在E中编码向量的L2距离）==========================
        z_e = add_Encoder(fake_images)
        L_enc = torch.norm(z - z_e, p=2)#计算欧式距离

        #计算总损失
        g_loss = w_adv*L_adv+w_con*L_con+w_enc*L_enc
        epoch_g_loss += g_loss.item()
        #对生成器、判别器的梯度清零
        reset_grad()#梯度清零
        g_loss.backward()#反向传播
        #参数更新
        G_E_optimizer.step()
        G_D_optimizer.step()
        add_E_optimizer.step()

        #打印训练信息
        if (i+1)%200 == 0:
            print('Epoch[{}/{}],step[{}/{}],d_loss:{:.4f},g_loss:{:.4f},D(x):{:.2f},D(G(z)):{:.2f},L_adv(对抗损失):{:2f},L_con(语义损失):{:2f},L_enc(编码损失):{:2f}'
                  .format(epoch+1,num_epoch,i+1,total_step,d_loss.item(),g_loss.item(),
                          real_score.mean().item(),fake_score.mean().item(),
                          L_adv.item(),L_con.item(),L_enc.item()))

    #保存假图片
    if (epoch+1)%10 == 0:
        save_image(denorm(fake_images),os.path.join(sample_dir,'fake_image-{}.png'.format(epoch+1)))
        save_image(denorm(images),os.path.join(sample_dir,'real_image-{}.png'.format(epoch+1)))

    # 记录每个 epoch 的平均损失
    D_loss_epoch.append(epoch_d_loss / len(train_loader))
    G_loss_epoch.append(epoch_g_loss / len(train_loader))


#==================================06保存模型=================================
torch.save(D.state_dict(),'D.ckpt')
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


#==================================07可视化结果=================================



