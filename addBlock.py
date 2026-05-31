__author__ = 'Eric'

"""
自己搭建的网络块文件
包括：
1、基本残差网络（分前向激活和非前向激活版本）
2、瓶颈结构残差块（Bottleneck Block）（分前向激活和非前向激活版本）
"""

#基本包
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#网络层可视化工具
from torchsummary import summary
from torchinfo import summary as sum
from torchviz import make_dot
import netron
import torch.onnx


class ResidualBlock(nn.Module):
    """
    基础残差块（ResNet风格）
    包含跳跃连接（Skip Connection），解决梯度消失问题
    包含前向激活（preactivated）,解决100层以上网络训练困难问题
    一次添加两个-三个（多一个下采样）卷积层
    """

    #初始化定义
    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None, activation='relu',preactivated=False):
        """
        tip:输入通道必须和输出通道匹配，不然，没法相加 : y = f(x) + x [f(x)与x的通道必须匹配]
        in_channels：输入通道数
        out_channels：输出通道数
        stride：步长
        downsample: 下采样层（尺寸/通道不匹配时使用），需要时传入卷积块，不需要为None,默认None
        activation：激活层，只有relu和leaky_relu
        preactivated:是否前向激活,需要为True,不需要为False（Pre-activation/ResNet-v2风格）
        """
        super(ResidualBlock, self).__init__()#继承父类的初始化

        # 下采样层（当输入输出尺寸不匹配时使用）
        self.downsample = downsample
        self.stride = stride
        self.preactivated = preactivated

        if not preactivated:
            # ========== 标准 ResNet (Original) ==========
            # 顺序：Conv → BN → ReLU → Conv → BN → Add → ReLU

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        else:
            # ========== 预激活 ResNet-v2 (Pre-activation) ==========
            # 顺序：BN → ReLU → Conv → BN → ReLU → Conv → Add
            # 注意：第一个BN的通道数是in_channels！

            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)

            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")


    #前向传播逻辑
    def forward(self, x):
        identity = x# 保存原始输入

        if not self.preactivated:
            # ========== 标准 ResNet 前向 ==========

            # 主路径：Conv → BN → Act → Conv → BN
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)  # 注意：这里先不加激活！

            # 下采样（如果需要）
            if self.downsample is not None:
                identity = self.downsample(x)

            # 跳跃连接
            out += identity
            out = self.activation(out)  # 最后加激活

        else:
            # ========== 预激活 ResNet-v2 前向 ==========

            # 预激活路径：BN → Act → Conv → BN → Act → Conv
            out = self.bn1(x)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            # 下采样（如果需要）
            if self.downsample is not None:
                identity = self.downsample(identity)

            # 跳跃连接（直接相加，不再加激活）
            out += identity
            # Pre-activation 模式下，这里不加激活！
            # 因为激活已经在卷积前做过了，再加会丢失identity的原始信息

        return out


class ResidualBottleneckBlock(nn.Module):
    """
    瓶颈结构残差块（Bottleneck Block）
    包含跳跃连接（Skip Connection），解决梯度消失问题
    包含前向激活（Pre-activation），解决100层以上网络训练困难问题
    包含瓶颈结构（1x1→3x3→1x1），减少计算量，支持更深网络
    一次添加三个-四个（多一个下采样）卷积层

    结构：
    - 标准模式 (ResNet-50/101/152):
    Conv1x1→BN→ReLU→
    Conv3x3→BN→ReLU→
    Conv1x1→BN→Add→ReLU
    - 预激活模式 (ResNet-v2):
    BN→ReLU→Conv1x1→
    BN→ReLU→Conv3x3→
    BN→ReLU→Conv1x1→Add

    注意：输出通道数通常是中间通道数的4倍（expansion=4）
    """

    # 扩展系数：Bottleneck输出通道是中间通道的4倍
    expansion = 4

    def __init__(self, in_channels, mid_channels, out_channels=None, stride=1,
                 downsample=None, activation='relu', preactivated=False):
        """
        Args:
            in_channels: 输入通道数
            mid_channels: 中间通道数（瓶颈层的窄通道）
            out_channels: 输出通道数（默认=mid_channels * 4）
            stride: 步长（放在3x3卷积，ResNet-v1.5风格）
            downsample: 下采样层（尺寸/通道不匹配时使用）
            activation: 激活函数 ('relu' 或 'leaky_relu')
            preactivated: 是否预激活（Pre-activation/ResNet-v2风格）
        """
        super(ResidualBottleneckBlock, self).__init__()

        # 如果out_channels未指定，按标准expansion=4计算
        if out_channels is None:
            out_channels = mid_channels * self.expansion

        self.downsample = downsample
        self.stride = stride
        self.preactivated = preactivated

        if not preactivated:
            # ========== 标准 ResNet (ResNet-v1/v1.5) ==========
            """
            顺序：
            Conv(1x1) → BN → ReLU →
            Conv(3x3, stride) → BN → ReLU →  ← stride在这里（ResNet-v1.5推荐）
            Conv(1x1) → BN → Add → ReLU
            """

            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)  # 降维
            self.bn1 = nn.BatchNorm2d(mid_channels)

            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)  # 特征提取+下采样
            self.bn2 = nn.BatchNorm2d(mid_channels)


            self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)  # 升维
            self.bn3 = nn.BatchNorm2d(out_channels)

        else:
            # ========== 预激活 ResNet-v2 (Pre-activation) ==========
            """
            顺序：
            BN → ReLU → Conv(1x1) →
            BN → ReLU → Conv(3x3, stride) →
            BN → ReLU → Conv(1x1) → Add
            """
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)

            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)

            self.bn3 = nn.BatchNorm2d(mid_channels)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        identity = x  # 保存原始输入

        if not self.preactivated:
            # ========== 标准 ResNet 前向 ==========
            """
            顺序：
            Conv(1x1) → BN → ReLU →
            Conv(3x3, stride) → BN → ReLU →  ← stride在这里（ResNet-v1.5推荐）
            Conv(1x1) → BN → Add → ReLU
            """

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.bn3(out)  # 先不加激活

            # 下采样（如果需要）
            if self.downsample is not None:
                identity = self.downsample(x)

            # 跳跃连接
            out += identity
            out = self.activation(out)  # 最后加激活

        else:
            # ========== 预激活 ResNet-v2 前向 ==========
            """
            顺序：
            BN → ReLU → Conv(1x1) →
            BN → ReLU → Conv(3x3, stride) →
            BN → ReLU → Conv(1x1) → Add
            """
            out = self.bn1(x)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.activation(out)
            out = self.conv3(out)

            # 下采样（如果需要）
            if self.downsample is not None:
                identity = self.downsample(identity)

            # 跳跃连接（直接相加，不加激活）
            out += identity

        return out



class DWConv2d(nn.Module):
    """
    深度可分离卷积
    Standard convolutions have the computational cost of:  （普通卷积计算复杂度）
    DK · DK · αM · αN · ρDF · ρDF
    Depthwise separable convolutions cost:（深度可分离卷积计算复杂度）
    DK · DK · αM · ρDF · ρDF + αM · αN · ρDF · ρDF = （DK · DK+ αN）·ρDF · ρDF· αM
    其中，
    DK为卷积和的尺寸（如：DK·DK 3x3）
    M为输入通道数
    N为输出通道数
    DF为特征图尺寸（如：DF·DF 255x255）

    由于DK · DK+N 《 DK · DK · N ，所以，这个优化在计算复杂度上是有效的
    理论上，复杂度达到原来卷积的 (1/N + 1/DK·DK)
    同时，α与ρ也为网络的深度和特征图尺寸进行优化，使得网络在整体的计算量进一步降低！

    """
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True,Width_Multiplier=1,firstBlock=False,endBlock=False):
        """
        深度可分离卷积参数（添加残差连接）
        :param in_channels:输入通道数
        :param out_channels:输出通道数
        :param kernel_size:卷积核尺寸
        :param stride:步长
        :param padding:边缘填充
        :param bias:是否加偏置
        :param Width_Multiplier:宽度乘子
        :param firstBlock:是否为第一层
        :param endBlock:是否为最后一层
        :return:输出特征图
        """
        super(DWConv2d, self).__init__()
        self.in_channels =in_channels
        self.out_channels = out_channels
        # 应用宽度乘子，并确保是整数
        self.in_channels_change = int(in_channels * Width_Multiplier)
        self.out_channels_change = int(out_channels * Width_Multiplier)
        # 确保至少为1
        self.in_channels_change = max(1, self.in_channels_change)
        self.out_channels_change = max(1, self.out_channels_change)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.firstBlock = firstBlock
        self.endBlock = endBlock
        self.use_residual = (self.in_channels_change == self.out_channels_change and stride == 1)


        #定义卷积
        if self.firstBlock == True:#如果是第一层，输入不进行缩放，输出缩放（因为输入缩放会导致和原始数据通道不匹配）
            self.Depthwise_Conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size = self.kernel_size,
                                   stride=self.stride, padding=self.padding, bias=self.bias,groups=self.in_channels)
            self.Pointwise_Conv = nn.Conv2d(self.in_channels, self.out_channels_change, kernel_size = 1,
                                   stride=1, padding=0, bias=self.bias)
            self.bn = nn.BatchNorm2d(self.out_channels_change)
        elif self.endBlock == True:#如果是最后一层，输出不进行缩放，输入缩放（因为输出缩放会导致和原始数据通道不匹配）
            self.Depthwise_Conv = nn.Conv2d(self.in_channels_change, self.in_channels_change, kernel_size = self.kernel_size,
                                   stride=self.stride, padding=self.padding, bias=self.bias,groups=self.in_channels_change)
            self.Pointwise_Conv = nn.Conv2d(self.in_channels_change, self.out_channels, kernel_size = 1,
                                   stride=1, padding=0, bias=self.bias)
            self.bn = nn.BatchNorm2d(self.out_channels)
        else:#如果不是第一层，那就输入输出都缩放
            self.Depthwise_Conv = nn.Conv2d(self.in_channels_change, self.in_channels_change, kernel_size = self.kernel_size,
                                   stride=self.stride, padding=self.padding, bias=self.bias,groups=self.in_channels_change)
            self.Pointwise_Conv = nn.Conv2d(self.in_channels_change, self.out_channels_change, kernel_size = 1,
                                   stride=1, padding=0, bias=self.bias)
            self.bn = nn.BatchNorm2d(self.out_channels_change)


    def forward(self,x):
        identity = x
        x = self.Depthwise_Conv(x)
        x = self.Pointwise_Conv(x)
        x = self.bn(x)
        if self.use_residual:
            x = x + identity
        return x


class DWConvTranspose2d(nn.Module):
    """
    深度可分离转置卷积（用于上采样）

    原理：将转置卷积分解为两个步骤：
    1. Depthwise Transpose Conv：每个通道独立进行上采样
    2. Pointwise Conv：混合通道信息并调整通道数

    优势：计算量约为普通转置卷积的 (1/N + 1/DK·DK)
    其中 DK 为卷积核尺寸，N 为输出通道数
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 output_padding=0, bias=True, Width_Multiplier=1, firstBlock=False, endBlock=False):
        """
        深度可分离转置卷积参数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小（通常为4，配合stride=2实现2倍上采样）
        :param stride: 步长（上采样倍数，通常为2）
        :param padding: 边缘填充（通常为1，配合kernel_size=4实现2倍上采样）
        :param output_padding: 输出填充（处理奇数尺寸，通常为0或1）
        :param bias: 是否使用偏置
        :param Width_Multiplier: 宽度乘子（控制通道数缩放）
        :param firstBlock: 是否为第一层（输入不缩放）
        :param endBlock: 是否为最后一层（输出不缩放）
        """
        super(DWConvTranspose2d, self).__init__()

        # 保存原始输入输出通道数
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 应用宽度乘子，并确保是整数
        # 输入通道缩放（乘以宽度因子）
        self.in_channels_change = int(in_channels * Width_Multiplier)
        # 输出通道缩放（乘以宽度因子）
        self.out_channels_change = int(out_channels * Width_Multiplier)

        # 确保缩放后的通道数至少为1（防止出现0通道）
        self.in_channels_change = max(1, self.in_channels_change)
        self.out_channels_change = max(1, self.out_channels_change)

        # 保存卷积参数
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias
        self.firstBlock = firstBlock
        self.endBlock = endBlock

        # ========== 定义卷积层 ==========

        if self.firstBlock == True:
            # 情况1：第一层（编码器的输入层或解码器的输入层）
            # 输入不缩放（因为原始数据通道不能被缩放）
            # 输出缩放（应用宽度乘子）

            # Depthwise转置卷积：每个通道独立上采样
            # 输入通道 = 原始输入通道（不缩放）
            # 输出通道 = 原始输入通道（保持不变）
            # groups = in_channels：每个通道独立卷积
            self.Depthwise_TransposeConv = nn.ConvTranspose2d(
                self.in_channels,           # 输入通道数（原始值）
                self.in_channels,           # 输出通道数（与输入相同）
                kernel_size=self.kernel_size,   # 卷积核大小
                stride=self.stride,             # 步长（控制上采样倍数）
                padding=self.padding,           # 填充
                output_padding=self.output_padding,  # 输出填充
                bias=self.bias,                 # 是否使用偏置
                groups=self.in_channels         # 分组数=输入通道数（深度卷积）
            )

            # Pointwise卷积：混合通道信息并调整通道数
            # 输入通道 = 原始输入通道
            # 输出通道 = 缩放后的输出通道
            # 1x1卷积用于通道混合和维度变换
            self.Pointwise_Conv = nn.Conv2d(
                self.in_channels,               # 输入通道数（原始值）
                self.out_channels_change,       # 输出通道数（缩放后）
                kernel_size=1,                  # 1x1卷积核
                stride=1,                       # 步长为1（不改变空间尺寸）
                padding=0,                      # 不填充
                bias=self.bias                  # 是否使用偏置
            )
            self.bn = nn.BatchNorm2d(self.out_channels_change)

        elif self.endBlock == True:
            # 情况2：最后一层（输出层）
            # 输入缩放（应用宽度乘子）
            # 输出不缩放（保持目标通道数，如RGB图像的3通道）

            # Depthwise转置卷积：每个通道独立上采样
            # 输入通道 = 缩放后的输入通道
            # 输出通道 = 缩放后的输入通道（保持不变）
            # groups = in_channels_change：每个通道独立卷积
            self.Depthwise_TransposeConv = nn.ConvTranspose2d(
                self.in_channels_change,        # 输入通道数（缩放后）
                self.in_channels_change,        # 输出通道数（与输入相同）
                kernel_size=self.kernel_size,   # 卷积核大小
                stride=self.stride,             # 步长（控制上采样倍数）
                padding=self.padding,           # 填充
                output_padding=self.output_padding,  # 输出填充
                bias=self.bias,                 # 是否使用偏置
                groups=self.in_channels_change  # 分组数=缩放后输入通道数
            )

            # Pointwise卷积：混合通道信息并调整到目标通道数
            # 输入通道 = 缩放后的输入通道
            # 输出通道 = 原始输出通道（不缩放，如3）
            self.Pointwise_Conv = nn.Conv2d(
                self.in_channels_change,        # 输入通道数（缩放后）
                self.out_channels,              # 输出通道数（原始值，不缩放）
                kernel_size=1,                  # 1x1卷积核
                stride=1,                       # 步长为1
                padding=0,                      # 不填充
                bias=self.bias                  # 是否使用偏置
            )
            self.bn = nn.BatchNorm2d(self.out_channels)

        else:
            # 情况3：中间层
            # 输入和输出都缩放（应用宽度乘子）

            # Depthwise转置卷积：每个通道独立上采样
            # 输入通道 = 缩放后的输入通道
            # 输出通道 = 缩放后的输入通道（保持不变）
            # groups = in_channels_change：每个通道独立卷积
            self.Depthwise_TransposeConv = nn.ConvTranspose2d(
                self.in_channels_change,        # 输入通道数（缩放后）
                self.in_channels_change,        # 输出通道数（与输入相同）
                kernel_size=self.kernel_size,   # 卷积核大小
                stride=self.stride,             # 步长（控制上采样倍数）
                padding=self.padding,           # 填充
                output_padding=self.output_padding,  # 输出填充
                bias=self.bias,                 # 是否使用偏置
                groups=self.in_channels_change  # 分组数=缩放后输入通道数
            )

            # Pointwise卷积：混合通道信息并调整通道数
            # 输入通道 = 缩放后的输入通道
            # 输出通道 = 缩放后的输出通道
            self.Pointwise_Conv = nn.Conv2d(
                self.in_channels_change,        # 输入通道数（缩放后）
                self.out_channels_change,       # 输出通道数（缩放后）
                kernel_size=1,                  # 1x1卷积核
                stride=1,                       # 步长为1
                padding=0,                      # 不填充
                bias=self.bias                  # 是否使用偏置
            )
            self.bn = nn.BatchNorm2d(self.out_channels_change)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入特征图
        :return: 输出特征图（上采样后的结果）
        """
        # 第一步：Depthwise转置卷积（空间上采样）
        # 每个通道独立进行上采样操作
        x = self.Depthwise_TransposeConv(x)

        # 第二步：Pointwise卷积（通道混合）
        # 使用1x1卷积混合通道信息并调整通道数
        x = self.Pointwise_Conv(x)
        x = self.bn(x)

        return x




if __name__ == '__main__':

    # ==================== 像 tf.keras.Sequential 一样直接搭 ====================
    # ============================== 模型定义 ===================================
    #使用 OrderedDict（给每层命名，更像TF）

    # model = nn.Sequential(OrderedDict([
    # ('resBlok', ResidualBottleneckBlock(512,30,512,preactivated=True)),
    # ('dowmSample',nn.Conv2d(512, 256, kernel_size=1,
    #                                stride=2, padding=0, bias=False))
    # ]))

    model = nn.Sequential(OrderedDict([
    ('DWConv2d', DWConv2d(512,128)),
    ('DWConv2d_1',DWConv2d(128,128))
    ]))
    model1 = nn.Sequential(OrderedDict([
    ('Conv2d', nn.Conv2d(512,128,3)),
    ('Conv2d_1',nn.Conv2d(128,128,3))
    ]))

    # ==========================================================================

    # 可以像字典一样访问
    #print(model.block1)  # 访问特定层

    # 直接使用
    #x = torch.randn(2, 3, 224, 224)
    #output = model(x)  # 自动顺序执行，无需forward
    #print(output.shape)  # [2, 10]

    # ================================== 打印模型架构 ============================

    # 1、类似 model.summary() 的表格输出
    summary(model, input_size=(512, 224, 224), device='cpu')#（输入通道，输入w尺寸，输入h尺寸）
    summary(model1, input_size=(512, 224, 224), device='cpu')

    # ================================== 打印模型架构 ============================
    # 2、torchinfo（推荐，信息更丰富）
        # 输出包含：
        # - 每层的输入/输出形状
        # - 参数量
        # - 乘加运算量（FLOPs估算）
        # - 内存占用
    '''
    sum(model,
        input_size=(2,512, 224, 224),  # (batch, channels, H, W)
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],#显示的列
        col_width=20,#每列显示的宽度
        row_settings=["var_names"],#行的显示设置
        verbose=1)
    '''

    # ================================== 打印模型架构 ============================
    # 3、可视化结构图（生成图片）torchviz
    '''
    x = torch.randn(2,512, 224, 224)
    y = model(x)

    # 生成计算图
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")  # 保存为PNG
    #dot.render("model_architecture", format="pdf")  # 保存为PDF
    #dot
    '''

    # # ================================== 打印模型架构 ============================
    # # 4、Netron（交互式可视化，最直观）
    #
    # x = torch.randn(2,512, 224, 224)
    #
    # # 导出为ONNX格式
    # torch.onnx.export(model, x, "model.onnx",
    #                   input_names=['input'],#输入节点名称	在 Netron 中显示为 'input'
    #                   output_names=['output'],#output_names	输出节点名称	在 Netron 中显示为 'output'
    #                   #动态维度	指定哪些维度可变（这里是 batch 维度）
    #                   #输入的第0维（batch）是动态的，可以变化
    #                   #输出的第0维（batch）也是动态的
    #                   dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    #                   )
    #
    # # 启动Netron查看（浏览器自动打开）
    # netron.start("model.onnx", browse=True)#模型地址，打开浏览器
    #
    # # 加入阻塞，防止进程结束
    # input("按回车键停止服务...")  # 或者 while True: pass

