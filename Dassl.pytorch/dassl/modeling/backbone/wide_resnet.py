"""
Modified from https://github.com/xternalz/WideResNet-pytorch
Adapted for Jittor framework
"""
import jittor as jt
import jittor.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # Jittor不支持inplace参数，移除inplace=True
        self.relu1 = nn.LeakyReLU(0.01)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        # shortcut卷积（当输入输出通道数不同时）
        self.convShortcut = (
            (not self.equalInOut) and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            ) or None
        )

    # Jittor用execute替代forward
    def execute(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        
        # 计算主分支
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = nn.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        
        # 残差连接：主分支 + 捷径分支
        return (x if self.equalInOut else self.convShortcut(x)) + out


class NetworkBlock(nn.Module):

    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0
    ):
        super().__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    # 第一层输入通道为in_planes，后续为out_planes
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    # 第一层使用指定stride，后续为1
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def execute(self, x):
        return self.layer(x)


class WideResNet(Backbone):

    def __init__(self, depth, widen_factor, dropRate=0.0):
        super().__init__()
        # 通道数配置：基础通道数 * 宽度因子
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        # 验证网络深度是否符合预期（WideResNet的深度计算方式）
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 三个网络块
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # 最终的批归一化和激活
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.01)

        self._out_features = nChannels[3]

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def execute(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # 替代adaptive_avg_pool2d(1)：全局平均池化到1x1
        out = out.mean(dim=[2, 3], keepdims=True)
        # 展平特征
        return out.view(out.size(0), -1)


@BACKBONE_REGISTRY.register()
def wide_resnet_28_2(**kwargs):
    """WideResNet-28-2"""
    return WideResNet(28, 2)


@BACKBONE_REGISTRY.register()
def wide_resnet_16_4(** kwargs):
    """WideResNet-16-4"""
    return WideResNet(16, 4)
