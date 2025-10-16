import jittor as jt
import jittor.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class PreActBlock(nn.Module):
    expansion = 1  # 残差块输出通道扩展系数（普通块为1）

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # Jittor 层与 PyTorch 参数完全兼容，直接替换模块前缀
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 当步长≠1 或 输入输出通道不匹配时，添加 shortcut 调整维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    # Jittor 用 execute 替代 forward 作为前向传播入口
    def execute(self, x):
        # 预激活逻辑：先 BN + ReLU，再卷积
        out = nn.relu(self.bn1(x))
        # 处理 shortcut（存在则用，否则直接用原输入）
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(nn.relu(self.bn2(out)))
        # 残差连接
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4  # 瓶颈块输出通道扩展系数（4倍）

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        # shortcut 维度调整逻辑（同普通块）
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def execute(self, x):
        # 预激活逻辑：BN + ReLU 在前，卷积在后
        out = nn.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(nn.relu(self.bn2(out)))
        out = self.conv3(nn.relu(self.bn3(out)))
        # 残差连接
        out += shortcut
        return out


class PreActResNet(Backbone):

    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64  # 初始输入通道数

        # 第一层卷积（无预激活，直接卷积）
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 构建四层残差网络（通过 _make_layer 批量创建残差块）
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 输出特征维度（512 * 块扩展系数）
        self._out_features = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        """批量创建残差块：第一个块调整步长，后续块步长为1"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # 更新输入通道数（当前块输出通道 = planes * 扩展系数）
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def execute(self, x):
        # 前向传播流程：卷积 → 四层残差 → 平均池化 → 展平
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 平均池化（核大小4，与CIFAR/SVHN输出尺寸适配）
        out = nn.avg_pool2d(out, 4)
        # 展平为 batch×feature 格式
        out = out.view(out.size(0), -1)
        return out


"""
Preact-ResNet18 was used for the CIFAR10 and
SVHN datasets (both are SSL tasks) in

- Wang et al. Semi-Supervised Learning by
Augmented Distribution Alignment. ICCV 2019.
"""


@BACKBONE_REGISTRY.register()
def preact_resnet18(**kwargs):
    """注册 PreActResNet18：使用普通残差块（PreActBlock），每层2个块"""
    return PreActResNet(PreActBlock, [2, 2, 2, 2])