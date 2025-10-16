"""
Reference
https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA
Adapted for Jittor framework
"""
import jittor as jt
import jittor.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class FeatureExtractor(Backbone):

    def __init__(self):
        super().__init__()
        # 1. 替换 PyTorch 层为 Jittor 对应层（参数完全兼容）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

        self._out_features = 2048

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    # 2. Jittor 用 execute 替代 forward 作为前向传播方法
    def execute(self, x):
        self._check_input(x)
        # 3. 替换 PyTorch 的 F.xxx 为 Jittor 的 nn.xxx（功能一致）
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = nn.relu(self.bn3(self.conv3(x)))
        # 4. 替换 torch.Tensor.view 为 jittor.Var.view（或 jt.view，写法一致）
        x = x.view(x.size(0), 8192)
        x = nn.relu(self.bn1_fc(self.fc1(x)))
        # 5. Jittor 的 dropout 支持 training 参数，与 PyTorch 行为一致
        x = nn.dropout(x, training=self.training)
        x = nn.relu(self.bn2_fc(self.fc2(x)))
        return x


@BACKBONE_REGISTRY.register()
def cnn_digit5_m3sda(**kwargs):
    """
    This architecture was used for the Digit-5 dataset in:

        - Peng et al. Moment Matching for Multi-Source
        Domain Adaptation. ICCV 2019.
    """
    return FeatureExtractor()