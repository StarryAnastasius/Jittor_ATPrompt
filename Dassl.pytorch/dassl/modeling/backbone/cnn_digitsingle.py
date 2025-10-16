"""
This model is built based on
https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
Adapted for Jittor framework
"""
import jittor as jt
import jittor.nn as nn

from dassl.utils import init_network_weights  # 需确保该函数已适配Jittor（参考此前权重初始化适配代码）

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class CNN(Backbone):

    def __init__(self):
        super().__init__()
        # 1. 替换 PyTorch 层为 Jittor 层（参数完全兼容，无需修改）
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc3 = nn.Linear(5 * 5 * 128, 1024)
        self.fc4 = nn.Linear(1024, 1024)

        self._out_features = 1024

    def _check_input(self, x):
        # 2. Jittor 张量 shape 属性与 PyTorch 一致，输入检查逻辑不变
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    # 3. Jittor 用 execute 替代 forward 作为前向传播入口
    def execute(self, x):
        self._check_input(x)
        # 4. 替换 PyTorch 的 F.xxx 为 Jittor 的 nn.xxx（功能完全对齐）
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.max_pool2d(x, 2)

        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool2d(x, 2)

        # 5. Jittor 张量 view 方法用法与 PyTorch 一致，展平操作不变
        x = x.view(x.size(0), -1)

        x = self.fc3(x)
        x = nn.relu(x)

        x = self.fc4(x)
        x = nn.relu(x)

        return x


@BACKBONE_REGISTRY.register()
def cnn_digitsingle(**kwargs):
    model = CNN()
    # 6. 权重初始化函数需提前适配 Jittor（参考此前：用 nn.init.kaiming_normal_ 等 Jittor 接口）
    init_network_weights(model, init_type="kaiming")
    return model