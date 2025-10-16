import jittor as jt
import jittor.nn as nn

from dassl.utils import init_network_weights  # 假设已适配Jittor

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class Convolution(nn.Module):
    """卷积块：Conv2d + ReLU"""
    def __init__(self, c_in, c_out):
        super().__init__()
        # Jittor的Conv2d参数与PyTorch兼容，移除inplace=True（Jittor不支持）
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    # Jittor用execute替代forward
    def execute(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):
    """主网络结构"""
    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden  # 输出特征维度

    def _check_input(self, x):
        """检查输入尺寸是否为32x32"""
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), f"Input to network must be 32x32, but got {H}x{W}"

    def execute(self, x):
        """前向传播（Jittor用execute替代forward）"""
        self._check_input(x)
        x = self.conv1(x)
        x = nn.max_pool2d(x, 2)  # F.max_pool2d -> nn.max_pool2d
        x = self.conv2(x)
        x = nn.max_pool2d(x, 2)
        x = self.conv3(x)
        x = nn.max_pool2d(x, 2)
        x = self.conv4(x)
        x = nn.max_pool2d(x, 2)
        # 展平特征（view用法与PyTorch一致）
        return x.view(x.size(0), -1)


@BACKBONE_REGISTRY.register()
def cnn_digitsdg(** kwargs):
    """
    用于DigitsDG数据集的网络，源自：
    Zhou et al. Deep Domain-Adversarial Image Generation for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64)
    # 初始化网络权重（需确保init_network_weights已适配Jittor）
    init_network_weights(model, init_type="kaiming")
    return model
    