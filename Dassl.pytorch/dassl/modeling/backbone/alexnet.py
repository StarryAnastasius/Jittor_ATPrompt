# 保留 PyTorch 的 model_zoo 用于加载权重
import torch
import torch.utils.model_zoo as model_zoo
# 其余部分使用 Jittor
import jittor as jt
import jittor.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


class AlexNet(Backbone):
    def __init__(self):
        super().__init__()
        # 模型结构替换为 Jittor 层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 用 Jittor 实现 AdaptiveAvgPool2d((6,6))
        self.avgpool = lambda x: nn.pool2d(x, kernel_size=x.shape[2:4], stride=1, op='mean')
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self._out_features = 4096

    # Jittor 用 execute 替代 forward
    def execute(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.flatten(x, 1)  # Jittor  flatten
        return self.classifier(x)


def init_pretrained_weights(model, model_url):
    # 保留 torch.model_zoo 加载权重
    pretrain_dict = model_zoo.load_url(model_url)
    # 将 PyTorch 张量（torch.Tensor）转换为 Jittor 变量（jt.Var）
    jittor_dict = {
        k: jt.array(v.cpu().numpy())  # 转换为 Jittor 格式
        for k, v in pretrain_dict.items()
    }
    # Jittor 模型加载参数
    model.load_parameters(jittor_dict, strict=False)


@BACKBONE_REGISTRY.register()
def alexnet(pretrained=True, **kwargs):
    model = AlexNet()

    if pretrained:
        init_pretrained_weights(model, model_urls["alexnet"])

    return model