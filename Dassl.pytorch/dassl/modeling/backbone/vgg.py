import jittor as jt
import jittor.nn as nn
import os
import requests
from tqdm import tqdm

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

# VGG模型预训练权重URL
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

# 自定义权重加载函数，替代torch.hub或model_zoo
def load_state_dict_from_url(url, progress=True):
    """下载并加载权重文件"""
    # 缓存目录
    cache_dir = os.path.expanduser("~/.cache/jittor/models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 提取文件名
    filename = url.split('/')[-1]
    cached_file = os.path.join(cache_dir, filename)
    
    # 如果没有缓存，下载文件
    if not os.path.exists(cached_file):
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(cached_file, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, unit_divisor=1024, disable=not progress
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    # 加载权重并转换为Jittor格式
    import torch  # 临时导入torch用于加载权重文件
    torch_state_dict = torch.load(cached_file, map_location='cpu')
    jittor_state_dict = {
        k: jt.array(v.numpy()) for k, v in torch_state_dict.items()
    }
    return jittor_state_dict


class VGG(Backbone):

    def __init__(self, features, init_weights=True):
        super().__init__()
        self.features = features
        # 用Jittor实现AdaptiveAvgPool2d((7,7))
        self.avgpool = lambda x: nn.adaptive_avg_pool2d(x, (7, 7))
        # 分类器部分（输出特征而非logits）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),  # 移除inplace=True
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),  # 移除inplace=True
            nn.Dropout(),
        )

        self._out_features = 4096

        if init_weights:
            self._initialize_weights()

    # Jittor用execute替代forward
    def execute(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.flatten(x, 1)  # 替代torch.flatten
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Jittor的初始化函数
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 移除inplace=True
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                # 移除inplace=True
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


# VGG配置参数（与原PyTorch版本一致）
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M"
    ],
    "E": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M"
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained):
    init_weights = False if pretrained else True
    model = VGG(
        make_layers(cfgs[cfg], batch_norm=batch_norm),
        init_weights=init_weights
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        # 加载权重到Jittor模型
        model.load_parameters(state_dict, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    return _vgg("vgg16", "D", False, pretrained)
