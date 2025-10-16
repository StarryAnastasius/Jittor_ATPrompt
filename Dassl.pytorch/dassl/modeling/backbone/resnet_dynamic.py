"""
Dynamic ResNet from `"Dynamic Domain Generalization" <https://github.com/MetaVisionLab/DDG>`_.
Adapted for Jittor framework
"""

from typing import Any, List, Type, Union, Callable, Optional
from collections import OrderedDict
import jittor as jt
import jittor.nn as nn
from jittor import Var  # 替换 PyTorch 的 Tensor
import os
import requests
from tqdm import tqdm
# 临时导入 torch 用于加载 PyTorch 格式的预训练权重（权重文件为 .pth 格式）
import torch

from dassl.modeling.ops import MixStyle, Conv2dDynamic  # 假设该模块已适配 Jittor

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

__all__ = [
    "resnet18_dynamic", "resnet50_dynamic", "resnet101_dynamic",
    "resnet18_dynamic_ms_l123", "resnet18_dynamic_ms_l12",
    "resnet18_dynamic_ms_l1", "resnet50_dynamic_ms_l123",
    "resnet50_dynamic_ms_l12", "resnet50_dynamic_ms_l1",
    "resnet101_dynamic_ms_l123", "resnet101_dynamic_ms_l12",
    "resnet101_dynamic_ms_l1"
]

model_urls = {
    "resnet18_dynamic":
    "https://csip.fzu.edu.cn/files/models/resnet18_dynamic-074db766.pth",
    "resnet50_dynamic":
    "https://csip.fzu.edu.cn/files/models/resnet50_dynamic-2c3b0201.pth",
    "resnet101_dynamic":
    "https://csip.fzu.edu.cn/files/models/resnet101_dynamic-c5f15780.pth",
}


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding（Jittor 适配）"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv3x3_dynamic(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    attention_in_channels: int = None
) -> Conv2dDynamic:
    """3x3 dynamic convolution（保持原接口，依赖 Conv2dDynamic 已适配 Jittor）"""
    return Conv2dDynamic(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        attention_in_channels=attention_in_channels
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution（Jittor 适配）"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


def load_state_dict_from_url(url: str, progress: bool = True) -> OrderedDict:
    """自定义权重加载：下载并转换 PyTorch 权重为 Jittor 格式"""
    # 1. 定义缓存路径
    cache_dir = os.path.expanduser("~/.cache/jittor/models/dynamic_resnet")
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    cached_file = os.path.join(cache_dir, filename)

    # 2. 下载权重（若未缓存）
    if not os.path.exists(cached_file):
        print(f"Downloading dynamic ResNet weights from: {url}")
        response = requests.get(url, stream=True, verify=True)
        total_size = int(response.headers.get("content-length", 0))
        
        with open(cached_file, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, disable=not progress
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        print(f"Weights saved to: {cached_file}")

    # 3. 加载 PyTorch 权重并转换为 Jittor Var
    torch_state_dict = torch.load(cached_file, map_location="cpu")
    jittor_state_dict = OrderedDict()
    for k, v in torch_state_dict.items():
        # PyTorch Tensor → Jittor Var
        jittor_state_dict[k] = jt.array(v.numpy())
    
    return jittor_state_dict


def load_state_dict(
    model: nn.Module,
    state_dict: OrderedDict,
    allowed_missing_keys: List = None
):
    r"""Jittor 适配的状态字典加载函数（保留原错误处理逻辑）"""
    # Jittor 用 load_parameters 替代 PyTorch 的 load_state_dict
    missing_keys, unexpected_keys = model.load_parameters(
        state_dict, strict=allowed_missing_keys is None, return_missing=True
    )

    msgs: List[str] = []
    raise_error = False
    if len(unexpected_keys) > 0:
        raise_error = True
        msgs.insert(
            0, "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join("'{}'".format(k) for k in unexpected_keys)
            )
        )
    if len(missing_keys) > 0:
        if allowed_missing_keys is None or sorted(missing_keys) != sorted(
            allowed_missing_keys
        ):
            raise_error = True
        msgs.insert(
            0, "Missing key(s) in state_dict: {}. ".format(
                ", ".join("'{}'".format(k) for k in missing_keys)
            )
        )
    if raise_error:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(msgs)
            )
        )
    if len(msgs) > 0:
        print(
            "\nInfo(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(msgs)
            )
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
      
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # Jittor 用 execute 替代 forward
    def execute(self, x: Var) -> Var:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # 移除 inplace=True
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    # Jittor 用 execute 替代 forward
    def execute(self, x: Var) -> Var:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockDynamic(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockDynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # 动态卷积层（依赖 Conv2dDynamic 已适配 Jittor）
        self.conv1 = conv3x3_dynamic(
            inplanes, planes, stride, attention_in_channels=inplanes
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()  # 移除 inplace=True
        self.conv2 = conv3x3_dynamic(
            planes, planes, attention_in_channels=inplanes
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # Jittor 用 execute 替代 forward，保持动态卷积的 attention_x 传递
    def execute(self, x: Var) -> Var:
        identity = x

        # 动态卷积需传入 attention_x（与原逻辑一致）
        out = self.conv1(x, attention_x=x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckDynamic(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckDynamic, self).__init__()
        if groups != 1:
            raise ValueError("BottleneckDynamic only supports groups=1")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BottleneckDynamic"
            )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # 移除 inplace=True
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_dynamic(
            width, width, stride, attention_in_channels=inplanes
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    # Jittor 用 execute 替代 forward，保持动态卷积逻辑
    def execute(self, x: Var) -> Var:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 动态卷积传递 attention_x
        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, BasicBlockDynamic,
                          BottleneckDynamic]],
        layers: List[int],
        has_fc: bool = True,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ms_class=None,
        ms_layers=None,
        ms_p=0.5,
        ms_a=0.1
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".
                format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        # 基础卷积层（Jittor 适配）
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()  # 移除 inplace=True
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        # 自适应平均池化（Jittor 支持该接口）
        self.avgpool = nn.adaptive_avg_pool2d((1, 1))

        self.has_fc = has_fc
        self._out_features = 512 * block.expansion
        if has_fc:
            self.fc = nn.Linear(self._out_features, num_classes)
            self._out_features = num_classes

        # MixStyle 配置（保留原逻辑，依赖 MixStyle 已适配 Jittor）
        if ms_class is not None and ms_layers is not None:
            self.ms_class = ms_class(p=ms_p, alpha=ms_a)
            for layer in ms_layers:
                assert layer in ["layer1", "layer2", "layer3"]
            self.ms_layers = ms_layers
        else:
            self.ms_class = None
            self.ms_layers = []

        # 权重初始化（Jittor 与 PyTorch 接口兼容）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 残差分支最后一层 BN 零初始化（保持原逻辑）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    # 合并 _forward_impl 逻辑到 execute（Jittor 推荐直接用 execute）
    def execute(self, x: Var) -> Var:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = jt.flatten(x, 1)  # 替换 torch.flatten
        if self.has_fc:
            x = self.fc(x)

        return x


def _resnet(
    arch: str, block: Type[Union[BasicBlock, Bottleneck, BasicBlockDynamic,
                                 BottleneckDynamic]], layers: List[int],
    pretrained: bool, progress: bool, **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # 加载预训练权重（转换为 Jittor 格式）
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        # 移除不需要的权重键（fc 层，若模型不包含 fc）
        removed_keys = []
        if model.has_fc is False or (model.has_fc and model._out_features != 1000):
            removed_keys = ["fc.weight", "fc.bias"]
        for key in removed_keys:
            if key in state_dict:
                state_dict.pop(key)
        # 配置允许缺失的键
        allowed_missing_keys = removed_keys if model.has_fc else None
        load_state_dict(model, state_dict, allowed_missing_keys)
    return model


@BACKBONE_REGISTRY.register()
def resnet18_dynamic(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet18_dynamic",
        BasicBlockDynamic, [2, 2, 2, 2],
        pretrained=pretrained,
        progress=True,
        has_fc=False
    )


@BACKBONE_REGISTRY.register()
def resnet50_dynamic(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet50_dynamic",
        BottleneckDynamic, [3, 4, 6, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False
    )


@BACKBONE_REGISTRY.register()
def resnet101_dynamic(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet101_dynamic",
        BottleneckDynamic, [3, 4, 23, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False
    )


@BACKBONE_REGISTRY.register()
def resnet18_dynamic_ms_l123(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet18_dynamic",
        BasicBlockDynamic, [2, 2, 2, 2],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"]
    )


@BACKBONE_REGISTRY.register()
def resnet18_dynamic_ms_l12(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet18_dynamic",
        BasicBlockDynamic, [2, 2, 2, 2],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"]
    )


@BACKBONE_REGISTRY.register()
def resnet18_dynamic_ms_l1(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet18_dynamic",
        BasicBlockDynamic, [2, 2, 2, 2],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )


@BACKBONE_REGISTRY.register()
def resnet50_dynamic_ms_l123(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet50_dynamic",
        BottleneckDynamic, [3, 4, 6, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"]
    )


@BACKBONE_REGISTRY.register()
def resnet50_dynamic_ms_l12(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet50_dynamic",
        BottleneckDynamic, [3, 4, 6, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"]
    )


@BACKBONE_REGISTRY.register()
def resnet50_dynamic_ms_l1(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet50_dynamic",
        BottleneckDynamic, [3, 4, 6, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )


@BACKBONE_REGISTRY.register()
def resnet101_dynamic_ms_l123(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet101_dynamic",
        BottleneckDynamic, [3, 4, 23, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"]
    )


@BACKBONE_REGISTRY.register()
def resnet101_dynamic_ms_l12(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet101_dynamic",
        BottleneckDynamic, [3, 4, 23, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"]
    )


@BACKBONE_REGISTRY.register()
def resnet101_dynamic_ms_l1(pretrained=True, **kwargs) -> ResNet:
    return _resnet(
        "resnet101_dynamic",
        BottleneckDynamic, [3, 4, 23, 3],
        pretrained=pretrained,
        progress=True,
        has_fc=False,
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )