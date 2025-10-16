"""
Credit to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Adapted for Jittor framework
"""
import functools
import jittor as jt
import jittor.nn as nn

from .build import NETWORK_REGISTRY


def init_network_weights(model, init_type="normal", gain=0.02):
    """Jittor 适配的网络权重初始化函数"""
    def _init_func(m):
        classname = m.__class__.__name__
        # 处理卷积层和全连接层权重
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)  # Jittor 无 .data 属性，直接操作权重
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                raise NotImplementedError(
                    f"initialization method {init_type} is not implemented"
                )
            # 处理偏置项
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        # 处理 BatchNorm2d
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        # 处理 InstanceNorm2d
        elif classname.find("InstanceNorm2d") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    model.apply(_init_func)


def get_norm_layer(norm_type="instance"):
    """生成 Jittor 兼容的归一化层（BatchNorm/InstanceNorm）"""
    if norm_type == "batch":
        # Jittor BatchNorm2d 与 PyTorch 参数兼容（affine=True 启用可学习参数）
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        # Jittor InstanceNorm2d 支持 track_running_stats 参数
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(
            f"normalization layer [{norm_type}] is not found"
        )
    return norm_layer


class ResnetBlock(nn.Module):
    """残差块（Jittor 适配，支持反射/复制/零填充）"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias
    ):
        conv_block = []
        p = 0  # 零填充的 padding 大小
        # 选择 padding 类型
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))  # 反射填充
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))  # 复制填充
        elif padding_type == "zero":
            p = 1  # 零填充，后续在 Conv2d 中设置 padding=p
        else:
            raise NotImplementedError(
                f"padding [{padding_type}] is not implemented"
            )

        # 第一层卷积：Conv → Norm → ReLU（移除 inplace=True）
        conv_block.extend([
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ])
        # 可选 dropout 层
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        # 第二层 padding（与第一层一致）
        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1

        # 第二层卷积：Conv → Norm（无激活，残差连接后激活）
        conv_block.extend([
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ])

        return nn.Sequential(*conv_block)

    # Jittor 用 execute 替代 forward
    def execute(self, x):
        # 残差连接：输入 + 卷积块输出
        return x + self.conv_block(x)


class LocNet(nn.Module):
    """定位网络（用于 STN 空间变换，Jittor 适配）"""
    def __init__(
        self,
        input_nc,
        nc=32,
        n_blocks=3,
        use_dropout=False,
        padding_type="zero",
        image_size=32,
    ):
        super().__init__()

        backbone = []
        # 初始卷积层：下采样（步长2）
        backbone.extend([
            nn.Conv2d(
                input_nc, nc, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(nc),
            nn.ReLU()  # 移除 inplace=True
        ])
        # 堆叠残差块 + 最大池化（下采样）
        for _ in range(n_blocks):
            backbone.append(
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=nn.BatchNorm2d,
                    use_dropout=use_dropout,
                    use_bias=False,
                )
            )
            backbone.append(nn.MaxPool2d(2, stride=2))  # Jittor MaxPool2d 与 PyTorch 兼容
        self.backbone = nn.Sequential(*backbone)

        # 计算下采样后的图像尺寸
        reduced_imsize = int(image_size * (0.5 ** (n_blocks + 1)))
        # 全连接层：输出 2x2 的变换矩阵参数
        self.fc_loc = nn.Linear(nc * (reduced_imsize ** 2), 2 * 2)

    # Jittor 用 execute 替代 forward
    def execute(self, x):
        x = self.backbone(x)
        # 展平特征图（batch × 特征数）
        x = x.view(x.size(0), -1)
        # 预测变换参数并通过 tanh 归一化
        x = self.fc_loc(x)
        x = jt.tanh(x)  # 替换 torch.tanh
        # 调整形状为 (batch, 2, 2)
        x = x.view(-1, 2, 2)
        # 构建 2x3 的变换矩阵（前两列是预测参数，第三列初始为 0）
        # Jittor 用 jt.zeros 替代 torch.Tensor.data.new_zeros
        theta = jt.zeros(x.size(0), 2, 3, dtype=x.dtype)
        theta[:, :, :2] = x
        return theta


class FCN(nn.Module):
    """全卷积网络（含 STN 空间变换、全局上下文融合，Jittor 适配）"""
    def __init__(
        self,
        input_nc,
        output_nc,
        nc=32,
        n_blocks=3,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        padding_type="reflect",
        gctx=True,
        stn=False,
        image_size=32,
    ):
        super().__init__()

        backbone = []
        p = 0  # 零填充的 padding 大小
        # 初始 padding 层
        if padding_type == "reflect":
            backbone.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            backbone.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        # 初始卷积层：Conv → Norm → ReLU
        backbone.extend([
            nn.Conv2d(
                input_nc, nc, kernel_size=3, stride=1, padding=p, bias=False
            ),
            norm_layer(nc),
            nn.ReLU()  # 移除 inplace=True
        ])

        # 堆叠残差块（无下采样，保持特征图尺寸）
        for _ in range(n_blocks):
            backbone.append(
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=False,
                )
            )
        self.backbone = nn.Sequential(*backbone)

        # 全局上下文融合层（可选）
        self.gctx_fusion = None
        if gctx:
            self.gctx_fusion = nn.Sequential(
                nn.Conv2d(
                    2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False
                ),
                norm_layer(nc),
                nn.ReLU()
            )

        # 回归层：输出扰动（通过 Tanh 归一化到 [-1, 1]）
        self.regress = nn.Sequential(
            nn.Conv2d(
                nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Tanh()
        )

        # 定位网络（STN，可选）
        self.locnet = None
        if stn:
            self.locnet = LocNet(
                input_nc, nc=nc, n_blocks=n_blocks, image_size=image_size
            )

    def init_loc_layer(self):
        """初始化 STN 的变换矩阵为单位矩阵（Jittor 适配）"""
        if self.locnet is not None:
            # 权重清零，偏置初始化为 [1,0,0,1]（单位变换）
            self.locnet.fc_loc.weight.zero_()  # 移除 .data
            # Jittor 用 jt.array 替代 torch.tensor，用 copy_ 赋值
            self.locnet.fc_loc.bias.copy_(
                jt.array([1, 0, 0, 1], dtype=jt.float32)
            )

    def stn(self, x):
        """STN 空间变换（Jittor 适配）"""
        theta = self.locnet(x)
        # 生成 affine grid（Jittor 用 nn.affine_grid 替代 F.affine_grid）
        grid = nn.affine_grid(theta, x.size())
        # 网格采样（Jittor 用 nn.grid_sample 替代 F.grid_sample）
        x_transformed = nn.grid_sample(x, grid)
        return x_transformed, theta

    # Jittor 用 execute 替代 forward
    def execute(self, x, lmda=1.0, return_p=False, return_stn_output=False):
        """
        Args:
            x (jt.Var): 输入批量张量
            lmda (float): 扰动乘数
            return_p (bool): 是否返回扰动 p
            return_stn_output (bool): 是否返回 STN 变换后的输入
        """
        theta = None
        # 若启用 STN，先进行空间变换
        if self.locnet is not None:
            x, theta = self.stn(x)
        input_stn = x  # 保存 STN 变换后的输入

        #  backbone 特征提取
        x = self.backbone(x)
        # 若启用全局上下文融合
        if self.gctx_fusion is not None:
            # 全局平均池化获取上下文特征
            c = nn.adaptive_avg_pool2d(x, (1, 1))  # 替换 F.adaptive_avg_pool2d
            # 扩展到与原特征图尺寸一致
            c = c.expand_as(x)
            # 拼接特征与上下文，再通过融合层
            x = jt.cat([x, c], dim=1)  # 替换 torch.cat
            x = self.gctx_fusion(x)

        # 预测扰动 p
        p = self.regress(x)
        # 生成带扰动的输出（输入 + lmda * 扰动）
        x_p = input_stn + lmda * p

        # 根据返回参数选择输出
        if return_stn_output:
            return x_p, p, input_stn
        if return_p:
            return x_p, p
        return x_p


@NETWORK_REGISTRY.register()
def fcn_3x32_gctx(**kwargs):
    """注册 3通道输入、32通道基础卷积、全局上下文融合的 FCN"""
    norm_layer = get_norm_layer(norm_type="instance")
    net = FCN(3, 3, nc=32, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type="normal", gain=0.02)
    return net


@NETWORK_REGISTRY.register()
def fcn_3x64_gctx(**kwargs):
    """注册 3通道输入、64通道基础卷积、全局上下文融合的 FCN"""
    norm_layer = get_norm_layer(norm_type="instance")
    net = FCN(3, 3, nc=64, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type="normal", gain=0.02)
    return net


@NETWORK_REGISTRY.register()
def fcn_3x32_gctx_stn(image_size=32, **kwargs):
    """注册带 STN 的 3x32 全局上下文融合 FCN"""
    norm_layer = get_norm_layer(norm_type="instance")
    net = FCN(
        3,
        3,
        nc=32,
        n_blocks=3,
        norm_layer=norm_layer,
        stn=True,
        image_size=image_size
    )
    init_network_weights(net, init_type="normal", gain=0.02)
    net.init_loc_layer()  # 初始化 STN 为单位变换
    return net


@NETWORK_REGISTRY.register()
def fcn_3x64_gctx_stn(image_size=224, **kwargs):
    """注册带 STN 的 3x64 全局上下文融合 FCN"""
    norm_layer = get_norm_layer(norm_type="instance")
    net = FCN(
        3,
        3,
        nc=64,
        n_blocks=3,
        norm_layer=norm_layer,
        stn=True,
        image_size=image_size
    )
    init_network_weights(net, init_type="normal", gain=0.02)
    net.init_loc_layer()  # 初始化 STN 为单位变换
    return net