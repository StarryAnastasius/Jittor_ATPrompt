import functools
import jittor as jt
import jittor.nn as nn

from .build import HEAD_REGISTRY


class MLP(nn.Module):

    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        activation="relu",
        bn=True,
        dropout=0.0,
    ):
        super().__init__()
        # 处理隐藏层参数：若输入为整数，转为单元素列表
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0, "Hidden layers cannot be empty"
        self.out_features = hidden_layers[-1]  # 输出特征维度为最后一层隐藏层维度

        mlp = []

        # 配置激活函数（Jittor 不支持 inplace=True，移除该参数）
        if activation == "relu":
            act_fn = functools.partial(nn.ReLU)  # 替换 torch.nn.ReLU
        elif activation == "leaky_relu":
            act_fn = functools.partial(nn.LeakyReLU)  # 替换 torch.nn.LeakyReLU
        else:
            raise NotImplementedError(f"Activation '{activation}' is not supported")

        # 构建 MLP 层序列（Linear → BN（可选）→ 激活 → Dropout（可选））
        for hidden_dim in hidden_layers:
            # 全连接层（Jittor 与 PyTorch 参数兼容）
            mlp.append(nn.Linear(in_features, hidden_dim))
            # 批量归一化（1D，适配全连接层输出）
            if bn:
                mlp.append(nn.BatchNorm1d(hidden_dim))
            # 激活函数
            mlp.append(act_fn())
            #  dropout 层（仅当 dropout > 0 时添加）
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            # 更新下一层输入特征维度
            in_features = hidden_dim

        self.mlp = nn.Sequential(*mlp)

    # Jittor 用 execute 替代 forward 作为前向传播入口
    def execute(self, x):
        return self.mlp(x)


@HEAD_REGISTRY.register()
def mlp(**kwargs):
    """注册 MLP 头，接口与原 PyTorch 版本完全一致"""
    return MLP(** kwargs)