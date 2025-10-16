"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import jittor as jt
import jittor.nn as nn

# from .radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None):
    """A function wrapper for building an optimizer."""
    print("\n===== 开始构建优化器 =====")
    print(f"优化器类型: {optim_cfg.NAME}")
    print(f"学习率 (LR): {optim_cfg.LR}")
    print(f"权重衰减 (WEIGHT_DECAY): {optim_cfg.WEIGHT_DECAY}")


    optim = optim_cfg.NAME.lower() 
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        print(f"错误：优化器类型 {optim} 不在支持列表中！")
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )
    if param_groups is None:
        print("参数组 (param_groups) 为None，开始自动构建参数组...")
        if staged_lr:
            print(f"启用分段学习率 (staged_lr)，新层: {new_layers}")
            if not isinstance(model, nn.Module):
                print("错误：staged_lr为True时，model必须是nn.Module实例！")
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )

            if isinstance(new_layers, str):
                new_layers = [new_layers] if new_layers is not None else []

            base_params = []
            new_params = []
            for name, module in model.named_children():
                if name in new_layers:
                    new_params.extend([p for p in module.parameters()])
                    print(f"新层参数: {name}，参数数量: {len(list(module.parameters()))}")
                else:
                    base_params.extend([p for p in module.parameters()])
                    print(f"基础层参数: {name}，参数数量: {len(list(module.parameters()))}")

            param_groups = [
                {
                    "params": base_params,
                    "lr": lr * base_lr_mult
                },
                {
                    "params": new_params
                },
            ]
            # 调试输出：参数组详情
            print(f"构建完成的参数组数量: {len(param_groups)}")
            print(f"基础参数组大小: {len(base_params)}，学习率: {lr * base_lr_mult}")
            print(f"新参数组大小: {len(new_params)}，学习率: {lr}")
        else:
            print("不启用分段学习率，使用模型所有参数...")
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model
            # 调试输出：参数总数
            param_count = len(list(param_groups)) if not isinstance(param_groups, dict) else len(param_groups["params"])
            print(f"参数组总参数数量: {param_count}")
    else:
        print(f"使用外部传入的参数组，参数组数量: {len(param_groups)}")

    # --------------------------
    # 调试输出：构建优化器
    # --------------------------
    optimizer = None
    print(f"\n开始初始化优化器: {optim}")

    if optim == "sgd":
        print(f"SGD参数 - 学习率: {lr}, 动量: {momentum}, 权重衰减: {weight_decay}")
        optimizer = jt.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )
    else:
        print(f"错误：优化器 {optim} 未实现！")
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    if optimizer is None:
        print("错误：优化器初始化失败！")
    else:
        print(f"优化器构建成功: {type(optimizer).__name__}")

    print("===== 优化器构建结束 =====\n")
    return optimizer
