"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
Adapted for Jittor framework.
"""
import jittor as jt
import jittor.nn as nn
from jittor import save, load
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import shutil

from .tools import mkdir_if_missing

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
    "open_all_layers",
    "open_specified_layers",
    "count_num_param",
    "load_pretrained_weights",
    "init_network_weights",
]


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    mkdir_if_missing(save_dir)

    if "state_dict" in state:
        pass

    # 保存模型
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    save(state, fpath)
    print(f"Checkpoint saved to {fpath}")

    # 记录当前模型文件名
    checkpoint_file = osp.join(save_dir, "checkpoint")
    with open(checkpoint_file, "w+") as checkpoint:
        checkpoint.write("{}\n".format(osp.basename(fpath)))

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))


def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    try:
        checkpoint = load(fpath)
    except Exception as e:
        print('Unable to load checkpoint from "{}": {}'.format(fpath, e))
        raise

    return checkpoint


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    """修复：兼容Jittor调度器无load_state_dict的问题"""
    with open(osp.join(fdir, "checkpoint"), "r") as checkpoint_file:
        model_name = checkpoint_file.readlines()[0].strip("\n")
        fpath = osp.join(fdir, model_name)

    print(f'Loading checkpoint from "{fpath}"')
    checkpoint = load_checkpoint(fpath)

    # 1. 加载模型参数
    model.load_parameters(checkpoint["state_dict"])
    print("Loaded model weights")

    # 2. 加载优化器
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state")
        except Exception as e:
            print(f"Warning: 优化器状态加载失败，使用默认参数: {e}")

    # 3. 加载调度器
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler_ckpt = checkpoint["scheduler"]
        try:
            scheduler.load_state_dict(scheduler_ckpt)
            print("Loaded scheduler state via load_state_dict")
        except AttributeError:
            print("Scheduler has no load_state_dict, manually setting last_epoch")
            if "last_epoch" in scheduler_ckpt:
                scheduler.last_epoch = scheduler_ckpt["last_epoch"]
            elif "successor_state" in scheduler_ckpt:
                successor_last_epoch = scheduler_ckpt["successor_state"].get("last_epoch", -1)
                scheduler.successor.last_epoch = successor_last_epoch
        except Exception as e:
            print(f"Scheduler load failed: {e}，手动恢复last_epoch")
            if hasattr(scheduler, "last_epoch"):
                scheduler.last_epoch = scheduler_ckpt.get("last_epoch", -1)

    # 获取起始epoch
    start_epoch = checkpoint["epoch"]
    print(f"Previous epoch: {start_epoch}")

    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100,
):
    if linear_decay:
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1.0-frac_done) * base_lr
    else:
        lr = base_lr * (gamma **(epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), f"{layer} is not an attribute"

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model=None, params=None):
    if model is not None:
        return sum(p.size for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].size
            else:
                s += p.size
        return s

    raise ValueError("model and params must provide at least one.")


def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model.load_parameters(new_state_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )


def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            weight = m.weight
            if init_type == "normal":
                nn.init.normal_(weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_gauss_(weight, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(weight, a=0, mode="fan_in")
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif classname.find("BatchNorm") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.apply(_init_func)