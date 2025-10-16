import os
import torch
import pickle
import numpy as np
from clip import clip  # 确保已安装PyTorch版本的CLIP

# --------------------------
# 配置绝对路径（根据你的环境修改）
# --------------------------
# atprompt文件夹的绝对路径（脚本所在目录）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# 转换后的权重保存目录（会自动创建）
WEIGHT_SAVE_DIR = os.path.join(BASE_DIR, "clip_converted_weights")
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)

def convert_clip_weight(backbone_name):
    """
    将指定CLIP模型的PyTorch权重转换为Jittor兼容格式
    backbone_name: 模型名称，如"ViT-B/32"、"RN50"等
    """
    # 1. 下载PyTorch格式的CLIP权重（使用CLIP自带的下载函数）
    print(f"开始下载/加载 {backbone_name} 的PyTorch权重...")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)  # 下载到~/.cache/clip目录
    
    # 2. 关键修复：加载TorchScript模型并提取state_dict
    print(f"加载PyTorch模型: {model_path}")
    # 对于TorchScript格式的模型，需要用torch.jit.load加载完整模型
    model = torch.jit.load(model_path, map_location="cpu")
    # 从模型中提取权重字典（state_dict）
    state_dict_torch = model.state_dict()
    
    # 3. 将PyTorch张量转换为numpy数组（剥离PyTorch依赖）
    state_dict_np = {}
    for key, value in state_dict_torch.items():
        if isinstance(value, torch.Tensor):
            state_dict_np[key] = value.numpy()  # 转为numpy数组
        else:
            state_dict_np[key] = value  # 非张量值直接保留
    
    # 4. 用绝对路径保存转换后的权重
    safe_backbone_name = backbone_name.replace("/", "-")
    save_path = os.path.join(WEIGHT_SAVE_DIR, f"clip_{safe_backbone_name}_jittor.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(state_dict_np, f)
    
    print(f"转换完成，保存至绝对路径: {save_path}")
    return save_path

if __name__ == "__main__":
    # 与配置文件中 cfg.MODEL.BACKBONE.NAME 保持一致
    BACKBONE_NAME = "ViT-B/16"  # 例如："RN50"、"ViT-B/32"等
    convert_clip_weight(BACKBONE_NAME)
