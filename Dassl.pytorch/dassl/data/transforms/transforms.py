import numpy as np
import random
import jittor as jt
from jittor import transform
from PIL import Image  # 导入PIL用于插值模式映射

from jittor.transform import (
    Resize, Compose, ToTensor, ImageNormalize, CenterCrop, RandomCrop, 
    RandomResizedCrop,
    RandomHorizontalFlip
)

# 导入自定义增强策略（需确保内部已适配Jittor）
from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch

AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "instance_norm",
    "random_crop",
    "random_translation",
    "center_crop",
    "cutout",
    "imagenet_policy",
    "cifar10_policy",
    "svhn_policy",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "gaussian_noise",
    "colorjitter",
    "randomgrayscale",
    # "gaussian_blur" （注：提供的API中无此变换，已移除）
]

# 插值模式映射：配置字符串 -> (Resize的mode(int), RandomResizedCrop的interpolation(PIL对象))
INTERP_MAPPING = {
    "bilinear": (Image.BILINEAR, Image.BILINEAR),
    "bicubic": (Image.BICUBIC, Image.BICUBIC),
    "nearest": (Image.NEAREST, Image.NEAREST),
}


class Random2DTranslation:
    """图像随机平移增强：放大后随机裁剪（基于jittor.transform.functional）"""
    def __init__(self, height, width, p=0.5, interpolation="bilinear"):
        self.height = height
        self.width = width
        self.p = p
        # 将插值字符串转为PIL插值对象（供transform.resize使用）
        self.interpolation = getattr(Image, interpolation.upper())

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return transform.resize(img, [self.height, self.width], self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = transform.resize(img, [new_height, new_width], self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        return transform.crop(resized_img, y1, x1, self.height, self.width)


class InstanceNormalization:
    """实例归一化（基于Jittor Var的张量操作）"""
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img - mean) / (std + self.eps)


class Cutout:
    """随机遮挡增强（numpy + Jittor Var）"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = jt.array(mask).expand_as(img)
        return img * mask


class GaussianNoise:
    """高斯噪声增强（Jittor randn生成噪声）"""
    def __init__(self, mean=0, std=0.15, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = jt.randn(img.size()) * self.std + self.mean
        return img + noise


def build_transform(cfg, is_train=True, choices=None):
    """构建图像变换流水线"""
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS
    for choice in choices:
        assert choice in AVAI_CHOICES, f"Unsupported transform: {choice}"

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"
    normalize = ImageNormalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    return _build_transform_train(cfg, choices, target_size, normalize) if is_train \
        else _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    """训练阶段变换流水线"""
    print("Building transform_train")
    tfm_train = []

    # 从配置中获取插值模式，并拆分给Resize和RandomResizedCrop的参数
    interp_str = cfg.INPUT.INTERPOLATION
    resize_mode, rrcrop_interp = INTERP_MAPPING[interp_str]
    input_size = cfg.INPUT.SIZE

    # 基础Resize（无随机裁剪时生效）
    if "random_crop" not in choices and "random_resized_crop" not in choices:
        print(f"+ resize to {target_size}")
        tfm_train.append(Resize(input_size, mode=resize_mode))  # Resize使用mode参数

    # 随机平移（自定义类）
    if "random_translation" in choices:
        print("+ random translation")
        tfm_train.append(Random2DTranslation(input_size[0], input_size[1], interpolation=interp_str))

    # 随机裁剪（RandomCrop）
    if "random_crop" in choices:
        print(f"+ random crop (padding={cfg.INPUT.CROP_PADDING})")
        tfm_train.append(RandomCrop(input_size, padding=cfg.INPUT.CROP_PADDING))

    # 随机Resize裁剪（RandomResizedCrop）
    if "random_resized_crop" in choices:
        print(f"+ random resized crop (size={input_size}, scale={cfg.INPUT.RRCROP_SCALE})")
        tfm_train.append(
            RandomResizedCrop(
                input_size, 
                scale=cfg.INPUT.RRCROP_SCALE, 
                interpolation=rrcrop_interp  # RandomResizedCrop使用interpolation参数
            )
        )

    # 随机水平翻转（RandomHorizontalFlip）
    if "random_flip" in choices:
        print("+ random flip")
        tfm_train.append(RandomHorizontalFlip())

    # 自动增强策略（自定义类）
    if "imagenet_policy" in choices:
        tfm_train.append(ImageNetPolicy())
    if "cifar10_policy" in choices:
        tfm_train.append(CIFAR10Policy())
    if "svhn_policy" in choices:
        tfm_train.append(SVHNPolicy())

    # RandAugment系列（自定义类）
    if "randaugment" in choices:
        tfm_train.append(RandAugment(cfg.INPUT.RANDAUGMENT_N, cfg.INPUT.RANDAUGMENT_M))
    if "randaugment_fixmatch" in choices:
        tfm_train.append(RandAugmentFixMatch(cfg.INPUT.RANDAUGMENT_N))
    if "randaugment2" in choices:
        tfm_train.append(RandAugment2(cfg.INPUT.RANDAUGMENT_N))

    # 转为Jittor Var
    tfm_train.append(ToTensor())

    # Cutout（自定义类）
    if "cutout" in choices:
        print(f"+ cutout (n_holes={cfg.INPUT.CUTOUT_N}, length={cfg.INPUT.CUTOUT_LEN})")
        tfm_train.append(Cutout(cfg.INPUT.CUTOUT_N, cfg.INPUT.CUTOUT_LEN))

    # 归一化（ImageNormalize）
    if "normalize" in choices:
        print(f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})")
        tfm_train.append(normalize)

    # 高斯噪声（自定义类）
    if "gaussian_noise" in choices:
        print(f"+ gaussian noise (mean={cfg.INPUT.GN_MEAN}, std={cfg.INPUT.GN_STD})")
        tfm_train.append(GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD))

    # 实例归一化（自定义类）
    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_train.append(InstanceNormalization())

    return Compose(tfm_train)


def _build_transform_test(cfg, choices, target_size, normalize):
    """测试阶段变换流水线"""
    print("Building transform_test")
    tfm_test = []

    # 从配置中获取插值模式，Resize使用mode参数
    interp_str = cfg.INPUT.INTERPOLATION
    resize_mode, _ = INTERP_MAPPING[interp_str]
    input_size = cfg.INPUT.SIZE

    # Resize短边到目标尺寸最大值
    tfm_test.append(Resize(max(input_size), mode=resize_mode))
    # 中心裁剪
    tfm_test.append(CenterCrop(input_size))
    # 转为Jittor Var
    tfm_test.append(ToTensor())

    # 归一化
    if "normalize" in choices:
        tfm_test.append(normalize)
    # 实例归一化
    if "instance_norm" in choices:
        tfm_test.append(InstanceNormalization())

    return Compose(tfm_test)