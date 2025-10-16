# import sys
# import pprint as pp
# import os
# import os.path as osp
# import pickle
# import tarfile
# import urllib.request
# from PIL import Image
# import numpy as np
#
# from dassl.utils import mkdir_if_missing
#
# # 标签映射保持不变
# cifar_label2name = {
#     0: "airplane",
#     1: "car",
#     2: "bird",
#     3: "cat",
#     4: "deer",
#     5: "dog",
#     6: "frog",  # 待丢弃类别
#     7: "horse",
#     8: "ship",
#     9: "truck",
# }
#
# stl_label2name = {
#     0: "airplane",
#     1: "bird",
#     2: "car",
#     3: "cat",
#     4: "deer",
#     5: "dog",
#     6: "horse",
#     7: "monkey",  # 待丢弃类别
#     8: "ship",
#     9: "truck",
# }
#
# new_name2label = {
#     "airplane": 0,
#     "bird": 1,
#     "car": 2,
#     "cat": 3,
#     "deer": 4,
#     "dog": 5,
#     "horse": 6,
#     "ship": 7,
#     "truck": 8,
# }
#
#
# def download_url(url, save_path):
#     """纯Python下载文件，替代torchvision的download功能"""
#     if not osp.exists(save_path):
#         print(f"下载 {url} 到 {save_path}...")
#         with urllib.request.urlopen(url) as response, open(save_path, "wb") as out_file:
#             out_file.write(response.read())
#
#
# def load_cifar10(root, train=True):
#     """手动解析CIFAR10数据集（替代torchvision.datasets.CIFAR10）"""
#     base_dir = osp.join(root, "cifar-10-batches-py")
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     tar_path = osp.join(root, "cifar-10-python.tar.gz")
#
#     # 下载并解压
#     if not osp.exists(base_dir):
#         download_url(url, tar_path)
#         with tarfile.open(tar_path, "r:gz") as tar:
#             tar.extractall(root)
#
#     # 读取数据
#     images = []
#     labels = []
#     if train:
#         # 训练集包含5个批次
#         for i in range(1, 6):
#             batch_path = osp.join(base_dir, f"data_batch_{i}")
#             with open(batch_path, "rb") as f:
#                 batch = pickle.load(f, encoding="latin1")  # CIFAR10用latin1编码
#             images.append(batch["data"])
#             labels.extend(batch["labels"])
#     else:
#         # 测试集
#         batch_path = osp.join(base_dir, "test_batch")
#         with open(batch_path, "rb") as f:
#             batch = pickle.load(f, encoding="latin1")
#         images.append(batch["data"])
#         labels.extend(batch["labels"])
#
#     # 转换为PIL图像（CIFAR10数据格式：3x32x32，存储为flattened数组）
#     images = np.concatenate(images).reshape(-1, 3, 32, 32)
#     images = images.transpose(0, 2, 3, 1)  # (N, 3, 32, 32) → (N, 32, 32, 3)
#     dataset = []
#     for img_arr, label in zip(images, labels):
#         img = Image.fromarray(img_arr.astype(np.uint8))  # 转换为PIL图像
#         dataset.append((img, label))
#     return dataset
#
#
# def load_stl10(root, split="train"):
#     """手动解析STL10数据集（替代torchvision.datasets.STL10）"""
#     base_dir = osp.join(root, "stl10_binary")
#     url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
#     tar_path = osp.join(root, "stl10_binary.tar.gz")
#
#     # 下载并解压
#     if not osp.exists(base_dir):
#         download_url(url, tar_path)
#         with tarfile.open(tar_path, "r:gz") as tar:
#             tar.extractall(root)
#
#     # 读取数据（STL10为二进制文件，格式：[label][3*96*96字节图像数据]）
#     split_map = {
#         "train": "train_X.bin",
#         "test": "test_X.bin",
#         "train_labels": "train_y.bin",
#         "test_labels": "test_y.bin"
#     }
#     data_path = osp.join(base_dir, split_map[split])
#     labels_path = osp.join(base_dir, split_map[f"{split}_labels"])
#
#     # 读取标签（注意：STL10标签从1开始，需减1转为0基）
#     with open(labels_path, "rb") as f:
#         labels = np.fromfile(f, dtype=np.uint8) - 1  # 转为0基索引
#
#     # 读取图像数据（3x96x96，RGB格式）
#     with open(data_path, "rb") as f:
#         images = np.fromfile(f, dtype=np.uint8).reshape(-1, 3, 96, 96)
#     images = images.transpose(0, 2, 3, 1)  # (N, 3, 96, 96) → (N, 96, 96, 3)
#
#     # 转换为PIL图像
#     dataset = []
#     for img_arr, label in zip(images, labels):
#         img = Image.fromarray(img_arr.astype(np.uint8))
#         dataset.append((img, label))
#     return dataset
#
#
# def extract_and_save_image(dataset, save_dir, discard, label2name):
#     """提取图像并保存（逻辑不变，仅依赖PIL和os）"""
#     if osp.exists(save_dir):
#         print(f'文件夹 "{save_dir}" 已存在')
#         return
#
#     print(f'正在将图像提取到 "{save_dir}" ...')
#     mkdir_if_missing(save_dir)
#
#     for i in range(len(dataset)):
#         img, label = dataset[i]
#         if label == discard:
#             continue
#         class_name = label2name[label]
#         label_new = new_name2label[class_name]
#         class_dir = osp.join(
#             save_dir,
#             f"{str(label_new).zfill(3)}_{class_name}"
#         )
#         mkdir_if_missing(class_dir)
#         impath = osp.join(class_dir, f"{str(i + 1).zfill(5)}.jpg")
#         img.save(impath)
#
#
# def download_and_prepare(name, root, discarded_label, label2name):
#     """下载并预处理数据集（完全去Torch化）"""
#     print(f"数据集: {name}")
#     print(f"根目录: {root}")
#     print("原始标签映射:")
#     pp.pprint(label2name)
#     print(f"需丢弃的标签: {discarded_label}")
#     print("新标签映射:")
#     pp.pprint(new_name2label)
#
#     # 加载数据集（使用自定义的load函数，不依赖torchvision）
#     if name == "cifar":
#         train = load_cifar10(root, train=True)
#         test = load_cifar10(root, train=False)
#     else:
#         train = load_stl10(root, split="train")
#         test = load_stl10(root, split="test")
#
#     # 保存路径
#     train_dir = osp.join(root, name, "train")
#     test_dir = osp.join(root, name, "test")
#
#     # 提取并保存
#     extract_and_save_image(train, train_dir, discarded_label, label2name)
#     extract_and_save_image(test, test_dir, discarded_label, label2name)
#
#
# if __name__ == "__main__":
#     # 用法：python prepare_datasets_no_torch.py <数据集根目录>
#     download_and_prepare("cifar", sys.argv[1], 6, cifar_label2name)  # 丢弃frog
#     download_and_prepare("stl", sys.argv[1], 7, stl_label2name)  # 丢弃monkey



import sys
import pprint as pp
import os.path as osp
from torchvision.datasets import STL10, CIFAR10

from dassl.utils import mkdir_if_missing

cifar_label2name = {
    0: "airplane",
    1: "car",  # the original name was 'automobile'
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",  # conflict class
    7: "horse",
    8: "ship",
    9: "truck",
}

stl_label2name = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",  # conflict class
    8: "ship",
    9: "truck",
}

new_name2label = {
    "airplane": 0,
    "bird": 1,
    "car": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "horse": 6,
    "ship": 7,
    "truck": 8,
}


def extract_and_save_image(dataset, save_dir, discard, label2name):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == discard:
            continue
        class_name = label2name[label]
        label_new = new_name2label[class_name]
        class_dir = osp.join(
            save_dir,
            str(label_new).zfill(3) + "_" + class_name
        )
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)


def download_and_prepare(name, root, discarded_label, label2name):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))
    print("Old labels:")
    pp.pprint(label2name)
    print("Discarded label: {}".format(discarded_label))
    print("New labels:")
    pp.pprint(new_name2label)

    if name == "cifar":
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False)
    else:
        train = STL10(root, split="train", download=True)
        test = STL10(root, split="test")

    train_dir = osp.join(root, name, "train")
    test_dir = osp.join(root, name, "test")

    extract_and_save_image(train, train_dir, discarded_label, label2name)
    extract_and_save_image(test, test_dir, discarded_label, label2name)


if __name__ == "__main__":
    download_and_prepare("cifar", sys.argv[1], 6, cifar_label2name)
    download_and_prepare("stl", sys.argv[1], 7, stl_label2name)
