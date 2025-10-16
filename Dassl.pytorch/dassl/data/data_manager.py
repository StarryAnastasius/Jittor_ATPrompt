import jittor as jt
import jittor.transform as jt_transform
from jittor.dataset import Dataset
from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler  
from .transforms import build_transform  
from .transforms import INTERP_MAPPING 
import json


def build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None,
        class_names=None
):
    # 若dataset_wrapper为None，显式导入并赋值为DatasetWrapper
    if dataset_wrapper is None:
        from dassl.data.data_manager import DatasetWrapper
        dataset_wrapper = DatasetWrapper

    # 1. 构建采样器
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    # 2. 创建DatasetWrapper实例（此时已预定义属性）
    dataset = dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train)
    

    # 动态计算正确的 total_len（batch 数 = 采样器总样本数 ÷ batch_size）
    sampler_total_samples = len(sampler)  # 获取采样器的总样本数（如 800）
    correct_total_len = sampler_total_samples // batch_size  # 计算实际 batch 数
    # 设置正确的 total_len
    dataset.set_attrs(total_len=correct_total_len)
    
    # 3. 原有属性设置
    dataset.set_attrs(
        batch_size=batch_size,    
        sampler=sampler,          
        num_workers=cfg.DATALOADER.NUM_WORKERS,  
        drop_last=is_train and len(data_source) >= batch_size,  
    )
    
    return dataset

class DataManager:
    def __init__(
            self,
            cfg,
            custom_tfm_train=None,
            custom_tfm_test=None,
            dataset_wrapper=None
    ):
        

        # 加载数据集
        dataset = build_dataset(cfg)

        # 构建变换部分添加打印
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            tfm_train = custom_tfm_train
        # 测试集预处理同理
        if custom_tfm_test is None:
    
            tfm_test = build_transform(cfg, is_train=False)
        else:
            tfm_test = custom_tfm_test

        # 构建带标签训练集加载器
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
            class_names=dataset.classnames
        )

        # 构建无标签训练集加载器
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # 构建验证集加载器
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # 构建测试集加载器
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # 数据集元信息
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # 数据集与加载器
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader


        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print("\n===== 数据集原始长度信息 =====")
        print(f"测试集原始长度 (len(dataset.test)): {len(self.dataset.test)}")
        print(f"训练集原始长度 (len(dataset.train_x)): {len(self.dataset.train_x)}")
        print(f"类别数: {self.num_classes}")
        print(f"SUBSAMPLE_CLASSES 配置: {cfg.DATASET.SUBSAMPLE_CLASSES}") 
        print("=============================\n")
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])



class DatasetWrapper(Dataset):  # 继承Jittor的Dataset基类

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        super().__init__()

        # -------------------------- 原有预定义属性--------------------------
        self.batch_size = None
        self.sampler = None
        self.num_workers = None
        self.drop_last = None
        self.total_len = 0 
        # -------------------------- 原有数据相关属性--------------------------
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform 
        self.is_train = is_train
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0


        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times because transform is None".format(self.k_tfm)
            )


        interp_mode = INTERP_MAPPING[cfg.INPUT.INTERPOLATION][0]
        to_tensor = []
        to_tensor.append(jt_transform.Resize(cfg.INPUT.SIZE, mode=interp_mode))
        to_tensor.append(jt_transform.ToTensor())
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = jt_transform.ImageNormalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor.append(normalize)
        self.to_tensor = jt_transform.Compose(to_tensor)

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img = tfm(img0)
            img_list.append(img)

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img