import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import sys
import os

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_backbone, build_head
from dassl.evaluation import build_evaluator


class SimpleNet(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = jt.nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def execute(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.rank = 0
        self.world_size = 1
        self.distributed = False

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before super().__init__()")
        if self.__dict__.get("_optims") is None:
            raise AttributeError("Cannot assign optim before super().__init__()")
        if self.__dict__.get("_scheds") is None:
            raise AttributeError("Cannot assign sched before super().__init__()")

        assert name not in self._models, "重复的模型名称"
        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        if self.rank != 0 and self.distributed:
            return
        names = self.get_model_names()
        for name in names:
            model_dict = self._models[name].state_dict()
            # 1. 优化器状态
            optim_dict = self._optims[name].state_dict() if self._optims[name] else None
            
            # 2. 调度器状态
            sched = self._scheds[name]
            sched_dict = None
            if sched is not None:
                if hasattr(sched, "state_dict"):
                    sched_dict = sched.state_dict()
                else:
                    sched_dict = {"last_epoch": sched.last_epoch}
                    print(f"保存调度器 {name} 的手动状态: {sched_dict}")

            # 3. 保存检查点
            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )


    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = any(not osp.exists(osp.join(directory, name)) for name in names)

        if file_missing:
            print("未找到 checkpoint，从头训练")
            return 0  # 无检查点时从 epoch 0 开始

        print(f"找到 checkpoint 在 {directory}（将恢复训练）")
        start_epoch = 0  # 初始化起始 epoch
        for name in names:
            path = osp.join(directory, name)
            # 接收 resume_from_checkpoint 返回的 start_epoch
            epoch = resume_from_checkpoint(path, self._models[name], self._optims[name], self._scheds[name])
            if epoch > start_epoch:
                start_epoch = epoch  # 取最大的 epoch 作为起始点
        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("未指定预训练模型路径，跳过加载")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f"模型不存在于 {model_path}")

            checkpoint = load_checkpoint(model_path)
            self._models[name].load_state_dict(checkpoint["state_dict"])
            if self.rank == 0:
                print(f"加载 {model_path} 到 {name}（epoch={checkpoint['epoch']}, val_result={checkpoint['val_result']:.2f}）")

        
        conv1_weight = self.model.image_encoder.conv1.weight  # 对应ModifiedResNet的conv1

        
    
    
    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)
        for name in names:
            self._models[name].train() if mode == "train" else self._models[name].eval()

    def update_lr(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._scheds[name]:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not jt.misc.isfinite(loss).all():
            raise FloatingPointError("损失为无穷或NaN！")

    def train(self, start_epoch, max_epoch):
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        sys.stdout.flush()
        sys.stdout.close()
        os._exit(0)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name]:
                self._optims[name].zero_grad()

    def model_backward(self, loss, names=None):
        """修复：通过优化器调用backward，而非直接对loss调用"""
        self.detect_anomaly(loss)
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name]:
                self._optims[name].backward(loss)

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name]:
                self._optims[name].step()  

    def model_backward_and_update(self, loss, names=None):
        """修复：使用Jittor优化器的step(loss)，一步完成zero_grad+backward+step"""
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name]:
                self._optims[name].step(loss)

class SimpleTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        self.build_model() 
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        pass

    def build_data_loader(self, custom_tfm_train=None, custom_tfm_test=None):
        dm = DataManager(
            self.cfg,
            custom_tfm_train=custom_tfm_train,
            custom_tfm_test=custom_tfm_test
        )
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

        from collections import defaultdict
        try:
            # DatasetWrapper → data_source 是 ListDataset → data_list 是原始样本列表
            train_samples = self.train_loader_x.dataset.data_source.data_list
            # 统计每个类别的样本数
            label_count = defaultdict(int)
            for sample in train_samples:
                label_count[sample.label] += 1
            # 计算总样本数和类别数
            total_samples = len(train_samples)
            actual_num_classes = len(label_count)
            sorted_labels = sorted(label_count.keys())
            class_sample_counts = [label_count[label] for label in sorted_labels]
            print(f"=== Jittor 训练集样本统计 ===")
            print(f"实际类别数: {actual_num_classes}（应=50）")
            print(f"实际总样本数: {total_samples}（应=800=50类×16shot）")
            print(f"每类样本数: {class_sample_counts}（应全≈16）")
        except Exception as e:
            print(f"统计样本数时出错: {str(e)}")

        print(f"train_loader_x 的 batch 数: {len(self.train_loader_x)}")
        print(f"单个 batch 样本数: {self.train_loader_x.batch_size}")
        print(f"train_loader_x 总样本数（batch数 × batch_size）: {len(self.train_loader_x) * self.train_loader_x.batch_size}")
        # 验证 dataset 长度（Jittor DataLoader.dataset 应对应总样本数）
        print(f"train_loader_x.dataset 长度: {len(self.train_loader_x.dataset)}")

    def get_current_lr(self, names=None):
        """修复：正确获取Jittor优化器的学习率"""
        names = self.get_model_names(names)
        optim = self._optims[names[0]]
        
        if optim is None:
            return self.cfg.OPTIM.LR
        
        try:
            # 尝试从优化器状态获取学习率
            if hasattr(optim, 'lr'):
                lr = optim.lr
                if isinstance(lr, jt.Var):
                    return lr.item()
                return float(lr)
            
            if hasattr(optim, 'param_groups'):
                for param_group in optim.param_groups:
                    if 'lr' in param_group:
                        lr = param_group['lr']
                        if isinstance(lr, jt.Var):
                            return lr.item()
                        return float(lr)
        except:
            pass
        # 如果无法获取，返回配置中的默认学习率
        return self.cfg.OPTIM.LR


    def build_model(self):
            cfg = self.cfg
            print("构建模型")
            # 初始化模型
            self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            print(f"参数数量: {count_num_param(self.model):,}")
            
            # 构建优化器和学习率调度器
            self.optim = build_optimizer(self.model, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("model", self.model, self.optim, self.sched)
            print(f"初始学习率: {self.get_current_lr():.4e}")
            

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.RESUME if self.cfg.RESUME else self.cfg.OUTPUT_DIR
        self.start_epoch = self.resume_model_if_exist(directory)
        self.epoch = self.start_epoch
        self.time_start = time.time()

    def after_train(self):
        print("训练完成")
        if not self.cfg.TEST.NO_TEST:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("加载验证集表现最好的模型")
                self.load_model(self.output_dir)
            else:
                print("加载最后一轮模型")
            self.test()
        elapsed = str(datetime.timedelta(seconds=round(time.time() - self.time_start)))
        print(f"耗时: {elapsed}")

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            self.cfg.TRAIN.CHECKPOINT_FREQ > 0
            and (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @jt.no_grad()
    def test(self, split=None, is_final=False):

        print("=== 验证类别列表 ===")
        print()
        if self.distributed and self.rank != 0:
            return 0.0
        self.set_model_mode("eval")
        self.evaluator.reset()
        split = self.cfg.TEST.SPLIT if split is None else split
        data_loader = self.val_loader if (split == "val" and self.val_loader) else self.test_loader
        print(f"在 *{split}* 集上评估")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        if is_final:
            sys.stdout.flush()
            sys.stdout.close()
            os._exit(0)
        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        return batch["img"], batch["label"]

    # def get_current_lr(self, names=None):
    #     names = self.get_model_names(names)
    #     return self._optims[names[0]].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_x, len_train_u = len(self.train_loader_x), len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_x, len_train_u)
        else:
            raise ValueError("未知的迭代计数方式")

        iter_x, iter_u = iter(self.train_loader_x), iter(self.train_loader_u)
        end = time.time()

        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(iter_x)
            except StopIteration:
                iter_x = iter(self.train_loader_x)
                batch_x = next(iter_x)
            try:
                batch_u = next(iter_u)
            except StopIteration:
                iter_u = iter(self.train_loader_u)
                batch_u = next(iter_u)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = (self.num_batches - self.batch_idx - 1) + (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * nb_remain)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))
            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        return batch_x["img"], batch_x["label"], batch_u["img"]


class TrainerX(SimpleTrainer):
    def before_epoch(self):
        self.num_batches = len(self.train_loader_x)

    def run_epoch(self):
        self.set_model_mode("train")
        print(f"train_loader_x长度（batch数）: {len(self.train_loader_x)}")
        print(f"单个batch样本数: {self.train_loader_x.batch_size if hasattr(self.train_loader_x, 'batch_size') else '未知'}")
        print(f"训练集总样本数: {len(self.train_loader_x.dataset) if hasattr(self.train_loader_x.dataset, '__len__') else '未知'}")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()

        model_names = self.get_model_names()
        main_scheduler = self._scheds[model_names[0]] if model_names and self._scheds else None
        print(f"当前使用的调度器: {type(main_scheduler).__name__ if main_scheduler else 'None'}")

        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if main_scheduler is not None:
                try:
                    main_scheduler.step()
                    # 打印调度器 step 后的状态
                    if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                        current_lr = self.get_current_lr()
                        print(f"[调度器验证] batch {self.batch_idx+1}: 学习率={current_lr:.6f}")
                except Exception as e:
                    print(f"调度器 step 错误: {str(e)}")

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                current_lr = self.get_current_lr()
                # 确保是数值类型
                if current_lr is None:
                    current_lr = self.cfg.OPTIM.LR
                elif isinstance(current_lr, jt.Var):
                    current_lr = current_lr.item()
                info += [f"lr {float(current_lr):.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx

    def parse_batch_train(self, batch):
        return batch["img"], batch["label"], batch["domain"]