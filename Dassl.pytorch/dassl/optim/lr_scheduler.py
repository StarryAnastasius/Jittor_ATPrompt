from jittor.optim import LRScheduler
from jittor.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


def _safe_get_lr_from_scheduler(sched):
    """Try several ways to get lr list from successor scheduler."""
    if hasattr(sched, "get_last_lr"):
        try:
            return sched.get_last_lr()
        except Exception:
            pass

    if hasattr(sched, "get_lr"):
        try:
            return sched.get_lr()
        except Exception:
            pass
    if hasattr(sched, "_last_lr"):
        return sched._last_lr
    if hasattr(sched, "base_lrs"):
        return list(sched.base_lrs)
    return None


class _BaseWarmupScheduler(LRScheduler):
    """热身学习率调度器基类：热身阶段使用自身逻辑，热身结束后交给 successor 并把 lr 写回 optimizer"""
    def __init__(self, optimizer, successor, warmup_epoch, last_epoch=-1):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        # 同步last_epoch
        if last_epoch != -1:
            try:
                self.successor.last_epoch = last_epoch
            except Exception:
                pass
        super().__init__(optimizer, last_epoch)


    def _write_lrs_to_optimizer(self, lrs):
        """把 lr 列表写回 optimizer.param_groups"""
        if lrs is None:
            return
        # 如果 lrs 是单一值，扩展为 param_groups 长度
        if not isinstance(lrs, (list, tuple)):
            lrs = [lrs] * len(self.optimizer.param_groups)
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g["lr"] = float(lr)

    def get_lr(self):
        raise NotImplementedError

    def get_last_lr(self):
        # 返回最后一次计算的 lr
        lrs = _safe_get_lr_from_scheduler(self)
        if lrs is not None:
            return lrs
        # fallback: base_lrs
        return list(getattr(self, "base_lrs", [g["lr"] for g in self.optimizer.param_groups]))

    def step(self, epoch=None):
        """统一 step：先推进 last_epoch，然后把计算得到的 lr 写回 optimizer"""
        if epoch is None:
            epoch = self.last_epoch + 1
        # 更新自身的 last_epoch
        self.last_epoch = epoch

        if self.last_epoch >= self.warmup_epoch:
            try:
                # 优先调用 successor.step(epoch)
                self.successor.step(epoch)
            except TypeError:
                # 某些实现可能不接受 epoch 参数
                try:
                    self.successor.step()
                except Exception:
                    pass
            # 从 successor 获取 lr
            lrs = _safe_get_lr_from_scheduler(self.successor)
        else:
            # 热身阶段，使用本类 get_lr()
            lrs = self.get_lr()

        # 写回 optimizer，确保实际生效
        self._write_lrs_to_optimizer(lrs)
        # 并保留最后 lr 到 _last_lr 字段
        self._last_lr = lrs


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    def __init__(self, optimizer, successor, warmup_epoch, cons_lr, last_epoch=-1):
        self.cons_lr = cons_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            lrs = _safe_get_lr_from_scheduler(self.successor)
            return lrs
        return [self.cons_lr for _ in self.base_lrs]

    def state_dict(self):
        successor_state = {}
        if hasattr(self.successor, "state_dict"):
            try:
                successor_state = self.successor.state_dict()
            except Exception:
                successor_state = {"last_epoch": getattr(self.successor, "last_epoch", -1)}
        else:
            successor_state = {"last_epoch": getattr(self.successor, "last_epoch", -1)}
        return {
            "last_epoch": self.last_epoch,
            "successor_state": successor_state,
            "cons_lr": self.cons_lr,
            "warmup_epoch": self.warmup_epoch
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict.get("last_epoch", -1)
        successor_state = state_dict.get("successor_state", {})
        if hasattr(self.successor, "load_state_dict"):
            try:
                self.successor.load_state_dict(successor_state)
            except Exception:
                # fallback
                self.successor.last_epoch = successor_state.get("last_epoch", -1)
        else:
            self.successor.last_epoch = successor_state.get("last_epoch", -1)
        self.cons_lr = state_dict.get("cons_lr", self.cons_lr)
        self.warmup_epoch = state_dict.get("warmup_epoch", self.warmup_epoch)


class LinearWarmupScheduler(_BaseWarmupScheduler):
    def __init__(self, optimizer, successor, warmup_epoch, min_lr, last_epoch=-1):
        self.min_lr = min_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch)

    def get_lr(self):
        print(f"[Warmup Debug] last_epoch={self.last_epoch}, warmup_epoch={self.warmup_epoch}")
        if self.last_epoch >= self.warmup_epoch:
            lrs = _safe_get_lr_from_scheduler(self.successor)
            print(f"[Warmup Debug] 已过warmup，使用余弦LR: {lrs}")
            return lrs
        if self.last_epoch <= 0:
            lr = self.min_lr
        else:
            # 线性公式：min_lr + (base_lr - min_lr) * (epoch / warmup_epoch)
            lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * (self.last_epoch / float(self.warmup_epoch))
        lrs = [lr for _ in self.base_lrs]
        print(f"[Warmup Debug] Warmup中，计算LR: {lrs}（min_lr={self.min_lr}, base_lr={self.base_lrs[0]}）")
        return lrs

    def step(self, epoch=None):
        print(f"[Warmup Step Debug] 开始step，当前last_epoch: {self.last_epoch}")
        super().step(epoch)
        print(f"[Warmup Step Debug] step后last_epoch: {self.last_epoch}")
        current_lr = _safe_get_lr_from_scheduler(self)
        print(f"[Warmup Step Debug] 当前学习率: {current_lr}")


    def state_dict(self):
        successor_state = {}
        if hasattr(self.successor, "state_dict"):
            try:
                successor_state = self.successor.state_dict()
            except Exception:
                successor_state = {"last_epoch": getattr(self.successor, "last_epoch", -1)}
        else:
            successor_state = {"last_epoch": getattr(self.successor, "last_epoch", -1)}
        return {
            "last_epoch": self.last_epoch,
            "successor_state": successor_state,
            "min_lr": self.min_lr,
            "warmup_epoch": self.warmup_epoch
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict.get("last_epoch", -1)
        successor_state = state_dict.get("successor_state", {})
        if hasattr(self.successor, "load_state_dict"):
            try:
                self.successor.load_state_dict(successor_state)
            except Exception:
                self.successor.last_epoch = successor_state.get("last_epoch", -1)
        else:
            self.successor.last_epoch = successor_state.get("last_epoch", -1)
        self.min_lr = state_dict.get("min_lr", self.min_lr)
        self.warmup_epoch = state_dict.get("warmup_epoch", self.warmup_epoch)


def build_lr_scheduler(optimizer, optim_cfg):
    """构建调度器（Jittor 版本）"""
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(f"scheduler必须为{AVAI_SCHEDS}之一，当前为{lr_scheduler}")

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]
        if not isinstance(stepsize, int):
            raise TypeError(f"single_step的stepsize必须为int，当前为{type(stepsize)}")
        if stepsize <= 0:
            stepsize = max_epoch
        successor = StepLR(optimizer, step_size=stepsize, gamma=gamma)

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(f"multi_step的stepsize必须为list，当前为{type(stepsize)}")
        successor = MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == "cosine":
        successor = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

    scheduler = successor

    # warmup
    if getattr(optim_cfg, "WARMUP_EPOCH", 0) > 0:
        # 获取当前是否是断点续训
        is_resume = getattr(optim_cfg, "RESUME", False) or (getattr(successor, "last_epoch", -1) != -1)
        if not optim_cfg.WARMUP_RECOUNT and is_resume:
            try:
                successor.last_epoch = optim_cfg.WARMUP_EPOCH
                print(f"断点续训：同步successor.last_epoch={optim_cfg.WARMUP_EPOCH}")
            except Exception:
                pass

        if optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, successor, optim_cfg.WARMUP_EPOCH, optim_cfg.WARMUP_MIN_LR
            )
            print(f"启用线性warmup：{optim_cfg.WARMUP_MIN_LR} → {optim_cfg.LR}（{optim_cfg.WARMUP_EPOCH}个epoch）")

    # 最终返回能写回 optimizer 的 scheduler
    return scheduler
