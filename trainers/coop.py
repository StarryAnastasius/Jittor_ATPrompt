import os.path as osp
import pickle
import jittor as jt
import jittor.nn as nn
import os
import numpy as np
import math
import traceback
import sys
from PIL import Image

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from jittor.transform import Compose, Resize, CenterCrop, ToTensor, ImageNormalize, RandomResizedCrop, RandomHorizontalFlip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

# --------------------------
# 预处理函数
# --------------------------
CLIP_INPUT_SIZE = 224
def clip_train_preprocess(input_size=CLIP_INPUT_SIZE):
    return Compose([
        RandomResizedCrop(
            size=input_size,
            scale=(0.08, 1.0),
            ratio=(3/4, 4/3),
            interpolation=Image.BICUBIC
        ),
        RandomHorizontalFlip(p=0.5),
        lambda img: img.convert("RGB"),
        ToTensor(),
        ImageNormalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

def clip_test_preprocess(input_size=CLIP_INPUT_SIZE):
    return Compose([
        Resize(input_size, mode=3),  # mode=3对应BICUBIC
        RandomHorizontalFlip(p=0.5),
        lambda img: img.convert("RGB"),
        ToTensor(),
        ImageNormalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

# --------------------------
# 模型加载函数
# --------------------------
def load_clip_to_cpu(cfg):
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    WEIGHT_SAVE_DIR = os.path.join(BASE_DIR, "clip_converted_weights")

    backbone_name = cfg.MODEL.BACKBONE.NAME
    safe_backbone_name = backbone_name.replace("/", "-")
    converted_weight_path = os.path.join(WEIGHT_SAVE_DIR, f"clip_{safe_backbone_name}_jittor.pkl")
    
    if not os.path.exists(converted_weight_path):
        raise FileNotFoundError(f"权重文件不存在: {converted_weight_path}\n请先运行转换脚本生成Jittor格式权重")
    
    with open(converted_weight_path, "rb") as f:
        state_dict_np = pickle.load(f)
    
    state_dict_jittor = {}
    for key, val in state_dict_np.items():
        if isinstance(val, np.ndarray):
            if np.issubdtype(val.dtype, np.floating):
                val_np = val.astype(np.float32)
            elif np.issubdtype(val.dtype, np.integer):
                val_np = val.astype(np.int64)
            else:
                val_np = val
            state_dict_jittor[key] = jt.array(val_np)
        else:
            state_dict_jittor[key] = val

    design_details = {
        "trainer": 'CoOp', 
        "vision_depth": 0, 
        "language_depth": 0, 
        "vision_ctx": 0, 
        "language_ctx": 0
    }
    model = clip.build_model(state_dict_jittor, design_details)
    return model


# --------------------------
# 模型组件
# --------------------------
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding.astype(jt.float32)
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.astype(jt.float32)
        self.dtype = jt.float32
        
        self.eot_token = clip._tokenizer.encoder["<|endoftext|>"]

    def execute(self, prompts, tokenized_prompts):
        prompts = prompts.astype(jt.float32)
        positional_emb = self.positional_embedding.astype(jt.float32)
        
        x = prompts + positional_emb
        x = x.permute(1, 0, 2).astype(jt.float32)
        x = self.transformer(x).astype(jt.float32)
        x = x.permute(1, 0, 2).astype(jt.float32)
        x = self.ln_final(x).astype(jt.float32)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        tokenized_prompts = tokenized_prompts.astype(jt.int64)
        
        eot_indices, _ = tokenized_prompts.argmax(dim=-1)
        eot_indices = eot_indices.clamp(0, seq_len - 1)

        eot_mask = (tokenized_prompts == self.eot_token).astype(jt.int64)
        correct_eot_indices = (eot_mask.argmax(dim=-1)[0]).clamp(0, seq_len - 1)

        indices = jt.stack([jt.arange(batch_size), eot_indices], dim=1).astype(jt.int64)
        x_emb = x[indices[:, 0], indices[:, 1]].astype(jt.float32)
        
        x = x_emb @ self.text_projection.astype(jt.float32)
        return x.astype(jt.float32)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = jt.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with jt.no_grad():
                embedding = clip_model.token_embedding(prompt).astype(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = jt.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = jt.empty(n_ctx, ctx_dim, dtype=dtype)
            jt.init.gauss_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]


        self.ctx = jt.Var(ctx_vectors)
        self.ctx.requires_grad = True

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        tokenized_prompts = jt.cat([clip.tokenize(p) for p in prompts])
        with jt.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).astype(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def execute(self,):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = jt.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = jt.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = jt.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = jt.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = jt.cat(prompts, dim=0)
        elif self.class_token_position == "attribute":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = jt.cat([prefix_i, ctx_i, class_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = jt.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = jt.array(4.6, dtype=jt.float32)
        self.logit_scale.stop_grad()
        self.dtype = jt.float32
        self.global_step = 0
        self.text_encoder.global_step = self.global_step 
        
        print("logit_scale初始值:", self.logit_scale)
        print("logit_scale.exp()的结果:", self.logit_scale.exp())

    def execute(self, image):
        image = image.astype(jt.float32)

        image_features = self.image_encoder(image).astype(jt.float32)
        prompts = self.prompt_learner().astype(jt.float32)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts).astype(jt.float32)

        # 特征归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.training and (jt.rank == 0) and (self.global_step % 100 == 0):
            img_norm = image_features.norm(dim=-1).mean().item()
            txt_norm = text_features.norm(dim=-1).mean().item()

        logit_scale = self.logit_scale.exp().astype(jt.float32)
        logits = (logit_scale * image_features @ text_features.permute(1, 0)).astype(jt.float32)

        return logits

def dump_dataloader_info(self, which="train"):
    try:
        if which == "train":
            loader = getattr(self, "train_loader_x", None) or getattr(self, "train_loader", None)
        else:
            loader = getattr(self, "test_loader", None)
        print(f"\n=== DATALOADER DUMP ({which}) ===")
        print("cfg.DATASET.NAME:", self.cfg.DATASET.NAME)
        print("cfg.DATASET.SUBSAMPLE_CLASSES:", getattr(self.cfg.DATASET, "SUBSAMPLE_CLASSES", None))
        print("cfg.DATASET.NUM_SHOTS:", getattr(self.cfg.DATASET, "NUM_SHOTS", None))
        print("cfg.INPUT.SIZE:", getattr(self.cfg.INPUT, "SIZE", None))
        if loader is None:
            print("No loader found for", which)
            return

        # dataloader length and batch_size & drop_last
        try:
            print("len(loader):", len(loader))
        except Exception:
            print("len(loader) unknown")
        try:
            bs = loader.batch_size
            print("batch_size:", bs)
        except Exception:
            print("cannot read batch_size attr")

        # underlying dataset
        ds = getattr(loader, "dataset", None)
        if ds is None and hasattr(self.dm, 'dataset'):
            ds = self.dm.dataset
        if ds is not None:
            try:
                print("dataset_len (len(ds)):", len(ds))
            except Exception:
                print("cannot get len(ds)")
            # If dataset has image paths or samples, try to dump first 10 sample ids/paths
            sample_paths = []
            for i in range(min(10, len(ds))):
                item = ds.__getitem__(i)
                # try common patterns
                if isinstance(item, dict):
                    if "img" in item:
                        sample_paths.append(f"sample[{i}] contains 'img' tensor")
                    if "path" in item:
                        sample_paths.append(item["path"])
                    if "file_name" in item:
                        sample_paths.append(item["file_name"])
                    if "index" in item:
                        sample_paths.append(str(item["index"]))
                else:
                    sample_paths.append(str(type(item)))
            print("sample preview (first up to 10):", sample_paths)
        else:
            print("Cannot access underlying dataset object from loader")

    except Exception as e:
        import traceback; traceback.print_exc()
        print("dump failed", e)

@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32"]

    def __init__(self, cfg):
        self.global_step = 0
        super().__init__(cfg)

    def build_model(self):
        cfg = self.cfg
        
        dump_dataloader_info(self, which="test")
        dump_dataloader_info(self, which="train")

        # 1. 加载CLIP模型
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        input_resolution = clip_model.visual.input_resolution
        

        print("临时构建数据加载器以获取类别名和检查数据泄露...")
        self.build_data_loader()  
        # 检查训练集与测试集重叠
        train_paths = [item.impath for item in self.dm.dataset.train_x]
        test_paths = [item.impath for item in self.dm.dataset.test]
        overlap = set(train_paths) & set(test_paths)
        print(f"[数据泄露] 训练集与测试集重叠样本数: {len(overlap)}")
        if len(overlap) > 0:
            print(f"[数据泄露] 重叠样本示例: {list(overlap)[:5]}")
        classnames = self.dm.dataset.classnames


        # 2. 构建CustomCLIP模型
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.global_step = self.global_step
        
        # 3. 冻结编码器权重
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad = False

        # 4. 构建优化器和调度器
        self.optim = build_optimizer(self.model.prompt_learner.parameters(), cfg.OPTIM)
        from dassl.optim.lr_scheduler import CosineAnnealingLR
        self.sched = CosineAnnealingLR(
            self.optim,
            T_max=cfg.OPTIM.MAX_EPOCH,
            eta_min=1e-6,
            last_epoch=-1
        )
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = None
        
        # 5. 重新构建数据加载器
        print("使用与CoOp_ATP对齐的预处理重新构建数据加载器...")
        self.build_data_loader(
            custom_tfm_train=clip_train_preprocess(input_size=input_resolution),
            custom_tfm_test=clip_test_preprocess(input_size=input_resolution)
        )
        
        # 6. 预处理验证打印
        train_tfm = self.train_loader_x.dataset.transform
        rrc = next(t for t in train_tfm.transforms if isinstance(t, RandomResizedCrop))
        print(f"[预处理验证] 训练集RandomResizedCrop.scale: {rrc.scale}")
        print(f"[预处理验证] 训练集预处理完整步骤: {[t.__class__.__name__ for t in train_tfm.transforms]}")

    def forward_backward(self, batch):
            image, label = self.parse_batch_train(batch)
            image = image.astype(jt.float32)
            label = label.astype(jt.int64)

            # 仅记录参数更新前的值
            ctx_before = self.model.prompt_learner.ctx[0, :5].clone().numpy()

            # 前向传播计算输出和损失
            output = self.model(image)
            loss = nn.cross_entropy_loss(output, label).astype(jt.float32)


            # 执行反向传播和参数更新
            self.model_backward_and_update(loss)

            # 验证参数更新
            ctx_after = self.model.prompt_learner.ctx[0, :5].numpy()
            ctx_diff = np.abs(ctx_after - ctx_before).mean()

            # 正确获取反向传播后的梯度
            ctx_grad = self.model.prompt_learner.ctx.opt_grad(self.optim)
            grad_norm = np.linalg.norm(ctx_grad.numpy()) if ctx_grad is not None else 0.0

            # 打印参数更新和梯度信息
            if self.global_step % 100 == 0:
                print(f"ctx参数更新差异: {ctx_diff:.6f}")
                print(f"ctx参数梯度L2范数: {grad_norm:.6f}")
                print(f"当前学习率: {self.optim.lr:.6f}")

            # 构建损失摘要
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item()
            }

            # 更新学习率,若为当前epoch最后一个batch
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            # 更新全局步数
            self.global_step += 1
            self.model.global_step = self.global_step
            self.model.text_encoder.global_step = self.global_step

            return loss_summary
    
    

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        print(f"DEBUG: Loading model file: {model_file} for names: {names}")

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            for key in ["token_prefix", "token_suffix"]:
                if key in state_dict:
                    del state_dict[key]
                    print(f"DEBUG: Deleted buffer key '{key}' from state_dict")

            print(f"Loading weights to {name} from {model_path} (epoch = {epoch})")
            self._models[name].load_state_dict(state_dict)