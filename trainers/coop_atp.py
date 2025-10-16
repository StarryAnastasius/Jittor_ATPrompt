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

import json

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
        Resize(input_size, mode=3),
        RandomHorizontalFlip(p=0.5),
        lambda img: img.convert("RGB"),
        ToTensor(),
        ImageNormalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def load_clip_to_cpu(cfg):
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    WEIGHT_SAVE_DIR = os.path.join(BASE_DIR, "clip_converted_weights")

    backbone_name = cfg.MODEL.BACKBONE.NAME
    safe_backbone_name = backbone_name.replace("/", "-")
    converted_weight_path = os.path.join(WEIGHT_SAVE_DIR, f"clip_{safe_backbone_name}_jittor.pkl")
    
    if not os.path.exists(converted_weight_path):
        raise FileNotFoundError(f"权重文件不存在: {converted_weight_path}")
    
    with open(converted_weight_path, "rb") as f:
        state_dict_np = pickle.load(f)
    
    # 权重转为Jittor张量
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

    # 构建CLIP模型
    design_details = {"trainer": 'CoOp', "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(state_dict_jittor, design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        # positional embedding shape used as sequence length reference
        self.positional_embedding = clip_model.positional_embedding.astype(jt.float32)  # [fixed: dtype=float32]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.astype(jt.float32)  # [fixed: dtype=float32]
        self.dtype = jt.float32

        # seq_len inferred from positional embedding
        try:
            self.seq_len = int(self.positional_embedding.shape[0])
        except Exception:
            self.seq_len = None
        # print(f"[debug] TextEncoder initialized. seq_len={self.seq_len}")

    def execute(self, prompts, tokenized_prompts):
        # prompts: (n_cls, seq_len, dim)
        # tokenized_prompts: (n_cls, seq_len)  (int64)
        prompts = prompts.astype(jt.float32)
        positional_emb = self.positional_embedding.astype(jt.float32)

        try:
            if self.seq_len is None:
                self.seq_len = int(positional_emb.shape[0])
        except Exception:
            pass

        x = prompts + positional_emb
        x = x.permute(1, 0, 2).astype(jt.float32)
        x = self.transformer(x).astype(jt.float32)
        x = x.permute(1, 0, 2).astype(jt.float32)
        x = self.ln_final(x).astype(jt.float32)

        batch_size = int(x.shape[0])
        seq_len = int(x.shape[1])
        #_safe_print_tensor_info("x (after ln_final)", x)

        # tokenized_prompts ensure int64
        tokenized_prompts = tokenized_prompts.astype(jt.int64)
        #_safe_print_tensor_info("tokenized_prompts", tokenized_prompts)

        # validate shapes: tokenized_prompts should match seq_len (or be at least compatible)
        if tokenized_prompts.dim() == 1:
            # unlikely, expand
            tokenized_prompts = tokenized_prompts.reshape(batch_size, -1)
            # print("[debug] tokenized_prompts was 1d, reshaped to", tokenized_prompts.shape)

        # If tokenized_prompts seq len mismatches x's seq_len, print debug and attempt fix by padding/trunc
        try:
            tok_seq_len = int(tokenized_prompts.shape[1])
        except Exception:
            tok_seq_len = None

        if tok_seq_len is None:
            pass
        else:
            if tok_seq_len != seq_len:
                if tok_seq_len < seq_len:
                    pad_amount = seq_len - tok_seq_len
                    pad = jt.zeros((batch_size, pad_amount), dtype=tokenized_prompts.dtype)
                    tokenized_prompts = jt.concat([tokenized_prompts, pad], dim=1)
                else:
                    tokenized_prompts = tokenized_prompts[:, :seq_len]

        # construct indices safely
        idx0 = jt.arange(batch_size).reshape(batch_size, 1).astype(jt.int64)

        # jittor argmax 返回 (indices, values)
        argmax_out = tokenized_prompts.argmax(dim=-1)

        if isinstance(argmax_out, (tuple, list)) and len(argmax_out) == 2:
            idx1 = argmax_out[0] 
            idx1 = idx1.astype(jt.int64)
            
            seq_len = self.seq_len  # 从TextEncoder初始化时的positional_embedding获取
            if seq_len is not None:
                idx1_min = idx1.min().item()
                idx1_max = idx1.max().item()
                assert idx1_min >= 0 and idx1_max < seq_len, \
                    f"索引超出有效范围！应在0~{seq_len-1}，实际{idx1_min}~{idx1_max}"
        else:
            np_tp = tokenized_prompts.numpy()
            idx1_np = np.argmax(np_tp, axis=-1).reshape(batch_size, 1).astype(np.int64)
            idx1 = jt.array(idx1_np)

        # 索引形状正确[batch_size, 1]
        try:
            idx1 = idx1.reshape(batch_size, 1).astype(jt.int64)
        except Exception:
            try:
                idx1 = idx1.view(batch_size, 1).astype(jt.int64)
            except Exception:
                print("[error] 无法调整idx1形状")
                raise

        # Clamp indices to valid range to avoid out-of-range getitem (prevents segfault)
        try:
            min_idx1 = int(idx1.min().item())
            max_idx1 = int(idx1.max().item())
        except Exception:
            min_idx1 = None
            max_idx1 = None

        # print(f"[debug] idx1 range before clamp: min={min_idx1}, max={max_idx1}, seq_len={seq_len}")

        # clamp to [0, seq_len-1]
        idx1 = jt.clamp(idx1, 0, seq_len - 1).astype(jt.int64)
        try:
            min_idx1_c = int(idx1.min().item())
            max_idx1_c = int(idx1.max().item())
        except Exception:
            min_idx1_c = None
            max_idx1_c = None

        indices = jt.concat([idx0, idx1], dim=1).astype(jt.int64)
        try:
            x_emb = x[indices[:, 0], indices[:, 1]].astype(jt.float32)
        except Exception as e:
            print("[error] Exception during indexing x[indices[:,0], indices[:,1]]")
            print("Context debug info:")
            print("Traceback:")
            traceback.print_exc()
            # Re-raise for upper-level handling
            raise

        # matrix multiply
        x = x_emb @ self.text_projection.astype(jt.float32)

        # final debug
        # _safe_print_tensor_info("x (text emb out)", x)
        return x.astype(jt.float32)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = jt.float32  # [fixed: dtype=float32]
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if False:
            ctx_init = "a photo of a"
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

        self.ctx = nn.Parameter(ctx_vectors)

        self.use_atp = cfg.TRAINER.ATPROMPT.USE_ATPROMPT
        self.atp_num = cfg.TRAINER.ATPROMPT.ATT_NUM

        print(f'self.use_atp is {self.use_atp}')
        print(f'self.atp_num is {self.atp_num}')

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if self.use_atp:
            print("USE ATPROPMT-ING 1")
            n_att1 = cfg.TRAINER.ATPROMPT.N_ATT1
            att1_text = cfg.TRAINER.ATPROMPT.ATT1_TEXT

            n_att2 = cfg.TRAINER.ATPROMPT.N_ATT2
            att2_text = cfg.TRAINER.ATPROMPT.ATT2_TEXT

            n_att3 = cfg.TRAINER.ATPROMPT.N_ATT3
            att3_text = cfg.TRAINER.ATPROMPT.ATT3_TEXT

            att_vectors_1 = jt.empty(n_att1, ctx_dim, dtype=dtype)
            att_vectors_2 = jt.empty(n_att2, ctx_dim, dtype=dtype)
            att_vectors_3 = jt.empty(n_att3, ctx_dim, dtype=dtype)

            jt.init.gauss_(att_vectors_1, std=0.01)
            prefix1 = " ".join(["X"] * n_att1)
            jt.init.gauss_(att_vectors_2, std=0.01)
            prefix2 = " ".join(["X"] * n_att2)
            jt.init.gauss_(att_vectors_3, std=0.01)
            prefix3 = " ".join(["X"] * n_att3)

            self.ctx_att1 = nn.Parameter(att_vectors_1)
            self.ctx_att2 = nn.Parameter(att_vectors_2)
            self.ctx_att3 = nn.Parameter(att_vectors_3)

            if self.atp_num == 1:
                prompts = [prefix1 + " " + att1_text + " " + prompt_prefix + " " + name + "." for name in classnames]
            elif self.atp_num == 2:
                prompts = [
                    prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prompt_prefix + " " + name + "."
                    for name in classnames]
            elif self.atp_num == 3:
                prompts = [
                    prefix1 + " " + att1_text + " " + prefix2 + " " + att2_text + " " + prefix3 + " " + att3_text + " " + prompt_prefix + " " + name + "."
                    for name in classnames]
            else:
                print("wrong parameter.")
                raise ValueError
        print(prompts)

        # tokenized prompts -> int64, embedding -> float32
        tokenized_prompts = jt.cat([clip.tokenize(p) for p in prompts]).astype(jt.int64)
        with jt.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).astype(dtype)
            print(f'embedding shape is {embedding.shape}')

        # register buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])

        if self.use_atp:
            print("USE ATPROPMT-ING 2")
            if self.atp_num == 1:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: 1 + n_att1 + 1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_ctx:, :])

            elif self.atp_num == 2:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: n_att1 + 1 + 1, :])
                self.register_buffer("token_middle2",
                                     embedding[:, 1 + n_att1 + 1 + n_att2: 1 + n_att1 + 1 + n_att2 + 1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_att2 + 1 + n_ctx:, :])

            elif self.atp_num == 3:
                self.register_buffer("token_middle1", embedding[:, 1 + n_att1: n_att1 + 1 + 1, :])
                self.register_buffer("token_middle2",
                                     embedding[:, 1 + n_att1 + 1 + n_att2: 1 + n_att1 + 1 + n_att2 + 1, :])
                self.register_buffer("token_middle3", embedding[:,
                                                      1 + n_att1 + 1 + n_att2 + 1 + n_att3: 1 + n_att1 + 1 + n_att2 + 1 + n_att3 + 1,
                                                      :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_att1 + 1 + n_att2 + 1 + n_att3 + 1 + n_ctx:, :])
            else:
                raise ValueError
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # jt.Var (int64)
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION


    def execute(self):
        ctx = self.ctx
        if self.use_atp:
            ctx_att1 = self.ctx_att1
            ctx_att2 = self.ctx_att2
            ctx_att3 = self.ctx_att3

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.use_atp:
                ctx_att1 = ctx_att1.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att2 = ctx_att2.unsqueeze(0).expand(self.n_cls, -1, -1)
                ctx_att3 = ctx_att3.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.use_atp:
            if self.atp_num == 1:
                middle_attribute1 = self.token_middle1
                prompts = jt.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 2:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                prompts = jt.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            elif self.atp_num == 3:
                middle_attribute1 = self.token_middle1
                middle_attribute2 = self.token_middle2
                middle_attribute3 = self.token_middle3
                prompts = jt.cat(
                    [
                        prefix,
                        ctx_att1,
                        middle_attribute1,
                        ctx_att2,
                        middle_attribute2,
                        ctx_att3,
                        middle_attribute3,
                        ctx,
                        suffix,
                    ],
                    dim=1,
                )
            else:
                raise ValueError
        else:
            prompts = jt.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        if self.class_token_position == "middle":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                ctx_att1_i = ctx_att1[i:i + 1, :, :] if self.use_atp else None
                mid_att1 = self.token_middle1[i:i + 1, :, :] if self.use_atp else None
                ctx_att2_i = ctx_att2[i:i + 1, :, :] if self.use_atp and self.atp_num >= 2 else None
                mid_att2 = self.token_middle2[i:i + 1, :, :] if self.use_atp and self.atp_num >= 2 else None

                prompt_parts = [prefix_i, ctx_att1_i, mid_att1, ctx_i, class_i, ctx_att2_i, mid_att2, suffix_i]
                prompt_parts = [part for part in prompt_parts if part is not None]
                prompt = jt.cat(prompt_parts, dim=1)
                prompts_list.append(prompt)
            prompts = jt.cat(prompts_list, dim=0)

        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]

                ctx_att1_i = ctx_att1[i:i + 1, :, :] if self.use_atp else None
                mid_att1 = self.token_middle1[i:i + 1, :, :] if self.use_atp else None
                ctx_att2_i = ctx_att2[i:i + 1, :, :] if self.use_atp and self.atp_num >= 2 else None
                mid_att2 = self.token_middle2[i:i + 1, :, :] if self.use_atp and self.atp_num >= 2 else None

                prompt_parts = [prefix_i, ctx_i, class_i, ctx_att1_i, mid_att1, ctx_att2_i, mid_att2, suffix_i]
                prompt_parts = [part for part in prompt_parts if part is not None]
                prompt = jt.cat(prompt_parts, dim=1)
                prompts_list.append(prompt)
            prompts = jt.cat(prompts_list, dim=0)

        return prompts.astype(jt.float32)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.astype(jt.float32)
        self.dtype = jt.float32

        try:
            self.image_encoder.float()
        except Exception:
            try:
                for name, p in self.image_encoder.named_parameters():
                    if p.dtype != jt.float32:
                        with jt.no_grad():
                            try:
                                self.image_encoder.__getattr__(name).assign(p.astype(jt.float32))
                            except Exception:
                                self.image_encoder.__dict__[name] = p.astype(jt.float32)
            except Exception:
                pass



    def execute(self, image):
        image = image.astype(jt.float32)

        
        
        # 2. 验证CLIP图像特征是否有效
        image_features = self.image_encoder(image).astype(jt.float32)
        
        
        # 3. 验证文本特征是否有效
        prompts = self.prompt_learner().astype(jt.float32)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts).astype(jt.float32)

        image_features = self.image_encoder(image).astype(jt.float32)

        # prompts = self.prompt_learner().astype(jt.float32)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts).astype(jt.float32)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
class CoOp_ATP(TrainerX):
    """Context Optimization (CoOp) with ATPrompt."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32"]

    def build_model(self):
        cfg = self.cfg
        
        dump_dataloader_info(self, which="test")
        dump_dataloader_info(self, which="train")

        # 1. 临时构建数据加载器获取类别名
        self.build_data_loader()
        classnames = self.dm.dataset.classnames

        # 2. 加载CLIP模型
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        input_resolution = clip_model.visual.input_resolution  # 获取模型输入尺寸

        # 3. 构建CustomCLIP + 冻结权重
        self.model = CustomCLIP(cfg, classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad = False

        # 4. 优化器配置
        self.optim = build_optimizer(self.model.prompt_learner.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # 5. 构建数据加载器
        self.build_data_loader(
            custom_tfm_train=clip_train_preprocess(input_size=input_resolution),  
            custom_tfm_test=clip_test_preprocess(input_size=input_resolution)    
        )

        self.scaler = None

    def forward_backward(self, batch):


        image, label = self.parse_batch_train(batch)

        # force float32
        image = image.astype(jt.float32)

        output = None
        try:
            output = self.model(image)
        except Exception as e:
            print("[error] Exception during model forward.")
            traceback.print_exc()
            raise

        loss = nn.cross_entropy_loss(output, label)

        logits = output  # shape [B, C]
        probs = jt.nn.softmax(logits, dim=1)
        top1_idx, top1_vals = probs.argmax(1)

        true_logits = logits[jt.arange(logits.shape[0]), label]


        try:
            loss = loss.astype(jt.float32)
        except Exception:
            try:
                loss = loss.float32()
            except Exception:
                pass
        param_name = "ctx"
        param_before = self.model.prompt_learner.ctx[0, :5].clone()

        self.model_backward_and_update(loss)

        param_after = self.model.prompt_learner.ctx[0, :5]


        try:
            output = output.astype(jt.float32)
        except Exception:
            pass

        loss_summary = {
            "loss": loss.item() if hasattr(loss, "item") else float(loss),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # print(f"[debug] loss_summary: {loss_summary}")
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"], batch["label"]

    def load_model(self, directory, epoch=None):

        if hasattr(self.model.prompt_learner, "use_atp") and self.model.prompt_learner.use_atp:
            print(f"\n=== 验证ATPrompt参数加载 ===")
            print(f"ctx_att1均值: {self.model.prompt_learner.ctx_att1.mean().item()}")
            print(f"ctx_att2均值: {self.model.prompt_learner.ctx_att2.mean().item()}")
        
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(f"DEBUG: get_model_names() returns: {names}")

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            print(f"DEBUG: Trying to load model from: {model_path}")

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            for key in ["token_prefix", "token_suffix", "token_middle1", "token_middle2", "token_middle3"]:
                if key in state_dict:
                    del state_dict[key]

            print("Loading weights to {} from {} (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict)
