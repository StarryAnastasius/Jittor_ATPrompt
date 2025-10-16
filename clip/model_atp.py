from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import jittor as jt
from jittor import nn
from jittor import attention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def execute(self, x: jt.Var):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(jt.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.attn = attention.MultiheadAttention(embed_dim, num_heads, bias=True)

    def execute(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = jt.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].astype(x.dtype)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x, _ = self.attn(q, k, v, need_weights=False)

        x = self.c_proj(x)
        return x[0]


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def execute(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.astype(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def execute(self, x: jt.Var):
        orig_type = x.dtype
        ret = super().execute(x.astype('float32'))
        return ret.astype(orig_type)


class QuickGELU(nn.Module):
    def execute(self, x: jt.Var):
        return x * jt.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: jt.Var = None):
        super().__init__()
        self.attn = attention.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: jt.Var):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def execute(self, x: jt.Var):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: jt.Var = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()
        self.attn = attention.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.add_prompt = add_prompt if i != 0 else False
        self.atprompt_num = design_details.get("atprompt_num", 0)

        if self.add_prompt:
            if self.text_layer:
                self.n_ctx_text = design_details["language_ctx"]
                ctx_vectors = jt.empty(self.n_ctx_text, d_model)
                nn.init.gauss_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
            else:
                self.n_ctx_visual = design_details["vision_ctx"]
                ctx_vectors = jt.empty(self.n_ctx_visual, d_model)
                nn.init.gauss_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)

    def attention(self, x: jt.Var):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def execute(self, x: jt.Var):
        if self.add_prompt:
            if not self.text_layer:
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).astype('float32')
                x = jt.cat([prefix, visual_context], dim=0)
            else:
                att_token_lens = self.atprompt_num * (self.n_ctx_text + 1)
                prefix = x[:1 + att_token_lens, :, :]
                suffix = x[1 + att_token_lens + self.n_ctx_text:, :, :]
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).astype('float32')
                x = jt.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: jt.Var = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()
        self.attn = attention.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.compound_prompt_nctx = design_details['maple_length']
        self.atprompt_num = design_details['atprompt_num']
        self.first_layer = (i == 0)

    def attention(self, x: jt.Var):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def execute(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        
        if not self.first_layer and len(compound_prompts_deeper) > 0:
            if not self.text_layer and not (counter > len(compound_prompts_deeper) - 1):
                prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                visual_context = compound_prompts_deeper[counter].expand(x.shape[1], -1, -1).permute(1, 0, 2).astype('float32')
                x = jt.cat([prefix, visual_context], dim=0)
                counter += 1
            elif self.text_layer and not (counter > len(compound_prompts_deeper) - 1):
                att_token_lens = self.atprompt_num * (self.compound_prompt_nctx + 1)
                prefix = x[:1 + att_token_lens, :, :]
                textual_context = compound_prompts_deeper[counter].expand(x.shape[1], -1, -1).permute(1, 0, 2).astype('float32')
                suffix = x[1 + att_token_lens + self.compound_prompt_nctx:, :, :]
                x = jt.cat([prefix, textual_context, suffix], dim=0)
                counter += 1
        
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: jt.Var = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        current_trainer = design_details['trainer']
        
        if current_trainer in ['IVLP', 'VPT']:
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_IVLP(width, heads, attn_mask, True, text_layer, i, design_details) 
                if prompts_needed > i else 
                ResidualAttentionBlock_IVLP(width, heads, attn_mask, False, text_layer, i, design_details)
                for i in range(layers)
            ])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                for i in range(layers)
            ])
        else:
            assert current_trainer in ['CoOp', 'CoCoOp']
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
            ])

    def execute(self, x: jt.Var):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = (design_details["vision_depth"] != 0)
        
        if self.VPT_shallow:
            n_ctx = design_details["vision_ctx"]
            ctx_vectors = jt.empty(n_ctx, width)
            nn.init.gauss_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * jt.randn(width))
        self.positional_embedding = nn.Parameter(scale * jt.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(
            width, layers, heads, 
            prompts_needed=self.prompt_till_layer_visual,
            design_details=design_details
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * jt.randn(width, output_dim))

    def execute(self, x: jt.Var):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = jt.cat([
            self.class_embedding.astype(x.dtype) + jt.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype),
            x
        ], dim=1)
        x = x + self.positional_embedding.astype(x.dtype)

        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).astype('float32')
            x = jt.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * jt.randn(width))
        self.positional_embedding = nn.Parameter(scale * jt.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * jt.randn(width, output_dim))

    def execute(self, x: jt.Var, shared_ctx, compound_deeper_prompts, att1_ctx, att2_ctx, token_middle1, token_middle2):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = jt.cat([
            self.class_embedding.astype(x.dtype) + jt.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype),
            x
        ], dim=1)
        x = x + self.positional_embedding.astype(x.dtype)

        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).astype('float32')
            x = jt.cat([x, visual_ctx], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        outputs = self.transformer([x, compound_deeper_prompts, 0, att1_ctx, att2_ctx, token_middle1, token_middle2])
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()
        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )

        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(jt.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(jt.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(jt.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.gauss_(self.token_embedding.weight, std=0.02)
        nn.init.gauss_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet) and self.visual.attnpool is not None:
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            for proj in [self.visual.attnpool.q_proj, self.visual.attnpool.k_proj,
                         self.visual.attnpool.v_proj, self.visual.attnpool.c_proj]:
                nn.init.gauss_(proj.weight, std=std)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.gauss_(block.attn.in_proj_weight, std=attn_std)
            nn.init.gauss_(block.attn.out_proj.weight, std=proj_std)
            nn.init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.gauss_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.gauss_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = jt.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(self.dtype)
        x = x + self.positional_embedding.astype(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).astype(self.dtype)
        x = x[jt.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.astype('float32')
            if l.bias is not None:
                l.bias.data = l.bias.data.astype('float32')

        if isinstance(l, attention.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                if hasattr(l, attr) and getattr(l, attr) is not None:
                    getattr(l, attr).data = getattr(l, attr).data.astype('float32')

        for name in ["text_projection", "proj"]:
            if hasattr(l, name) and getattr(l, name) is not None:
                getattr(l, name).data = getattr(l, name).data.astype('float32')

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1,2,3,4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
