import os
import sys
import argparse
import jittor as jt

from clip.simple_tokenizer import SimpleTokenizer 
from clip_jittor import clip  

# "ViT-B/16"
# "RN50"
def load_clip_to_cpu(backbone_name="ViT-B/16"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    state_dict = jt.load(model_path)

    model = clip.build_model(state_dict)

    return model



fpath = "./compound_prompt_weights/train_base/food101/shots_16/cocoop/vit_b16_c4_ep10_batch1_ctxv1/seed1/prompt_learner/model.pth.tar-5"
topk = 10

assert os.path.exists(fpath), f"文件不存在: {fpath}"

print(f"返回前-{topk}个匹配的单词")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"词嵌入的大小: {token_embedding.shape}")

# 加载prompt权重
# 如果是Jittor保存的权重，可以直接用jt.load
prompt_learner = jt.load(fpath)["state_dict"]
# 提取输入tokens
ctx = prompt_learner["prompt_learner.ctx"]
ctx = ctx.float()
# 提取中间tokens
intermediate_embeddings = []
depth = 9 - 1  # 原代码中的深度计算
for i in range(depth):
    # 提取prompt嵌入
    query = f'prompt_learner.compound_prompts_text.{i}'
    temp = prompt_learner[query].float()
    intermediate_embeddings.append(temp)

print(f"上下文的大小: {ctx.shape}")

# 合并所有层的上下文嵌入
all_layer_ctx = [ctx] + intermediate_embeddings

for idx, single_ctx in enumerate(all_layer_ctx):
    print(f"显示第 {idx + 1} 层的CTX向量结果:")
    ctx = single_ctx
    if ctx.dim() == 2:
        # 通用上下文
        # Jittor中cdist的用法与PyTorch类似
        distance = jt.cdist(ctx, token_embedding)
        print(f"距离矩阵的大小: {distance.shape}")
        # 排序并获取前k个索引
        sorted_idxs = jt.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            # 将索引转换为单词
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            # 获取对应的距离值
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            print(f"{m + 1}: {words} {dist}")

    elif ctx.dim() == 3:
        raise NotImplementedError("暂不支持特定类别上下文的处理")
