import hashlib
import os
import urllib.request
import warnings
from typing import Union, List, Callable, Tuple  # 导入Tuple
import numpy as np
import jittor as jt
from PIL import Image
from tqdm import tqdm

from .model_atp import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from jittor.transform import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _transform(n_px):
    from jittor.transform import Compose, Resize, CenterCrop, ToTensor, ImageNormalize

    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def available_models() -> List[str]:
    return list(_MODELS.keys())




def load(
    name: str, 
    device: Union[str, None] = "cuda" if jt.has_cuda else "cpu", 
    jit=False
) -> Tuple[jt.nn.Module, Callable[[Image.Image], jt.Var]]:  
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"模型 {name} 未找到；可用模型 = {available_models()}")

    state_dict = jt.load(model_path)
    model = build_model(state_dict)

    if device is not None:
        model = model.to(device)

    input_resolution = model.visual.input_resolution

    def manual_preprocess(image):
        import numpy as np
        from PIL import Image
        import jittor as jt
        # 确保是 RGB
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            img = np.array(image).astype("float32") / 255.0
        else:
            img = np.array(image).astype("float32")
            if img.max() > 1.0:
                img = img / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3,1,1)
        std  = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3,1,1)
        img = (img.transpose(2,0,1) - mean) / std
        return jt.array(img).unsqueeze(0)
    return model, manual_preprocess




def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> jt.Var:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = jt.zeros((len(all_tokens), context_length), dtype="int64")

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"输入 {texts[i]} 长于上下文长度 {context_length}")
        result[i, :len(tokens)] = jt.array(tokens, dtype="int64")

    return result