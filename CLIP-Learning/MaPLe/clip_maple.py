"""MaPLe 版 CLIP 模型 (基于 openai/CLIP 官方源码改写, 复刻 MaPLe 的 prompt 注入逻辑)

本文件是官方 `clip/model.py` 的 MaPLe 变体, 关键改动:
1. ResidualAttentionBlock_MaPLe.forward(inputs):
   inputs = [x, compound_prompts_deeper, counter]  (列表, LND 格式)
   - 文本层: 把 compound prompt 插入到 CLS 之后、原文本 token 之前
   - 视觉层: 把新的视觉 deep prompt 替换序列末尾的可学习 token
2. VisionTransformer_MaPLe.forward(x, shared_ctx, compound_deeper_prompts):
   - 浅层 shared_ctx 拼在序列末尾 (NLD, dim=1)
   - 深层由 transformer 内部逐层替换
3. CLIP: 暴露 encode_image_with_prompts / encode_text_with_prompts
   - 文本侧: token -> positional -> transformer(list) -> ln_final -> eot 投影
   - 视觉侧: 直接调用 VisionTransformer_MaPLe

设计细节 design_details 固定 trainer='MaPLe'。
"""

from collections import OrderedDict

import os

import numpy as np
import torch
import torch.nn as nn
from clip import clip
from clip.model import LayerNorm, QuickGELU
from clip.clip import _download, _MODELS, available_models


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 design_details=None, text_layer=False, i=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
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
        self.first_layer = (i == 0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                if not self.text_layer:
                    # 视觉侧: 移除前一层的可学习 token (位于序列末尾)
                    prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                    visual_context = compound_prompts_deeper[counter]
                    visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    x = torch.cat([prefix, visual_context], dim=0)
                    counter += 1
                else:
                    # 文本侧: x 形状 [77, NCLS, DIM]
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx:, :, :]
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    counter += 1
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class TextTransformer_MaPLe(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer=True, i=i)
            for i in range(layers)
        ])

    def forward(self, x, compound_prompts_deeper):
        return self.resblocks([x, compound_prompts_deeper, 0])


class _VisionTransformerInner(nn.Module):
    """包装视觉 transformer 的残差块, 使 state_dict key 与预训练权重一致
    (visual.transformer.resblocks.{i})。"""

    def __init__(self, blocks):
        super().__init__()
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        # x 是 [tensor, compound_deeper_prompts, counter] 列表
        return self.resblocks(x)


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int,
                 heads: int, output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.prompt_till_layer_visual = 0
        self.transformer = _VisionTransformerInner([
            ResidualAttentionBlock_MaPLe(width, heads, design_details=design_details,
                                        text_layer=False, i=i)
            for i in range(layers)
        ])
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1)
            x = torch.cat([x, visual_ctx], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer([x, compound_deeper_prompts, 0])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class ModifiedResNet_MaPLe(nn.Module):
    """占位: 本实验只用 ViT-B/32, 不实现 RN 版 MaPLe。"""
    pass


class CLIP_MaPLe(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int,
                 vision_width: int, vision_patch_size: int, context_length: int,
                 vocab_size: int, transformer_width: int, transformer_heads: int,
                 transformer_layers: int, design_details):
        super().__init__()
        self.context_length = context_length
        self.design_details = design_details

        if isinstance(vision_layers, (tuple, list)):
            raise NotImplementedError("MaPLe-CIFAR10 实验仅支持 ViT 视觉骨干")

        vision_heads = vision_width // 64
        self.visual = VisionTransformer_MaPLe(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_heads, output_dim=embed_dim,
            design_details=design_details,
        )

        attn_mask = torch.empty(context_length, context_length)
        attn_mask.fill_(float("-inf"))
        attn_mask.triu_(1)  # 上三角为 -inf -> 因果掩码
        self.transformer = TextTransformer_MaPLe(
            width=transformer_width, layers=transformer_layers, heads=transformer_heads,
            attn_mask=attn_mask,
            design_details=design_details,
        )

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dtype = torch.float32

    def encode_image_with_prompts(self, image, shared_ctx, compound_deeper_prompts):
        return self.visual(image.type(self.dtype), shared_ctx, compound_deeper_prompts)

    def encode_text_with_prompts(self, text, compound_prompts_deeper_text):
        # text: (n_cls, context_length) token ids
        x = self.token_embedding(text).type(self.dtype)  # (n_cls, L, D)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer(x, compound_prompts_deeper_text)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def build_name_to_update(self):
        return "prompt_learner"


def build_model_maple(state_dict: dict, maple_length: int = 2):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise NotImplementedError("MaPLe-CIFAR10 实验仅支持 ViT 视觉骨干")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict
                                 if k.startswith("transformer.resblocks")))

    design_details = {
        "trainer": 'MaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": maple_length,
    }

    model = CLIP_MaPLe(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads,
        transformer_layers, design_details,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 本实验采用 float32 精度以保持数值稳定 (XPU/CPU 兼容); 故不做 half 转换
    model.load_state_dict(state_dict)
    return model.float().eval()


def convert_weights(model: nn.Module):
    """把模型中的浮点权重转成 float16 (官方 CLIP build_model 的做法)。"""
    def _convert_module(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            m.weight.data = m.weight.data.half()
            if m.bias is not None:
                m.bias.data = m.bias.data.half()
    model.apply(_convert_module)


def load_clip_to_cpu(backbone_name="ViT-B/32", maple_length=2):
    if backbone_name in _MODELS:
        model_path = _download(_MODELS[backbone_name],
                               root=os.path.expanduser(os.path.join("~", ".cache", "clip")))
    elif backbone_name in available_models():
        raise RuntimeError(f"{backbone_name} 不在内置下载列表")
    else:
        model_path = backbone_name

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model_maple(state_dict or model.state_dict(), maple_length=maple_length)
    return model


__all__ = ["CLIP_MaPLe", "load_clip_to_cpu", "build_model_maple", "design_details"]
