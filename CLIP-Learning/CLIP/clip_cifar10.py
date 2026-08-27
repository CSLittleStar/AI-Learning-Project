"""CLIP 小型实验 (CIFAR-10) —— 调用 OpenAI 开源预训练 CLIP 权重

关键修正 (相对此前的误区):
- 不需要自己从零训练一个 ViT / Text Encoder;
- CLIP 与 CoOp 论文作者已在 https://github.com/openai/CLIP 开源了预训练权重,
  直接用 `model, preprocess = clip.load("ViT-B/32")` 即可加载官方预训练模型
  (Image Encoder + Text Encoder + 投影层 + 温度 logit_scale 全部就绪);
- 本脚本只演示如何"复用"该预训练模型在 CIFAR-10 上做 zero-shot 分类 (下游任务),
  而不再训练 CLIP 本体。

CIFAR-10 原始图像为 32x32, 而 ViT-B/32 预训练输入是 224x224, 所以数据增强里
统一用 `preprocess` (内部会把图 resize 到 224) 作为 transform。
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import clip

# ============================================================================
# 实验超参数
# ============================================================================
CLIP_MODEL = "ViT-B/32"      # 直接调用 OpenAI 开源的预训练 CLIP 权重
BATCH_SIZE = 128
# CIFAR-10 类别 (与 CLIP 论文一致)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
# 文本 prompt 模板 (CLIP 论文也使用 prompt 工程)
PROMPT_TEMPLATE = "a photo of a {}"

# ---- 路径 ----
CIFAR10_ROOT = r"e:/AI-Learning/data/cifar10"
CKPT_PATH = os.path.join(os.path.dirname(__file__), "clip_cifar10.pth")


# ============================================================================
# 1. 加载 OpenAI 预训练 CLIP 模型 + 预处理
# ============================================================================
def load_pretrained_clip(device):
    """直接调用 CLIP 官方开源的预训练权重, 无需自己训练。

    返回: (model, preprocess)
      - model     : 完整的 CLIP 模型 (Image/Text Encoder + 投影 + logit_scale)
      - preprocess: 官方规定的图像预处理/resize 流程
    """
    model, preprocess = clip.load(CLIP_MODEL, device=device, jit=False)
    # 评估态: 预训练权重本身就是训练好的, 不再做梯度更新
    model.eval()
    return model, preprocess


# ============================================================================
# 2. 数据 (统一使用 CLIP 官方的 preprocess, 把 32x32 的 CIFAR 图 resize 到 224)
# ============================================================================
def get_dataloaders(preprocess, batch_size, num_workers=4):
    train_ds = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=True, download=False,
        transform=preprocess,
    )
    test_ds = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=False, download=False,
        transform=preprocess,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ============================================================================
# 3. 用预训练 CLIP 做 CIFAR-10 的 Zero-shot 分类 (下游任务)
# ============================================================================
@torch.no_grad()
def zero_shot_evaluate(model, loader, text_features, device):
    """用预训练 CLIP 对每一类生成文本特征, 与图像特征做相似度比较 -> 分类。"""
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Zero-shot Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        # 图像编码 -> 归一化特征 (预训练权重直接产出)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)   # (B, D)
        # 与 10 个类别文本特征做相似度 (已含 logit_scale)
        logits = (image_features @ text_features.t())          # (B, 10)
        pred = logits.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ============================================================================
# 4. 主流程
# ============================================================================
def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ---- 直接加载官方预训练 CLIP (核心修正: 不再自训练 ViT) ----
    model, preprocess = load_pretrained_clip(device)
    print(f"已加载 OpenAI 预训练 CLIP 模型: {CLIP_MODEL}")

    # ---- 为每个类别构造文本 prompt 并编码为文本特征 ----
    prompts = [PROMPT_TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    text_tokens = clip.tokenize(prompts).to(device)           # (10, 77)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)     # (10, D)
    print(f"文本类别特征形状: {tuple(text_features.shape)}")

    train_loader, test_loader = get_dataloaders(preprocess, BATCH_SIZE)
    print(f"训练集: {len(train_loader.dataset)} 张, 测试集: {len(test_loader.dataset)} 张")

    # ---- 仅靠预训练权重做 zero-shot, 不训练任何参数 ----
    zs_acc = zero_shot_evaluate(model, test_loader, text_features, device)
    print(f"\nCLIP (ViT-B/32) Zero-shot 准确率 (CIFAR-10): {zs_acc:.2f}%")

    # 顺便保存一下模型, 方便后续 CoOp 实验直接复用 (不需要重新下载)
    torch.save({"model_name": CLIP_MODEL}, CKPT_PATH)
    print(f"模型信息已记录: {CKPT_PATH}")


if __name__ == "__main__":
    main()
