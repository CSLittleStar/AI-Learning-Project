"""CLIP 小型实验 (CIFAR-10)"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ViT 实现所在位置: e:/AI-Learning/Transformer-Learning/ViT/vit.py
sys.path.insert(0, r"e:/AI-Learning/Transformer-Learning/ViT")
from vit import (  # noqa: E402
    PatchEmbedding,
    MultiHeadSelfAttention,
    MLP,
    EncoderLayer,
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGE_SIZE,
    PATCH_SIZE,
    IN_CHANNELS,
    get_device,
)

# ============================================================================
# 实验超参数
# ============================================================================
# ---- Image Encoder (ViT) ----
IMG_DIM = 128          # ViT 嵌入维度 D (参考 vit.py 的 HIDDEN_DIM)
IMG_LAYERS = 4
IMG_HEADS = 4
IMG_MLP = 512          # 参考 vit.py 可切换的 mlp_dim=512
IMG_DROPOUT = 0.1

# ---- Text Encoder ----
TEXT_LAYERS = 4
TEXT_WIDTH = 256
TEXT_HEADS = 4
TEXT_MLP = 512
TEXT_DROPOUT = 0.1
TEXT_CTX = 32          # 文本序列最大长度 (含 [CLS]/[EOS])

# ---- 投影 / 训练 ----
PROJECTION_DIM = 256
LEARNING_RATE = 1e-3        # 峰值 LR (warmup 后达到)
LR_MIN = 1e-5              # cosine 末尾保留的最小 LR, 避免过早归零
BATCH_SIZE = 128
EPOCHS = 30
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0

# ---- 数据 ----
CIFAR10_ROOT = r"e:/AI-Learning/data/cifar10"
CKPT_PATH = os.path.join(os.path.dirname(__file__), "clip_cifar10.pth")

# CIFAR-10 类别 (与 vit.py 一致)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
# 文本 prompt 模板 (CLIP 论文也使用 prompt 工程, 这里用最简单模板)
PROMPT_TEMPLATE = "a photo of a {}"


# ============================================================================
# 1. Image Encoder (ViT) —— 输出经投影映射到统一空间
# ============================================================================
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim=PROJECTION_DIM):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, IMG_DIM
        )
        num_patches = self.patch_embed.num_patches  # 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, IMG_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, IMG_DIM))
        self.pos_drop = nn.Dropout(IMG_DROPOUT)
        self.layers = nn.ModuleList([
            EncoderLayer(IMG_DIM, IMG_HEADS, IMG_MLP, IMG_DROPOUT)
            for _ in range(IMG_LAYERS)
        ])
        self.norm = nn.LayerNorm(IMG_DIM)
        self.proj = nn.Linear(IMG_DIM, proj_dim)  # 投影到统一空间
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_feat = x[:, 0]                 # (B, IMG_DIM)
        return self.proj(cls_feat)         # (B, proj_dim)


# ============================================================================
# 2. Text Encoder (Transformer) —— 输出经投影映射到统一空间
# ============================================================================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, ctx=TEXT_CTX, proj_dim=PROJECTION_DIM):
        super().__init__()
        self.ctx = ctx
        self.token_embed = nn.Embedding(vocab_size, TEXT_WIDTH)
        self.pos_embed = nn.Parameter(torch.zeros(1, ctx, TEXT_WIDTH))
        self.drop = nn.Dropout(TEXT_DROPOUT)
        self.layers = nn.ModuleList([
            EncoderLayer(TEXT_WIDTH, TEXT_HEADS, TEXT_MLP, TEXT_DROPOUT)
            for _ in range(TEXT_LAYERS)
        ])
        self.norm = nn.LayerNorm(TEXT_WIDTH)
        self.proj = nn.Linear(TEXT_WIDTH, proj_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens):
        # tokens: (B, L) 已经过 padding/截断
        B, L = tokens.shape
        tok = self.token_embed(tokens)             # (B, L, W)
        pos = self.pos_embed[:, :L, :]
        x = self.drop(tok + pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # 取 [EOS] (序列末尾, 即最后一个有效 token) 的特征作为整句表示
        feat = x[torch.arange(B), (tokens != 0).sum(1) - 1]  # (B, W)
        return self.proj(feat)                      # (B, proj_dim)


# ============================================================================
# 3. CLIP 模型 (双塔 + 可学习温度)
# ============================================================================
class CLIP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size)
        # 可学习温度 logit_scale (CLIP 论文用 exp 绑定到 [ln(1/100), ln(100)])
        self.logit_scale = nn.Parameter(torch.log(torch.ones(1) * 1.0 / 0.07))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        # L2 归一化 (论文在对比前对特征做 normalize)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        # 相似度 = exp(tau) * I·T^T
        logit_scale = self.logit_scale.exp().clamp(1 / 100, 100)
        logits = logit_scale * image_features @ text_features.t()
        return logits


def clip_loss(logits):
    """对称的对比损失 (image->text 与 text->image 两个方向)。"""
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# ============================================================================
# 4. 极简 tokenizer + 英文词表 (仅覆盖 CIFAR-10 类别单词 + 模板词)
# ============================================================================
def build_vocab():
    words = set()
    for c in CIFAR10_CLASSES:
        for w in c.split():
            words.add(w)
    for w in PROMPT_TEMPLATE.replace("{}", "").split():
        words.add(w)
    # pad 占 0, 其余单词从 1 开始连续编号, eos 追加在最后
    vocab = {"<pad>": 0}
    for i, w in enumerate(sorted(words)):
        vocab[w] = i + 1
    vocab["<eos>"] = len(vocab)     # 紧跟在单词之后
    return vocab


def encode_prompts(vocab):
    """为每个类别生成一条文本 token 序列 (长度截断到 TEXT_CTX)。"""
    prompts = [PROMPT_TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    seqs = []
    for p in prompts:
        toks = [vocab[w] for w in p.split() if w in vocab]
        # 加 [EOS]
        toks.append(vocab["<eos>"])
        if len(toks) > TEXT_CTX:
            toks = toks[:TEXT_CTX]
        seqs.append(toks)
    max_len = max(len(s) for s in seqs)
    # padding 到 max_len (pad token = 0), forward 中会取最后一个非 pad 位置
    padded = [s + [0] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)


# ============================================================================
# 5. 数据
# ============================================================================
def build_transforms(is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(batch_size, num_workers=4):
    train_ds = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=True, download=False,
        transform=build_transforms(True),
    )
    test_ds = datasets.CIFAR10(
        root=CIFAR10_ROOT, train=False, download=False,
        transform=build_transforms(False),
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
# 6. 学习率调度 (warmup + cosine)
# ============================================================================
def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    # cosine 下限: 末尾保留 LR_MIN / LEARNING_RATE 比例, 不衰减到 0
    lr_min_ratio = LR_MIN / LEARNING_RATE

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        # 从 1.0 余弦退火到 lr_min_ratio
        return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 7. Zero-shot 评估
# ============================================================================
@torch.no_grad()
def zero_shot_evaluate(model, loader, text_tokens, device):
    model.eval()
    correct, total = 0, 0
    # 预计算文本特征 (10 个类别)
    text_tokens = text_tokens.to(device)
    text_feat = model.text_encoder(text_tokens)
    text_feat = F.normalize(text_feat, dim=-1)     # (10, D)
    logit_scale = model.logit_scale.exp().clamp(1 / 100, 100)
    for images, labels in tqdm(loader, desc="Zero-shot Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        image_feat = model.image_encoder(images)
        image_feat = F.normalize(image_feat, dim=-1)   # (B, D)
        logits = logit_scale * image_feat @ text_feat.t()  # (B, 10)
        pred = logits.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ============================================================================
# 8. 训练
# ============================================================================
def main():
    device = get_device()
    print(f"使用设备: {device}")

    vocab = build_vocab()
    text_tokens = encode_prompts(vocab).to(device)  # (10, L), 预置到 device
    print(f"词表大小: {len(vocab)}, 文本序列长度: {text_tokens.shape[1]}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    print(f"训练集: {len(train_loader.dataset)} 张, 测试集: {len(test_loader.dataset)} 张")

    model = CLIP(vocab_size=len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CLIP 参数量: {n_params:.3f} M")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer, EPOCHS, len(train_loader), WARMUP_EPOCHS
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            # CLIP 标准对比: 每张图对应其自身类别的文本, 构成 B×B 相似度矩阵,
            # 对角线 (i,i) 才是正样本 (图像 i 与 "a photo of a <label_i>" 匹配)
            text_tokens_b = text_tokens[labels]                # (B, L)
            images_b = images                                 # (B, 3, H, W)

            optimizer.zero_grad()
            logits = model(images_b, text_tokens_b)            # (B, B)
            loss = clip_loss(logits)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total

        # ---- zero-shot eval ----
        zs_acc = zero_shot_evaluate(model, test_loader, text_tokens, device)
        lr = optimizer.param_groups[0]["lr"]
        tau = model.logit_scale.exp().item()
        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Zero-shot Acc: {zs_acc:.2f}% | "
            f"LR: {lr:.2e} | tau: {tau:.2f}"
        )

        if zs_acc > best_acc:
            best_acc = zs_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> 最佳模型已保存: {CKPT_PATH} (acc={best_acc:.2f}%)")

    print(f"\n训练完成. 最佳 Zero-shot 准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
