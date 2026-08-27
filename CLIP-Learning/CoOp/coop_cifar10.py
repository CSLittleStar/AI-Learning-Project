"""CoOp 实验 (CIFAR-10) —— 基于 OpenAI 开源预训练 CLIP 权重

与 CLIP 实验的区别 (核心来自论文 "Learning to Prompt for Vision-Language Models"):
- 冻结 CLIP 的 Image Encoder 和 Text Encoder 全部参数 (含投影层与 logit_scale)
- 不再使用人工设计的固定文本 prompt ("a photo of a {class}"),
  而是把 prompt 的前 N 个词替换为可学习的连续向量 (context vectors) [V]_1 ... [V]_M
- 文本输入变为:  [V]_1 [V]_2 ... [V]_M [CLASS]
  * [V] 为可学习向量 (维度 = CLIP Text Transformer 的 width, 即 512)
  * [CLASS] 为类别名的 token embedding (由冻结的 CLIP token_embedding 查表得到)
- 仅用少量标注数据 + 分类损失 反向传播更新 [V] (prompt context)

关键修正 (相对此前的误区):
- 不需要自训练一个 ViT / Text Encoder 来当 CLIP;
- 直接用 `model, preprocess = clip.load("ViT-B/32")` 加载 OpenAI 开源预训练权重,
  再在其上训练 CoOp 的可学习 prompt。CoOp 本身并不训练 CLIP, 整个预训练模型全部冻结。

本实现默认 Unified Context (所有类别共享同一组 [V]), 并额外提供 Class-Specific
Context (CSC, 每个类别独立 [V]) 的开关。
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import clip

# ============================================================================
# 实验超参数
# ============================================================================
CLIP_MODEL = "ViT-B/32"      # 直接调用 OpenAI 开源的预训练 CLIP 权重
BATCH_SIZE = 128
EPOCHS = 30
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.02
GRAD_CLIP = 1.0
LR_MIN = 1e-5

# ---- CoOp 专属 ----
N_CTX = 4                   # prompt 中可学习 context 向量的个数 [V]_1...[V]_M
CSC = False                 # False=Unified Context (共享), True=Class-Specific Context
INIT_CTX = "uniform"        # context 初始化方式: "uniform"(用模板词 embedding) 或 "zero"
CTX_LR = 2e-3              # prompt context 学习率 (CoOp 论文标准值, 避免 fp16 下发散)
TEMP_LR = 1e-5              # 温度 logit_scale 的学习率 (可选微调)
LEARNING_RATE = CTX_LR

# ---- 路径 ----
CIFAR10_ROOT = r"e:/AI-Learning/data/cifar10"
CKPT_PATH = os.path.join(os.path.dirname(__file__), "coop_cifar10.pth")

# CIFAR-10 类别
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
# CoOp 的类别名 token (用模板词 "a photo of a" 初始化 context, 但类别本身只取类名)
PROMPT_TEMPLATE = "a photo of a {}"


# ============================================================================
# 1. 加载 OpenAI 预训练 CLIP (冻结, 仅用于编码)
# ============================================================================
def load_pretrained_clip(device):
    model, preprocess = clip.load(CLIP_MODEL, device=device, jit=False)
    model.eval()
    return model, preprocess


# ============================================================================
# 2. CoOp Prompt Context —— 可学习的连续向量
# ============================================================================
class PromptContext(nn.Module):
    """可学习的 prompt context 向量。

    Unified Context: shape (N_CTX, WIDTH), 所有类别共享。
    Class-Specific : shape (NUM_CLASS, N_CTX, WIDTH), 每个类别独立。

    关键: context 用 CLIP 模板词 "a photo of a" 的 embedding 初始化
    (CoOp 论文的标准做法), 而不是随机初始化。原因——
    冻结的 Text Encoder 是在 "a photo of a {class}" 这类完整文本上预训练的,
    若 context 用随机极小向量初始化, 进 Transformer 的序列分布与预训练严重脱节,
    text_features 会崩坏, 导致训练几乎不提升。用模板词 embedding 锚定初始
    分布后, text_features 起点即接近 CLIP 水平, 再微调即可小幅超越。
    """

    def __init__(self, n_ctx, width, n_cls, csc=False, init="uniform",
                 ctx_init_embed=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.csc = csc
        ctx_dim = (n_cls, n_ctx, width) if csc else (n_ctx, width)
        if ctx_init_embed is not None:
            # 用模板词 embedding 初始化 (首选): 取前 n_ctx 个词, 不足则补零
            emb = ctx_init_embed[:n_ctx]            # (min(n_ctx,len), W)
            ctx_vectors = torch.zeros(ctx_dim)
            ctx_vectors[:emb.size(0)] = emb
            if csc:
                ctx_vectors = ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1).clone()
        elif init == "zero":
            ctx_vectors = torch.zeros(ctx_dim)
        else:
            ctx_vectors = torch.empty(ctx_dim)
            nn.init.trunc_normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self, class_emb):
        """把 context 拼到每个类别 embedding 序列前面。

        class_emb: (NUM_CLASS, 1, WIDTH) 每个类别的 [CLASS] token embedding
        返回: (NUM_CLASS, N_CTX + 1, WIDTH)

        可学习 ctx 参数保持 fp32 (优化更稳定), 这里把 class_emb 对齐到 ctx 的
        dtype 再拼接, 避免 mixed dtype; 进入 CLIP transformer 前的统一 .type(self.dtype)
        在 encode_text_with_context 中处理。
        """
        if self.csc:
            ctx = self.ctx
        else:
            ctx = self.ctx.unsqueeze(0).expand(class_emb.size(0), -1, -1)
        class_emb = class_emb.to(ctx.dtype)              # 对齐可学习参数 dtype
        return torch.cat([ctx, class_emb], dim=1)


# ============================================================================
# 3. CoOp 模型 (冻结 CLIP 双塔, 仅训练 context)
# ============================================================================
class CoOp(nn.Module):
    def __init__(self, clip_model, n_cls, n_ctx=N_CTX, csc=CSC, init=INIT_CTX):
        super().__init__()
        self.clip = clip_model                       # 预训练 CLIP (冻结)
        self.n_ctx = n_ctx
        self.csc = csc

        # 取出 CLIP Text Transformer 的关键部件, 用于把 [V]+[CLASS] 编码成文本特征
        self.token_embedding = clip_model.token_embedding      # 冻结查表
        self.transformer = clip_model.transformer              # 冻结
        self.ln_final = clip_model.ln_final                    # 冻结
        self.text_projection = clip_model.text_projection      # 冻结
        self.logit_scale = clip_model.logit_scale             # 冻结 (沿用预训练值)
        self.dtype = clip_model.dtype

        ctx_init_embed = None
        if init == "uniform":
            # 用模板词 "a photo of a" 的 embedding 初始化 context (CoOp 标准做法)
            ctx_ids = clip.tokenize(
                PROMPT_TEMPLATE.replace("{}", "")
            ).squeeze(0).to(next(clip_model.parameters()).device)
            with torch.no_grad():
                ctx_init_embed = self.token_embedding(ctx_ids).float()  # (L, W)
        self.prompt = PromptContext(
            n_ctx, self.token_embedding.embedding_dim, n_cls,
            csc=csc, init=init, ctx_init_embed=ctx_init_embed,
        )

    def encode_image(self, images):
        # 输入图像对齐 CLIP 权重 dtype (fp16/fp32 皆可), 避免 layer_norm 的 mixed dtype 报错
        feat = self.clip.encode_image(images.type(self.dtype))
        return F.normalize(feat.float(), dim=-1)

    def encode_text_with_context(self, class_emb):
        """给定类别 embedding, 构造 CoOp prompt 并编码。

        class_emb: (NUM_CLASS, 1, WIDTH)
        返回: (NUM_CLASS, D) 的归一化文本特征

        对齐 CLIP 预训练: CLIP Text Encoder 取 [EOS] 后的特征来聚合整句语义,
        因此这里 [CLASS] 之后追加一个 [EOS] token 的 embedding 并取末位特征。
        """
        prompt_emb = self.prompt(class_emb)                  # (NUM_CLASS, N_CTX+1, W)
        NUM_CLASS = prompt_emb.shape[0]
        # 追加 [EOS] embedding (从预训练模型的 vocab 动态取 EOS id)
        eos_id = int(self.clip.token_embedding.weight.shape[0] - 1)
        eos_ids = torch.full((NUM_CLASS,), eos_id, dtype=torch.long,
                             device=prompt_emb.device)
        with torch.no_grad():
            eos_emb = self.token_embedding(eos_ids).unsqueeze(1)   # (C,1,W)
        # 全程与 CLIP 预训练权重 dtype (self.dtype, 通常为 fp16) 保持一致,
        # 避免 transformer 输出 (fp16) 与 text_projection (fp16) 再与 fp32 混算。
        prompt_emb = prompt_emb.type(self.dtype)
        eos_emb = eos_emb.type(self.dtype)
        prompt_emb = torch.cat([prompt_emb, eos_emb], dim=1)       # (C, N_CTX+2, W)

        x = prompt_emb
        # CLIP 的位置编码: 从 0 开始, 长度 = 序列长度
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        x = x + self.clip.positional_embedding[pos].type(self.dtype)
        x = x.permute(1, 0, 2)                               # (L, C, W)

        # CLIP 预训练 Text Transformer 的 resblock 自带 77x77 因果 attn_mask;
        # 当 CoOp prompt 序列长度 != 77 时尺寸不匹配, 这里临时替换为
        # 与当前 seq_len 对齐的下三角因果掩码, 跑完再恢复, 避免覆盖预训练权重。
        causal = torch.empty(seq_len, seq_len, device=x.device).fill_(float("-inf")).triu_(1).bool()
        attn_masks = [b.attn_mask for b in self.transformer.resblocks]
        for b in self.transformer.resblocks:
            b.attn_mask = causal
        try:
            x = self.transformer(x)
        finally:
            for b, m in zip(self.transformer.resblocks, attn_masks):
                b.attn_mask = m

        x = x.permute(1, 0, 2)                               # (C, L, W)
        x = self.ln_final(x).type(self.dtype)                # 保持 fp16
        # 取 [EOS] 位置 (序列末位) 特征来聚合整句语义, 与 CLIP 预训练一致
        eos_idx = seq_len - 1
        feat = x[:, eos_idx, :]                             # (C, W), fp16
        feat = feat @ self.text_projection                  # (C, D), fp16
        feat = feat.float()                                 # 转回 fp32 再做归一化
        return F.normalize(feat, dim=-1)

    def forward(self, images, class_emb):
        image_features = self.encode_image(images)                    # (B, D), fp32
        text_features = self.encode_text_with_context(class_emb)      # (NUM_CLASS, D), fp32
        # logit_scale 用 fp32 标量, 避免 fp16 下 100 倍放大导致 softmax 溢出 -> nan
        logit_scale = self.logit_scale.exp().float()
        logits = logit_scale * image_features @ text_features.t()      # (B, NUM_CLASS), fp32
        return logits


# ============================================================================
# 4. 类别 embedding (冻结查表, 不进入梯度)
# ============================================================================
def build_class_embeddings(coop, device):
    """为每个类别生成 [CLASS] token 的 embedding (冻结查表, 不学习)。

    返回 class_emb: (NUM_CLASS, 1, WIDTH)
    """
    class_tokens = []
    for c in CIFAR10_CLASSES:
        # CIFAR-10 类别均为单词, 取该单词的 token id
        tok = clip.tokenize(c).squeeze(0)
        # tokenize 形如 [<start>, w, <end>], 取中间的实际单词 token
        word_id = tok[1].item() if len(tok) > 1 else tok[0].item()
        class_tokens.append(word_id)
    tok_ids = torch.tensor(class_tokens, dtype=torch.long, device=device)
    with torch.no_grad():
        # 取 CLIP 权重 dtype (通常为 fp16), 与 encode_text_with_context 全程保持一致
        emb = coop.token_embedding(tok_ids).to(coop.dtype)     # (NUM_CLASS, W)
    emb = emb.unsqueeze(1)                              # (NUM_CLASS, 1, W)
    return emb


# ============================================================================
# 5. 数据 (统一用 CLIP 官方 preprocess 把 32x32 的 CIFAR 图 resize 到 224)
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
# 6. 学习率调度 (warmup + cosine)
# ============================================================================
def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    lr_min_ratio = LR_MIN / LEARNING_RATE

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 7. 评估 (CoOp 用分类准确率)
# ============================================================================
@torch.no_grad()
def evaluate(coop, loader, class_emb, device, use_amp=False, autocast_ctx=None):
    coop.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        if use_amp:
            with autocast_ctx():
                logits = coop(images, class_emb)        # (B, NUM_CLASS)
        else:
            logits = coop(images, class_emb)            # (B, NUM_CLASS)
        pred = logits.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ============================================================================
# 8. 训练 (仅优化 prompt context, CLIP 全部冻结)
# ============================================================================
def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ---- 直接加载官方预训练 CLIP (核心修正: 不再自训练 ViT) ----
    clip_model, preprocess = load_pretrained_clip(device)
    print(f"已加载 OpenAI 预训练 CLIP 模型: {CLIP_MODEL}")

    train_loader, test_loader = get_dataloaders(preprocess, BATCH_SIZE)
    print(f"训练集: {len(train_loader.dataset)} 张, 测试集: {len(test_loader.dataset)} 张")

    coop = CoOp(
        clip_model,
        n_cls=len(CIFAR10_CLASSES),
        n_ctx=N_CTX,
        csc=CSC,
        init=INIT_CTX,
    ).to(device)

    # ---- 冻结 CLIP 全部参数, 仅保留 prompt context 可训练 ----
    for name, p in coop.named_parameters():
        if "prompt." in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    coop.logit_scale.requires_grad = False   # 温度沿用预训练值
    print("已冻结 CLIP (Image/Text Encoder + 投影 + logit_scale), 仅 prompt context 可训练")

    class_emb = build_class_embeddings(coop, device)      # (NUM_CLASS, 1, W)
    print(f"类别 embedding 形状: {tuple(class_emb.shape)}")

    trainable = [p for p in coop.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable) / 1e3
    print(f"可训练参数量: {n_trainable:.2f} K (其余 CLIP 参数已全部冻结)")

    optimizer = optim.AdamW(
        trainable, lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer, EPOCHS, len(train_loader), WARMUP_EPOCHS
    )

    # ---- 混合精度 (AMP): XPU/GPU 上 CLIP 为 fp16, 训练时易数值溢出 -> loss=nan。
    #      用 autocast 做前向 fp16 加速, GradScaler 缩放梯度防止 fp16 下溢/溢出。
    #      (CPU 设备无 amp, 直接用 fp32 训练, 同样稳定)
    #      统一使用 torch.amp (PyTorch 2.x 推荐 API, 兼容 xpu/cuda)。
    use_amp = device != "cpu"
    if use_amp:
        autocast_ctx = lambda: torch.amp.autocast(device, dtype=torch.float16)
        scaler = torch.amp.GradScaler(device)
    else:
        autocast_ctx = None
        scaler = None

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        coop.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast_ctx():
                    logits = coop(images, class_emb)          # (B, NUM_CLASS)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                if GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = coop(images, class_emb)              # (B, NUM_CLASS)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                if GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ctx_grad_norm = (coop.prompt.ctx.grad.norm().item()
                         if coop.prompt.ctx.grad is not None else 0.0)
        train_loss = running_loss / total
        acc = evaluate(coop, test_loader, class_emb, device,
                       use_amp=use_amp, autocast_ctx=autocast_ctx)
        lr = optimizer.param_groups[0]["lr"]
        tau = coop.logit_scale.exp().item()
        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {acc:.2f}% | "
            f"LR: {lr:.2e} | tau: {tau:.2f} | "
            f"ctx_grad: {ctx_grad_norm:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(coop.state_dict(), CKPT_PATH)
            print(f"  -> 最佳模型已保存: {CKPT_PATH} (acc={best_acc:.2f}%)")

    print(f"\n训练完成. 最佳 CoOp 准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
