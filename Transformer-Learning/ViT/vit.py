"""ViT 复现"""

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================================
# 设备: 优先 XPU (Intel Arc), 其次 CUDA, 最后 CPU
# ============================================================================
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# 超参数
# ============================================================================
IMAGE_SIZE = 32
PATCH_SIZE = 4
IN_CHANNELS = 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2      # 64
NUM_TOKENS = NUM_PATCHES + 1                       # 65 (含 CLS)
HIDDEN_DIM = 128                                   # D
NUM_LAYERS = 4                                     # Encoder Layers
NUM_HEADS = 4                                      # 每头维度 128/4 = 32
MLP_HIDDEN_DIM = 256                               # 可用 --mlp-dim 512 切换
DROPOUT = 0.1

# CIFAR-100 (预训练) / CIFAR-10 (微调) 的数据目录与类别数
PRETRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FINETUNE_DATA_DIR = r"e:/AI-Learning/CNN-Learning/data/cifar10"
PRETRAIN_NUM_CLASSES = 100
FINETUNE_NUM_CLASSES = 10

PRETRAIN_CKPT = os.path.join(os.path.dirname(__file__), "vit_cifar100_pretrain.pth")
FINETUNE_CKPT = os.path.join(os.path.dirname(__file__), "vit_cifar10_finetune.pth")

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ----------------------------------------------------------------------------
# 1. Patch Embedding (论文 3.1: 切 patch -> 展平 -> 线性投影到 D 维)
# ----------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """把 (B,3,32,32) 切成 64 个 4x4 patch, 每个 patch 投影成 128 维 token。

    等价实现: 用 kernel=stride=patch_size 的 Conv2d 一次完成
    "展平 P^2*C=48 维 + 乘投影矩阵 E(48xD)"。
    """

    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=IN_CHANNELS, embed_dim=HIDDEN_DIM):
        super().__init__()
        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 卷积核大小=步长=patch_size, 所以每个卷积窗口恰好是一个不重叠 patch
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"输入尺寸应为 {self.image_size}x{self.image_size}, 实际 {H}x{W}"
        x = self.proj(x)              # (B, D, 8, 8): 8x8=64 个 patch 的 D 维表示
        x = x.flatten(2)              # (B, D, 64): 空间维展平成 patch 序列 （压缩最后两个空间维度）
        x = x.transpose(1, 2)         # (B, 64, D): 转成 Transformer 的 (B, L, D)   （做一轮维度交换，根据下标交换两个维度，从B * 128 * 64 转换成 B * 64 * 128）
        return x


# ----------------------------------------------------------------------------
# 2. 多头自注意力 (论文 A.1 / Transformer 3.2.2)
# ----------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads     # 128/4 = 32
        self.scale = self.head_dim ** -0.5         # 1/sqrt(d_k)

        # 一次性投影出 Q,K,V (等价三个独立 Linear, 但更省 kernel 调用)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)      # 多头融合的输出投影
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape                                 # L=65 (CLS + 64 patch)
        qkv = self.qkv(x)                                 # (B, L, 3D)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ViT 是双向自注意力, 无需 causal mask; 也没有 padding, 故不传 mask
        scores = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, L, L)
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                     # (B, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)         # 拼回 (B, L, D)
        out = self.proj_drop(self.proj(out))
        return out


# ----------------------------------------------------------------------------
# 3. MLP Block (论文式: MLP(x) = W2 * GELU(W1 x + b1) + b2)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, embed_dim=HIDDEN_DIM, hidden_dim=MLP_HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()                 # ViT 用 GELU, 而非原始 Transformer 的 ReLU
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


# ----------------------------------------------------------------------------
# 4. Encoder Layer (Pre-LN: LN -> MHA -> 残差 -> LN -> MLP -> 残差)
# ----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                 mlp_hidden_dim=MLP_HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)

    def forward(self, x):
        # 注意 LN 在子层"之前" (Pre-LN), 残差是干净的恒等路径, 深层更易收敛
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------------------------------------------------------
# 5. 完整 ViT
# ----------------------------------------------------------------------------
class ViT(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=IN_CHANNELS, num_classes=PRETRAIN_NUM_CLASSES,
                 embed_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 mlp_hidden_dim=MLP_HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches            # 64

        # [CLS] token: 可学习参数, 经 Encoder 后其输出作为整图全局表示
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 可学习位置编码 (论文 3.1: 1D learnable position embedding), 长度 65
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)                   # 最后一层 LN
        self.head = nn.Linear(embed_dim, num_classes)         # 分类器: 单层 Linear

        self._init_weights()

    def _init_weights(self):
        # 论文附录: pos_embed / cls_token 用小标准差截断正态初始化
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

    def forward_features(self, x):
        """返回 CLS token 经 Encoder + LN 后的全局特征 (B, D)。"""
        B = x.size(0)
        x = self.patch_embed(x)                               # (B, 64, D)

        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                        # (B, 65, D)
        x = self.pos_drop(x + self.pos_embed)                 # 加位置信息

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0]                                        # 取 CLS 位置 -> (B, D)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def reset_head(self, num_classes):
        """微调时替换分类头 (论文 3.1: 迁移时换成 D x K 的新零初始化头)。"""
        self.head = nn.Linear(self.head.in_features, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        return self


# ============================================================================
# 数据
# ============================================================================
# 各数据集自身的通道均值/方差
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(is_train: bool, mean, std):
    """ViT 在小数据集上极易过拟合, 故训练侧使用随机裁剪 + 水平翻转增强。"""
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_dataloaders(stage: str, batch_size: int, num_workers: int):
    """stage='pretrain' -> CIFAR-100 (自动下载); 'finetune' -> CIFAR-10 (本地已存在)。"""
    if stage == "pretrain":
        os.makedirs(PRETRAIN_DATA_DIR, exist_ok=True)
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        train_ds = datasets.CIFAR100(
            root=PRETRAIN_DATA_DIR, train=True, download=True,
            transform=build_transforms(True, mean, std),
        )
        test_ds = datasets.CIFAR100(
            root=PRETRAIN_DATA_DIR, train=False, download=True,
            transform=build_transforms(False, mean, std),
        )
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        # 本地已有 cifar-10-batches-py, 无需下载
        train_ds = datasets.CIFAR10(
            root=FINETUNE_DATA_DIR, train=True, download=False,
            transform=build_transforms(True, mean, std),
        )
        test_ds = datasets.CIFAR10(
            root=FINETUNE_DATA_DIR, train=False, download=False,
            transform=build_transforms(False, mean, std),
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
# 训练 / 评估
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()          # 每个 iteration 更新 (warmup + cosine)

        running_loss += loss.item() * labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs, base_lr):
    """线性 warmup + 余弦退火 (论文附录 B.1 的训练策略)。"""
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run(stage, args):
    device = get_device()
    print(f"使用设备: {device}")
    print(f"阶段: {stage}")

    num_classes = PRETRAIN_NUM_CLASSES if stage == "pretrain" else FINETUNE_NUM_CLASSES
    train_loader, test_loader = get_dataloaders(stage, args.batch_size, args.num_workers)
    print(f"训练集: {len(train_loader.dataset)} 张, 测试集: {len(test_loader.dataset)} 张, "
          f"类别数: {num_classes}")

    # ---- 构建模型 ----
    model = ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
        num_classes=num_classes, embed_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, mlp_hidden_dim=args.mlp_dim, dropout=DROPOUT,
    )

    # ---- 微调: 载入 CIFAR-100 预训练权重 (分类头形状不同, 需剔除后重建) ----
    if stage == "finetune":
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f"未找到预训练权重 {args.pretrained}, 请先运行: python vit.py --stage pretrain"
            )
        state = torch.load(args.pretrained, map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.reset_head(num_classes)
        print(f"已加载预训练权重: {args.pretrained}")
        print(f"  未匹配(新建)参数: {list(missing)} | 多余参数: {list(unexpected)}")

    model = model.to(device)
    print(f"ViT 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print(f"配置: Patch={PATCH_SIZE} Patches={NUM_PATCHES} Tokens={NUM_TOKENS} "
          f"D={HIDDEN_DIM} Layers={NUM_LAYERS} Heads={NUM_HEADS} "
          f"MLP={args.mlp_dim} Dropout={DROPOUT}")

    # ---- 优化器: AdamW + label smoothing (ViT 标准配方) ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer, args.epochs, len(train_loader), args.warmup_epochs, args.lr
    )

    ckpt_path = PRETRAIN_CKPT if stage == "pretrain" else FINETUNE_CKPT
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, args.grad_clip
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | LR: {lr:.2e}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> 最佳模型已保存: {ckpt_path} (acc={best_acc:.2f}%)")

    print(f"\n{stage} 完成. 最佳测试准确率: {best_acc:.2f}%")
    return best_acc


def parse_args():
    p = argparse.ArgumentParser(description="ViT 复现: CIFAR-100 预训练 + CIFAR-10 微调")
    p.add_argument("--stage", choices=["pretrain", "finetune", "all"], default="all",
                   help="pretrain=CIFAR-100 预训练; finetune=CIFAR-10 微调; all=依次执行")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=30,
                   help="stage=all 时微调阶段的 epoch 数")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="预训练学习率")
    p.add_argument("--finetune-lr", type=float, default=1e-4,
                   help="微调学习率 (通常比预训练小一个量级)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlp-dim", type=int, default=MLP_HIDDEN_DIM, choices=[256, 512],
                   help="MLP 隐藏层维度, 论文风格为 D 的 2~4 倍")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pretrained", type=str, default=PRETRAIN_CKPT,
                   help="微调时载入的预训练权重路径")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage in ("pretrain", "all"):
        run("pretrain", args)

    if args.stage in ("finetune", "all"):
        # 微调阶段: 更小的 lr、更少的 epoch、更短的 warmup
        if args.stage == "all":
            args.epochs = args.finetune_epochs
        args.lr = args.finetune_lr
        args.warmup_epochs = min(args.warmup_epochs, max(1, args.epochs // 10))
        run("finetune", args)


if __name__ == "__main__":
    main()
