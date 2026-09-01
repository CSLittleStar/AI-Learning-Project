"""
MaPLe 在 CIFAR-10 上的下游分类实验 (train / val / test)

复刻 MaPLe (Multi-modal Prompt Learning) 论文思想:
- 文本侧: 学习一段可训练 context (ctx), 并在 Text Transformer 的多个深层注入 compound prompts;
- 视觉侧: 把文本 ctx 投影到视觉维度(768), 在 Vision Transformer 的多个深层注入视觉 deep prompts;
- 跨模态耦合: 视觉 deep prompts 由文本 ctx 经线性层投影得到。

本脚本:
1. 调用 CLIP 预训练权重 (ViT-B/32);
2. 冻结 CLIP 的 image encoder / text encoder 主干, 仅训练 Prompt Learner;
3. 把 CIFAR-10 划分为 训练集(45000) / 验证集(5000) / 测试集(10000);
4. 完整执行 train + val EarlyStopping + test, 并保存最优权重。

运行:
    python maple_cifar10.py
依赖: torch, torchvision, pillow, clip(openai)
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip as _clip  # openai CLIP, 用于 tokenize / 常量

# 让脚本可导入同目录的 MaPLe 改版 CLIP 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_maple import load_clip_to_cpu  # noqa: E402

DATA_ROOT = r"E:\AI-Learning\data"                 # CIFAR-10 数据根目录
CLASSNAMES_FILE = r"E:\AI-Learning\CLIP-Learning\CLIP\cifar10_classnames.txt"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DEVICE = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- 超参数 -----------------------
BACKBONE = "ViT-B/32"      # CLIP 视觉/文本骨干
N_CTX = 2                  # 文本侧 prompt 长度 (MaPLe 论文用 2)
N_CTX_VISION = 2           # 视觉侧 prompt 长度
CTX_INIT = "a photo of a"  # 可学习 ctx 的初始化文本
N_CLASS = 10
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0035
WEIGHT_DECAY = 5e-4
PATIENCE = 8               # 验证集早停耐心


# ============================================================================
# 1. Prompt Learner (多模态)
# ============================================================================
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg_ctx, classnames, clip_model):
        super().__init__()
        n_ctx = cfg_ctx["n_ctx"]
        n_ctx_v = cfg_ctx["n_ctx_vision"]
        # Prompt Learner 参数使用 float32 训练, 进入 CLIP 的 half 计算图时再转换
        ctx_dim = clip_model.token_embedding.weight.shape[1]  # 文本侧维度 512 (ViT-B)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, "CLIP 输入分辨率需为 224"

        # ----- 文本侧可学习 ctx -----
        if cfg_ctx["ctx_init"]:
            # 用 CTX_INIT 的词嵌入初始化 ctx 向量
            ctx_init = cfg_ctx["ctx_init"]
            ctx_init_tokens = _clip.tokenize(ctx_init).to(DEVICE)
            with torch.no_grad():
                embedding = clip_model.token_embedding(ctx_init_tokens).float()
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].clone()
            prompt_n_ctx = n_ctx
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_n_ctx = n_ctx

        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, ctx_dim) float32
        self.prompt_n_ctx = prompt_n_ctx

        # ----- 文本侧深层 compound prompts (跨模态共享, 第0维=1) -----
        # MaPLe 官方中文本深层 prompt 也是 class-specific, 但为避免视觉侧注入时
        # expand 到 batch 维冲突, 此处统一采用共享形式 (n_ctx, ctx_dim)。
        # 浅层 ctx 本就共享; 类区分度由 class token 提供。
        prompt_deep = []
        for i in range(12 - 1):  # 文本 transformer 共 12 层, 第 0 层不注入, 深层共 11
            p = torch.empty(1, n_ctx, ctx_dim)
            nn.init.normal_(p, std=0.02)
            prompt_deep.append(nn.Parameter(p))
        self.compound_prompts_text = nn.ParameterList(prompt_deep)

        # ----- 视觉侧 deep prompts (由文本投影) -----
        single_layer = nn.Linear(ctx_dim, 768, bias=False)
        self.compound_prompt_projections = nn.ModuleList(
            [single_layer for _ in range(11)]
        )
        self.n_ctx_v = n_ctx_v
        self.n_cls = N_CLASS

    def construct_prompts(self, ctx, prefix, suffix, use_prefix=False):
        # 文本 token 结构: [CLS] <ctx> <class token> <eot>
        # ctx: (NCLS, n_ctx, dim); prefix: (NCLS, 1, dim); suffix: (NCLS, *, dim)
        if use_prefix:
            p = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            p = torch.cat([ctx, suffix], dim=1)
        return p

    def forward(self):
        ctx = self.ctx.to(DEVICE)
        # 文本侧: 把 ctx 扩展到每个类 (NCLS, n_ctx, dim)
        ctx_text = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        compound_prompts_text = [p.to(DEVICE) for p in self.compound_prompts_text]

        # 视觉侧: 用第 0 层文本 ctx 作为 shared_ctx, 投影到视觉维度
        shared_ctx = self.compound_prompt_projections[0](ctx).unsqueeze(0)  # (1, n_ctx, 768)
        # 视觉深层: 把文本 compound prompts 投影到视觉维度
        compound_deeper_prompts_vision = []
        for layer in range(1, 12):
            if layer - 1 < len(compound_prompts_text):
                p = compound_prompts_text[layer - 1]
                p = self.compound_prompt_projections[layer - 1](p)
                compound_deeper_prompts_vision.append(p)
            else:
                compound_deeper_prompts_vision.append(None)

        return ctx_text, compound_prompts_text, shared_ctx, compound_deeper_prompts_vision


# ============================================================================
# 2. Custom CLIP (MaPLe)
# ============================================================================
class CustomCLIP(nn.Module):
    def __init__(self, cfg_ctx, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg_ctx, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.classnames = classnames

        # 构造每类的 token ids (MaPLe: 文本 token 由 [CLS] + ctx + classname + eot 组成)
        self.register_buffer("tokenized_prompts", self._build_tokenized_prompts())

    def _build_tokenized_prompts(self):
        n_ctx = self.prompt_learner.prompt_n_ctx
        prompts = [f"{' '.join(['X'] * n_ctx)} {name}." for name in self.classnames]
        return _clip.tokenize(prompts).to(DEVICE)

    def build_text_features(self, tokenized_prompts, compound_prompts_deeper_text, ctx_text):
        # MaPLe 文本编码:
        #   token 序列 = [CLS] <可学习 ctx> <class token...> <eot>
        #   用可学习 ctx 替换占位 token 的前 n_ctx 个位置 (紧接 CLS 之后)
        x = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # (NCLS, L, D)
        n_ctx = ctx_text.shape[1]
        x[:, 1: 1 + n_ctx, :] = ctx_text  # 注入可学习 context
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.clip_model.transformer(x, compound_prompts_deeper_text)
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] \
            @ self.clip_model.text_projection
        return x

    def forward(self, image):
        ctx_text, compound_prompts_text, shared_ctx, compound_deeper_prompts_vision = \
            self.prompt_learner()
        image_features = self.clip_model.encode_image_with_prompts(
            image, shared_ctx, compound_deeper_prompts_vision)
        text_features = self.build_text_features(
            self.tokenized_prompts, compound_prompts_text, ctx_text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


# ============================================================================
# 3. 数据与划分
# ============================================================================
def build_transforms():
    return Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def prepare_datasets():
    transform = build_transforms()
    trainval = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_set = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)

    # 训练集 45000 / 验证集 5000 (按类分层抽样, 保证分布一致)
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    targets = np.array(trainval.targets)
    for c in range(N_CLASS):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = len(idx) // 10  # 每类 5000/10 = 500
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_set = Subset(trainval, train_idx)
    val_set = Subset(trainval, val_idx)
    print(f"数据集划分 -> 训练: {len(train_set)}  验证: {len(val_set)}  测试: {len(test_set)}")
    return train_set, val_set, test_set


# ============================================================================
# 4. 评测
# ============================================================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ============================================================================
# 5. 训练主流程
# ============================================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    classnames = [l.strip() for l in open(CLASSNAMES_FILE, encoding="utf-8") if l.strip()]

    print(f"设备: {DEVICE} | 骨干: {BACKBONE} | 加载 CLIP 预训练权重 ...")
    clip_model = load_clip_to_cpu(BACKBONE, maple_length=N_CTX).to(DEVICE)

    cfg_ctx = {
        "n_ctx": N_CTX,
        "n_ctx_vision": N_CTX_VISION,
        "ctx_init": CTX_INIT,
        "use_class_specific": True,   # 文本深层 prompt 按类独立
    }

    model = CustomCLIP(cfg_ctx, classnames, clip_model).to(DEVICE)

    # 冻结 CLIP 主干, 仅训练 Prompt Learner
    for name, p in model.named_parameters():
        if "prompt_learner" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数量: {sum(p.numel() for p in trainable):,}")

    train_set, val_set, test_set = prepare_datasets()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    optimizer = torch.optim.SGD(trainable, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc, wait = 0.0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
        scheduler.step()

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        history.append({"epoch": epoch, "loss": running_loss / n,
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"[Epoch {epoch:02d}/{EPOCHS}] loss={running_loss/n:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={scheduler.get_last_lr()[0]:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save({
                "epoch": epoch,
                "best_val_acc": best_acc,
                "prompt_learner": model.prompt_learner.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, "maple_cifar10_best.pt"))
            print(f"  -> 保存最优权重 (val_acc={best_acc:.4f})")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  早停: 验证集 {PATIENCE} 轮未提升")
                break

    print(f"\n训练完成, 用时 {time.time()-t0:.1f}s, 最优 val_acc={best_acc:.4f}")
    with open(os.path.join(CHECKPOINT_DIR, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ---- 测试 ----
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "maple_cifar10_best.pt"), map_location=DEVICE)
    model.prompt_learner.load_state_dict(ckpt["prompt_learner"])
    test_acc = evaluate(model, test_loader)
    per_class = evaluate_per_class(model, test_loader, classnames)
    print(f"\n[TEST] 最优权重在 CIFAR-10 测试集上准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    for name, acc in zip(classnames, per_class):
        print(f"    {name:15s}: {acc:.4f}")

    with open(os.path.join(CHECKPOINT_DIR, "test_result.json"), "w", encoding="utf-8") as f:
        json.dump({"test_acc": test_acc, "per_class": dict(zip(classnames, per_class))},
                  f, ensure_ascii=False, indent=2)


def evaluate_per_class(model, loader, classnames):
    model.eval()
    correct = [0] * N_CLASS
    total = [0] * N_CLASS
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred = model(images).argmax(dim=1)
            for p, t in zip(pred.cpu().numpy(), labels.cpu().numpy()):
                if p == t:
                    correct[t] += 1
                total[t] += 1
    return [correct[c] / total[c] for c in range(N_CLASS)]


if __name__ == "__main__":
    main()
