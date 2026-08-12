"""VGG-16 神经网络框架（PyTorch 实现）。"""
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class VGG16(nn.Module):
    """VGG-16 神经网络框架。"""

    def __init__(self, num_classes: int = 10, init_method: str = "he") -> None:
        super().__init__()
        self.num_classes = num_classes

        # ------------------- 特征提取 (卷积部分) -------------------
        # 每个 block 说明: (out_channels, num_conv) — 卷积后接一个 2x2 最大池化
        # 输入 224x224x3
        self.features = nn.Sequential(
            # Block 1: 224*224*3 -> 224*224*64 -> 112*112*64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112*112*64 -> 112*112*128 -> 56*56*128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56*56*128 -> 56*56*256 -> 28*28*256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28*28*256 -> 28*28*512 -> 14*14*512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14*14*512 -> 7*7*512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 经 5 次池化后: 224 -> 7x7, 通道 512 -> 展平 512 * 7 * 7 = 25088

        # ------------------- 分类部分 (全连接) -------------------
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # 最后一层输出原始分数 (logits), 由损失函数负责 softmax
            nn.Linear(4096, num_classes),
        )

        if init_method == "he":
            self._initialize_weights_he()
        elif init_method == "paper":
            self._initialize_weights_paper()
        else:
            raise ValueError(f"Unknown init_method: {init_method!r} (use 'he' or 'paper')")

    def _initialize_weights_he(self) -> None:
        """Kaiming/He 初始化 (推荐, 默认)。

        PyTorch 官方 torchvision 的 VGG 默认即用 kaiming_normal_ + ReLU。
        He 初始化保证前向输出方差在前向传播中保持稳定 (对 ReLU 网络),
        使深层网络的梯度不至于消失/爆炸, 在小数据集上能稳定收敛。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def _initialize_weights_paper(self) -> None:
        """按论文附录 A 的 small-random 初始化。

        权重从 N(0, 0.01) 采样, bias 设为 0。该方案在 ImageNet 上配合预训练/长训练周期可用,
        但在 CIFAR-10 上从头训练时初始权重过小 (信号逐层衰减), 会导致 loss 卡在
        随机基线 (ln(10)≈2.3) 附近无法收敛。仅作论文复现对照。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# ======================================================================
#  训练 / 测试脚本 (CIFAR-10, 输入上采样到 224x224)
# ======================================================================

# CIFAR-10 的 10 个类别名称
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# 数据集根目录
DATA_DIR = r"e:/AI-Learning/CNN-Learning/data/cifar10"

# 训练超参数
NUM_CLASSES = 10
NUM_EPOCHS = 50          # 原 10 轮不足以在 CIFAR-10 上收敛, 增大到 50
BATCH_SIZE = 64          # 在 XPU 上 224x224 输入可尝试 64 (显存不足则调回 32)
LEARNING_RATE = 0.001    # 原 0.01 在 CIFAR-10 小数据集上过大, 降为 0.001 稳定收敛
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 模型保存路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vgg16_cifar10.pth")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 与准确率。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在给定数据上评估模型，返回平均 loss 与准确率。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Eval"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def build_transforms(is_train: bool):
    """构造数据增强 / 预处理。

    CIFAR-10 原图是 32x32, 而 VGG-16 原始结构期望 224x224 输入,
    因此这里统一上采样到 224x224。
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 通道均值
        std=(0.2023, 0.1994, 0.2010),   # CIFAR-10 通道标准差
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            normalize,
        ])


def get_dataloaders():
    """加载 CIFAR-10 训练集与测试集。"""
    train_ds = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=False,
        transform=build_transforms(is_train=True),
    )
    test_ds = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False,
        transform=build_transforms(is_train=False),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, test_loader


def main():
    # Intel Arc B580 (XPU) 优先，其次 CUDA，最后 CPU
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = VGG16(num_classes=NUM_CLASSES, init_method="he").to(device)
    print(
        "VGG-16 Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 40], gamma=0.1
    )

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% "
            f"| LR: {lr:.2e}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved to {MODEL_PATH} (acc={best_acc:.2f}%)")

    print(f"\nTraining finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
