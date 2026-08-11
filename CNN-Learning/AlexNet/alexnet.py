"""AlexNet 基本神经网络框架（PyTorch 实现）。

原始论文: Krizhevsky et al., "ImageNet Classification with Deep Convolutional
Neural Networks", NeurIPS 2012。

本文件提供:
- 特征提取部分 (5 个卷积层 + 池化 + LRN + Dropout)
- 分类部分 (3 个全连接层 + Dropout)
"""
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ------------------- 特征提取 (卷积部分) -------------------
        self.features = nn.Sequential(
            # Conv1: 11x11 卷积, 步幅 4, padding 2 -> 输出 96 通道
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                      stride=4, padding=2, groups=1),
            nn.ReLU(inplace=True),
            # LRN: 局部响应归一化
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            # MaxPool: 3x3, 步幅 2 (重叠池化)
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27

            # Conv2: 5x5, padding 2 -> 256 通道
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13

            # Conv3: 3x3, padding 1 -> 384 通道
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True),

            # Conv4: 3x3, padding 1 -> 384 通道
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),

            # Conv5: 3x3, padding 1 -> 256 通道
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
        )

        # ------------------- 分类部分 (全连接) -------------------
        # 经过特征提取后, 256 通道, 空间尺寸 6x6 -> 展平为 256*6*6 = 9216
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            # 最后一层输出原始分数 (logits), 由损失函数负责 softmax
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化。

        原论文使用 weight ~ N(0, 0.01) + bias 常数，但那需要 ImageNet 百万级
        数据 + 长时间训练才能收敛。对于 CIFAR-10 这类小数据集，直接采用
        PyTorch 的默认初始化（Conv2d 用 kaiming_uniform_，Linear 用
        kaiming_uniform_ + uniform bias）收敛更快、更稳定，且不改变网络结构。
        """
        # 无需手动初始化：nn.Conv2d / nn.Linear 构造时已带 PyTorch 默认初始化。
        # 这里保留一个 reset 以便显式确认（默认初始化即最优）。
        pass

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

# 数据集根目录 (本仓库 data/cifar10 下包含 cifar-10-batches-py 文件夹,
# torchvision 的 CIFAR10 要求 root 指向包含 cifar-10-batches-py 的目录)
DATA_DIR = r"e:/AI-Learning/CNN-Learning/data/cifar10"

# 训练超参数
NUM_CLASSES = 10
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01     # 原论文用 0.01；相对 1e-3 更利于小权重快速收敛
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 模型保存路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "alexnet_cifar10.pth")


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

    CIFAR-10 原图是 32x32, 而 AlexNet 原始结构期望 224x224 输入,
    因此这里统一上采样到 224x224 (用双三次插值, 与原论文 resize 思路一致)。

    数据增强包含两类 (仅训练时启用):
      1. 几何增强: 随机裁剪 + 水平翻转 (论文提到的平移/镜像)。
      2. RGB 强度扰动: 对应论文 4.1 节 "Altering the intensities of the
         RGB channels", 这里用 ColorJitter 对亮度/对比度/饱和度做随机线性
         扰动 (对 PCA 通道强度扰动的近似)。
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 通道均值
        std=(0.2023, 0.1994, 0.2010),   # CIFAR-10 通道标准差
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2),   # 平移（几何增强，削弱）
            transforms.RandomHorizontalFlip(),      # 水平翻转（几何增强）
            transforms.ToTensor(),                  # 图像转张量，像素值从[0,255]归一化到[0,1]，变成(C,H,W)格式
            # RGB 通道强度扰动（削弱: 从 0.4 降到 0.1, 减小对拟合的干扰）:
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.Resize(224, antialias=True), # 32x32 -> 224x224（抗锯齿更好）
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True), # 32x32 -> 224x224（抗锯齿更好）
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

    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    print(
        "AlexNet Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    criterion = nn.CrossEntropyLoss()
    # 原论文使用 SGD + momentum，并配合 weight decay 正则化
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    # 学习率余弦退火（论文中使用类似策略逐步降低 lr）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
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

        # 保存当前最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved to {MODEL_PATH} (acc={best_acc:.2f}%)")

    print(f"\nTraining finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
