"""ResNet-18 (Deep Residual Learning for Image Recognition, He et al. 2016) 复现。"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = torch.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """瓶颈残差块: 1×1 -> 3×3 -> 1×1, 用于 ResNet-50/101/152 (论文 Fig.5 right)。
    1×1 负责降维与升维, 中间的 3×3 处理较小通道数, 最后1*1大幅降低计算量。
    （本文件按需求保留此实现代码，ResNet-18 不调用它，可复用于更深的变体。）
    """

    expansion = 4  # 瓶颈块末端通道扩张为 4 倍

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        out = torch.relu(self.bn3(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# ======================================================================
#  ResNet-18 主网络
# ======================================================================

# 论文 Table 1: ResNet-18 每个 stage 包含的 block 数量
RESNET18_LAYERS = [2, 2, 2, 2]


class ResNet18(nn.Module):
    """ResNet-18 (论文 Table 1, 共 18 层: 1 conv1 + 8 残差卷积层 + 1 fc)。

    结构:
        conv1 : 7×7, 64, stride 2
        maxpool: 3×3, stride 2
        conv2_x: 64  通道, 2 × BasicBlock, 无下采样 (stride=1)
        conv3_x: 128 通道, 2 × BasicBlock, stride=2
        conv4_x: 256 通道, 2 × BasicBlock, stride=2
        conv5_x: 512 通道, 2 × BasicBlock, stride=2
        avgpool: 全局平均池化
        fc     : num_classes 路全连接
    """

    def __init__(self, num_classes: int = 1000, in_channels: int = 3) -> None:
        super().__init__()
        self.in_planes = 64

        # ------------------- conv1 (论文 Table 1: 7×7, 64, stride 2) -------------------
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # ------------------- 最大池化 (论文 Fig.3: 3×3, stride 2) -------------------
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ------------------- 四个残差 stage (conv2_x ~ conv5_x) -------------------
        self.layer1 = self._make_stage(BasicBlock, 64, RESNET18_LAYERS[0], stride=1)
        self.layer2 = self._make_stage(BasicBlock, 128, RESNET18_LAYERS[1], stride=2)
        self.layer3 = self._make_stage(BasicBlock, 256, RESNET18_LAYERS[2], stride=2)
        self.layer4 = self._make_stage(BasicBlock, 512, RESNET18_LAYERS[3], stride=2)

        # ------------------- 全局平均池化 + 全连接 -------------------
        # 论文 3.3 / 4.1: global average pooling -> fc -> softmax
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        # 每个 stage 第一个 block 负责下采样 (stride) 与通道对齐, 其余 stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming/He 初始化 (论文 3.4 采用 rectifier 初始化, 等价于 He 初始化)。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ======================================================================
#  训练 / 测试脚本 (以 CIFAR-10 为例, 论文 4.2 数据增强)
# ======================================================================

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

DATA_DIR = r"e:/AI-Learning/CNN-Learning/data/cifar10"

NUM_CLASSES = 10
NUM_EPOCHS = 100
BATCH_SIZE = 128         # 论文 CIFAR-10 用 128
LEARNING_RATE = 0.01     # 注意: 本文件是 ImageNet 版 ResNet-18 (7x7 conv1 + maxpool),
                         # 直接用于 32x32 CIFAR-10 时特征图会被过度下采样, 激活方差易放大。
                         # 论文的 0.1 初始 lr 在此结构下会引发梯度/激活数值爆炸 -> loss=nan,
                         # 故将初始 lr 降为 0.01 以稳定训练 (仅调整训练超参, 不改网络结构)。
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4      # 论文 4.2: weight decay 0.0001
GRAD_CLIP = 5.0          # 梯度范数裁剪上限: 兜底防止数值爆炸导致 loss=nan

MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet18_cifar10.pth")


def build_transforms(is_train: bool):
    """论文 4.2 的 CIFAR-10 数据增强: 四周各 padding 4px, 随机 32×32 裁剪 + 水平翻转。"""
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders():
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


def train_one_epoch(model, loader, criterion, optimizer, device):
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
        # 梯度裁剪: 防止数值爆炸 (loss=nan 的根因之一)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=loss.item())
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    # 构建 ResNet-18 (ImageNet 预训练结构, 此处用于 CIFAR-10 则 num_classes=10)
    model = ResNet18(num_classes=NUM_CLASSES).to(device)
    print(
        f"ResNet-18 Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    # 论文: lr 在 32k / 48k 迭代处除以 10; 这里按 epoch 阶梯衰减。
    # 因初始 lr 已降为 0.01, 将衰减点相应延后以充分利用训练。
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1
    )

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
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
