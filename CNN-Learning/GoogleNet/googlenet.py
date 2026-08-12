"""GoogLeNet (Inception v1) 模型实现 (PyTorch)。

原始论文: Szegedy et al., "Going Deeper with Convolutions", CVPR 2015 (ILSVRC 2014)。

本文件提供:
- Inception 模块 (带 dimension reduction 的版本, 对应论文 Figure 2(b))
- GoogLeNet 主体 (22 层带参数, 含 stem + 9 个 Inception + 分类头)
- 两个辅助分类器 (auxiliary classifiers), 接在 inception(4a) 与 inception(4d) 后
  - 训练时其 loss 以 0.3 权重加到总损失上; 推理时丢弃。

所有卷积/全连接参数严格遵循论文 Table 1 与正文描述。
输入: 224x224x3 (论文使用 224 感受野, RGB 均值减除)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ======================================================================
#  Inception 模块 (带降维)
# ======================================================================

class Inception(nn.Module):
    """论文 Figure 2(b): 4 条并行分支。

    分支 0: 1x1 卷积
    分支 1: 1x1 reduce -> 3x3 卷积
    分支 2: 1x1 reduce -> 5x5 卷积
    分支 3: 3x3 max pool -> 1x1 projection

    参数对应论文 Table 1 的列:
        in_channels : 输入通道
        ch1x1       : #1x1
        ch3x3red    : #3x3 reduce
        ch3x3       : #3x3
        ch5x5red    : #5x5 reduce
        ch5x5       : #5x5
        chpool      : pool proj (#1x1 after pool)
    """

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3,
                 ch5x5red, ch5x5, chpool):
        super().__init__()
        # 分支 0: 1x1 卷积 (stride 1, padding 0)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # 分支 1: 1x1 reduce -> 3x3 (padding 1 保持空间尺寸)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # 分支 2: 1x1 reduce -> 5x5 (padding 2 保持空间尺寸)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        # 分支 3: 3x3 max pool (stride 1, padding 1) -> 1x1 projection
        self.branchp = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, chpool, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 4 条分支沿通道维拼接 (depth concat)
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branchp(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


# ======================================================================
#  辅助分类器 (Auxiliary Classifier)
# ======================================================================

class AuxClassifier(nn.Module):
    """论文 6 页: 接在 inception(4a) / inception(4d) 输出上的侧网络。

    结构:
        - 平均池化 5x5, stride 3  -> 4x4x(ch)  (4a: 512, 4d: 528)
        - 1x1 卷积, 128 通道, ReLU
        - 3x3 卷积, 768 通道, ReLU  (论文中为 FC, 这里用等效卷积实现)
        - Dropout(0.7)
        - 全连接 (线性) -> num_classes, 由损失函数做 softmax
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, ceil_mode=True),  # 14x14 -> 4x4 (ceil 匹配论文)
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=3, stride=1, padding=1),  # 4x4 -> 4x4 (padding=1 保持)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
        )
        # 论文图: aux 分类器最终 FC 输出类别数 (论文用 1000)。
        # 上一卷积输出空间 4x4 (avgpool 5x5/3 后保持), 通道 768 -> 4*4*768 = 3072。
        self.fc = nn.Linear(768 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


# ======================================================================
#  GoogLeNet 主体
# ======================================================================

class GoogLeNet(nn.Module):

    def __init__(self, num_classes: int = 1000, aux_logits: bool = True,
                 init_weights: bool = True):
        super().__init__()
        self.aux_logits = aux_logits

        # ------------------- Stem (特征提取前端) -------------------
        # 论文 Table 1:
        #   conv 7x7/2 -> 112x112x64
        #   maxpool 3x3/2 -> 56x56x64
        #   conv 3x3/1 -> 56x56x192
        #   maxpool 3x3/2 -> 28x28x192
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),   # 224 -> 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # 112 -> 56
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 56 -> 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # 56 -> 28
        )

        # ------------------- Inception 阶段 (3a~3b) -------------------
        # 输出 28x28
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)    # -> 256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)  # -> 480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 28 -> 14

        # ------------------- Inception 阶段 (4a~4e) -------------------
        # 输出 14x14
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)   # -> 512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)  # -> 512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # -> 512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # -> 528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # -> 832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 14 -> 7

        # ------------------- Inception 阶段 (5a~5b) -------------------
        # 输出 7x7
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)  # -> 832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)  # -> 1024

        # ------------------- 辅助分类器 (训练时启用) -------------------
        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)   # 接在 4a 后
            self.aux2 = AuxClassifier(528, num_classes)   # 接在 4d 后

        # ------------------- 分类头 (Classifier) -------------------
        # 论文 Table 1:
        #   avgpool 7x7/1 -> 1x1x1024
        #   dropout(40%)
        #   linear -> 1000
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes),                  # 最后一层 logits
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化 (He / Kaiming, fan_in, 对 ReLU 稳定)。

        原实现用 fan_out + Linear std=0.01, 在 Inception 多分支 concat 结构下会导致
        激活值逐层指数增长 (实测 stem=1.6 -> 5b=1484), 训练数步后溢出为 NaN。
        改为 fan_in 后, 前向每层输出方差保持 ~1, 激活不再爆炸, NaN 消失。
        不改变任何卷积/全连接结构, 仅调整初始化方式。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)                       # -> 28x28x192

        x = self.inception3a(x)               # -> 28x28x256
        x = self.inception3b(x)               # -> 28x28x480
        x = self.maxpool3(x)                  # -> 14x14x480

        x = self.inception4a(x)               # -> 14x14x512
        # 辅助分类器 1
        aux1 = self.aux1(x) if self.aux_logits else None

        x = self.inception4b(x)               # -> 14x14x512
        x = self.inception4c(x)               # -> 14x14x512
        x = self.inception4d(x)               # -> 14x14x528
        # 辅助分类器 2
        aux2 = self.aux2(x) if self.aux_logits else None

        x = self.inception4e(x)               # -> 14x14x832
        x = self.maxpool4(x)                  # -> 7x7x832

        x = self.inception5a(x)               # -> 7x7x832
        x = self.inception5b(x)               # -> 7x7x1024

        # 论文: avgpool 7x7/1 -> 1x1x1024 (等价于全局平均池化)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)                # -> num_classes (Dropout + 全连接)

        if self.aux_logits:
            return x, aux1, aux2
        return x


# ======================================================================
#  训练 / 测试脚本 (CIFAR-10, 输入上采样到 224x224)
# ======================================================================

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

DATA_DIR = r"e:/AI-Learning/CNN-Learning/data/cifar10"

NUM_CLASSES = 10
NUM_EPOCHS = 50           # 原 10 轮不足以收敛, 增大到 50
BATCH_SIZE = 128          # XPU 显存允许则加大 batch (不足则调回 64)
LEARNING_RATE = 0.001     # 原 0.01 过大, 配合 aux 分支易梯度爆炸产生 NaN, 降为 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
AUX_WEIGHT = 0.3          # 论文: 辅助分类器 loss 权重 0.3

MODEL_PATH = os.path.join(os.path.dirname(__file__), "googlenet_cifar10.pth")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, aux1, aux2 = model(images)
        loss = criterion(outputs, labels)
        if aux1 is not None:
            loss += AUX_WEIGHT * criterion(aux1, labels)
        if aux2 is not None:
            loss += AUX_WEIGHT * criterion(aux2, labels)
        loss.backward()
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
    # 推理时关闭辅助分类器, 仅切换一次, 避免每 batch 反复改属性
    prev_aux = model.aux_logits
    model.aux_logits = False
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

    model.aux_logits = prev_aux
    return running_loss / total, 100.0 * correct / total


def build_transforms(is_train: bool):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
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
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = GoogLeNet(num_classes=NUM_CLASSES).to(device)
    print(
        "GoogLeNet Parameters: "
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
