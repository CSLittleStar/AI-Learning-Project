import torch
import torch.nn as nn            # torch的神经网络模块
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
在test2.py的MLP基础上，做成MNIST手写数据分类
"""

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()             # 调用父类初始化
        self.flatten = nn.Flatten()  # 28x28 -> 784 flatten用于铺平图片数据
        self.layers = nn.Sequential(    # Sequential 用于组合多个层
            nn.Linear(784, 256),    # 784 -> 256
            nn.ReLU(),                                       # ReLU激活函数转非线性
            nn.Linear(256, 128),    # 256 -> 128
            nn.ReLU(),
            nn.Linear(128, 10),     # 128 -> 10
        )

    def forward(self, x):       # 前向传播方法
        x = self.flatten(x)             # 铺平图片数据
        logits = self.layers(x)         # 展平后送入layers，得到输出logits=（batch, 10）
        return logits


def get_data_loaders(batch_size=64):
    """加载 MNIST 数据集并返回训练/测试 DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # 转为张量并归一化到 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 标准均值/标准差
    ])

    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # shuffle是否打乱数据，训练一般为true
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 切换到训练模式
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   # data是需要预测的数据图，target是真实数据的标签

        optimizer.zero_grad()          # 清空历史梯度
        output = model(data)           # 前向传播，output 形状 (batch, 10)
        loss = criterion(output, target)  # 交叉熵损失（内部含 softmax）
        loss.backward()                # 反向传播计算梯度
        optimizer.step()               # Adam 更新参数

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")

    print(f"Epoch {epoch} 平均损失: {total_loss / len(train_loader):.6f}")


def test(model, device, test_loader, criterion):
    model.eval()   # 切换到评估模式（关闭 dropout/batchnorm 等）
    test_loss = 0.0
    correct = 0
    with torch.no_grad():   # 推理阶段不计算梯度，节省显存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 取概率最大的类别
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"测试集: 平均损失 {test_loss:.4f}, "
          f"准确率 {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")


if __name__ == "__main__":
    # 自动选择 GPU / CPU
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print("使用设备:", device)

    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    model = MLP().to(device)
    print(model)

    # 使用交叉熵损失函数（适用于多分类，已内置 softmax）

    criterion = nn.CrossEntropyLoss()

    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_data_loaders(batch_size)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    # 保存训练好的模型
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("模型已保存到 mnist_mlp.pth")
