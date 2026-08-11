import torch
import torch.nn as nn       # torch的神经网络模块


class MLP(nn.Module):
    """多层感知机 (MLP)

    结构:
        输入 28x28 图像
         -> Flatten (784)
         -> 全连接 784 -> 256 (ReLU)
         -> 全连接 256 -> 128 (ReLU)
         -> 全连接 128 -> 10 (输出 10 类)
    """

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


if __name__ == "__main__":
    model = MLP()
    print(model)

    # 用一个随机批次验证前向传播
    x = torch.randn(64, 1, 28, 28)  # (batch, channel, height, width)
    out = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)  # 期望: torch.Size([64, 10])

