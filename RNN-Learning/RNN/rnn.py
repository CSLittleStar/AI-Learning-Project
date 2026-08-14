"""
循环神经网络 (RNN) 数字序列预测 —— 代码复现
============================================
本文件包含三部分：
  1. VanillaRNN (from scratch)    —— 完全从零实现前向 + BPTT 反向传播，对应笔记里的公式
  2. TorchRNN                     —— 使用 PyTorch 内置 nn.RNN 的版本（对照）
  3. 数据 / 训练 / 可视化          —— 以"正弦波序列预测"作为数字序列预测任务

任务说明：
  给定一个长度为 T 的数字序列窗口 x = [x_1, x_2, ..., x_T]，
  目标是预测下一个时刻的值 y = x_{T+1}（多对一 / 滑动窗口预测）。

公式（与笔记 RNN-Note.md 一致，这里用 tanh 激活）：
  h(t) = tanh( W_hh * h(t-1) + W_xh * x(t) + b_h )
  y(t) = W_hy * h(t) + b_y
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能保存图片
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 工具：设备选择（优先 XPU / CUDA，否则 CPU）
# ----------------------------------------------------------------------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# 1. 从零实现 Vanilla RNN（含 BPTT 反向传播）
# ============================================================================
class VanillaRNN:
    """
    一个极简的"单隐藏层 + 输出层"的 RNN，完全手写前向与反向传播 (BPTT)。
    参数：
        input_size  输入维度（数字序列中每个时间步的特征数，这里为 1）
        hidden_size 隐藏状态维度
        output_size 输出维度（预测下一个值，为 1）
        learning_rate 学习率
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # 参数初始化（小随机值，避免对称 + 缓解梯度问题）
        # W_xh: 输入 -> 隐藏；W_hh: 隐藏 -> 隐藏；b_h: 隐藏偏置
        # W_hy: 隐藏 -> 输出；b_y: 输出偏置
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        BPTT 前向。
        inputs: list[np.ndarray]，每个元素形状 (input_size, 1)，长度 = 时间步 T
        返回：
            ys       预测序列（每个时间步的输出）
            hiddens  隐藏状态序列（包含初始 h0）
        """
        T = len(inputs)
        h = np.zeros((self.hidden_size, 1))   # h(0) = 0
        hiddens = [h]                          # 保存 h(0)..h(T)
        ys = []
        for x in inputs:
            h = np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)  # h(t)
            y = self.W_hy @ h + self.b_y                              # y(t)
            hiddens.append(h)
            ys.append(y)
        return ys, hiddens

    def backward(self, inputs, targets, ys, hiddens):
        """
        BPTT 反向传播，计算各参数梯度并按梯度下降更新。
        targets: list[np.ndarray]，每个 (output_size, 1)
        """
        T = len(inputs)
        # 梯度累加器（五个参数）
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        # 输出层梯度初始为 0 隐藏态（dh 沿时间回传）
        dh_next = np.zeros((self.hidden_size, 1))

        # 从最后一个时间步往回扫（沿时间反向传播）
        for t in reversed(range(T)):
            dy = ys[t] - targets[t]                       # MSE 对 y 的导数 (2*0.5*(y-y*))
            dW_hy += dy @ hiddens[t + 1].T                     # 细节：.T，因为是列向量，所以要转置
            db_y += dy                                         # 偏置梯度直接等于dy

            # 把输出层梯度传导回隐藏态
            dh = self.W_hy.T @ dy + dh_next
            # tanh 的导数：dh_raw = dh * (1 - h^2)
            h_t = hiddens[t + 1]
            dh_raw = dh * (1 - h_t * h_t)
            db_h += dh_raw
            dW_xh += dh_raw @ inputs[t].T
            dW_hh += dh_raw @ hiddens[t].T
            dh_next = self.W_hh.T @ dh_raw                 # 传给上一时间步
        
        """
        # 梯度裁剪（笔记提到的缓解梯度爆炸手段）
        for g in (dW_xh, dW_hh, db_h, dW_hy, db_y):
            np.clip(g, -5.0, 5.0, out=g)
        
        """
        # 参数更新
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.b_h -= self.lr * db_h
        self.W_hy -= self.lr * dW_hy
        self.b_y -= self.lr * db_y

    def train_step(self, inputs, target):
        """单步训练：输入序列 -> 预测其最后一个时间步对应的下一值。"""
        # 这里让 RNN 在每个时间步都输出，用最后一个输出作为预测
        ys, hiddens = self.forward(inputs)
        # 目标：预测序列整体往后一位。为简单起见，用"最后一个时间步的输出"对 target 回归。
        # 构造每个时间步的目标（输入整体左移一位）
        n = len(inputs)
        targets = inputs[1:] + [target]   # 退化为 teacher forcing 的多对一
        self.backward(inputs, targets, ys, hiddens)
        # 以最后一个预测对最终 target 的损失作为监控
        loss = 0.5 * np.sum((ys[-1] - target) ** 2)
        return float(loss)


# ============================================================================
# 2. PyTorch 内置 RNN 版本（对照 / 生产可用）
# ============================================================================
class TorchRNN(nn.Module):
    """
    使用 nn.RNN 构建的多对一序列预测网络。
    由于内置 RNN 自动处理 BPTT，无需手写反向传播。
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)        # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                      # 取最后一个时间步（多对一）
        pred = self.fc(last)                      # (batch, 1)
        return pred


# ============================================================================
# 3. 数据：正弦波数字序列（滑动窗口）
# ============================================================================
class SineDataset(Dataset):
    """
    生成正弦波连续序列，用滑动窗口切分成 (输入窗口, 下一时刻值)。
    例如窗口长度 20，则样本为 [s_i .. s_{i+19}] -> s_{i+20}
    """

    def __init__(self, total=2000, window=20, step=1):
        t = np.linspace(0, total * 0.05, total)
        self.series = np.sin(t).astype(np.float32)
        self.window = window
        self.step = step
        self.X, self.Y = self._build()

    def _build(self):
        X, Y = [], []
        for i in range(0, len(self.series) - self.window - 1, self.step):
            x = self.series[i:i + self.window]
            y = self.series[i + self.window]
            X.append(x)
            Y.append(y)
        X = np.array(X, dtype=np.float32).reshape(-1, self.window, 1)
        Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
        return X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================================
# 4. 训练 + 可视化
# ============================================================================
def train_torch():
    device = get_device()
    print(f"[TorchRNN] device = {device}")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TorchRNN(input_size=1, hidden_size=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss = {avg:.6f}")

    # 评估可视化
    model.eval()
    xs = dataset.X[:200]
    ys = dataset.Y[:200]
    with torch.no_grad():
        preds = model(torch.from_numpy(xs).to(device)).cpu().numpy()
    _plot(ys, preds, "TorchRNN", window)
    torch.save(model.state_dict(), __file__.replace(".py", "_torch_model.pt"))
    print("[TorchRNN] 已保存模型权重与预测对比图")
    return model


def train_scratch():
    """训练手写 VanillaRNN 并可视化（作为教学复现）。"""
    print("[VanillaRNN] 从零训练（教学用，迭代较慢）")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    n_samples = len(dataset.X)

    model = VanillaRNN(input_size=1, hidden_size=24, output_size=1, learning_rate=0.01)
    epochs = 40
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        for idx in perm:
            x_seq = dataset.X[idx].reshape(window, 1, 1)  # (T,1,1)
            inputs = [x_seq[t] for t in range(window)]     # list of (1,1)
            target = dataset.Y[idx].reshape(1, 1)
            total_loss += model.train_step(inputs, target)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss = {total_loss / n_samples:.6f}")

    # 在部分样本上做预测
    ys, preds = [], []
    for i in range(200):
        x_seq = dataset.X[i].reshape(window, 1, 1)
        inputs = [x_seq[t] for t in range(window)]
        ys.append(dataset.Y[i].item())
        preds.append(model.forward(inputs)[0][-1].item())
    _plot(np.array(ys), np.array(preds), "VanillaRNN", window)
    print("[VanillaRNN] 已保存预测对比图")
    return model


def _plot(truth, pred, name, window):
    # 设置 matplotlib 支持中文显示（屏蔽乱码警告）
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    plt.figure(figsize=(10, 4))
    plt.plot(truth, label="Ground Truth", color="black", linewidth=1.2)
    plt.plot(pred, label="Prediction", color="red", linestyle="--", linewidth=1.2)
    plt.title(f"{name} Number-Sequence Prediction (window={window})")
    plt.xlabel("sample index")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = __file__.replace(".py", f"_{name}_predict.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  已保存图片: {out}")


def main():
    print("=" * 60)
    print("RNN 数字序列预测 —— 代码复现")
    print("=" * 60)
    # 1) PyTorch 内置 RNN（推荐，训练快效果好）
    train_torch()
    # 2) 从零手写 RNN（教学复现，验证前向/反向传播正确性）
    print()
    train_scratch()


if __name__ == "__main__":
    main()
