"""长短期记忆网络 (LSTM) 数字序列预测 —— 代码复现"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ============================================================================
# 1. 从零实现 VanillaLSTM（含 BPTT 反向传播）
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(out):
    # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    return out * (1.0 - out)


def tanh_grad(out):
    # d(tanh)/dx = 1 - tanh^2
    return 1.0 - out * out


class VanillaLSTM:
    """
    手写单隐层 LSTM + 输出层。
    input_size  输入维度（这里为 1）
    hidden_size 隐藏 / 细胞状态维度
    output_size 输出维度（预测下一个值，为 1）
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # LSTM 4 个门控共用一份 [h; x] 拼接输入，维度 = hidden + input
        D = hidden_size + input_size
        scale = 0.01
        # 每个门: W (hidden_size, D), b (hidden_size, 1)
        self.W_f = np.random.randn(hidden_size, D) * scale
        self.W_i = np.random.randn(hidden_size, D) * scale
        self.W_g = np.random.randn(hidden_size, D) * scale
        self.W_o = np.random.randn(hidden_size, D) * scale
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_g = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        # 输出层
        self.W_hy = np.random.randn(output_size, hidden_size) * scale
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        inputs: list[np.ndarray]，每个 (input_size, 1)，长度 = T
        返回各时间步缓存，供反向传播使用。
        """
        T = len(inputs)
        H = self.hidden_size
        h = np.zeros((H, 1))
        c = np.zeros((H, 1))

        # 缓存
        h_list, c_list = [h], [c]          # h(0), c(0)
        f_list, i_list, g_list, o_list = [], [], [], []
        x_list = []                         # 每个时间步的拼接 [h_prev; x]

        for x in inputs:
            concat = np.concatenate([h, x], axis=0)   # (D, 1)
            x_list.append(concat)
            f = sigmoid(self.W_f @ concat + self.b_f)
            i = sigmoid(self.W_i @ concat + self.b_i)
            g = np.tanh(self.W_g @ concat + self.b_g)
            o = sigmoid(self.W_o @ concat + self.b_o)
            c = f * c + i * g
            h = o * np.tanh(c)
            f_list.append(f); i_list.append(i)
            g_list.append(g); o_list.append(o)
            h_list.append(h); c_list.append(c)

        ys = [self.W_hy @ h_list[t + 1] + self.b_y for t in range(T)]
        cache = dict(h=h_list, c=c_list, f=f_list, i=i_list,
                     g=g_list, o=o_list, x=x_list, y=ys)
        return ys, cache

    def backward(self, inputs, targets, cache):
        """BPTT 反向传播，沿时间回传 c、h 的梯度。"""
        T = len(inputs)
        H = self.hidden_size
        # 梯度累加器
        dW_f = np.zeros_like(self.W_f); db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i); db_i = np.zeros_like(self.b_i)
        dW_g = np.zeros_like(self.W_g); db_g = np.zeros_like(self.b_g)
        dW_o = np.zeros_like(self.W_o); db_o = np.zeros_like(self.b_o)
        dW_hy = np.zeros_like(self.W_hy); db_y = np.zeros_like(self.b_y)

        h_list = cache["h"]; c_list = cache["c"]
        f_list = cache["f"]; i_list = cache["i"]
        g_list = cache["g"]; o_list = cache["o"]
        x_list = cache["x"]; ys = cache["y"]

        # 时间维度上初始梯度为 0
        dh_next = np.zeros((H, 1))
        dc_next = np.zeros((H, 1))

        for t in reversed(range(T)):
            # ---- 输出层梯度 ----
            dy = ys[t] - targets[t]
            dW_hy += dy @ h_list[t + 1].T
            db_y += dy
            dh = self.W_hy.T @ dy + dh_next

            # ---- 由 h(t) 反推到 c(t) 与 o(t) ----
            # h = o * tanh(c)
            c_t = c_list[t + 1]
            tanh_c = np.tanh(c_t)
            do = dh * tanh_c
            do_raw = do * sigmoid_grad(o_list[t])      # 对线性组合的导数
            dc = dh * o_list[t] * tanh_grad(tanh_c) + dc_next   # 来自 h 与下一时刻 c

            # ---- 由 c(t) 反推三个门 ----
            # c_t = f*c_{t-1} + i*g
            df_raw = (dc * c_list[t]) * sigmoid_grad(f_list[t])
            di_raw = (dc * g_list[t]) * sigmoid_grad(i_list[t])
            dg_raw = (dc * i_list[t]) * tanh_grad(g_list[t])

            # ---- 线性组合对 W、b、以及 [h_prev; x] 的梯度 ----
            for dW, db, raw in ((dW_f, db_f, df_raw), (dW_i, db_i, di_raw),
                                (dW_g, db_g, dg_raw), (dW_o, db_o, do_raw)):
                dW += raw @ x_list[t].T
                db += raw

            # [h_prev; x] 的梯度，用于回传 dh_next（只取 h 部分）
            dz = (self.W_f.T @ df_raw + self.W_i.T @ di_raw +
                  self.W_g.T @ dg_raw + self.W_o.T @ do_raw)
            dh_next = dz[:H, :]               # 前 H 行对应 h(t-1)
            dc_next = f_list[t] * dc          # c 的直连通道：c(t-1) 项
        
        """
        # 梯度裁剪
        for g in (dW_f, dW_i, dW_g, dW_o, db_f, db_i, db_g, db_o, dW_hy, db_y):
            np.clip(g, -5.0, 5.0, out=g)
        """
        
        # 参数更新
        self.W_f -= self.lr * dW_f; self.b_f -= self.lr * db_f
        self.W_i -= self.lr * dW_i; self.b_i -= self.lr * db_i
        self.W_g -= self.lr * dW_g; self.b_g -= self.lr * db_g
        self.W_o -= self.lr * dW_o; self.b_o -= self.lr * db_o
        self.W_hy -= self.lr * dW_hy; self.b_y -= self.lr * db_y

    def train_step(self, inputs, target):
        ys, cache = self.forward(inputs)
        targets = inputs[1:] + [target]
        self.backward(inputs, targets, cache)
        return float(0.5 * np.sum((ys[-1] - target) ** 2))


# ============================================================================
# 2. PyTorch 内置 LSTM 版本（对照）
# ============================================================================
class TorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)   # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                  # 多对一：取最后时间步
        return self.fc(last)


# ============================================================================
# 3. 数据：正弦波数字序列（滑动窗口，与 RNN 版本一致）
# ============================================================================
class SineDataset(Dataset):
    def __init__(self, total=2000, window=20, step=1):
        t = np.linspace(0, total * 0.05, total)
        self.series = np.sin(t).astype(np.float32)
        self.window = window
        self.step = step
        self.X, self.Y = self._build()

    def _build(self):
        X, Y = [], []
        for i in range(0, len(self.series) - self.window - 1, self.step):
            X.append(self.series[i:i + self.window])
            Y.append(self.series[i + self.window])
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
    print(f"[TorchLSTM] device = {device}")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TorchLSTM(input_size=1, hidden_size=32).to(device)
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

    model.eval()
    xs = dataset.X[:200]
    ys = dataset.Y[:200]
    with torch.no_grad():
        preds = model(torch.from_numpy(xs).to(device)).cpu().numpy()
    _plot(ys, preds, "TorchLSTM", window)
    torch.save(model.state_dict(), __file__.replace(".py", "_torch_model.pt"))
    print("[TorchLSTM] 已保存模型权重与预测对比图")


def train_scratch():
    print("[VanillaLSTM] 从零训练（教学用，验证门控前向/反向传播正确性）")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    n_samples = len(dataset.X)

    model = VanillaLSTM(input_size=1, hidden_size=24, output_size=1, learning_rate=0.01)
    epochs = 40
    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        for idx in perm:
            x_seq = dataset.X[idx].reshape(window, 1, 1)
            inputs = [x_seq[t] for t in range(window)]
            target = dataset.Y[idx].reshape(1, 1)
            total_loss += model.train_step(inputs, target)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss = {total_loss / n_samples:.6f}")

    ys, preds = [], []
    for i in range(200):
        x_seq = dataset.X[i].reshape(window, 1, 1)
        inputs = [x_seq[t] for t in range(window)]
        ys.append(dataset.Y[i].item())
        preds.append(model.forward(inputs)[0][-1].item())
    _plot(np.array(ys), np.array(preds), "VanillaLSTM", window)
    print("[VanillaLSTM] 已保存预测对比图")


def _plot(truth, pred, name, window):
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
    print("LSTM 数字序列预测 —— 代码复现")
    print("=" * 60)
    train_torch()
    print()
    train_scratch()


if __name__ == "__main__":
    main()
