"""
GRU (Gated Recurrent Unit) 数字序列预测 —— 代码复现
====================================================
复现论文《Learning Phrase Representations using RNN
Encoder–Decoder for Statistical Machine Translation》
(Kyunghyun Cho et al., 2014) 提出的 GRU 隐藏单元
与其 Encoder–Decoder 架构思想。

文件名统一为 gru.py，放置在 RNN-Learning/GRU 下。
与 ../RNN/rnn.py、../LSTM/lstm.py 保持一致的风格。

本文件包含四部分：
  1. VanillaGRU (from scratch)   —— 完全从零实现 GRU 前向 + BPTT 反向传播
  2. TorchGRU                   —— 使用 PyTorch 内置 nn.GRU 的版本（对照）
  3. EncoderDecoderGRU          —— 复现论文 Encoder–Decoder 架构（自回归多步预测）
  4. 数据 / 训练 / 可视化         —— 同样以"正弦波序列预测"作为数字序列预测任务

GRU 单元公式（论文 Eq. 5–8，本复现与原论文一致）：
  重置门 r(t) = sigmoid(W_r · x(t) + U_r · h(t-1) + b_r)
  更新门 z(t) = sigmoid(W_z · x(t) + U_z · h(t-1) + b_z)
  候选状态 h̃(t) = tanh(W · x(t) + U · (r(t) ⊙ h(t-1)) + b_h)   # 关键：r⊙h，而非完整 h
  隐藏状态 h(t) = z(t) ⊙ h(t-1) + (1 - z(t)) ⊙ h̃(t)           # 门控线性插值
  输出：   y(t) = W_hy · h(t) + b_y

与 LSTM 相比，GRU 只有两个门（重置门、更新门），没有独立的细胞状态 c(t)，
隐藏状态即最终输出。候选状态用"重置门对上一时刻隐藏态做逐元素屏蔽 (r⊙h)"，
使得单元能丢弃与未来无关的信息，从而学到不同时间尺度的依赖——这正是论文
2.3 节强调的"adaptively remembers and forgets"。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能保存图片
import matplotlib.pyplot as plt


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# 1. 从零实现 VanillaGRU（含 BPTT 反向传播）
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(out):
    # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    return out * (1.0 - out)


def tanh_grad(out):
    # d(tanh)/dx = 1 - tanh^2
    return 1.0 - out * out


class VanillaGRU:
    """
    手写单隐层 GRU + 输出层。
    input_size  输入维度（这里为 1）
    hidden_size 隐藏状态维度
    output_size 输出维度（预测下一个值，为 1）
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        H, D = hidden_size, input_size
        scale = 0.01
        # 三个线性变换各用一份 (W 输入相关, U 隐藏相关, b 偏置)
        # r: 重置门, z: 更新门, h: 候选状态（tanh）
        self.W_r = np.random.randn(H, D) * scale
        self.U_r = np.random.randn(H, H) * scale
        self.b_r = np.zeros((H, 1))

        self.W_z = np.random.randn(H, D) * scale
        self.U_z = np.random.randn(H, H) * scale
        self.b_z = np.zeros((H, 1))

        self.W_h = np.random.randn(H, D) * scale
        self.U_h = np.random.randn(H, H) * scale
        self.b_h = np.zeros((H, 1))

        # 输出层
        self.W_hy = np.random.randn(output_size, H) * scale
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        inputs: list[np.ndarray]，每个 (input_size, 1)，长度 = T
        返回各时间步缓存，供反向传播使用。
        """
        T = len(inputs)
        H = self.hidden_size
        h = np.zeros((H, 1))   # h(0) = 0

        h_list = [h]                 # 保存 h(0)..h(T)
        r_list, z_list, tilde_list = [], [], []
        x_list, hprev_list = [], []  # 候选状态拆分出的输入项，便于反向

        for x in inputs:
            r = sigmoid(self.W_r @ x + self.U_r @ h + self.b_r)        # 重置门
            z = sigmoid(self.W_z @ x + self.U_z @ h + self.b_z)        # 更新门
            h_tilde = np.tanh(self.W_h @ x + self.U_h @ (r * h) + self.b_h)  # 候选状态
            h = z * h + (1 - z) * h_tilde                            # 门控线性插值

            x_list.append(x); hprev_list.append(h)
            r_list.append(r); z_list.append(z); tilde_list.append(h_tilde)
            h_list.append(h)

        ys = [self.W_hy @ h_list[t + 1] + self.b_y for t in range(T)]
        cache = dict(h=h_list, r=r_list, z=z_list, tilde=tilde_list,
                     x=x_list, hprev=hprev_list, y=ys)
        return ys, cache

    def backward(self, inputs, targets, cache):
        """BPTT 反向传播，沿时间回传 h 的梯度。"""
        T = len(inputs)
        H = self.hidden_size
        # 梯度累加器
        dW_r = np.zeros_like(self.W_r); dU_r = np.zeros_like(self.U_r); db_r = np.zeros_like(self.b_r)
        dW_z = np.zeros_like(self.W_z); dU_z = np.zeros_like(self.U_z); db_z = np.zeros_like(self.b_z)
        dW_h = np.zeros_like(self.W_h); dU_h = np.zeros_like(self.U_h); db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy); db_y = np.zeros_like(self.b_y)

        h_list = cache["h"]; r_list = cache["r"]; z_list = cache["z"]
        tilde_list = cache["tilde"]; x_list = cache["x"]; hprev = cache["hprev"]
        ys = cache["y"]

        dh_next = np.zeros((H, 1))   # 时间维度上初始梯度为 0

        for t in reversed(range(T)):
            # ---- 输出层梯度 ----
            dy = ys[t] - targets[t]
            dW_hy += dy @ h_list[t + 1].T
            db_y += dy
            dh = self.W_hy.T @ dy + dh_next   # 来自输出层 + 下一时刻

            # ---- 由 h(t) 反推 z、h̃、r ----
            # h = z*h_prev + (1-z)*h̃
            h_prev = h_list[t]
            z = z_list[t]; h_tilde = tilde_list[t]
            dz = dh * (h_prev - h_tilde) * sigmoid_grad(z)              # 对 z 线性项
            dh_tilde = dh * (1 - z) * tanh_grad(h_tilde)                # 对 h̃ 线性项

            # ---- h̃ 线性项拆回到 W_h、U_h(r⊙h)、b_h ----
            # h̃ = tanh(W_h x + U_h (r⊙h_prev) + b_h)
            dW_h += dh_tilde @ x_list[t].T
            db_h += dh_tilde
            # U_h 作用在 (r⊙h_prev) 上
            dU_h_contrib = dh_tilde @ (r_list[t] * h_prev).T
            dU_h += dU_h_contrib
            # (r⊙h_prev) 对 r 与 h_prev 的梯度
            d_rh = self.U_h.T @ dh_tilde                  # 形状 (H,1)，对应 (r⊙h_prev)
            dr = d_rh * h_prev * sigmoid_grad(r_list[t])  # 对 r 的线性项
            dW_r += dr @ x_list[t].T
            dU_r += dr @ h_prev.T
            db_r += dr
            # h_prev 经过两个路径回传：z*h_prev 与 r*h_prev
            dh_next = dh * z + d_rh * r_list[t]

        # 梯度裁剪（与 RNN/LSTM 复现一致，缓解梯度爆炸）
        for g in (dW_r, dU_r, db_r, dW_z, dU_z, db_z,
                  dW_h, dU_h, db_h, dW_hy, db_y):
            np.clip(g, -5.0, 5.0, out=g)

        # 参数更新
        self.W_r -= self.lr * dW_r; self.U_r -= self.lr * dU_r; self.b_r -= self.lr * db_r
        self.W_z -= self.lr * dW_z; self.U_z -= self.lr * dU_z; self.b_z -= self.lr * db_z
        self.W_h -= self.lr * dW_h; self.U_h -= self.lr * dU_h; self.b_h -= self.lr * db_h
        self.W_hy -= self.lr * dW_hy; self.b_y -= self.lr * db_y

    def train_step(self, inputs, target):
        ys, cache = self.forward(inputs)
        targets = inputs[1:] + [target]
        self.backward(inputs, targets, cache)
        return float(0.5 * np.sum((ys[-1] - target) ** 2))


# ============================================================================
# 2. PyTorch 内置 GRU 版本（对照）
# ============================================================================
class TorchGRU(nn.Module):
    """
    使用 nn.GRU 构建的多对一序列预测网络。
    PyTorch 的 nn.GRU 默认按论文公式实现（候选状态用 r⊙h_{t-1}）。
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.gru(x, hidden)   # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                  # 多对一：取最后时间步
        return self.fc(last)


# ============================================================================
# 3. Encoder–Decoder 架构（复现论文 Sec. 2.2 思想）
# ============================================================================
class EncoderDecoderGRU(nn.Module):
    """
    复现论文的 Encoder–Decoder：
      Encoder 把变长输入序列编码为固定向量 c（取最后隐藏态）；
      Decoder 以 c 为初始条件，自回归地生成目标序列。

    这里用"正弦波"做演示：给定前 window 个值，让 Decoder 自回归
    预测后续若干步（多步滚动预测），体现论文中"序列 -> 向量 -> 序列"
    的核心思想，而不依赖机器翻译语料。
    """

    def __init__(self, input_size=1, hidden_size=32, decoder_steps=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_steps = decoder_steps
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x_src, y_tgt=None, teacher_forcing=True):
        """
        x_src: (batch, src_len, input_size)  输入序列
        y_tgt: (batch, tgt_len, input_size)  训练时的目标序列（可选，用于 teacher forcing）
        返回预测序列 (batch, out_steps, input_size)
        """
        batch = x_src.size(0)
        # ---- Encoder ----
        _, h_n = self.encoder(x_src)            # h_n: (1, batch, hidden)
        c = h_n[-1]                              # 取最后一层隐藏态作为上下文向量 c

        # ---- Decoder 自回归生成 ----
        # 初始输入为全 0（等价于论文 e(y_0) 的全零嵌入），隐藏态初始化为 c
        dec_input = torch.zeros(batch, 1, self.hidden_size if False else 1, device=x_src.device)
        hidden = c.unsqueeze(0)                 # (1, batch, hidden)
        outputs = []
        out_steps = self.decoder_steps if y_tgt is None else y_tgt.size(1)
        for t in range(out_steps):
            out, hidden = self.decoder(dec_input, hidden)   # out: (batch,1,hidden)
            pred = self.fc(out[:, 0, :])                     # (batch, input_size)
            outputs.append(pred.unsqueeze(1))
            if y_tgt is not None and teacher_forcing:
                dec_input = y_tgt[:, t:t + 1, :]             # teacher forcing
            else:
                dec_input = pred.unsqueeze(1)                # 自回归（用自身预测）
        return torch.cat(outputs, dim=1)                     # (batch, out_steps, input_size)


# ============================================================================
# 4. 数据：正弦波数字序列（滑动窗口，与 RNN/LSTM 版本一致）
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
# 5. 训练 + 可视化
# ============================================================================
def train_torch():
    device = get_device()
    print(f"[TorchGRU] device = {device}")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TorchGRU(input_size=1, hidden_size=32).to(device)
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
    _plot(ys, preds, "TorchGRU", window)
    torch.save(model.state_dict(), __file__.replace(".py", "_torch_model.pt"))
    print("[TorchGRU] 已保存模型权重与预测对比图")


def train_scratch():
    print("[VanillaGRU] 从零训练（教学用，验证门控前向/反向传播正确性）")
    window = 20
    dataset = SineDataset(total=2000, window=window, step=1)
    n_samples = len(dataset.X)

    model = VanillaGRU(input_size=1, hidden_size=24, output_size=1, learning_rate=0.01)
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
    _plot(np.array(ys), np.array(preds), "VanillaGRU", window)
    print("[VanillaGRU] 已保存预测对比图")


def train_encdec():
    """演示论文 Encoder–Decoder 架构：自回归多步滚动预测正弦波。"""
    print("[EncoderDecoderGRU] 训练 Encoder–Decoder（自回归多步预测）")
    device = get_device()
    # 构造 (输入序列 -> 后续若干步) 的样本
    series = np.sin(np.linspace(0, 2000 * 0.05, 2000)).astype(np.float32)
    window, pred_steps = 20, 10
    X, Y = [], []
    for i in range(0, len(series) - window - pred_steps, 1):
        X.append(series[i:i + window])
        Y.append(series[i + window:i + window + pred_steps])
    X = np.array(X, dtype=np.float32).reshape(-1, window, 1)
    Y = np.array(Y, dtype=np.float32).reshape(-1, pred_steps, 1)

    model = EncoderDecoderGRU(input_size=1, hidden_size=32, decoder_steps=pred_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(list(zip(X, Y)), batch_size=64, shuffle=True)
    epochs = 40
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, yb, teacher_forcing=True)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(X)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss = {avg:.6f}")

    # 自回归（无 teacher forcing）生成，验证 Encoder–Decoder 的多步预测能力
    model.eval()
    with torch.no_grad():
        gen = model(torch.from_numpy(X[:1]).to(device), y_tgt=None, teacher_forcing=False)
    truth = Y[:1].reshape(-1)
    pred = gen.cpu().numpy().reshape(-1)
    _plot_encdec(truth, pred, window, pred_steps)
    torch.save(model.state_dict(), __file__.replace(".py", "_encdec_model.pt"))
    print("[EncoderDecoderGRU] 已保存模型权重与自回归预测对比图")


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


def _plot_encdec(truth, pred, window, pred_steps):
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    plt.figure(figsize=(10, 4))
    plt.plot(range(pred_steps), truth, label="Ground Truth", color="black",
             linewidth=1.5, marker="o")
    plt.plot(range(pred_steps), pred, label="GRU Encoder–Decoder", color="red",
             linestyle="--", linewidth=1.5, marker="x")
    plt.title(f"Encoder–Decoder Autoregressive Prediction (target steps={pred_steps})")
    plt.xlabel("target time step")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = __file__.replace(".py", "_EncoderDecoder_predict.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  已保存图片: {out}")


def main():
    print("=" * 60)
    print("GRU 数字序列预测 —— 代码复现 (Cho et al., 2014)")
    print("=" * 60)
    train_torch()
    print()
    train_scratch()
    print()
    train_encdec()


if __name__ == "__main__":
    main()
