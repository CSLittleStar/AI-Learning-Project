"""
Recurrent Neural Network (LSTM) with Dropout Regularization
============================================================

复现论文:
    Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
    "Recurrent Neural Network Regularization", ICLR 2015
    https://arxiv.org/abs/1409.2329

核心思想:
    1. 使用 LSTM (Long Short-Term Memory) 单元的多层深度 RNN。
    2. 提出一种针对 RNN/LSTM 的 dropout 正则化方案:
       - dropout 只作用在 **非循环连接 (non-recurrent connections)** 上,
         即当前层输入 h_{t-1}^{(l-1)} (来自下一层相同时间步) 与输出之间。
       - **循环连接 (recurrent connections)** h_{t-1}^{(l)} (同一层上一时间步)
         不做 dropout,以免破坏 LSTM 长期记忆能力。

论文中的符号与公式 (Section 3):
    - 下标 t 表示时间步, 上标 l 表示层数。
    - 所有隐藏状态 h 均为 n 维向量。
    - T^{n,m} 为仿射变换 (W x + b)。
    - LSTM 动态 (Graves et al., 2013):
          [ i ]       [ sigmoid ]            [ h_{t-1}^{(l)} ]
          [ f ]   =   [ sigmoid ] T^{2n,4n} [              ]
          [ o ]       [ sigmoid ]            [ h_{t}^{(l-1)} ]
          [ g ]       [  tanh  ]
          c_t^{(l)} = f * c_{t-1}^{(l)} + i * g
          h_t^{(l)} = o * tanh(c_t^{(l)})
      其中 * 表示逐元素乘法, sigmoid / tanh 逐元素作用。

    - 加入 dropout (只作用于非循环连接, 即输入 h_{t-1}^{(l-1)} 处):
          [ i ]       [ sigmoid ]            [ h_{t-1}^{(l)}     ]
          [ f ]   =   [ sigmoid ] T^{2n,4n} [ D(h_{t}^{(l-1)}) ]
          [ o ]       [ sigmoid ]            [                  ]
          [ g ]       [  tanh  ]
          c_t^{(l)} = f * c_{t-1}^{(l)} + i * g
          h_t^{(l)} = o * tanh(c_t^{(l)})

本实现用 PyTorch 完整还原上述结构, 并在 forward 时:
    - 对"层间输入"(非循环连接)使用同一个 dropout mask (变分 dropout),
      保证同一序列前向传播中 mask 一致, 且不在循环连接上做 dropout。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """
    单个 LSTM 单元 (对应论文公式 Section 3.1)。

    这里把循环输入 (h_{t-1}^{(l)}, 来自上一时间步同一层) 与
    层间输入 (h_{t}^{(l-1)}, 来自下一层同一时间步) 分开处理,
    以便只对层间输入施加 dropout (非循环连接)。

    参数:
        input_size : 本层输入特征维度 (即 h_{t}^{(l-1)} 的维度)
        hidden_size: 隐藏状态维度 n
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 循环部分的仿射变换: T^{n,n} 作用在 h_{t-1}^{(l)} 上, 输出 4n (i,f,o,g)
        self.W_recur = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # 层间部分的仿射变换: T^{n,n} 作用在 h_{t}^{(l-1)} 上, 输出 4n (i,f,o,g)
        # 若这是第 0 层, 则 layer 输入为词向量/嵌入, 维度为 input_size
        self.W_input = nn.Linear(input_size, 4 * hidden_size, bias=True)

    def forward(self, x_t, h_prev, c_prev, dropout_mask=None):
        """
        单步前向传播。

        参数:
            x_t        : 层间输入 h_{t}^{(l-1)}, shape (batch, input_size)
                         若已做 dropout 则传入已 mask 的结果。
            h_prev     : 循环输入 h_{t-1}^{(l)}, shape (batch, hidden_size)
            c_prev     : 上一记忆单元 c_{t-1}^{(l)}, shape (batch, hidden_size)
            dropout_mask: 应用于 x_t 的 dropout mask (在外部预先生成),
                          None 表示不 dropout。
        返回:
            h_t, c_t
        """
        if dropout_mask is not None:
            x_t = x_t * dropout_mask

        # 论文公式: [i,f,o,g]^T = T^{2n,4n} [h_{t-1}^{(l)}; h_{t}^{(l-1)}]
        gates = self.W_recur(h_prev) + self.W_input(x_t)  # (batch, 4n)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_gate)

        # c_t = f * c_{t-1} + i * g
        c_t = f * c_prev + i * g
        # h_t = o * tanh(c_t)
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class RegularizedLSTM(nn.Module):
    """
    多层 LSTM RNN, 实现论文的 dropout 正则化方案。

    参数:
        input_size : 输入(词向量/嵌入)维度
        hidden_size: 每层隐藏状态维度 n
        num_layers : 层数 L
        dropout    : 非循环连接上的 dropout 概率 (论文: 中等 0.5, 大型 0.65)。
                     设为 0 即退化为普通 LSTM。
        vocab_size : 词表大小 (用于输出投影到词表)。若提供则包含输出层。
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.0, vocab_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 构建多层 LSTM cell。
        # 第 0 层输入维度为 input_size, 其余层输入维度为 hidden_size。
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_dim = input_size if l == 0 else hidden_size
            self.cells.append(LSTMCell(in_dim, hidden_size))

        # 输出层: 将最顶层隐藏状态 h_t^{(L)} 投影到词表分布
        self.has_output = vocab_size is not None
        if self.has_output:
            self.decoder = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # 论文: 权重均匀初始化在 [-0.05, 0.05] (中等) / [-0.04, 0.04] (大型) 区间。
        # 这里默认使用 [-0.05, 0.05], 可通过 reset_parameters(scale) 调整。
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.05, 0.05)

    def reset_parameters(self, lo: float, hi: float):
        """按给定区间重新均匀初始化权重 (对应论文不同规模设定的初始化范围)。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, lo, hi)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, lo, hi)

    def _init_hidden(self, batch_size, device):
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0, c0

    def forward(self, x, hidden=None):
        """
        前向传播 (按时间步展开)。

        参数:
            x      : 输入序列, shape (batch, seq_len, input_size)
            hidden : 初始 (h, c) 元组, 每层一个; 若 None 则初始化为 0。
                     论文: 把当前 minibatch 的最终隐藏状态作为下一个 minibatch
                     的初始隐藏状态 (successive minibatches 顺序遍历训练集)。
                     注意: 此处 hidden 不固定为 0, 支持跨 minibatch 传递。
        返回:
            output : 最顶层每个时间步的隐藏状态, shape (batch, seq_len, hidden_size)
            hidden : 最终 (h, c)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # hidden[l] = (h, c) for layer l
        if hidden is None:
            hidden = [self._init_hidden(batch_size, device) for _ in range(self.num_layers)]
        else:
            hidden = list(hidden)

        # 论文关键: 对每个层、每个序列实例, 生成 **同一时间步内一致的 dropout mask**。
        # 变分 dropout —— 同一个序列前向传播中 mask 固定, 不随 t 改变,
        # 且只对层间输入 (非循环连接) 使用; 循环连接 h_{t-1}^{(l)} 不使用。
        if self.dropout > 0:
            # 每个 layer 一个 mask, 应用到该层输入的 batch 上
            masks = []
            for _ in range(self.num_layers):
                # (batch, input_dim) 的 mask, input_dim 对第0层是 input_size,
                # 其余层是 hidden_size
                m = torch.empty(batch_size, self.hidden_size, device=device)
                m = (torch.rand_like(m) > self.dropout).float() / (1 - self.dropout)
                masks.append(m)
        else:
            masks = [None] * self.num_layers

        # 保存每层每个时间步的输出, 用于给上一层下一时间步提供输入
        # layer_inputs[l] 是传给第 l 层的输入序列
        layer_inputs = [x] + [None] * self.num_layers

        final_hidden = []
        for l in range(self.num_layers):
            h, c = hidden[l]
            mask = masks[l]
            # 第 l 层的输入序列
            inp = layer_inputs[l]  # (batch, seq_len, in_dim)
            outs = []
            for t in range(seq_len):
                x_t = inp[:, t, :]  # (batch, in_dim)
                h, c = self.cells[l](x_t, h, c, dropout_mask=mask)
                outs.append(h.unsqueeze(1))
            layer_out = torch.cat(outs, dim=1)  # (batch, seq_len, hidden_size)
            layer_inputs[l + 1] = layer_out
            final_hidden.append((h, c))

        output = layer_inputs[self.num_layers]  # 最顶层输出

        if self.has_output:
            # 将每个时间步的隐藏状态映射到词表 logits
            # (batch*seq_len, hidden_size) -> (batch*seq_len, vocab)
            logits = self.decoder(output.reshape(-1, self.hidden_size))
            logits = logits.reshape(batch_size, seq_len, -1)
            return logits, final_hidden

        return output, final_hidden


def build_medium_model(vocab_size: int):
    """
    论文 "Medium regularized LSTM" 配置 (Table 1, Section 4.1):
        - 2 层, 每层 650 单元
        - 50% dropout 在非循环连接 (= 0.5)
        - 权重均匀初始化于 [-0.05, 0.05]
    """
    model = RegularizedLSTM(
        input_size=650, hidden_size=650, num_layers=2,
        dropout=0.5, vocab_size=vocab_size,
    )
    model.reset_parameters(-0.05, 0.05)
    return model


def build_large_model(vocab_size: int):
    """
    论文 "Large regularized LSTM" 配置 (Section 4.1):
        - 2 层, 每层 1500 单元
        - 65% dropout 在非循环连接 (= 0.65)
        - 权重均匀初始化于 [-0.04, 0.04]
    """
    model = RegularizedLSTM(
        input_size=1500, hidden_size=1500, num_layers=2,
        dropout=0.65, vocab_size=vocab_size,
    )
    model.reset_parameters(-0.04, 0.04)
    return model


if __name__ == "__main__":
    # 简单的功能自检: 构造一个小模型, 跑通前向并验证形状。
    torch.manual_seed(0)
    vocab = 1000
    batch, seq_len = 20, 35  # 论文: minibatch=20, unroll=35
    model = RegularizedLSTM(
        input_size=200, hidden_size=200, num_layers=2,
        dropout=0.5, vocab_size=vocab,
    )

    # 用嵌入层把词 id 映射为向量
    embed = nn.Embedding(vocab, 200)
    inp = torch.randint(0, vocab, (batch, seq_len))
    x = embed(inp)

    logits, hidden = model(x)
    print("input shape :", tuple(x.shape))
    print("logits shape:", tuple(logits.shape))  # 期望 (batch, seq_len, vocab)
    print("num layers  :", len(hidden))
    print("OK - 前向传播通过")
