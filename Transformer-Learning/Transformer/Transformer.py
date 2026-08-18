"""Transformer 模型实现 + 训练 —— 复现论文《Attention Is All You Need》"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 设备: 优先 XPU (Intel Arc), 退回 CPU
# ============================================================================
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# 1. 位置编码 (Positional Encoding, 论文 3.5 节)
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)


# ----------------------------------------------------------------------------
# 2. 缩放点积注意力 (Scaled Dot-Product Attention, 论文 3.2.1 节)
# ----------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# ----------------------------------------------------------------------------
# 3. 多头注意力 (Multi-Head Attention, 论文 3.2.2 节)
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # split_heads 将 d_model 切成 n_heads 个 d_k 头，并行计算注意力；
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)                      # 取 batch 大小, 用于后面 reshape 还原形状

        def split_heads(x):
            x = x.view(batch_size, -1, self.n_heads, self.d_k)  # 把最后一维 d_model 拆成 (n_heads, d_k), 如 (B,L,256)->(B,L,4,64)
            return x.permute(0, 2, 1, 3)                        # 把 head 维提前 -> (B, n_heads, L, d_k), 每个头在独立维度并行计算

        q = split_heads(self.w_q(query))   # 先线性投影到 (B,L,d_model), 再切多头 -> (B, n_heads, L, d_k)
        k = split_heads(self.w_k(key))     # key 同理投影并切头
        v = split_heads(self.w_v(value))   # value 同理投影并切头

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # 2D (L,L) -> (1,1,L,L), 靠广播作用到所有 batch 和头
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)               # 3D (B,L,L) -> (B,1,L,L), 靠广播作用到所有头

        context, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout=self.dropout   # 计算缩放点积注意力, 得每头输出 (B, n_heads, L, d_k) 及注意力权重
        )
        context = (
            context.permute(0, 2, 1, 3)   # 头维挪回: (B, n_heads, L, d_k) -> (B, L, n_heads, d_k)
            .contiguous()                  # permute 后内存不连续, 重置布局以便 view
            .view(batch_size, -1, self.d_model)  # 多头拼接回 (B, L, d_model)
        )
        return self.w_o(context), attn_weights  # 用 w_o 做输出投影(多头融合), 与注意力权重一起返回


# ----------------------------------------------------------------------------
# 4. 前馈网络 (Position-wise FFN, 论文 3.3 节)
# ----------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # FFN的公式：ReLU(xW1 + b1)W2 + b2


# ----------------------------------------------------------------------------
# 5. Encoder / Decoder
# ----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Encoder层前向传播：attention计算，add&norm，ffn计算，add&norm
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_out)) # 内部做add，外部做norm
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_out))
        return src


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, src, src_mask=None):
        # 路径如下：embedding+pos_encoding准备输入，layer是encoder的计算（按层）
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        # 流程：masked的attention + cross_attention + ffn （中间都要做add&norm）
        attn1, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn1))
        attn2, _ = self.cross_attn(tgt, enc_output, enc_output, attn_mask=src_mask)
        tgt = self.norm2(tgt + self.dropout2(attn2))
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x


# ----------------------------------------------------------------------------
# 6. 完整 Transformer
# ----------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(src_vocab_size, d_model, n_heads,
                               n_encoder_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads,
                               n_decoder_layers, d_ff, dropout, max_len)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return self.generator(dec_output)


# ----------------------------------------------------------------------------
# 7. 掩码工具
# ----------------------------------------------------------------------------
def generate_square_subsequent_mask(sz, device):
    """decoder causal mask: True 处(未来位置)被屏蔽。"""
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


def create_padding_mask(seq, pad_idx):
    """(B, L) -> (B, 1, 1, L), True 处为 <pad>。"""
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


# ----------------------------------------------------------------------------
# 8. 小型双语数据集 (本地 TSV: de \t en)
# ----------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """
    从 TSV 文件加载平行句对 (每行: 'src \t tgt', 用制表符分隔两个语言)。
    自动构建源/目标词表, 并将句子转为 token id 序列 (含 <bos>/<eos>)。
    """

    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, path, max_len=30):
        self.max_len = max_len
        self.src_vocab = {  # 预留特殊符
            "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
        }
        self.tgt_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.src_inv = {v: k for k, v in self.src_vocab.items()}
        self.tgt_inv = {v: k for k, v in self.tgt_vocab.items()}

        self.pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                src_text, tgt_text = line.split("\t", 1)
                src_tokens = self._tokenize(src_text)
                tgt_tokens = self._tokenize(tgt_text)
                src_ids = self._build_vocab(src_tokens, self.src_vocab, self.src_inv)
                tgt_ids = self._build_vocab(tgt_tokens, self.tgt_vocab, self.tgt_inv)
                if src_ids and tgt_ids:
                    self.pairs.append((src_ids, tgt_ids))

    @staticmethod
    def _tokenize(text):
        return text.strip().lower().split()

    def _build_vocab(self, tokens, vocab, inv):
        ids = []
        for t in tokens:
            if t not in vocab:
                idx = len(vocab)
                vocab[t] = idx
                inv[idx] = t
            ids.append(vocab[t])
        return ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        src_ids = [self.BOS] + src_ids + [self.EOS]
        tgt_ids = [self.BOS] + tgt_ids + [self.EOS]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0, max_len=30):
    """将变长句对 pad 到本 batch 最大长度。"""
    srcs, tgts = zip(*batch)
    src_len = min(max(len(s) for s in srcs), max_len)
    tgt_len = min(max(len(t) for t in tgts), max_len)

    src_padded, tgt_padded = [], []
    for s, t in zip(srcs, tgts):
        s = s[:src_len]
        t = t[:tgt_len]
        src_padded.append(F.pad(s, (0, src_len - len(s)), value=pad_idx))
        tgt_padded.append(F.pad(t, (0, tgt_len - len(t)), value=pad_idx))
    return torch.stack(src_padded), torch.stack(tgt_padded)


# ----------------------------------------------------------------------------
# 9. 训练 + 推理
# ----------------------------------------------------------------------------
def train_transformer(epochs=30, batch_size=8, use_hf_dataset=False):
    device = get_device()
    print(f"使用设备: {device}")

    here = os.path.dirname(os.path.abspath(__file__))

    # ---- 准备数据 ----
    if use_hf_dataset:
        # 可选: 使用 HuggingFace 真实平行语料 (需 `pip install datasets` 且联网)
        # 例: Helsinki-NLP/opus-100, 配置 "de-en"
        from datasets import load_dataset
        raw = load_dataset("Helsinki-NLP/opus-100", "de-en", split="train[:5000]")
        # 简单内存构建 (此处省略完整词表构建, 仅作接口示例)
        raise NotImplementedError(
            "use_hf_dataset 需要自行实现词表构建; 默认使用本地 TSV 数据集。"
        )
    else:
        data_path = os.path.join(here, "deu_eng_small.tsv")
        dataset = TranslationDataset(data_path, max_len=30)
        src_vocab_size = len(dataset.src_vocab)
        tgt_vocab_size = len(dataset.tgt_vocab)
        print(f"数据集句对数: {len(dataset)}, "
              f"词表: de={src_vocab_size}, en={tgt_vocab_size}")

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=0, max_len=30),
    )

    # ---- 模型 ----
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    PAD = 0

    # ---- 训练循环 ----
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]          # decoder 输入 (去掉最后的 <eos>)
            tgt_label = tgt[:, 1:]           # 监督目标 (去掉开头的 <bos>)

            src_mask = create_padding_mask(src, PAD)
            L = tgt_input.size(1)
            causal = generate_square_subsequent_mask(L, device)
            tgt_pad = create_padding_mask(tgt_input, PAD).squeeze(1).squeeze(1)
            tgt_mask = (causal.unsqueeze(0) | tgt_pad.unsqueeze(1).unsqueeze(2)).to(torch.bool)

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_label.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * src.size(0)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:03d}/{epochs} | loss = {avg_loss:.4f} | lr = {scheduler.get_last_lr()[0]:.5f}")

    # ---- 贪心推理示例 ----
    print("\n=== 推理测试 ===")
    model.eval()
    inv_src, inv_tgt = dataset.src_inv, dataset.tgt_inv
    test_src = ["ich liebe dich .", "wir essen brot .", "sie singt ein lied ."]
    for sent in test_src:
        ids = [TranslationDataset.BOS] + [
            dataset.src_vocab.get(w, TranslationDataset.UNK) for w in sent.split()
        ] + [TranslationDataset.EOS]
        src_tensor = torch.tensor([ids], dtype=torch.long).to(device)
        src_mask = create_padding_mask(src_tensor, PAD)
        with torch.no_grad():
            enc_out = model.encoder(src_tensor, src_mask)
            ys = torch.tensor([[TranslationDataset.BOS]], dtype=torch.long).to(device)
            for _ in range(30):
                L = ys.size(1)
                causal = generate_square_subsequent_mask(L, device)
                tgt_pad = create_padding_mask(ys, PAD).squeeze(1).squeeze(1)
                tgt_mask = (causal.unsqueeze(0) | tgt_pad.unsqueeze(1).unsqueeze(2)).to(torch.bool)
                out = model.decoder(ys, enc_out, tgt_mask=tgt_mask)
                nxt = model.generator(out[:, -1, :]).argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, nxt], dim=1)
                if nxt.item() == TranslationDataset.EOS:
                    break
        out_tokens = [inv_tgt.get(i, "<unk>") for i in ys[0].tolist()
                      if i not in (TranslationDataset.BOS, TranslationDataset.EOS, TranslationDataset.PAD)]
        print(f"  DE: {sent}")
        print(f"  EN: {' '.join(out_tokens)}")


if __name__ == "__main__":
    train_transformer(epochs=60, batch_size=8, use_hf_dataset=False)
