"""BERT 模型实现"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 设备: 优先 XPU (Intel Arc), 退回 CPU
# ----------------------------------------------------------------------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# 1. 缩放点积注意力 (对应 Vaswani et al. 2017, 3.2.1)
# ----------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        # attn_mask 中为 True 的位置被屏蔽(置 -inf)
        scores = scores.masked_fill(attn_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# ----------------------------------------------------------------------------
# 2. 多头自注意力 (3.2.2)
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attn_mask=None):
        batch_size = hidden_states.size(0)

        def split_heads(x):
            # (B, L, d_model) -> (B, n_heads, L, d_k)
            x = x.view(batch_size, -1, self.n_heads, self.d_k)
            return x.permute(0, 2, 1, 3)

        q = split_heads(self.w_q(hidden_states))
        k = split_heads(self.w_k(hidden_states))
        v = split_heads(self.w_v(hidden_states))

        if attn_mask is not None:
            # 将 mask 广播到 (B, n_heads, L_q, L_k)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        context, attn_weights = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout=self.dropout
        )
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.w_o(context), attn_weights


# ----------------------------------------------------------------------------
# 3. 前馈网络 (3.3) —— 使用 GELU 激活 (BERT 原始实现)
# ----------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GELU 激活 (BERT 论文/原始代码使用 exact GELU)
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ----------------------------------------------------------------------------
# 4. Transformer Encoder 层 (Post-LN: LayerNorm(x + Sublayer(x)))
# ----------------------------------------------------------------------------
class BertEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states, attn_mask=None):
        # 注意力子层: 残差 + LayerNorm
        attn_out, _ = self.attention(hidden_states, attn_mask=attn_mask)
        hidden_states = self.norm1(hidden_states + self.dropout1(attn_out))
        # 前馈子层: 残差 + LayerNorm
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout2(ffn_out))
        return hidden_states


# ----------------------------------------------------------------------------
# 5. 输入嵌入: token + segment + position 三者求和
# ----------------------------------------------------------------------------
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # BERT 使用可学习的位置嵌入 (非正弦)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # segment (token type) 嵌入: 区分句子 A / 句子 B
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # 位置 id 缓冲区: 0,1,2,... 供任意长度序列直接使用
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).unsqueeze(0)
        )

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.position_ids[:, :seq_len]
        # 三者求和 (BERT 不做 embedding 缩放)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


# ----------------------------------------------------------------------------
# 6. BERT 主体 (Encoder 堆叠) + 预训练头
# ----------------------------------------------------------------------------
class BertModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = None, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings,
            type_vocab_size, dropout,
        )
        d_ff = intermediate_size or 4 * hidden_size
        self.encoder = nn.ModuleList(
            [BertEncoderLayer(hidden_size, num_attention_heads, d_ff, dropout)
             for _ in range(num_hidden_layers)]
        )
        # 池化层 (用于 [CLS] 的序列级表示, 论文中分类任务使用)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        input_ids:      (B, L) token id
        token_type_ids: (B, L) 0=句子A, 1=句子B
        attention_mask: (B, L) 1=真实 token, 0=<pad> (会被转成注意力屏蔽)
        返回:
            sequence_output: (B, L, H) 每层 token 的上下文表示
            pooled_output:   (B, H)     [CLS] 经池化后的序列表示
        """
        # pad 位置屏蔽: True 表示被屏蔽
        extended_mask = None
        if attention_mask is not None:
            extended_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # 扩展到 (B, 1, 1, L)

        hidden_states = self.embeddings(input_ids, token_type_ids)
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attn_mask=extended_mask)

        sequence_output = hidden_states
        # pooled_output: 取 [CLS](首 token) 再过一层 Tanh 线性 (BERT 惯例)
        first_token = sequence_output[:, 0, :]
        pooled_output = self.pooler_activation(self.pooler(first_token))
        return sequence_output, pooled_output


# ----------------------------------------------------------------------------
# 7. 预训练任务头: MLM + NSP
# ----------------------------------------------------------------------------
class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, word_embeddings):
        super().__init__()
        # MLM: 先一个非线性变换(LayerNorm+线性, dense), 再投影到词表
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # 与输入词嵌入共享权重 (BERT 原始实现, 可大幅减少参数量)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = word_embeddings.weight
        # NSP: 序列对是否连续 (二分类)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        mlm_hidden = self.transform_act(self.dense(sequence_output))
        mlm_hidden = self.LayerNorm(mlm_hidden)
        prediction_scores = self.decoder(mlm_hidden)       # (B, L, vocab)
        seq_relationship_score = self.seq_relationship(pooled_output)  # (B, 2)
        return prediction_scores, seq_relationship_score


class BertForPreTraining(nn.Module):
    """完整预训练模型: BERT 主体 + MLM/NSP 头。"""

    def __init__(self, vocab_size: int, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = None, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size, dropout,
        )
        self.cls = BertPreTrainingHeads(
            hidden_size, vocab_size, self.bert.embeddings.word_embeddings
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        return prediction_scores, seq_relationship_score


# ----------------------------------------------------------------------------
# 7.5 下游 Fine-tuning 任务头 (对应论文 3.2 节 Fine-tuning)
# 做法: 取 BERT 主体输出, 在最上层接一个(或少量)线性层,
#       与 BERT 一起端到端微调。除 dropout 外不引入新结构。
# ----------------------------------------------------------------------------
class BertForSequenceClassification(nn.Module):
    """句子/句对分类 (论文表1: MNLI, QQP, RTE, SST-2, CoLA, MRPC, QNLI ...)

    取 [CLS] 的 pooled_output 接线性分类头。
    """

    def __init__(self, vocab_size: int, num_labels: int = 2, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = None, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size, dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(self.dropout(pooled_output))   # (B, num_labels)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


class BertForTokenClassification(nn.Module):
    """词级分类 / 序列标注 (论文表1: NER 等)

    对每一个 token 的 sequence_output 接分类头。
    """

    def __init__(self, vocab_size: int, num_labels: int = 2, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = None, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size, dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(self.dropout(sequence_output))  # (B, L, num_labels)
        loss = None
        if labels is not None:
            # 忽略 padding: attention_mask==0 的位置不参与损失
            loss_fct = F.cross_entropy
            if attention_mask is not None:
                active = attention_mask.view(-1).bool()
                active_logits = logits.view(-1, logits.size(-1))[active]
                active_labels = labels.view(-1)[active]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss


class BertForQuestionAnswering(nn.Module):
    """SQuAD 抽取式问答 (论文表1: SQuAD v1.1)

    问题+段落拼成一个序列 ([CLS] 问题 [SEP] 段落 [SEP]),
    预测答案在段落中的起止 token 位置: 两个向量对序列打分。
    """

    def __init__(self, vocab_size: int, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = None, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, max_position_embeddings, type_vocab_size, dropout,
        )
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 0=start, 1=end

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(sequence_output)       # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)         # (B, L)
        end_logits = end_logits.squeeze(-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            # start/end 标签为 (B,) 标量位置, 直接对 (B, L) logits 求交叉熵。
            # 为忽略 padding 的影响, 把 padding 位置的打分压到 -inf 后再归一。
            if attention_mask is not None:
                neg = (1.0 - attention_mask) * -1e9
                start_logits = start_logits + neg
                end_logits = end_logits + neg
            loss = (
                F.cross_entropy(start_logits, start_positions)
                + F.cross_entropy(end_logits, end_positions)
            ) / 2
        return start_logits, end_logits, loss


# ----------------------------------------------------------------------------
# 8. 预训练目标构造: 随机遮盖 15% (MLM) + NSP 句对标签
# 论文策略: 选中的 15% token 中
#   80% 替换为 [MASK]
#   10% 替换为随机 token
#   10% 保持不变
# ----------------------------------------------------------------------------
class MaskingStrategy:
    MASK_TOKEN_ID = 103  # [MASK]
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    PAD_TOKEN_ID = 0

    @staticmethod
    def make_mlm_labels(input_ids: torch.Tensor, mask_prob: float = 0.15,
                        rng=None):
        """
        返回 (masked_ids, mlm_labels)
          mlm_labels: -100 表示不参与 MLM 损失, 否则为原始 token id
        """
        if rng is None:
            rng = torch.Generator()
        labels = input_ids.clone()
        # 候选位置: 非 [PAD]/[CLS]/[SEP] 的 token
        special = torch.isin(
            input_ids,
            torch.tensor([MaskingStrategy.CLS_TOKEN_ID,
                          MaskingStrategy.SEP_TOKEN_ID,
                          MaskingStrategy.PAD_TOKEN_ID]),
        )
        cand = ~special
        prob = torch.rand(input_ids.shape, generator=rng, device=input_ids.device)
        masked_pos = (prob < mask_prob) & cand

        labels[~masked_pos] = -100  # 仅对选中位置计算损失

        masked_ids = input_ids.clone()
        # 80% -> [MASK]
        r = torch.rand(input_ids.shape, generator=rng, device=input_ids.device)
        mask_token = masked_pos & (r < 0.8)
        random_token = masked_pos & (r >= 0.8) & (r < 0.9)
        # 10% 不变 (保持不变即可)

        masked_ids[mask_token] = MaskingStrategy.MASK_TOKEN_ID
        if random_token.any():
            vocab_size = int(input_ids.max()) + 1
            rand_ids = torch.randint(
                0, vocab_size, (random_token.sum().item(),),
                generator=rng, device=input_ids.device,
            )
            masked_ids[random_token] = rand_ids
        return masked_ids, labels


# ----------------------------------------------------------------------------
# 9. 预设规格
# ----------------------------------------------------------------------------
PRETRAINED_CONFIGS = {
    "bert-base-uncased": dict(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, max_position_embeddings=512,
    ),
    "bert-large-uncased": dict(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        intermediate_size=4096, max_position_embeddings=512,
    ),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 10. 演示 (小词表 sanity check)
# 验证模型前向形状/流程, 并演示论文中的单句/句对输入、MLM/NSP 目标构造。
# 真实预训练需要 30k WordPiece 词表 + BooksCorpus/Wikipedia 语料,
# 此处仅用极小词表走通流程并核对张量形状。
# ----------------------------------------------------------------------------
PAD, CLS, SEP, MASK = 0, 101, 102, 103


def build_pair(sent_a_tokens, sent_b_tokens):
    """构造 (input_ids, token_type_ids, attention_mask)"""
    ids = [CLS] + sent_a_tokens + [SEP]
    type_a = [0] * (len(sent_a_tokens) + 2)
    if sent_b_tokens is not None:
        ids = ids + sent_b_tokens + [SEP]
        type_b = [1] * (len(sent_b_tokens) + 1)
        type_ids = type_a + type_b
    else:
        type_ids = type_a
    attn = [1] * len(ids)
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([type_ids], dtype=torch.long),
        torch.tensor([attn], dtype=torch.long),
    )


def demo():
    device = get_device()
    torch.manual_seed(0)
    print(f"设备: {device}")

    # 用一个缩小版 BERT_BASE (hidden=128, 层数=2) 做 sanity check
    vocab_size = 320
    model = BertForPreTraining(
        vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512,
    ).to(device)
    model.eval()
    print(f"演示模型参数量: {count_parameters(model):,}")

    # ---- 单句 ----
    sent_a = [200, 201, 202, 203]           # 假设词 id
    input_ids, tok_types, attn = build_pair(sent_a, None)
    print("\n单句输入 ids:        ", input_ids.tolist())
    print("单句 token_type:     ", tok_types.tolist())

    with torch.no_grad():
        seq_out, pooled = model.bert(input_ids.to(device), tok_types.to(device),
                                     attn.to(device))
    print("单句 sequence_out 形状:", tuple(seq_out.shape), "(B, L, H)")
    print("单句 pooled_output 形状:", tuple(pooled.shape), "(B, H)")

    # ---- 句对 (NSP 正例) ----
    sent_b = [204, 205, 206]
    input_ids, tok_types, attn = build_pair(sent_a, sent_b)
    print("\n句对输入 ids:        ", input_ids.tolist())
    print("句对 token_type:     ", tok_types.tolist(), "(0=A, 1=B)")

    with torch.no_grad():
        pred_scores, nsp_scores = model(
            input_ids.to(device), tok_types.to(device), attn.to(device)
        )
    print("MLM 预测 logits 形状:", tuple(pred_scores.shape), "(B, L, vocab)")
    print("NSP 预测 logits 形状:", tuple(nsp_scores.shape), "(B, 2)")

    # ---- MLM 遮盖演示 ----
    mlm_ids, mlm_labels = MaskingStrategy.make_mlm_labels(input_ids)
    print("\nMLM 遮盖后 ids:      ", mlm_ids.tolist())
    print("MLM 标签( -100 忽略):", mlm_labels.tolist())

    # ---- 计算一次预训练损失 ----
    model.train()
    pred_scores, nsp_scores = model(mlm_ids.to(device), tok_types.to(device),
                                    attn.to(device))
    mlm_loss = torch.nn.functional.cross_entropy(
        pred_scores.view(-1, vocab_size), mlm_labels.view(-1).to(device)
    )
    nsp_loss = torch.nn.functional.cross_entropy(
        nsp_scores, torch.tensor([1], dtype=torch.long, device=device)  # 标签: 1=IsNext
    )
    print(f"\n示例 MLM loss: {mlm_loss.item():.4f} | NSP loss: {nsp_loss.item():.4f}")

    # ---- 验证 BERT_BASE / BERT_LARGE 规格参数量 ----
    print("\n--- 论文规格参数量核对 ---")
    base = BertForPreTraining(
        vocab_size=30522, **PRETRAINED_CONFIGS["bert-base-uncased"]
    ).to("cpu")
    print(f"BERT_BASE  : {count_parameters(base):,} (论文 ~110M)")
    large = BertForPreTraining(
        vocab_size=30522, **PRETRAINED_CONFIGS["bert-large-uncased"]
    ).to("cpu")
    print(f"BERT_LARGE : {count_parameters(large):,} (论文 ~340M)")

    # ---- Fine-tuning 任务演示 (论文 3.2 节) ----
    print("\n--- Fine-tuning 任务头演示 ---")
    cfg = dict(vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2,
               num_attention_heads=4, intermediate_size=512)

    # 1) 句子/句对分类 (如 SST-2, MNLI)
    cls_model = BertForSequenceClassification(num_labels=2, **cfg).to(device)
    logits, loss = cls_model(input_ids.to(device), tok_types.to(device), attn.to(device),
                             labels=torch.tensor([1], dtype=torch.long, device=device))
    print(f"序列分类 logits 形状: {tuple(logits.shape)} (B, num_labels), loss={loss.item():.4f}")

    # 2) 词级分类 / 序列标注 (如 NER)
    tok_model = BertForTokenClassification(num_labels=5, **cfg).to(device)
    # 构造每 token 标签 (padding 处填 -100 忽略, 与 attention_mask 对齐)
    tok_labels = torch.tensor([[-100, 0, 1, 2, 3, -100, 4, 0, 1, -100]],
                              dtype=torch.long, device=device)
    tok_logits, tok_loss = tok_model(input_ids.to(device), tok_types.to(device),
                                     attn.to(device), labels=tok_labels)
    print(f"序列标注 logits 形状: {tuple(tok_logits.shape)} (B, L, num_labels), loss={tok_loss.item():.4f}")

    # 3) SQuAD 抽取式问答 (起止位置预测)
    qa_model = BertForQuestionAnswering(**cfg).to(device)
    # 假设答案在序列中第 7 个 token (id=204, 落在句 B) 起止
    start_pos = torch.tensor([7], dtype=torch.long, device=device)
    end_pos = torch.tensor([8], dtype=torch.long, device=device)
    s_logits, e_logits, qa_loss = qa_model(
        input_ids.to(device), tok_types.to(device), attn.to(device),
        start_positions=start_pos, end_positions=end_pos,
    )
    print(f"SQuAD start/end logits 形状: {tuple(s_logits.shape)} (B, L), loss={qa_loss.item():.4f}")


if __name__ == "__main__":
    demo()
