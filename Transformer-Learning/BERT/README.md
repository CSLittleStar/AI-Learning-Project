# BERT 复现 (BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)

基于论文 [Devlin et al., 2019] 复现的 BERT 架构。**仅实现模型结构（Transformer Encoder + 预训练头），
不包含完整预训练数据管线**（BooksCorpus + Wikipedia / WordPiece 分词）。

## 文件
- `bert.py` — BERT 模型核心实现 + 演示（直接运行 `python bert.py` 执行 sanity check）

## 论文关键实现点
| 项 | 设定 |
|----|------|
| 结构 | 多层双向 Transformer **Encoder**（无 Decoder） |
| BERT_BASE | `L=12, H=768, A=12, d_ff=3072`，参数量 **~110M** |
| BERT_LARGE | `L=24, H=1024, A=16, d_ff=4096`，参数量 **~340M** |
| 输入嵌入 | `Token + Segment + Position` 三者求和（**可学习位置嵌入**，非正弦，不乘 √d） |
| 子层 | 残差 + LayerNorm（Post-LN，eps=1e-12）；注意力/FFN 后 Dropout |
| 激活 | **GELU**（非 ReLU） |
| 预训练目标 | Masked LM（随机遮盖 15%：80%→[MASK], 10%→随机, 10%→不变）+ NSP |
| 权重共享 | 输入词嵌入与 MLM 输出投影共享权重（与原始实现一致） |
| 特殊符号 | `[PAD]=0, [CLS]=101, [SEP]=102, [MASK]=103` |

## 提供的模型类（预训练 + 下游微调）
对应论文第 3 节：

| 类 | 用途 | 论文任务 |
|----|------|----------|
| `BertModel` | BERT 主体（Encoder 堆叠），输出 `sequence_output` / `pooled_output` | — |
| `BertForPreTraining` | MLM + NSP 预训练头 | 预训练 |
| `BertForSequenceClassification` | 取 `[CLS]` pooled 接线性头 | 句对/单句分类 (MNLI, QQP, SST-2, CoLA, MRPC, RTE, QNLI) |
| `BertForTokenClassification` | 每个 token 接分类头 | 序列标注 (NER) |
| `BertForQuestionAnswering` | start/end 位置打分 | SQuAD v1.1 抽取式问答 |

Fine-tuning 做法（论文 3.2 节）：在 BERT 主体之上加一个（或极少量）线性层，与 BERT
**端到端一起微调**，除 dropout 外不引入新结构。

## 运行
```bash
cd Transformer-Learning/BERT
python bert.py         # 参数量核对 + 单句/句对前向 + MLM/NSP + 三类 Fine-tuning 演示
```

## 用法示例
```python
from bert import (BertForPreTraining, BertForSequenceClassification,
                  BertForTokenClassification, BertForQuestionAnswering,
                  PRETRAINED_CONFIGS)

# 直接按论文规格构建 BERT_BASE
model = BertForPreTraining(vocab_size=30522, **PRETRAINED_CONFIGS["bert-base-uncased"])

# 预训练: 返回 (mlm_logits, nsp_logits)
mlm_logits, nsp_logits = model(input_ids, token_type_ids, attention_mask)

# 句对分类 (SST-2 / MNLI 等)
cls_model = BertForSequenceClassification(num_labels=2,
                 vocab_size=30522, **PRETRAINED_CONFIGS["bert-base-uncased"])
logits, loss = cls_model(input_ids, token_type_ids, attention_mask, labels)

# 序列标注 (NER), 每 token 一个标签
tok_model = BertForTokenClassification(num_labels=9,
                 vocab_size=30522, **PRETRAINED_CONFIGS["bert-base-uncased"])

# SQuAD 抽取式问答, 预测答案起止位置
qa_model = BertForQuestionAnswering(vocab_size=30522,
                 **PRETRAINED_CONFIGS["bert-base-uncased"])
start_logits, end_logits, loss = qa_model(input_ids, token_type_ids,
                                           attention_mask, start_positions, end_positions)
```
