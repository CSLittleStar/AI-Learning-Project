# Transformer模型的代码实现

## 代码记录
- torch.nn.Linear(in_features, out_features, bias=True)：全连接层，in_features输入特征维度，out_features输出特征维度，bias是否添加偏置项
- 

## Transformer模型基础
### Attention is all you need
- 从RNN/LSTM到Transformer的改变：
    - RNN长距离难搞，且天然串行
    - LSTM尽量解决，但长距离仍然困难
    - Transformer提出：让序列中的每个词直接看到其他词。即Self-Attention自注意力。
- Attention的做法：根据当前词与其他词之间的相关程度，为不同词分配不同权重。最终做加权组合。
- Self-attention：序列内部的元素彼此计算注意力。即每一个 token 都可以与所有 token 建立关系。
- Q、K、V的概念：
    - Q：Query查询；K：Key键；V：Value值
    - e.g. 去图书馆找书
        - 我提出 Query：我想找机器学习相关的书
        - 书架上的书具有属性 Key：这本书是什么类型；Value：这本书具体包含什么内容
        - 首先比较 Q <--> K，找到相关内容后再取 V
    - 总结流程：Q ---比较---> K ---Attention---> V 
- Self-Attention核心公式：
    - $$
      \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
      $$
    - 第一步：计算Q和K的相似度，找出“每一个词应该关注其他哪些词”，得到一个注意力分数矩阵
    - 第二步：除以$\sqrt{d_k}$ ，防止点积结果过大导致 softmax 梯度消失（维度越高，点积方差越大）
    - 第三步：Softmax，将分数转换为概率/权重（这些权重累加后等于1）
    - 第四步：加权V，根据注意力权重，从其他token中提取信息
- Transformer整体架构：Encoder-Decoder架构
    - Input --> Encoder(Selft-Attention --> Feed-Forward) --> Encoder_output --> Decoder(Masked Self-Attention --> Cross-Attention --> Feed Forward) --> Linear --> Softmax --> Output
    - Encoder：由多个完全相同的Encoder Layer堆叠，每个Layer存在两个核心模块
        - Multi-Head Self-Attention：让模型从多个不同的表示子空间学习 token 之间的关系。
            - 多个Attention，不同head学习不同类型的关系
            - e.g. 主语 ↔ 动词；代词 ↔ 名词；相邻词；长距离依赖
            - $$
              \begin{aligned}
              \text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^{O} \\
              \text{where} \quad \text{head}_i &= \text{Attention}(QW_i^{Q}, KW_i^{K}, VW_i^{V})
              \end{aligned}
              $$
            - $W_i^{Q} \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^{K} \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^{V} \in \mathbb{R}^{d_{\text{model}} \times d_v}$：$h$ 个头各自的投影矩阵
            - $W^{O} \in \mathbb{R}^{hd_v \times d_{\text{model}}}$：输出投影矩阵
            - 多头机制让模型在不同子空间并行关注不同位置的不同表示子空间

        - Feed Forward Network：FFN(x)=max(0, x * W1 + b1) * W2 + b2
            - Attention负责不同token之间的信息交流
            - FFN负责对每一个token自己进行特征变换
            - Linear -> ReLU -> Linear
        - Encoder Layer 的残差连接（Add & Norm）模块流程图：
            ```
            Input
            │
            ├───────────────┐
            ↓               │
            Multi-Head        │
            Self-Attention    │
            ↓               │
            Add & Norm ←──────┘
            │
            ├───────────────┐
            ↓               │
            Feed Forward      │
            ↓               │
            Add & Norm ←──────┘
            │
            Output
            ```
        - 每个子层（Self-Attention、FFN）外都包了 **残差连接 + LayerNorm（Add & Norm）**
        - 残差连接缓解深层网络梯度消失（Residual 的思想）
        - LayerNorm 稳定训练、加速收敛
        - Positional Encoding（位置编码）：确认顺序（Input = Embedding + PositionalEncoding）
    - Decoder：三个部分
        - Masked Self-Attention：
            - 为什么要mask？废话，输出只可能是前推后，否则不是作弊嘛//
            - causal mask/masked self-attention：保证模型只能看到当前位置之前的信息
            - $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}} + M\right)V $$
            - $M$ 为掩码矩阵：被遮挡的位置填入 $-\infty$（softmax 后权重趋于 0），使第 $i$ 个位置只能看到 $\le i$ 的位置。
        - Cross Attention
            - 也称Encoder-Decoder Attention，让Decoder关注Encoder的输出
            - Encoder --- 提供源语言信息 ---> Cross-Attention <--- Decoder
        - Feed Forward
- 反向传播计算梯度
    - Q、K、V的对应可训练参数：Wq、Wk、Wv
    - Token Embedding：Embedding矩阵
    - Multi-Head Attention输出：Wo
    - FFN：W1、W2、b1、b2
    - LayerNorm：γ、β
    - 输出层：Wout、bout
    - 一些求导的细节：b1、b2、γ、β、bout都是累加的导数，因为在公式中为了让矩阵计算成立，同一个b都是给每个token使用的，因此求导时要对每个token求和。

- 论文提出的核心内容
    - 用Attention代替RNN
    - self-attention解决长依赖问题
    - multi-head attention同时学习不同类型的关系
    - positional encoding让模型直到token的顺序
    - encoder-decoder transformer
    - 高度并行化
- 一些额外的知识点：
    - Add&Norm：Add是残差；Norm是LayerNorm归一化
    - Embedding矩阵：可供训练的token向量查找表，用以表示相近语义的token
    - 缩放点积注意力：就是Self-Attention那个核心公式
    - 位置编码的实现方式：偶数采用sin，奇数采用cos。

### BERT
- 提出问题：
    - Transformer的词语只存在一个固定向量，无法在不同语言下表达不同含义。即缺乏Contextualized Representation（上下文相关表示）
    - RNN/LSTM可以利用上下文，但存在方向的问题。即只能左边到右边，而BERT的突破在于让模型在预训练过程中真正利用双向上下文。
- BERT = Transformer Encoder + 双向预训练 + 下游任务微调
    - BERT没用Transformer Decoder。因为它只做语言理解。
    - 输入前存在预处理，将[CLS], [SEP]作为特殊token插入token化的文本中，再进行Embedding
    - BERT输入Embedding = TokenEmbedding + PositionEmbedding + SegmentEmbedding
        - SegmentEmbedding的作用：告诉Transformer当前的Token属于第一个句子，还是第二个句子（亦或是第i个句子）
    - 重要思想：Pre-training + Fine-tuning
- 预训练Pre-training：
    - 用MLM和NSP学习通用语言知识
    - 核心创新：Masked Language Model
        - 随机遮住输入中的一些词，让模型预测
        - 特殊设计：避免Pre-training / Fine-tuning mismatch，80%的输入被遮住，10%的输入被替换，10%的输入不变
        - 学习token与上下文之间的关系
    - Next Sentence Prediction
        - 正样本标签[IsNext]，负样本标签[NotNext]
        - 输入两个句子，判断第二个句子是否是第一个句子的下一句。
        - 通过[CLS]的表示进行二分类：[CLS] Sentence A [SEP] Sentence B [SEP] --> Transformer --> [CLS] vector --> Classifier --> IsNext/NotNext
        - 学习句子与句子之间的关系
- 微调Fine-tuning：
    - 选择具体下游任务作为训练目标，学会具体任务
    - 分类：
        - 文本分类（[CLS] This movie is great! [SEP]）：输入取[CLS]对应的向量h_CLS，y=softmax(W * h_CLS + b)
        - 两句话分类（[CLS] Sentence A [SEP] Sentence B [SEP]）：主要使用h_CLS
    - QA问答（[CLS] Question [SEP] Passage [SEP]）：模型输出每个token表示为h1, h2, ..., hn ，分别预测答案开始位置和答案结束位置。
    - 命名实体识别NER
- 创新点总结：
    - （核心）深度双向Transformer表示（Deep Bidirectional）：通过Masked Language Model让每个Token同时利用左右上下文
    - 统一的预训练—微调框架（Pre-training / Fine-tuning）：一个预训练模型->增加一个简单任务层->Fine-tuning
    - 把 Transformer Encoder 推向通用 NLP Backbone
- 一些问题：
    - MLM存在[Mask]不一致的问题，Pre-training / Fine-tuning mismatch
    - 训练效率问题：相比从左到右的模型收敛略慢，双向性换来了训练目标上的一些效率损失。
    - 存在最大长度限制（SequenceLength≤512）


### ViT
#### 模型概述：
- 输入 = Patch Embedding + CLS + Position Embedding
    - Patch Embedding：将一张图片切成patch，作为token进Transformer。最终得到$z_{i}$
    - [CLS] Token：输入变为$z_{0}=[x_{class}; x^{1}_{p}E; x^{2}_{p}E]; ...; x^{N}_{p}E$，经过Encoder后，$z_{l} = [z^{0}_{L}; ...]$。其中，$z^{0}_{L}$是整张图的全局表示。
    - Position Embedding：$z_{0}=[x_{class}; x^{1}_{p}E; x^{2}_{p}E]; ...; x^{N}_{p}E + E_{pos}$
- 进入Transformer Encoder：
    - layerNorm
    - Multi-Head Attention
    - Residual + LayerNorm
    - MLP（约等于Transformer的FFN）：$MLP(x) = W_{2} GELU(W_{1}x + b_{1}) + b_{2}$
    - Residual
- $z^{0}_{L}$进入分类Head，完成Classification。
- 预训练+微调：在完成上述预训练，得到一个通用结果后，可以进行微调，采用别的数据集，让 ViT 学习具有迁移价值的通用视觉表示。使模型适应具体下游数据分布和任务。
#### 研究：
- 隐藏维度D：输入时是N = $P^{2}C$ 的维度，需要更正成Transformer的维度D，因此需要一个矩阵 E = N * D$ ，最终 (1*N) (N*D)=D
- Transformer叠层的作用：
    - Head是横向的，每个Encoder Layer有多个Head，并行工作处理各自的内容。
    - Encoder Layer是纵向的，网络越深，可以进行更复杂的逐层特征变换。
- Patch的作用：
    - patch越小，细节越多，Token越多，计算量暴涨
- ViT与CNN比较：
    - CNN：有限的数据集下，做局部学习
    - Transformer：大规模数据集中，做全局学习
### CLIP
#### 传统CNN/ViT模型的局限性：
- 类别固定：需要根据分类器找类别，没有的/没训练的就无法输出识别
- 需要人工标注：一般需要人工根据图片标注标签，而CLIP观察了本身存在的“图片+自然语言”
- 传统视觉模型迁移能力有限：训练得到分类，换任务通常要重构数据集+重新训练；CLIP希望能一次预训练->自然语言描述任务->zero-shot完成任务

#### 核心思想：不要直接告诉模型“这个图片属于xx类”，而是告诉模型“这个图片和哪段文字匹配”
- Contrastive Language–Image Pre-training。（CLIP）：学习Image Encoder 和 Text Encoder，使匹配的图文表示相互接近，不匹配的表示相互远离。

#### 整体架构
- Image --- Image Encoder (CNN/ViT) ---> Image Feature --- Projection ---> Image Embedding
- Text  --- Text Encoder (类BERT)   ---> Text Feature  --- Projection ---> Text Embedding
- Image Embedding ---> Contrastive Learning <--- Text Embedding
- 注意：CLIP并非一个Encoder，而是Image Encoder + Text Encoder + Projection + Contrastive Loss组成的完整模型

#### projection投影层
- 本质上是一个可学习的线性变换，把Image Encoder 和 Text Encoder 输出的特征，映射到同一个、维度一致的多模态共享 embedding 空间中。
- 最简单的情况就是个Linear Layer
- 更重要的作用是：学习如何把视觉特征和文本特征映射到一个能够进行语义比较的共同空间。
- 负责把已经提取出的特征“转换到适合跨模态比较的空间”。
- 这部分需要学习的参数，学习的是“怎样把两个模态的特征对齐”

#### 如何对齐
- similarity(Image_i, Text_j)
- 形成一个N * N的相似度矩阵
- 理想情况是：对角线最大，即Image_i <-> Text_i
- 核心损失函数：
    - CLIP实质是两个方向的对比，Image -> Text , Text -> Image
    - 所以Contrastive Loss = Image-to-Text Loss + Text-to-Image Loss
    - 反向推导的学习参数调整：similarity
- 对齐的步骤
    - 首先要让两个Encoder的输出维度相同，否则算不了一点
    - 其次不能直接比较，因为数值尺度并不一定相同。需要做L2归一化
    - 然后计算Cosine Similarity，即similarity(I, T) = 归一化的I * 归一化的T （为什么是点积？x * y = ||x| * |y|| * cosθ推导）
    - 取一个temperature参数，计算的similarity * 该参数，放大对齐的差异性
#### 怎么实现Zero-shot Classification？
- Zero-shot Learning 是指模型在训练阶段没有见过某些目标类别的训练样本，但在测试阶段仍然能够识别这些类别。
- “zero”：对于测试目标类别，训练阶段拥有 0 个该类别的样本。
    - 如何让模型知道？：需要一种能够描述类别的“辅助信息（semantic information）”。
    - 传统的做法往往利用：类别属性；类别描述；Word Embedding；文本语义；知识图谱等
- Zero-shot Image Classification：自然语言本身就变成了分类器的类别描述。
    - 不需要提前固定cls neuron， 类别的语义描述替代
- 概念区分：
    - Zero-shot：新类别训练样本=0
    - One-shot：新类别训练样本=1
    - Few-shot：少量
    - Full-shot：大量

#### 创新点：
1. 自然语言作为视觉监督
2. 图像和文本共享语义空间
3. Constrastive Learning
4. Zero-shot Transfer

#### 问题：
- 不擅长计数
- 不擅长处理复杂的空间关系
- Fine-grained Recognition能力有限
- Prompt敏感
- 数据本身存在噪声和偏差

#### 一些思考：
- CLIP的输入Embedding也需要ViT的CLS Token，但这不是用来做类别分类的，而是一个全局图像的全局语义表示，做出Image Embedding