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
    - 缩放点积注意力：
    - 位置编码的实现方式：
### BERT

### ViT


### CLIP