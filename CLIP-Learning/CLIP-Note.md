## 代码实现内容
### CoOp
#### CoOp的可训练参数
~~~
self.ctx = nn.Parameter(
    torch.empty(n_ctx, ctx_dim)
)
~~~
- 文字->向量的替换
- 采用随机初始化或者人工Prompt初始化
    - 随机：按照一个正态分布最忌初始化，跟a photo of a 没任何关系
    - 人工Prompt：设置ctx_init="a photo of a"，用tokenizer和token embedding转化成v1~N的形式（CoOp实验采用的这种）
### MaPLe
#### MaPLe的Text Prompt初始化
~~~
prompt = clip.tokenize(ctx_init)

embedding = clip_model.token_embedding(prompt)

ctx_vectors = embedding[0, 1:1+n_ctx, :]
~~~
#### maple的F映射实现：
~~~
self.proj = nn.Linear(ctx_dim, 768)
~~~

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

### CoOp
#### 创新
- CLIP的问题：Prompt敏感，也就是仅仅改变prompt中的几个词，就可能造成很大的性能差异
    - 文中提出：对于Caltech101，仅仅给Prompt加了一个‘a’，准确率就可能提高超过5%
    - 对于不同数据集，还需要人为加入不同的领域信息
    - 总结一句话：CLIP 的图像 Encoder 很强，但最终分类效果高度依赖 Text Encoder 的输入 Prompt。
- CoOp的改进：把 CLIP 原本需要人工设计的 Prompt，变成可以通过少量标注数据自动学习的“连续向量”。
    - 将CLIP的"a photo of a [CLASS]" 改为 $[V]_{1}​[V]_{2}​⋯[V]_{M}​[CLASS]$
    - $ [V]_{1} , ​[V]_{2} , ​⋯ , [V]_{M}$是可学习的连续向量
    - 用少量训练图片计算分类Loss，反向传播到[V]，但Text Encoder和Image Encoder本身全部冻结，从而让真正更新的只有Prompt Context
    - 重要理解：CoOp并非训练CLIP，整个预训练模型都是冻结的，只训练Prompt Context
- CoOp的两个版本设计：
    - Unified Context：所有类别共享同一组Prompt（少量数据时，通用性挺好）
    - Class-Specific Context，CSC：每个类别都有自己独立的 Prompt（数据少时反而不好，参数太多，容易过拟合）
- 整体架构：
    - ![CoOp 整体架构](./CoOp%20Image.png)
    - 整体先按照CLIP的设计思路，训练好Image Encoder和Text Encoder，然后冻结预训练的所有参数
    - 在文本输入中加入一组可学习的Context Vector
    - 跟CLIP计算中唯一的不同在于，输入的Text有了改变，其他的计算过程一致
---

### 实验代码误区修正 (重要)
- **误区**：此前 `clip_cifar10.py` / `coop_cifar10.py` 自己从零实现 ViT / Text Encoder 并训练，这是错误的。
- **正解 (依据 CLIP / CoOp 论文)**：作者已在 https://github.com/openai/CLIP 开源了 CLIP 预训练模型权重，无需自己预训练。直接用：
  ```python
  import clip
  model, preprocess = clip.load("ViT-B/32")   # 加载官方开源预训练权重
  ```
- **CLIP 实验 (`clip_cifar10.py`)**：调用预训练模型后，仅用 CIFAR-10 做下游 zero-shot 分类（构造 "a photo of a {class}" 文本特征，与图像特征算相似度），不训练任何 CLIP 参数。
- **CoOp 实验 (`coop_cifar10.py`)**：同样先 `clip.load("ViT-B/32")` 加载并**冻结**整个预训练 CLIP（Image/Text Encoder + 投影 + logit_scale 全部 requires_grad=False），只训练可学习的 prompt context `[V]_1...[V]_M`（唯一的 `prompt.ctx` 参数）。
- **关键坑**：CLIP 预训练 Text Transformer 的 resblock 自带 77×77 因果 attn_mask；CoOp 的 prompt 序列长度 ≠ 77，直接调用会尺寸不匹配。需在 forward 时把 attn_mask 临时替换为当前序列长度对齐的下三角掩码（用完恢复），才能复用预训练权重。
- **依赖**：必须安装 OpenAI 官方 CLIP 包 (`pip install git+https://github.com/openai/CLIP.git`)，而不是同名的 CLI 工具包（后者没有 `clip.load`）。


    
### MaPLe
#### 创新
- 同时学习视觉 Prompt 和语言 Prompt，并且让两者建立强耦合关系。
    - Multi-model：同时作用于vision、language
    - Prompt Learning：学习少量Prompt参数、冻结CLIP
    - Coupled：让视觉 Prompt 显式依赖于语言 Prompt，从而避免两个模态分别学习出相互独立的解决方案。
- Deep Prompt：在 Transformer 的不同深度阶段，都可以加入不同的 Prompt。
    - 在不同 Transformer blocks 中使用独立的 context prompts，以获得分阶段的上下文表示。
    - 这招有效的原理：浅层transformer做局部/基础特征；中层做更复杂的语义关系；深层做高级语义表示
- 总结：MaPLe ≈ CoOp + VisionPrompt + DeepPrompt + Cross-model Coupling
#### 模型架构
- 假定数据集形式为$ D = {(x_{i}, y_{i})}$，其中$x_{i}$是图像，$y_{i}$是图像对应的文本描述。
- 文本prompt构造为[Language Context] + "$y_{i}$"
- Language Prompt -> Text Encoder -> Text Feature
- 同时：Language Prompt -> Coupling Function -> Vision Prompt -> Image Encoder -> Image Feature
- Z_image = ImageEncoder(x, Pv) , Z_text = TextEncoder(y, Pl)
- 计算similarity(Z_image, Z_text)，得到分类概率
- L = CrossEntropy(prediction, y)
- 反向传播：Loss -> Language Prompt; Loss -> Coupling Function Vision Prompt
- ![MaPLe 整体架构](./MaPLe%20Image.png)
#### 实现细节
- 输入的text是怎么初始化的？
    - 第一层Language Prompt：N_CTX=2, CTX_INIT="a photo of a", PROMPT_DEPTH=9；embedding形成[P1, P2, {CLS}]
    - 初始化的$P^{0}_{L}$就是上述的embedding
    - 训练开始后，Pl就是可学习参数了
    - 简单讲，输入初始化等价于CoOp
- 怎么做的Couple？Pl和Pv是如何产生的？
    - $P_{V} = F(P_{L})$
    - F 是一个可学习的线性投影层，作用是Pl转Pv去的时候，保证维度相同，所以做个映射
    - 反向时：Loss -> Pv -> F -> Pl
- 怎么在中间层新增Text Prompt的？这个Text Prompt是怎么来的？需要反向传播时训练吗？
    - 首先，需要更新
    - 其次，P0可以通过文本初始化，即[P0, P1, ..., PN]的token embedding结果；而Pv的text是默认随机初始化的，后续可学习改进
    - 注意：
        - Text的输入有两部分内容，一个是W0 = [P0, P1, ..., PN, [CLS]] + Position Embedding；另一个是P0，也就是Pv的初始阶段
        - 之后的EncoderLayer中，Wi是根据上一层的Wi-1得到的，P不存在这种关系，Pv每层独立，以参数形式自学习。
- Z_image和Z_text是怎么得到的？
    - 两边Encoder最终的结果，非独立参数，无需训练
