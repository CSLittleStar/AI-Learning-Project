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

### 以下内容不再做实验验证
### KgCoOp (CVPR23)
- 创新：直接在特征空间约束“学习后的 Prompt 不要离原始 Prompt 太远”。
- 在CoOp上加入一个Loss_Kg，来自CoOp的V1~Vm提示词与CLIP的“A photo of a”提示词的损失计算
- 最终的损失Loss = Loss_CE + Loss_Kg * λ

### Prograd (ICCV23)
- 创新：不直接约束 Prompt，而是约束“Prompt 的梯度更新方向”。
- 三个向量：
    - Gd表示下游任务希望的梯度方向
    - Gg表示通用CLIP做zero-shot prediction的预测方向
    - Gprograd表示实际的梯度方向
- 做法：
    - CoOp一同操作到相似度计算，得到一个预测的Gprograd的方向向量，分别与图片的标签label做Loss_CE，与Gg做Loss_KL
    - Loss_CE反向的梯度表示为Gd，Loss_KL反向的梯度表示为Gg
    - 将其返回到learnable context这边准备学习
    - 如果Gd和Gg的正交方向小于90°，说明下游任务大体还是没偏离通用CLIP的，Gprograd的梯度按照Gd，也就是下游任务学习
    - 如果Gd和Gg的正交方向大于90°，说明下游任务偏离了通用CLIP的，Gprograd的梯度做法是“把Gd投影到Gg正交的方向上”
- 总结，也算是CoOp的微调，重点是通过正交化的设计，避免few-shot的下游任务影响模型的整体性能，同时也避免了过拟合。

### PromptSRC (ICCV23)
- 创新：可学习 Prompt + 原始 CLIP知识约束 + 图像特征约束 + Prompt自身约束
- 三个正则化约束
    - Textual Alignment Regularization(promptsrc学习的textprompt与clip的约束，类似KgCoOp)：L_text = 1 - cos(z_clip_text, z_prompt_text)
    - Visual Feature Alignment(让 Prompt Learning 后的表示不要偏离原来的 CLIP视觉知识)：计算方法同上
    - 直接约束 Prompt 参数本身，避免参数过度变化。即用训练好的Prompt与初始的Prompt之间的差值作为loss
    - 最终的损失Loss = Loss_CE + Loss_text * λ1 + Loss_vision * λ2 + Loss_prompt * λ3
- 总结：通过多个自调节约束，让 Prompt Learning 在适应下游任务和保持 CLIP 通用知识之间取得平衡。
- 细节说明：
    - PromptSRC的可训练参数有两个：Pv和Pt
    - 从输入开始，先是CLIP的text prompt与Pt_0做concat拼接，图像跟Pv_0做拼接，进入各自的encoder
    - 然后，每层encoder是上一层的result+Pvt_i得到的，这套流程类似maple，只是没做跨模态
    - prompt的Loss计算，是根据CLIP的text-vision得到的similarity和PromptSRC训练得到的text-vision的similarity，做KL Divergence（某周概率分布）。
        -   $$
            \mathcal{L}_{\mathrm{SCL-logits}}
            =
            D_{\mathrm{KL}}(P_p \parallel P)
            =
            \sum_{i=1}^{C}
            P_p(i)
            \log
            \frac{P_p(i)}{P(i)}
            $$
        - $$ P_p=\operatorname{softmax}(S_p) $$
        - $$ P=\operatorname{softmax}(S) $$
        - $$ S_p=\operatorname{sim}(\tilde{f}_p,\tilde{g}_p) $$
        - $$ S=\operatorname{sim}(\tilde{f},\tilde{g}) $$
        - P_p：PromptSRC 的图文匹配概率分布
        - P：原始 CLIP 的图文匹配概率分布
        - C：类别数量
### CoPrompt (ICLR24)
- 创新：约束“训练模型”和“原始CLIP”的预测/表示保持一致。
    - Consistency Constraint：类似PromptSRC，加入prompt的文本端和图像端，跟原始CLIP的文本图像的结果做余弦算Loss。
    - Perturbation：不仅要求两个模型对原始输入一致，还要求它们对扰动后的输入也保持一致。
        - 图像做增强，文本做语义等价
        - 防止过拟合
        - LLM生成的new text单独进入Text Encoder
        - 原始CLIP的文本+可学习Prompt一块儿进入另一个Text Encoder
        - 原始图像+可学习Prompt进入Image Encoder
        - 经过增强的扰动图像单独进入另一个Image Encoder
    - Prompt + Adapter：Prompt调整输入空间，Adapter调整特征。输出空间
- 最终Loss = Loss_CE + Loss_Consistency * λ
### PromptKD (CVPR2024)
- 创新：通过知识蒸馏大幅度节省模型参数量、内存、运行时间，也可以让小模型学习大模型的基类特异能力
- teacher-student：
    - teacher代表原始CLIP，其text encoder得到的输出text feature会保存为一个pre-stored text feature
    - teacher的text feature和image feature会相乘得到一个logits
    - 学生CLIP只有一个image Encoder，在可学习prompt的输入下进encoder，得到image feature，再通过projector转换特征维度，匹配teacher text feature后，与其相乘，得到logits。
    - 下一步进行蒸馏，从teacher的logits蒸馏到student的logits
- 做法：
    - 首先微调大模型，将大模型的文本特征保留（基类微调）
    - 同时喂给大模型和小模型的无标签的图片，和大模型保存下来的文本特征做对比，并且用project将小模型视觉特征维度映射到和大模型文本特征可以匹配的维度，之后让小模型对齐大模型。只更新prompt和project。
    - 用小模型做测试
- 大白话总结：
    - 首先，Teacher做一个准备，按照CLIP操作正常训练，得到了text-feature和image-feature。
    - 然后，冻结Image-Encoder，固定Text-feature。
    - 接下来做蒸馏：
        - 将图片输入分别放入冻结的teacher-image-encoder，和可学习prompt+project的student-image-encoder
        - logits的计算所需要的text-feature全由teacher固定的那个买单
        - 这样可以计算出teacher-logits和student-logits
        - 让student-logits尽可能学习到teacher-logits，以此自调节prompt+project。
    - 最终用测试图片放入学习完毕的student-image-encoder，老样子用text-feature计算logits，这是测试集做测试的logits结果。
- 本质是蒸馏logits，也就是teacher对各类别的预测分布。
- ![PromptKD 整体架构](./PromptKD%20Image.png)
### DePT (CVPR2024)
- 创新：不是仅学习一个整体的prompt，而是把prompt的作用分解，分别学习
- 一个prompt分成两个分类头
    - CAT Head：隔离出来吸收基类知识
        - 本质上承担了CLIP/CoOp原来的zero-shot classification的任务
        - 即判断图片是什么类别（image-classification）
        - 保证不把分类能力整坏
    - ITM Head：保证原始空间不被破坏
        - 学习的是一种更加明确的Image-Text Matching能力，即image-text是否匹配
        - 保证text-image的语义对应关系能够得到更细致的学习
         - ITM Head损失就是CoOp原本CLIP Classification Loss的计算
    - 评测时，base评测用混合（1-λ）*ITM Head + λ*CAT Head；new评测尽量用ITM
- 可训练参数：
    - CAT损失函数更新CAT Head内部参数
    - 总损失函数更新提示词（Loss = L_ITM + L_CAT * λ
- 总结：分解prompt learning中的知识，通用知识与下游任务特定知识分别建模/学习，从而降低CoOp中prompt对few-shot下游数据过拟合的问题，同时尽可能保留CLIP的泛化能力。
### MMA
- 创新1：不是整个 CLIP 都进行 Adapter，而是选择性地在高层加入 Adapter。
    - 论文实验最终发现，从 第5层开始加入 MMA，一直到第12层，Base/Novel/HM 的综合权衡最好。
- 创新2：不是分别训练 Vision Adapter 和 Text Adapter
    - shared projection链接vision和text的信息
    - 让两个模态的数据经过同一个可学习函数
- 核心结构：Shared Projection
    - Text和Vision各有一个down->shared->up的流程
    - 两边的shared阶段可共享自身参数
    - down和up分别用于适配不同模态：down确保两边到shared的维度相同，up确保shared之后回到各自维度
    - 一共五个参数：W_vision_down, W_vision_up, W_text_down, W_text_up, W_shared_projection
- 总结：MMA 通过在视觉和文本 Adapter 中共享 Projection 参数，使两个模态的下游学习梯度能够共同更新共享参数，从而建立跨模态的参数级联系，缓解 Vision-Language 的语义鸿沟并改善特征对齐；同时结合仅在高层加入 Adapter、冻结低层的策略，在 few-shot 场景下兼顾任务判别性与预训练知识的泛化能力。
    - MMA主要解决的问题是：vision和language在微调时“各学各的”
    - 上述问题在few-shot下更加严重，因为每个类别缺少训练样本，单独学习容易过拟合
    - MMA的做法，让两边在训练中途的参数层面链接，可以做到梯度“相互影响”
    - 注意：依旧是先拿到CLIP的预训练模型，针对不同下游任务做MMA的操作，所以是微调，而不是重新训练
### MMRL
- 创新：Shared Learnable Representation Space（R）——建立一组与具体模态无关的共享表示 Token。
    - 通过两个Proj分别映射到Image和Text
    - 插入在高层（经典Transformer深层做具体特征）
    - 最终的R可看作是模型下游任务的特别特征
- 模型架构：
    - 核心参数：
        - R：MMRL新增的可训练参数，在Encoder高层参与Transformer计算
        - C：Class Token，CLIP原本的全局语义表示
        - E：Text的特征
    - 第一个Patch Proj：把共享的R映射成适合ImageEncoder使用的R'，跟Text Encoder得到的E'做similarity（Loss_r）
    - 第二个Patch Proj：把CLIP全局的语义C映射成C'，跟Image Encoder得到的E'做similarity（Loss_c）
    - Text Proj：同理，映射TextEncoder的E->E'
    - 两个similarity是按照(1-α) * Loss_r + α * Loss_c做的
    - 除此之外，还有个两个loss，采用cosine相似度计算，也就是text proj与第二个patch proj，结果分别同frozen encoder，即原始CLIP的text与patch计算得到，记为Loss_v, Loss_t
    - 最终的Loss = (1-α) * Loss_r + α * Loss_c + λ * (Loss_v + Loss_t)
    - 注意：α和λ都不是可训练参数，而是人为设定的常量
- 总结：
    - 整体看的关键还是α和λ的设定。λ是通用性的约束，决定模型能偏离CLIP的范围；α则是在该范围内往下游任务学习的趋势。这俩手动设定造成的不确定因素有点多啊。
    - 至于搞同一个可训练参数，放image和text一块儿训练，这套打通两侧的流程倒是见过几次了，不稀奇。

### DPC (CVPR2025)
- 创新：复制一份prompt，继续在下游任务上训练微调，原版的通用prompt冻结。最终两者加权得到既适合下游的prompt，又适合通用的prompt。
- 大体架构：
    - 原版的CLIP或者CoOp先训练，得到预训练模型
    - 冻结参数和原版的Prompt，将prompt复制出一份
    - 在base上训练测试时做成mixed prompt，训练复制的这份prompt
    - 在new上采用冻结的原始prompt（相当于复制训练的这份加权值为0）
- 具体细节：
    - 1、在做训练时，DPC采用了硬负样本优化(HNO)做法，先根据被冻结的模型，找出K个negative objects，让DPC在这些难啃的骨头上做优化。
    - 这个道理可以理解，因为DPC训练的prompt底子就是被冻结的模型，没必要再训练本来模型就能分辨的数据，找难啃的骨头优化效率更高。
    - 2、加权：加权的参数是手动设置的，不进入训练。
    - 这操作跟MMRL有点类似啊，实质上这个加权参数对结果的影响很大，而且针对不同的数据集，恐怕还需要特调，不能一套参数用到底，否则结果不好看。
- 总结：该论文的做法比较简单粗暴，相当于在原有模型的基础上，继续做下游任务的训练。但是考虑到不能过拟合，就想出了一个复制prompt，一半训练一半冻结的操作。最终用加权来混合两个prompt的内容。
    - 注1：为什么能混合？论文有提出一个实验，分析了两个prompt的特征，指出尽管经过了额外的训练，两者在特征维度上具有相似的趋势，因此相加不会影响大体特征的方向。
    - 注2：有什么缺点？很明显，DPC只针对base做了强行微调，还是会遗忘通用知识的，只是根据混合加权参数或多或少限制。不过现在基本上都在将base和new解耦，只要base尽量不影响new，让new出现明显下降，基本上不算大问题。
    - 注3：现有的方向主要有这么几种做法：
        - 损失函数设计（一致性约束、梯度）：PromptSRC、KgCoOp
        - 提示词限制：DePT、DPC
        - 引入新提取词/结构：MMA、MMRL
        - 引入外部知识（大模型生成等）：CoPrompt