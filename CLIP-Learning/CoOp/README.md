# CoOp 实验 (CIFAR-10)

基于 `../CLIP/` 中已训练好的 CLIP 权重，实现论文 *Learning to Prompt for Vision-Language Models* 的 CoOp 方法。

## 与 CLIP 实验的核心区别

| 项目 | CLIP 实验 | CoOp 实验 |
| --- | --- | --- |
| 文本 prompt | 人工固定：`a photo of a {class}` | 可学习连续向量：`[V]_1...[V]_M [CLASS]` |
| 训练参数 | 整个双塔 + 温度 | **仅** prompt context `[V]` + 温度 |
| 编码器 | 全部可训练 | **冻结** Image/Text Encoder 与投影层 |
| 损失 | 对称对比损失 | 分类交叉熵（image→class） |
| 评估 | Zero-shot 准确率 | 直接分类准确率 |

核心思想：**不重新训练 CLIP**，只学习"如何描述任务"的连续 prompt 向量，让预训练强大的
Image Encoder 与 Text Encoder 在少量标注数据下更好地对齐到当前任务。

## 关键参数（`coop_cifar10.py` 顶部）

- `N_CTX`：可学习 context 向量个数 `[V]_1...[V]_M`，默认 4。
- `CSC`：`False` = Unified Context（所有类别共享同一组 `[V]`）；`True` = Class-Specific Context（每类独立 `[V]`）。
- `CTX_LR`：prompt context 学习率，默认 `1e-2`（比 CLIP 训练 LR 大，因为只训少量参数）。
- `INIT_CTX`：context 初始化方式，`"uniform"` 或 `"zero"`。
- `CLIP_CKPT`：CLIP 预训练权重路径，指向 `../CLIP/clip_cifar10.pth`（相对路径，迁移文件夹仍有效）。
- `CKPT_PATH`：CoOp 模型保存路径，即本目录 `coop_cifar10.pth`。

## 运行

```bash
cd CLIP-Learning/CoOp
python coop_cifar10.py
```

前置条件：
- `../CLIP/clip_cifar10.pth` 已存在（CLIP 实验的训练产物）。
- `e:/AI-Learning/Transformer-Learning/ViT/vit.py` 存在（提供 ViT 组件）。
- CIFAR-10 数据位于 `e:/AI-Learning/data/cifar10`（与 CLIP 实验一致）。

## 实现要点

1. `load_clip_and_freeze`：用 `strict=False` 加载 CLIP 权重（prompt 模块为新增，缺失属正常），
   随后把名字不含 `prompt.` 的参数全部 `requires_grad=False`，仅保留 prompt context 与 `logit_scale` 可训练。
2. `PromptContext`：生成 `(N_CTX, TEXT_WIDTH)`（Unified）或 `(NUM_CLASS, N_CTX, TEXT_WIDTH)`（CSC）的可学习向量。
3. `CoOp.encode_text_with_context`：把 context 拼到类别 embedding 前面，复用 Text Encoder 的
   Transformer 主体（位置编码、EncoderLayer、LayerNorm、投影）编码，取序列最后一个位置作为文本特征。
4. 类别 embedding 由冻结的 `token_embed` 查表得到，不进入梯度图。

## 训练无效排查记录 (重要)

初版运行 30 轮后 train loss 卡在 ~0.64、准确率仅较 CLIP 提升 0.2%、tau 恒为 9.92。
定位到以下实现隐患并已修复：

1. **冻结的 Encoder 在 train() 下执行了 Dropout (主因)**
   `model.train()` 会把冻结的 Image/Text Encoder 也设为 train 模式，其内部
   `EncoderLayer` 的 `Dropout(p=0.1)` 在每次前向随机丢弃激活，给 prompt context 的
   梯度注入噪声，导致无法收敛。修复: 训练时调用 `set_encoders_eval(model)` 把两个
   encoder 拉回 eval (关闭 dropout)，prompt 仅为 Parameter 不受影响。

2. **`logit_scale` (tau) 不应参与训练**
   原实现 `logit_scale.requires_grad=True`，现已在 `load_clip_and_freeze` 中冻结，
   沿用 CLIP 预训练值 (CoOp 论文规范)。

3. **取特征位置与 CLIP 预训练不一致**
   CLIP 的 TextEncoder 取最后一个非 pad 位置 (`<eos>`) 聚合整句语义；原 CoOp 实现取
   `[CLASS]` 单 token 位置，特征分布与预训练不同。修复: 在 `[CLASS]` 后追加 `[EOS]`
   embedding 并取该位置特征，与 CLIP 对齐。

若重新训练后仍无明显提升，可观察每轮打印的 `ctx_grad` / `ctx_norm`:
- 若 `ctx_grad≈0` 说明 ctx 梯度断流 (需排查冻结逻辑)；
- 若 `ctx_grad` 正常但 acc 不动，可尝试调大 `N_CTX`、开启 `CSC=True`、
  或提高 `CTX_LR`。

## 核心根因 (第二次排查) —— context 初始化方式

第一次修复 (dropout / logit_scale / 取特征位置) 后准确率仍只提升 0.2%，
真正原因在 **prompt context 的初始化**:

- 原实现: context `[V]_1..[V]_M` 用 `trunc_normal_(std=0.02)` 随机初始化,
  幅度仅 ±0.04 量级; 而冻结 Text Encoder 是从 CLIP 预训练的
  "a photo of a {class}" 完整文本学来的。
- 把 "a photo of a" 这 4 个模板词直接删掉、换成随机极小向量, 再拼上类名单 token,
  进 Transformer 的序列分布与预训练严重脱节 (前面几乎全零、最后突然一个大类名),
  冻结 Transformer 输出异常, text_features 分布崩坏, 类间相对关系几乎不变
  -> 准确率几乎不提升。

**修复 (CoOp 论文标准做法)**: 用 CLIP 模板词 "a photo of a" 的 token embedding
初始化 context 向量, 让初始文本分布与预训练对齐, text_features 起点即接近 CLIP
水平, 训练只需小幅微调即可超越。见 `main()` 中 `load_clip_and_freeze` 之后用
`model.text_encoder.embed_tokens` 查表并 `copy_` 到 `model.prompt.ctx.data` 的段落,
以及 `PromptContext` 对 `ctx_init_embed` 的处理。
