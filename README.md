# 论文学习与代码实现

## 使用的数据集

本项目在训练与测试过程中使用了以下两个公开数据集，但出于体积考虑**未将数据集本身上传**至仓库，仅保留了对应的空目录占位（`data/cifar10/` 与 `data/MNIST/`，内含 `.gitkeep` 说明文件）。请在使用前自行下载并放置到对应目录：

- **CIFAR-10**：60,000 张 32×32 彩色图像，10 个类别（50,000 张训练 / 10,000 张测试）。用于 AlexNet 在本地数据集上的训练示例，对应 `data/cifar10/`。
- **MNIST**：70,000 张 28×28 手写数字灰度图像，10 个类别（60,000 张训练 / 10,000 张测试）。用于多层感知机（MLP）等基础模型的训练示例，对应 `data/MNIST/`。

> 说明：训练得到的模型权重文件（`.pth`）同样因体积过大未上传，运行代码后会重新生成。

## 环境配置
具体参考requirements.txt

本环境面向 **Intel Arc B580（XPU）** 显卡：
- 必须使用 PyTorch 官方 XPU 索引提供的 `torch` / `torchvision`（即 `2.13.0+xpu` / `0.28.0+xpu`），普通的 CUDA 版无法在 Intel GPU 上运行。
- 代码中设备选择已改为优先 `xpu`，其次 `cuda`，最后 `cpu`（`torch.xpu.is_available()`）。

### 安装（在已激活虚拟环境后）
```powershell
# 创建并激活虚拟环境（如使用项目已有 .venv 可跳过）
python -m venv .venv
.venv\Scripts\Activate.ps1

# 安装 PyTorch XPU 版（关键：指定 xpu 索引）
pip install torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu

# 安装其余依赖
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu
```

### 验证 XPU 可用
```python
import torch
print(torch.xpu.is_available())   # 应为 True
print(torch.xpu.device_count())   # 应为 1（B580）
```
运行虚拟环境：.venv\Scripts\Activate.ps1

## 神经网络框架
- 设计数据（输入/输出）
- 选择网络结构
- 搭建层级模块
- 选择激活函数
- 选择损失函数
- 初始化参数
- 前向传播计算输出
- 反向传播计算梯度
- 更新参数
### 输入
### 多层感知机MLP（全连接层（线性层）+激活函数）
- MLP就是由多个全连接层堆叠而成，并通过激活函数引入非线性能力的前馈神经网络。
### 损失函数
- CrossEntropyLoss：带softmax的交叉熵损失函数（最终结果越大，说明偏差越大）
	- softmax：将全连接层的输出转换为各类别的概率分布（p_j=e^(z_j) / 累加z的(e^z)）
		- 在AlexNet中，最后的全连接层输出1000个数，对应ImageNet的1000个类别。这些数可以理解为模型对每个类别的打分（logits）
		- 每个类别一个概率；所有类别加起来等于1；分数越高的类别，softmax后概率越大；

	- 交叉熵：衡量预测分布p与真实分布q之间的差异（-累加j的（q_j * log(p_j)））
### torch.optim.Adam优化器
- 根据loss反向传播得到的梯度，自动更新模型参数（权重、偏置），让loss逐步下降
- Adam = 动量+自适应学习率
	动量：记住历史梯度方向，加速收敛，减少震荡
	自适应学习率：每个参数根据自身梯度，动态调整步长（梯度大的步长小，梯度小的步长大）

