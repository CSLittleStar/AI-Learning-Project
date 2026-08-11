import torch
"""
测试torch的基本使用
"""


"""
x = torch.tensor([1, 2, 3])
print(x.shape)
print(x.dtype)
print(x.device)
"""

"""
x = torch.randn(32, 3, 224, 224)
print(x)
"""

"""
x = torch.tensor(2.0, requires_grad=True)      # 创建一个x=2的标量，并开启梯度计算
y = x ** 2                                          # y = x^2
y.backward()    # 计算y关于x的导数                           
print(x.grad)   # 打印x的梯度，即y关于x的导数，即2x           # 由于开启了requires_grad，所以x的.grad属性会被自动创建，并记录求导的结果
"""

