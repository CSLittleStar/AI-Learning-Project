"""诊断 GoogLeNet 在 XPU 上 NaN 的来源：逐步前向+反向，定位爆炸点。"""
import torch
import torch.nn as nn
from googlenet import GoogLeNet, AuxClassifier

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print("device:", device)

model = GoogLeNet(num_classes=10).to(device)
model.train()

x = torch.randn(4, 3, 224, 224, device=device)
y = torch.randint(0, 10, (4,), device=device)

criterion = nn.CrossEntropyLoss()

# 逐层前向，捕捉 NaN 出现位置
print("\n=== 前向逐段检查 ===")
with torch.no_grad():
    t = model.stem(x)
    print("stem:", torch.isnan(t).any().item(), "max:", t.abs().max().item())
    t = model.inception3a(t); t = model.inception3b(t); t = model.maxpool3(t)
    print("after 3b:", torch.isnan(t).any().item(), "max:", t.abs().max().item())
    t = model.inception4a(t)
    a4a = t
    print("4a:", torch.isnan(t).any().item(), "max:", t.abs().max().item())
    t = model.inception4b(t); t = model.inception4c(t)
    t = model.inception4d(t)
    a4d = t
    print("4d:", torch.isnan(t).any().item(), "max:", t.abs().max().item())
    t = model.inception4e(t); t = model.maxpool4(t)
    t = model.inception5a(t); t = model.inception5b(t)
    print("5b:", torch.isnan(t).any().item(), "max:", t.abs().max().item())

# 检查 aux 分支尺寸
print("\n=== Aux 分支尺寸检查 ===")
with torch.no_grad():
    o1 = model.aux1(a4a)
    o2 = model.aux2(a4d)
    print("aux1 out shape:", o1.shape, "nan:", torch.isnan(o1).any().item())
    print("aux2 out shape:", o2.shape, "nan:", torch.isnan(o2).any().item())

# 完整前向 + 反向
print("\n=== 完整前向+反向 ===")
model.train()
out, aux1, aux2 = model(x)
print("main out nan:", torch.isnan(out).any().item(), "max:", out.abs().max().item())
print("aux1 nan:", torch.isnan(aux1).any().item(), "max:", aux1.abs().max().item())
print("aux2 nan:", torch.isnan(aux2).any().item(), "max:", aux2.abs().max().item())
loss = criterion(out, y) + 0.3*criterion(aux1, y) + 0.3*criterion(aux2, y)
print("loss:", loss.item())
loss.backward()
nan_grad = sum(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
print("params with nan grad:", nan_grad)
