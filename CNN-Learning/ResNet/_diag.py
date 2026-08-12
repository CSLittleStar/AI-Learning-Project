import importlib.util
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

spec = importlib.util.spec_from_file_location('r18', 'resnet-18.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

device = torch.device('xpu')
normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
train_ds = datasets.CIFAR10(root=r'e:/AI-Learning/CNN-Learning/data/cifar10', train=True, download=False,
    transform=transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), normalize]))
loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

model = m.ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=m.LEARNING_RATE, momentum=m.MOMENTUM, weight_decay=m.WEIGHT_DECAY)

print('lr=', m.LEARNING_RATE, 'grad_clip=', m.GRAD_CLIP)
nan_flag = False
for epoch in range(3):
    for i, (x, y) in enumerate(loader):
        if i >= 10:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), m.GRAD_CLIP)
        optimizer.step()
        if torch.isnan(loss).any():
            print(f'  [NAN] epoch{epoch} step{i}')
            nan_flag = True
            break
    if nan_flag:
        break
    print(f'epoch {epoch}: last loss ok -> simulate full epoch loss stable')
print('NO NAN' if not nan_flag else 'STILL NAN')
