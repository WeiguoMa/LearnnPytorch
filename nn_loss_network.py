import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, Linear, Flatten, MaxPool2d, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class SQNN(nn.Module):
    def __init__(self):
        super(SQNN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# writer = SummaryWriter('logs_loss_network')

loss = nn.CrossEntropyLoss()
sqnn = SQNN()
step = 1

for data in dataloader:
    imgs, targets = data
    outputs = sqnn(imgs)
    loss_va = loss(outputs, targets)
    # loss_va.backward()        # 反向传播，得到 Grad