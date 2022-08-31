import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)

    def forward(self, input):
        output = self.linear1(input)
        # Linear mapping
        return output



lnn = LNN()
# writer = SummaryWriter()

for data in dataloader:
    imgs, targets = data
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs) # --> One line
    output = lnn(output)