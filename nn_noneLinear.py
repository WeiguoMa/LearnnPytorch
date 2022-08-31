

import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, -0.5],
#                      [-1, 3]])
#
# input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class NLN(nn.Module):
    def __init__(self):
        super(NLN, self).__init__()
        self.relu1 = ReLU(inplace=False)
        # ReLU is a function described as {x<0, then x=0} + {x>=0, then x=x}
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

nln = NLN()
writer = SummaryWriter("PNP_logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = nln(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

