"""
最大池化：
尽可能的保留数据类型，但减小整体数据量，增加训练效率。

"""



import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# input = torch.reshape(input, (-1, 1, 5, 5))

class PNN(nn.Module):
    def __init__(self):
        super(PNN, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,
                                  ceil_mode=True)
                                  # Default Stride_size=Kernal_size
    def forward(self, input):
        output = self.maxpool1(input)
        return output

pnn = PNN()

writer = SummaryWriter("maxpool_logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = pnn(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()