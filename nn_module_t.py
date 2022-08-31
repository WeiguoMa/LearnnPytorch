import torch
from torch import nn


class Weiguo(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


weiguo = Weiguo()
x = torch.tensor(1.0)
output = weiguo(x)
print(output)