# -*- coding: UTF-8 -*-
# Author: WeiguoM
# Mail: weiguo.m@iphy.ac.cn
import torch
from torch import nn
from torch.nn import Conv2d, Linear, Flatten, MaxPool2d, Sequential
# 搭建神经网络
class SQNN(nn.Module):
    def __init__(self):
        super(SQNN, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    sqnn = SQNN()
    input = torch.ones((64, 3, 32, 32))
    output = sqnn(input)
    print(output.shape)