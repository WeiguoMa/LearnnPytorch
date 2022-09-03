# -*- coding: UTF-8 -*-
# Author: WeiguoM
# Mail: weiguo.m@iphy.ac.cn


import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])


axes_rotation = 1       # AR=1 数组比较方向为横向； AR=0 数组比较方向为竖向；
print(outputs.argmax(axes_rotation)) # argmax 输出为方向上最大值的索引

preds = outputs.argmax(axes_rotation)
targets = torch.tensor([0, 1])

print(preds == targets)  # 进行比较

print((preds == targets).sum())     # 输出相等的个数