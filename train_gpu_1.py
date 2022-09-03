# -*- coding: UTF-8 -*-
# Author: WeiguoM
# Mail: weiguo.m@iphy.ac.cn

"""
网络模型

数据（输入、标注）

损失函数

.cuda() 调用并返回

"""

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
# from model_sqnn import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, Linear, Flatten, MaxPool2d, Sequential
import time


train_data = torchvision.datasets.CIFAR10('./dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True,
                                          target_transform=None)
test_data = torchvision.datasets.CIFAR10('./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True,
                                         target_transform=None)

train_data_size = len(train_data)
test_data_size = len(test_data)

print('The length of the train_data: {}'.format(train_data_size))
print('The length of the test_data: {}'.format(test_data_size))


# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)



# 创建网络模型
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
sqnn = SQNN()
if torch.cuda.is_available():
    sqnn =sqnn.cuda()       # 为网络模型提供 Cuda 方法

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()            # 为损失函数提供 Cuda 方法


# 优化器
learning_rating = 0.01
optimizer = torch.optim.SGD(sqnn.parameters(), lr=learning_rating)

# 设置训练网络的一些参数
total_train_step = 0        # --> 记录训练次数
total_test_step = 0         # --> 记录测试的次数
epoch = 10                  # --> 训练的轮数

# 添加 Tensorboard
writer = SummaryWriter('logs_train_gpu1')

star_time = time.time()          # 计时标记
for i in range(epoch):
    print('---------第 {} 轮训练开始---------'.format(i+1))

    # 训练步骤开始
    sqnn.train()            # 网络进入训练状态
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            ## 为训练数据提供 Cuda 方法
            imgs = imgs.cuda()
            targets = targets.cuda()
            ## 为训练数据提供 Cuda 方法
        outputs = sqnn(imgs)
        loss_va = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss_va.backward()
        optimizer.step()

        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - star_time)
            print('训练次数：{}，Loss：{}'.format(total_train_step, loss_va.item()))
            # Item makes number a number.
            writer.add_scalar('train_loss', loss_va.item(), global_step=total_train_step)

        total_train_step += 1

    # ------------------------------------训练完成----------------------进行测试------------------------------------ #
    # 测试步骤开始
    sqnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    axes_orientation = 1
    with torch.no_grad():       # 测试中不需要梯度对参数的优化
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                ## 为测试数据提供 Cuda 方法
                imgs = imgs.cuda()
                targets = targets.cuda()
                ## 为测试数据提供 Cuda 方法
            outputs = sqnn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(axes_orientation) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print('整体测试集上的 Loss：{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy/test_data_size))
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size,total_test_step)
    writer.add_scalar('test_loss', total_test_loss, global_step=total_test_step)
    total_test_step += 1

    torch.save(sqnn, './model_trained_CIFAR10/sqnn_{}.pth'.format(i))
    print('----------模型已保存----------')

writer.close()