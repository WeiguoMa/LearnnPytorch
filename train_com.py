# -*- coding: UTF-8 -*-
# Author: WeiguoM
# Mail: weiguo.m@iphy.ac.cn

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_sqnn import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10('./dataset', train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True,
                                       target_transform=None)
test_data = torchvision.datasets.CIFAR10('./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True,
                                         target_transform=None)

print('The length of the train_data: {}'.format(len(train_data)))
print('The length of the test_data: {}'.format(len(test_data)))

#%%

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


#%%

# 创建网络模型
sqnn = SQNN()
#%%

# 损失函数
loss_fn = nn.CrossEntropyLoss()

#%%

# 优化器
learning_rating = 0.01
optimizer = torch.optim.SGD(sqnn.parameters(), lr=learning_rating)
#%%

# 设置训练网络的一些参数
total_train_step = 0        # --> 记录训练次数
total_test_step = 0         # --> 记录测试的次数
epoch = 10                  # --> 训练的轮数

# 添加 Tensorboard
writer = SummaryWriter('logs_train')



for i in range(epoch):
    print('---------第 {} 轮训练开始---------'.format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = sqnn(imgs)
        loss_va = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss_va.backward()
        optimizer.step()

        if total_train_step % 100 == 0:
            print('训练次数：{}，Loss：{}'.format(total_train_step, loss_va.item()))
            # Item makes number a number.
            writer.add_scalar('train_loss', loss_va.item(), global_step=total_train_step)

        total_train_step += 1

#%%

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = sqnn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print('整体测试集上的 Loss：{}'.format(total_test_loss))
    writer.add_scalar('test_loss', total_test_loss, global_step=total_test_step)
    total_test_step += 1

    torch.save(sqnn, 'sqnn_{}.pth'.format(i))
    print('----------模型已保存----------')

writer.close()