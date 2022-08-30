import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0,
                         # drop_last 对 batch_size 不能整除的部分进行舍去
                         drop_last=False)

# 测试数据集中第一张图片及其对应 target
# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter('dataloader')
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("TestData", imgs, step)
    step = step + 1

writer.close()

"""
Data in Test_loader 将作为神经网络的输入
"""