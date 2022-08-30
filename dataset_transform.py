import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=dataset_transform)

# img, target = test_set[0]

print(test_set[0])

writer = SummaryWriter('ds_trans_logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('TestSet', img, i)

writer.close()
#%%
