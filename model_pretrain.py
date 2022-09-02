import torchvision.datasets
import torchvision.models
from torch import nn

# train_data = torchvision.datasets.ImageNet('./dataset', split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

# 添加现有的 Pythorch 模型
vgg_false = torchvision.models.vgg16(weights=False)     # False 代表导入的模型中，参数未经训练
vgg16_true = torchvision.models.vgg16(weights=True)

train_data = torchvision.datasets.CIFAR10('./dataset', download=True, train=True,
                                          transform=torchvision.transforms.ToTensor())

# 向现有网络中添加新模组
vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))

print(vgg16_true)

# 修改现有网络中的模组（重新赋值）
print(vgg_false)
vgg_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg_false)