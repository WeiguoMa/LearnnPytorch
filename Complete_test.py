# -*- coding: UTF-8 -*-
# Author: WeiguoM
# Mail: weiguo.m@iphy.ac.cn
import torch
import torchvision.transforms
from PIL import Image



img_path = './test_images/02.png'
image = Image.open(img_path)
image = image.convert('RGB')
print(image)

transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transfrom(image)
# print(image.shape)

model = torch.load('./model_trained_CIFAR10/sqnn_29.pth')

# print(model)

image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(image.cuda())
# print(output)

print(output.argmax(1))