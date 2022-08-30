from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np

# Usage of Python --> Totensor data type
# From transform.Totensor，解决两个问题
# 2， Tensor 数据类型的特点，以及为什么需要这个类型的数据。


img_path = 'D:\\Python_project\\PNeural_project\\dataset\\train\\ants_image\\0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter('D:\\Python_project\\PNeural_project\\logs')

# 1. Transform 的使用方法；
tensor_trans = transforms.ToTensor()        # 函数重定向声明
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
