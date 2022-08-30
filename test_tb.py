from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = 'dataset/train/bees_image/16838648_415acd9e3f.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array.shape)

writer.add_image('test', img_array, 2, dataformats='HWC')


writer.close()
