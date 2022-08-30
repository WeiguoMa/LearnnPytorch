from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('images/Postcard_4.jpg')

# Usage of Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("TOTENSOR", img_tensor)

# Usage of Normalize
trans_norm = transforms.Normalize([0.5,0.5, 0.5],[0.5,0.5, 0.5])
img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Usage of Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# print(img_resize)
# img_resize PIL --> img_resize Tensor
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize, 0)


# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL Image -> PIL -> Tensor Image
# The input of the second item 'trans_totensor' should match
#    the output of the first item 'trans_resize_2'
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((500,1000))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()



#%%

"""
需要注意函数的输入输出类型匹配
注意阅读函数说明文档
关注方法需要什么参数

"""