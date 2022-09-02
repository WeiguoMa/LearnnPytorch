import torch
import torchvision.models

# Load_Method 1 --> Save_Method 1
model1 = torch.load('vgg16_method1.pth')     # Root:str
print(model1)
#%%
# Load_Method 2 --> Save_Method 2
vgg16 = torchvision.models.vgg16(weights=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(model2)
#%%
