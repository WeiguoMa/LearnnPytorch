import torch
import torchvision.models

vgg16 = torchvision.models.vgg16(weights=False)

# Save_method 1, Constructure + Parameter
torch.save(vgg16, 'vgg16_method1.pth')

#Save_method 2, Only Parameter
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
#%%
