"""
1.计算实际输出和目标之间的差距；
2.为我们更新输出提供一定的依据（反向传播）；

"""
import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')      # default reduction = 'mean'
result = loss(inputs, targets)

loss_mse = MSELoss()                # Squared
result_mse = loss_mse(inputs, targets)

print(result_mse)

print(result)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()     # 交叉熵
result_cross = loss_cross(x, y)
print(result_cross)
