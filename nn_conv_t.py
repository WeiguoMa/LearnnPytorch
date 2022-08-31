import torch
import torch.nn.functional as F
# torch.nn.functional is a deterministic function, but not a Module.
# Inference as: https://blog.csdn.net/wangweiwells/article/details/100531264
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# Required by the function conv INPUT[Input tensor of shape(minibatch, in_channels, iH, iW)]
input = torch.reshape(input, (1, 1, 5, 5))
# Required by the function conv Weight[filters of shape(out_channels, in_channels/groups, kH, kW)]
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# print(input.shape)
# print(kernel.shape)

# Stride as the step of kernel
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=3)
print(output3)

# Padding is used to enlarge the input with PADDING-NUMBER
output4 = F.conv2d(input, kernel, stride=1, padding=1)
print(output4)


"""
Output matrix tends to be as 5th Squire, cause the full step goes to 5.
An explicit explanation could be reached with the link blow.
https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

"""
