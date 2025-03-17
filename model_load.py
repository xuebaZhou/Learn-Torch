import torch
import torchvision

from torch import nn
from model_save import *

# 方式 1  -> 对应 model_save 中的保存方式 1
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)
#
# # 方式 2,加载模型
vgg16 = torchvision.models.vgg16(weights=None)
# # 通过python字典形式加载vgg16的模型架构，其中放入torch.load("vgg16_method2.pth")获得字典
vgg16.load_state_dict(torch.load("vgg16_method2.pth", weights_only=False))
# model = torch.load("vgg16_method2.pth")

# print(model)
print(vgg16)


# 陷阱1
# 真实情况下是不会复制这个模型过来的，而是定义在一个单独的文件夹下，然后导入进来即可
class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


#  james = James()  这一步可以不写了
model = torch.load("james_method1.pth", weights_only=False)  # 只写这个会报错，说啥 __init__ 没有初始化
print(model)
