import torch
import torchvision
from torch import nn

# 下面表示vgg16使用到的网络当中的模型参数是没有经过训练的，是初始化了一个参数
vgg16 = torchvision.models.vgg16(weights=None)

# 模型保存方式 1， 后面指定的是保存的路径
# 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式 2
# 模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


#  陷阱
class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


james = James()

torch.save(james, "james_method1.pth")
