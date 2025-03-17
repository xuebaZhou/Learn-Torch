import torch
from torch import nn


# 搭建神经网络
# 将网络架构单独的放在一个文件里面，便于后续进行调用，这样可以减少代码量
class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    james = James()
    # 64 batch size，3 通道，32*32的图片
    _input = torch.ones((64, 3, 32, 32))
    output = james(_input)
    print(output.shape)
