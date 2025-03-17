import torch
from torch import nn
import torch.nn.functional as F


class James(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output1 = input + 100
        output2 = output1 * output1
        return output1, output2


def Module():
    james = James()  # 实例化类
    x = torch.tensor(1.0)
    output1, output2 = james(x)

    print("output1 = {} , output2 = {}".format(output1, output2))


def Cov1d_2d():
    input = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]])

    kernel = torch.tensor([[1, 2, 1],
                           [0, 1, 0],
                           [2, 1, 0]])

    print(input.shape)
    print(kernel.shape)
    # 发现上面的尺寸不符合PyTorch要求的尺寸，下面进行调整.  reshape可以按照我们的要求以一定的方式进行转换

    input = torch.reshape(input, (1, 1, 5, 5))
    kernel = torch.reshape(kernel, (1, 1, 3, 3))

    output1 = F.conv2d(input, kernel, stride=1)
    print(output1)

    output2 = F.conv2d(input, kernel, stride=2)
    print(output2)

    output3 = F.conv2d(input, kernel, stride=1, padding=1)
    print(output3)


if __name__ == "__main__":
    # Module()
    Cov1d_2d()
