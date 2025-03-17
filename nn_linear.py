import torchvision
import torch
from torch import nn

from torch.nn import Linear
from torch.utils.data import DataLoader

dataset_pool = torchvision.datasets.FashionMNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                                 download=True)

dataloader = DataLoader(dataset=dataset_pool, batch_size=64, drop_last=True)  # 这里需要丢弃掉最后一组，因为最后一组的图片没有64张


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        # 根据 FashionMNIST 图像的尺寸 (28x28) 和单通道，计算输入特征数
        self.linear1 = Linear(in_features=28 * 28 * 64, out_features=10)

    def forward(self, x):
        output = self.linear1(x)
        return output


def linear_run():
    james = James()
    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)
        # output = torch.reshape(imgs, (1, 1, 1, -1))
        # reshape 功能强大，可以指定尺寸进行变换
        output = torch.flatten(imgs)
        # flatten 会将 tensor 变成一行
        print(output.shape)
        output = james(output)
        print(output.shape)


if __name__ == "__main__":
    linear_run()
