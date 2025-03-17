import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
        # 这里的 inplace 表示在不在原来的位置进行替换，True 表示在原来的位置进行替换，一般使用默认的False

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


def run():
    input = torch.tensor([[1, -0.5],
                          [-1, 3]])
    # input = torch.reshape(input, (-1, 1, 2, 2))  # 新版的不需要，老版的需要指定 batchsize
    # print(input.shape)
    james = James()
    output = james(input)
    print(output)


def sigmoid_run():
    dataset = torchvision.datasets.FashionMNIST("./data", train=False, download=True,
                                                transform=torchvision.transforms.ToTensor())

    dataloader = DataLoader(dataset, batch_size=64)

    james = James()

    writer = SummaryWriter("logs")
    step = 0
    for data in dataloader:
        imgs, targets = data
        writer.add_images("Sigmoid_input", imgs, step)
        output = james(imgs)
        writer.add_images("sigmoid_output", output, step)
        step += 1

    writer.close()


if __name__ == "__main__":
    # run()
    sigmoid_run()
