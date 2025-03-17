import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.maxpool1 = MaxPool2d(3, ceil_mode=True)
        self.maxpool2 = MaxPool2d(3, ceil_mode=False)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


def run():
    input_ = torch.tensor([[1, 2, 0, 3, 1],
                           [0, 1, 2, 3, 1],
                           [1, 2, 1, 0, 0],
                           [5, 2, 3, 1, 1],
                           [2, 1, 0, 1, 1]], dtype=torch.float32)

    input_ = torch.reshape(input_, (-1, 1, 5, 5))

    james = James()
    output = james(input_)
    print(output)


trans_pool = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset_pool = torchvision.datasets.FashionMNIST("./data", train=False, transform=trans_pool, download=True)
dataloader = DataLoader(dataset=dataset_pool, batch_size=64)


def run_maxpool():
    james = James()
    writer = SummaryWriter("logs")
    step = 0

    for data in dataloader:
        imgs, targets = data
        writer.add_images("maxpool_input", imgs, step)
        output = james(imgs)
        writer.add_images("maxpool_output", output, step)
        step = step + 1

    writer.close()


if __name__ == "__main__":
    run_maxpool()
    # run()
