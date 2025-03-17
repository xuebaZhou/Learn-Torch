import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# 这里需要加入dtype=torch.float32，不然会报错。
targets = torch.tensor([1, 2, 5], dtype=torch.float32)


def run_L1Loss():
    loss = L1Loss(reduction="sum")
    result = loss(inputs, targets)
    print(result)


def run_MSELoss():
    loss_mse = MSELoss()
    results_mse = loss_mse(inputs, targets)
    print(results_mse)


def run_CrossEntropyLoss():
    x = torch.tensor([0.1, 0.2, 0.3])
    y = torch.tensor([1])
    x = torch.reshape(x, (1, 3))
    loss_cross = CrossEntropyLoss()
    result_cross = loss_cross(x, y)
    print(result_cross)


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
dataloader = DataLoader(dataset, batch_size=1)


def run():
    james = James()
    loss = nn.CrossEntropyLoss()
    for data in dataloader:
        imgs, targets = data
        outputs = james(imgs)
        result_loss = loss(outputs, targets)
        result_loss.backward()
        print(result_loss)


if __name__ == "__main__":
    run_L1Loss()
    run_MSELoss()
    run_CrossEntropyLoss()
    run()
