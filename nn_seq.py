import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.cov1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.cov2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.cov3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 4 * 4, 64)
        self.linear2 = nn.Linear(64, 10)

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = F.relu(self.cov1(x))
        x = self.maxpool1(x)
        x = F.relu(self.cov2(x))
        x = self.maxpool2(x)
        x = F.relu(self.cov3(x))
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.model1(x)
        return x


def run_original():
    james = James()
    print(james)
    input = torch.ones((64, 3, 32, 32))
    output = james(input)
    print(output.shape)

    writer = SummaryWriter("logs_seq")
    writer.add_graph(james, input)
    writer.close()


if __name__ == "__main__":
    run_original()
