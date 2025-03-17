import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 定义模型
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


# 数据集和数据加载器
# transform = torchvision.transforms.ToTensor()
trans_loss = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    # XXX add something if you want
])
dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=trans_loss, download=True)  # 修改为训练集
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)  # 调整batch_size和num_workers


def run():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    # 实例化模型并移动到指定设备
    james = James().to(device)

    # 定义优化器和损失函数
    optim = torch.optim.SGD(james.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter("logs_loss")

    for epoch in range(20000):
        running_loss = 0.0  # 每一轮开始前，将loss设置为0
        for data in dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到指定设备

            outputs = james(imgs)
            result_loss = loss_fn(outputs, targets)

            optim.zero_grad()  # 将每个节点对应的梯度清零
            result_loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(james.parameters(), max_norm=1.0)

            optim.step()  # 对每个参数进行调优

            running_loss += result_loss.item()  # 累加当前批次的loss

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # 使用SummaryWriter记录损失值
        writer.add_scalar('training loss_1', avg_loss, epoch + 1)

        # 如果损失变为 nan，则提前终止训练
        if torch.isnan(torch.tensor(avg_loss)):
            print("Loss became NaN. Stopping training.")
            break
    writer.close()


if __name__ == "__main__":
    run()
