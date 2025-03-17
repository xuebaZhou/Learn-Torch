import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用之前的网络模型,将模型数据转到GPU上
model = MyLeNet5().to(device)

# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器,momentum用于加速收敛过程，可以帮助优化过程更快地穿过平坦的损失函数区域
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 学习率每隔10轮，变为原来的0.1
# 学习率调度器  gamma=0.1：表示每次调整学习率时，学习率将会乘以 0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)  # 转移到GPU上进行运算
        output = model(X)  # 调用模型计算输出
        cur_loss = loss_fn(output, y)  # 计算损失函数，将输出值和真实值作交叉熵运算
        # 我只关心最大值的索引，不关心最大值
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum((y == pred) / output.shape[0])

        # 开始反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("train_loss" + str(loss / n))
    print("train_acc" + str(current / n))


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad(): # 关掉计算梯度的相关计算
        for batch, (X, y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)  # 计算损失函数，将输出值和真实值作交叉熵运算
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum((y == pred) / output.shape[0])
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss" + str(loss / n))
        print("val_acc" + str(current / n))

        return current / n


# 开始训练
epochs = 50
min_acc = 0
for t in range(epochs):
    print(f'epoch{t + 1}\n----------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.makedirs(folder)
        min_acc = a
        print('save as model')
        torch.save(model.state_dict(), 'sava_model/best_model.pth')
print("Done")
