import torch
import torchvision
import time

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 定义训练的设备 ,语法糖，适应不同的情况
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

# 创建网络模型
james = James()
james = james.to(device)  # 也可以直接写 james.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # 也可以直接写 loss_fn.to(device) ，不需要重新赋值

# 优化器
# learning_rate = 0.01
# 1e-2 = 1 x (10)^(-2) = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(james.parameters(), lr=learning_rate)

# 设置训练网路的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 20

# 添加 tensorboard
writer = SummaryWriter("logs_train")
# 记录开始时间
start_time = time.time()
for i in range(epoch):
    print("--------第 {} 轮训练开始---------".format(1 + i))

    # 训练步骤开始
    james.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = james(imgs)
        loss = loss_fn(outputs, targets)

        # 用优化器前，梯度清零
        optimizer.zero_grad()
        # 使用反向传播，得到每个参数节点的梯度
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss_1", loss.item(), total_train_step)

    # 在这里面的代码就没有梯度了。就保证不会对这里面的代码进行调优
    # 测试步骤开始
    james.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = james(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss_1", total_test_loss, total_test_step)
    writer.add_scalar("test_acc_1", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(james, "./all_pth/james_{}.pth".format(i))
    print("模型已保存")

writer.close()
