import os

import torch
import torchvision

from torchvision import transforms, datasets
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

dataset = torchvision.datasets.FashionMNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # 定义批量大小为 64


class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


def run_error():
    james = James()
    writer = SummaryWriter("logs")
    epoch = 0
    step = 0
    for data in dataloader:
        imgs, targets = data
        output = james(imgs)
        print("Epoch : {} , input_size = {}".format(epoch, imgs.shape))
        # Epoch : 0 , input_size = torch.Size([64, 1, 28, 28])
        print("Epoch : {} , output_size = {}".format(epoch, output.shape))
        # Epoch : 0 , output_size = torch.Size([64, 3, 26, 26])

        writer.add_images("in_e", imgs, step)
        writer.add_images("out_e", output, step)
        writer.add_images("out_e_plus", output, step)
        # 这里进行了两次卷积后的输出，尽管卷积操作相同，但我们发现在 tensorboard 中展示的图片颜色不一致
        # 查阅资料可知：卷积层生成的特征图通常包含负值或大于1的值，但图像像素值需要在0到1之间或者0到255之间的整数值。
        # 此外，这些特征图并不直接对应于可视化的RGB颜色空间，这导致了你观察到的颜色不一致现象。
        epoch = epoch + 1
        step = step + 1

    writer.close()


def run_right():
    james = James()
    writer = SummaryWriter("logs")
    epoch = 0
    step = 0
    for data in dataloader:
        imgs, targets = data
        output = james(imgs)
        print("Epoch : {} , input_size = {}".format(epoch, imgs.shape))
        # Epoch : 0 , input_size = torch.Size([64, 1, 28, 28])
        print("Epoch : {} , output_size = {}".format(epoch, output.shape))
        # Epoch : 0 , output_size = torch.Size([64, 3, 26, 26])

        writer.add_images("in_r", imgs, step)

        # 我希望将其变成 torch.Size([64, 3, 26, 26]) -> torch.Size([xxx, 1, 26, 26]),
        # 因为 FashionMNIST 数据集是黑白照片集，其通道数只有1，而不是传统的RGB三通道
        # 第一个数不知道是多少的时候直接写 -1 ，程序会根据后面进行计算
        output = torch.reshape(output, (-1, 1, 26, 26))
        print("Epoch : {} , reshape_output_size = {}".format(epoch, output.shape))
        writer.add_images("out_r", output, step)

        epoch = epoch + 1
        step = step + 1

    writer.close()


#  =====================================================================================================================
#  下面的代码是将之前的蚂蚁蜜蜂数据集进行卷积操作

class James_bees_ants(nn.Module):
    def __init__(self):
        super(James_bees_ants, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


class CustomAntsAndBeesImageDataset(Dataset):
    def __init__(self, root_dir, target_dir, transform=None):
        self.img_dir = os.path.join(root_dir, target_dir)
        self.transform = transform
        # 获取目录中的所有图像文件名
        self.img_names = [f for f in os.listdir(self.img_dir) if
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def run_ants_and_bees():
    root_dir = "data/hymenoptera_data/train"
    ants_target_dir = "ants_image"
    bees_target_dir = "bees_image"

    # img_ants_dir = os.path.join(root_dir, ants_target_dir)
    # img_bees_dir = os.path.join(root_dir, bees_target_dir)

    trans_img = transforms.Compose([
        transforms.Resize((256, 256)),  # 确保所有图像都调整为256x256
        transforms.ToTensor(),
        # 如果需要的话
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ants_dataset = CustomAntsAndBeesImageDataset(root_dir=root_dir, target_dir=ants_target_dir, transform=trans_img)
    bees_dataset = CustomAntsAndBeesImageDataset(root_dir=root_dir, target_dir=bees_target_dir, transform=trans_img)

    ants_loader = DataLoader(dataset=ants_dataset, batch_size=64, shuffle=True)
    bees_loader = DataLoader(dataset=bees_dataset, batch_size=64, shuffle=True)

    james = James_bees_ants()
    writer = SummaryWriter("logs")
    epoch = 0
    step = 0
    # 输出蚂蚁数据集的原始图片和卷积后的图片
    for data in ants_loader:
        imgs, targets = data
        output = james(imgs)
        print("Epoch : {} , ants_input_size = {}".format(epoch, imgs.shape))
        print("Epoch : {} , ants_output_size = {}".format(epoch, output.shape))

        writer.add_images("ants_in", imgs, step)

        #  将 6 通道转换回 3 通道
        output = torch.reshape(output, (-1, 3, 254, 254))

        print("Epoch : {} , ants_reshape_output_size = {}".format(epoch, output.shape))
        writer.add_images("ants_out", output, step)

        epoch = epoch + 1
        step = step + 1

    for data in bees_loader:
        imgs, targets = data
        output = james(imgs)
        print("Epoch : {} , bees_input_size = {}".format(epoch, imgs.shape))
        print("Epoch : {} , bees_output_size = {}".format(epoch, output.shape))

        writer.add_images("bees_in", imgs, step)

        #  将 6 通道转换回 3 通道
        output = torch.reshape(output, (-1, 3, 254, 254))

        print("Epoch : {} , bees_reshape_output_size = {}".format(epoch, output.shape))
        writer.add_images("bees_out", output, step)

        epoch = epoch + 1
        step = step + 1

    writer.close()


#  =====================================================================================================================
#  使用torchvision.datasets.ImageFolder来快速加载按上述方式组织的数据集
# 定义图像变换
def use_ImageFolder():

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    ants_train_dataset = datasets.ImageFolder(root='data/hymenoptera_data/train', transform=transform)
    ants_train_loader = DataLoader(ants_train_dataset, batch_size=64, shuffle=True)

    james = James_bees_ants()
    writer = SummaryWriter("logs")
    epoch = 0
    step = 0
    # 输出蚂蚁数据集的原始图片和卷积后的图片
    for data in ants_train_loader:
        imgs, targets = data
        output = james(imgs)
        print("Epoch : {} , use_ImageFolder_ants_input_size = {}".format(epoch, imgs.shape))
        print("Epoch : {} , use_ImageFolder_ants_output_size = {}".format(epoch, output.shape))

        # writer.add_images("use_ImageFolder_ants_in", imgs, step)

        #  将 6 通道转换回 3 通道
        output = torch.reshape(output, (-1, 3, 254, 254))

        print("Epoch : {} , use_ImageFolder_ants_reshape_output_size = {}".format(epoch, output.shape))
        writer.add_images("use_ImageFolder_ants_out", output, step)

        epoch = epoch + 1
        step = step + 1

    writer.close()


if __name__ == "__main__":
    # run_right()
    # run_error()
    # run_ants_and_bees()
    use_ImageFolder()
