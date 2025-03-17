import torchvision
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def run_FashionMNIST():
    tens = torchvision.transforms.Compose([
        # transforms.CenterCrop(10),
        # transforms.RandomCrop(size=5, padding=None, pad_if_needed=False),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),   # FashionMNIST 是灰白照片，没有RGB三通道，只有一个
    ])

    # 准备的测试数据集
    # test_data = torchvision.datasets.FashionMNIST("./data", train=False, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST("./data", train=False, transform=tens)

    # batch_size=4 表示每次区四个数据进行打包  最后的 drop_last=False 保留最后不足64张的图片
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    # 测试数据集中第一张图片及target
    img, target = test_data[0]
    print(img.shape)
    print(target)
    writer = SummaryWriter("logs")
    for epoch in range(2):  # shuffle=True 则两次生成顺序不是一样的
        step = 0
        for data in test_loader:
            imgs, targets = data
            # print(imgs.shape)
            # print(targets)
            writer.add_images("Epochs: {}".format(epoch + 2), imgs, step)
            step = step + 1
    writer.close()


# 自定义数据集类
class CustomImageDataset(Dataset):
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
        image = Image.open(img_path).convert('RGB')  # 确保图像模式为RGB

        if self.transform:
            image = self.transform(image)

        return image, img_name  # 返回张量和图像名称


def run_ants():
    # 定义根目录和目标子目录
    root_dir = "data/hymenoptera_data/train"
    target_dir = "ants_image"
    # 获取目标目录的完整路径
    img_dir = os.path.join(root_dir, target_dir)
    # 定义变换：将PIL图像转换为张量
    transform = transforms.Compose([
        # transforms.CenterCrop(10),
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.ToTensor()
    ])
    # 创建数据集实例
    dataset = CustomImageDataset(root_dir=root_dir, target_dir=target_dir, transform=transform)
    # 使用DataLoader打包数据集
    test_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    writer = SummaryWriter("logs")
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("test_data_loader_11", imgs, step)
        step = step + 1

    writer.close()


if __name__ == "__main__":
    # run_ants()
    run_FashionMNIST()
