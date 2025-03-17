from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 相当于创建了一个全局变量，便于后面进行使用
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接这两个地址
        self.img_path = os.listdir(self.path)  # 获取图片下所有的地址

    def __getitem__(self, idx):  #
        img_name = self.img_path[idx]  # 获取特定的图片地址
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼接这三个字符串
        img = Image.open(img_item_path)  # 返回图像的所有数据，具体的方法可以在python控制台进行查看
        label = self.label_dir  # ants or bees
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "data/hymenoptera_data/train"
ants_label_dir = "ants"  # 获得蚂蚁的数据集
bees_label_dir = "bees"  # 获得蜜蜂的数据集

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset  # 将两个小数据集进行拼接
