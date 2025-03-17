import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  # 数据集很小，只进行转为tensor操作。后面想进行其余的操作可以再写
])

# train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
# 对于大型的数据集 coco，可以查找源代码找到下载链接，然后迅雷下载即可。

train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=dataset_transform, download=True)
# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()  # PIL有show的方法，但是tensor没有

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
