from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import writer

# python 的用法  --》 tensor数据类型
# 通过 transforms.ToTensor 去解决两个问题

# 2、 为什么我们需要 Tensor 数据类型

# 绝对路径  D:\计算机汇总\Pytorch\learn-torch\data\hymenoptera_data\train\ants_image\6240329_72c01e663e.jpg
# 相对路径  data/hymenoptera_data/train/ants_image/6240329_72c01e663e.jpg
img_path = "data/hymenoptera_data/train/ants_image/6240329_72c01e663e.jpg"
# 这里还是使用相对路径，因为绝对路径 \ 会被 windows 当做转义符
img = Image.open(img_path)
# print(img)

wirter = SummaryWriter("logs")

# 1、 transforms该如何使用 （python）  :从 transforms 中选择一个 Class 进行一个创建.然后在进行使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
