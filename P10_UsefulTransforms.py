from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/hymenoptera_data/train/ants_image/9715481_b3cb4114ff.jpg")  # PIL数据类型
# print(img)

# ToTensor 的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)  # tensor 数据类型
writer.add_image("ToTensor_ssss", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 这地方由于是RGB。需要输入三个值，具体详见源码。值可以任意改动
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Norm", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
print(img_resize.size())
writer.add_image("Resize", img_resize, 0)

#  Compose - resize - 2
# 用法： Compose() 中的参数需要是一个列表，数据需要是 transforms 类型，得到 Compose([transforms参数1，transforms参数2,...])
# 后一个参数的输入需要与前一个参数的输出进行匹配
trans_resize_2 = transforms.Resize(512)
# PUL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])

img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_Random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_Random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)  # for every i ,随机裁剪一张256*256 的正方形图片出来
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
