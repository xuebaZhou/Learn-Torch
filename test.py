import torch
import torchvision
from PIL import Image
from torch import nn

# CIFAR-10 数据集的类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 获得照片的路径位置
airplane_image_path = "./image/airplane.png"
bird_image_path = "./image/bird.png"
deer_image_path = "./image/deer.png"
dog_image_path = "./image/dog.png"
ship_image_path = "./image/ship.png"

# 将照片以PIL格式打开
airplane_image = Image.open(airplane_image_path)
bird_image = Image.open(bird_image_path)
deer_image = Image.open(deer_image_path)
dog_image = Image.open(dog_image_path)
ship_image = Image.open(ship_image_path)

# 打印一下照片的类型
print(airplane_image)
print(bird_image)
print(deer_image)
print(dog_image)
print(ship_image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # 模型的输入就是32*32的
    torchvision.transforms.ToTensor()
])

airplane_image = transform(airplane_image)
bird_image = transform(bird_image)
deer_image = transform(deer_image)
dog_image = transform(dog_image)
ship_image = transform(ship_image)

print(airplane_image.shape)
print(bird_image.shape)
print(deer_image.shape)
print(dog_image.shape)
print(ship_image.shape)


# 模型架构
class James(nn.Module):
    def __init__(self):
        super(James, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device('cpu')
# 由于我的模型是用 GPU 上训练出来的，因此在 CPU 上检验的话需要加上 map_location=device
model_1 = torch.load("./all_pth/james_0.pth", map_location=device, weights_only=False)
model_20 = torch.load("./all_pth/james_19.pth", map_location=device, weights_only=False)
print(model_1)
print(model_20)

# 网络训练是需要 batchsize的，因此需要reshape一下
airplane_image = torch.reshape(airplane_image, (1, 3, 32, 32))
bird_image = torch.reshape(bird_image, (1, 3, 32, 32))
deer_image = torch.reshape(deer_image, (1, 3, 32, 32))
dog_image = torch.reshape(dog_image, (1, 3, 32, 32))
ship_image = torch.reshape(ship_image, (1, 3, 32, 32))

# 下面三行不写也没啥，但是还是建议写一下，养成良好的代码书写习惯
model_1.eval()
model_20.eval()

with torch.no_grad():
    airplane_output_use_model_1 = model_1(airplane_image)
    airplane_output_use_model_20 = model_20(airplane_image)

    bird_output_use_model_1 = model_1(bird_image)
    bird_output_use_model_20 = model_20(bird_image)

    deer_output_use_model_1 = model_1(deer_image)
    deer_output_use_model_20 = model_20(deer_image)

    dog_output_use_model_1 = model_1(dog_image)
    dog_output_use_model_20 = model_20(dog_image)

    ship_output_use_model_1 = model_1(ship_image)
    ship_output_use_model_20 = model_20(ship_image)

print("airplane_output_use_model_1 is {}".format(classes[airplane_output_use_model_1.argmax(1).item()]))
print("airplane_output_use_model_20 is {}".format(classes[airplane_output_use_model_20.argmax(1).item()]))

print("bird_output_use_model_1 is {}".format(classes[bird_output_use_model_1.argmax(1).item()]))
print("bird_output_use_model_20 is {}".format(classes[bird_output_use_model_20.argmax(1).item()]))

print("deer_output_use_model_1 is {}".format(classes[deer_output_use_model_1.argmax(1).item()]))
print("deer_output_use_model_20 is {}".format(classes[deer_output_use_model_20.argmax(1).item()]))

print("dog_output_use_model_1 is {}".format(classes[dog_output_use_model_1.argmax(1).item()]))
print("dog_output_use_model_20 is {}".format(classes[dog_output_use_model_20.argmax(1).item()]))

print("ship_output_use_model_1 is {}".format(classes[ship_output_use_model_1.argmax(1).item()]))
print("ship_output_use_model_20 is {}".format(classes[ship_output_use_model_20.argmax(1).item()]))

# print("airplane_output_use_model_1 is {} : ".format(airplane_output_use_model_1))
# print(airplane_output_use_model_1.argmax(1))
# print("airplane_output_use_model_20 is {} : ".format(airplane_output_use_model_20))
# print(airplane_output_use_model_20.argmax(1))
#
# print("bird_output_use_model_1 is {} : ".format(bird_output_use_model_1))  # 这种输出不利于解读，需要转换为利于解读的方式
# print(bird_output_use_model_1.argmax(1))
# print("bird_output_use_model_20 is {} : ".format(bird_output_use_model_20))
# print(bird_output_use_model_20.argmax(1))
#
# print("deer_output_use_model_1 is {} : ".format(deer_output_use_model_1))
# print(deer_output_use_model_1.argmax(1))
# print("deer_output_use_model_20 is {} : ".format(deer_output_use_model_20))
# print(deer_output_use_model_20.argmax(1))
#
# print("dog_output_use_model_1 is {} : ".format(dog_output_use_model_1))
# print(dog_output_use_model_1.argmax(1))
# print("dog_output_use_model_20 is {} : ".format(dog_output_use_model_20))
# print(dog_output_use_model_20.argmax(1))
#
# print("ship_output_use_model_1 is {} : ".format(ship_output_use_model_1))
# print(ship_output_use_model_1.argmax(1))
# print("ship_output_use_model_20 is {} : ".format(ship_output_use_model_20))
# print(ship_output_use_model_20.argmax(1))
